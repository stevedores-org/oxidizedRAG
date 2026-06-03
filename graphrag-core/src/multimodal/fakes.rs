//! In-memory test doubles for the multimodal seams.
//!
//! Closes stevedores-org/oxidizedRAG#147 (`FakeEmbeddingEngine`) and #149
//! (`MemoryVectorStore`). Both are intended for unit and integration tests
//! of downstream code; they are not production-quality storage or
//! inference backends.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use sha2::{Digest, Sha256};

use super::embedding::{
    Embedding, EmbeddingEngine, EmbeddingError, EmbeddingMetadata, ModelInfo,
};
use super::store::{SearchFilters, SearchResult, VectorStore, VectorStoreError};
use super::types::{ContentType, MultimodalContent};

// ===========================================================================
// FakeEmbeddingEngine — #147
// ===========================================================================

/// How [`FakeEmbeddingEngine`] derives the embedding vector.
#[derive(Debug, Clone)]
enum VectorMode {
    /// Return the same fixed vector for every input.
    Fixed(Vec<f32>),
    /// Derive a deterministic vector from a SHA-256 hash of the input,
    /// so identical inputs always yield identical vectors and distinct
    /// inputs yield (almost certainly) distinct vectors.
    HashSeeded(usize),
}

/// In-memory [`EmbeddingEngine`] for tests.
///
/// Builder pattern:
///
/// ```rust
/// use graphrag_core::multimodal::fakes::FakeEmbeddingEngine;
///
/// let engine = FakeEmbeddingEngine::with_fixed_vector(vec![1.0, 0.0, 0.0]);
/// assert_eq!(engine.dimensions(), 3);
/// assert_eq!(engine.call_count(), 0);
/// ```
///
/// Failure injection (see the `*_with_failure_returns_local_inference_error`
/// unit test for an executable example).
pub struct FakeEmbeddingEngine {
    mode: VectorMode,
    failure: Option<String>,
    delay: Option<Duration>,
    call_count: AtomicUsize,
}

impl FakeEmbeddingEngine {
    /// Default fake: hash-seeded 8-dim vectors, no failure, no delay.
    pub fn new() -> Self {
        Self {
            mode: VectorMode::HashSeeded(8),
            failure: None,
            delay: None,
            call_count: AtomicUsize::new(0),
        }
    }

    /// Return `vector` for every input. Dimensionality is taken from
    /// `vector.len()`.
    pub fn with_fixed_vector(vector: Vec<f32>) -> Self {
        Self {
            mode: VectorMode::Fixed(vector),
            failure: None,
            delay: None,
            call_count: AtomicUsize::new(0),
        }
    }

    /// Derive each vector deterministically from a SHA-256 of the input
    /// bytes. Identical inputs → identical vectors; distinct inputs →
    /// distinct vectors (modulo astronomical collisions).
    pub fn with_hash_seeded_vector(dimensions: usize) -> Self {
        Self {
            mode: VectorMode::HashSeeded(dimensions),
            failure: None,
            delay: None,
            call_count: AtomicUsize::new(0),
        }
    }

    /// Inject a failure: subsequent calls return
    /// [`EmbeddingError::LocalInference`] with this message.
    pub fn with_failure(mut self, message: String) -> Self {
        self.failure = Some(message);
        self
    }

    /// Inject latency: subsequent calls sleep for `delay` before returning.
    /// Useful for testing timeout / concurrency behaviour.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    /// Number of times [`Self::embed`] has been called since construction
    /// or the last [`Self::reset_call_count`].
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }

    /// Reset the call counter to zero.
    pub fn reset_call_count(&self) {
        self.call_count.store(0, Ordering::Relaxed);
    }

    /// Dimensionality of vectors this engine produces.
    pub fn dimensions(&self) -> usize {
        match &self.mode {
            VectorMode::Fixed(v) => v.len(),
            VectorMode::HashSeeded(d) => *d,
        }
    }

    fn content_bytes(content: &MultimodalContent) -> &[u8] {
        match content {
            MultimodalContent::Text(s) => s.as_bytes(),
            MultimodalContent::Image { bytes, .. }
            | MultimodalContent::Audio { bytes, .. }
            | MultimodalContent::Video { bytes, .. }
            | MultimodalContent::Pdf { bytes } => bytes,
        }
    }

    fn derive_vector(&self, content: &MultimodalContent) -> Vec<f32> {
        match &self.mode {
            VectorMode::Fixed(v) => v.clone(),
            VectorMode::HashSeeded(dim) => hash_to_vector(Self::content_bytes(content), *dim),
        }
    }
}

impl Default for FakeEmbeddingEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn hash_to_vector(bytes: &[u8], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    let mut counter: u32 = 0;
    while out.len() < dim {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher.update(counter.to_le_bytes());
        let digest = hasher.finalize();
        for chunk in digest.chunks_exact(4) {
            if out.len() == dim {
                break;
            }
            // Map each 4-byte chunk to f32 in [-1.0, 1.0].
            let raw = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let normalized = (raw as f64 / u32::MAX as f64) * 2.0 - 1.0;
            out.push(normalized as f32);
        }
        counter = counter.wrapping_add(1);
    }
    out
}

fn content_hash(content: &MultimodalContent) -> String {
    let mut hasher = Sha256::new();
    hasher.update(FakeEmbeddingEngine::content_bytes(content));
    hex_encode(hasher.finalize().as_slice())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[async_trait]
impl EmbeddingEngine for FakeEmbeddingEngine {
    async fn embed(&self, content: MultimodalContent) -> Result<Embedding, EmbeddingError> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        if let Some(d) = self.delay {
            tokio::time::sleep(d).await;
        }
        if let Some(msg) = &self.failure {
            return Err(EmbeddingError::LocalInference(msg.clone()));
        }
        let vector = self.derive_vector(&content);
        Ok(Embedding {
            id: format!("fake-{}", self.call_count.load(Ordering::Relaxed)),
            content_hash: content_hash(&content),
            vector,
            content_type: content.content_type(),
            metadata: EmbeddingMetadata {
                mime_type: None,
                model: Some("fake-embedding-engine".into()),
                tags: HashMap::new(),
            },
            created_at: Utc::now(),
        })
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "fake-embedding-engine".into(),
            dimensions: self.dimensions(),
            supported_types: vec![
                ContentType::Text,
                ContentType::Image,
                ContentType::Audio,
                ContentType::Video,
                ContentType::Pdf,
            ],
        }
    }
}

// ===========================================================================
// MemoryVectorStore — #149
// ===========================================================================

/// Lightweight introspection on the in-memory store.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoreStats {
    /// Total embeddings currently stored.
    pub total: usize,
    /// Per-modality count.
    pub by_content_type: HashMap<ContentType, usize>,
}

/// In-memory [`VectorStore`] for tests. Exact cosine similarity, no
/// indexing — O(n) per query — but that's fine at test sizes.
pub struct MemoryVectorStore {
    inner: Mutex<HashMap<String, Embedding>>,
}

impl MemoryVectorStore {
    /// Empty store.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Empty store with `capacity` pre-allocated.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(HashMap::with_capacity(capacity)),
        }
    }

    /// Snapshot the store's contents.
    pub fn stats(&self) -> StoreStats {
        let guard = self.inner.lock().expect("MemoryVectorStore mutex poisoned");
        let mut by_content_type: HashMap<ContentType, usize> = HashMap::new();
        for e in guard.values() {
            *by_content_type.entry(e.content_type).or_insert(0) += 1;
        }
        StoreStats {
            total: guard.len(),
            by_content_type,
        }
    }
}

impl Default for MemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        None
    } else {
        Some(dot / denom)
    }
}

fn matches_filters(embedding: &Embedding, filters: &SearchFilters) -> bool {
    if let Some(types) = &filters.content_types {
        if !types.contains(&embedding.content_type) {
            return false;
        }
    }
    if let Some((start, end)) = &filters.date_range {
        if embedding.created_at < *start || embedding.created_at > *end {
            return false;
        }
    }
    for (key, value) in &filters.metadata_filters {
        match embedding.metadata.tags.get(key) {
            Some(v) if v == value => continue,
            _ => return false,
        }
    }
    true
}

#[async_trait]
impl VectorStore for MemoryVectorStore {
    async fn store(&self, embedding: Embedding) -> Result<(), VectorStoreError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;
        guard.insert(embedding.id.clone(), embedding);
        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        filters: SearchFilters,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;
        let mut scored: Vec<SearchResult> = guard
            .values()
            .filter(|e| matches_filters(e, &filters))
            .filter_map(|e| {
                cosine_similarity(query, &e.vector).map(|score| SearchResult {
                    embedding: e.clone(),
                    score,
                })
            })
            .filter(|r| filters.min_score.map_or(true, |min| r.score >= min))
            .collect();
        // Descending by score.
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    async fn get(&self, id: &str) -> Result<Option<Embedding>, VectorStoreError> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;
        Ok(guard.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;
        guard.remove(id);
        Ok(())
    }

    async fn count(&self) -> Result<usize, VectorStoreError> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| VectorStoreError::Backend(e.to_string()))?;
        Ok(guard.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> MultimodalContent {
        MultimodalContent::Text(s.into())
    }

    #[tokio::test]
    async fn fake_engine_returns_fixed_vector_and_increments_count() {
        let engine = FakeEmbeddingEngine::with_fixed_vector(vec![1.0, 0.0, 0.0]);
        let e1 = engine.embed(text("hello")).await.expect("embed");
        let e2 = engine.embed(text("world")).await.expect("embed");
        assert_eq!(e1.vector, vec![1.0, 0.0, 0.0]);
        assert_eq!(e2.vector, vec![1.0, 0.0, 0.0]);
        assert_eq!(engine.call_count(), 2);
        engine.reset_call_count();
        assert_eq!(engine.call_count(), 0);
    }

    #[tokio::test]
    async fn fake_engine_hash_seeded_is_deterministic_per_input() {
        let engine = FakeEmbeddingEngine::with_hash_seeded_vector(16);
        let a1 = engine.embed(text("hello")).await.expect("embed");
        let a2 = engine.embed(text("hello")).await.expect("embed");
        let b = engine.embed(text("world")).await.expect("embed");
        assert_eq!(a1.vector, a2.vector, "same input → same vector");
        assert_ne!(a1.vector, b.vector, "different input → different vector");
        assert_eq!(a1.vector.len(), 16);
    }

    #[tokio::test]
    async fn fake_engine_with_failure_returns_local_inference_error() {
        let engine = FakeEmbeddingEngine::new().with_failure("boom".into());
        let err = engine.embed(text("hi")).await.unwrap_err();
        match err {
            EmbeddingError::LocalInference(msg) => assert_eq!(msg, "boom"),
            other => panic!("expected LocalInference, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn fake_engine_batch_default_calls_embed_per_item() {
        let engine = FakeEmbeddingEngine::with_fixed_vector(vec![0.5; 4]);
        let inputs = vec![text("a"), text("b"), text("c")];
        let out = engine.embed_batch(inputs).await.expect("embed_batch");
        assert_eq!(out.len(), 3);
        assert_eq!(engine.call_count(), 3);
    }

    #[tokio::test]
    async fn memory_store_round_trips_an_embedding() {
        let store = MemoryVectorStore::new();
        let engine = FakeEmbeddingEngine::with_hash_seeded_vector(4);
        let e = engine.embed(text("hello")).await.expect("embed");
        let id = e.id.clone();

        store.store(e.clone()).await.expect("store");
        assert_eq!(store.count().await.expect("count"), 1);

        let fetched = store.get(&id).await.expect("get").expect("present");
        assert_eq!(fetched.id, id);
        assert_eq!(fetched.vector, e.vector);

        store.delete(&id).await.expect("delete");
        assert_eq!(store.count().await.expect("count"), 0);
    }

    #[tokio::test]
    async fn memory_store_search_orders_by_cosine_similarity_descending() {
        let store = MemoryVectorStore::new();
        let engine = FakeEmbeddingEngine::with_hash_seeded_vector(8);

        let a = engine.embed(text("alpha")).await.expect("embed");
        let b = engine.embed(text("beta")).await.expect("embed");
        let c = engine.embed(text("gamma")).await.expect("embed");

        store.store_batch(vec![a.clone(), b.clone(), c.clone()])
            .await
            .expect("store_batch");

        // Query is exactly `a.vector` — `a` should rank #1.
        let results = store
            .search(&a.vector, 3, SearchFilters::default())
            .await
            .expect("search");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].embedding.id, a.id);
        // Cosine of identical vectors is 1.0.
        assert!((results[0].score - 1.0).abs() < 1e-5);
        // Results are in non-increasing order.
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[tokio::test]
    async fn memory_store_filters_by_content_type() {
        let store = MemoryVectorStore::new();
        let engine = FakeEmbeddingEngine::with_hash_seeded_vector(4);
        let txt = engine.embed(text("hello")).await.expect("embed");
        let img = engine
            .embed(MultimodalContent::Image {
                bytes: vec![1, 2, 3],
                mime: "image/png".into(),
            })
            .await
            .expect("embed");
        store.store(txt.clone()).await.expect("store");
        store.store(img.clone()).await.expect("store");

        let filters = SearchFilters {
            content_types: Some(vec![ContentType::Image]),
            ..Default::default()
        };
        let results = store.search(&img.vector, 10, filters).await.expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].embedding.id, img.id);
    }

    #[tokio::test]
    async fn memory_store_search_by_type_helper_matches_filtered_search() {
        let store = MemoryVectorStore::new();
        let engine = FakeEmbeddingEngine::with_hash_seeded_vector(4);
        let txt = engine.embed(text("hello")).await.expect("embed");
        store.store(txt.clone()).await.expect("store");

        let via_helper = store
            .search_by_type(&txt.vector, 10, ContentType::Text)
            .await
            .expect("search_by_type");
        let via_filters = store
            .search(
                &txt.vector,
                10,
                SearchFilters {
                    content_types: Some(vec![ContentType::Text]),
                    ..Default::default()
                },
            )
            .await
            .expect("search");
        assert_eq!(via_helper.len(), via_filters.len());
        assert_eq!(via_helper[0].embedding.id, via_filters[0].embedding.id);
    }

    #[tokio::test]
    async fn memory_store_min_score_filters_results() {
        let store = MemoryVectorStore::new();
        let engine = FakeEmbeddingEngine::with_hash_seeded_vector(4);
        let a = engine.embed(text("alpha")).await.expect("embed");
        let b = engine.embed(text("beta")).await.expect("embed");
        store.store(a.clone()).await.expect("store");
        store.store(b.clone()).await.expect("store");

        let filters = SearchFilters {
            min_score: Some(0.9999),
            ..Default::default()
        };
        let results = store.search(&a.vector, 10, filters).await.expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].embedding.id, a.id);
    }

    #[tokio::test]
    async fn store_stats_breaks_down_by_content_type() {
        let store = MemoryVectorStore::new();
        let engine = FakeEmbeddingEngine::with_hash_seeded_vector(4);
        store.store(engine.embed(text("a")).await.expect("e")).await.expect("s");
        store.store(engine.embed(text("b")).await.expect("e")).await.expect("s");
        store
            .store(
                engine
                    .embed(MultimodalContent::Image {
                        bytes: vec![1, 2],
                        mime: "image/png".into(),
                    })
                    .await
                    .expect("e"),
            )
            .await
            .expect("s");
        let stats = store.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.by_content_type.get(&ContentType::Text), Some(&2));
        assert_eq!(stats.by_content_type.get(&ContentType::Image), Some(&1));
    }
}
