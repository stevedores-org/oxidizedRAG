//! Transparent caching wrapper for pipeline stages.
//!
//! Provides content-addressed caching at the stage boundary, enabling
//! incremental updates and avoiding recomputation of unchanged inputs.

use super::{Stage, StageMeta, StageError, ContentHashable};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// In-memory cache for stage outputs.
///
/// Uses a simple HashMap with content-based keys.
pub struct StageCache {
    entries: Mutex<HashMap<String, Vec<u8>>>,
}

impl StageCache {
    /// Create a new in-memory stage cache.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Get a cached value by key.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.entries.lock().unwrap().get(key).cloned()
    }

    /// Store a value in the cache.
    pub fn set(&self, key: String, value: Vec<u8>) {
        self.entries.lock().unwrap().insert(key, value);
    }

    /// Clear all cache entries.
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }

    /// Get cache statistics.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.lock().unwrap().is_empty()
    }
}

impl Default for StageCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper that adds transparent caching to any Stage<I, O>.
///
/// Cache is populated based on content hashes of inputs. Non-deterministic
/// stages (those with `metadata().deterministic == false`) bypass caching.
pub struct CachedStage<I, O> {
    inner: Arc<dyn Stage<I, O>>,
    cache: Arc<StageCache>,
    enabled: bool,
}

impl<I, O> CachedStage<I, O>
where
    I: ContentHashable + serde::Serialize + Send + Sync + 'static,
    O: serde::Serialize + serde::de::DeserializeOwned + Send + Sync + 'static,
{
    /// Create a new cached stage wrapper.
    pub fn new(stage: Arc<dyn Stage<I, O>>, cache: Arc<StageCache>) -> Self {
        // Only enable caching for deterministic stages
        let enabled = stage.metadata().deterministic;
        Self {
            inner: stage,
            cache,
            enabled,
        }
    }

    /// Get the cache key for a given input.
    ///
    /// Format: `{stage_name}@{version}:{input_hash}`
    fn cache_key(&self, input: &I) -> String {
        format!(
            "{}@{}:{}",
            self.inner.name(),
            self.inner.version(),
            input.content_hash()
        )
    }

    /// Get a reference to the inner stage.
    pub fn inner(&self) -> &Arc<dyn Stage<I, O>> {
        &self.inner
    }

    /// Get a reference to the cache.
    pub fn cache(&self) -> &Arc<StageCache> {
        &self.cache
    }

    /// Check if caching is enabled for this stage.
    pub fn caching_enabled(&self) -> bool {
        self.enabled
    }
}

#[async_trait]
impl<I, O> Stage<I, O> for CachedStage<I, O>
where
    I: ContentHashable + serde::Serialize + Send + Sync + 'static,
    O: serde::Serialize + serde::de::DeserializeOwned + Send + Sync + 'static,
{
    async fn execute(&self, input: I) -> Result<O, StageError> {
        // If caching is disabled, pass through to inner stage
        if !self.enabled {
            return self.inner.execute(input).await;
        }

        let key = self.cache_key(&input);

        // Try cache lookup
        if let Some(cached_bytes) = self.cache.get(&key) {
            if let Ok(output) = bincode::deserialize::<O>(&cached_bytes) {
                return Ok(output);
            }
        }

        // Cache miss - execute stage
        let output = self.inner.execute(input).await?;

        // Store in cache (best-effort, ignore serialization errors)
        if let Ok(bytes) = bincode::serialize(&output) {
            self.cache.set(key, bytes);
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn version(&self) -> &str {
        self.inner.version()
    }

    fn metadata(&self) -> StageMeta {
        self.inner.metadata()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::types::{ChunkBatch, DocumentChunk, EmbeddingBatch, EmbeddingRecord};

    /// Mock stage for testing that returns uppercase input.
    struct MockStringStage;

    #[async_trait]
    impl Stage<String, String> for MockStringStage {
        async fn execute(&self, input: String) -> Result<String, StageError> {
            Ok(input.to_uppercase())
        }

        fn name(&self) -> &str {
            "mock-string"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }
    }

    /// Mock stage that returns an error.
    struct FailingStage;

    #[async_trait]
    impl Stage<String, String> for FailingStage {
        async fn execute(&self, _input: String) -> Result<String, StageError> {
            Err(StageError {
                stage_name: "failing".to_string(),
                message: "intentional failure".to_string(),
                details: None,
            })
        }

        fn name(&self) -> &str {
            "failing"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn metadata(&self) -> StageMeta {
            StageMeta {
                deterministic: true,
                ..Default::default()
            }
        }
    }

    /// Non-deterministic mock stage.
    struct NonDeterministicStage;

    #[async_trait]
    impl Stage<String, String> for NonDeterministicStage {
        async fn execute(&self, input: String) -> Result<String, StageError> {
            Ok(format!("{}-{}", input, uuid::Uuid::new_v4()))
        }

        fn name(&self) -> &str {
            "non-det"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn metadata(&self) -> StageMeta {
            StageMeta {
                deterministic: false,
                ..Default::default()
            }
        }
    }

    #[tokio::test]
    async fn test_cache_hit() {
        let cache = Arc::new(StageCache::new());
        let stage = Arc::new(MockStringStage);
        let cached = CachedStage::new(stage, cache.clone());

        let input = "hello".to_string();
        let result1 = cached.execute(input.clone()).await.unwrap();
        assert_eq!(result1, "HELLO");
        assert_eq!(cache.len(), 1, "Cache should have 1 entry after first execution");

        let result2 = cached.execute(input).await.unwrap();
        assert_eq!(result2, "HELLO");
        assert_eq!(cache.len(), 1, "Cache should still have 1 entry");
    }

    #[tokio::test]
    async fn test_cache_miss_different_input() {
        let cache = Arc::new(StageCache::new());
        let stage = Arc::new(MockStringStage);
        let cached = CachedStage::new(stage, cache.clone());

        let result1 = cached.execute("hello".to_string()).await.unwrap();
        assert_eq!(result1, "HELLO");

        let result2 = cached.execute("world".to_string()).await.unwrap();
        assert_eq!(result2, "WORLD");

        assert_eq!(cache.len(), 2, "Cache should have 2 entries for different inputs");
    }

    #[tokio::test]
    async fn test_non_deterministic_stage_bypasses_cache() {
        let cache = Arc::new(StageCache::new());
        let stage = Arc::new(NonDeterministicStage);
        let cached = CachedStage::new(stage, cache.clone());

        let input = "hello".to_string();
        let result1 = cached.execute(input.clone()).await.unwrap();
        let result2 = cached.execute(input).await.unwrap();

        assert_ne!(result1, result2, "Non-deterministic stage should return different results");
        assert!(cache.is_empty(), "Cache should be empty for non-deterministic stage");
    }

    #[tokio::test]
    async fn test_error_propagates() {
        let cache = Arc::new(StageCache::new());
        let stage = Arc::new(FailingStage);
        let cached = CachedStage::new(stage, cache.clone());

        let result = cached.execute("hello".to_string()).await;
        assert!(result.is_err(), "Should propagate error from inner stage");
        assert!(cache.is_empty(), "Cache should be empty after error");
    }

    #[tokio::test]
    async fn test_metadata_delegation() {
        let cache = Arc::new(StageCache::new());
        let stage = Arc::new(MockStringStage);
        let cached = CachedStage::new(stage, cache);

        assert_eq!(cached.name(), "mock-string");
        assert_eq!(cached.version(), "1.0.0");
        assert!(cached.metadata().deterministic);
    }

    #[tokio::test]
    async fn test_chunk_batch_caching() {
        let cache = Arc::new(StageCache::new());

        // Create a simple stage that identity-returns ChunkBatch
        struct IdentityChunkStage;

        #[async_trait]
        impl Stage<ChunkBatch, ChunkBatch> for IdentityChunkStage {
            async fn execute(&self, input: ChunkBatch) -> Result<ChunkBatch, StageError> {
                Ok(input)
            }

            fn name(&self) -> &str {
                "identity-chunk"
            }

            fn version(&self) -> &str {
                "1.0.0"
            }
        }

        let stage = Arc::new(IdentityChunkStage);
        let cached = CachedStage::new(stage, cache.clone());

        let batch = ChunkBatch {
            id: "batch-1".to_string(),
            chunks: vec![DocumentChunk {
                id: "chunk-1".to_string(),
                content: "test".to_string(),
                source: "test.rs".to_string(),
                line_range: Some((1, 1)),
                metadata: Default::default(),
            }],
            corpus_hash: "hash123".to_string(),
        };

        let result1 = cached.execute(batch.clone()).await.unwrap();
        let result2 = cached.execute(batch).await.unwrap();

        assert_eq!(result1, result2);
        assert_eq!(cache.len(), 1, "Should have cached the batch");
    }

    #[tokio::test]
    async fn test_embedding_batch_caching() {
        let cache = Arc::new(StageCache::new());

        struct IdentityEmbeddingStage;

        #[async_trait]
        impl Stage<EmbeddingBatch, EmbeddingBatch> for IdentityEmbeddingStage {
            async fn execute(&self, input: EmbeddingBatch) -> Result<EmbeddingBatch, StageError> {
                Ok(input)
            }

            fn name(&self) -> &str {
                "identity-embedding"
            }

            fn version(&self) -> &str {
                "1.0.0"
            }
        }

        let stage = Arc::new(IdentityEmbeddingStage);
        let cached = CachedStage::new(stage, cache.clone());

        let batch = EmbeddingBatch {
            id: "batch-1".to_string(),
            embeddings: vec![EmbeddingRecord {
                chunk_id: "chunk-1".to_string(),
                vector: vec![0.1, 0.2, 0.3],
                metadata: Default::default(),
            }],
            config_hash: "config-1".to_string(),
        };

        let _result1 = cached.execute(batch.clone()).await.unwrap();
        let _result2 = cached.execute(batch).await.unwrap();

        assert_eq!(cache.len(), 1, "Should have cached the embedding batch");
    }

    #[test]
    fn test_cache_key_format() {
        let cache = Arc::new(StageCache::new());
        let stage = Arc::new(MockStringStage);
        let cached = CachedStage::new(stage, cache);

        let key = cached.cache_key(&"hello".to_string());
        assert!(key.contains("mock-string@1.0.0"), "Key should include name and version");
        assert!(key.contains(':'), "Key should use colon separator");
    }
}
