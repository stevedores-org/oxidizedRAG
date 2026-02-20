//! Stage<I,O> trait and typed batch contracts for pipeline stages.
//!
//! Provides the core abstraction for composable, typed pipeline stages.

use crate::core::{ChunkId, Result, TextChunk};
use async_trait::async_trait;
use std::fmt::Debug;

#[cfg(feature = "incremental")]
use crate::graph::incremental::GraphDelta;

use crate::retrieval::SearchResult;

// ============================================================================
// Stage Trait
// ============================================================================

/// A typed, async pipeline stage transforming input `I` into output `O`.
///
/// Stages are the fundamental building blocks of processing pipelines.
/// They can be composed sequentially via `PipelineBuilder`.
#[async_trait]
pub trait Stage<I: Send + 'static, O: Send + 'static>: Send + Sync {
    /// Process the input and produce an output.
    async fn process(&self, input: I) -> Result<O>;

    /// Human-readable name of this stage (for logging/tracing).
    fn name(&self) -> &str;
}

// ============================================================================
// Typed Batch Newtypes
// ============================================================================

/// A batch of text chunks ready for embedding.
#[derive(Debug, Clone)]
pub struct ChunkBatch(pub Vec<TextChunk>);

impl ChunkBatch {
    /// Create a new chunk batch.
    pub fn new(chunks: Vec<TextChunk>) -> Self {
        Self(chunks)
    }

    /// Number of chunks in the batch.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A batch of embeddings keyed by chunk ID.
#[derive(Debug, Clone)]
pub struct EmbeddingBatch(pub Vec<(ChunkId, Vec<f32>)>);

impl EmbeddingBatch {
    /// Create a new embedding batch.
    pub fn new(embeddings: Vec<(ChunkId, Vec<f32>)>) -> Self {
        Self(embeddings)
    }

    /// Number of embeddings in the batch.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A graph delta produced by entity extraction, wrapping `GraphDelta`.
#[derive(Debug, Clone)]
pub struct EntityGraphDelta {
    /// The underlying graph delta (available when `incremental` feature is enabled).
    #[cfg(feature = "incremental")]
    pub delta: GraphDelta,
    /// Placeholder when `incremental` feature is disabled.
    #[cfg(not(feature = "incremental"))]
    _private: (),
}

impl EntityGraphDelta {
    /// Create a new entity graph delta from a `GraphDelta`.
    #[cfg(feature = "incremental")]
    pub fn new(delta: GraphDelta) -> Self {
        Self { delta }
    }

    /// Create an empty entity graph delta (non-incremental).
    #[cfg(not(feature = "incremental"))]
    pub fn empty() -> Self {
        Self { _private: () }
    }
}

/// A set of retrieval results from a search stage.
#[derive(Debug, Clone)]
pub struct RetrievalSet(pub Vec<SearchResult>);

impl RetrievalSet {
    /// Create a new retrieval set.
    pub fn new(results: Vec<SearchResult>) -> Self {
        Self(results)
    }

    /// Number of results.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ChunkId;

    /// A no-op stage that passes input through unchanged (compile test).
    struct NoopStage;

    #[async_trait]
    impl Stage<String, String> for NoopStage {
        async fn process(&self, input: String) -> Result<String> {
            Ok(input)
        }
        fn name(&self) -> &str {
            "noop"
        }
    }

    #[tokio::test]
    async fn test_noop_stage_compiles() {
        let stage = NoopStage;
        let result = stage.process("hello".to_string()).await.unwrap();
        assert_eq!(result, "hello");
        assert_eq!(stage.name(), "noop");
    }

    #[tokio::test]
    async fn test_trait_object_creation() {
        let stage: Box<dyn Stage<String, String>> = Box::new(NoopStage);
        let result = stage.process("world".to_string()).await.unwrap();
        assert_eq!(result, "world");
    }

    #[test]
    fn test_chunk_batch_construction() {
        let batch = ChunkBatch::new(vec![]);
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_embedding_batch_construction() {
        let batch = EmbeddingBatch::new(vec![
            (ChunkId::new("c1".to_string()), vec![0.1, 0.2]),
            (ChunkId::new("c2".to_string()), vec![0.3, 0.4]),
        ]);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_retrieval_set_construction() {
        let set = RetrievalSet::new(vec![]);
        assert!(set.is_empty());
    }

    #[test]
    fn test_entity_graph_delta_construction() {
        #[cfg(feature = "incremental")]
        {
            use crate::graph::incremental::{DeltaStatus, GraphDelta, UpdateId};
            let delta = GraphDelta {
                delta_id: UpdateId::new(),
                timestamp: chrono::Utc::now(),
                changes: vec![],
                dependencies: vec![],
                status: DeltaStatus::Pending,
                rollback_data: None,
            };
            let egd = EntityGraphDelta::new(delta);
            assert!(egd.delta.changes.is_empty());
        }

        #[cfg(not(feature = "incremental"))]
        {
            let _egd = EntityGraphDelta::empty();
        }
    }
}
