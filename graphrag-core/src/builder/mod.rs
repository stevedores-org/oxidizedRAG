//! GraphRAG builder module
//!
//! This module provides a builder pattern for constructing GraphRAG instances
//! and a config-driven `PipelineBuilder` for chaining `Stage<I,O>` objects.

use crate::core::{GraphRAGError, Result};

/// Builder for GraphRAG instances
#[derive(Debug, Clone, Default)]
pub struct GraphRAGBuilder {
    // Configuration fields
}

impl GraphRAGBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the GraphRAG instance
    pub fn build(self) -> Result<GraphRAG> {
        Err(GraphRAGError::Config {
            message: "GraphRAG builder not yet implemented".to_string(),
        })
    }
}

/// GraphRAG instance (placeholder)
#[derive(Debug)]
pub struct GraphRAG {
    // Fields will be added during implementation
}

impl GraphRAG {
    /// Create a builder
    pub fn builder() -> GraphRAGBuilder {
        GraphRAGBuilder::new()
    }
}

// ============================================================================
// Pipeline Builder
// ============================================================================

#[cfg(feature = "async")]
mod pipeline_builder {
    use crate::config::Config;
    use crate::core::{GraphRAGError, Result};
    use crate::pipeline::stage::Stage;
    use async_trait::async_trait;
    use std::fmt;

    /// A type-erased pipeline step that transforms `Vec<u8>` (serialized) input/output.
    /// This enables heterogeneous stage chaining in a single pipeline.
    #[async_trait]
    trait ErasedStage: Send + Sync {
        async fn process_erased(&self, input: Vec<u8>) -> Result<Vec<u8>>;
        fn name(&self) -> &str;
    }

    /// Wraps a concrete `Stage<I,O>` into an erased stage using serde.
    struct TypedStageWrapper<S, I, O> {
        stage: S,
        _phantom: std::marker::PhantomData<(I, O)>,
    }

    // Manual Send+Sync: safe because PhantomData doesn't hold data
    unsafe impl<S: Send, I, O> Send for TypedStageWrapper<S, I, O> {}
    unsafe impl<S: Sync, I, O> Sync for TypedStageWrapper<S, I, O> {}

    #[async_trait]
    impl<S, I, O> ErasedStage for TypedStageWrapper<S, I, O>
    where
        S: Stage<I, O> + Send + Sync,
        I: serde::de::DeserializeOwned + Send + 'static,
        O: serde::Serialize + Send + 'static,
    {
        async fn process_erased(&self, input: Vec<u8>) -> Result<Vec<u8>> {
            let decoded: I = serde_json::from_slice(&input).map_err(|e| {
                GraphRAGError::Config {
                    message: format!("Pipeline stage input deserialization failed: {e}"),
                }
            })?;
            let output = self.stage.process(decoded).await?;
            serde_json::to_vec(&output).map_err(|e| GraphRAGError::Config {
                message: format!("Pipeline stage output serialization failed: {e}"),
            })
        }

        fn name(&self) -> &str {
            self.stage.name()
        }
    }

    /// A built pipeline that processes data through a chain of stages.
    pub struct Pipeline {
        stages: Vec<Box<dyn ErasedStage>>,
        config_snapshot: String,
    }

    impl fmt::Debug for Pipeline {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let names: Vec<&str> = self.stages.iter().map(|s| s.name()).collect();
            f.debug_struct("Pipeline")
                .field("stages", &names)
                .finish()
        }
    }

    impl Pipeline {
        /// Run the pipeline with a serializable input, returning serialized output.
        pub async fn run<I, O>(&self, input: I) -> Result<O>
        where
            I: serde::Serialize + Send + 'static,
            O: serde::de::DeserializeOwned + Send + 'static,
        {
            if self.stages.is_empty() {
                return Err(GraphRAGError::Config {
                    message: "Pipeline has no stages".to_string(),
                });
            }

            let mut data = serde_json::to_vec(&input).map_err(|e| GraphRAGError::Config {
                message: format!("Pipeline input serialization failed: {e}"),
            })?;

            for stage in &self.stages {
                data = stage.process_erased(data).await?;
            }

            serde_json::from_slice(&data).map_err(|e| GraphRAGError::Config {
                message: format!("Pipeline output deserialization failed: {e}"),
            })
        }

        /// Get the number of stages in the pipeline.
        pub fn stage_count(&self) -> usize {
            self.stages.len()
        }

        /// Get the effective config snapshot (for `--print-effective-config`).
        pub fn effective_config(&self) -> &str {
            &self.config_snapshot
        }

        /// Get the names of all stages in order.
        pub fn stage_names(&self) -> Vec<&str> {
            self.stages.iter().map(|s| s.name()).collect()
        }
    }

    /// Config-driven builder that chains `Stage<I,O>` objects into a `Pipeline`.
    pub struct PipelineBuilder {
        stages: Vec<Box<dyn ErasedStage>>,
        config: Config,
    }

    impl PipelineBuilder {
        /// Create a new pipeline builder with the given config.
        pub fn new(config: Config) -> Self {
            Self {
                stages: Vec::new(),
                config,
            }
        }

        /// Add a typed stage to the pipeline.
        ///
        /// Stages are executed in the order they are added.
        pub fn add_stage<S, I, O>(mut self, stage: S) -> Self
        where
            S: Stage<I, O> + Send + Sync + 'static,
            I: serde::de::DeserializeOwned + Send + 'static,
            O: serde::Serialize + Send + 'static,
        {
            self.stages.push(Box::new(TypedStageWrapper {
                stage,
                _phantom: std::marker::PhantomData,
            }));
            self
        }

        /// Build the pipeline, snapshotting the effective config.
        pub fn build(self) -> Result<Pipeline> {
            if self.stages.is_empty() {
                return Err(GraphRAGError::Config {
                    message: "Cannot build empty pipeline".to_string(),
                });
            }

            let config_snapshot =
                toml::to_string_pretty(&self.config).unwrap_or_else(|_| "{}".to_string());

            Ok(Pipeline {
                stages: self.stages,
                config_snapshot,
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::config::Config;
        use crate::pipeline::stage::{ChunkBatch, EmbeddingBatch};
        use crate::core::{ChunkId, TextChunk, DocumentId};

        /// Stage that converts a String into a ChunkBatch (single chunk).
        struct StringToChunkStage;

        #[async_trait]
        impl Stage<String, ChunkBatch> for StringToChunkStage {
            async fn process(&self, input: String) -> Result<ChunkBatch> {
                let chunk = TextChunk {
                    id: ChunkId::new("chunk_0".to_string()),
                    document_id: DocumentId::new("doc_0".to_string()),
                    content: input,
                    start_offset: 0,
                    end_offset: 0,
                    embedding: None,
                    entities: vec![],
                    metadata: Default::default(),
                };
                Ok(ChunkBatch::new(vec![chunk]))
            }
            fn name(&self) -> &str {
                "string_to_chunk"
            }
        }

        /// Stage that converts ChunkBatch to EmbeddingBatch (fake embeddings).
        struct ChunkToEmbeddingStage;

        #[async_trait]
        impl Stage<ChunkBatch, EmbeddingBatch> for ChunkToEmbeddingStage {
            async fn process(&self, input: ChunkBatch) -> Result<EmbeddingBatch> {
                let embeddings = input
                    .0
                    .iter()
                    .map(|c| (c.id.clone(), vec![0.1_f32; 8]))
                    .collect();
                Ok(EmbeddingBatch::new(embeddings))
            }
            fn name(&self) -> &str {
                "chunk_to_embedding"
            }
        }

        #[tokio::test]
        async fn test_two_stage_pipeline() {
            let config = Config::default();
            let pipeline = PipelineBuilder::new(config)
                .add_stage(StringToChunkStage)
                .add_stage(ChunkToEmbeddingStage)
                .build()
                .unwrap();

            assert_eq!(pipeline.stage_count(), 2);
            assert_eq!(
                pipeline.stage_names(),
                vec!["string_to_chunk", "chunk_to_embedding"]
            );

            let result: EmbeddingBatch = pipeline
                .run("hello world".to_string())
                .await
                .unwrap();
            assert_eq!(result.len(), 1);
            assert_eq!(result.0[0].0, ChunkId::new("chunk_0".to_string()));
        }

        #[test]
        fn test_config_snapshot() {
            let config = Config::default();
            let pipeline = PipelineBuilder::new(config)
                .add_stage(StringToChunkStage)
                .build()
                .unwrap();

            let snapshot = pipeline.effective_config();
            assert!(!snapshot.is_empty());
            // Should contain some config keys
            assert!(snapshot.contains("chunk_size") || snapshot.contains("approach"));
        }

        #[test]
        fn test_empty_pipeline_fails() {
            let config = Config::default();
            let result = PipelineBuilder::new(config).build();
            assert!(result.is_err());
        }
    }
}

#[cfg(feature = "async")]
pub use pipeline_builder::{Pipeline, PipelineBuilder};
