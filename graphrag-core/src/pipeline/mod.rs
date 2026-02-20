//! Pipeline DAG Architecture - typed, composable stages with explicit contracts.
//!
//! Provides first-class pipeline composition with type-safe stage boundaries,
//! deterministic DAG generation, and per-stage caching.
//!
//! ## Architecture
//!
//! The pipeline is organized as a directed acyclic graph (DAG) of stages,
//! where each stage has explicitly typed inputs and outputs.
//!
//! ```text
//! ChunkBatch → [Chunking Stage] → [Embedding Stage] → EmbeddingBatch → ...
//! ```
//!
//! Each stage:
//! - Implements the `Stage<I, O>` trait
//! - Is independently testable
//! - Can be swapped for alternative implementations
//! - Registers with a StageRegistry for discovery
//!
//! ## Core Types
//!
//! - `ChunkBatch`: Documents and chunks
//! - `EmbeddingBatch`: Vectors and sources
//! - `EntityGraphDelta`: Graph changes (add/remove/update)
//! - `RetrievalSet`: Ranked query results
//!
//! ## Example
//!
//! ```ignore
//! let mut registry = StageRegistry::new();
//! registry.register("chunker", "1.0.0", "Split documents into chunks");
//!
//! let chunker = MyChunker::new();
//! let batch = ChunkBatch { /* ... */ };
//!
//! let result = chunker.execute(batch).await?;
//! ```

pub mod registry;
pub mod stage;
pub mod builder;
pub mod types;

/// Stage-level caching/memoization.
#[cfg(feature = "async")]
pub mod cached_stage;

pub use registry::{StageId, StageRegistry};
pub use stage::{Stage, StageMeta, StageError};
pub use types::{ChunkBatch, DocumentChunk, EmbeddingBatch, EmbeddingRecord,
                EntityGraphDelta, GraphNode, GraphEdge, RetrievalSet, RankedResult,
                ScoreBreakdown};

#[cfg(feature = "async")]
pub use cached_stage::CachedStage;
