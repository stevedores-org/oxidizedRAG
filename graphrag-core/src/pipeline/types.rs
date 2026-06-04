//! Core batch types for pipeline stages.
//!
//! These types define the contracts between pipeline stages,
//! enabling type-safe composition and explicit data flow.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A batch of document chunks ready for processing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkBatch {
    /// Unique batch identifier
    pub id: String,
    /// List of chunks with metadata
    pub chunks: Vec<DocumentChunk>,
    /// Source corpus hash for caching
    pub corpus_hash: String,
}

/// A single document chunk with metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentChunk {
    /// Unique chunk ID
    pub id: String,
    /// Chunk content
    pub content: String,
    /// Source file path
    pub source: String,
    /// Line range in source (start, end)
    pub line_range: Option<(usize, usize)>,
    /// Metadata (language, type, etc)
    pub metadata: HashMap<String, String>,
}

/// A batch of embeddings with source references.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingBatch {
    /// Unique batch identifier
    pub id: String,
    /// List of embeddings
    pub embeddings: Vec<EmbeddingRecord>,
    /// Config hash that produced these embeddings
    pub config_hash: String,
}

/// A single embedding with source reference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingRecord {
    /// Chunk ID this embedding represents
    pub chunk_id: String,
    /// Dense vector (typically 384-1536 dims)
    pub vector: Vec<f32>,
    /// Metadata (model, timestamp, etc)
    pub metadata: HashMap<String, String>,
}

/// A delta representing graph changes (add/remove/update).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EntityGraphDelta {
    /// Unique delta ID
    pub id: String,
    /// Nodes added to graph
    pub added_nodes: Vec<GraphNode>,
    /// Node IDs removed
    pub removed_nodes: Vec<String>,
    /// Edges added
    pub added_edges: Vec<GraphEdge>,
    /// Edge IDs removed
    pub removed_edges: Vec<String>,
    /// Nodes with updated payloads
    pub updated_nodes: Vec<GraphNode>,
}

/// A node in the entity graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphNode {
    /// Unique node ID
    pub id: String,
    /// Node label/type
    pub label: String,
    /// Node properties
    pub properties: serde_json::Value,
}

/// An edge in the entity graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphEdge {
    /// Edge ID
    pub id: String,
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Relationship type
    pub rel_type: String,
    /// Edge properties
    pub properties: serde_json::Value,
}

/// A set of ranked retrieval results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalSet {
    /// Unique result set ID
    pub id: String,
    /// Query that produced these results
    pub query: String,
    /// Ranked results
    pub results: Vec<RankedResult>,
    /// Retrieval config hash
    pub config_hash: String,
}

/// A single ranked retrieval result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RankedResult {
    /// Chunk ID
    pub chunk_id: String,
    /// Final fused score (0-1)
    pub score: f32,
    /// Component scores
    pub scores: ScoreBreakdown,
    /// Chunk content preview
    pub preview: String,
}

/// Detailed score breakdown for a result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreBreakdown {
    /// Vector similarity score
    pub vector_score: f32,
    /// BM25/keyword score
    pub keyword_score: f32,
    /// Graph/relationship score
    pub graph_score: f32,
    /// Metadata for debugging
    pub metadata: HashMap<String, f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_batch_serde_roundtrip() {
        let batch = ChunkBatch {
            id: "batch-1".to_string(),
            chunks: vec![DocumentChunk {
                id: "chunk-1".to_string(),
                content: "fn hello() {}".to_string(),
                source: "main.rs".to_string(),
                line_range: Some((1, 1)),
                metadata: Default::default(),
            }],
            corpus_hash: "abc123".to_string(),
        };

        let json = serde_json::to_string(&batch).expect("serialize");
        let deserialized: ChunkBatch = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(batch, deserialized);
    }

    #[test]
    fn test_retrieval_set_serde_roundtrip() {
        let result_set = RetrievalSet {
            id: "results-1".to_string(),
            query: "find function".to_string(),
            results: vec![RankedResult {
                chunk_id: "chunk-1".to_string(),
                score: 0.95,
                scores: ScoreBreakdown {
                    vector_score: 0.9,
                    keyword_score: 0.8,
                    graph_score: 0.7,
                    metadata: Default::default(),
                },
                preview: "fn hello() {}".to_string(),
            }],
            config_hash: "config-1".to_string(),
        };

        let json = serde_json::to_string(&result_set).expect("serialize");
        let deserialized: RetrievalSet = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(result_set, deserialized);
    }

    #[test]
    fn test_entity_graph_delta_serde_roundtrip() {
        let delta = EntityGraphDelta {
            id: "delta-1".to_string(),
            added_nodes: vec![GraphNode {
                id: "node-1".to_string(),
                label: "Function".to_string(),
                properties: serde_json::json!({"name": "hello"}),
            }],
            removed_nodes: vec![],
            added_edges: vec![],
            removed_edges: vec![],
            updated_nodes: vec![],
        };

        let json = serde_json::to_string(&delta).expect("serialize");
        let deserialized: EntityGraphDelta = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(delta, deserialized);
    }
}
