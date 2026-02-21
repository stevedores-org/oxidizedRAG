//! Core batch types for pipeline stages.
//!
//! These types define the contracts between pipeline stages,
//! enabling type-safe composition and explicit data flow.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use super::hashable::ContentHashable;
use sha2::{Sha256, Digest};

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

// ContentHashable implementations for cache key generation

impl ContentHashable for String {
    fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

impl ContentHashable for ChunkBatch {
    fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // Hash corpus_hash (already deterministic)
        hasher.update(self.corpus_hash.as_bytes());

        // Hash chunk IDs in sorted order for determinism
        let mut chunk_ids: Vec<_> = self.chunks.iter().map(|c| &c.id).collect();
        chunk_ids.sort();
        for id in chunk_ids {
            hasher.update(id.as_bytes());
            hasher.update(b"\x00"); // Separator
        }

        format!("{:x}", hasher.finalize())
    }
}

impl ContentHashable for EmbeddingBatch {
    fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // Hash config_hash (already deterministic)
        hasher.update(self.config_hash.as_bytes());

        // Hash embedding IDs in sorted order
        let mut embedding_ids: Vec<_> = self.embeddings.iter().map(|e| &e.chunk_id).collect();
        embedding_ids.sort();
        for id in embedding_ids {
            hasher.update(id.as_bytes());
            hasher.update(b"\x00"); // Separator
        }

        format!("{:x}", hasher.finalize())
    }
}

impl ContentHashable for EntityGraphDelta {
    fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // Hash delta ID as baseline
        hasher.update(self.id.as_bytes());
        hasher.update(b"\x01");

        // Hash added node IDs in sorted order
        let mut added_node_ids: Vec<_> = self.added_nodes.iter().map(|n| &n.id).collect();
        added_node_ids.sort();
        for id in added_node_ids {
            hasher.update(id.as_bytes());
            hasher.update(b"\x00");
        }

        // Hash removed node IDs in sorted order
        let mut removed_nodes = self.removed_nodes.clone();
        removed_nodes.sort();
        for id in &removed_nodes {
            hasher.update(id.as_bytes());
            hasher.update(b"\x00");
        }

        // Hash added edge IDs in sorted order
        let mut added_edge_ids: Vec<_> = self.added_edges.iter().map(|e| &e.id).collect();
        added_edge_ids.sort();
        for id in added_edge_ids {
            hasher.update(id.as_bytes());
            hasher.update(b"\x00");
        }

        // Hash removed edge IDs in sorted order
        let mut removed_edges = self.removed_edges.clone();
        removed_edges.sort();
        for id in &removed_edges {
            hasher.update(id.as_bytes());
            hasher.update(b"\x00");
        }

        // Hash updated node IDs in sorted order
        let mut updated_node_ids: Vec<_> = self.updated_nodes.iter().map(|n| &n.id).collect();
        updated_node_ids.sort();
        for id in updated_node_ids {
            hasher.update(id.as_bytes());
            hasher.update(b"\x00");
        }

        format!("{:x}", hasher.finalize())
    }
}

impl ContentHashable for RetrievalSet {
    fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // Hash query and config
        hasher.update(self.query.as_bytes());
        hasher.update(b"\x01");
        hasher.update(self.config_hash.as_bytes());
        hasher.update(b"\x01");

        // Hash result chunk IDs in sorted order
        let mut result_ids: Vec<_> = self.results.iter().map(|r| &r.chunk_id).collect();
        result_ids.sort();
        for id in result_ids {
            hasher.update(id.as_bytes());
            hasher.update(b"\x00");
        }

        format!("{:x}", hasher.finalize())
    }
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

    #[test]
    fn test_chunk_batch_content_hash_deterministic() {
        let batch = ChunkBatch {
            id: "batch-1".to_string(),
            chunks: vec![
                DocumentChunk {
                    id: "chunk-1".to_string(),
                    content: "content".to_string(),
                    source: "main.rs".to_string(),
                    line_range: Some((1, 1)),
                    metadata: Default::default(),
                },
                DocumentChunk {
                    id: "chunk-2".to_string(),
                    content: "more content".to_string(),
                    source: "lib.rs".to_string(),
                    line_range: Some((2, 3)),
                    metadata: Default::default(),
                },
            ],
            corpus_hash: "abc123".to_string(),
        };

        let hash1 = batch.content_hash();
        let hash2 = batch.content_hash();
        assert_eq!(hash1, hash2, "Same batch should produce same hash");
    }

    #[test]
    fn test_chunk_batch_hash_order_independent() {
        let batch1 = ChunkBatch {
            id: "batch-1".to_string(),
            chunks: vec![
                DocumentChunk {
                    id: "chunk-1".to_string(),
                    content: "content".to_string(),
                    source: "main.rs".to_string(),
                    line_range: None,
                    metadata: Default::default(),
                },
                DocumentChunk {
                    id: "chunk-2".to_string(),
                    content: "more".to_string(),
                    source: "lib.rs".to_string(),
                    line_range: None,
                    metadata: Default::default(),
                },
            ],
            corpus_hash: "abc123".to_string(),
        };

        let batch2 = ChunkBatch {
            id: "batch-1".to_string(),
            chunks: vec![
                DocumentChunk {
                    id: "chunk-2".to_string(),
                    content: "more".to_string(),
                    source: "lib.rs".to_string(),
                    line_range: None,
                    metadata: Default::default(),
                },
                DocumentChunk {
                    id: "chunk-1".to_string(),
                    content: "content".to_string(),
                    source: "main.rs".to_string(),
                    line_range: None,
                    metadata: Default::default(),
                },
            ],
            corpus_hash: "abc123".to_string(),
        };

        assert_eq!(
            batch1.content_hash(),
            batch2.content_hash(),
            "Different order should produce same hash"
        );
    }

    #[test]
    fn test_embedding_batch_content_hash() {
        let batch = EmbeddingBatch {
            id: "batch-1".to_string(),
            embeddings: vec![
                EmbeddingRecord {
                    chunk_id: "chunk-1".to_string(),
                    vector: vec![0.1, 0.2, 0.3],
                    metadata: Default::default(),
                },
                EmbeddingRecord {
                    chunk_id: "chunk-2".to_string(),
                    vector: vec![0.4, 0.5, 0.6],
                    metadata: Default::default(),
                },
            ],
            config_hash: "config-1".to_string(),
        };

        let hash = batch.content_hash();
        assert!(!hash.is_empty(), "Hash should not be empty");
        assert_eq!(hash.len(), 64, "SHA256 hex should be 64 chars");
    }

    #[test]
    fn test_retrieval_set_content_hash() {
        let result_set = RetrievalSet {
            id: "results-1".to_string(),
            query: "find function".to_string(),
            results: vec![
                RankedResult {
                    chunk_id: "chunk-1".to_string(),
                    score: 0.95,
                    scores: ScoreBreakdown {
                        vector_score: 0.9,
                        keyword_score: 0.8,
                        graph_score: 0.7,
                        metadata: Default::default(),
                    },
                    preview: "fn hello() {}".to_string(),
                },
                RankedResult {
                    chunk_id: "chunk-2".to_string(),
                    score: 0.85,
                    scores: ScoreBreakdown {
                        vector_score: 0.8,
                        keyword_score: 0.7,
                        graph_score: 0.6,
                        metadata: Default::default(),
                    },
                    preview: "fn world() {}".to_string(),
                },
            ],
            config_hash: "config-1".to_string(),
        };

        let hash = result_set.content_hash();
        assert!(!hash.is_empty(), "Hash should not be empty");
        assert_eq!(hash.len(), 64, "SHA256 hex should be 64 chars");
    }

    #[test]
    fn test_entity_graph_delta_content_hash() {
        let delta = EntityGraphDelta {
            id: "delta-1".to_string(),
            added_nodes: vec![
                GraphNode {
                    id: "node-1".to_string(),
                    label: "Function".to_string(),
                    properties: serde_json::json!({"name": "hello"}),
                },
                GraphNode {
                    id: "node-2".to_string(),
                    label: "Variable".to_string(),
                    properties: serde_json::json!({"type": "int"}),
                },
            ],
            removed_nodes: vec!["old-node".to_string()],
            added_edges: vec![GraphEdge {
                id: "edge-1".to_string(),
                source: "node-1".to_string(),
                target: "node-2".to_string(),
                rel_type: "calls".to_string(),
                properties: serde_json::json!({}),
            }],
            removed_edges: vec![],
            updated_nodes: vec![],
        };

        let hash = delta.content_hash();
        assert!(!hash.is_empty(), "Hash should not be empty");
        assert_eq!(hash.len(), 64, "SHA256 hex should be 64 chars");
    }
}
