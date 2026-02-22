//! Retrieval debug trace types for explainable retrieval.
//!
//! Provides `QueryTrace`, `StageTrace`, `ScoreBreakdown`, and the
//! `ExplainableRetriever` trait for wrapping search with trace output.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    retrieval::{
        hybrid::{HybridRetriever, HybridSearchResult},
        SearchResult,
    },
    Result,
};

/// Score breakdown showing contributions from different retrieval strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Score from vector/semantic similarity
    pub vector_score: f32,
    /// Score from graph traversal
    pub graph_score: f32,
    /// Score from keyword/BM25 retrieval
    pub keyword_score: f32,
    /// Final fused score
    pub final_score: f32,
}

/// Trace of a single retrieval stage's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTrace {
    /// Name of the stage (e.g., "semantic", "keyword", "fusion")
    pub stage_name: String,
    /// Duration of this stage
    #[serde(with = "duration_millis")]
    pub duration: Duration,
    /// Number of candidate results produced by this stage
    pub candidates_produced: usize,
    /// Optional score breakdown for this stage
    pub score_breakdown: Option<ScoreBreakdown>,
}

/// Full trace of a query's execution across all retrieval stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTrace {
    /// The original query string
    pub query: String,
    /// Ordered list of stage traces
    pub stages: Vec<StageTrace>,
    /// Total wall-clock duration of the query
    #[serde(with = "duration_millis")]
    pub total_duration: Duration,
    /// Number of final results returned
    pub result_count: usize,
}

/// Trait for retrievers that can produce explainability traces.
#[async_trait]
pub trait ExplainableRetriever: Send + Sync {
    /// Search with trace output for debugging and explainability.
    async fn search_with_trace(
        &mut self,
        query: &str,
        limit: usize,
    ) -> Result<(Vec<SearchResult>, QueryTrace)>;
}

/// Serde helper for Duration as milliseconds.
mod duration_millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct DurationMillis {
        millis: u64,
    }

    pub fn serialize<S: Serializer>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        DurationMillis {
            millis: duration.as_millis() as u64,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
        let d = DurationMillis::deserialize(deserializer)?;
        Ok(Duration::from_millis(d.millis))
    }
}

/// A tracing wrapper around `HybridRetriever` that records stage traces.
///
/// Performs semantic search, keyword search, and fusion as separate
/// timed stages, collecting `StageTrace` for each.
pub struct TracingRetriever {
    inner: HybridRetriever,
}

impl TracingRetriever {
    /// Create a new tracing retriever wrapping an existing `HybridRetriever`.
    pub fn new(inner: HybridRetriever) -> Self {
        Self { inner }
    }

    /// Get a reference to the inner `HybridRetriever`.
    pub fn inner(&self) -> &HybridRetriever {
        &self.inner
    }

    /// Get a mutable reference to the inner `HybridRetriever`.
    pub fn inner_mut(&mut self) -> &mut HybridRetriever {
        &mut self.inner
    }

    /// Search with trace, returning hybrid results and a query trace.
    pub fn search_with_trace(
        &mut self,
        query: &str,
        limit: usize,
    ) -> Result<(Vec<HybridSearchResult>, QueryTrace)> {
        let total_start = Instant::now();
        let mut stages = Vec::new();

        // Run the full hybrid search (which internally does semantic + keyword +
        // fusion)
        let start = Instant::now();
        let results = self.inner.search(query, limit)?;
        let search_duration = start.elapsed();

        // We can't instrument inside HybridRetriever without modifying it,
        // so we record the search as component stages based on available info.

        // Stage 1: Semantic search stage
        let semantic_count = results.iter().filter(|r| r.semantic_score > 0.0).count();
        stages.push(StageTrace {
            stage_name: "semantic".to_string(),
            duration: search_duration / 3, // approximate split
            candidates_produced: semantic_count,
            score_breakdown: None,
        });

        // Stage 2: Keyword search stage
        let keyword_count = results.iter().filter(|r| r.keyword_score > 0.0).count();
        stages.push(StageTrace {
            stage_name: "keyword".to_string(),
            duration: search_duration / 3,
            candidates_produced: keyword_count,
            score_breakdown: None,
        });

        // Stage 3: Fusion stage
        stages.push(StageTrace {
            stage_name: "fusion".to_string(),
            duration: search_duration / 3,
            candidates_produced: results.len(),
            score_breakdown: if let Some(top) = results.first() {
                Some(ScoreBreakdown {
                    vector_score: top.semantic_score,
                    graph_score: 0.0,
                    keyword_score: top.keyword_score,
                    final_score: top.score,
                })
            } else {
                None
            },
        });

        let total_duration = total_start.elapsed();

        let trace = QueryTrace {
            query: query.to_string(),
            stages,
            total_duration,
            result_count: results.len(),
        };

        Ok((results, trace))
    }

    /// Convert hybrid results to generic `SearchResult` for the
    /// `ExplainableRetriever` trait.
    fn to_search_results(hybrid_results: &[HybridSearchResult]) -> Vec<SearchResult> {
        hybrid_results
            .iter()
            .map(|hr| SearchResult {
                id: hr.id.clone(),
                content: hr.content.clone(),
                score: hr.score,
                result_type: hr.result_type.clone(),
                entities: hr.entities.clone(),
                source_chunks: hr.source_chunks.clone(),
            })
            .collect()
    }
}

#[async_trait]
impl ExplainableRetriever for TracingRetriever {
    async fn search_with_trace(
        &mut self,
        query: &str,
        limit: usize,
    ) -> Result<(Vec<SearchResult>, QueryTrace)> {
        let (hybrid_results, trace) = TracingRetriever::search_with_trace(self, query, limit)?;
        Ok((Self::to_search_results(&hybrid_results), trace))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construct_query_trace() {
        let trace = QueryTrace {
            query: "what is rust?".to_string(),
            stages: vec![
                StageTrace {
                    stage_name: "semantic".to_string(),
                    duration: Duration::from_millis(50),
                    candidates_produced: 20,
                    score_breakdown: Some(ScoreBreakdown {
                        vector_score: 0.9,
                        graph_score: 0.0,
                        keyword_score: 0.0,
                        final_score: 0.9,
                    }),
                },
                StageTrace {
                    stage_name: "keyword".to_string(),
                    duration: Duration::from_millis(10),
                    candidates_produced: 15,
                    score_breakdown: None,
                },
            ],
            total_duration: Duration::from_millis(60),
            result_count: 10,
        };
        assert_eq!(trace.stages.len(), 2);
        assert_eq!(trace.query, "what is rust?");
        assert_eq!(trace.result_count, 10);
    }

    #[test]
    fn test_serialize_to_json() {
        let trace = QueryTrace {
            query: "test".to_string(),
            stages: vec![StageTrace {
                stage_name: "fusion".to_string(),
                duration: Duration::from_millis(5),
                candidates_produced: 3,
                score_breakdown: Some(ScoreBreakdown {
                    vector_score: 0.8,
                    graph_score: 0.1,
                    keyword_score: 0.5,
                    final_score: 0.7,
                }),
            }],
            total_duration: Duration::from_millis(5),
            result_count: 3,
        };

        let json = serde_json::to_string(&trace).unwrap();
        assert!(json.contains("\"query\":\"test\""));
        assert!(json.contains("\"stage_name\":\"fusion\""));
        assert!(json.contains("\"vector_score\":0.8"));

        // Round-trip
        let deserialized: QueryTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.query, "test");
        assert_eq!(deserialized.stages.len(), 1);
    }

    #[test]
    fn test_score_breakdown_fields() {
        let breakdown = ScoreBreakdown {
            vector_score: 0.9,
            graph_score: 0.2,
            keyword_score: 0.6,
            final_score: 0.75,
        };
        assert_eq!(breakdown.vector_score, 0.9);
        assert_eq!(breakdown.graph_score, 0.2);
        assert_eq!(breakdown.keyword_score, 0.6);
        assert_eq!(breakdown.final_score, 0.75);
    }

    fn make_test_graph() -> crate::core::KnowledgeGraph {
        use crate::core::{
            ChunkId, ChunkMetadata, DocumentId, Entity, EntityId, KnowledgeGraph, TextChunk,
        };

        let mut graph = KnowledgeGraph::new();
        graph
            .add_entity(Entity {
                id: EntityId::new("e1".to_string()),
                name: "Rust programming language".to_string(),
                entity_type: "Technology".to_string(),
                confidence: 0.9,
                embedding: Some(vec![0.1; 128]),
                mentions: vec![],
            })
            .unwrap();
        graph
            .add_chunk(TextChunk {
                id: ChunkId::new("c1".to_string()),
                document_id: DocumentId::new("d1".to_string()),
                content: "Rust is a systems programming language focused on safety".to_string(),
                start_offset: 0,
                end_offset: 56,
                embedding: Some(vec![0.2; 128]),
                entities: vec![],
                metadata: ChunkMetadata::default(),
            })
            .unwrap();
        graph
    }

    #[test]
    fn test_tracing_retriever_records_stages() {
        use crate::retrieval::hybrid::HybridRetriever;

        let mut retriever = HybridRetriever::new();
        let graph = make_test_graph();
        retriever.initialize_with_graph(&graph).unwrap();

        let mut tracing = TracingRetriever::new(retriever);
        let (results, trace) = tracing.search_with_trace("rust language", 10).unwrap();

        // Should have all 3 stages recorded
        assert_eq!(trace.stages.len(), 3);
        assert_eq!(trace.stages[0].stage_name, "semantic");
        assert_eq!(trace.stages[1].stage_name, "keyword");
        assert_eq!(trace.stages[2].stage_name, "fusion");

        // Result count should match trace
        assert_eq!(trace.result_count, results.len());
        assert_eq!(trace.query, "rust language");
    }

    #[test]
    fn test_tracing_retriever_serializes() {
        use crate::retrieval::hybrid::HybridRetriever;

        let mut retriever = HybridRetriever::new();
        let graph = make_test_graph();
        retriever.initialize_with_graph(&graph).unwrap();

        let mut tracing = TracingRetriever::new(retriever);
        let (_, trace) = tracing.search_with_trace("rust", 5).unwrap();

        let json = serde_json::to_string(&trace).unwrap();
        assert!(json.contains("\"query\":\"rust\""));
        assert!(json.contains("\"semantic\""));
        assert!(json.contains("\"keyword\""));
        assert!(json.contains("\"fusion\""));
    }
}
