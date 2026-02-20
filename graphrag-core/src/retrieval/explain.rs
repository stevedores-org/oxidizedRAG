//! Retrieval debug trace types for explainable retrieval.
//!
//! Provides `QueryTrace`, `StageTrace`, `ScoreBreakdown`, and the
//! `ExplainableRetriever` trait for wrapping search with trace output.

use crate::retrieval::SearchResult;
use crate::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

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
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

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
}
