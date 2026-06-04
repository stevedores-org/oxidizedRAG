//! AIVCS Integration Adapter - bridges RAG run recorder to AIVCS recording
//! system
//!
//! Converts RagRunRecorder events into AIVCS-compatible events for persistent
//! storage and version control integration.

use serde_json::json;

use crate::run_recorder::RagRunRecorder;

/// Converts a RAG run into AIVCS-compatible events
///
/// This adapter transforms high-level RAG events (retrieval, LLM calls, etc.)
/// into the AIVCS event model for persistent tracking and comparison.
///
/// # Example
///
/// ```ignore
/// let mut recorder = RagRunRecorder::new("What is Rust?");
/// recorder.record_retrieval("chunk 1", 5, 0.95, 100);
/// recorder.record_llm_call("prompt", "response", 150, 500);
///
/// let aivcs_events = RagToAivcsAdapter::convert_run(&recorder, "rag-agent");
/// // aivcs_events contains structured tool call entries for storage
/// ```
pub struct RagToAivcsAdapter;

impl RagToAivcsAdapter {
    /// Convert RAG run events to AIVCS tool-call format
    ///
    /// Each RAG operation (retrieval, LLM call) becomes an AIVCS tool call,
    /// enabling unified run comparison and analysis.
    pub fn convert_run(recorder: &RagRunRecorder, agent_name: &str) -> Vec<serde_json::Value> {
        recorder
            .events()
            .iter()
            .map(|event| {
                json!({
                    "seq": event.seq,
                    "event_type": event.event_type,
                    "tool_name": format!("rag.{}", event.event_type),
                    "timestamp": event.timestamp,
                    "duration_ms": event.duration_ms,
                    "metadata": event.metadata,
                    "agent": agent_name,
                })
            })
            .collect()
    }

    /// Create AIVCS-compatible metadata from RAG run summary
    pub fn summarize_run(recorder: &RagRunRecorder) -> serde_json::Value {
        let summary = recorder.summary();
        json!({
            "run_id": summary.run_id,
            "query": summary.query,
            "event_count": summary.event_count,
            "total_duration_ms": summary.total_duration_ms,
            "retrieval_count": summary.retrieval_count,
            "llm_calls": summary.llm_calls,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_run_creates_tool_calls() {
        let mut recorder = RagRunRecorder::new("test query");
        recorder.record_retrieval("chunk 1", 5, 0.95, 100);
        recorder.record_llm_call("prompt", "response", 150, 500);

        let converted = RagToAivcsAdapter::convert_run(&recorder, "test-agent");
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0]["tool_name"], "rag.retrieval_complete");
        assert_eq!(converted[1]["tool_name"], "rag.llm_called");
    }

    #[test]
    fn test_convert_run_preserves_metadata() {
        let mut recorder = RagRunRecorder::new("test query");
        recorder.record_retrieval("chunk 1", 5, 0.95, 100);

        let converted = RagToAivcsAdapter::convert_run(&recorder, "test-agent");
        assert_eq!(converted[0]["agent"], "test-agent");
        assert_eq!(converted[0]["metadata"]["query_chunk"], "chunk 1");
        assert_eq!(converted[0]["metadata"]["retrieved_count"], 5);
    }

    #[test]
    fn test_summarize_run_includes_metrics() {
        let mut recorder = RagRunRecorder::new("test query");
        recorder.record_retrieval("chunk 1", 5, 0.95, 100);
        recorder.record_llm_call("prompt", "response", 150, 500);

        let summary = RagToAivcsAdapter::summarize_run(&recorder);
        assert_eq!(summary["query"], "test query");
        assert_eq!(summary["event_count"], 2);
        assert_eq!(summary["retrieval_count"], 1);
        assert_eq!(summary["llm_calls"], 1);
    }

    #[test]
    fn test_summarize_run_has_valid_duration() {
        let mut recorder = RagRunRecorder::new("test");
        recorder.record_retrieval("chunk", 1, 0.8, 50);

        let summary = RagToAivcsAdapter::summarize_run(&recorder);
        assert!(summary["total_duration_ms"].is_u64());
        assert_eq!(summary["event_count"], 1);
    }
}
