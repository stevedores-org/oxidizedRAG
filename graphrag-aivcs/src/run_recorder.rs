//! RAG Run Recorder - tracks RAG agent executions using AIVCS
//!
//! Records query processing, retrieval operations, and LLM interactions
//! as discrete events in an AIVCS run for later analysis and comparison.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use uuid::Uuid;

/// Represents a single RAG query execution event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagEvent {
    /// Unique event ID
    pub id: String,
    /// Sequence number (starts at 1)
    pub seq: u64,
    /// Event type (query_started, retrieval_complete, llm_called, response_generated, etc.)
    pub event_type: String,
    /// Event timestamp (ISO 8601)
    pub timestamp: String,
    /// Event metadata (query, retrieved_chunks, tokens_used, etc.)
    pub metadata: serde_json::Value,
    /// Duration in milliseconds (for operation events)
    pub duration_ms: Option<u64>,
}

/// Tracks a complete RAG query run with all events
#[derive(Debug)]
pub struct RagRunRecorder {
    run_id: String,
    query: String,
    events: Vec<RagEvent>,
    event_seq: u64,
    start_time: Instant,
}

impl RagRunRecorder {
    /// Create a new RAG run recorder for a query
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            run_id: Uuid::new_v4().to_string(),
            query: query.into(),
            events: Vec::new(),
            event_seq: 0,
            start_time: Instant::now(),
        }
    }

    /// Get the run ID
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Get the original query
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Record a RAG event (retrieval, LLM call, etc.)
    pub fn record_event(
        &mut self,
        event_type: impl Into<String>,
        metadata: serde_json::Value,
        duration_ms: Option<u64>,
    ) {
        self.event_seq += 1;
        let event = RagEvent {
            id: Uuid::new_v4().to_string(),
            seq: self.event_seq,
            event_type: event_type.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            metadata,
            duration_ms,
        };
        self.events.push(event);
    }

    /// Record a retrieval event
    pub fn record_retrieval(
        &mut self,
        query_chunk: &str,
        retrieved_count: usize,
        score: f32,
        duration_ms: u64,
    ) {
        self.record_event(
            "retrieval_complete",
            serde_json::json!({
                "query_chunk": query_chunk,
                "retrieved_count": retrieved_count,
                "score": score,
            }),
            Some(duration_ms),
        );
    }

    /// Record an LLM call event
    pub fn record_llm_call(
        &mut self,
        prompt: &str,
        response: &str,
        tokens_used: u32,
        duration_ms: u64,
    ) {
        self.record_event(
            "llm_called",
            serde_json::json!({
                "prompt_length": prompt.len(),
                "response": response,
                "tokens_used": tokens_used,
            }),
            Some(duration_ms),
        );
    }

    /// Get all recorded events
    pub fn events(&self) -> &[RagEvent] {
        &self.events
    }

    /// Get total run duration
    pub fn total_duration_ms(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }

    /// Create a summary of the run
    pub fn summary(&self) -> RagRunSummary {
        RagRunSummary {
            run_id: self.run_id.clone(),
            query: self.query.clone(),
            event_count: self.events.len(),
            total_duration_ms: self.total_duration_ms(),
            retrieval_count: self
                .events
                .iter()
                .filter(|e| e.event_type == "retrieval_complete")
                .count(),
            llm_calls: self
                .events
                .iter()
                .filter(|e| e.event_type == "llm_called")
                .count(),
        }
    }
}

/// Summary of a RAG run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagRunSummary {
    pub run_id: String,
    pub query: String,
    pub event_count: usize,
    pub total_duration_ms: u128,
    pub retrieval_count: usize,
    pub llm_calls: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_run_recorder() {
        let recorder = RagRunRecorder::new("What is Rust?");
        assert_eq!(recorder.query(), "What is Rust?");
        assert!(!recorder.run_id().is_empty());
        assert_eq!(recorder.events().len(), 0);
    }

    #[test]
    fn test_record_retrieval_event() {
        let mut recorder = RagRunRecorder::new("test query");
        recorder.record_retrieval("chunk 1", 5, 0.95, 100);

        assert_eq!(recorder.events().len(), 1);
        assert_eq!(recorder.events()[0].event_type, "retrieval_complete");
        assert_eq!(recorder.events()[0].seq, 1);
    }

    #[test]
    fn test_record_llm_call() {
        let mut recorder = RagRunRecorder::new("test query");
        recorder.record_llm_call("prompt", "response", 150, 500);

        assert_eq!(recorder.events().len(), 1);
        assert_eq!(recorder.events()[0].event_type, "llm_called");
        assert_eq!(recorder.events()[0].seq, 1);
    }

    #[test]
    fn test_run_summary() {
        let mut recorder = RagRunRecorder::new("test query");
        recorder.record_retrieval("chunk", 5, 0.95, 100);
        recorder.record_llm_call("prompt", "response", 150, 500);

        let summary = recorder.summary();
        assert_eq!(summary.event_count, 2);
        assert_eq!(summary.retrieval_count, 1);
        assert_eq!(summary.llm_calls, 1);
    }
}
