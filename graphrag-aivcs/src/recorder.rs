//! Wraps [`aivcs_core::GraphRunRecorder`] with RAG-specific event helpers.

use std::sync::Arc;

use chrono::Utc;
use oxidized_state::{RunEvent, RunId, RunLedger, RunMetadata, RunSummary, StorageError};

use aivcs_core::recording::GraphRunRecorder;

use crate::spec::GraphRagSpec;

/// Records RAG pipeline events (retrieval, LLM calls) into an AIVCS run ledger.
///
/// Uses [`GraphRunRecorder`] for lifecycle (start/finish with observability hooks)
/// and appends RAG-specific events directly to the ledger.
pub struct RagRunRecorder {
    inner: GraphRunRecorder,
    ledger: Arc<dyn RunLedger>,
    seq: u64,
}

impl RagRunRecorder {
    /// Start a new run in the ledger bound to the given spec.
    pub async fn start(
        ledger: Arc<dyn RunLedger>,
        spec: &GraphRagSpec,
    ) -> Result<Self, StorageError> {
        let digest = spec.content_digest();
        let metadata = RunMetadata {
            git_sha: None,
            agent_name: spec.agent_name.clone(),
            tags: serde_json::json!({"source": "graphrag-aivcs"}),
        };
        let inner = GraphRunRecorder::start(ledger.clone(), &digest, metadata).await?;
        Ok(Self {
            inner,
            ledger,
            seq: 0,
        })
    }

    /// Record a retrieval-complete event.
    pub async fn record_retrieval(
        &mut self,
        count: usize,
        score: f32,
        duration_ms: u64,
    ) -> Result<(), StorageError> {
        self.seq += 1;
        let event = RunEvent {
            seq: self.seq,
            kind: "rag.retrieval_complete".to_string(),
            payload: serde_json::json!({
                "retrieved_count": count,
                "score": score,
                "duration_ms": duration_ms,
            }),
            timestamp: Utc::now(),
        };
        self.ledger.append_event(self.inner.run_id(), event).await?;
        aivcs_core::obs::emit_event_appended(
            &self.inner.run_id().to_string(),
            "rag.retrieval_complete",
            self.seq,
        );
        Ok(())
    }

    /// Record an LLM call event.
    pub async fn record_llm_call(
        &mut self,
        prompt_len: usize,
        tokens: u32,
        duration_ms: u64,
    ) -> Result<(), StorageError> {
        self.seq += 1;
        let event = RunEvent {
            seq: self.seq,
            kind: "rag.llm_called".to_string(),
            payload: serde_json::json!({
                "prompt_len": prompt_len,
                "tokens_used": tokens,
                "duration_ms": duration_ms,
            }),
            timestamp: Utc::now(),
        };
        self.ledger.append_event(self.inner.run_id(), event).await?;
        aivcs_core::obs::emit_event_appended(
            &self.inner.run_id().to_string(),
            "rag.llm_called",
            self.seq,
        );
        Ok(())
    }

    /// Finalize the run as completed.
    pub async fn finish_ok(self, total_duration_ms: u64) -> Result<(), StorageError> {
        let summary = RunSummary {
            total_events: self.seq,
            final_state_digest: None,
            duration_ms: total_duration_ms,
            success: true,
        };
        self.inner.finish_ok(summary).await
    }

    /// Finalize the run as failed.
    pub async fn finish_err(self, _error: &str) -> Result<(), StorageError> {
        let summary = RunSummary {
            total_events: self.seq,
            final_state_digest: None,
            duration_ms: 0,
            success: false,
        };
        self.inner.finish_err(summary).await
    }

    /// Return a reference to the underlying run ID.
    pub fn run_id(&self) -> &RunId {
        self.inner.run_id()
    }
}
