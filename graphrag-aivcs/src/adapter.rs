//! Convenience factory that bundles a [`RunLedger`] for easy wiring.

use std::sync::Arc;

use oxidized_state::{fakes::MemoryRunLedger, RunLedger, StorageError};

use crate::recorder::RagRunRecorder;
use crate::spec::GraphRagSpec;

/// Convenience wrapper that holds a ledger and creates [`RagRunRecorder`]s.
pub struct RagAdapter {
    ledger: Arc<dyn RunLedger>,
}

impl RagAdapter {
    /// Create an adapter backed by an in-memory ledger (useful for tests).
    pub fn in_memory() -> Self {
        Self {
            ledger: Arc::new(MemoryRunLedger::new()),
        }
    }

    /// Create an adapter with a caller-provided ledger.
    pub fn with_ledger(ledger: Arc<dyn RunLedger>) -> Self {
        Self { ledger }
    }

    /// Start a new tracked RAG run.
    pub async fn start_run(&self, spec: &GraphRagSpec) -> Result<RagRunRecorder, StorageError> {
        RagRunRecorder::start(self.ledger.clone(), spec).await
    }

    /// Return a reference to the underlying ledger (useful for assertions in tests).
    pub fn ledger(&self) -> &Arc<dyn RunLedger> {
        &self.ledger
    }
}
