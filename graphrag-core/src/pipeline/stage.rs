//! Stage trait - the core abstraction for pipeline composition.
//!
//! Defines the interface for pipeline stages with typed inputs/outputs,
//! enabling compile-time validation and swappable implementations.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Error type for stage execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageError {
    pub stage_name: String,
    pub message: String,
    pub details: Option<String>,
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}{}",
            self.stage_name,
            self.message,
            self.details
                .as_ref()
                .map(|d| format!(": {}", d))
                .unwrap_or_default()
        )
    }
}

impl std::error::Error for StageError {}

/// A pipeline stage with typed input/output contracts.
///
/// Each stage is independently testable and composable.
/// The generic parameters I and O enforce type safety at compile time.
#[async_trait]
pub trait Stage<I, O>: Send + Sync
where
    I: Send + Sync,
    O: Send + Sync,
{
    /// Execute the stage on the given input.
    async fn execute(&self, input: I) -> Result<O, StageError>;

    /// Human-readable stage name.
    fn name(&self) -> &str;

    /// Stage version for compatibility checking.
    fn version(&self) -> &str;

    /// Optional: metadata about the stage.
    fn metadata(&self) -> StageMeta {
        StageMeta::default()
    }
}

/// Metadata about a stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageMeta {
    /// Human description
    pub description: Option<String>,
    /// Author/maintainer
    pub author: Option<String>,
    /// Is this stage deterministic?
    pub deterministic: bool,
    /// Typical execution time (ms)
    pub typical_latency_ms: Option<u64>,
    /// Memory footprint estimate (MB)
    pub memory_mb: Option<u64>,
    /// Configuration schema (JSON schema)
    pub config_schema: Option<serde_json::Value>,
}

impl Default for StageMeta {
    fn default() -> Self {
        Self {
            description: None,
            author: None,
            deterministic: true,
            typical_latency_ms: None,
            memory_mb: None,
            config_schema: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock stage for testing.
    struct MockStage;

    #[async_trait]
    impl Stage<String, String> for MockStage {
        async fn execute(&self, input: String) -> Result<String, StageError> {
            Ok(input.to_uppercase())
        }

        fn name(&self) -> &str {
            "mock-stage"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }
    }

    #[tokio::test]
    async fn test_stage_execution() {
        let stage = MockStage;
        let result = stage.execute("hello".to_string()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "HELLO");
    }

    #[tokio::test]
    async fn test_stage_metadata() {
        let stage = MockStage;
        assert_eq!(stage.name(), "mock-stage");
        assert_eq!(stage.version(), "1.0.0");
        assert!(stage.metadata().deterministic);
    }

    #[test]
    fn test_stage_error_display() {
        let err = StageError {
            stage_name: "test".to_string(),
            message: "failed".to_string(),
            details: Some("details".to_string()),
        };
        assert!(err.to_string().contains("[test]"));
        assert!(err.to_string().contains("failed"));
    }
}
