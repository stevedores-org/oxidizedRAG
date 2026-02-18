//! GraphRAG AIVCS Integration
//!
//! Provides run tracking, versioning, and evaluation capabilities for GraphRAG agents
//! using AIVCS (AI Agent Version Control System).
//!
//! This module enables:
//! - Tracking RAG agent runs and experiments
//! - Versioning RAG configurations and knowledge graphs
//! - Evaluating code generation quality
//! - Comparing multi-run results
//! - Integration with AIVCS for version control

pub mod run_recorder;
pub mod config_hasher;
pub mod aivcs_adapter;

pub use run_recorder::RagRunRecorder;
pub use config_hasher::RagConfigDigest;
pub use aivcs_adapter::RagToAivcsAdapter;

/// GraphRAG AIVCS integration version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
