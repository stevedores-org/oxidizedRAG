//! AIVCS integration for oxidizedRAG.
//!
//! Bridges GraphRAG `ask()` runs to the AIVCS ledger for run tracking,
//! content-addressed config specs, and observability hooks.

pub mod adapter;
pub mod recorder;
pub mod spec;

pub use adapter::RagAdapter;
pub use recorder::RagRunRecorder;
pub use spec::GraphRagSpec;
