//! Pipeline builder - construct DAGs from configuration.
//!
//! TODO (Story 1.2): Implement config-driven pipeline construction
//! - Parse TOML/YAML to stage sequence
//! - Resolve dependencies
//! - Validate no cycles
//! - Hash pipeline for content-addressing

pub struct PipelineBuilder;

impl PipelineBuilder {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
