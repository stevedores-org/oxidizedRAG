//! Stage registry for dynamic stage discovery and management.

use std::collections::HashMap;

/// Identifies a stage by name and version.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StageId {
    pub name: String,
    pub version: String,
}

impl StageId {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

impl std::fmt::Display for StageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

/// Registry for discovering and managing stages.
///
/// Stages are registered by (name, version) tuple.
/// This enables deterministic DAG construction from config.
pub struct StageRegistry {
    stages: HashMap<StageId, String>, // Maps StageId to description
}

impl StageRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            stages: HashMap::new(),
        }
    }

    /// Register a stage.
    pub fn register(&mut self, name: &str, version: &str, description: &str) -> bool {
        let id = StageId::new(name, version);
        if self.stages.contains_key(&id) {
            return false; // Already registered
        }
        self.stages.insert(id, description.to_string());
        true
    }

    /// Check if a stage is registered.
    pub fn has_stage(&self, name: &str, version: &str) -> bool {
        self.stages.contains_key(&StageId::new(name, version))
    }

    /// Get stage description.
    pub fn get_description(&self, name: &str, version: &str) -> Option<&str> {
        self.stages
            .get(&StageId::new(name, version))
            .map(|s| s.as_str())
    }

    /// List all registered stages.
    pub fn list_stages(&self) -> Vec<(StageId, String)> {
        self.stages
            .iter()
            .map(|(id, desc)| (id.clone(), desc.clone()))
            .collect()
    }

    /// Get count of registered stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl Default for StageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_id_display() {
        let id = StageId::new("chunker", "1.0.0");
        assert_eq!(id.to_string(), "chunker@1.0.0");
    }

    #[test]
    fn test_registry_register() {
        let mut registry = StageRegistry::new();
        assert!(registry.register("chunker", "1.0.0", "Document chunking stage"));
        assert!(!registry.register("chunker", "1.0.0", "Duplicate"));
    }

    #[test]
    fn test_registry_lookup() {
        let mut registry = StageRegistry::new();
        registry.register("chunker", "1.0.0", "Document chunking");

        assert!(registry.has_stage("chunker", "1.0.0"));
        assert!(!registry.has_stage("chunker", "2.0.0"));
        assert!(!registry.has_stage("embedder", "1.0.0"));

        assert_eq!(
            registry.get_description("chunker", "1.0.0"),
            Some("Document chunking")
        );
    }

    #[test]
    fn test_registry_list() {
        let mut registry = StageRegistry::new();
        registry.register("chunker", "1.0.0", "Chunking");
        registry.register("embedder", "1.0.0", "Embedding");

        let stages = registry.list_stages();
        assert_eq!(stages.len(), 2);
        assert_eq!(registry.stage_count(), 2);
    }
}
