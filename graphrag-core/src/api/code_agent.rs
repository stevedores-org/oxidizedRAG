//! Code-Agent API layer for codebase indexing and querying.
//!
//! Provides traits and types for indexing source code, querying definitions
//! and references, and building context packs for LLM consumption.

use std::{collections::HashMap, path::PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;

// ============================================================================
// Supporting Types
// ============================================================================

/// A location in source code.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CodeLocation {
    /// File path relative to the repository root.
    pub file: PathBuf,
    /// Line number (1-indexed).
    pub line: usize,
    /// Column number (1-indexed, optional).
    pub column: Option<usize>,
}

/// A code match result from a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMatch {
    /// Location of the match.
    pub location: CodeLocation,
    /// The matched code snippet.
    pub snippet: String,
    /// Relevance score (0.0–1.0).
    pub score: f32,
    /// Language of the matched file.
    pub language: String,
    /// Symbol name if applicable (function, class, etc.).
    pub symbol: Option<String>,
}

/// Statistics about the indexed codebase.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexStats {
    /// Total number of indexed files.
    pub files_indexed: usize,
    /// Total number of indexed symbols (functions, classes, etc.).
    pub symbols_indexed: usize,
    /// Total lines of code indexed.
    pub lines_of_code: usize,
    /// Languages detected and file counts per language.
    pub languages: HashMap<String, usize>,
}

/// A packed context bundle for LLM consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPack {
    /// The query that generated this context.
    pub query: String,
    /// Relevant code snippets, ordered by relevance.
    pub snippets: Vec<CodeMatch>,
    /// Summary of the codebase structure relevant to the query.
    pub structure_summary: Option<String>,
    /// Total token estimate for the context.
    pub estimated_tokens: usize,
}

// ============================================================================
// Traits
// ============================================================================

/// Trait for indexing source code files and directories.
#[async_trait]
pub trait CodebaseIndexer: Send + Sync {
    /// Index a single source file.
    async fn index_file(&mut self, path: &std::path::Path) -> Result<()>;

    /// Recursively index a directory of source files.
    async fn index_directory(&mut self, path: &std::path::Path) -> Result<IndexStats>;

    /// Re-index only files that changed since last indexing.
    async fn reindex_changed(&mut self) -> Result<IndexStats>;
}

/// Trait for querying an indexed codebase.
#[async_trait]
pub trait CodeQueryEngine: Send + Sync {
    /// Find the definition of a symbol by name.
    async fn find_definition(&self, symbol: &str) -> Result<Vec<CodeMatch>>;

    /// Find all references to a symbol.
    async fn find_references(&self, symbol: &str) -> Result<Vec<CodeMatch>>;

    /// Generate an explanation of a function's purpose and behavior.
    async fn explain_function(&self, symbol: &str) -> Result<String>;

    /// Find code similar to the given snippet or description.
    async fn find_similar_code(&self, query: &str, limit: usize) -> Result<Vec<CodeMatch>>;
}

/// Builder for constructing a `ContextPack`.
#[derive(Debug, Default)]
pub struct ContextPackBuilder {
    query: String,
    snippets: Vec<CodeMatch>,
    structure_summary: Option<String>,
    max_tokens: usize,
}

impl ContextPackBuilder {
    /// Create a new builder for the given query.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            snippets: Vec::new(),
            structure_summary: None,
            max_tokens: 4096,
        }
    }

    /// Add a code match snippet.
    pub fn add_snippet(mut self, snippet: CodeMatch) -> Self {
        self.snippets.push(snippet);
        self
    }

    /// Set the structure summary.
    pub fn with_structure_summary(mut self, summary: impl Into<String>) -> Self {
        self.structure_summary = Some(summary.into());
        self
    }

    /// Set the maximum token budget.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Build the context pack, estimating tokens from snippet lengths.
    pub fn build(self) -> ContextPack {
        // Rough token estimate: ~4 chars per token
        let estimated_tokens: usize = self
            .snippets
            .iter()
            .map(|s| s.snippet.len() / 4)
            .sum::<usize>()
            + self
                .structure_summary
                .as_ref()
                .map(|s| s.len() / 4)
                .unwrap_or(0);

        ContextPack {
            query: self.query,
            snippets: self.snippets,
            structure_summary: self.structure_summary,
            estimated_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_objects_compile() {
        // Verify trait objects can be created (compile test)
        fn _takes_indexer(_: &dyn CodebaseIndexer) {}
        fn _takes_query_engine(_: &dyn CodeQueryEngine) {}
    }

    #[test]
    fn test_context_pack_builder() {
        let snippet = CodeMatch {
            location: CodeLocation {
                file: PathBuf::from("src/main.rs"),
                line: 10,
                column: Some(1),
            },
            snippet: "fn main() { println!(\"hello\"); }".to_string(),
            score: 0.95,
            language: "rust".to_string(),
            symbol: Some("main".to_string()),
        };

        let pack = ContextPackBuilder::new("what does main do?")
            .add_snippet(snippet)
            .with_structure_summary("Entry point of the application")
            .with_max_tokens(2048)
            .build();

        assert_eq!(pack.query, "what does main do?");
        assert_eq!(pack.snippets.len(), 1);
        assert!(pack.structure_summary.is_some());
        assert!(pack.estimated_tokens > 0);
    }

    #[test]
    fn test_code_location_serde() {
        let loc = CodeLocation {
            file: PathBuf::from("src/lib.rs"),
            line: 42,
            column: Some(5),
        };
        let json = serde_json::to_string(&loc).unwrap();
        let deserialized: CodeLocation = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, loc);
    }

    #[test]
    fn test_index_stats_default() {
        let stats = IndexStats::default();
        assert_eq!(stats.files_indexed, 0);
        assert_eq!(stats.symbols_indexed, 0);
        assert_eq!(stats.lines_of_code, 0);
        assert!(stats.languages.is_empty());
    }

    #[test]
    fn test_empty_context_pack() {
        let pack = ContextPackBuilder::new("empty query").build();
        assert_eq!(pack.query, "empty query");
        assert!(pack.snippets.is_empty());
        assert_eq!(pack.estimated_tokens, 0);
    }
}
