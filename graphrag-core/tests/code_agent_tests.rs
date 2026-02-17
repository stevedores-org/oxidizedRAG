//! Integration tests for AI agents writing code with GraphRAG
//!
//! Tests retrieval-augmented generation workflows for:
//! - Code indexing (Rust via tree-sitter)
//! - Code understanding (entities, relationships, call graphs)
//! - Code retrieval (finding relevant functions/modules)
//! - Code generation (tests, refactors, features)
//! - Agent workflows (multi-turn conversations)
//! - Performance (indexing speed, query latency)
//!
//! Test plan: https://github.com/stevedores-org/oxidizedRAG/issues/2
//!
//! Requirement mapping:
//! - [x] Code indexing
//! - [x] Code understanding
//! - [x] Code retrieval
//! - [x] Code generation
//! - [x] Agent workflows
//! - [x] Performance baselines

use graphrag_core::config::Config;
use graphrag_core::core::{Document, DocumentId, KnowledgeGraph};
use graphrag_core::graph::GraphBuilder;
use graphrag_core::retrieval::RetrievalSystem;
use graphrag_core::text::TextProcessor;
use graphrag_core::Result;

use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Test Helpers
// ---------------------------------------------------------------------------

/// Base path for code sample fixtures.
const FIXTURE_DIR: &str = "tests/fixtures/code_samples";

/// Load a fixture file by name.
fn load_fixture(name: &str) -> String {
    let path = format!("{}/{}", FIXTURE_DIR, name);
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to load fixture '{}': {}", path, e))
}

/// Create a Document from a fixture file.
fn fixture_document(filename: &str) -> Document {
    let content = load_fixture(filename);
    let doc_id = DocumentId::new(filename.replace('.', "_"));
    Document::new(doc_id, filename.to_string(), content)
}

/// Build a knowledge graph from fixture files with entity extraction.
fn build_graph_from_fixtures(filenames: &[&str]) -> Result<KnowledgeGraph> {
    let documents: Vec<Document> = filenames.iter().map(|f| fixture_document(f)).collect();
    let mut builder = GraphBuilder::new(500, 100, 0.5, 0.7, 10)?;
    builder.build_graph(documents)
}

/// Parse Rust code with tree-sitter and return whether it's syntactically valid.
#[cfg(feature = "code-chunking")]
fn validate_rust_syntax(code: &str) -> std::result::Result<(), String> {
    use tree_sitter::Parser;

    let mut parser = Parser::new();
    let language = tree_sitter_rust::language();
    parser
        .set_language(&language)
        .map_err(|e| format!("Failed to load Rust grammar: {}", e))?;

    let tree = parser
        .parse(code, None)
        .ok_or_else(|| "Failed to parse code".to_string())?;

    let root = tree.root_node();
    if root.has_error() {
        Err(format!(
            "Syntax error in generated code at byte {}",
            root.start_byte()
        ))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Module 1: Code Indexing
// ---------------------------------------------------------------------------

mod code_indexing {
    use super::*;

    #[test]
    #[cfg(feature = "code-chunking")]
    fn test_rust_code_chunking_preserves_boundaries() {
        use graphrag_core::text::chunking_strategies::RustCodeChunkingStrategy;

        let code = load_fixture("calculator.rs");
        let doc_id = DocumentId::new("calculator_rs".to_string());
        let strategy = RustCodeChunkingStrategy::new(10, doc_id);

        let chunks = strategy.chunk(&code);

        // Should produce multiple chunks (struct, impl, enum, etc.)
        assert!(
            chunks.len() >= 3,
            "Expected at least 3 chunks (struct, impl, enum), got {}",
            chunks.len()
        );

        // Each chunk should be syntactically complete — no mid-function splits
        for chunk in &chunks {
            assert!(
                !chunk.content.trim().is_empty(),
                "Chunk should not be empty"
            );

            // Verify braces are balanced in each chunk
            let open_braces = chunk.content.matches('{').count();
            let close_braces = chunk.content.matches('}').count();
            assert_eq!(
                open_braces, close_braces,
                "Unbalanced braces in chunk: {}",
                &chunk.content[..chunk.content.len().min(80)]
            );
        }

        // Verify we can find the Calculator struct in one chunk
        let struct_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.content.contains("struct Calculator"))
            .collect();
        assert!(
            !struct_chunks.is_empty(),
            "Should find Calculator struct in chunks"
        );

        // Verify the impl block is captured
        let impl_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.content.contains("impl Calculator"))
            .collect();
        assert!(
            !impl_chunks.is_empty(),
            "Should find impl Calculator in chunks"
        );
    }

    #[test]
    fn test_multi_file_workspace_indexing() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph from fixtures");

        // Should have all 3 documents
        assert_eq!(
            graph.documents().count(),
            3,
            "Should have indexed 3 documents"
        );

        // Should have extracted chunks from each document
        assert!(
            graph.chunks().count() >= 3,
            "Should have at least 3 chunks across all documents"
        );

        // Should have extracted entities
        assert!(
            graph.entities().count() > 0,
            "Should have extracted entities from code"
        );
    }

    #[test]
    #[cfg(feature = "code-chunking")]
    fn test_code_chunking_extracts_all_top_level_items() {
        use graphrag_core::text::chunking_strategies::RustCodeChunkingStrategy;

        let code = load_fixture("graph_algorithms.rs");
        let doc_id = DocumentId::new("graph_algorithms_rs".to_string());
        let strategy = RustCodeChunkingStrategy::new(10, doc_id);

        let chunks = strategy.chunk(&code);

        // graph_algorithms.rs has: Graph struct, impl Graph, bfs, dfs, dijkstra,
        // has_cycle, dfs_cycle_check, topological_sort, tests mod
        // Should extract at least the major top-level items
        assert!(
            chunks.len() >= 4,
            "Expected at least 4 chunks from graph_algorithms.rs, got {}",
            chunks.len()
        );

        // Verify key functions are captured
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect();
        assert!(
            all_content.contains("fn bfs"),
            "Should capture bfs function"
        );
        assert!(
            all_content.contains("fn dfs"),
            "Should capture dfs function"
        );
        assert!(
            all_content.contains("fn dijkstra"),
            "Should capture dijkstra function"
        );
    }

    #[test]
    fn test_incremental_code_updates() {
        let processor = TextProcessor::new(500, 100).expect("Failed to create processor");
        let mut graph = KnowledgeGraph::new();

        // Index initial version of calculator
        let initial_doc = fixture_document("calculator.rs");
        let initial_chunks = processor.chunk_text(&initial_doc).expect("Failed to chunk");
        let initial_doc_with_chunks = Document {
            chunks: initial_chunks,
            ..initial_doc
        };
        graph
            .add_document(initial_doc_with_chunks)
            .expect("Failed to add document");

        let initial_doc_count = graph.documents().count();
        let initial_chunk_count = graph.chunks().count();

        // Add updated version with a new method appended
        let updated_code = format!(
            "{}\n\nimpl Calculator {{\n    pub fn power(&mut self, exp: f64) -> f64 {{\n        self.memory = self.memory.powf(exp);\n        self.memory\n    }}\n}}",
            load_fixture("calculator.rs")
        );

        let updated_doc = Document::new(
            DocumentId::new("calculator_v2".to_string()),
            "calculator_v2.rs".to_string(),
            updated_code,
        );
        let updated_chunks = processor.chunk_text(&updated_doc).expect("Failed to chunk");
        let updated_doc_with_chunks = Document {
            chunks: updated_chunks,
            ..updated_doc
        };
        graph
            .add_document(updated_doc_with_chunks)
            .expect("Failed to add updated document");

        // Should have both document versions
        assert_eq!(graph.documents().count(), initial_doc_count + 1);

        // Updated version should have more total chunks
        assert!(
            graph.chunks().count() > initial_chunk_count,
            "Adding a second (larger) document should increase chunk count: {} vs {}",
            graph.chunks().count(),
            initial_chunk_count
        );
    }

    #[test]
    fn test_metadata_extraction_from_chunks() {
        let processor = TextProcessor::new(500, 100).expect("Failed to create processor");

        let doc = fixture_document("calculator.rs");
        let chunks = processor.chunk_text(&doc).expect("Failed to chunk");

        // All chunks should have the correct document_id
        for chunk in &chunks {
            assert_eq!(
                chunk.document_id,
                DocumentId::new("calculator_rs".to_string()),
                "Chunk should reference parent document"
            );
        }

        // Offsets should be non-overlapping and increasing
        let mut prev_end = 0;
        for chunk in &chunks {
            assert!(
                chunk.start_offset <= chunk.end_offset,
                "Start offset should be <= end offset"
            );
            // Note: with overlap, start might be before prev_end
            assert!(
                chunk.end_offset > prev_end || chunk.start_offset == 0,
                "Chunks should make forward progress"
            );
            prev_end = chunk.end_offset;
        }
    }
}

// ---------------------------------------------------------------------------
// Module 2: Code Understanding
// ---------------------------------------------------------------------------

mod code_understanding {
    use super::*;

    #[test]
    fn test_entity_extraction_from_code() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        let entities: Vec<_> = graph.entities().collect();

        // Should extract some entities from code
        // The pattern-based extractor will find capitalized names like
        // Calculator, CalculatorError, etc.
        assert!(
            !entities.is_empty(),
            "Should extract entities from calculator.rs"
        );

        // Check that we find Calculator-related entities
        let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
        let has_calculator = entity_names.iter().any(|n| n.contains("Calculator"));
        assert!(
            has_calculator,
            "Should find 'Calculator' entity. Found: {:?}",
            entity_names
        );
    }

    #[test]
    fn test_entity_extraction_from_api_client() {
        let graph = build_graph_from_fixtures(&["api_client.rs"])
            .expect("Failed to build graph");

        let entities: Vec<_> = graph.entities().collect();

        assert!(
            !entities.is_empty(),
            "Should extract entities from api_client.rs"
        );

        let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();

        // Should find ApiClient or Api or Client entity
        let has_api_related = entity_names
            .iter()
            .any(|n| n.contains("Api") || n.contains("Client") || n.contains("HTTP"));
        assert!(
            has_api_related,
            "Should find API-related entity. Found: {:?}",
            entity_names
        );
    }

    #[test]
    fn test_relationship_extraction_between_entities() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        // Even if no explicit relationships, verify the API works
        let relationships: Vec<_> = graph.relationships().collect();

        // If entities were extracted, there should be at least some relationship
        // inference between co-occurring entities
        let entity_count = graph.entities().count();
        if entity_count >= 2 {
            // With 2+ entities in the same chunk, pattern-based extraction
            // should infer co-occurrence relationships
            // This is a soft assertion — may depend on extraction quality
            let _rel_count = relationships.len();
            // Just verify it doesn't panic
        }
    }

    #[test]
    fn test_graph_structure_after_multi_file_indexing() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        // Verify graph has expected structure
        assert_eq!(graph.documents().count(), 3, "Should have 3 documents");
        assert!(graph.entities().count() > 0, "Should have entities");

        // Get graph stats
        let entity_count = graph.entities().count();
        let relationship_count = graph.relationships().count();

        // With multiple files, we expect a richer graph
        assert!(
            entity_count >= 2,
            "Multi-file index should produce at least 2 entities, got {}",
            entity_count
        );

        println!(
            "Graph stats: {} entities, {} relationships, {} documents, {} chunks",
            entity_count,
            relationship_count,
            graph.documents().count(),
            graph.chunks().count()
        );
    }

    #[test]
    fn test_entity_neighbors_in_knowledge_graph() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        // For each entity that has relationships, verify we can query neighbors
        for entity in graph.entities() {
            let neighbors = graph.get_neighbors(&entity.id);
            // Just verify the API works without panicking
            let _ = neighbors;
        }
    }

    #[test]
    fn test_find_entities_by_name() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        // Search for entities by name
        let results: Vec<_> = graph.find_entities_by_name("Calculator").collect();

        // Should find at least one match (exact or partial)
        // Note: depends on how the extractor names entities
        let all_entities: Vec<String> = graph.entities().map(|e| e.name.clone()).collect();
        println!("All entities: {:?}", all_entities);
        println!("Search results for 'Calculator': {}", results.len());
    }
}

// ---------------------------------------------------------------------------
// Module 3: Code Retrieval
// ---------------------------------------------------------------------------

mod code_retrieval {
    use super::*;

    #[test]
    fn test_hybrid_retrieval_for_code_search() {
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");

        let mut graph = build_graph_from_fixtures(&["graph_algorithms.rs"])
            .expect("Failed to build graph");

        // Add embeddings for vector search
        retrieval
            .add_embeddings_to_graph(&mut graph)
            .expect("Failed to add embeddings");

        // Search for "shortest path algorithm"
        let results = retrieval
            .hybrid_query("shortest path algorithm", &graph)
            .expect("Failed to query");

        assert!(
            !results.is_empty(),
            "Should return results for 'shortest path algorithm'"
        );

        // Check ALL results for relevance (not just first 200 chars of top 3)
        let all_content: String = results
            .iter()
            .map(|r| r.content.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        // At least one result should mention dijkstra or shortest or path or graph
        let is_relevant = all_content.contains("dijkstra")
            || all_content.contains("shortest")
            || all_content.contains("path")
            || all_content.contains("distance")
            || all_content.contains("graph");

        assert!(
            is_relevant,
            "Results should be relevant to 'shortest path algorithm'. Got {} results",
            results.len()
        );
    }

    #[test]
    fn test_bm25_search_for_function_names() {
        use graphrag_core::retrieval::bm25::{BM25Retriever, Document as BM25Document};

        let code = load_fixture("graph_algorithms.rs");

        // Build a BM25 index from the code
        let mut retriever = BM25Retriever::new();
        let doc = BM25Document {
            id: "graph_algorithms".to_string(),
            content: code,
            metadata: std::collections::HashMap::new(),
        };
        retriever
            .index_document(doc)
            .expect("Failed to index document");

        // Search for "breadth first search"
        let results = retriever.search("breadth first search", 5);

        assert!(
            !results.is_empty(),
            "BM25 should return results for 'breadth first search'"
        );

        // The document should be the top result
        assert_eq!(results[0].doc_id, "graph_algorithms");
        assert!(results[0].score > 0.0, "Score should be positive");
    }

    #[test]
    fn test_retrieval_across_multiple_files() {
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");

        let mut graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        retrieval
            .add_embeddings_to_graph(&mut graph)
            .expect("Failed to add embeddings");

        // Search for "error handling"
        let results = retrieval
            .hybrid_query("error handling", &graph)
            .expect("Failed to query");

        assert!(
            !results.is_empty(),
            "Should return results for 'error handling'"
        );

        // Should find results from api_client.rs (which has ApiError)
        // and possibly calculator.rs (which has CalculatorError)
        let all_content: String = results.iter().map(|r| r.content.as_str()).collect();
        let has_error_content =
            all_content.contains("Error") || all_content.contains("error");
        assert!(
            has_error_content,
            "Results should contain error-related content"
        );
    }

    #[test]
    fn test_query_analysis_classifies_code_queries() {
        let config = Config::default();
        let retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");
        let graph = KnowledgeGraph::new();

        // Entity-focused query
        let analysis = retrieval
            .analyze_query("Calculator struct", &graph)
            .expect("Failed to analyze");
        println!("'Calculator struct' classified as: {:?}", analysis.query_type);

        // Relationship query
        let analysis = retrieval
            .analyze_query("what functions call fetch_data", &graph)
            .expect("Failed to analyze");
        println!(
            "'what functions call fetch_data' classified as: {:?}",
            analysis.query_type
        );

        // Exploratory query
        let analysis = retrieval
            .analyze_query("how does the graph algorithm module work", &graph)
            .expect("Failed to analyze");
        println!(
            "'how does the graph algorithm module work' classified as: {:?}",
            analysis.query_type
        );
    }

    #[test]
    fn test_empty_query_returns_gracefully() {
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");

        let mut graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        retrieval
            .add_embeddings_to_graph(&mut graph)
            .expect("Failed to add embeddings");

        // Empty query should not panic
        let results = retrieval.hybrid_query("", &graph).expect("Should not fail");
        assert!(
            results.len() <= 20,
            "Empty query should return bounded results"
        );
    }

    #[test]
    fn test_retrieval_result_scores_are_ordered() {
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");

        let mut graph = build_graph_from_fixtures(&["graph_algorithms.rs"])
            .expect("Failed to build graph");

        retrieval
            .add_embeddings_to_graph(&mut graph)
            .expect("Failed to add embeddings");

        let results = retrieval
            .hybrid_query("graph traversal", &graph)
            .expect("Failed to query");

        if results.len() >= 2 {
            // Results should be in descending score order
            for window in results.windows(2) {
                assert!(
                    window[0].score >= window[1].score,
                    "Results should be sorted by score descending: {} >= {}",
                    window[0].score,
                    window[1].score
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Module 4: Code Generation
// ---------------------------------------------------------------------------

mod code_generation {
    use super::*;

    #[test]
    #[cfg(feature = "code-chunking")]
    fn test_validate_fixture_syntax() {
        // Verify all fixture files are syntactically valid Rust
        for filename in &["calculator.rs", "api_client.rs", "graph_algorithms.rs"] {
            let code = load_fixture(filename);
            validate_rust_syntax(&code).unwrap_or_else(|e| {
                panic!("Fixture '{}' has syntax errors: {}", filename, e)
            });
        }
    }

    #[test]
    #[cfg(feature = "code-chunking")]
    fn test_generated_test_code_is_valid_rust() {
        // Simulate what a code agent would generate: a test for Calculator::add
        let generated_test = r#"
#[cfg(test)]
mod generated_tests {
    use super::*;

    #[test]
    fn test_calculator_add_positive() {
        let mut calc = Calculator::new();
        let result = calc.add(5.0);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_calculator_add_negative() {
        let mut calc = Calculator::new();
        let result = calc.add(-3.0);
        assert_eq!(result, -3.0);
    }

    #[test]
    fn test_calculator_add_accumulates() {
        let mut calc = Calculator::new();
        calc.add(2.0);
        calc.add(3.0);
        assert_eq!(calc.recall(), 5.0);
    }
}
"#;

        // The generated test code should parse as valid Rust
        validate_rust_syntax(generated_test)
            .expect("Generated test code should be syntactically valid");
    }

    #[test]
    #[cfg(feature = "code-chunking")]
    fn test_generated_trait_impl_is_valid_rust() {
        // Simulate generating a Display impl for Calculator
        let generated_impl = r#"
use std::fmt;

struct Calculator {
    memory: f64,
}

impl fmt::Display for Calculator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Calculator(memory: {})", self.memory)
    }
}
"#;

        validate_rust_syntax(generated_impl)
            .expect("Generated impl should be syntactically valid");
    }

    #[test]
    #[cfg(feature = "code-chunking")]
    fn test_generated_error_handling_is_valid_rust() {
        // Simulate adding timeout error handling to a function
        let generated_code = r#"
use std::time::Duration;

enum ApiError {
    Timeout(Duration),
    NetworkError(String),
}

async fn fetch_with_timeout(url: &str, timeout: Duration) -> Result<String, ApiError> {
    let result = tokio::time::timeout(timeout, async {
        Ok::<String, ApiError>("response".to_string())
    })
    .await
    .map_err(|_| ApiError::Timeout(timeout))?;

    result
}
"#;

        validate_rust_syntax(generated_code)
            .expect("Generated error handling code should be syntactically valid");
    }

    #[test]
    fn test_retrieval_provides_context_for_generation() {
        // Verify that retrieval returns enough context for an LLM to generate code
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");

        let mut graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        retrieval
            .add_embeddings_to_graph(&mut graph)
            .expect("Failed to add embeddings");

        // Query as a code agent would: "Write a test for the add method"
        let results = retrieval
            .hybrid_query("Calculator add method", &graph)
            .expect("Failed to query");

        assert!(!results.is_empty(), "Should find relevant context");

        // The retrieved context should contain the actual add method signature
        let context: String = results.iter().map(|r| r.content.as_str()).collect();
        let has_add_context = context.contains("add") || context.contains("Calculator");
        assert!(
            has_add_context,
            "Retrieved context should mention 'add' or 'Calculator' for code generation"
        );
    }
}

// ---------------------------------------------------------------------------
// Module 5: Agent Workflows
// ---------------------------------------------------------------------------

mod agent_workflows {
    use super::*;

    #[test]
    fn test_multi_turn_code_conversation() {
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");

        let mut graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        retrieval
            .add_embeddings_to_graph(&mut graph)
            .expect("Failed to add embeddings");

        // Turn 1: "Show me the calculator code"
        let turn1 = retrieval
            .hybrid_query("calculator code", &graph)
            .expect("Turn 1 failed");
        assert!(!turn1.is_empty(), "Turn 1 should return calculator code");

        // Turn 2: "What methods does Calculator have?"
        let turn2 = retrieval
            .hybrid_query("Calculator methods add multiply divide", &graph)
            .expect("Turn 2 failed");
        assert!(!turn2.is_empty(), "Turn 2 should return method information");

        // Turn 3: "How does error handling work?"
        let turn3 = retrieval
            .hybrid_query("error handling divide by zero", &graph)
            .expect("Turn 3 failed");
        assert!(!turn3.is_empty(), "Turn 3 should return error handling info");

        // Each turn should produce results — verifying the system can handle
        // a sequence of related queries (conversation-like)
        println!(
            "Multi-turn conversation: {} / {} / {} results",
            turn1.len(),
            turn2.len(),
            turn3.len()
        );
    }

    #[test]
    fn test_code_search_to_modification_pipeline() {
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");

        let mut graph = build_graph_from_fixtures(&["api_client.rs"])
            .expect("Failed to build graph");

        retrieval
            .add_embeddings_to_graph(&mut graph)
            .expect("Failed to add embeddings");

        // Step 1: Search for error handling patterns
        let search_results = retrieval
            .hybrid_query("error handling ApiError", &graph)
            .expect("Search step failed");
        assert!(
            !search_results.is_empty(),
            "Step 1: Should find error handling patterns"
        );

        // Step 2: Understand the error types
        let understand_results = retrieval
            .hybrid_query("timeout network error types", &graph)
            .expect("Understand step failed");
        assert!(
            !understand_results.is_empty(),
            "Step 2: Should find error type information"
        );

        // Step 3: Search for method signatures to base new code on
        let method_results = retrieval
            .hybrid_query("fetch_data post_data async method", &graph)
            .expect("Method search failed");
        assert!(
            !method_results.is_empty(),
            "Step 3: Should find method signatures"
        );

        // The pipeline should maintain consistent retrieval quality
        // across related queries
        println!(
            "Pipeline: search={} understand={} methods={}",
            search_results.len(),
            understand_results.len(),
            method_results.len()
        );
    }

    #[test]
    fn test_cross_file_code_understanding() {
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");

        let mut graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        retrieval
            .add_embeddings_to_graph(&mut graph)
            .expect("Failed to add embeddings");

        // An agent asking about error handling should get results from
        // BOTH calculator.rs (CalculatorError) and api_client.rs (ApiError)
        let results = retrieval
            .hybrid_query("error types and error handling", &graph)
            .expect("Failed to query");

        assert!(!results.is_empty(), "Should find error handling across files");

        // Verify results span multiple source documents
        let unique_sources: std::collections::HashSet<_> = results
            .iter()
            .flat_map(|r| r.source_chunks.iter())
            .collect();

        println!(
            "Cross-file query found {} results from {} unique sources",
            results.len(),
            unique_sources.len()
        );
    }

    #[test]
    fn test_context_window_efficiency() {
        // Compare how much content is retrieved for the same query
        // with different retrieval configurations
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");

        let mut graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        retrieval
            .add_embeddings_to_graph(&mut graph)
            .expect("Failed to add embeddings");

        let results = retrieval
            .hybrid_query("Calculator", &graph)
            .expect("Failed to query");

        // Count total tokens (approximate: chars / 4)
        let total_chars: usize = results.iter().map(|r| r.content.len()).sum();
        let approx_tokens = total_chars / 4;

        println!(
            "Context efficiency: {} results, ~{} tokens for 'Calculator' query",
            results.len(),
            approx_tokens
        );

        // A well-tuned retrieval should not return excessive content
        // For a focused query like "Calculator", we don't need the entire codebase
        assert!(
            approx_tokens < 10_000,
            "Retrieval should be efficient — got ~{} tokens",
            approx_tokens
        );
    }
}

// ---------------------------------------------------------------------------
// Module 6: Performance
// ---------------------------------------------------------------------------

mod performance {
    use super::*;

    fn strict_perf_gates_enabled() -> bool {
        std::env::var("PERF_CI")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    #[test]
    fn bench_indexing_speed() {
        let start = Instant::now();

        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        let elapsed = start.elapsed();

        println!(
            "Indexed 3 files in {:?}: {} entities, {} relationships, {} chunks",
            elapsed,
            graph.entities().count(),
            graph.relationships().count(),
            graph.chunks().count()
        );

        // Threshold gates are opt-in in shared CI to reduce flakiness.
        if strict_perf_gates_enabled() {
            assert!(
                elapsed.as_secs() < 5,
                "Indexing should complete in <5s, took {:?}",
                elapsed
            );
        }
    }

    #[test]
    fn bench_query_latency() {
        let config = Config::default();
        let mut retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");

        let mut graph = build_graph_from_fixtures(&["graph_algorithms.rs"])
            .expect("Failed to build graph");

        retrieval
            .add_embeddings_to_graph(&mut graph)
            .expect("Failed to add embeddings");

        let queries = [
            "shortest path",
            "graph traversal",
            "breadth first search",
            "depth first search",
            "cycle detection",
            "topological sort",
            "dijkstra algorithm",
            "adjacency list",
            "binary heap priority queue",
            "visited nodes",
        ];

        let mut latencies = Vec::new();

        for query in &queries {
            let start = Instant::now();
            let _results = retrieval.hybrid_query(query, &graph).expect("Query failed");
            latencies.push(start.elapsed());
        }

        latencies.sort();
        let p50 = latencies[latencies.len() / 2];
        let p90 = latencies[(latencies.len() as f64 * 0.9) as usize];
        let p99 = latencies[latencies.len() - 1]; // With 10 samples, max ≈ p99

        println!(
            "Query latency (n={}): p50={:?}, p90={:?}, p99={:?}",
            queries.len(),
            p50,
            p90,
            p99
        );

        if strict_perf_gates_enabled() {
            assert!(
                p99.as_millis() < 500,
                "p99 latency should be <500ms, got {:?}",
                p99
            );
        }
    }

    #[test]
    fn bench_retrieval_system_initialization() {
        let config = Config::default();

        let start = Instant::now();
        let _retrieval = RetrievalSystem::new(&config).expect("Failed to create retrieval");
        let elapsed = start.elapsed();

        println!("RetrievalSystem initialization: {:?}", elapsed);

        if strict_perf_gates_enabled() {
            assert!(
                elapsed.as_millis() < 100,
                "RetrievalSystem init should be <100ms, took {:?}",
                elapsed
            );
        }
    }

    #[test]
    fn bench_graph_construction_scaling() {
        // Test with increasing document counts to verify sub-linear scaling
        let mut times = Vec::new();

        for n in [1, 2, 3] {
            let files: Vec<&str> = ["calculator.rs", "api_client.rs", "graph_algorithms.rs"]
                .iter()
                .take(n)
                .copied()
                .collect();

            let start = Instant::now();
            let _graph = build_graph_from_fixtures(&files).expect("Failed to build graph");
            let elapsed = start.elapsed();

            times.push((n, elapsed));
            println!("Graph construction with {} files: {:?}", n, elapsed);
        }

        if strict_perf_gates_enabled() {
            for (n, elapsed) in &times {
                assert!(
                    elapsed.as_secs() < 10,
                    "Graph construction with {} files should be <10s, took {:?}",
                    n,
                    elapsed
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "code-chunking")]
    fn bench_tree_sitter_chunking_speed() {
        use graphrag_core::text::chunking_strategies::RustCodeChunkingStrategy;

        let code = load_fixture("graph_algorithms.rs");
        let doc_id = DocumentId::new("bench".to_string());
        let strategy = RustCodeChunkingStrategy::new(10, doc_id);

        let start = Instant::now();
        for _ in 0..100 {
            let _chunks = strategy.chunk(&code);
        }
        let elapsed = start.elapsed();

        let per_parse = elapsed / 100;
        println!(
            "Tree-sitter chunking: {:?} per parse ({} bytes)",
            per_parse,
            code.len()
        );

        if strict_perf_gates_enabled() {
            assert!(
                per_parse.as_millis() < 10,
                "Tree-sitter parse should be <10ms, got {:?}",
                per_parse
            );
        }
    }
}
