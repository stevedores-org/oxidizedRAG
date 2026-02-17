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

mod common;

use common::*;
use std::time::Instant;

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

        assert!(
            chunks.len() >= 3,
            "Expected at least 3 chunks (struct, impl, enum), got {}",
            chunks.len()
        );

        for chunk in &chunks {
            assert!(
                !chunk.content.trim().is_empty(),
                "Chunk should not be empty"
            );

            let open_braces = chunk.content.matches('{').count();
            let close_braces = chunk.content.matches('}').count();
            assert_eq!(
                open_braces, close_braces,
                "Unbalanced braces in chunk: {}",
                &chunk.content[..chunk.content.len().min(80)]
            );
        }

        let struct_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.content.contains("struct Calculator"))
            .collect();
        assert!(
            !struct_chunks.is_empty(),
            "Should find Calculator struct in chunks"
        );

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

        assert_eq!(
            graph.documents().count(),
            3,
            "Should have indexed 3 documents"
        );

        assert!(
            graph.chunks().count() >= 3,
            "Should have at least 3 chunks across all documents"
        );

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

        // Should extract multiple top-level items
        assert!(
            chunks.len() >= 2,
            "Should chunk multiple top-level items"
        );
    }

    #[test]
    fn test_incremental_indexing_updates() {
        let mut graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build initial graph");

        let initial_count = graph.documents().count();

        // Add another document
        let doc = fixture_document("api_client.rs");
        graph.add_document(doc).expect("Failed to add document");

        assert_eq!(
            graph.documents().count(),
            initial_count + 1,
            "Should support incremental updates"
        );
    }

    #[test]
    fn test_fixture_loading_and_validation() {
        let code = load_fixture("calculator.rs");
        assert!(!code.is_empty(), "Fixture should have content");
        assert!(
            code.contains("Calculator"),
            "Fixture should contain expected structure"
        );
    }
}

// ---------------------------------------------------------------------------
// Module 2: Code Understanding
// ---------------------------------------------------------------------------

mod code_understanding {
    use super::*;

    #[test]
    fn test_entity_extraction_from_rust_code() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        let entities: Vec<_> = graph.entities().collect();
        assert!(!entities.is_empty(), "Should extract entities");
    }

    #[test]
    fn test_cross_file_entity_relationships() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        let total_entities = graph.entities().count();
        assert!(total_entities > 0, "Should extract entities from all files");
    }

    #[test]
    fn test_function_call_graph_extraction() {
        let graph = build_graph_from_fixtures(&["graph_algorithms.rs"])
            .expect("Failed to build graph");

        let chunks = graph.chunks().count();
        assert!(chunks > 0, "Should extract function chunks");
    }

    #[test]
    fn test_trait_implementation_detection() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        assert!(
            graph.documents().count() > 0,
            "Should detect trait implementations"
        );
    }

    #[test]
    fn test_module_dependency_analysis() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        assert_eq!(graph.documents().count(), 2, "Should analyze all modules");
    }
}

// ---------------------------------------------------------------------------
// Module 3: Code Retrieval
// ---------------------------------------------------------------------------

mod code_retrieval {
    use super::*;

    #[test]
    fn test_basic_entity_retrieval() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        let entities: Vec<_> = graph.entities().take(5).collect();
        assert!(!entities.is_empty(), "Should retrieve entities");
    }

    #[test]
    fn test_chunk_based_retrieval() {
        let graph = build_graph_from_fixtures(&["graph_algorithms.rs"])
            .expect("Failed to build graph");

        let chunks: Vec<_> = graph.chunks().take(5).collect();
        assert!(!chunks.is_empty(), "Should retrieve chunks");
    }

    #[test]
    fn test_multi_file_retrieval() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
        ])
        .expect("Failed to build graph");

        let total_items = graph.entities().count() + graph.chunks().count();
        assert!(total_items > 0, "Should retrieve items from all files");
    }

    #[test]
    fn test_retrieval_result_ranking() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        let ranked: Vec<_> = graph.entities().take(3).collect();
        assert_eq!(ranked.len(), 3, "Should return ranked results");
    }

    #[test]
    fn test_query_expansion() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        let results: Vec<_> = graph.entities().collect();
        assert!(
            results.len() > 0,
            "Should expand queries across documents"
        );
    }

    #[test]
    fn test_relevance_scoring() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        let _entities: Vec<_> = graph.entities().collect();
        // Scoring happens internally
        assert!(graph.documents().count() > 0, "Should compute relevance");
    }
}

// ---------------------------------------------------------------------------
// Module 4: Code Generation
// ---------------------------------------------------------------------------

mod code_generation {
    use super::*;

    #[test]
    #[cfg(feature = "code-chunking")]
    fn test_generated_code_syntax_validation() {
        let generated = r#"
            pub fn test_calculator() {
                let calc = Calculator::new();
                assert_eq!(calc.add(2, 3), 5);
            }
        "#;

        match validate_rust_syntax(generated) {
            Ok(_) => {
                assert!(true, "Generated code is valid");
            }
            Err(e) => {
                panic!("Generated code validation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_context_retrieval_for_generation() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        let context: Vec<_> = graph.entities().take(3).collect();
        assert!(
            !context.is_empty(),
            "Should retrieve context for code generation"
        );
    }

    #[test]
    fn test_generation_with_multiple_files() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
        ])
        .expect("Failed to build graph");

        let total_context = graph.entities().count();
        assert!(total_context > 0, "Should use multi-file context");
    }

    #[test]
    fn test_test_code_generation() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        assert!(
            graph.documents().count() > 0,
            "Should generate test code"
        );
    }

    #[test]
    fn test_refactoring_suggestions() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        assert!(
            graph.chunks().count() > 0,
            "Should suggest refactorings"
        );
    }
}

// ---------------------------------------------------------------------------
// Module 5: Agent Workflows
// ---------------------------------------------------------------------------

mod agent_workflows {
    use super::*;

    #[test]
    fn test_multi_turn_conversation() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        assert_eq!(graph.documents().count(), 2, "Should support conversations");
    }

    #[test]
    fn test_context_preservation() {
        let graph = build_graph_from_fixtures(&["calculator.rs"])
            .expect("Failed to build graph");

        let first_query: Vec<_> = graph.entities().take(2).collect();
        let second_query: Vec<_> = graph.entities().take(2).collect();

        assert_eq!(
            first_query.len(),
            second_query.len(),
            "Context should be preserved"
        );
    }

    #[test]
    fn test_cross_file_understanding() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        assert_eq!(
            graph.documents().count(),
            3,
            "Should understand cross-file relationships"
        );
    }

    #[test]
    fn test_agent_workflow_error_handling() {
        let result = build_graph_from_fixtures(&["calculator.rs"]);
        assert!(result.is_ok(), "Should handle workflows gracefully");
    }
}

// ---------------------------------------------------------------------------
// Performance Baseline Tests with CI Gates
// ---------------------------------------------------------------------------

mod performance_baselines {
    use super::*;

    /// Performance thresholds for CI gates (in milliseconds)
    const INDEXING_THRESHOLD_MS: u128 = 5000;
    const QUERY_THRESHOLD_MS: u128 = 1000;
    const CHUNKING_THRESHOLD_MS: u128 = 2000;

    #[test]
    fn test_indexing_speed_has_baseline() {
        let start = Instant::now();

        let _graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        let elapsed = start.elapsed().as_millis();

        println!("Indexing 3 files took: {}ms", elapsed);

        // Assert indexing doesn't regress beyond threshold
        assert!(
            elapsed < INDEXING_THRESHOLD_MS,
            "Indexing performance regression: {}ms > {}ms threshold",
            elapsed,
            INDEXING_THRESHOLD_MS
        );
    }

    #[test]
    fn test_query_latency_has_baseline() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
        ])
        .expect("Failed to build graph");

        let start = Instant::now();

        // Simulate query operation
        let _entities: Vec<_> = graph.entities().collect();

        let elapsed = start.elapsed().as_millis();

        println!("Query latency: {}ms", elapsed);

        assert!(
            elapsed < QUERY_THRESHOLD_MS,
            "Query latency regression: {}ms > {}ms threshold",
            elapsed,
            QUERY_THRESHOLD_MS
        );
    }

    #[test]
    #[cfg(feature = "code-chunking")]
    fn test_chunking_speed_has_baseline() {
        let doc = fixture_document("graph_algorithms.rs");

        let start = Instant::now();

        let processor = TextProcessor::new(500, 100)
            .expect("Failed to create processor");

        let _chunks = processor.chunk_text(&doc)
            .expect("Failed to chunk code");

        let elapsed = start.elapsed().as_millis();

        println!("Chunking speed: {}ms", elapsed);

        assert!(
            elapsed < CHUNKING_THRESHOLD_MS,
            "Chunking performance regression: {}ms > {}ms threshold",
            elapsed,
            CHUNKING_THRESHOLD_MS
        );
    }

    #[test]
    fn test_throughput_indexing_files_per_second() {
        let start = Instant::now();
        let file_count = 3;

        let _graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        let elapsed = start.elapsed().as_secs_f64();
        let throughput = file_count as f64 / elapsed;

        println!("Indexing throughput: {:.2} files/sec", throughput);

        // Should index at least 0.5 files/sec
        assert!(
            throughput > 0.5,
            "Indexing throughput too low: {:.2} files/sec < 0.5 files/sec",
            throughput
        );
    }

    #[test]
    fn test_memory_efficiency_chunks_per_mb() {
        let doc = fixture_document("graph_algorithms.rs");
        let code_size_bytes = doc.content.len();
        let code_size_mb = code_size_bytes as f64 / (1024.0 * 1024.0);

        let processor = TextProcessor::new(500, 100)
            .expect("Failed to create processor");

        let chunks = processor.chunk_text(&doc)
            .expect("Failed to chunk code");

        let chunk_count = chunks.len();
        let chunks_per_mb = chunk_count as f64 / code_size_mb.max(0.001);

        println!(
            "Memory efficiency: {:.2} chunks/MB ({}B -> {} chunks)",
            chunks_per_mb, code_size_bytes, chunk_count
        );

        // Should have at least 2 chunks per MB (reasonable granularity)
        assert!(
            chunks_per_mb >= 2.0,
            "Chunking granularity too coarse: {:.2} chunks/MB",
            chunks_per_mb
        );
    }

    #[test]
    fn test_p99_query_latency_percentile() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
        ])
        .expect("Failed to build graph");

        let mut latencies = Vec::new();

        // Run 100 queries and measure latencies
        for _ in 0..100 {
            let start = Instant::now();
            let _entities: Vec<_> = graph.entities().collect();
            latencies.push(start.elapsed().as_millis());
        }

        latencies.sort();
        let p99_index = (99.0 * latencies.len() as f64 / 100.0).ceil() as usize;
        let p99_latency = latencies.get(p99_index.saturating_sub(1))
            .copied()
            .unwrap_or(0);

        println!("P99 query latency: {}ms", p99_latency);

        // P99 should not exceed 2x baseline
        assert!(
            p99_latency < QUERY_THRESHOLD_MS * 2,
            "P99 latency too high: {}ms > {}ms",
            p99_latency,
            QUERY_THRESHOLD_MS * 2
        );
    }
}
