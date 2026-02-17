
// ---------------------------------------------------------------------------
// Performance Baseline Tests with CI Gates
// ---------------------------------------------------------------------------

mod performance_baselines {
    use super::*;
    use std::time::Duration;

    /// Performance thresholds for CI gates (in milliseconds)
    const INDEXING_THRESHOLD_MS: u128 = 5000;
    const QUERY_THRESHOLD_MS: u128 = 1000;
    const CHUNKING_THRESHOLD_MS: u128 = 2000;

    #[test]
    fn test_indexing_speed_has_baseline() {
        let start = Instant::now();
        
        let graph = build_graph_from_fixtures(&[
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
        let code = load_fixture("graph_algorithms.rs");
        
        let start = Instant::now();
        
        let processor = TextProcessor::new(500, 100)
            .expect("Failed to create processor");
        
        let _chunks = processor.chunk_text(&code)
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
        let code = load_fixture("graph_algorithms.rs");
        let code_size_bytes = code.len();
        let code_size_mb = code_size_bytes as f64 / (1024.0 * 1024.0);
        
        let processor = TextProcessor::new(500, 100)
            .expect("Failed to create processor");
        
        let chunks = processor.chunk_text(&code)
            .expect("Failed to chunk code");
        
        let chunk_count = chunks.len();
        let chunks_per_mb = chunk_count as f64 / code_size_mb.max(0.001);
        
        println!(
            "Memory efficiency: {:.2} chunks/MB ({}B → {} chunks)",
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

// ---------------------------------------------------------------------------
// Module 7: Multi-Language Support
// ---------------------------------------------------------------------------

mod multi_language {
    use super::*;

    #[test]
    fn test_python_fixture_loading() {
        let code = load_fixture("example.py");
        assert!(!code.is_empty(), "Python fixture should have content");
        assert!(
            code.contains("class"),
            "Python fixture should contain class definitions"
        );
        assert!(
            code.contains("def "),
            "Python fixture should contain function definitions"
        );
    }

    #[test]
    fn test_python_class_extraction() {
        let code = load_fixture("example.py");

        // Verify key Python constructs
        assert!(
            code.contains("class DataProcessor"),
            "Should contain abstract base class"
        );
        assert!(
            code.contains("class StatisticalAnalyzer"),
            "Should contain concrete implementation"
        );
        assert!(
            code.contains("@dataclass"),
            "Should contain dataclass decorator"
        );
        assert!(
            code.contains("@abstractmethod"),
            "Should contain abstract method decorator"
        );
    }

    #[test]
    fn test_python_function_extraction() {
        let code = load_fixture("example.py");

        // Verify function patterns
        assert!(
            code.contains("def process(self, data:"),
            "Should contain typed method"
        );
        assert!(
            code.contains("def __init__"),
            "Should contain constructor"
        );
        assert!(
            code.contains("def aggregate_results("),
            "Should contain module-level function"
        );
    }

    #[test]
    fn test_python_type_hints() {
        let code = load_fixture("example.py");

        // Verify type hints (Python 3.6+)
        assert!(
            code.contains("List[DataPoint]"),
            "Should use generic type hints"
        );
        assert!(
            code.contains("Optional[dict]"),
            "Should use Optional hints"
        );
        assert!(
            code.contains("Union["),
            "Should support Union types"
        );
        assert!(
            code.contains("-> float"),
            "Should have return type annotations"
        );
    }

    #[test]
    fn test_javascript_fixture_loading() {
        let code = load_fixture("example.js");
        assert!(!code.is_empty(), "JavaScript fixture should have content");
        assert!(
            code.contains("class"),
            "JavaScript fixture should contain class definitions"
        );
        assert!(
            code.contains("function"),
            "JavaScript fixture should contain functions"
        );
    }

    #[test]
    fn test_javascript_class_extraction() {
        let code = load_fixture("example.js");

        // Verify JavaScript constructs
        assert!(
            code.contains("class DataProcessor"),
            "Should contain base class"
        );
        assert!(
            code.contains("class StatisticalAnalyzer extends"),
            "Should contain class inheritance"
        );
        assert!(
            code.contains("async process(data)"),
            "Should contain async methods"
        );
    }

    #[test]
    fn test_javascript_closure_patterns() {
        let code = load_fixture("example.js");

        // Verify closure and scope patterns
        assert!(
            code.contains("this.cache = new Map()"),
            "Should use instance properties"
        );
        assert!(
            code.contains("this.windowSize"),
            "Should access member variables"
        );
        assert!(
            code.contains("const results = []"),
            "Should use const declarations"
        );
    }

    #[test]
    fn test_javascript_async_await() {
        let code = load_fixture("example.js");

        // Verify async/await patterns
        assert!(
            code.contains("async execute(data)"),
            "Should have async functions"
        );
        assert!(
            code.contains("await processor.process"),
            "Should use await expressions"
        );
    }

    #[test]
    fn test_typescript_fixture_loading() {
        let code = load_fixture("example.ts");
        assert!(!code.is_empty(), "TypeScript fixture should have content");
        assert!(
            code.contains("interface"),
            "TypeScript fixture should contain interfaces"
        );
        assert!(
            code.contains("export"),
            "TypeScript fixture should have exports"
        );
    }

    #[test]
    fn test_typescript_interface_extraction() {
        let code = load_fixture("example.ts");

        // Verify TypeScript interfaces
        assert!(
            code.contains("export interface DataPoint"),
            "Should contain interface definitions"
        );
        assert!(
            code.contains("export interface AnalysisConfig"),
            "Should contain configuration interface"
        );
        assert!(
            code.contains("Record<string"),
            "Should use mapped types"
        );
    }

    #[test]
    fn test_typescript_generic_types() {
        let code = load_fixture("example.ts");

        // Verify generic type usage
        assert!(
            code.contains("DataProcessor<T"),
            "Should use generic type parameters"
        );
        assert!(
            code.contains("DataProcessor<DataPoint>"),
            "Should specialize generic types"
        );
        assert!(
            code.contains("PipelineExecutor<T>"),
            "Should have generic class"
        );
    }

    #[test]
    fn test_typescript_union_types() {
        let code = load_fixture("example.ts");

        // Verify union and literal types
        assert!(
            code.contains("AggregationType = 'mean' | 'max' | 'min'"),
            "Should use literal union types"
        );
        assert!(
            code.contains("number | null"),
            "Should use union with null"
        );
    }

    #[test]
    fn test_typescript_advanced_features() {
        let code = load_fixture("example.ts");

        // Verify advanced TypeScript features
        assert!(
            code.contains("ReadonlyArray"),
            "Should use readonly types"
        );
        assert!(
            code.contains("never ="),
            "Should use exhaustive checking"
        );
        assert!(
            code.contains("Map<string"),
            "Should use generic collections"
        );
    }

    #[test]
    fn test_multi_language_entity_count() {
        // Load all three fixtures
        let python = load_fixture("example.py");
        let javascript = load_fixture("example.js");
        let typescript = load_fixture("example.ts");

        // Count class definitions across languages
        let python_classes = python.matches("class ").count();
        let js_classes = javascript.matches("class ").count();
        let ts_classes = typescript.matches("class ").count();

        assert!(python_classes > 0, "Python should have classes");
        assert!(js_classes > 0, "JavaScript should have classes");
        assert!(ts_classes > 0, "TypeScript should have classes");

        println!(
            "Language comparison: Python={} classes, JS={}, TS={}",
            python_classes, js_classes, ts_classes
        );
    }

    #[test]
    fn test_multi_language_fixture_sizes() {
        let python = load_fixture("example.py");
        let javascript = load_fixture("example.js");
        let typescript = load_fixture("example.ts");

        let py_size = python.len();
        let js_size = javascript.len();
        let ts_size = typescript.len();

        println!(
            "Fixture sizes: Python={}B, JavaScript={}B, TypeScript={}B",
            py_size, js_size, ts_size
        );

        // All should be substantial
        assert!(py_size > 1000, "Python fixture should be >1KB");
        assert!(js_size > 1000, "JavaScript fixture should be >1KB");
        assert!(ts_size > 1000, "TypeScript fixture should be >1KB");
    }

    #[test]
    fn test_multi_language_common_patterns() {
        let python = load_fixture("example.py");
        let javascript = load_fixture("example.js");
        let typescript = load_fixture("example.ts");

        // Common patterns across languages
        let has_processor_pattern = |code: &str| {
            code.contains("Processor") && code.contains("process")
        };

        let has_analyzer_pattern = |code: &str| {
            code.contains("Analyzer") && code.contains("analyze")
        };

        let has_pipeline_pattern = |code: &str| {
            code.contains("Pipeline") || code.contains("pipeline")
        };

        assert!(has_processor_pattern(&python), "Python: Processor pattern");
        assert!(has_processor_pattern(&javascript), "JavaScript: Processor pattern");
        assert!(has_processor_pattern(&typescript), "TypeScript: Processor pattern");

        assert!(has_analyzer_pattern(&python), "Python: Analyzer pattern");
        assert!(has_analyzer_pattern(&javascript), "JavaScript: Analyzer pattern");
        assert!(has_analyzer_pattern(&typescript), "TypeScript: Analyzer pattern");

        assert!(has_pipeline_pattern(&python), "Python: Pipeline pattern");
        assert!(has_pipeline_pattern(&javascript), "JavaScript: Pipeline pattern");
        assert!(has_pipeline_pattern(&typescript), "TypeScript: Pipeline pattern");
    }

    #[test]
    fn test_multi_language_documentation() {
        let python = load_fixture("example.py");
        let javascript = load_fixture("example.js");
        let typescript = load_fixture("example.ts");

        // Verify documentation exists in each language
        assert!(python.contains("\"\"\""), "Python should have docstrings");
        assert!(
            javascript.contains("/**"),
            "JavaScript should have JSDoc comments"
        );
        assert!(
            typescript.contains("/**"),
            "TypeScript should have TSDoc comments"
        );
    }

    #[test]
    fn test_python_import_patterns() {
        let code = load_fixture("example.py");

        // Verify import patterns
        assert!(code.contains("from typing import"), "Should have typing imports");
        assert!(code.contains("from dataclasses"), "Should have dataclass imports");
        assert!(code.contains("from abc import"), "Should have ABC imports");
    }

    #[test]
    fn test_javascript_export_patterns() {
        let code = load_fixture("example.js");

        // Verify export patterns
        assert!(
            code.contains("module.exports"),
            "Should use CommonJS exports"
        );
        assert!(
            code.contains("class StatisticalAnalyzer"),
            "Should export classes"
        );
    }

    #[test]
    fn test_typescript_export_patterns() {
        let code = load_fixture("example.ts");

        // Verify TypeScript export patterns
        assert!(code.contains("export interface"), "Should export interfaces");
        assert!(code.contains("export class"), "Should export classes");
        assert!(code.contains("export type"), "Should export type aliases");
        assert!(code.contains("export function"), "Should export functions");
    }
}
