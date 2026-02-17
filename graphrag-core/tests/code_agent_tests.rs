
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
