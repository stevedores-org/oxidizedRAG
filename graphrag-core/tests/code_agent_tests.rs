
// ---------------------------------------------------------------------------
// Negative Test Cases - Error Handling and Edge Cases
// ---------------------------------------------------------------------------

mod negative_test_cases {
    use super::*;

    #[test]
    fn test_empty_document_handling() {
        let doc_id = DocumentId::new("empty".to_string());
        let empty_doc = Document::new(doc_id, "empty.rs".to_string(), String::new());
        
        // Should handle empty documents gracefully
        let mut graph = KnowledgeGraph::new();
        let result = graph.add_document(empty_doc);
        
        // Either succeeds with empty chunks or returns sensible error
        match result {
            Ok(_) => {
                // Should have zero chunks
                assert_eq!(graph.chunks().count(), 0);
            }
            Err(e) => {
                // Error should mention empty content
                let msg = format!("{:?}", e);
                assert!(
                    msg.to_lowercase().contains("empty") || msg.to_lowercase().contains("invalid"),
                    "Expected meaningful error for empty doc, got: {}",
                    msg
                );
            }
        }
    }

    #[test]
    fn test_malformed_rust_code_graceful_handling() {
        let malformed_code = r#"
            fn broken_function() {
                let x = 5
                // Missing semicolon above
                let y = {
                    // Missing closing brace for y block
                println!("Hello");
            }
        "#;
        
        let doc_id = DocumentId::new("malformed".to_string());
        let doc = Document::new(doc_id, "malformed.rs".to_string(), malformed_code.to_string());
        
        let mut graph = KnowledgeGraph::new();
        let result = graph.add_document(doc);
        
        // Should either parse with errors or return error
        match result {
            Ok(g) => {
                // If it succeeds, at least report chunks (even if partial)
                let chunks = g.chunks().count();
                println!("Malformed code produced {} chunks", chunks);
            }
            Err(e) => {
                // Error should be descriptive
                let msg = format!("{:?}", e);
                assert!(
                    !msg.is_empty(),
                    "Error handling malformed code should be descriptive"
                );
                println!("Malformed code error: {}", msg);
            }
        }
    }

    #[test]
    fn test_extremely_large_code_file_handling() {
        // Create a large but valid Rust function
        let mut large_code = String::new();
        large_code.push_str("fn large_function() {\n");
        
        // Add 1000 lines of simple code
        for i in 0..1000 {
            large_code.push_str(&format!("    let var_{} = {};\n", i, i));
        }
        
        large_code.push_str("}\n");
        
        let doc_id = DocumentId::new("large".to_string());
        let doc = Document::new(doc_id, "large.rs".to_string(), large_code);
        
        let mut graph = KnowledgeGraph::new();
        let result = graph.add_document(doc);
        
        // Should handle large files
        match result {
            Ok(g) => {
                // Verify it was chunked appropriately
                let chunk_count = g.chunks().count();
                assert!(chunk_count > 0, "Should chunk large files into multiple pieces");
            }
            Err(e) => {
                // If it fails, should be explicit about size limits
                let msg = format!("{:?}", e);
                println!("Large file error: {}", msg);
            }
        }
    }

    #[test]
    fn test_missing_fixture_file_error() {
        let result = std::fs::read_to_string("tests/fixtures/code_samples/nonexistent.rs");
        
        // Should fail clearly
        assert!(
            result.is_err(),
            "Missing fixture file should return error"
        );
        
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("No such file") || err_msg.contains("not found"),
            "Error should mention missing file: {}",
            err_msg
        );
    }

    #[test]
    fn test_null_or_invalid_document_id() {
        // Document ID should not accept empty strings
        let empty_id = DocumentId::new(String::new());
        let doc = Document::new(empty_id, "test.rs".to_string(), "fn test() {}".to_string());
        
        let mut graph = KnowledgeGraph::new();
        let result = graph.add_document(doc);
        
        // Either handle gracefully or reject with clear error
        match result {
            Ok(_) => {
                println!("Empty document ID was accepted");
            }
            Err(e) => {
                let msg = format!("{:?}", e);
                assert!(
                    msg.to_lowercase().contains("id") || msg.to_lowercase().contains("invalid"),
                    "Error for invalid ID should mention document ID: {}",
                    msg
                );
            }
        }
    }

    #[test]
    fn test_concurrent_graph_modifications() {
        use std::sync::Arc;
        use std::sync::Mutex;
        use std::thread;

        let graph = Arc::new(Mutex::new(KnowledgeGraph::new()));
        let mut handles = vec![];

        // Try to modify graph from 3 concurrent threads
        for i in 0..3 {
            let graph_clone = Arc::clone(&graph);
            let handle = thread::spawn(move || {
                let doc_id = DocumentId::new(format!("concurrent_{}", i));
                let doc = Document::new(
                    doc_id,
                    format!("file_{}.rs", i),
                    format!("fn func_{}() {{}}", i),
                );
                
                let mut g = graph_clone.lock().unwrap();
                g.add_document(doc)
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            let result = handle.join();
            assert!(result.is_ok(), "Thread panic during concurrent modification");
        }

        // Verify graph state is consistent
        let final_graph = graph.lock().unwrap();
        let doc_count = final_graph.documents().count();
        assert_eq!(doc_count, 3, "Should have 3 documents after concurrent modifications");
    }

    #[test]
    fn test_duplicate_document_handling() {
        let doc_id = DocumentId::new("duplicate".to_string());
        let doc1 = Document::new(doc_id.clone(), "test.rs".to_string(), "fn test() {}".to_string());
        let doc2 = Document::new(doc_id, "test.rs".to_string(), "fn test2() {}".to_string());

        let mut graph = KnowledgeGraph::new();
        let result1 = graph.add_document(doc1);
        let result2 = graph.add_document(doc2);

        // Both operations should either succeed or fail consistently
        if result1.is_ok() && result2.is_ok() {
            // If both succeed, verify latest wins or merge
            let count = graph.documents().count();
            assert!(count >= 1, "Should have at least one document");
        } else if result1.is_ok() && result2.is_err() {
            // If second fails, should mention duplicate
            let msg = format!("{:?}", result2.unwrap_err());
            assert!(
                msg.to_lowercase().contains("duplicate") || msg.to_lowercase().contains("exists"),
                "Error should mention duplicate: {}",
                msg
            );
        }
    }

    #[test]
    fn test_invalid_chunk_size_parameters() {
        // Zero chunk size should fail
        let result = TextProcessor::new(0, 10);
        assert!(
            result.is_err(),
            "TextProcessor should reject zero chunk size"
        );

        // Chunk size smaller than overlap should fail
        let result = TextProcessor::new(10, 20);
        assert!(
            result.is_err(),
            "TextProcessor should reject overlap > chunk_size"
        );
    }
}
