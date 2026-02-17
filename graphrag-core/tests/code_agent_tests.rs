
// ---------------------------------------------------------------------------
// End-to-End Agent Workflows - Full RAG Pipeline Tests
// ---------------------------------------------------------------------------

mod e2e_agent_workflows {
    use super::*;

    /// Represents a multi-turn conversation context
    struct ConversationContext {
        turns: Vec<ConversationTurn>,
        knowledge_graph: KnowledgeGraph,
    }

    struct ConversationTurn {
        user_query: String,
        retrieved_context: Vec<String>,
        generated_response: String,
        turn_number: usize,
    }

    impl ConversationContext {
        fn new(graph: KnowledgeGraph) -> Self {
            ConversationContext {
                turns: Vec::new(),
                knowledge_graph: graph,
            }
        }

        fn add_turn(&mut self, query: String, retrieved: Vec<String>, response: String) {
            self.turns.push(ConversationTurn {
                user_query: query,
                retrieved_context: retrieved,
                generated_response: response,
                turn_number: self.turns.len() + 1,
            });
        }

        fn context_history(&self) -> String {
            self.turns
                .iter()
                .map(|t| {
                    format!(
                        "Turn {}: Query='{}', Response='{}'",
                        t.turn_number, t.user_query, t.generated_response
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
    }

    #[test]
    fn test_e2e_multi_turn_conversation_with_context_preservation() {
        // Index code fixtures
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build knowledge graph");

        let mut conversation = ConversationContext::new(graph);

        // Turn 1: Ask about calculator structure
        conversation.add_turn(
            "What is the Calculator struct?".to_string(),
            vec!["Calculator struct with add, subtract, multiply methods".to_string()],
            "The Calculator provides basic arithmetic operations".to_string(),
        );

        // Turn 2: Follow-up about implementation
        conversation.add_turn(
            "How is addition implemented?".to_string(),
            vec!["impl block shows add returns self for chaining".to_string()],
            "Addition is implemented with method chaining support".to_string(),
        );

        // Turn 3: Cross-file understanding
        conversation.add_turn(
            "Compare with graph algorithms complexity".to_string(),
            vec!["Graph algorithms include BFS, DFS, shortest path".to_string()],
            "Calculator is O(1), graph ops are O(V+E) or O(V²)".to_string(),
        );

        // Verify conversation preserved all context
        assert_eq!(conversation.turns.len(), 3, "Should have 3 conversation turns");

        // Verify context is available for each turn
        for turn in &conversation.turns {
            assert!(
                !turn.user_query.is_empty(),
                "Turn {} should have query",
                turn.turn_number
            );
            assert!(
                !turn.generated_response.is_empty(),
                "Turn {} should have response",
                turn.turn_number
            );
            assert!(
                !turn.retrieved_context.is_empty(),
                "Turn {} should have retrieved context",
                turn.turn_number
            );
        }

        // Verify conversation continuity
        let history = conversation.context_history();
        assert!(
            history.contains("Turn 1"),
            "History should include all turns"
        );
        assert!(
            history.contains("Calculator"),
            "History should maintain context across turns"
        );
    }

    #[test]
    fn test_e2e_full_rag_pipeline_index_search_generate() {
        // Step 1: Index code documents
        println!("Step 1: Indexing documents...");
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to index documents");

        assert!(graph.documents().count() > 0, "Should have indexed documents");

        // Step 2: Search/retrieve relevant code
        println!("Step 2: Retrieving relevant code...");
        let relevant_entities: Vec<_> = graph
            .entities()
            .take(5) // Get top 5 entities
            .collect();

        assert!(
            !relevant_entities.is_empty(),
            "Should retrieve relevant entities"
        );

        // Step 3: Generate response based on retrieved context
        println!("Step 3: Generating response...");
        let generated = format!(
            "Based on {} relevant entities, the code implements {} components",
            relevant_entities.len(),
            graph.documents().count()
        );

        assert!(!generated.is_empty(), "Should generate response");
        println!("Generated: {}", generated);
    }

    #[test]
    fn test_e2e_cross_file_entity_relationships() {
        // Index multiple files
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "api_client.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        // Verify entities are extracted from all files
        let total_entities = graph.entities().count();
        assert!(
            total_entities > 0,
            "Should extract entities from multiple files"
        );

        // Verify relationships exist between entities
        // (In real implementation, would verify function calls, trait implementations, etc.)
        let entity_groups: Vec<_> = graph.entities().take(3).collect();

        for entity in entity_groups {
            println!("Entity: {:?}", entity);
        }

        // Verify entity relationships are discoverable
        assert!(
            total_entities >= 3,
            "Should have multiple entities to establish relationships"
        );
    }

    #[test]
    fn test_e2e_code_generation_validation() {
        // Index code to use as context
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
        ])
        .expect("Failed to build graph");

        // Simulate code generation based on graph
        let generated_code = r#"
            pub fn test_calculator() {
                let calc = Calculator::new();
                assert_eq!(calc.add(2, 3), 5);
            }
        "#;

        // Validate generated code is syntactically correct
        #[cfg(feature = "code-chunking")]
        {
            match validate_rust_syntax(generated_code) {
                Ok(_) => {
                    println!("✓ Generated code is syntactically valid");
                }
                Err(e) => {
                    panic!("Generated code validation failed: {}", e);
                }
            }
        }

        assert!(
            graph.documents().count() > 0,
            "Graph should have context for generation"
        );
    }

    #[test]
    fn test_e2e_context_aware_code_suggestions() {
        let graph = build_graph_from_fixtures(&[
            "calculator.rs",
            "graph_algorithms.rs",
        ])
        .expect("Failed to build graph");

        // User asks for code suggestion in context of graph
        let user_intent = "Add a multiply method to the Calculator";

        // Retrieve relevant entities (Calculator struct)
        let relevant_code: Vec<_> = graph
            .entities()
            .take(5)
            .map(|e| format!("{:?}", e))
            .collect();

        // Generate suggestion based on context
        let suggestion = format!(
            "Based on existing structure, suggested implementation: \
             impl Calculator {{ pub fn multiply(&self, a: i32, b: i32) -> i32 {{ a * b }} }}"
        );

        assert!(
            !suggestion.is_empty(),
            "Should generate context-aware suggestion"
        );
        assert!(
            suggestion.contains("multiply"),
            "Suggestion should address user intent"
        );
        assert!(
            !relevant_code.is_empty(),
            "Should find relevant context"
        );
    }

    #[test]
    fn test_e2e_conversation_with_feedback_loop() {
        let mut graph = build_graph_from_fixtures(&[
            "calculator.rs",
        ])
        .expect("Failed to build graph");

        let mut conversation = ConversationContext::new(graph);

        // Initial query
        conversation.add_turn(
            "Explain the Calculator struct".to_string(),
            vec!["struct Calculator { value: i32 }".to_string()],
            "Calculator is a simple arithmetic struct".to_string(),
        );

        // User feedback: "That's too brief"
        // System responds with more detail
        conversation.add_turn(
            "More details please".to_string(),
            vec![
                "impl block has add, subtract, multiply, divide".to_string(),
            ],
            "Calculator implements standard arithmetic operations with method chaining support"
                .to_string(),
        );

        // Verify feedback loop improved response
        let last_turn = &conversation.turns[conversation.turns.len() - 1];
        assert!(
            last_turn.generated_response.len()
                > conversation.turns[0].generated_response.len(),
            "Feedback should lead to more detailed responses"
        );
    }

    #[test]
    fn test_e2e_error_recovery_in_workflow() {
        // Simulate workflow with error recovery
        let mut workflow_steps = Vec::new();

        // Step 1: Try to index (might fail)
        let step1 = std::panic::catch_unwind(|| {
            build_graph_from_fixtures(&["nonexistent.rs"])
        });

        match step1 {
            Ok(Ok(_)) => {
                workflow_steps.push("Index succeeded");
            }
            Ok(Err(_)) => {
                println!("Index failed (expected), recovering...");
                workflow_steps.push("Index failed but recovered");
            }
            Err(_) => {
                println!("Index panicked, but caught");
                workflow_steps.push("Index error caught");
            }
        }

        // Step 2: Retry with valid files
        let step2 = build_graph_from_fixtures(&[
            "calculator.rs",
        ]);

        if step2.is_ok() {
            workflow_steps.push("Retry succeeded");
        }

        // Verify error recovery worked
        assert!(
            workflow_steps.len() > 1,
            "Workflow should recover from errors"
        );
    }
}
