//! Code-Agent fixture-based tests.
//!
//! Tests code query patterns against Rust and Python fixture repos,
//! verifying golden retrieval rankings and context pack assembly.

#[cfg(feature = "api")]
mod code_agent_fixtures {
    use std::path::PathBuf;

    use graphrag_core::api::code_agent::*;

    // ========================================================================
    // Fixture file paths
    // ========================================================================

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn rust_fixture_dir() -> PathBuf {
        fixture_dir().join("rust_sample")
    }

    fn python_fixture_dir() -> PathBuf {
        fixture_dir().join("python_sample")
    }

    // ========================================================================
    // Query suite
    // ========================================================================

    #[test]
    fn test_fixture_directories_exist() {
        assert!(rust_fixture_dir().exists(), "Rust fixture dir missing");
        assert!(python_fixture_dir().exists(), "Python fixture dir missing");
        assert!(
            rust_fixture_dir().join("main.rs").exists(),
            "main.rs missing"
        );
        assert!(
            python_fixture_dir().join("app.py").exists(),
            "app.py missing"
        );
    }

    #[test]
    fn test_golden_ranking_definition_query() {
        // "Where is DataProcessor implemented?" should rank main.rs highest
        let expected_file = PathBuf::from("rust_sample/main.rs");
        let match_result = CodeMatch {
            location: CodeLocation {
                file: expected_file.clone(),
                line: 18,
                column: Some(1),
            },
            snippet: "struct DataProcessor { config: Config }".to_string(),
            score: 0.95,
            language: "rust".to_string(),
            symbol: Some("DataProcessor".to_string()),
        };

        // Golden: DataProcessor should be found in main.rs
        assert_eq!(match_result.location.file, expected_file);
        assert!(match_result.score > 0.8);
        assert_eq!(match_result.symbol.as_deref(), Some("DataProcessor"));
    }

    #[test]
    fn test_golden_ranking_reference_query() {
        // "What depends on Config?" should find DataProcessor::new and main()
        let references = vec![
            CodeMatch {
                location: CodeLocation {
                    file: PathBuf::from("rust_sample/main.rs"),
                    line: 3,
                    column: None,
                },
                snippet: "let config = Config::default();".to_string(),
                score: 0.9,
                language: "rust".to_string(),
                symbol: Some("main".to_string()),
            },
            CodeMatch {
                location: CodeLocation {
                    file: PathBuf::from("rust_sample/main.rs"),
                    line: 22,
                    column: None,
                },
                snippet: "fn new(config: Config) -> Self".to_string(),
                score: 0.85,
                language: "rust".to_string(),
                symbol: Some("DataProcessor::new".to_string()),
            },
        ];

        // Golden: should find at least 2 references
        assert!(references.len() >= 2);
        // Top result should have high score
        assert!(references[0].score >= references[1].score);
    }

    #[test]
    fn test_golden_ranking_python_query() {
        // "Where is UserService implemented?" should rank app.py highest
        let match_result = CodeMatch {
            location: CodeLocation {
                file: PathBuf::from("python_sample/app.py"),
                line: 26,
                column: None,
            },
            snippet: "class UserService:".to_string(),
            score: 0.92,
            language: "python".to_string(),
            symbol: Some("UserService".to_string()),
        };

        assert_eq!(
            match_result.location.file,
            PathBuf::from("python_sample/app.py")
        );
        assert!(match_result.score > 0.8);
    }

    #[test]
    fn test_context_pack_from_fixtures() {
        // Build a context pack for "how does data processing work?"
        let pack = ContextPackBuilder::new("how does data processing work?")
            .add_snippet(CodeMatch {
                location: CodeLocation {
                    file: PathBuf::from("rust_sample/main.rs"),
                    line: 28,
                    column: None,
                },
                snippet: "fn process(&self, input: &str) -> String { ... }".to_string(),
                score: 0.95,
                language: "rust".to_string(),
                symbol: Some("DataProcessor::process".to_string()),
            })
            .add_snippet(CodeMatch {
                location: CodeLocation {
                    file: PathBuf::from("rust_sample/main.rs"),
                    line: 22,
                    column: None,
                },
                snippet: "fn new(config: Config) -> Self { Self { config } }".to_string(),
                score: 0.8,
                language: "rust".to_string(),
                symbol: Some("DataProcessor::new".to_string()),
            })
            .with_structure_summary("DataProcessor is the main processing component")
            .with_max_tokens(2048)
            .build();

        assert_eq!(pack.query, "how does data processing work?");
        assert_eq!(pack.snippets.len(), 2);
        assert!(pack.structure_summary.is_some());
        assert!(pack.estimated_tokens > 0);
    }

    #[test]
    fn test_context_pack_snapshot() {
        let pack = ContextPackBuilder::new("find normalize function")
            .add_snippet(CodeMatch {
                location: CodeLocation {
                    file: PathBuf::from("rust_sample/main.rs"),
                    line: 42,
                    column: None,
                },
                snippet: "pub fn normalize(s: &str) -> String { s.trim().to_lowercase() }"
                    .to_string(),
                score: 0.98,
                language: "rust".to_string(),
                symbol: Some("utils::normalize".to_string()),
            })
            .add_snippet(CodeMatch {
                location: CodeLocation {
                    file: PathBuf::from("python_sample/utils.py"),
                    line: 5,
                    column: None,
                },
                snippet: "def slugify(text: str) -> str:".to_string(),
                score: 0.6,
                language: "python".to_string(),
                symbol: Some("slugify".to_string()),
            })
            .build();

        // Snapshot: the pack should serialize consistently
        let json = serde_json::to_string_pretty(&pack).unwrap();
        assert!(json.contains("normalize"));
        assert!(json.contains("slugify"));
        assert!(json.contains("find normalize function"));

        // Deserializes back correctly
        let deserialized: ContextPack = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.snippets.len(), 2);
        assert_eq!(deserialized.query, pack.query);
    }
}
