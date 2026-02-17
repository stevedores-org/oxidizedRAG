//! Shared test fixtures and helpers for RAG agent tests

pub use graphrag_core::core::{Document, DocumentId, KnowledgeGraph};
use graphrag_core::graph::GraphBuilder;
pub use graphrag_core::text::TextProcessor;
pub use graphrag_core::Result;
use std::fs;

pub const FIXTURE_DIR: &str = "tests/fixtures/code_samples";

/// Load a fixture file by name
pub fn load_fixture(name: &str) -> String {
    let path = format!("{}/{}", FIXTURE_DIR, name);
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to load fixture '{}': {}", path, e))
}

/// Create a Document from a fixture file
pub fn fixture_document(filename: &str) -> Document {
    let content = load_fixture(filename);
    let doc_id = DocumentId::new(filename.replace('.', "_"));
    Document::new(doc_id, filename.to_string(), content)
}

/// Index fixture files into a GraphRAG knowledge graph
pub fn index_fixtures(filenames: &[&str], chunk_size: usize) -> Result<KnowledgeGraph> {
    let mut graph = KnowledgeGraph::new();
    let processor = TextProcessor::new(chunk_size, chunk_size / 5)?;

    for filename in filenames {
        let doc = fixture_document(filename);
        let chunks = processor.chunk_text(&doc)?;
        let doc_with_chunks = Document {
            chunks,
            ..doc
        };
        graph.add_document(doc_with_chunks)?;
    }

    Ok(graph)
}

/// Build a knowledge graph from fixture files with entity extraction
pub fn build_graph_from_fixtures(filenames: &[&str]) -> Result<KnowledgeGraph> {
    let documents: Vec<Document> = filenames.iter().map(|f| fixture_document(f)).collect();
    let mut builder = GraphBuilder::new(500, 100, 0.5, 0.7, 10)?;
    builder.build_graph(documents)
}

/// Parse Rust code with tree-sitter and validate syntax
#[cfg(feature = "code-chunking")]
pub fn validate_rust_syntax(code: &str) -> std::result::Result<(), String> {
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

