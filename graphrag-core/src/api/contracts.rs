//! Shared deployment contract types for server/CLI communication.
//!
//! These types define the wire format for GraphRAG API requests and responses,
//! ensuring consistent serialization between server and client implementations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Options controlling query behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptions {
    pub include_sources: bool,
    pub include_scores: bool,
    pub fusion_method: Option<String>,
}

/// A query request sent to the GraphRAG API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    pub limit: usize,
    pub options: QueryOptions,
}

/// A single result item returned from a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub id: String,
    pub content: String,
    pub score: f32,
    pub source: Option<String>,
}

/// The response to a query request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub results: Vec<QueryResult>,
    pub total: usize,
    pub query_time_ms: u64,
}

/// A single document to be indexed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDocument {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

/// A request to index one or more documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRequest {
    pub documents: Vec<IndexDocument>,
}

/// The response after an indexing operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResponse {
    pub indexed: usize,
    pub errors: Vec<String>,
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: serialize to JSON and deserialize back, asserting round-trip equality.
    fn roundtrip<T>(value: &T) -> T
    where
        T: Serialize + for<'de> Deserialize<'de> + std::fmt::Debug,
    {
        let json = serde_json::to_string(value).expect("serialize");
        serde_json::from_str(&json).expect("deserialize")
    }

    #[test]
    fn test_query_options_roundtrip() {
        let opts = QueryOptions {
            include_sources: true,
            include_scores: false,
            fusion_method: Some("rrf".to_string()),
        };
        let rt = roundtrip(&opts);
        assert_eq!(rt.include_sources, opts.include_sources);
        assert_eq!(rt.include_scores, opts.include_scores);
        assert_eq!(rt.fusion_method, opts.fusion_method);
    }

    #[test]
    fn test_query_options_none_fusion() {
        let opts = QueryOptions {
            include_sources: false,
            include_scores: true,
            fusion_method: None,
        };
        let rt = roundtrip(&opts);
        assert!(rt.fusion_method.is_none());
    }

    #[test]
    fn test_query_request_roundtrip() {
        let req = QueryRequest {
            query: "What is GraphRAG?".to_string(),
            limit: 10,
            options: QueryOptions {
                include_sources: true,
                include_scores: true,
                fusion_method: None,
            },
        };
        let rt = roundtrip(&req);
        assert_eq!(rt.query, req.query);
        assert_eq!(rt.limit, req.limit);
        assert_eq!(rt.options.include_sources, true);
    }

    #[test]
    fn test_query_result_roundtrip() {
        let result = QueryResult {
            id: "doc-1".to_string(),
            content: "GraphRAG combines graphs with RAG.".to_string(),
            score: 0.95,
            source: Some("paper.pdf".to_string()),
        };
        let rt = roundtrip(&result);
        assert_eq!(rt.id, result.id);
        assert_eq!(rt.content, result.content);
        assert_eq!(rt.score, result.score);
        assert_eq!(rt.source, result.source);
    }

    #[test]
    fn test_query_result_no_source() {
        let result = QueryResult {
            id: "doc-2".to_string(),
            content: "Some content".to_string(),
            score: 0.5,
            source: None,
        };
        let rt = roundtrip(&result);
        assert!(rt.source.is_none());
    }

    #[test]
    fn test_query_response_roundtrip() {
        let resp = QueryResponse {
            results: vec![
                QueryResult {
                    id: "1".to_string(),
                    content: "First".to_string(),
                    score: 0.9,
                    source: None,
                },
                QueryResult {
                    id: "2".to_string(),
                    content: "Second".to_string(),
                    score: 0.8,
                    source: Some("src.txt".to_string()),
                },
            ],
            total: 2,
            query_time_ms: 42,
        };
        let rt = roundtrip(&resp);
        assert_eq!(rt.results.len(), 2);
        assert_eq!(rt.total, resp.total);
        assert_eq!(rt.query_time_ms, resp.query_time_ms);
    }

    #[test]
    fn test_query_response_empty() {
        let resp = QueryResponse {
            results: vec![],
            total: 0,
            query_time_ms: 1,
        };
        let rt = roundtrip(&resp);
        assert!(rt.results.is_empty());
    }

    #[test]
    fn test_index_document_roundtrip() {
        let mut metadata = HashMap::new();
        metadata.insert("author".to_string(), "Alice".to_string());
        metadata.insert("year".to_string(), "2025".to_string());

        let doc = IndexDocument {
            id: "doc-42".to_string(),
            content: "Document content here.".to_string(),
            metadata,
        };
        let rt = roundtrip(&doc);
        assert_eq!(rt.id, doc.id);
        assert_eq!(rt.content, doc.content);
        assert_eq!(rt.metadata.get("author").unwrap(), "Alice");
        assert_eq!(rt.metadata.get("year").unwrap(), "2025");
    }

    #[test]
    fn test_index_document_empty_metadata() {
        let doc = IndexDocument {
            id: "doc-0".to_string(),
            content: "No metadata".to_string(),
            metadata: HashMap::new(),
        };
        let rt = roundtrip(&doc);
        assert!(rt.metadata.is_empty());
    }

    #[test]
    fn test_index_request_roundtrip() {
        let req = IndexRequest {
            documents: vec![
                IndexDocument {
                    id: "a".to_string(),
                    content: "Alpha".to_string(),
                    metadata: HashMap::new(),
                },
                IndexDocument {
                    id: "b".to_string(),
                    content: "Beta".to_string(),
                    metadata: HashMap::new(),
                },
            ],
        };
        let rt = roundtrip(&req);
        assert_eq!(rt.documents.len(), 2);
        assert_eq!(rt.documents[0].id, "a");
        assert_eq!(rt.documents[1].id, "b");
    }

    #[test]
    fn test_index_response_roundtrip() {
        let resp = IndexResponse {
            indexed: 5,
            errors: vec!["failed doc-3".to_string()],
        };
        let rt = roundtrip(&resp);
        assert_eq!(rt.indexed, resp.indexed);
        assert_eq!(rt.errors, resp.errors);
    }

    #[test]
    fn test_index_response_no_errors() {
        let resp = IndexResponse {
            indexed: 10,
            errors: vec![],
        };
        let rt = roundtrip(&resp);
        assert_eq!(rt.indexed, 10);
        assert!(rt.errors.is_empty());
    }

    #[test]
    fn test_health_response_roundtrip() {
        let resp = HealthResponse {
            status: "healthy".to_string(),
            version: "0.1.0".to_string(),
            uptime_seconds: 3600,
        };
        let rt = roundtrip(&resp);
        assert_eq!(rt.status, resp.status);
        assert_eq!(rt.version, resp.version);
        assert_eq!(rt.uptime_seconds, resp.uptime_seconds);
    }
}
