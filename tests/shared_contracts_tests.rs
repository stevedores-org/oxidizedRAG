//! Cross-crate serialization tests for shared deployment contracts.
//!
//! These tests verify that the contract types defined in `graphrag_core::api::contracts`
//! serialize to JSON and deserialize back correctly when used from outside the
//! defining crate, ensuring wire-format compatibility between server and CLI.
//!
//! Gated behind `#[cfg(feature = "api")]` because the contracts module lives
//! inside the `api` feature of graphrag-core. Note: the `api` feature currently
//! has pre-existing compilation issues, so these tests may not run until those
//! are resolved.

#![cfg(feature = "api")]

use graphrag_core::api::contracts::{
    HealthResponse, IndexDocument, IndexRequest, IndexResponse, QueryOptions, QueryRequest,
    QueryResponse, QueryResult,
};
use std::collections::HashMap;

/// Helper: serialize a value to JSON and deserialize it back.
fn roundtrip_json<T>(value: &T) -> T
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de> + std::fmt::Debug,
{
    let json = serde_json::to_string(value).expect("failed to serialize to JSON");
    serde_json::from_str(&json).expect("failed to deserialize from JSON")
}

// ---------------------------------------------------------------------------
// QueryRequest round-trip
// ---------------------------------------------------------------------------

#[test]
fn query_request_roundtrip() {
    let req = QueryRequest {
        query: "How does GraphRAG work?".to_string(),
        limit: 5,
        options: QueryOptions {
            include_sources: true,
            include_scores: true,
            fusion_method: Some("rrf".to_string()),
        },
    };

    let rt = roundtrip_json(&req);

    assert_eq!(rt.query, "How does GraphRAG work?");
    assert_eq!(rt.limit, 5);
    assert!(rt.options.include_sources);
    assert!(rt.options.include_scores);
    assert_eq!(rt.options.fusion_method.as_deref(), Some("rrf"));
}

#[test]
fn query_request_none_fusion_method() {
    let req = QueryRequest {
        query: "test".to_string(),
        limit: 1,
        options: QueryOptions {
            include_sources: false,
            include_scores: false,
            fusion_method: None,
        },
    };

    let rt = roundtrip_json(&req);

    assert!(rt.options.fusion_method.is_none());
}

// ---------------------------------------------------------------------------
// QueryResponse round-trip
// ---------------------------------------------------------------------------

#[test]
fn query_response_roundtrip() {
    let resp = QueryResponse {
        results: vec![
            QueryResult {
                id: "doc-1".to_string(),
                content: "GraphRAG combines graphs with RAG.".to_string(),
                score: 0.95,
                source: Some("paper.pdf".to_string()),
            },
            QueryResult {
                id: "doc-2".to_string(),
                content: "Knowledge graphs improve retrieval.".to_string(),
                score: 0.87,
                source: None,
            },
        ],
        total: 2,
        query_time_ms: 42,
    };

    let rt = roundtrip_json(&resp);

    assert_eq!(rt.results.len(), 2);
    assert_eq!(rt.total, 2);
    assert_eq!(rt.query_time_ms, 42);

    assert_eq!(rt.results[0].id, "doc-1");
    assert_eq!(rt.results[0].content, "GraphRAG combines graphs with RAG.");
    assert_eq!(rt.results[0].score, 0.95);
    assert_eq!(rt.results[0].source.as_deref(), Some("paper.pdf"));

    assert_eq!(rt.results[1].id, "doc-2");
    assert!(rt.results[1].source.is_none());
}

#[test]
fn query_response_empty_results() {
    let resp = QueryResponse {
        results: vec![],
        total: 0,
        query_time_ms: 1,
    };

    let rt = roundtrip_json(&resp);

    assert!(rt.results.is_empty());
    assert_eq!(rt.total, 0);
}

// ---------------------------------------------------------------------------
// IndexRequest round-trip
// ---------------------------------------------------------------------------

#[test]
fn index_request_roundtrip() {
    let mut metadata = HashMap::new();
    metadata.insert("author".to_string(), "Alice".to_string());
    metadata.insert("year".to_string(), "2025".to_string());

    let req = IndexRequest {
        documents: vec![
            IndexDocument {
                id: "doc-1".to_string(),
                content: "First document content.".to_string(),
                metadata: metadata.clone(),
            },
            IndexDocument {
                id: "doc-2".to_string(),
                content: "Second document content.".to_string(),
                metadata: HashMap::new(),
            },
        ],
    };

    let rt = roundtrip_json(&req);

    assert_eq!(rt.documents.len(), 2);
    assert_eq!(rt.documents[0].id, "doc-1");
    assert_eq!(rt.documents[0].content, "First document content.");
    assert_eq!(rt.documents[0].metadata.get("author").unwrap(), "Alice");
    assert_eq!(rt.documents[0].metadata.get("year").unwrap(), "2025");

    assert_eq!(rt.documents[1].id, "doc-2");
    assert!(rt.documents[1].metadata.is_empty());
}

// ---------------------------------------------------------------------------
// IndexResponse round-trip
// ---------------------------------------------------------------------------

#[test]
fn index_response_roundtrip() {
    let resp = IndexResponse {
        indexed: 10,
        errors: vec!["failed doc-3".to_string(), "timeout doc-7".to_string()],
    };

    let rt = roundtrip_json(&resp);

    assert_eq!(rt.indexed, 10);
    assert_eq!(rt.errors.len(), 2);
    assert_eq!(rt.errors[0], "failed doc-3");
    assert_eq!(rt.errors[1], "timeout doc-7");
}

#[test]
fn index_response_no_errors() {
    let resp = IndexResponse {
        indexed: 5,
        errors: vec![],
    };

    let rt = roundtrip_json(&resp);

    assert_eq!(rt.indexed, 5);
    assert!(rt.errors.is_empty());
}

// ---------------------------------------------------------------------------
// HealthResponse round-trip
// ---------------------------------------------------------------------------

#[test]
fn health_response_roundtrip() {
    let resp = HealthResponse {
        status: "healthy".to_string(),
        version: "0.1.0".to_string(),
        uptime_seconds: 3600,
    };

    let rt = roundtrip_json(&resp);

    assert_eq!(rt.status, "healthy");
    assert_eq!(rt.version, "0.1.0");
    assert_eq!(rt.uptime_seconds, 3600);
}

// ---------------------------------------------------------------------------
// JSON format verification
// ---------------------------------------------------------------------------

#[test]
fn query_request_json_has_expected_keys() {
    let req = QueryRequest {
        query: "test".to_string(),
        limit: 3,
        options: QueryOptions {
            include_sources: true,
            include_scores: false,
            fusion_method: None,
        },
    };

    let json_str = serde_json::to_string(&req).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert!(value.get("query").is_some());
    assert!(value.get("limit").is_some());
    assert!(value.get("options").is_some());

    let options = value.get("options").unwrap();
    assert!(options.get("include_sources").is_some());
    assert!(options.get("include_scores").is_some());
}

#[test]
fn health_response_json_has_expected_keys() {
    let resp = HealthResponse {
        status: "ok".to_string(),
        version: "1.0.0".to_string(),
        uptime_seconds: 0,
    };

    let json_str = serde_json::to_string(&resp).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert!(value.get("status").is_some());
    assert!(value.get("version").is_some());
    assert!(value.get("uptime_seconds").is_some());
}
