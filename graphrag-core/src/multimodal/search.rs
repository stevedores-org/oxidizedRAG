//! Cross-modal search seam: compose an
//! [`EmbeddingEngine`](crate::multimodal::embedding::EmbeddingEngine) and a
//! [`VectorStore`](crate::multimodal::store::VectorStore) into a unified
//! retrieval surface.

use async_trait::async_trait;
use thiserror::Error;

use super::embedding::EmbeddingError;
use super::store::{SearchFilters, SearchResult, VectorStoreError};
use super::types::MultimodalContent;

/// Errors returned by [`CrossModalSearcher`] implementations.
#[derive(Debug, Error)]
pub enum SearchError {
    /// Embedding the query failed.
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
    /// Vector store rejected the search.
    #[error("vector store error: {0}")]
    Store(#[from] VectorStoreError),
    /// Filter combination yielded zero candidates (distinct from
    /// "search ran but found nothing" — that returns `Ok(vec![])`).
    #[error("no candidates matched the filters")]
    NoCandidates,
}

/// Async trait for end-to-end query → results.
///
/// The canonical impl (planned for [meta-epic #182] phase 6.1) composes an
/// `EmbeddingEngine` with a `VectorStore`: embed the query, search, return.
/// Hybrid retrievers (vector + BM25 + PageRank) can implement this trait
/// alongside, reusing the existing `graphrag-core/src/retrieval/` machinery.
///
/// [meta-epic #182]: https://github.com/stevedores-org/oxidizedRAG/issues/182
#[async_trait]
pub trait CrossModalSearcher: Send + Sync {
    /// Find the `k` nearest results for a multimodal query.
    async fn search(
        &self,
        query: MultimodalContent,
        k: usize,
    ) -> Result<Vec<SearchResult>, SearchError>;

    /// Filtered variant. Default impl delegates to
    /// [`Self::search_with_filters`] callers that don't need filters; the
    /// other direction is the more common override.
    async fn search_with_filters(
        &self,
        query: MultimodalContent,
        k: usize,
        filters: SearchFilters,
    ) -> Result<Vec<SearchResult>, SearchError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_error_from_embedding_error_preserves_message() {
        let e: SearchError = EmbeddingError::Auth("bad key".into()).into();
        assert!(e.to_string().contains("bad key"));
    }

    #[test]
    fn search_error_from_store_error_preserves_message() {
        let e: SearchError = VectorStoreError::NotFound("abc".into()).into();
        assert!(e.to_string().contains("abc"));
    }
}
