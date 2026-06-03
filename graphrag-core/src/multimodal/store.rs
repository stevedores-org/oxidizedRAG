//! Vector storage seam for multimodal embeddings.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::embedding::Embedding;
use super::types::ContentType;

/// Filters applied to a [`VectorStore::search`] call. All fields are
/// optional; a default `SearchFilters` returns the unfiltered top-k.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchFilters {
    /// Restrict to these content types. Empty / `None` = no restriction.
    pub content_types: Option<Vec<ContentType>>,
    /// Restrict to embeddings whose `created_at` falls in `[start, end]`.
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Tag-equality filter on [`crate::multimodal::embedding::EmbeddingMetadata::tags`].
    /// All entries must match.
    pub metadata_filters: HashMap<String, String>,
    /// Drop results with a similarity score below this threshold.
    pub min_score: Option<f32>,
}

/// One hit from a [`VectorStore::search`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// The stored embedding.
    pub embedding: Embedding,
    /// Similarity score (cosine in canonical impls — higher is closer).
    pub score: f32,
}

/// Errors returned by [`VectorStore`] implementations.
#[derive(Debug, Error)]
pub enum VectorStoreError {
    /// Dimensionality mismatch between input and store.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Dimensionality the store expects.
        expected: usize,
        /// Dimensionality the caller provided.
        actual: usize,
    },
    /// Backend storage failure (disk, network, sqlite, etc.).
    #[error("storage backend error: {0}")]
    Backend(String),
    /// Requested ID was not found.
    #[error("not found: {0}")]
    NotFound(String),
    /// Invalid filter or query argument.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

/// Async trait every multimodal vector store implements.
///
/// Default impls of `search_by_type` and `count` delegate to the primitive
/// methods; backends with native indexes should override them.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert a single embedding. Backends are free to dedup on
    /// [`Embedding::content_hash`] or upsert by [`Embedding::id`].
    async fn store(&self, embedding: Embedding) -> Result<(), VectorStoreError>;

    /// Insert a batch. Default impl falls back to sequential inserts —
    /// backends with native batch APIs should override.
    async fn store_batch(&self, embeddings: Vec<Embedding>) -> Result<(), VectorStoreError> {
        for e in embeddings {
            self.store(e).await?;
        }
        Ok(())
    }

    /// k-nearest-neighbour search with optional filtering.
    async fn search(
        &self,
        query: &[f32],
        k: usize,
        filters: SearchFilters,
    ) -> Result<Vec<SearchResult>, VectorStoreError>;

    /// Convenience: search restricted to one content type. Default
    /// implementation wraps [`Self::search`] with a single-type filter.
    async fn search_by_type(
        &self,
        query: &[f32],
        k: usize,
        content_type: ContentType,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let filters = SearchFilters {
            content_types: Some(vec![content_type]),
            ..Default::default()
        };
        self.search(query, k, filters).await
    }

    /// Fetch by stable [`Embedding::id`].
    async fn get(&self, id: &str) -> Result<Option<Embedding>, VectorStoreError>;

    /// Delete by stable [`Embedding::id`]. Returns `Ok(())` whether or not
    /// the ID existed — backends may distinguish via logs.
    async fn delete(&self, id: &str) -> Result<(), VectorStoreError>;

    /// Total number of embeddings in the store.
    async fn count(&self) -> Result<usize, VectorStoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_filters_match_anything() {
        let f = SearchFilters::default();
        assert!(f.content_types.is_none());
        assert!(f.date_range.is_none());
        assert!(f.metadata_filters.is_empty());
        assert!(f.min_score.is_none());
    }

    #[test]
    fn dimension_mismatch_error_carries_expected_and_actual() {
        let err = VectorStoreError::DimensionMismatch { expected: 1408, actual: 1536 };
        let msg = err.to_string();
        assert!(msg.contains("1408"));
        assert!(msg.contains("1536"));
    }
}
