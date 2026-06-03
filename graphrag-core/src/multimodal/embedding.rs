//! Embedding type + [`EmbeddingEngine`] trait seam for multimodal backends.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::types::{ContentType, MultimodalContent};

/// A computed embedding for a piece of content.
///
/// Dimensionality is **not** fixed on the type — backends report their
/// dimensionality via [`ModelInfo::dimensions`]. Callers that need to
/// validate dimensionality at insert time check
/// [`Embedding::vector`]`.len()` against the dimension reported by the
/// store or engine they're handing it to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// Stable identifier (UUID v4 by convention).
    pub id: String,
    /// Hex-encoded content hash (SHA-256 over the source bytes) for
    /// deduplication. Engines that compute this must agree on the
    /// canonicalization rules.
    pub content_hash: String,
    /// The embedding vector. Length is backend-dependent.
    pub vector: Vec<f32>,
    /// Discriminant of the source content.
    pub content_type: ContentType,
    /// Backend-supplied metadata (model name, mime type, custom tags).
    pub metadata: EmbeddingMetadata,
    /// UTC timestamp at which the embedding was produced.
    pub created_at: DateTime<Utc>,
}

/// Free-form metadata carried alongside an [`Embedding`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    /// Source MIME type, if known.
    pub mime_type: Option<String>,
    /// Model that produced the embedding (e.g. `"gemini-embedding-2-preview"`).
    pub model: Option<String>,
    /// Caller-supplied tags. Storage backends may index on these.
    pub tags: HashMap<String, String>,
}

/// Backend-reported model identity, used by callers to validate compatibility
/// (dimensionality match, model-name match, modality coverage).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (e.g. `"gemini-embedding-2-preview"`).
    pub name: String,
    /// Output vector dimensionality.
    pub dimensions: usize,
    /// Content types this model accepts as input.
    pub supported_types: Vec<ContentType>,
}

/// Errors returned by [`EmbeddingEngine`] implementations.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// Backend rejected the input as unsupported (wrong modality, oversized,
    /// disallowed MIME type, etc.).
    #[error("unsupported input: {0}")]
    Unsupported(String),
    /// Transport error talking to a remote backend.
    #[error("transport error: {0}")]
    Transport(String),
    /// Backend authentication failure (missing or invalid API key).
    #[error("authentication failed: {0}")]
    Auth(String),
    /// Backend returned an unexpected response shape.
    #[error("protocol error: {0}")]
    Protocol(String),
    /// Local model loading / inference failed.
    #[error("local inference error: {0}")]
    LocalInference(String),
}

/// Async trait every multimodal embedding backend implements.
///
/// Implementations live in their own crates / modules and are wired up via
/// `Arc<dyn EmbeddingEngine>`. The trait deliberately mirrors the existing
/// text-only [`crate::embeddings::EmbeddingProvider`] but takes
/// [`MultimodalContent`] and returns rich [`Embedding`] values.
#[async_trait]
pub trait EmbeddingEngine: Send + Sync {
    /// Embed a single piece of content.
    async fn embed(&self, content: MultimodalContent) -> Result<Embedding, EmbeddingError>;

    /// Embed a batch. Default impl falls back to sequential `embed` calls —
    /// backends with a real batch endpoint should override this.
    async fn embed_batch(
        &self,
        batch: Vec<MultimodalContent>,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        let mut out = Vec::with_capacity(batch.len());
        for item in batch {
            out.push(self.embed(item).await?);
        }
        Ok(out)
    }

    /// Report the underlying model's identity and capabilities.
    fn model_info(&self) -> ModelInfo;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_info_round_trips_through_json() {
        let info = ModelInfo {
            name: "gemini-embedding-2-preview".into(),
            dimensions: 1408,
            supported_types: vec![
                ContentType::Text,
                ContentType::Image,
                ContentType::Audio,
            ],
        };
        let json = serde_json::to_string(&info).expect("serialize");
        let back: ModelInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(info, back);
    }

    #[test]
    fn embedding_metadata_default_is_empty() {
        let meta = EmbeddingMetadata::default();
        assert!(meta.mime_type.is_none());
        assert!(meta.model.is_none());
        assert!(meta.tags.is_empty());
    }

    #[test]
    fn embedding_error_display_carries_message() {
        let err = EmbeddingError::Auth("missing GEMINI_API_KEY".into());
        assert!(err.to_string().contains("missing GEMINI_API_KEY"));
    }
}
