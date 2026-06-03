//! Ingestion seam: turn a [`ContentSource`] into one or more
//! [`MultimodalContent`] values.

use async_trait::async_trait;
use thiserror::Error;

use super::types::{ContentSource, MultimodalContent};

/// Errors returned by [`MultimodalIngestor`] implementations.
#[derive(Debug, Error)]
pub enum IngestError {
    /// I/O failure reading from the filesystem.
    #[error("io error: {0}")]
    Io(String),
    /// HTTP failure fetching a [`ContentSource::RemoteUrl`].
    #[error("http error: {0}")]
    Http(String),
    /// MIME type could not be determined and no hint was provided.
    #[error("unknown mime type for {0}")]
    UnknownMime(String),
    /// MIME type is known but not supported by this ingestor.
    #[error("unsupported mime type {mime} for source {origin}")]
    UnsupportedMime {
        /// Detected MIME type.
        mime: String,
        /// Source that produced it (path or URL).
        ///
        /// Named `origin` rather than `source` to avoid thiserror's
        /// implicit `#[source]` field convention.
        origin: String,
    },
    /// Glob pattern was invalid.
    #[error("invalid glob pattern: {0}")]
    InvalidPattern(String),
    /// Ingestor exceeded a configured size, time, or concurrency limit.
    #[error("limit exceeded: {0}")]
    LimitExceeded(String),
}

/// Async trait every multimodal content ingestor implements.
///
/// A [`ContentSource::Directory`] typically expands to many
/// [`MultimodalContent`] values; a [`ContentSource::LocalFile`] typically
/// produces exactly one. Ingestors are free to decide; callers must not
/// assume a one-to-one mapping.
#[async_trait]
pub trait MultimodalIngestor: Send + Sync {
    /// Materialize one source into one or more content values.
    async fn ingest(&self, source: ContentSource) -> Result<Vec<MultimodalContent>, IngestError>;

    /// Materialize many sources. Default impl falls back to sequential
    /// `ingest` calls — implementations with concurrency should override.
    async fn ingest_batch(
        &self,
        sources: Vec<ContentSource>,
    ) -> Result<Vec<MultimodalContent>, IngestError> {
        let mut out = Vec::new();
        for src in sources {
            out.extend(self.ingest(src).await?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_mime_error_carries_both_fields() {
        let err = IngestError::UnsupportedMime {
            mime: "application/x-tar".into(),
            origin: "blob.tar".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("application/x-tar"));
        assert!(msg.contains("blob.tar"));
    }
}
