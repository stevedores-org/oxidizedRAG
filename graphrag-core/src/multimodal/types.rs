//! Content type primitives shared across the multimodal seams.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// The five content modalities the multimodal stack handles end-to-end.
///
/// Used both as a runtime tag on [`MultimodalContent`] / [`crate::multimodal::embedding::Embedding`]
/// and as a filter value on [`crate::multimodal::store::SearchFilters`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    /// UTF-8 text.
    Text,
    /// Raster image (PNG / JPEG / WebP / etc.).
    Image,
    /// Audio clip (MP3 / WAV / OGG / etc.).
    Audio,
    /// Video clip (MP4 / WebM / etc.).
    Video,
    /// PDF document.
    Pdf,
}

/// A piece of content ready to embed.
///
/// Variants carry the raw bytes (or text). MIME types are kept alongside the
/// bytes so embedding engines that need to dispatch on container format don't
/// have to re-sniff. `MultimodalContent` is **not** the same as
/// [`ContentSource`]: the source describes *where* the content lives
/// (filesystem, URL, in-memory blob); this enum describes *what* the content
/// is once a [`MultimodalIngestor`](crate::multimodal::ingest::MultimodalIngestor)
/// has materialized it.
///
/// ## Example
///
/// ```rust
/// use graphrag_core::multimodal::{ContentType, MultimodalContent};
///
/// let txt = MultimodalContent::Text("hello world".into());
/// assert_eq!(txt.content_type(), ContentType::Text);
///
/// let img = MultimodalContent::Image {
///     bytes: vec![0x89, 0x50, 0x4E, 0x47],
///     mime: "image/png".into(),
/// };
/// assert_eq!(img.content_type(), ContentType::Image);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MultimodalContent {
    /// UTF-8 text payload.
    Text(String),
    /// Raster image with its IANA media type.
    Image {
        /// Raw image bytes.
        bytes: Vec<u8>,
        /// IANA media type (e.g. `image/png`, `image/jpeg`).
        mime: String,
    },
    /// Audio payload with its IANA media type.
    Audio {
        /// Raw audio bytes.
        bytes: Vec<u8>,
        /// IANA media type (e.g. `audio/mpeg`, `audio/wav`).
        mime: String,
    },
    /// Video payload with its IANA media type.
    Video {
        /// Raw video bytes.
        bytes: Vec<u8>,
        /// IANA media type (e.g. `video/mp4`, `video/webm`).
        mime: String,
    },
    /// PDF document.
    Pdf {
        /// Raw PDF bytes.
        bytes: Vec<u8>,
    },
}

impl MultimodalContent {
    /// Return the discriminant for filtering, dispatch, and tagging.
    pub fn content_type(&self) -> ContentType {
        match self {
            Self::Text(_) => ContentType::Text,
            Self::Image { .. } => ContentType::Image,
            Self::Audio { .. } => ContentType::Audio,
            Self::Video { .. } => ContentType::Video,
            Self::Pdf { .. } => ContentType::Pdf,
        }
    }

    /// Byte length of the underlying payload. For [`Self::Text`], this is the
    /// UTF-8 byte length, **not** the character count.
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Text(s) => s.len(),
            Self::Image { bytes, .. }
            | Self::Audio { bytes, .. }
            | Self::Video { bytes, .. }
            | Self::Pdf { bytes } => bytes.len(),
        }
    }
}

/// Where a piece of content lives before ingestion.
///
/// [`MultimodalIngestor`](crate::multimodal::ingest::MultimodalIngestor)
/// implementations resolve a [`ContentSource`] into one or more
/// [`MultimodalContent`] values — e.g. globbing a directory, downloading a
/// URL, or wrapping in-memory bytes.
///
/// ## Example
///
/// ```rust
/// use std::path::PathBuf;
/// use graphrag_core::multimodal::ContentSource;
///
/// let f = ContentSource::LocalFile { path: PathBuf::from("README.md"), mime_type: None };
/// let u = ContentSource::RemoteUrl { url: "https://example.com/img.png".into() };
/// let d = ContentSource::DirectContent { bytes: vec![1, 2, 3], mime_type: "application/octet-stream".into() };
/// let dir = ContentSource::Directory { path: PathBuf::from("docs/"), pattern: "**/*.md".into() };
/// # let _ = (f, u, d, dir);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContentSource {
    /// A single file on the local filesystem.
    LocalFile {
        /// Path to the file.
        path: PathBuf,
        /// Optional MIME type hint. If `None`, the ingestor sniffs from
        /// extension and magic bytes.
        mime_type: Option<String>,
    },
    /// A remote URL (HTTP or HTTPS).
    RemoteUrl {
        /// Fully-qualified URL.
        url: String,
    },
    /// In-memory bytes with an explicit MIME type — useful for streams,
    /// stdin, or callers that already have the payload in hand.
    ///
    /// Issue #153 named this field `data`; the repo's clippy config bans
    /// `data` as a standalone identifier, so it's `bytes` here, matching
    /// the [`MultimodalContent`] variants.
    DirectContent {
        /// Raw payload bytes.
        bytes: Vec<u8>,
        /// IANA media type — required, since there's no path to sniff.
        mime_type: String,
    },
    /// A directory plus a glob pattern. Ingestors expand this into one
    /// [`Self::LocalFile`] per match.
    Directory {
        /// Root path to glob from.
        path: PathBuf,
        /// Glob pattern relative to `path` (e.g. `"**/*.{txt,pdf}"`).
        pattern: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_type_dispatches_per_variant() {
        assert_eq!(MultimodalContent::Text("hi".into()).content_type(), ContentType::Text);
        assert_eq!(
            MultimodalContent::Image { bytes: vec![1], mime: "image/png".into() }.content_type(),
            ContentType::Image
        );
        assert_eq!(
            MultimodalContent::Audio { bytes: vec![1], mime: "audio/mpeg".into() }.content_type(),
            ContentType::Audio
        );
        assert_eq!(
            MultimodalContent::Video { bytes: vec![1], mime: "video/mp4".into() }.content_type(),
            ContentType::Video
        );
        assert_eq!(
            MultimodalContent::Pdf { bytes: vec![1] }.content_type(),
            ContentType::Pdf
        );
    }

    #[test]
    fn byte_len_matches_payload() {
        assert_eq!(MultimodalContent::Text("hello".into()).byte_len(), 5);
        assert_eq!(
            MultimodalContent::Image { bytes: vec![0; 32], mime: "image/png".into() }.byte_len(),
            32
        );
        assert_eq!(MultimodalContent::Pdf { bytes: vec![0; 100] }.byte_len(), 100);
    }

    #[test]
    fn content_round_trips_through_json() {
        let original = MultimodalContent::Image {
            bytes: vec![0x89, 0x50, 0x4E, 0x47],
            mime: "image/png".into(),
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let back: MultimodalContent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, back);
    }

    #[test]
    fn source_round_trips_through_json() {
        let original = ContentSource::Directory {
            path: PathBuf::from("docs"),
            pattern: "**/*.md".into(),
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let back: ContentSource = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, back);
    }
}
