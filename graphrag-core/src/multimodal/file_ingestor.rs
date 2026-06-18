//! File-system implementation of [`MultimodalIngestor`].

use std::fs;
use std::path::{Path, PathBuf};

use async_trait::async_trait;

use super::ingest::{IngestError, MultimodalIngestor};
use super::types::{ContentSource, MultimodalContent};

/// Local file ingestor for multimodal content.
///
/// Supports individual local files, in-memory direct content, and directory
/// expansion via glob patterns. Remote URLs are intentionally left to the
/// planned remote ingestor.
#[derive(Debug, Clone, Default)]
pub struct FileIngestor {
    max_bytes: Option<u64>,
}

impl FileIngestor {
    /// Create a file ingestor with no size limit.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a file ingestor with a maximum payload size.
    pub fn with_max_bytes(max_bytes: u64) -> Self {
        Self {
            max_bytes: Some(max_bytes),
        }
    }

    fn ensure_size(&self, len: u64, origin: &str) -> Result<(), IngestError> {
        if let Some(max) = self.max_bytes {
            if len > max {
                return Err(IngestError::LimitExceeded(format!(
                    "{origin} is {len} bytes, exceeds {max} byte limit"
                )));
            }
        }
        Ok(())
    }

    fn ingest_local_file(
        &self,
        path: PathBuf,
        mime_type: Option<String>,
    ) -> Result<Vec<MultimodalContent>, IngestError> {
        let origin = path.display().to_string();
        let metadata = fs::metadata(&path).map_err(|e| IngestError::Io(e.to_string()))?;
        if !metadata.is_file() {
            return Err(IngestError::Io(format!("{origin} is not a file")));
        }
        self.ensure_size(metadata.len(), &origin)?;

        let bytes = fs::read(&path).map_err(|e| IngestError::Io(e.to_string()))?;
        let mime = mime_type
            .or_else(|| sniff_mime(&bytes))
            .or_else(|| mime_from_extension(&path))
            .ok_or_else(|| IngestError::UnknownMime(origin.clone()))?;

        content_from_bytes(bytes, &mime, origin).map(|content| vec![content])
    }

    fn ingest_directory(
        &self,
        path: PathBuf,
        pattern: String,
    ) -> Result<Vec<MultimodalContent>, IngestError> {
        let glob_pattern = path.join(pattern).display().to_string();
        let mut paths = glob::glob(&glob_pattern)
            .map_err(|e| IngestError::InvalidPattern(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| IngestError::InvalidPattern(e.to_string()))?;
        paths.sort();

        let mut out = Vec::new();
        for file_path in paths.into_iter().filter(|p| p.is_file()) {
            out.extend(self.ingest_local_file(file_path, None)?);
        }
        Ok(out)
    }
}

#[async_trait]
impl MultimodalIngestor for FileIngestor {
    async fn ingest(&self, source: ContentSource) -> Result<Vec<MultimodalContent>, IngestError> {
        match source {
            ContentSource::LocalFile { path, mime_type } => self.ingest_local_file(path, mime_type),
            ContentSource::DirectContent { bytes, mime_type } => {
                self.ensure_size(bytes.len() as u64, "direct content")?;
                content_from_bytes(bytes, &mime_type, "direct content".into()).map(|c| vec![c])
            },
            ContentSource::Directory { path, pattern } => self.ingest_directory(path, pattern),
            ContentSource::RemoteUrl { url } => Err(IngestError::UnsupportedMime {
                mime: "remote-url".into(),
                origin: url,
            }),
        }
    }
}

fn sniff_mime(bytes: &[u8]) -> Option<String> {
    infer::get(bytes).map(|kind| kind.mime_type().to_string())
}

fn mime_from_extension(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    let mime = match ext.as_str() {
        "txt" | "md" | "markdown" => "text/plain",
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        _ => return None,
    };
    Some(mime.to_string())
}

fn content_from_bytes(
    bytes: Vec<u8>,
    mime: &str,
    origin: String,
) -> Result<MultimodalContent, IngestError> {
    match mime {
        m if m.starts_with("text/") => String::from_utf8(bytes)
            .map(MultimodalContent::Text)
            .map_err(|e| IngestError::UnsupportedMime {
                mime: format!("{mime}; invalid utf-8: {e}"),
                origin,
            }),
        "application/pdf" => Ok(MultimodalContent::Pdf { bytes }),
        m if m.starts_with("image/") => Ok(MultimodalContent::Image {
            bytes,
            mime: mime.to_string(),
        }),
        m if m.starts_with("audio/") => Ok(MultimodalContent::Audio {
            bytes,
            mime: mime.to_string(),
        }),
        m if m.starts_with("video/") => Ok(MultimodalContent::Video {
            bytes,
            mime: mime.to_string(),
        }),
        _ => Err(IngestError::UnsupportedMime {
            mime: mime.into(),
            origin,
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn ingests_text_file_from_extension() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("note.txt");
        fs::write(&path, "hello multimodal").expect("write");

        let out = FileIngestor::new()
            .ingest(ContentSource::LocalFile {
                path,
                mime_type: None,
            })
            .await
            .expect("ingest");

        assert_eq!(
            out,
            vec![MultimodalContent::Text("hello multimodal".into())]
        );
    }

    #[tokio::test]
    async fn ingests_png_by_magic_bytes() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("image.bin");
        fs::write(
            &path,
            [
                0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n', 0, 0, 0, 0,
            ],
        )
        .expect("write");

        let out = FileIngestor::new()
            .ingest(ContentSource::LocalFile {
                path,
                mime_type: None,
            })
            .await
            .expect("ingest");

        assert!(matches!(
            &out[0],
            MultimodalContent::Image { mime, .. } if mime == "image/png"
        ));
    }

    #[tokio::test]
    async fn expands_directory_glob_in_sorted_order() {
        let dir = tempdir().expect("tempdir");
        fs::write(dir.path().join("b.txt"), "b").expect("write");
        fs::write(dir.path().join("a.txt"), "a").expect("write");
        fs::write(dir.path().join("skip.bin"), [1, 2, 3]).expect("write");

        let out = FileIngestor::new()
            .ingest(ContentSource::Directory {
                path: dir.path().to_path_buf(),
                pattern: "*.txt".into(),
            })
            .await
            .expect("ingest");

        assert_eq!(
            out,
            vec![
                MultimodalContent::Text("a".into()),
                MultimodalContent::Text("b".into())
            ]
        );
    }

    #[tokio::test]
    async fn direct_content_uses_explicit_mime() {
        let out = FileIngestor::new()
            .ingest(ContentSource::DirectContent {
                bytes: b"hello".to_vec(),
                mime_type: "text/plain".into(),
            })
            .await
            .expect("ingest");

        assert_eq!(out, vec![MultimodalContent::Text("hello".into())]);
    }

    #[tokio::test]
    async fn rejects_unknown_mime() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("payload.unknown");
        fs::write(&path, [1, 2, 3]).expect("write");

        let err = FileIngestor::new()
            .ingest(ContentSource::LocalFile {
                path,
                mime_type: None,
            })
            .await
            .unwrap_err();

        assert!(matches!(err, IngestError::UnknownMime(_)));
    }

    #[tokio::test]
    async fn enforces_size_limit_before_reading() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("large.txt");
        let mut file = fs::File::create(&path).expect("create");
        file.write_all(b"too large").expect("write");

        let err = FileIngestor::with_max_bytes(3)
            .ingest(ContentSource::LocalFile {
                path,
                mime_type: None,
            })
            .await
            .unwrap_err();

        assert!(matches!(err, IngestError::LimitExceeded(_)));
    }
}
