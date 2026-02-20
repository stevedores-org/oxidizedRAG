//! Content hashing trait for deterministic cache key generation.
//!
//! Enables types to produce stable, deterministic SHA256 hashes of their content
//! for use in content-addressed caching.

/// Trait for types that can produce deterministic content hashes for caching.
///
/// This trait is used to generate stable, reproducible cache keys based on content.
/// Implementations MUST ensure:
/// - Same content always produces the same hash
/// - Order-independent hashing (e.g., sorted collections)
/// - Version/schema changes don't invalidate unrelated hashes
pub trait ContentHashable {
    /// Generate a deterministic SHA256 hash of this content.
    ///
    /// Returns a hex-encoded SHA256 hash string.
    fn content_hash(&self) -> String;
}
