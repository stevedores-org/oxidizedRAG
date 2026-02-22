//! Stage-level caching/memoization for pipeline stages.
//!
//! `CachedStage<I,O>` wraps any `Stage<I,O>` with content-hash caching
//! using sha2 for cache keys and moka for in-memory TTL-based caching.

use std::{hash::Hash, sync::Arc, time::Duration};

use async_trait::async_trait;
#[cfg(feature = "caching")]
use moka::future::Cache;
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};

use crate::{core::Result, pipeline::stage::Stage};

/// A cached wrapper around any `Stage<I,O>`.
///
/// Caches stage outputs based on a content hash of the input.
/// Requires `I: Serialize + Hash` and `O: Serialize + DeserializeOwned +
/// Clone`.
pub struct CachedStage<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    inner: Arc<dyn Stage<I, O>>,
    #[cfg(feature = "caching")]
    cache: Cache<String, Vec<u8>>,
    #[cfg(not(feature = "caching"))]
    _phantom: std::marker::PhantomData<(I, O)>,
    ttl: Duration,
}

impl<I, O> CachedStage<I, O>
where
    I: Serialize + Hash + Send + 'static,
    O: Serialize + DeserializeOwned + Clone + Send + 'static,
{
    /// Create a new cached stage wrapping the given inner stage.
    ///
    /// `max_capacity` controls maximum number of cached entries.
    /// `ttl` is the time-to-live for each cache entry.
    pub fn new(inner: Arc<dyn Stage<I, O>>, max_capacity: u64, ttl: Duration) -> Self {
        Self {
            inner,
            #[cfg(feature = "caching")]
            cache: Cache::builder()
                .max_capacity(max_capacity)
                .time_to_live(ttl)
                .build(),
            #[cfg(not(feature = "caching"))]
            _phantom: std::marker::PhantomData,
            ttl,
        }
    }

    /// Compute a cache key from the input using SHA-256.
    fn cache_key(input: &I) -> String
    where
        I: Serialize,
    {
        let serialized = serde_json::to_vec(input).unwrap_or_default();
        let hash = Sha256::digest(&serialized);
        hex::encode(hash)
    }

    /// Get the configured TTL.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }
}

// hex encoding helper (avoiding a dependency)
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes
            .as_ref()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

#[async_trait]
impl<I, O> Stage<I, O> for CachedStage<I, O>
where
    I: Serialize + Hash + Send + Sync + 'static,
    O: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    async fn process(&self, input: I) -> Result<O> {
        let key = Self::cache_key(&input);

        // Check cache
        #[cfg(feature = "caching")]
        {
            if let Some(cached_bytes) = self.cache.get(&key).await {
                if let Ok(output) = serde_json::from_slice::<O>(&cached_bytes) {
                    return Ok(output);
                }
            }
        }

        // Cache miss — run inner stage
        let output = self.inner.process(input).await?;

        // Store in cache
        #[cfg(feature = "caching")]
        {
            if let Ok(bytes) = serde_json::to_vec(&output) {
                self.cache.insert(key, bytes).await;
            }
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        // Delegate to inner stage name with a prefix would lose the &str lifetime,
        // so we just delegate directly.
        self.inner.name()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::core::Result;

    /// A counting stage that tracks how many times process() is called.
    struct CountingStage {
        call_count: AtomicU64,
    }

    impl CountingStage {
        fn new() -> Self {
            Self {
                call_count: AtomicU64::new(0),
            }
        }

        fn calls(&self) -> u64 {
            self.call_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl Stage<String, String> for CountingStage {
        async fn process(&self, input: String) -> Result<String> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            Ok(format!("processed:{}", input))
        }
        fn name(&self) -> &str {
            "counting"
        }
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let inner = Arc::new(CountingStage::new());
        let cached = CachedStage::new(inner.clone(), 100, Duration::from_secs(60));

        let result = cached.process("hello".to_string()).await.unwrap();
        assert_eq!(result, "processed:hello");
        assert_eq!(inner.calls(), 1);
    }

    #[cfg(feature = "caching")]
    #[tokio::test]
    async fn test_cache_hit() {
        let inner = Arc::new(CountingStage::new());
        let cached = CachedStage::new(inner.clone(), 100, Duration::from_secs(60));

        // First call — miss
        let r1 = cached.process("hello".to_string()).await.unwrap();
        assert_eq!(r1, "processed:hello");
        assert_eq!(inner.calls(), 1);

        // Second call — hit
        let r2 = cached.process("hello".to_string()).await.unwrap();
        assert_eq!(r2, "processed:hello");
        assert_eq!(inner.calls(), 1); // inner not called again
    }

    #[cfg(feature = "caching")]
    #[tokio::test]
    async fn test_cache_ttl_expiry() {
        let inner = Arc::new(CountingStage::new());
        let cached = CachedStage::new(
            inner.clone(),
            100,
            Duration::from_millis(50), // Very short TTL
        );

        cached.process("hello".to_string()).await.unwrap();
        assert_eq!(inner.calls(), 1);

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(100)).await;

        // moka eviction may require a get after expiry
        cached.process("hello".to_string()).await.unwrap();
        // After TTL, inner should be called again
        assert!(inner.calls() >= 2);
    }
}
