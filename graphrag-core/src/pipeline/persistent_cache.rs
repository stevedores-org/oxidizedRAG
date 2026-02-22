//! Persistent cache backend using RocksDB.
//!
//! Provides a disk-based cache that survives process restarts and enables
//! cache sharing across pipeline runs.

/// Trait for pluggable cache backends.
///
/// Enables multiple implementations (in-memory, RocksDB, Redis, etc.)
pub trait PersistentCacheBackend: Send + Sync {
    /// Retrieve a value from the cache by key.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, String>;

    /// Store a value in the cache.
    fn set(&self, key: String, value: Vec<u8>) -> Result<(), String>;

    /// Delete a value from the cache.
    fn delete(&self, key: &str) -> Result<(), String>;

    /// Check if a key exists in the cache.
    fn contains(&self, key: &str) -> Result<bool, String>;

    /// Clear all entries from the cache.
    fn clear(&self) -> Result<(), String>;

    /// Get the number of entries in the cache.
    fn len(&self) -> Result<usize, String>;

    /// Check if cache is empty.
    fn is_empty(&self) -> Result<bool, String> {
        self.len().map(|len| len == 0)
    }

    /// Get cache statistics.
    fn stats(&self) -> Result<CacheStats, String> {
        Ok(CacheStats::default())
    }

    /// Compact the cache to reclaim space.
    fn compact(&self) -> Result<(), String> {
        Ok(())
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Cache size in bytes
    pub size_bytes: u64,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of evictions
    pub evictions: u64,
}

impl CacheStats {
    /// Calculate hit rate as percentage (0-100).
    pub fn hit_rate_percent(&self) -> f64 {
        let total = (self.hits + self.misses) as f64;
        if total == 0.0 {
            0.0
        } else {
            (self.hits as f64 / total) * 100.0
        }
    }
}

#[cfg(feature = "persistent-cache")]
mod rocksdb_backend {
    use std::{
        path::{Path, PathBuf},
        sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        },
    };

    use super::{CacheStats, PersistentCacheBackend};

    /// RocksDB-backed persistent cache.
    pub struct RocksDBCache {
        db: Arc<rocksdb::DB>,
        path: PathBuf,
        stats: CacheStatistics,
    }

    /// Thread-safe statistics tracking.
    struct CacheStatistics {
        hits: AtomicU64,
        misses: AtomicU64,
        evictions: AtomicU64,
    }

    impl Default for CacheStatistics {
        fn default() -> Self {
            Self {
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                evictions: AtomicU64::new(0),
            }
        }
    }

    impl RocksDBCache {
        /// Create a new RocksDB cache at the given path.
        pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, String> {
            let path = path.as_ref().to_path_buf();

            // Create parent directory if it doesn't exist
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create cache directory: {}", e))?;
            }

            // Open or create RocksDB
            let mut opts = rocksdb::Options::default();
            opts.create_if_missing(true);
            opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

            let db = rocksdb::DB::open(&opts, &path)
                .map_err(|e| format!("Failed to open RocksDB: {}", e))?;

            Ok(Self {
                db: Arc::new(db),
                path,
                stats: CacheStatistics::default(),
            })
        }

        /// Get cache path.
        pub fn path(&self) -> &Path {
            &self.path
        }

        /// Get internal statistics.
        fn get_stats(&self) -> CacheStats {
            CacheStats {
                total_entries: 0, // RocksDB doesn't provide direct count
                size_bytes: 0,    // Would need to iterate
                hits: self.stats.hits.load(Ordering::Relaxed),
                misses: self.stats.misses.load(Ordering::Relaxed),
                evictions: self.stats.evictions.load(Ordering::Relaxed),
            }
        }
    }

    impl PersistentCacheBackend for RocksDBCache {
        fn get(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
            match self.db.get(key.as_bytes()) {
                Ok(Some(value)) => {
                    self.stats.hits.fetch_add(1, Ordering::Relaxed);
                    Ok(Some(value))
                },
                Ok(None) => {
                    self.stats.misses.fetch_add(1, Ordering::Relaxed);
                    Ok(None)
                },
                Err(e) => Err(format!("RocksDB get error: {}", e)),
            }
        }

        fn set(&self, key: String, value: Vec<u8>) -> Result<(), String> {
            self.db
                .put(key.as_bytes(), &value)
                .map_err(|e| format!("RocksDB set error: {}", e))
        }

        fn delete(&self, key: &str) -> Result<(), String> {
            self.db
                .delete(key.as_bytes())
                .map_err(|e| format!("RocksDB delete error: {}", e))
        }

        fn contains(&self, key: &str) -> Result<bool, String> {
            self.get(key).map(|opt| opt.is_some())
        }

        fn clear(&self) -> Result<(), String> {
            // RocksDB doesn't have a direct clear, so we'd need to delete the DB and
            // recreate it For now, just return an error indicating this isn't
            // supported
            Err("Clear not supported - use delete key by key or recreate cache".to_string())
        }

        fn len(&self) -> Result<usize, String> {
            // Counting all keys is expensive, return estimate or error
            Err("len() not efficiently supported - consider using stats instead".to_string())
        }

        fn stats(&self) -> Result<CacheStats, String> {
            Ok(self.get_stats())
        }

        fn compact(&self) -> Result<(), String> {
            self.db.compact_range(None::<&[u8]>, None::<&[u8]>);
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use tempfile::TempDir;

        use super::*;

        #[test]
        fn test_rocksdb_cache_new() {
            let dir = TempDir::new().unwrap();
            let cache = RocksDBCache::new(dir.path()).unwrap();
            assert!(cache.path().exists());
        }

        #[test]
        fn test_rocksdb_cache_set_get() {
            let dir = TempDir::new().unwrap();
            let cache = RocksDBCache::new(dir.path()).unwrap();

            let key = "test-key".to_string();
            let value = vec![1, 2, 3, 4, 5];

            cache.set(key.clone(), value.clone()).unwrap();
            let retrieved = cache.get(&key).unwrap();
            assert_eq!(retrieved, Some(value));
        }

        #[test]
        fn test_rocksdb_cache_delete() {
            let dir = TempDir::new().unwrap();
            let cache = RocksDBCache::new(dir.path()).unwrap();

            let key = "delete-key".to_string();
            let value = vec![1, 2, 3];

            cache.set(key.clone(), value).unwrap();
            assert!(cache.contains(&key).unwrap());

            cache.delete(&key).unwrap();
            assert!(!cache.contains(&key).unwrap());
        }

        #[test]
        fn test_rocksdb_cache_stats() {
            let dir = TempDir::new().unwrap();
            let cache = RocksDBCache::new(dir.path()).unwrap();

            // First access is a miss
            let _ = cache.get("nonexistent");

            // Second access is a hit
            cache.set("key".to_string(), vec![1, 2, 3]).unwrap();
            let _ = cache.get("key");

            let stats = cache.stats().unwrap();
            assert_eq!(stats.hits, 1);
            assert_eq!(stats.misses, 1);
        }

        #[test]
        fn test_rocksdb_cache_compression() {
            let dir = TempDir::new().unwrap();
            let cache = RocksDBCache::new(dir.path()).unwrap();

            // Store a large value (should be compressed)
            let large_value = vec![42u8; 10_000];
            cache.set("large".to_string(), large_value.clone()).unwrap();

            let retrieved = cache.get("large").unwrap();
            assert_eq!(retrieved, Some(large_value));
        }

        #[test]
        fn test_rocksdb_cache_compact() {
            let dir = TempDir::new().unwrap();
            let cache = RocksDBCache::new(dir.path()).unwrap();

            // Add and delete some data
            for i in 0..10 {
                cache.set(format!("key-{}", i), vec![i as u8; 100]).unwrap();
            }

            for i in 0..5 {
                cache.delete(&format!("key-{}", i)).unwrap();
            }

            // Compact should succeed
            cache.compact().unwrap();
        }
    }
}

#[cfg(feature = "persistent-cache")]
pub use rocksdb_backend::RocksDBCache;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            hits: 80,
            misses: 20,
            ..Default::default()
        };
        assert_eq!(stats.hit_rate_percent(), 80.0);
    }

    #[test]
    fn test_cache_stats_hit_rate_zero() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate_percent(), 0.0);
    }

    #[test]
    fn test_cache_stats_hit_rate_perfect() {
        let stats = CacheStats {
            hits: 100,
            misses: 0,
            ..Default::default()
        };
        assert_eq!(stats.hit_rate_percent(), 100.0);
    }
}
