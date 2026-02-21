//! Comprehensive integration tests for stage-level caching.

use async_trait::async_trait;
use graphrag_core::pipeline::{
    Stage, StageMeta, StageError, CachedStage, StageCache, ContentHashable, ChunkBatch,
    DocumentChunk, EmbeddingBatch, EmbeddingRecord,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Mock stage that counts executions
struct CountingChunkStage {
    execution_count: Arc<AtomicUsize>,
}

#[async_trait]
impl Stage<ChunkBatch, ChunkBatch> for CountingChunkStage {
    async fn execute(&self, input: ChunkBatch) -> Result<ChunkBatch, StageError> {
        self.execution_count.fetch_add(1, Ordering::SeqCst);
        Ok(input)
    }

    fn name(&self) -> &str {
        "counting-chunker"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn metadata(&self) -> StageMeta {
        StageMeta {
            deterministic: true,
            description: Some("Mock chunker for testing".to_string()),
            ..Default::default()
        }
    }
}

/// Mock stage for embedding
struct CountingEmbeddingStage {
    execution_count: Arc<AtomicUsize>,
}

#[async_trait]
impl Stage<ChunkBatch, EmbeddingBatch> for CountingEmbeddingStage {
    async fn execute(&self, input: ChunkBatch) -> Result<EmbeddingBatch, StageError> {
        self.execution_count.fetch_add(1, Ordering::SeqCst);

        let embeddings: Vec<EmbeddingRecord> = input
            .chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| EmbeddingRecord {
                chunk_id: chunk.id.clone(),
                vector: vec![0.1 * (i as f32), 0.2 * (i as f32), 0.3 * (i as f32)],
                metadata: Default::default(),
            })
            .collect();

        Ok(EmbeddingBatch {
            id: format!("{}-embeddings", input.id),
            embeddings,
            config_hash: "test-config".to_string(),
        })
    }

    fn name(&self) -> &str {
        "counting-embedder"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn metadata(&self) -> StageMeta {
        StageMeta {
            deterministic: true,
            description: Some("Mock embedder for testing".to_string()),
            ..Default::default()
        }
    }
}

#[tokio::test]
async fn test_cache_hit_reduces_executions() {
    let count = Arc::new(AtomicUsize::new(0));
    let stage = Arc::new(CountingChunkStage {
        execution_count: count.clone(),
    });
    let cache = Arc::new(StageCache::new());
    let cached = CachedStage::new(stage, cache);

    let batch = ChunkBatch {
        id: "batch-1".to_string(),
        chunks: vec![DocumentChunk {
            id: "chunk-1".to_string(),
            content: "test content".to_string(),
            source: "test.rs".to_string(),
            line_range: Some((1, 1)),
            metadata: Default::default(),
        }],
        corpus_hash: "hash1".to_string(),
    };

    // First execution
    let result1 = cached.execute(batch.clone()).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1, "Should execute once");

    // Second execution with same input (cache hit)
    let result2 = cached.execute(batch).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1, "Should not execute again (cache hit)");

    assert_eq!(result1, result2);
}

#[tokio::test]
async fn test_cache_miss_on_different_input() {
    let count = Arc::new(AtomicUsize::new(0));
    let stage = Arc::new(CountingChunkStage {
        execution_count: count.clone(),
    });
    let cache = Arc::new(StageCache::new());
    let cached = CachedStage::new(stage, cache);

    let batch1 = ChunkBatch {
        id: "batch-1".to_string(),
        chunks: vec![DocumentChunk {
            id: "chunk-1".to_string(),
            content: "content 1".to_string(),
            source: "test.rs".to_string(),
            line_range: None,
            metadata: Default::default(),
        }],
        corpus_hash: "hash1".to_string(),
    };

    let batch2 = ChunkBatch {
        id: "batch-2".to_string(),
        chunks: vec![DocumentChunk {
            id: "chunk-2".to_string(),
            content: "content 2".to_string(),
            source: "test.rs".to_string(),
            line_range: None,
            metadata: Default::default(),
        }],
        corpus_hash: "hash2".to_string(),
    };

    // First execution
    let _result1 = cached.execute(batch1).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);

    // Second execution with different input (cache miss)
    let _result2 = cached.execute(batch2).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 2, "Should execute for different input");
}

#[tokio::test]
async fn test_cache_hit_with_embedding_batch() {
    let count = Arc::new(AtomicUsize::new(0));
    let stage = Arc::new(CountingEmbeddingStage {
        execution_count: count.clone(),
    });
    let cache = Arc::new(StageCache::new());
    let cached = CachedStage::new(stage, cache);

    let batch = ChunkBatch {
        id: "batch-1".to_string(),
        chunks: vec![
            DocumentChunk {
                id: "chunk-1".to_string(),
                content: "content 1".to_string(),
                source: "test.rs".to_string(),
                line_range: None,
                metadata: Default::default(),
            },
            DocumentChunk {
                id: "chunk-2".to_string(),
                content: "content 2".to_string(),
                source: "test.rs".to_string(),
                line_range: None,
                metadata: Default::default(),
            },
        ],
        corpus_hash: "hash1".to_string(),
    };

    // First execution
    let _result1 = cached.execute(batch.clone()).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);

    // Second execution (cache hit)
    let _result2 = cached.execute(batch).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1, "Should use cache");
}

#[tokio::test]
async fn test_shared_cache_across_stages() {
    let count1 = Arc::new(AtomicUsize::new(0));
    let count2 = Arc::new(AtomicUsize::new(0));

    let stage1 = Arc::new(CountingChunkStage {
        execution_count: count1.clone(),
    });
    let stage2 = Arc::new(CountingEmbeddingStage {
        execution_count: count2.clone(),
    });

    let shared_cache = Arc::new(StageCache::new());
    let cached1 = CachedStage::new(stage1, shared_cache.clone());
    let cached2 = CachedStage::new(stage2, shared_cache.clone());

    let batch = ChunkBatch {
        id: "batch-1".to_string(),
        chunks: vec![DocumentChunk {
            id: "chunk-1".to_string(),
            content: "test".to_string(),
            source: "test.rs".to_string(),
            line_range: None,
            metadata: Default::default(),
        }],
        corpus_hash: "hash1".to_string(),
    };

    // Execute stage 1 twice
    let _result1 = cached1.execute(batch.clone()).await.unwrap();
    let _result2 = cached1.execute(batch).await.unwrap();
    assert_eq!(count1.load(Ordering::SeqCst), 1, "Stage 1 cached");

    // Execute stage 2 once (different stage, different cache key)
    let batch2 = ChunkBatch {
        id: "batch-2".to_string(),
        chunks: vec![DocumentChunk {
            id: "chunk-1".to_string(),
            content: "test".to_string(),
            source: "test.rs".to_string(),
            line_range: None,
            metadata: Default::default(),
        }],
        corpus_hash: "hash1".to_string(),
    };

    let _result3 = cached2.execute(batch2).await.unwrap();
    assert_eq!(count2.load(Ordering::SeqCst), 1);

    // Shared cache should have 2 entries
    assert_eq!(shared_cache.len(), 2, "Shared cache should have entries from both stages");
}

#[tokio::test]
async fn test_cache_clear() {
    let count = Arc::new(AtomicUsize::new(0));
    let stage = Arc::new(CountingChunkStage {
        execution_count: count.clone(),
    });
    let cache = Arc::new(StageCache::new());
    let cached = CachedStage::new(stage, cache.clone());

    let batch = ChunkBatch {
        id: "batch-1".to_string(),
        chunks: vec![DocumentChunk {
            id: "chunk-1".to_string(),
            content: "test".to_string(),
            source: "test.rs".to_string(),
            line_range: None,
            metadata: Default::default(),
        }],
        corpus_hash: "hash1".to_string(),
    };

    // Execute and cache
    let _result1 = cached.execute(batch.clone()).await.unwrap();
    assert_eq!(cache.len(), 1);

    // Clear cache
    cache.clear();
    assert!(cache.is_empty());

    // Execute again (should not use cache)
    let _result2 = cached.execute(batch).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 2, "Should execute again after cache clear");
}

#[test]
fn test_content_hash_stability() {
    let batch = ChunkBatch {
        id: "batch-1".to_string(),
        chunks: vec![
            DocumentChunk {
                id: "chunk-1".to_string(),
                content: "test".to_string(),
                source: "test.rs".to_string(),
                line_range: None,
                metadata: Default::default(),
            },
            DocumentChunk {
                id: "chunk-2".to_string(),
                content: "test2".to_string(),
                source: "lib.rs".to_string(),
                line_range: None,
                metadata: Default::default(),
            },
        ],
        corpus_hash: "hash1".to_string(),
    };

    let hash1 = batch.content_hash();
    let hash2 = batch.content_hash();

    assert_eq!(hash1, hash2, "Hash should be deterministic");
    assert_eq!(hash1.len(), 64, "SHA256 hex should be 64 chars");
}

#[test]
fn test_content_hash_order_independent() {
    let batch1 = ChunkBatch {
        id: "batch-1".to_string(),
        chunks: vec![
            DocumentChunk {
                id: "a".to_string(),
                content: "content".to_string(),
                source: "test.rs".to_string(),
                line_range: None,
                metadata: Default::default(),
            },
            DocumentChunk {
                id: "b".to_string(),
                content: "more".to_string(),
                source: "lib.rs".to_string(),
                line_range: None,
                metadata: Default::default(),
            },
        ],
        corpus_hash: "hash1".to_string(),
    };

    let mut batch2_chunks = batch1.chunks.clone();
    batch2_chunks.reverse();

    let batch2 = ChunkBatch {
        id: "batch-1".to_string(),
        chunks: batch2_chunks,
        corpus_hash: "hash1".to_string(),
    };

    // Even though chunks are in different order, content hash should be the same
    // because IDs are sorted before hashing
    assert_eq!(
        batch1.content_hash(),
        batch2.content_hash(),
        "Different order should produce same hash"
    );
}
