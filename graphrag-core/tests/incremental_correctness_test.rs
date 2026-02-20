//! Correctness tests for incremental graph updates.
//!
//! Validates that incrementally-built graphs produce identical state
//! to graphs built from scratch, and that delete/rollback operations
//! maintain consistency.

use graphrag_core::core::{Entity, EntityId, KnowledgeGraph, Relationship};
use graphrag_core::graph::incremental::{
    ChangeData, ChangeRecord, ChangeType, ConflictResolver, ConflictStrategy, DeltaStatus,
    GraphDelta, IncrementalConfig, IncrementalGraphStore, Operation, ProductionGraphStore,
    UpdateId,
};

fn create_test_entity(i: usize) -> Entity {
    Entity {
        id: EntityId::new(format!("entity-{i}")),
        name: format!("Entity {i}"),
        entity_type: "test".to_string(),
        confidence: 0.9,
        mentions: vec![],
        embedding: Some(vec![i as f32 * 0.1; 8]),
    }
}

fn create_test_relationship(src: usize, tgt: usize) -> Relationship {
    Relationship {
        source: EntityId::new(format!("entity-{src}")),
        target: EntityId::new(format!("entity-{tgt}")),
        relation_type: "related_to".to_string(),
        confidence: 0.8,
        context: vec![],
    }
}

/// Build a graph synchronously (full rebuild).
fn build_full_graph(entity_count: usize, edge_count: usize) -> KnowledgeGraph {
    let mut graph = KnowledgeGraph::new();

    for i in 0..entity_count {
        graph.add_entity(create_test_entity(i)).unwrap();
    }

    for i in 0..edge_count.min(entity_count.saturating_sub(1)) {
        graph
            .add_relationship(create_test_relationship(i, i + 1))
            .unwrap();
    }

    graph
}

#[tokio::test]
async fn test_incremental_vs_full_rebuild_parity() {
    let entity_count = 200;
    let edge_count = 100;

    // Build via full rebuild
    let full_graph = build_full_graph(entity_count, edge_count);

    // Build incrementally via ProductionGraphStore
    let config = IncrementalConfig::default();
    let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
    let graph = KnowledgeGraph::new();
    let mut store = ProductionGraphStore::new(graph, config, resolver);

    for i in 0..entity_count {
        store.upsert_entity(create_test_entity(i)).await.unwrap();
    }

    for i in 0..edge_count.min(entity_count.saturating_sub(1)) {
        store
            .upsert_relationship(create_test_relationship(i, i + 1))
            .await
            .unwrap();
    }

    // Validate parity
    let stats = store.get_graph_statistics().await.unwrap();
    let full_entities: Vec<_> = full_graph.entities().collect();
    let full_rels = full_graph.get_all_relationships();

    assert_eq!(
        stats.node_count,
        full_entities.len(),
        "Entity count mismatch"
    );
    assert_eq!(
        stats.edge_count,
        full_rels.len(),
        "Relationship count mismatch"
    );
}

#[tokio::test]
async fn test_delete_entity_removes_entity_and_relationships() {
    let config = IncrementalConfig::default();
    let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
    let graph = KnowledgeGraph::new();
    let mut store = ProductionGraphStore::new(graph, config, resolver);

    // Add 3 entities in a chain: 0 -> 1 -> 2
    for i in 0..3 {
        store.upsert_entity(create_test_entity(i)).await.unwrap();
    }
    store
        .upsert_relationship(create_test_relationship(0, 1))
        .await
        .unwrap();
    store
        .upsert_relationship(create_test_relationship(1, 2))
        .await
        .unwrap();

    let stats_before = store.get_graph_statistics().await.unwrap();
    assert_eq!(stats_before.node_count, 3);
    assert_eq!(stats_before.edge_count, 2);

    // Delete middle entity
    let entity_id = EntityId::new("entity-1".into());
    store.delete_entity(&entity_id).await.unwrap();

    let stats_after = store.get_graph_statistics().await.unwrap();
    assert_eq!(stats_after.node_count, 2, "Entity should be removed");
    // petgraph removes all edges when a node is removed
    assert_eq!(
        stats_after.edge_count, 0,
        "Relationships involving deleted entity should be gone"
    );
}

#[tokio::test]
async fn test_delete_relationship_leaves_entities() {
    let config = IncrementalConfig::default();
    let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
    let graph = KnowledgeGraph::new();
    let mut store = ProductionGraphStore::new(graph, config, resolver);

    for i in 0..2 {
        store.upsert_entity(create_test_entity(i)).await.unwrap();
    }
    store
        .upsert_relationship(create_test_relationship(0, 1))
        .await
        .unwrap();

    let src = EntityId::new("entity-0".into());
    let tgt = EntityId::new("entity-1".into());
    store
        .delete_relationship(&src, &tgt, "related_to")
        .await
        .unwrap();

    let stats = store.get_graph_statistics().await.unwrap();
    assert_eq!(stats.node_count, 2, "Entities should remain");
    assert_eq!(stats.edge_count, 0, "Relationship should be removed");
}

#[tokio::test]
async fn test_rollback_delta_restores_state() {
    let config = IncrementalConfig::default();
    let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
    let graph = KnowledgeGraph::new();
    let mut store = ProductionGraphStore::new(graph, config, resolver);

    // Add initial entities
    for i in 0..5 {
        store.upsert_entity(create_test_entity(i)).await.unwrap();
    }

    let stats_before = store.get_graph_statistics().await.unwrap();
    assert_eq!(stats_before.node_count, 5);

    // Apply a delta that adds more entities
    let delta = GraphDelta {
        delta_id: UpdateId::new(),
        timestamp: chrono::Utc::now(),
        changes: (5..10)
            .map(|i| ChangeRecord {
                change_id: UpdateId::new(),
                timestamp: chrono::Utc::now(),
                change_type: ChangeType::EntityAdded,
                entity_id: Some(EntityId::new(format!("entity-{i}"))),
                document_id: None,
                operation: Operation::Upsert,
                data: ChangeData::Entity(create_test_entity(i)),
                metadata: std::collections::HashMap::new(),
            })
            .collect(),
        dependencies: vec![],
        status: DeltaStatus::Pending,
        rollback_data: None,
    };

    let delta_id = store.apply_delta(delta).await.unwrap();

    let stats_after_apply = store.get_graph_statistics().await.unwrap();
    assert_eq!(stats_after_apply.node_count, 10);

    // Rollback the delta
    store.rollback_delta(&delta_id).await.unwrap();

    let stats_after_rollback = store.get_graph_statistics().await.unwrap();
    assert_eq!(
        stats_after_rollback.node_count, 5,
        "Should be back to original 5 entities after rollback"
    );
}

#[tokio::test]
async fn test_consistency_report_detects_issues() {
    let config = IncrementalConfig::default();
    let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
    let graph = KnowledgeGraph::new();
    let mut store = ProductionGraphStore::new(graph, config, resolver);

    // Add isolated entity (orphan — no relationships)
    store.upsert_entity(create_test_entity(0)).await.unwrap();

    // Add entity without embedding
    let mut entity_no_embed = create_test_entity(1);
    entity_no_embed.embedding = None;
    store.upsert_entity(entity_no_embed).await.unwrap();

    let report = store.validate_consistency().await.unwrap();

    assert!(report.orphaned_entities.len() >= 2, "Should detect orphans");
    assert!(
        report.missing_embeddings.len() >= 1,
        "Should detect missing embedding"
    );
}

#[tokio::test]
async fn test_change_log_tracks_operations() {
    let config = IncrementalConfig::default();
    let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
    let graph = KnowledgeGraph::new();
    let mut store = ProductionGraphStore::new(graph, config, resolver);

    let before = chrono::Utc::now();

    for i in 0..3 {
        store.upsert_entity(create_test_entity(i)).await.unwrap();
    }

    let log = store.get_change_log(Some(before)).await.unwrap();
    assert_eq!(log.len(), 3, "Should have 3 change log entries");
}

#[tokio::test]
async fn test_batch_upsert_entities() {
    let config = IncrementalConfig::default();
    let resolver = ConflictResolver::new(ConflictStrategy::KeepNew);
    let graph = KnowledgeGraph::new();
    let mut store = ProductionGraphStore::new(graph, config, resolver);

    let entities: Vec<Entity> = (0..50).map(|i| create_test_entity(i)).collect();
    let ids = store
        .batch_upsert_entities(entities, ConflictStrategy::KeepNew)
        .await
        .unwrap();

    assert_eq!(ids.len(), 50);

    let stats = store.get_graph_statistics().await.unwrap();
    assert_eq!(stats.node_count, 50);
}
