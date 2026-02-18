//! Integration tests for graphrag-aivcs using MemoryRunLedger.

use std::sync::Arc;

use graphrag_aivcs::{GraphRagSpec, RagAdapter, RagRunRecorder};
use oxidized_state::{fakes::MemoryRunLedger, RunLedger, RunStatus};

fn test_spec() -> GraphRagSpec {
    GraphRagSpec::new("test-rag-agent", r#"{"model":"gpt-4","top_k":10}"#)
}

// 1. recorder_start_creates_run
#[tokio::test]
async fn recorder_start_creates_run() {
    let ledger: Arc<dyn RunLedger> = Arc::new(MemoryRunLedger::new());
    let spec = test_spec();

    let recorder = RagRunRecorder::start(ledger.clone(), &spec)
        .await
        .expect("start");
    assert!(!recorder.run_id().to_string().is_empty());

    recorder.finish_ok(10).await.expect("finish_ok");
}

// 2. retrieval_event_is_persisted
#[tokio::test]
async fn retrieval_event_is_persisted() {
    let ledger: Arc<dyn RunLedger> = Arc::new(MemoryRunLedger::new());
    let spec = test_spec();

    let mut recorder = RagRunRecorder::start(ledger.clone(), &spec)
        .await
        .expect("start");
    let run_id = recorder.run_id().clone();

    recorder
        .record_retrieval(5, 0.85, 120)
        .await
        .expect("record_retrieval");
    recorder.finish_ok(200).await.expect("finish_ok");

    let events = ledger.get_events(&run_id).await.expect("get_events");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, "rag.retrieval_complete");
    assert_eq!(events[0].payload["retrieved_count"], 5);
}

// 3. llm_event_is_persisted
#[tokio::test]
async fn llm_event_is_persisted() {
    let ledger: Arc<dyn RunLedger> = Arc::new(MemoryRunLedger::new());
    let spec = test_spec();

    let mut recorder = RagRunRecorder::start(ledger.clone(), &spec)
        .await
        .expect("start");
    let run_id = recorder.run_id().clone();

    recorder
        .record_llm_call(512, 150, 800)
        .await
        .expect("record_llm_call");
    recorder.finish_ok(900).await.expect("finish_ok");

    let events = ledger.get_events(&run_id).await.expect("get_events");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, "rag.llm_called");
    assert_eq!(events[0].payload["tokens_used"], 150);
}

// 4. finish_ok_sets_completed_status
#[tokio::test]
async fn finish_ok_sets_completed_status() {
    let ledger: Arc<dyn RunLedger> = Arc::new(MemoryRunLedger::new());
    let spec = test_spec();

    let recorder = RagRunRecorder::start(ledger.clone(), &spec)
        .await
        .expect("start");
    let run_id = recorder.run_id().clone();

    recorder.finish_ok(50).await.expect("finish_ok");

    let record = ledger.get_run(&run_id).await.expect("get_run");
    assert_eq!(record.status, RunStatus::Completed);
}

// 5. finish_err_sets_failed_status
#[tokio::test]
async fn finish_err_sets_failed_status() {
    let ledger: Arc<dyn RunLedger> = Arc::new(MemoryRunLedger::new());
    let spec = test_spec();

    let recorder = RagRunRecorder::start(ledger.clone(), &spec)
        .await
        .expect("start");
    let run_id = recorder.run_id().clone();

    recorder
        .finish_err("retrieval timeout")
        .await
        .expect("finish_err");

    let record = ledger.get_run(&run_id).await.expect("get_run");
    assert_eq!(record.status, RunStatus::Failed);
}

// 6. spec_digest_is_deterministic
#[test]
fn spec_digest_is_deterministic() {
    let config = r#"{"model":"gpt-4","top_k":10}"#;
    let s1 = GraphRagSpec::new("agent", config);
    let s2 = GraphRagSpec::new("agent", config);

    assert_eq!(s1.config_digest, s2.config_digest);
    assert_eq!(s1.content_digest(), s2.content_digest());
}

// 7. spec_digest_changes_on_mutation
#[test]
fn spec_digest_changes_on_mutation() {
    let s1 = GraphRagSpec::new("agent", r#"{"model":"gpt-4","top_k":10}"#);
    let s2 = GraphRagSpec::new("agent", r#"{"model":"gpt-4","top_k":20}"#);

    assert_ne!(s1.config_digest, s2.config_digest);
}

// 8. adapter_in_memory_works_end_to_end
#[tokio::test]
async fn adapter_in_memory_works_end_to_end() {
    let adapter = RagAdapter::in_memory();
    let spec = test_spec();

    let mut recorder = adapter.start_run(&spec).await.expect("start_run");
    let run_id = recorder.run_id().clone();

    recorder
        .record_retrieval(10, 0.92, 50)
        .await
        .expect("retrieval");
    recorder
        .record_llm_call(256, 80, 300)
        .await
        .expect("llm_call");
    recorder.finish_ok(400).await.expect("finish_ok");

    let record = adapter.ledger().get_run(&run_id).await.expect("get_run");
    assert_eq!(record.status, RunStatus::Completed);

    let events = adapter
        .ledger()
        .get_events(&run_id)
        .await
        .expect("get_events");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].kind, "rag.retrieval_complete");
    assert_eq!(events[1].kind, "rag.llm_called");
}
