# Technical Design Document: AI Inference KV & Rust-Native RAG Vector Storage

| Field | Value |
|-------|-------|
| **Status** | Draft — for research group review |
| **Authors** | Stevedores engineering (synthesized from architecture review, June 2026) |
| **Audience** | Research group, platform, oxidizedRAG / oxidizedgraph maintainers |
| **Repositories** | [oxidizedRAG](https://github.com/stevedores-org/oxidizedRAG), [oxidizedgraph](https://github.com/stevedores-org/oxidizedgraph) |
| **Related** | oxidizedgraph [#18](https://github.com/stevedores-org/oxidizedgraph/issues/18) (agent orchestration roadmap), oxidizedRAG multimodal foundation (PR #185, merged to `develop`) |

---

## 1. Executive summary

This document records findings from a storage-layer review for **AI inference** and **retrieval-augmented generation (RAG)** in the Stevedores Rust stack. It separates two concerns that are often conflated:

1. **Key–value stores** — low-latency application cache and coordination (prompt cache, sessions, rate limits, tool outputs).
2. **Vector stores + embedders** — semantic retrieval for RAG (chunk embeddings, similarity search, metadata filters).

**Recommendations:**

| Layer | Primary recommendation | Alternative |
|-------|------------------------|-------------|
| Inference hot cache (KV) | **Valkey** (Redis-compatible, BSD-3) | **Dragonfly** at extreme QPS/RAM pressure |
| RAG embeddings (Rust) | **Remote APIs today**; add **fastembed-rs** for local ONNX onboarding | **candle** when GPU/pure-Rust inference is implemented |
| RAG vector index (prod) | **Qdrant** via `qdrant-client` | **LanceDB** after placeholder server integration is completed |
| RAG vector index (dev/edge) | **instant-distance** / **usearch** (in-process HNSW) | **voy** for WASM paths |

These choices align with **existing oxidizedRAG features** (`vector-hnsw`, `neural-embeddings`, server `qdrant` / `lancedb` options) and **oxidizedgraph Phase 1** (traced runs, tool policy, quality gates) without mandating immediate rewrites. The alignment is not uniform: `qdrant` and text HTTP embedding providers are usable now; local Candle inference, LanceDB storage, and hybrid sparse+dense retrieval still need implementation work.

---

## 2. Problem statement

### 2.1 Inference workloads

Agent and LLM serving systems need:

- Sub-millisecond reads for repeated prompts and completions
- TTL-based eviction (stale completions)
- Shared state across horizontally scaled inference replicas
- Optional pub/sub, counters, and streams for rate limiting and job coordination

A general-purpose **KV store** satisfies this. It does **not** store transformer attention KV tensors (that remains inside the inference engine, e.g. vLLM, llama.cpp).

### 2.2 RAG workloads

GraphRAG and multimodal pipelines need:

- Deterministic **embedding** of text (and eventually image/audio) into dense vectors
- **Approximate nearest neighbor (ANN)** search at scale
- **Payload filters** (tenant, document ID, ACL, modality, graph node ID)
- Optional **hybrid** retrieval (BM25 + dense) for SKU-like exact matches
- Rust-native paths for CLI, WASM, and server binaries without Python runtime dependency

### 2.3 Constraints (Stevedores)

| Constraint | Implication |
|------------|-------------|
| Rust-first | Prefer crates with mature Rust clients or in-process libraries |
| Commercial OSS | BSD/Apache/MIT + managed cloud options (ElastiCache Valkey, Qdrant Cloud, etc.) |
| oxidizedRAG today | `instant-distance`, HTTP embedding providers, optional `candle` scaffolding, `qdrant-client`, LanceDB server placeholder |
| oxidizedgraph today | Orchestration + guardrails on `develop`; no dedicated vector tier yet |
| License hygiene | Avoid SSPL-only dependencies for greenfield defaults (favor Valkey over Redis 7.4+ for new caches) |

---

## 3. Key–value storage for AI inference

### 3.1 Evaluation criteria

| Criterion | Weight |
|-----------|--------|
| Latency (p99 read) | High |
| Memory efficiency | High |
| Redis API compatibility | Medium (ecosystem) |
| OSS license clarity | High |
| Managed commercial offering | Medium |
| HA / clustering maturity | Medium |

### 3.2 Candidates

| System | License | Role | Notes |
|--------|---------|------|-------|
| **Valkey** | BSD-3 | **Recommended default** | Linux Foundation / AWS backing; Redis-compatible |
| **Dragonfly** | Apache-2.0 | High-throughput cache | Multithreaded; fewer nodes per GB |
| **Redis** | RSAL/SSPL (newer) | Incumbent | Best vendor support; license review required |
| **Memcached** | BSD | Simple blob cache | No rich structures; no hybrid features |
| **etcd** | Apache-2.0 | Coordination only | Not for prompt/completion cache |

### 3.3 Recommended topology

```text
                    ┌─────────────────┐
  Clients / Agents  │  Inference API  │
                    │  (Axum / graph) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────────┐
        │  Valkey  │  │ Vector   │  │ Object store │
        │  cache   │  │ tier     │  │ (artifacts)  │
        └──────────┘  └──────────┘  └──────────────┘
```

**Cache key patterns:**

| Key pattern | Value | TTL |
|-------------|-------|-----|
| `cache:prompt:{sha256(model+prompt+params)}` | completion blob | 1h–24h |
| `session:{thread_id}` | serialized agent state | session lifetime |
| `embed:{model}:{text_hash}` | float vector (binary/json) | 7d |
| `ratelimit:{tenant}:{window}` | counter | window size |

### 3.4 Integration with oxidizedgraph

oxidizedgraph already provides **in-graph state** (`AgentState`) and **checkpointing**. Valkey sits **beside** the graph:

- **Hot path:** cache LLM/tool results before writing to checkpoint DB
- **Cross-replica:** share `run_id` / `thread_id` metadata (complements `RunContext` from Phase 1)
- **Not in scope:** replacing `TransitionLog` persistence (remain crate-native or SurrealDB optional feature)

**Phase proposal (oxidizedgraph):**

| Phase | Deliverable |
|-------|-------------|
| P0 | Document Valkey key schema in `docs/` |
| P1 | Optional `CacheSink` trait + Valkey backend behind feature flag |
| P2 | Prompt/completion cache node in agent graphs |

---

## 4. RAG: embeddings and vector storage (Rust-native)

### 4.1 Embedding layer

| Backend | Crate | When to use |
|---------|-------|-------------|
| **Remote** | `reqwest` + provider API | Production-ready today; OpenAI, Voyage, Cohere, Jina, Mistral, Together |
| **fastembed-rs** | `fastembed` | Proposed local default for server/CLI; ONNX models; fast onboarding |
| **candle** | `candle-core`, `candle-transformers` | Pure Rust; Metal/CUDA; currently scaffolded, local inference follow-up required |
| **ort** | `ort` | ONNX Runtime; maximum model zoo |

**Trait boundary (current and recommended):**

Text-only paths already use `Embedder` / `AsyncEmbedder` in `graphrag-core`. PR #185 added multimodal seams under `graphrag-core/src/multimodal/`, including `EmbeddingEngine`, `Embedding`, `ModelInfo`, and `MultimodalContent`. New work should bridge or implement these existing traits instead of introducing a third parallel `EmbeddingModel` API.

```rust
#[async_trait]
pub trait EmbeddingEngine: Send + Sync {
    async fn embed(&self, content: MultimodalContent) -> Result<Embedding, EmbeddingError>;
    async fn embed_batch(&self, batch: Vec<MultimodalContent>) -> Result<Vec<Embedding>, EmbeddingError>;
    fn model_info(&self) -> ModelInfo;
}
```

`MultimodalContent` currently covers `Text`, `Image`, `Audio`, `Video`, and `Pdf`. Per-backend adapters should declare modality support through `ModelInfo::supported_types`.

### 4.2 Vector index layer

| Tier | Technology | oxidizedRAG feature | Scale |
|------|------------|---------------------|-------|
| **L0 — in-process** | `instant-distance` HNSW | `vector-hnsw` | &lt; ~1M vectors / process |
| **L0 — WASM** | `voy` | `graphrag-wasm` | Browser/edge bundles |
| **L1 — embedded columnar** | LanceDB | `lancedb` server feature exists; methods are placeholder | Large static corpora on S3/disk after implementation |
| **L2 — vector database** | Qdrant | `graphrag-server` default | Production dense RAG + filters; hybrid templates are follow-up work |

**Trait boundary (current and recommended):**

Text vector search already has `VectorStore` / `AsyncVectorStore` in `graphrag-core/src/core/traits.rs`. PR #185 added a multimodal `VectorStore` under `graphrag-core/src/multimodal/store.rs` with `store`, `store_batch`, `search`, `get`, `delete`, and `count`. New Qdrant/Lance/HNSW backends should implement those existing surfaces or provide explicit adapters between them.

```rust
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn store(&self, embedding: Embedding) -> Result<(), VectorStoreError>;
    async fn store_batch(&self, embeddings: Vec<Embedding>) -> Result<(), VectorStoreError>;
    async fn search(&self, query: &[f32], k: usize, filters: SearchFilters) -> Result<Vec<SearchResult>, VectorStoreError>;
}
```

Implementations: `InMemoryHnswStore`, `QdrantStore`, `LanceStore`.

### 4.3 Hybrid retrieval

Pure ANN misses exact tokens (IDs, SKUs, names). Recommended pattern:

```text
Query ─┬─► Tantivy BM25 (Rust) ────┐
       │                           ├─► Fusion / rerank ─► context
       └─► Qdrant dense ANN ────────┘
```

Qdrant supports sparse + dense in one system, but the current server store only creates dense cosine collections. Tantivy remains valuable for full-text inside Rust-only deployments. The first implementation should make the fusion contract explicit before choosing whether BM25 lives in Tantivy or Qdrant sparse vectors.

### 4.4 Alignment with oxidizedRAG codebase

Current `graphrag-core` / server layout:

| Component | Today | TDD target |
|-----------|-------|------------|
| ANN | `instant-distance` | Keep for tests, CLI, WASM |
| Neural embed | HTTP providers ready; `candle` scaffolding fails loud until implemented | Add `fastembed` feature flag as default local server embedder |
| Server vectors | `qdrant` default feature | Remain production dense-vector default; add payload indexes and hybrid templates |
| LanceDB | server feature placeholder; core Lance path disabled | Implement table/create/add/search/delete and decide whether core needs a Lance feature |
| Multimodal | PR #185 trait seams merged | Implement per-backend `EmbeddingEngine` and multimodal `VectorStore` adapters |

---

## 5. Reference architecture (combined)

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         oxidizedRAG / Agent graph                         │
├──────────────────────────────────────────────────────────────────────────┤
│  Ingest: chunk → enrich → embed (API today; fastembed|candle follow-up)  │
│          → upsert vectors (Qdrant|Lance|HNSW) + graph edges (core)      │
│  Query:  embed query → ANN + filter → (optional BM25) → rerank → LLM   │
├──────────────────────────────────────────────────────────────────────────┤
│  Sidecar caches (Valkey): prompt cache | session | embed cache            │
└──────────────────────────────────────────────────────────────────────────┘
```

**Data planes:**

| Plane | Technology | Durability |
|-------|------------|------------|
| Graph / entities | oxidizedRAG core (SQLite / workspace) | Durable |
| Vectors | Qdrant today; LanceDB after implementation | Durable |
| Hot cache | Valkey | Ephemeral (TTL) |
| Run audit | oxidizedgraph `TransitionLog` / AIVCS | Durable |

---

## 6. Decision log (ADRs)

| ID | Decision | Rationale | Status |
|----|----------|-----------|--------|
| ADR-KV-01 | Default inference cache: **Valkey** | BSD-3, Redis-compatible, broad managed offerings | Proposed |
| ADR-KV-02 | Reject Redis as default for *new* greenfield caches | License uncertainty on SSPL/RSAL lines | Proposed |
| ADR-RAG-01 | Production vectors: **Qdrant** | Payload filters, hybrid, mature `qdrant-client` | Proposed |
| ADR-RAG-02 | Dev/edge vectors: **instant-distance** | Already shipped; zero ops | Accepted (in tree) |
| ADR-RAG-03 | Default embedder for new deployments: **remote API today**, **fastembed** local follow-up | Avoids relying on unfinished Candle local inference | Proposed |
| ADR-RAG-04 | Retain **candle** for pure-Rust / Metal / CUDA paths | Strategic, but implementation remains pending | Proposed |
| ADR-RAG-05 | LanceDB for re-embed / data-lake batches | Columnar + object storage; current server code is placeholder | Proposed |

---

## 7. Non-goals

- Replacing transformer **KV cache** inside LLM runtimes (vLLM, etc.)
- Using Valkey/Redis as the **primary** vector database
- Mandating a single embedder for all modalities in v1
- PostgreSQL-only stack (valid for some teams; not the default here)

---

## 8. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Embed model drift changes retrieval quality | Version `model_id` in vector payload; re-embed job |
| In-process HNSW OOM at scale | Hard cap + spill to Qdrant |
| Valkey cache stampede | Jittered TTL; single-flight lock key |
| LanceDB placeholder mistaken for production backend | Dedicated implementation PR with integration tests, not only dependency resolution |
| Multimodal embed cost | Route heavy modalities to API; cache by content hash in Valkey |

---

## 9. Implementation roadmap

### Phase A — Documentation & trait alignment (2–3 weeks)

- [x] Land multimodal `EmbeddingEngine` + `VectorStore` seams in `graphrag-core` (PR #185)
- [ ] Add adapters or guidance connecting text-only `AsyncEmbedder` / `AsyncVectorStore` to multimodal `EmbeddingEngine` / `VectorStore`
- [ ] Document Valkey key schema for agent/inference caches
- [ ] Decide whether LanceDB belongs in core, server, or a dedicated storage crate

### Phase B — Production wiring (4–6 weeks)

- [ ] `fastembed` feature on `graphrag-server` with config switch
- [ ] Qdrant collection templates (payload indexes: `tenant_id`, `doc_id`, `node_id`, `modality`)
- [ ] Hybrid retrieval spike (Tantivy + Qdrant fusion)
- [ ] LanceDB implementation for create/add/search/delete with integration tests

### Phase C — Platform integration (6–10 weeks)

- [ ] oxidizedgraph optional `ValkeyCacheSink` for prompt/session cache, based from oxidizedgraph `develop`
- [ ] OTel spans: cache hit/miss, embed latency, ANN recall@k
- [ ] Keep durable run audit in AIVCS / `TransitionLog`; use Valkey only for ephemeral cache/session metadata
- [ ] Managed service runbooks (ElastiCache Valkey, Qdrant Cloud)

---

## 10. Open questions for research group

1. **Embedding default:** fastembed vs candle vs remote-only for production GraphRAG server?
2. **Hybrid retrieval:** Tantivy-in-Rust vs Qdrant sparse-only for BM25+dense?
3. **Multimodal vectors:** single collection with `modality` filter vs per-modality collections?
4. **Cache coherency:** should graph checkpoint IDs be stored in Valkey session blobs?
5. **Governance:** embedding dimension locks per collection — who owns rotation policy?

Please comment on this document via the tracking issue (see §11) or in the research group channel.

---

## 11. Distribution

| Channel | Action |
|---------|--------|
| Repository | `docs/TDD-ai-inference-storage-and-rag-vectors.md` (this file) |
| GitHub | Tracking issue in `stevedores-org/oxidizedRAG` (research label) |
| Cross-link | `docs/architecture.md` § storage (pending) |

---

## 12. References

- [Valkey](https://valkey.io/) — BSD-3 Redis fork
- [Dragonfly](https://www.dragonflydb.io/) — Apache-2.0 Redis-compatible
- [Qdrant](https://qdrant.tech/) — vector search engine, Apache-2.0
- [LanceDB](https://lancedb.com/) — embedded vector lake
- [fastembed-rs](https://github.com/Anush008/fastembed-rs)
- [instant-distance](https://github.com/insta-rs/instant-distance)
- [oxidizedgraph Issue #18](https://github.com/stevedores-org/oxidizedgraph/issues/18) — autonomous orchestration roadmap
- oxidizedRAG `graphrag-core/Cargo.toml` — `vector-hnsw`, `neural-embeddings`, `qdrant` features

---

*End of document.*
