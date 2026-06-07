# TDD — Multimodal Embeddings + Cluster-Native Service

**Status**: Proposed
**Author**: filed by tooling, owner: maintainers
**Created**: 2026-06-02
**Tracking issue**: TBD (filed alongside this doc)

This Technical Design Doc rationalizes the 20 open issues on `stevedores-org/oxidizedRAG` (as of 2026-06-02) into a single sequenced plan. It does not introduce new product scope — it organizes what is already filed into a delivery order the team can actually ship against, and calls out where the existing issues are too ambitious or already partially solved.

## 1. Motivation

Three independent threads are currently filed as parallel epics:

| Thread | Scope | Filed as |
|---|---|---|
| **A. Bug debt** | Silent zero-vector embeddings; duplicate `RagRunRecorder` types | #167, #178 |
| **B. Multimodal RAG via Gemini 2** | New `EmbeddingEngine`/`VectorStore`/`Ingestor`/`Searcher` traits, Gemini client, SQLite vector store, batch pipeline, orchestrator | #145, #146–#161 (17 issues) |
| **C. Cluster deployment** | Package server mode as a K8s `Deployment`/`Service` on AKS, Nix-built OCI image, Workload Identity to Cosmos + Blob | #171 (epic), #172 |

Today these threads are filed as a flat list of ~20 issues with overlapping acceptance criteria, several broken dependency chains, and a quality bar (`100% test coverage` repeated per issue) that does not match the team's actual cadence.

This TDD reframes them as **three epics, eight phases**, each phase landable as 1–3 PRs against `develop`. It does not change the *what* — only the *when* and the *unit-of-work*.

## 2. Current state (audited 2026-06-02)

### What already works

- `EmbeddingProvider` (text-only, `async`) in `graphrag-core/src/embeddings/mod.rs` — used by `HttpEmbeddingProvider` (OpenAI, Voyage, Cohere, Jina, Mistral, Together AI).
- Pipeline DAG, content-hash stage cache, dual sync/async trait pattern (see `graphrag-core/src/core/traits.rs`, `graphrag-core/src/pipeline/`).
- `graphrag-server` Actix-web REST API.
- `graphrag-aivcs` ledger-aware async `recorder::RagRunRecorder` (per PR #179).
- `flake.nix` builds the workspace via Nix (no docker output yet).

### What is broken

- `HuggingFaceEmbeddings::embed` (`graphrag-core/src/embeddings/huggingface.rs:186-212`) returns `Ok(vec![0.0; dims])` when `neural-embeddings` is enabled — semantically wrong; passes through the pipeline silently.
- `graphrag-core/src/embeddings/neural.rs` is a one-line `// TODO:` stub.
- `graphrag-aivcs` carries **two** `RagRunRecorder` types — legacy sync `run_recorder::RagRunRecorder` (re-exported from `lib.rs`, used by `aivcs_adapter` + examples) and async ledger-aware `recorder::RagRunRecorder` (used by integration tests). The duplicate surface is per-PR-#179 a known follow-up.

### What is green-field

- None of `MultimodalContent`, `ContentSource`, `MultimodalIngestor`, `EmbeddingEngine` (multimodal variant), `VectorStore` (multimodal variant), `CrossModalSearcher`, `BatchEmbeddingPipeline`, `RagOrchestrator` exist today. All 17 children of #145 are new types.
- No `deploy/`, `infra/`, or `kustomize/` directory exists. `flake.nix` does not yet expose a `dockerImage` output.
- No `gemini` crate exists in the workspace.

## 3. Non-goals

- Re-architecting the existing `EmbeddingProvider` text trait. The multimodal `EmbeddingEngine` is **additive** — text-only callers stay on `EmbeddingProvider`.
- Implementing every backend listed in the children of #145 in one go. SQLite vector store, full Candle local inference, and remote URL ingestion can each be follow-on PRs.
- Cross-repo coordination for `crossplane-heaven#10` (Workload Identity), `dockworker.ai#6` (image build canonicals), or `lornu.ai#2347` (Python client). Stub the integration surface; defer the binding.
- "100% test coverage" as a hard gate. The repo today is at ~75% on `graphrag-core` per CI; mandating 100% per-PR will stall the queue. The target is **happy-path + error-path coverage on the public API of each new type**.

## 4. Design

### 4.1 Multimodal embedding seam — new `EmbeddingEngine` trait

The existing `EmbeddingProvider` is text-only (`async fn embed(&self, text: &str) -> Result<Vec<f32>>`). Multimodal needs a new trait alongside, not a replacement:

```rust
// graphrag-core/src/multimodal/types.rs
#[derive(Debug, Clone)]
pub enum MultimodalContent {
    Text(String),
    Image { bytes: Vec<u8>, mime: String },
    Audio { bytes: Vec<u8>, mime: String },
    Video { bytes: Vec<u8>, mime: String },
    Pdf  { bytes: Vec<u8> },
}

#[derive(Debug, Clone)]
pub enum ContentSource {
    LocalFile { path: PathBuf, mime_type: Option<String> },
    RemoteUrl { url: String },
    DirectContent { data: Vec<u8>, mime_type: String },
    Directory { path: PathBuf, pattern: String },
}

// graphrag-core/src/multimodal/engine.rs
#[async_trait]
pub trait EmbeddingEngine: Send + Sync {
    async fn embed(&self, content: MultimodalContent) -> Result<Embedding, EmbeddingError>;
    async fn embed_batch(&self, batch: Vec<MultimodalContent>) -> Result<Vec<Embedding>, EmbeddingError>;
    fn model_info(&self) -> ModelInfo;
}
```

`Embedding` carries `id`, `content_hash`, `vector: Vec<f32>`, `content_type`, `metadata`, `created_at` — matching the shape in #146 but **not** hard-coding the 1408-dim Gemini-2 width. Dimensionality is reported via `model_info()`.

### 4.2 Vector storage seam — `VectorStore` trait

```rust
// graphrag-core/src/multimodal/store.rs
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn store(&self, embedding: Embedding) -> Result<(), VectorStoreError>;
    async fn store_batch(&self, embeddings: Vec<Embedding>) -> Result<(), VectorStoreError>;
    async fn search(&self, query: &[f32], k: usize, filters: SearchFilters) -> Result<Vec<SearchResult>, VectorStoreError>;
    async fn get(&self, id: &str) -> Result<Option<Embedding>, VectorStoreError>;
    async fn delete(&self, id: &str) -> Result<(), VectorStoreError>;
    async fn count(&self) -> Result<usize, VectorStoreError>;
}
```

In-memory impl (DashMap-backed) ships in the same PR as the trait, so unit tests downstream don't have to mock.

### 4.3 Ingestion seam — `MultimodalIngestor` trait

```rust
#[async_trait]
pub trait MultimodalIngestor: Send + Sync {
    async fn ingest(&self, source: ContentSource) -> Result<Vec<MultimodalContent>, IngestError>;
    async fn ingest_batch(&self, sources: Vec<ContentSource>) -> Result<Vec<MultimodalContent>, IngestError>;
}
```

Two impls follow as separate PRs: `FileIngestor` (local FS + glob, MIME detect via `infer`) and `RemoteUrlIngestor` (reqwest with retry).

### 4.4 Search seam — `CrossModalSearcher`

```rust
#[async_trait]
pub trait CrossModalSearcher: Send + Sync {
    async fn search(&self, query: MultimodalContent, k: usize) -> Result<Vec<SearchResult>, SearchError>;
    async fn search_with_filters(&self, query: MultimodalContent, filters: SearchFilters) -> Result<Vec<SearchResult>, SearchError>;
}
```

Default impl composes `EmbeddingEngine + VectorStore`. Pluggable so callers can swap in a hybrid (vector + BM25 + PageRank) impl later — consistent with the existing `graphrag-core/src/retrieval/` pattern.

### 4.5 Cluster deployment — Kustomize on AKS hub

Per the repo's standing rule (Kustomize, not Helm), `deploy/kustomize/` lays out:

```
deploy/kustomize/
  base/
    deployment.yaml      # graphrag-server, image pinned to ghcr.io/stevedores-org/oxidizedrag@sha256:...
    service.yaml         # ClusterIP, port 8080
    serviceaccount.yaml  # name only — annotation injected by XFederatedIdentity from crossplane-heaven#10
    kustomization.yaml
  overlays/
    aks-hub/
      kustomization.yaml # namespace, replicas, resource sizing, image tag patch
```

The Nix flake gains a `packages.<system>.dockerImage` output using `nix2container.buildImage`, labeled `agent-build/source=oxidizedRAG` and `agent-build/sha=$(git rev-parse HEAD)`. The Kustomize overlay pins the image by digest, not tag.

Workload Identity binding (Cosmos + Blob) lives in `crossplane-heaven#10` and is **not** authored here — the ServiceAccount manifest ships with no Azure annotations, and gets the federated identity patched in by the platform layer.

## 5. Sequenced delivery

Eight phases, each one PR (or a small ordered chain) against `develop`. Earlier phases unblock later ones; phases marked **(parallel)** can be opened concurrently with siblings.

| # | Phase | Closes / advances | PR scope | Risk |
|---|---|---|---|---|
| 0 | **TDD lands** | this doc | docs/tdd-multimodal-and-service.md | none |
| 1 | **Fail-loud HF embeddings** | #167 | `huggingface.rs`: replace `Ok(vec![0.0; ..])` with `Err`; delete `neural.rs` stub or gate the whole module out | low |
| 2 | **`graphrag-aivcs` recorder consolidation** | #178 | Pick one of {legacy sync, async ledger-aware}; delete the other; update `aivcs_adapter` and examples | low |
| 3 | **Multimodal types (foundation)** *(parallel-safe)* | #146, #148, #153, #156 | New `graphrag-core/src/multimodal/{mod,types,engine,store,ingest,search}.rs`. Types + traits only, no impls. Re-exported from `lib.rs` behind `multimodal` feature flag. | low |
| 4 | **Test doubles** *(blocked by 3)* | #147, #149 | `FakeEmbeddingEngine` + `MemoryVectorStore` in the same module. | low |
| 5 | **File + URL ingestors** *(blocked by 3)* | #154, #155 | `FileIngestor` (infer + glob) and `RemoteUrlIngestor` (reqwest, retry, rate-limit). | medium — new deps |
| 6 | **Gemini client + engine** *(blocked by 3)* | #151, #152 | New `gemini` workspace crate; `GeminiEmbeddingEngine` impl behind `gemini` feature flag. Mocked HTTP in tests. | medium — API key handling, mock recording |
| 7 | **Searcher + batch pipeline + orchestrator** *(blocked by 4)* | #157, #158, #159 | `CrossModalSearcherImpl`, `BatchEmbeddingPipeline`, `RagOrchestrator` builder. | medium |
| 8 | **SQLite vector store** *(blocked by 3)* | #150 | `SqliteVectorStore` behind `sqlite-vector` feature. | medium — sqlite-vss / sqlite-vec choice |
| 9 | **Integration tests + docs** *(blocked by 5–8)* | #160, #161 | `tests/multimodal_integration.rs` with fixtures; `ARCHITECTURE.md`, `docs/MULTIMODAL_DESIGN.md`, `docs/API_REFERENCE.md`, runnable `examples/`. | low |
| 10 | **K8s deployment manifests** *(parallel — does not depend on 3-9)* | #172 | `deploy/kustomize/{base,overlays/aks-hub}/*.yaml`, readiness probe wiring in `graphrag-server`, docs in `deploy/README.md`. SA annotation deferred to `crossplane-heaven#10`. | low |
| 11 | **Nix → OCI image** *(blocked by 10)* | #172 | `flake.nix`: add `packages.<system>.dockerImage` via `nix2container`; `agent-build/*` labels; reproducibility check in CI. | medium — substituter cache + label conformance |
| 12 | **Workload Identity wire-up** *(blocked by `crossplane-heaven#10`)* | #171, #172 | Patch overlay to consume the `XFederatedIdentity`-managed SA annotation; readiness gates on Cosmos + Blob auth. **Not opened until upstream lands.** | high — cross-repo |

Out of these, **phases 0–4 and 10** are landable today with no external blockers. Phases 6, 8, 11 introduce new dependencies (Gemini API key, sqlite-vec/vss, nix2container) and need their tooling decisions captured in the PR description.

## 6. Test strategy

Per-PR bar (replaces the per-issue "100% test coverage" claim):

1. **Trait/type PRs (3, 4, 6 type defs)**: doctest on each public type; one round-trip serde test where applicable; one constructor test per impl.
2. **Impl PRs (5, 6, 7, 8)**: happy-path async test using `FakeEmbeddingEngine` / `MemoryVectorStore`; one explicit error path (network failure, missing file, malformed input).
3. **Integration phase (9)**: end-to-end test with real fixtures (`tests/fixtures/sample.{txt,pdf,png,mp3}`) but Gemini mocked. Real Gemini hit gated behind `GEMINI_API_KEY` env var, skipped in CI.
4. **K8s phases (10–12)**: `kustomize build deploy/kustomize/overlays/aks-hub | kubeval` in CI. Deployment-level smoke test deferred to cluster.

Coverage measurement (`cargo tarpaulin`) lands as a CI job alongside phase 9, not gated per-PR.

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Gemini embedding 2 API surface shifts between preview and GA | Pin to a specific model name + version; wrap the HTTP client so the rest of the codebase doesn't depend on Gemini-specific shapes. |
| sqlite-vec vs sqlite-vss choice changes the schema | Defer the choice to phase 8; trait abstracts it. |
| Workload Identity binding in `crossplane-heaven#10` lands later than expected | Manifests ship with a placeholder SA; deployment fails closed (no static creds path). |
| Nix `dockerImage` reproducibility check flakes under different substituters | Use `stevedores-1` substituter only; cache `nix2container` output. |
| The 17-issue tree under #145 keeps growing | This TDD is the contract — new multimodal features get filed as follow-ons referencing the relevant phase. |

## 8. Open questions

1. **Embedding dimensionality** — Gemini 2 is 1408. The existing text providers vary (1536 OpenAI small, 1024 Voyage, etc.). Do we need a per-store dimensionality assertion at insert time, or rely on backend rejecting mismatch?
2. **`graphrag-aivcs` recorder consolidation direction** — keep the legacy sync API and grow it ledger-aware, or delete it and migrate `aivcs_adapter` + examples to the async one? PR #179 leaves both alive.
3. **Where does `graphrag-server` expose the multimodal query API** — same Actix routes (`/query`) with a polymorphic body, or a new `/query/multimodal` route? Affects `lornu.ai#2347` Python client design.

These are decided in their respective PR threads, not in this doc.

## 9. References

- Open issues snapshot (2026-06-02): #145, #146, #147, #148, #149, #150, #151, #152, #153, #154, #155, #156, #157, #158, #159, #160, #161, #167, #171, #172, #178
- Cross-repo dependencies: `stevedores-org/crossplane-heaven#10`, `lornu-ai/dockworker.ai#6`, `lornu-ai/lornu.ai#2347`
- Internal docs: `docs/architecture.md`, `docs/ci.md`, `AGENTS.md`, `CLAUDE.md`
