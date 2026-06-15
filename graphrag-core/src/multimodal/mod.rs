//! Multimodal RAG primitives — types and async trait seams for
//! cross-modal embedding, ingestion, storage, and search.
//!
//! This module is **additive** to the existing text-only
//! [`crate::embeddings::EmbeddingProvider`]; it does not replace it.
//! Text-only callers stay on `EmbeddingProvider`. Callers that need to
//! embed images, audio, video, or PDFs alongside text use the seams
//! defined here.
//!
//! ## Status
//!
//! Types + traits only. No backends yet. Tracked in:
//! - Foundation: stevedores-org/oxidizedRAG#146, #148, #153, #156
//! - In-memory fakes: #147, #149 (next PR)
//! - File / URL ingestors: #154, #155
//! - Gemini backend: #151, #152
//! - SQLite vector store: #150
//! - Searcher / pipeline / orchestrator: #157, #158, #159
//!
//! See [TDD](../../../docs/tdd-multimodal-and-service.md) and meta-epic
//! stevedores-org/oxidizedRAG#182.
//!
//! ## Layout
//!
//! - [`types`] — [`types::MultimodalContent`], [`types::ContentType`], [`types::ContentSource`]
//! - [`embedding`] — [`embedding::Embedding`], [`embedding::EmbeddingEngine`] trait, [`embedding::EmbeddingError`]
//! - [`store`] — [`store::VectorStore`] trait, [`store::SearchFilters`], [`store::SearchResult`], [`store::VectorStoreError`]
//! - [`ingest`] — [`ingest::MultimodalIngestor`] trait, [`ingest::IngestError`]
//! - [`search`] — [`search::CrossModalSearcher`] trait, [`search::SearchError`]

pub mod embedding;
pub mod fakes;
pub mod ingest;
pub mod search;
pub mod store;
pub mod types;

pub use embedding::{Embedding, EmbeddingEngine, EmbeddingError, EmbeddingMetadata, ModelInfo};
pub use ingest::{IngestError, MultimodalIngestor};
pub use search::{CrossModalSearcher, SearchError};
pub use store::{SearchFilters, SearchResult, VectorStore, VectorStoreError};
pub use types::{ContentSource, ContentType, MultimodalContent};
