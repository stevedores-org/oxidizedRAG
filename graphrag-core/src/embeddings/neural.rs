//! Neural embedding models using Candle for local inference.
//!
//! This module is gated behind the `neural-embeddings` feature flag.
//!
//! ## Status
//!
//! Not implemented. Tracked in stevedores-org/oxidizedRAG#167. The previous
//! placeholder (a one-line `// TODO`) led the feature flag to advertise a
//! capability the crate did not provide. Until a real `NeuralEmbeddingProvider`
//! lands, attempting to use the local-Candle path on
//! [`crate::embeddings::huggingface::HuggingFaceEmbeddings`] returns
//! [`crate::core::error::GraphRAGError::Embedding`] rather than silently
//! producing zero vectors.
//!
//! Use one of the HTTP providers in [`crate::embeddings::api_providers`]
//! (`HttpEmbeddingProvider::{openai, voyage_ai, cohere, jina_ai, mistral,
//! together_ai}`) until local inference is wired up.

// Intentionally empty. Adding a real `NeuralEmbeddingProvider` is the next
// step on #167. Until then the module exists only to document the gap.
