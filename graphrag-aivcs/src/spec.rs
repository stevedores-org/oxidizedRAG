//! Content-addressed specification for a GraphRAG configuration.

use oxidized_state::ContentDigest;
use sha2::{Digest, Sha256};

/// Hashes a GraphRAG configuration into a [`ContentDigest`] for AIVCS tracking.
pub struct GraphRagSpec {
    /// 64-char hex SHA-256 of the serialized config.
    pub config_digest: String,
    /// Human-readable agent/pipeline name.
    pub agent_name: String,
}

impl GraphRagSpec {
    /// Create a new spec by hashing `config_json` with SHA-256.
    pub fn new(agent_name: impl Into<String>, config_json: &str) -> Self {
        let hash = Sha256::digest(config_json.as_bytes());
        Self {
            config_digest: hex::encode(hash),
            agent_name: agent_name.into(),
        }
    }

    /// Return a [`ContentDigest`] derived from `config_digest`.
    pub fn content_digest(&self) -> ContentDigest {
        ContentDigest::try_from(self.config_digest.clone())
            .expect("config_digest is a valid 64-char hex string")
    }
}
