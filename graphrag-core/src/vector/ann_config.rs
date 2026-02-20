//! ANN (Approximate Nearest Neighbor) configuration and performance profiles.
//!
//! Provides `ANNConfig` for parameterizing vector index construction and search,
//! with built-in presets for common use cases.

use serde::{Deserialize, Serialize};

/// Distance metric for vector similarity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (default).
    #[default]
    Cosine,
    /// Euclidean (L2) distance.
    Euclidean,
    /// Dot-product similarity.
    DotProduct,
}

/// Configuration for ANN index construction and search.
///
/// Controls the trade-off between speed, memory, and recall quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ANNConfig {
    /// Number of neighbors to consider during index construction.
    /// Higher values produce better recall but slower build times.
    pub ef_construction: usize,
    /// Number of neighbors to consider during search.
    /// Higher values produce better recall but slower queries.
    pub ef_search: usize,
    /// Maximum number of connections per node in the HNSW graph.
    /// Higher values improve recall but increase memory usage.
    pub m: usize,
    /// Distance metric for similarity computation.
    pub distance_metric: DistanceMetric,
}

impl ANNConfig {
    /// Create a custom ANN configuration.
    pub fn new(ef_construction: usize, ef_search: usize, m: usize, distance_metric: DistanceMetric) -> Self {
        Self {
            ef_construction,
            ef_search,
            m,
            distance_metric,
        }
    }

    /// Fast profile: prioritizes speed over recall.
    ///
    /// Good for interactive/real-time use cases where latency matters more
    /// than finding the absolute best matches.
    pub fn fast() -> Self {
        Self {
            ef_construction: 100,
            ef_search: 50,
            m: 12,
            distance_metric: DistanceMetric::Cosine,
        }
    }

    /// Balanced profile: reasonable trade-off between speed and recall.
    ///
    /// Suitable for most production workloads.
    pub fn balanced() -> Self {
        Self {
            ef_construction: 200,
            ef_search: 100,
            m: 16,
            distance_metric: DistanceMetric::Cosine,
        }
    }

    /// Precise profile: prioritizes recall over speed.
    ///
    /// Best for offline/batch use cases where accuracy is critical.
    pub fn precise() -> Self {
        Self {
            ef_construction: 400,
            ef_search: 200,
            m: 24,
            distance_metric: DistanceMetric::Cosine,
        }
    }

    /// Builder-style setter for distance metric.
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }
}

impl Default for ANNConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_profile_params() {
        let config = ANNConfig::fast();
        assert_eq!(config.ef_construction, 100);
        assert_eq!(config.ef_search, 50);
        assert_eq!(config.m, 12);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
    }

    #[test]
    fn test_balanced_profile_params() {
        let config = ANNConfig::balanced();
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 100);
        assert_eq!(config.m, 16);
    }

    #[test]
    fn test_precise_profile_params() {
        let config = ANNConfig::precise();
        assert_eq!(config.ef_construction, 400);
        assert_eq!(config.ef_search, 200);
        assert_eq!(config.m, 24);
    }

    #[test]
    fn test_custom_config() {
        let config = ANNConfig::new(150, 75, 20, DistanceMetric::Euclidean);
        assert_eq!(config.ef_construction, 150);
        assert_eq!(config.ef_search, 75);
        assert_eq!(config.m, 20);
        assert_eq!(config.distance_metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_builder_with_distance_metric() {
        let config = ANNConfig::fast().with_distance_metric(DistanceMetric::DotProduct);
        assert_eq!(config.ef_search, 50); // fast params preserved
        assert_eq!(config.distance_metric, DistanceMetric::DotProduct);
    }

    #[test]
    fn test_default_is_balanced() {
        let default = ANNConfig::default();
        let balanced = ANNConfig::balanced();
        assert_eq!(default.ef_construction, balanced.ef_construction);
        assert_eq!(default.ef_search, balanced.ef_search);
        assert_eq!(default.m, balanced.m);
    }

    #[test]
    fn test_serde_roundtrip() {
        let config = ANNConfig::precise().with_distance_metric(DistanceMetric::Euclidean);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ANNConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.ef_construction, config.ef_construction);
        assert_eq!(deserialized.ef_search, config.ef_search);
        assert_eq!(deserialized.m, config.m);
        assert_eq!(deserialized.distance_metric, config.distance_metric);
    }
}
