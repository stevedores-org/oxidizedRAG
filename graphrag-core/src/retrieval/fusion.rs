//! Pluggable fusion policies for hybrid retrieval ranking.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;

use crate::config::HybridFusionConfig;

/// Source channel that produced a ranked retrieval candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RetrievalSource {
    /// Embedding/vector-similarity retrieval.
    Vector,
    /// Keyword/statistical retrieval (for example BM25).
    Keyword,
    /// Graph traversal/path-based retrieval.
    Graph,
}

/// Input candidate with score and rank for policy fusion.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RankedResult {
    /// Stable candidate identifier.
    pub id: String,
    /// Retrieval source that produced this candidate.
    pub source: RetrievalSource,
    /// Raw source score before policy fusion.
    pub score: f32,
    /// Rank inside the source-specific candidate list (0-based).
    pub rank: usize,
}

/// Output candidate after policy fusion and score aggregation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FusedResult {
    /// Stable candidate identifier.
    pub id: String,
    /// Final fused score used for ordering.
    pub score: f32,
    /// Per-source contribution scores.
    pub per_source_scores: HashMap<RetrievalSource, f32>,
}

/// Metrics emitted per policy run for tuning and comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionMetrics {
    /// Fusion policy name that generated this report.
    pub policy_name: String,
    /// Minimum fused score in the output set.
    pub score_min: f32,
    /// Maximum fused score in the output set.
    pub score_max: f32,
    /// Mean fused score in the output set.
    pub score_mean: f32,
    /// Recall@k against a provided relevant-id set.
    pub recall_at_k: f32,
}

/// Weighted-sum policy configuration.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeightedSum {
    /// Weight for vector-source scores.
    pub vector_weight: f32,
    /// Weight for keyword-source scores.
    pub keyword_weight: f32,
    /// Weight for graph-source scores.
    pub graph_weight: f32,
}

/// Reciprocal Rank Fusion policy configuration.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReciprocalRankFusion {
    /// RRF constant (higher values flatten rank contributions).
    pub k: f32,
}

/// Cascade fusion policy configuration.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CascadeFusion {
    /// Early-stop threshold for terminating later fusion stages.
    pub early_stop_score: f32,
}

/// Trait for pluggable hybrid retrieval fusion strategies.
#[async_trait]
pub trait FusionPolicy: Send + Sync {
    /// Fuse source-ranked candidates into a top-k final ordering.
    async fn fuse(&self, results: Vec<RankedResult>, top_k: usize) -> Vec<FusedResult>;
    /// Stable policy identifier.
    fn name(&self) -> &str;
}

#[async_trait]
impl FusionPolicy for WeightedSum {
    async fn fuse(&self, results: Vec<RankedResult>, top_k: usize) -> Vec<FusedResult> {
        fuse_with_weighted_sum(results, top_k, self)
    }

    fn name(&self) -> &str {
        "weighted_sum"
    }
}

#[async_trait]
impl FusionPolicy for ReciprocalRankFusion {
    async fn fuse(&self, results: Vec<RankedResult>, top_k: usize) -> Vec<FusedResult> {
        let mut combined: HashMap<String, FusedResult> = HashMap::new();
        for result in results {
            let contrib = 1.0 / (self.k + result.rank as f32 + 1.0);
            let entry = combined
                .entry(result.id.clone())
                .or_insert_with(|| FusedResult {
                    id: result.id.clone(),
                    score: 0.0,
                    per_source_scores: HashMap::new(),
                });
            entry.score += contrib;
            *entry.per_source_scores.entry(result.source).or_insert(0.0) += contrib;
        }
        sort_and_limit(combined, top_k)
    }

    fn name(&self) -> &str {
        "rrf"
    }
}

#[async_trait]
impl FusionPolicy for CascadeFusion {
    async fn fuse(&self, results: Vec<RankedResult>, top_k: usize) -> Vec<FusedResult> {
        let mut grouped: HashMap<RetrievalSource, Vec<RankedResult>> = HashMap::new();
        for result in results {
            grouped.entry(result.source).or_default().push(result);
        }

        for values in grouped.values_mut() {
            values.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let order = [
            RetrievalSource::Keyword,
            RetrievalSource::Graph,
            RetrievalSource::Vector,
        ];

        let mut chosen: HashMap<String, FusedResult> = HashMap::new();
        for source in order {
            if let Some(stage_results) = grouped.get(&source) {
                for result in stage_results {
                    let entry = chosen
                        .entry(result.id.clone())
                        .or_insert_with(|| FusedResult {
                            id: result.id.clone(),
                            score: 0.0,
                            per_source_scores: HashMap::new(),
                        });
                    entry.score = entry.score.max(result.score);
                    entry.per_source_scores.insert(source, result.score);
                }

                let best = chosen.values().map(|v| v.score).fold(0.0_f32, f32::max);
                if best >= self.early_stop_score {
                    break;
                }
            }
        }

        let mut output: Vec<FusedResult> = chosen.into_values().collect();
        output.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        output.truncate(top_k);
        output
    }

    fn name(&self) -> &str {
        "cascade"
    }
}

/// Select a concrete fusion policy from `HybridFusionConfig`.
pub fn policy_from_config(config: &HybridFusionConfig) -> Box<dyn FusionPolicy> {
    match config.policy.as_str() {
        "rrf" => Box::new(ReciprocalRankFusion { k: config.rrf_k }),
        "cascade" => Box::new(CascadeFusion {
            early_stop_score: config.cascade_early_stop_score,
        }),
        _ => Box::new(WeightedSum {
            vector_weight: config.weights.keywords,
            keyword_weight: config.weights.bm25,
            graph_weight: config.weights.graph,
        }),
    }
}

/// Compute score-distribution and recall metrics for a fused output set.
pub fn compute_metrics(
    policy_name: &str,
    fused: &[FusedResult],
    relevant_ids: &HashSet<String>,
) -> FusionMetrics {
    if fused.is_empty() {
        return FusionMetrics {
            policy_name: policy_name.to_string(),
            score_min: 0.0,
            score_max: 0.0,
            score_mean: 0.0,
            recall_at_k: 0.0,
        };
    }

    let score_min = fused.iter().map(|f| f.score).fold(f32::INFINITY, f32::min);
    let score_max = fused
        .iter()
        .map(|f| f.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let score_mean = fused.iter().map(|f| f.score).sum::<f32>() / fused.len() as f32;

    let retrieved_relevant = fused
        .iter()
        .filter(|f| relevant_ids.contains(&f.id))
        .count() as f32;
    let recall_at_k = if relevant_ids.is_empty() {
        0.0
    } else {
        retrieved_relevant / relevant_ids.len() as f32
    };

    FusionMetrics {
        policy_name: policy_name.to_string(),
        score_min,
        score_max,
        score_mean,
        recall_at_k,
    }
}

fn fuse_with_weighted_sum(
    results: Vec<RankedResult>,
    top_k: usize,
    policy: &WeightedSum,
) -> Vec<FusedResult> {
    let mut combined: HashMap<String, FusedResult> = HashMap::new();
    for result in results {
        let source_weight = match result.source {
            RetrievalSource::Vector => policy.vector_weight,
            RetrievalSource::Keyword => policy.keyword_weight,
            RetrievalSource::Graph => policy.graph_weight,
        };
        let weighted = result.score * source_weight;
        let entry = combined
            .entry(result.id.clone())
            .or_insert_with(|| FusedResult {
                id: result.id.clone(),
                score: 0.0,
                per_source_scores: HashMap::new(),
            });
        entry.score += weighted;
        *entry.per_source_scores.entry(result.source).or_insert(0.0) += weighted;
    }
    sort_and_limit(combined, top_k)
}

fn sort_and_limit(combined: HashMap<String, FusedResult>, top_k: usize) -> Vec<FusedResult> {
    let mut out: Vec<FusedResult> = combined.into_values().collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    out.truncate(top_k);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_results() -> Vec<RankedResult> {
        vec![
            RankedResult {
                id: "A".to_string(),
                source: RetrievalSource::Vector,
                score: 0.9,
                rank: 0,
            },
            RankedResult {
                id: "B".to_string(),
                source: RetrievalSource::Vector,
                score: 0.8,
                rank: 1,
            },
            RankedResult {
                id: "A".to_string(),
                source: RetrievalSource::Keyword,
                score: 0.7,
                rank: 0,
            },
            RankedResult {
                id: "C".to_string(),
                source: RetrievalSource::Graph,
                score: 0.95,
                rank: 0,
            },
        ]
    }

    #[tokio::test]
    async fn deterministic_top_k_weighted_sum() {
        let policy = WeightedSum {
            vector_weight: 0.5,
            keyword_weight: 0.3,
            graph_weight: 0.2,
        };
        let first = policy.fuse(fixed_results(), 3).await;
        let second = policy.fuse(fixed_results(), 3).await;
        assert_eq!(first, second);
        assert_eq!(first[0].id, "A");
    }

    #[tokio::test]
    async fn policy_swap_is_safe_and_produces_results() {
        let cfg = HybridFusionConfig::default();
        let weighted = policy_from_config(&cfg);
        assert_eq!(weighted.name(), "weighted_sum");
        assert!(!weighted.fuse(fixed_results(), 2).await.is_empty());

        let mut rrf_cfg = cfg.clone();
        rrf_cfg.policy = "rrf".to_string();
        let rrf = policy_from_config(&rrf_cfg);
        assert_eq!(rrf.name(), "rrf");
        assert!(!rrf.fuse(fixed_results(), 2).await.is_empty());

        let mut cascade_cfg = cfg;
        cascade_cfg.policy = "cascade".to_string();
        let cascade = policy_from_config(&cascade_cfg);
        assert_eq!(cascade.name(), "cascade");
        assert!(!cascade.fuse(fixed_results(), 2).await.is_empty());
    }

    #[tokio::test]
    async fn recall_metrics_available_per_policy() {
        let policy = ReciprocalRankFusion { k: 60.0 };
        let fused = policy.fuse(fixed_results(), 3).await;
        let relevant: HashSet<String> = ["A".to_string(), "C".to_string()].into_iter().collect();
        let metrics = compute_metrics(policy.name(), &fused, &relevant);
        assert_eq!(metrics.policy_name, "rrf");
        assert!(metrics.recall_at_k >= 0.5);
        assert!(metrics.score_max >= metrics.score_min);
    }

    #[tokio::test]
    async fn empty_and_tie_cases_are_stable() {
        let policy = WeightedSum {
            vector_weight: 1.0,
            keyword_weight: 1.0,
            graph_weight: 1.0,
        };

        let empty = policy.fuse(Vec::new(), 10).await;
        assert!(empty.is_empty());

        let ties = vec![
            RankedResult {
                id: "x".to_string(),
                source: RetrievalSource::Vector,
                score: 1.0,
                rank: 0,
            },
            RankedResult {
                id: "y".to_string(),
                source: RetrievalSource::Vector,
                score: 1.0,
                rank: 0,
            },
        ];
        let out = policy.fuse(ties, 2).await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, "x");
        assert_eq!(out[1].id, "y");
    }
}
