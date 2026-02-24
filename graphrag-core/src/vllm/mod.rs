//! vLLM / llm-d LLM integration
//!
//! This module provides integration with vLLM and llm-d inference servers
//! via their OpenAI-compatible API.

use crate::core::{GraphRAGError, Result};

/// vLLM configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VllmConfig {
    /// Enable vLLM integration
    pub enabled: bool,
    /// Base URL of the vLLM server (e.g. "http://localhost:8000")
    pub base_url: String,
    /// Model identifier
    pub model: String,
    /// Optional API key for authenticated deployments
    pub api_key: Option<String>,
    /// Timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for generation (0.0 - 2.0)
    pub temperature: Option<f32>,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "http://localhost:8000".to_string(),
            model: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
            api_key: None,
            timeout_seconds: 30,
            max_tokens: Some(2000),
            temperature: Some(0.7),
        }
    }
}

/// vLLM client for LLM inference via the OpenAI-compatible API
#[derive(Debug, Clone)]
pub struct VllmClient {
    config: VllmConfig,
    #[cfg(feature = "ureq")]
    client: ureq::Agent,
}

impl VllmClient {
    /// Create a new vLLM client
    pub fn new(config: VllmConfig) -> Self {
        Self {
            config: config.clone(),
            #[cfg(feature = "ureq")]
            client: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(config.timeout_seconds))
                .build(),
        }
    }

    /// Access the config
    pub fn config(&self) -> &VllmConfig {
        &self.config
    }

    /// Generate a chat completion using the OpenAI-compatible endpoint.
    #[cfg(feature = "ureq")]
    pub fn chat_completion(&self, prompt: &str) -> Result<String> {
        let endpoint = format!("{}/v1/chat/completions", self.config.base_url);

        let mut request_body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": false,
        });

        if let Some(max_tokens) = self.config.max_tokens {
            request_body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(temperature) = self.config.temperature {
            request_body["temperature"] = serde_json::json!(temperature);
        }

        let mut req = self
            .client
            .post(&endpoint)
            .set("Content-Type", "application/json");

        if let Some(ref api_key) = self.config.api_key {
            req = req.set("Authorization", &format!("Bearer {api_key}"));
        }

        let response = req.send_json(&request_body).map_err(|e| {
            GraphRAGError::Generation {
                message: format!("vLLM API request failed: {e}"),
            }
        })?;

        let json_response: serde_json::Value =
            response.into_json().map_err(|e| GraphRAGError::Generation {
                message: format!("Failed to parse vLLM response: {e}"),
            })?;

        // OpenAI format: choices[0].message.content
        json_response["choices"]
            .as_array()
            .and_then(|choices| choices.first())
            .and_then(|choice| choice["message"]["content"].as_str())
            .map(Self::strip_think_tags)
            .ok_or_else(|| GraphRAGError::Generation {
                message: format!("Invalid vLLM response format: {json_response:?}"),
            })
    }

    /// Generate embeddings using the OpenAI-compatible endpoint.
    #[cfg(feature = "ureq")]
    pub fn embeddings(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        let endpoint = format!("{}/v1/embeddings", self.config.base_url);

        let input_value = if inputs.len() == 1 {
            serde_json::json!(inputs[0])
        } else {
            serde_json::json!(inputs)
        };

        let request_body = serde_json::json!({
            "model": self.config.model,
            "input": input_value,
        });

        let mut req = self
            .client
            .post(&endpoint)
            .set("Content-Type", "application/json");

        if let Some(ref api_key) = self.config.api_key {
            req = req.set("Authorization", &format!("Bearer {api_key}"));
        }

        let response = req.send_json(&request_body).map_err(|e| {
            GraphRAGError::Generation {
                message: format!("vLLM embeddings request failed: {e}"),
            }
        })?;

        let json_response: serde_json::Value =
            response.into_json().map_err(|e| GraphRAGError::Generation {
                message: format!("Failed to parse vLLM embeddings response: {e}"),
            })?;

        json_response["data"]
            .as_array()
            .map(|entries| {
                entries.iter()
                    .filter_map(|item| {
                        item["embedding"].as_array().map(|emb| {
                            emb.iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect()
                        })
                    })
                    .collect()
            })
            .ok_or_else(|| GraphRAGError::Generation {
                message: format!("Invalid embeddings response: {json_response:?}"),
            })
    }

    /// Generate chat completion (fallback when ureq feature is disabled)
    #[cfg(not(feature = "ureq"))]
    pub fn chat_completion(&self, _prompt: &str) -> Result<String> {
        Err(GraphRAGError::Generation {
            message: "ureq feature required for vLLM integration".to_string(),
        })
    }

    /// Generate embeddings (fallback when ureq feature is disabled)
    #[cfg(not(feature = "ureq"))]
    pub fn embeddings(&self, _inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        Err(GraphRAGError::Generation {
            message: "ureq feature required for vLLM integration".to_string(),
        })
    }

    /// Remove `<think>...</think>` tags from LLM output (Qwen3 and similar models).
    fn strip_think_tags(text: &str) -> String {
        let mut result = text.to_string();
        while let Some(start) = result.find("<think>") {
            if let Some(end) = result[start..].find("</think>") {
                let end_pos = start + end + "</think>".len();
                result.replace_range(start..end_pos, "");
            } else {
                result.replace_range(start..start + "<think>".len(), "");
                break;
            }
        }
        result.trim().to_string()
    }
}

/// Async vLLM generator implementing `AsyncLanguageModel`.
#[cfg(feature = "async-traits")]
pub struct AsyncVllmGenerator {
    client: VllmClient,
}

#[cfg(feature = "async-traits")]
impl AsyncVllmGenerator {
    /// Create a new async vLLM generator
    pub fn new(config: VllmConfig) -> Self {
        Self {
            client: VllmClient::new(config),
        }
    }
}

#[cfg(feature = "async-traits")]
#[async_trait::async_trait]
impl crate::core::traits::AsyncLanguageModel for AsyncVllmGenerator {
    type Error = GraphRAGError;

    async fn complete(&self, prompt: &str) -> Result<String> {
        self.client.chat_completion(prompt)
    }

    async fn complete_with_params(
        &self,
        prompt: &str,
        _params: crate::core::traits::GenerationParams,
    ) -> Result<String> {
        self.client.chat_completion(prompt)
    }

    async fn is_available(&self) -> bool {
        self.client.config.enabled
    }

    async fn model_info(&self) -> crate::core::traits::ModelInfo {
        crate::core::traits::ModelInfo {
            name: self.client.config.model.clone(),
            version: None,
            max_context_length: Some(8192),
            supports_streaming: false,
        }
    }
}

/// vLLM embedding provider implementing `EmbeddingProvider`.
#[cfg(feature = "async-traits")]
pub struct VllmEmbeddingProvider {
    client: VllmClient,
    dimensions: usize,
    initialized: bool,
}

#[cfg(feature = "async-traits")]
impl VllmEmbeddingProvider {
    /// Create a new vLLM embedding provider
    pub fn new(config: VllmConfig, dimensions: usize) -> Self {
        Self {
            client: VllmClient::new(config),
            dimensions,
            initialized: false,
        }
    }
}

#[cfg(feature = "async-traits")]
#[async_trait::async_trait]
impl crate::embeddings::EmbeddingProvider for VllmEmbeddingProvider {
    async fn initialize(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.client.embeddings(&[text])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| GraphRAGError::Generation {
                message: "No embedding returned".to_string(),
            })
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.client.embeddings(texts)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn is_available(&self) -> bool {
        self.client.config.enabled
    }

    fn provider_name(&self) -> &str {
        "vllm"
    }
}

/// Sync adapter for using `VllmClient` as an `LLMInterface` (e.g. for `AnswerGenerator`).
#[cfg(feature = "async")]
pub struct VllmLLMAdapter {
    client: VllmClient,
}

#[cfg(feature = "async")]
impl VllmLLMAdapter {
    /// Create a new sync adapter wrapping a VllmClient
    pub fn new(config: VllmConfig) -> Self {
        Self {
            client: VllmClient::new(config),
        }
    }
}

#[cfg(feature = "async")]
impl crate::generation::LLMInterface for VllmLLMAdapter {
    fn generate_response(&self, prompt: &str) -> Result<String> {
        self.client.chat_completion(prompt)
    }

    fn generate_summary(&self, content: &str, max_length: usize) -> Result<String> {
        let prompt = format!(
            "Summarize the following in at most {max_length} characters:\n\n{content}"
        );
        self.client.chat_completion(&prompt)
    }

    fn extract_key_points(&self, content: &str, num_points: usize) -> Result<Vec<String>> {
        let prompt = format!(
            "Extract {num_points} key points from:\n\n{content}\n\nReturn one point per line."
        );
        let response = self.client.chat_completion(&prompt)?;
        Ok(response
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(num_points)
            .map(String::from)
            .collect())
    }
}
