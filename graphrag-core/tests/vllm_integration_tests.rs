//! Integration tests for the vLLM / llm-d module.
//!
//! All tests use wiremock — no real LLM calls are made.

#[cfg(feature = "vllm")]
mod vllm_tests {
    use graphrag_core::core::traits::{AsyncLanguageModel, GenerationParams};
    use graphrag_core::embeddings::EmbeddingProvider;
    use graphrag_core::vllm::{AsyncVllmGenerator, VllmConfig, VllmEmbeddingProvider};
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn make_config(base_url: &str) -> VllmConfig {
        VllmConfig {
            enabled: true,
            base_url: base_url.to_string(),
            model: "test-model".to_string(),
            api_key: None,
            timeout_seconds: 5,
            max_tokens: Some(100),
            temperature: Some(0.7),
        }
    }

    fn chat_response(content: &str) -> serde_json::Value {
        json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })
    }

    fn embedding_response(embeddings: &[Vec<f32>]) -> serde_json::Value {
        let data: Vec<_> = embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                json!({
                    "object": "embedding",
                    "index": i,
                    "embedding": emb
                })
            })
            .collect();
        json!({
            "object": "list",
            "data": data,
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        })
    }

    // ─── Chat Completion Tests ───────────────────────────────────────────

    #[tokio::test]
    async fn test_vllm_generator_complete_returns_content() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_response("Hello from vLLM!")),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let gen = AsyncVllmGenerator::new(make_config(&mock_server.uri()));
        let result = gen.complete("Hi").await;

        assert!(result.is_ok(), "complete failed: {:?}", result.err());
        assert_eq!(result.unwrap(), "Hello from vLLM!");
    }

    #[tokio::test]
    async fn test_vllm_generator_complete_with_params() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_response("Parameterized response")),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let gen = AsyncVllmGenerator::new(make_config(&mock_server.uri()));
        let params = GenerationParams {
            temperature: Some(0.5),
            max_tokens: Some(50usize),
            top_p: None,
            stop_sequences: None,
        };
        let result = gen.complete_with_params("Prompt", params).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Parameterized response");
    }

    #[tokio::test]
    async fn test_vllm_generator_is_available_when_enabled() {
        let gen = AsyncVllmGenerator::new(VllmConfig {
            enabled: true,
            ..VllmConfig::default()
        });
        assert!(gen.is_available().await);
    }

    #[tokio::test]
    async fn test_vllm_generator_not_available_when_disabled() {
        let gen = AsyncVllmGenerator::new(VllmConfig {
            enabled: false,
            ..VllmConfig::default()
        });
        assert!(!gen.is_available().await);
    }

    #[tokio::test]
    async fn test_vllm_model_info_returns_configured_name() {
        let gen = AsyncVllmGenerator::new(VllmConfig {
            model: "my-custom-model".to_string(),
            ..VllmConfig::default()
        });
        let info = gen.model_info().await;
        assert_eq!(info.name, "my-custom-model");
    }

    #[tokio::test]
    async fn test_vllm_generator_returns_error_on_5xx() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .expect(1)
            .mount(&mock_server)
            .await;

        let gen = AsyncVllmGenerator::new(make_config(&mock_server.uri()));
        let result = gen.complete("fail").await;

        assert!(result.is_err(), "should return error on 500");
    }

    #[tokio::test]
    async fn test_vllm_generator_handles_timeout() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_response("delayed"))
                    .set_delay(std::time::Duration::from_secs(10)),
            )
            .mount(&mock_server)
            .await;

        let config = VllmConfig {
            timeout_seconds: 1, // 1 second timeout
            ..make_config(&mock_server.uri())
        };
        let gen = AsyncVllmGenerator::new(config);
        let result = gen.complete("timeout test").await;

        assert!(result.is_err(), "should timeout");
    }

    #[tokio::test]
    async fn test_vllm_generator_sends_api_key_header() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("Authorization", "Bearer test-secret-key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(chat_response("authed response")),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let config = VllmConfig {
            api_key: Some("test-secret-key".to_string()),
            ..make_config(&mock_server.uri())
        };
        let gen = AsyncVllmGenerator::new(config);
        let result = gen.complete("auth test").await;

        assert!(result.is_ok(), "authed request should succeed: {:?}", result.err());
        assert_eq!(result.unwrap(), "authed response");
    }

    // ─── Embedding Tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_vllm_embedding_single_input() {
        let mock_server = MockServer::start().await;

        let expected_embedding = vec![0.1f32, 0.2, 0.3, 0.4];

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(embedding_response(&[expected_embedding.clone()])),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let mut provider = VllmEmbeddingProvider::new(make_config(&mock_server.uri()), 4);
        provider.initialize().await.unwrap();

        let result = provider.embed("test text").await;
        assert!(result.is_ok(), "embed failed: {:?}", result.err());

        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 4);
        assert!((embedding[0] - 0.1).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_vllm_embedding_batch() {
        let mock_server = MockServer::start().await;

        let embeddings = vec![
            vec![0.1f32, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
        ];

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(embedding_response(&embeddings)),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let mut provider = VllmEmbeddingProvider::new(make_config(&mock_server.uri()), 3);
        provider.initialize().await.unwrap();

        let result = provider.embed_batch(&["text1", "text2"]).await;
        assert!(result.is_ok(), "batch embed failed: {:?}", result.err());

        let batch = result.unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].len(), 3);
        assert_eq!(batch[1].len(), 3);
    }

    // ─── Pipeline Integration Test ───────────────────────────────────────

    #[tokio::test]
    async fn test_rag_pipeline_with_vllm_generation() {
        use graphrag_core::generation::{AnswerGenerator, GenerationConfig};

        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(chat_response(
                    "Machine learning is a subset of AI that enables systems to learn from data.",
                )),
            )
            .mount(&mock_server)
            .await;

        let config = make_config(&mock_server.uri());
        let adapter = graphrag_core::vllm::VllmLLMAdapter::new(config);

        let gen_config = GenerationConfig::default();
        let generator = AnswerGenerator::new(Box::new(adapter), gen_config);
        assert!(
            generator.is_ok(),
            "AnswerGenerator creation should succeed"
        );
    }

    // ─── Think Tag Stripping Test ────────────────────────────────────────

    #[tokio::test]
    async fn test_vllm_generator_strips_think_tags() {
        let mock_server = MockServer::start().await;

        let think_response = "<think>Let me reason about this...</think>The answer is 42.";

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(chat_response(think_response)),
            )
            .expect(1)
            .mount(&mock_server)
            .await;

        let gen = AsyncVllmGenerator::new(make_config(&mock_server.uri()));
        let result = gen.complete("question").await;

        assert!(result.is_ok());
        let answer = result.unwrap();
        assert!(
            !answer.contains("<think>"),
            "think tags should be stripped, got: {answer}"
        );
        assert!(
            answer.contains("The answer is 42"),
            "content after think tags should remain, got: {answer}"
        );
    }
}
