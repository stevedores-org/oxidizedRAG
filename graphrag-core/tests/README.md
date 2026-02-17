# Code Agent Test Suite

Primary suite: `graphrag-core/tests/code_agent_tests.rs`

Issue traceability:
- Issue: https://github.com/stevedores-org/oxidizedRAG/issues/2
- Coverage:
  - code indexing
  - code understanding
  - code retrieval
  - code generation
  - agent workflows
  - performance baselines

Fixtures:
- `graphrag-core/tests/fixtures/code_samples/calculator.rs`
- `graphrag-core/tests/fixtures/code_samples/api_client.rs`
- `graphrag-core/tests/fixtures/code_samples/graph_algorithms.rs`

Running:
- Core suite: `cargo test -p graphrag-core --test code_agent_tests`
- Strict perf gates: `PERF_CI=1 cargo test -p graphrag-core --test code_agent_tests performance`
