# CLAUDE.md — oxidizedRAG

## What is this?

High-performance Rust GraphRAG engine. Multi-crate workspace with WASM support.

## Build & check

```bash
# Preferred: use Nix dev shell (provides all tools)
nix develop    # or: nix --extra-experimental-features 'nix-command flakes' develop

# Inside dev shell (or with tools installed):
just fmt                    # cargo fmt --check (uses nightly rustfmt)
just clippy                 # cargo clippy --workspace --all-targets -- -D warnings
just test                   # cargo test --workspace
just bench                  # compile benchmarks
just doc                    # build docs
just ci                     # full local CI (all of the above)
just flake-check            # nix flake check

# Without just:
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Workspace crates

| Crate | Purpose |
|-------|---------|
| `graphrag-core` | Core library — graph construction, retrieval, NLP, caching |
| `graphrag-server` | HTTP API server (axum) |
| `graphrag-cli` | TUI terminal client (ratatui) |
| `graphrag-wasm` | WASM bindings (leptos, wasm-bindgen) |
| `graphrag-aivcs` | AIVCS adapter for persistence & versioning |

## Key conventions

- **Always target `develop`** for PRs — `main` is releases only
- **Nightly rustfmt** required — `rustfmt.toml` uses nightly-only options (`imports_granularity`, `wrap_comments`, etc.)
- **`-D warnings`** enforced — clippy warnings are errors in CI
- **CI runs on x86_64-linux** via `nix flake check` in GitHub Actions
- **Attic binary cache** at `nix-cache.stevedores.org` speeds up CI builds

## Formatting

This repo uses **nightly rustfmt**. The Nix dev shell provides it. Outside the dev shell:

```bash
rustup run nightly cargo fmt --all          # format
rustup run nightly cargo fmt --all -- --check  # check only
```

If rustfmt complains about "unstable features only available in nightly channel", you're running stable rustfmt. Use the nightly toolchain.

## Testing

```bash
cargo test --workspace                      # all tests
cargo test -p graphrag-core                 # single crate
cargo test -p graphrag-core -- test_name    # single test
cargo nextest run --workspace               # parallel (if installed)
```

## Common issues

| Issue | Fix |
|-------|-----|
| `unstable features are only available in nightly channel` | Use nightly rustfmt: `rustup run nightly cargo fmt` |
| `darwin.apple_sdk_11_0 has been removed` | macOS nixpkgs compat issue — CI runs Linux, unaffected |
| `failed to resolve mod neural` | `graphrag-core/src/embeddings/neural.rs` stub must exist |
| `qdrant-client build.rs PermissionDenied` | Nix sandbox issue — `doNotLinkInheritedArtifacts = true` in flake.nix |

## Architecture notes

- **Trait-based design**: core traits in `graphrag-core/src/core/traits.rs`
- **Pipeline system**: `graphrag-core/src/pipeline/` — staged processing with caching
- **LightRAG**: `graphrag-core/src/lightrag/` — concept graphs, dual retrieval, lazy pipelines
- **RoGRAG**: `graphrag-core/src/rograg/` — reasoning-oriented retrieval
- **Embeddings**: supports HuggingFace, OpenAI, Voyage AI, Cohere, ONNX, Candle
- **Storage**: SQLite, SurrealDB, LanceDB, Qdrant, Voy (WASM)
