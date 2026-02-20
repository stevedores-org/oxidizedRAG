set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
  @just --list

hooks-install:
  ./.githooks/install.sh

fmt:
  cargo fmt --all

fmt-check:
  cargo fmt --all -- --check

clippy:
  cargo clippy --workspace --all-targets -- -D warnings

test:
  cargo test --workspace

bench:
  cargo test --workspace --benches --no-run

doc:
  cargo doc --workspace --no-deps

check:
  cargo check --workspace

ci:
  just fmt-check
  just clippy
  just test
  just bench
  just doc

flake-check:
  nix flake check --print-build-logs
