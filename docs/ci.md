# CI Architecture

This repository uses a three-layer CI strategy with a hybrid GitHub status lane.

## Layer 1: Local Pre-Commit

- Hook path: `.githooks/pre-commit`
- Runs:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Installation:
  - `./.githooks/install.sh`

## Layer 2: Full Local CI

- Command shortcuts are defined in `justfile`
- Main command:
  - `just ci`
- Equivalent to:
  - fmt + clippy + tests + benches compile + docs build

## Layer 3: Self-Hosted Polling Runner

- Runner script: `ci/runner.sh`
- Systemd unit: `ci/oxidizedrag-ci.service`
- Behavior:
  - Polls `origin/develop`
  - Fast-forwards local checkout to latest commit
  - Runs `nix flake check`
  - Optionally pushes successful build outputs to Attic cache

## Hybrid GitHub Lane

- Workflow: `.github/workflows/ci.yml`
- Runs `nix flake check` on pushes and PRs to `develop`/`main`
- Uses DeterminateSystems `magic-nix-cache` for fast, reproducible builds

## Nix Cache (Attic)

- Cache endpoint: `https://nix-cache.stevedores.org`
- Login example:
  - `attic login stevedores https://nix-cache.stevedores.org $ATTIC_TOKEN`

For self-hosted service environments, place `ATTIC_TOKEN` in an env file loaded by systemd.
