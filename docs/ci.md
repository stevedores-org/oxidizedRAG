# CI Architecture

This repository uses a three-layer CI strategy with a hybrid GitHub status lane.

## Layer 1: Local Pre-Commit

- Hook path: `.githooks/pre-commit`
- Runs:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
- Installation:
  - `./.githooks/install.sh`

### Hook setup

1. Run `./.githooks/install.sh`.
2. Confirm hook path:
   - `git config core.hooksPath`
3. Validate hook script:
   - `bash -n .githooks/pre-commit`

## Layer 2: Full Local CI

- Command shortcuts are defined in `justfile`
- Main command:
  - `just ci`
- Equivalent to:
  - fmt + clippy + tests + benches compile + docs build

Useful commands:
- `just fmt`
- `just fmt-check`
- `just clippy`
- `just test`
- `just bench`
- `just doc`

## Layer 3: Self-Hosted Polling Runner

- Runner script: `ci/runner.sh`
- Systemd unit: `ci/oxidizedrag-ci.service`
- Behavior:
  - Polls `origin/develop`
  - Fast-forwards local checkout to latest commit
  - Runs `nix flake check`
  - Optionally pushes successful build outputs to Attic cache

### Systemd setup

1. Copy service unit to systemd:
   - `sudo cp ci/oxidizedrag-ci.service /etc/systemd/system/`
2. Provide environment file:
   - `sudo tee /etc/oxidizedrag-ci.env`
3. Add token value:
   - `ATTIC_TOKEN=...`
4. Enable and start:
   - `sudo systemctl daemon-reload`
   - `sudo systemctl enable --now oxidizedrag-ci.service`
5. Verify logs:
   - `journalctl -u oxidizedrag-ci.service -f`

## Hybrid GitHub Lane

- Workflow: `.github/workflows/ci.yml`
- Runs `nix flake check` on pushes and PRs to `develop`/`main`
- Uses DeterminateSystems `magic-nix-cache` for fast, reproducible builds
- Also runs an explicit `cargo test --workspace` inside `nix develop`

## Nix Cache (Attic)

- Cache endpoint: `https://nix-cache.stevedores.org`
- Login example:
  - `attic login stevedores https://nix-cache.stevedores.org $ATTIC_TOKEN`

For self-hosted service environments, place `ATTIC_TOKEN` in an env file loaded by systemd.

## Troubleshooting

- Runner refuses to update branch:
  - Check for dirty working tree (`git status --short`).
  - Runner intentionally refuses destructive updates on non-clean repos.
- `nix flake check` fails:
  - Reproduce locally with `just flake-check`.
- Hook not running:
  - Re-run `./.githooks/install.sh` and verify `core.hooksPath`.
- Attic push skipped:
  - Confirm `ATTIC_TOKEN` is set in `/etc/oxidizedrag-ci.env`.
  - Validate auth manually: `attic cache info stevedores`.
