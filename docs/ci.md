# CI Architecture

This repository uses a three-layer CI strategy with a hybrid GitHub status lane.

## Layer 1: Local Pre-Commit

- Hook path: `.githooks/pre-commit`
- Priority order (first available tool is used):
  1. **local-ci** (unified CI pipeline, if installed)
  2. **Nix** (via `nix develop`)
  3. **Cargo** (bare local installation)
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

### Using local-ci in pre-commit

To use unified CI pipeline in pre-commit hooks (optional):

1. Install local-ci:
   - `just local-ci-install`
   - Or manually: `git clone https://github.com/stevedores-org/local-ci && cd local-ci && make build`
2. Hook will automatically detect and use it
3. Check hook behavior:
   - `bash -x .githooks/pre-commit` (debug mode)

## Layer 2: Full Local CI (Optional: Unified via local-ci)

### Option A: Unified pipeline with local-ci (Recommended)

- Tool: `local-ci` (from https://github.com/stevedores-org/local-ci)
- Configuration: `.local-ci.toml`
- Install:
  - `just local-ci-install`
- Run full pipeline:
  - `just local-ci` (or `local-ci`)
- Run fix mode:
  - `just local-ci-fix` (or `local-ci --fix`)
- Run selected stages:
  - `local-ci fmt clippy`
  - `local-ci test`
- Benefits:
  - Unified configuration across all tools
  - Fast cached stage runs
  - Consistent output formatting
  - Built-in caching strategy
- Current limitation:
  - `local-ci` currently uses built-in stage definitions (`fmt`, `clippy`, `test`, `check`)
  - `.local-ci.toml` is forward-compatible policy documentation and not enforced yet by the binary

### Option B: Traditional direct commands

- Command shortcuts defined in `justfile`
- Main command:
  - `just ci`
- Equivalent to:
  - fmt + clippy + tests + benches compile + docs build

Useful commands:
- `just fmt` — Format code
- `just fmt-check` — Check formatting
- `just clippy` — Run clipper linter
- `just test` — Run tests
- `just bench` — Compile benches
- `just doc` — Build docs
- `just check` — Quick cargo check

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

## Local-CI Configuration

The `.local-ci.toml` file defines a unified pipeline with the following stages:

`local-ci` currently executes built-in stages and does not load this file yet.
Treat `.local-ci.toml` as desired policy documentation for upcoming schema-backed config support.

## Troubleshooting

### local-ci issues
- local-ci not found:
  - Install: `just local-ci-install`
  - Or manually build: `git clone https://github.com/stevedores-org/local-ci && make build`
  - Add to PATH or ensure GOPATH/bin is in PATH
- Stage fails to run:
  - Check configuration: `cat .local-ci.toml`
  - Run with verbose output: `local-ci --verbose fmt clippy test`
  - Verify tools are installed: `cargo audit --version`, `cargo deny --version`

### Pre-commit hook issues
- Hook not running:
  - Re-run: `./.githooks/install.sh`
  - Verify: `git config core.hooksPath`
- Hook runs wrong tool:
  - Check tool priority: local-ci > nix > cargo
  - Debug: `bash -x .githooks/pre-commit`

### General CI issues
- Runner refuses to update branch:
  - Check for dirty working tree: `git status --short`
  - Runner intentionally refuses destructive updates on non-clean repos.
- `nix flake check` fails:
  - Reproduce locally: `just flake-check`
- Attic push skipped:
  - Confirm `ATTIC_TOKEN` is set in `/etc/oxidizedrag-ci.env`
  - Validate auth: `attic cache info stevedores`
