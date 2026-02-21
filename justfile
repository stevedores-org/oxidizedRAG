set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
  @just --list

# Install git hooks
hooks-install:
  ./.githooks/install.sh

# Install local-ci tool (optional, for unified CI pipeline)
local-ci-install:
  #!/usr/bin/env bash
  if command -v local-ci >/dev/null 2>&1; then
    echo "local-ci is already installed"
  else
    echo "Installing local-ci..."
    if [[ -d ../local-ci ]]; then
      cd ../local-ci && make build && mv local-ci $(go env GOPATH)/bin/ || echo "Please install from https://github.com/stevedores-org/local-ci"
    else
      echo "Please clone https://github.com/stevedores-org/local-ci and run 'make build'"
    fi
  fi

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

# Run full CI via local-ci tool (if installed)
local-ci:
  #!/usr/bin/env bash
  if command -v local-ci >/dev/null 2>&1; then
    local-ci
  else
    echo "⚠️ local-ci not installed. Run 'just local-ci-install' first, or use 'just ci' for standard pipeline"
    echo "   For setup: https://github.com/stevedores-org/local-ci"
  fi

# Run local-ci with auto-fix (formatting)
local-ci-fix:
  #!/usr/bin/env bash
  if command -v local-ci >/dev/null 2>&1; then
    local-ci --fix
  else
    echo "local-ci not installed. Run: just local-ci-install"
  fi
