#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/engineering/code/oxidizedRAG}"
BRANCH="${BRANCH:-develop}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "$REPO_DIR"

echo "[runner] starting for branch=$BRANCH poll=${POLL_SECONDS}s"

while true; do
  git fetch origin "$BRANCH"

  LOCAL_SHA="$(git rev-parse HEAD)"
  REMOTE_SHA="$(git rev-parse "origin/$BRANCH")"

  if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
    echo "[runner] new commit detected: $LOCAL_SHA -> $REMOTE_SHA"
    git checkout "$BRANCH"
    git reset --hard "origin/$BRANCH"

    if nix flake check --print-build-logs; then
      echo "[runner] checks passed"

      if command -v attic >/dev/null 2>&1 && [[ -n "${ATTIC_TOKEN:-}" ]]; then
        echo "[runner] pushing closure to attic cache"
        attic login stevedores https://nix-cache.stevedores.org "$ATTIC_TOKEN" || true
        path="$(nix build .#default --print-out-paths --no-link | tail -n 1)"
        attic push stevedores "$path" || true
      fi
    else
      echo "[runner] checks failed"
    fi
  fi

  sleep "$POLL_SECONDS"
done
