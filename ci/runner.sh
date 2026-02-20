#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/engineering/code/oxidizedRAG}"
BRANCH="${BRANCH:-develop}"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() {
  printf "[%s] [runner] %s\n" "$(date +"%Y-%m-%d %H:%M:%S")" "$*"
}

cd "$REPO_DIR"

log "starting for branch=$BRANCH poll=${POLL_SECONDS}s"

while true; do
  git fetch origin "$BRANCH" || {
    log "git fetch failed for branch=$BRANCH"
    sleep "$POLL_SECONDS"
    continue
  }

  LOCAL_SHA="$(git rev-parse HEAD)"
  REMOTE_SHA="$(git rev-parse "origin/$BRANCH")"

  if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
    log "new commit detected: $LOCAL_SHA -> $REMOTE_SHA"

    if [[ -n "$(git status --porcelain)" ]]; then
      log "refusing to run on dirty working tree"
      sleep "$POLL_SECONDS"
      continue
    fi

    git checkout "$BRANCH" || {
      log "git checkout failed for branch=$BRANCH"
      sleep "$POLL_SECONDS"
      continue
    }

    git merge --ff-only "origin/$BRANCH" || {
      log "git fast-forward failed; refusing to mutate repository state"
      sleep "$POLL_SECONDS"
      continue
    }

    if nix flake check --print-build-logs; then
      log "checks passed"

      if command -v attic >/dev/null 2>&1 && [[ -n "${ATTIC_TOKEN:-}" ]]; then
        if ! attic cache info stevedores >/dev/null 2>&1; then
          log "no cached attic auth; logging in"
          attic login stevedores https://nix-cache.stevedores.org "$ATTIC_TOKEN" || {
            log "attic login failed; skipping cache push"
            sleep "$POLL_SECONDS"
            continue
          }
        fi

        log "pushing closure to attic cache"
        path="$(nix build .#default --print-out-paths --no-link | tail -n 1)" || {
          log "nix build failed; skipping cache push"
          sleep "$POLL_SECONDS"
          continue
        }
        attic push stevedores "$path" || log "attic push failed"
      fi
    else
      log "checks failed"
    fi
  fi

  sleep "$POLL_SECONDS"
done
