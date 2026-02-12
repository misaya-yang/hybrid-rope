#!/usr/bin/env bash
set -euo pipefail

# Commit and push any updated artifacts in this repo.
#
# Intended flow:
#   scripts/pull_a100_350m_artifacts.sh
#   scripts/commit_and_push.sh "Update 350M artifacts (auto)"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

msg="${1:-Update artifacts}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  :
fi

git add -A

if git diff --cached --quiet; then
  echo "[git] nothing to commit"
  exit 0
fi

git commit -m "$msg"
git push
echo "[git] pushed"

