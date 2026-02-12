#!/usr/bin/env bash
set -euo pipefail

# Pull non-weight artifacts from the A100 run back into this repo.
#
# Usage:
#   scripts/pull_a100_350m_artifacts.sh
#
# Optional:
#   export A100_HOST="root@117.50.192.217"
#   export A100_PORT="23"
#
# Authentication:
# - Prefer SSH keys / ssh-agent.
# - If you must use a password, export A100_SSH_PASS and install sshpass.
#   (We intentionally do NOT store passwords in this repo.)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

A100_HOST="${A100_HOST:-root@117.50.192.217}"
A100_PORT="${A100_PORT:-23}"

REMOTE_BASE="/opt/dfrope/results/350m_final"
LOCAL_BASE="${ROOT}/results/350m_final"

mkdir -p "${LOCAL_BASE}"

scp_cmd=(scp -P "${A100_PORT}" -o StrictHostKeyChecking=no)
ssh_cmd=(ssh -p "${A100_PORT}" -o StrictHostKeyChecking=no)

if [[ -n "${A100_SSH_PASS:-}" ]]; then
  if command -v sshpass >/dev/null 2>&1; then
    scp_cmd=(sshpass -p "${A100_SSH_PASS}" "${scp_cmd[@]}")
    ssh_cmd=(sshpass -p "${A100_SSH_PASS}" "${ssh_cmd[@]}")
  else
    echo "A100_SSH_PASS is set but sshpass is not installed." >&2
    exit 2
  fi
fi

echo "[pull] ${A100_HOST}:${REMOTE_BASE} -> ${LOCAL_BASE}"

# Always pull run.log (it grows over time).
("${scp_cmd[@]}" "${A100_HOST}:${REMOTE_BASE}/run.log" "${LOCAL_BASE}/run.log" 2>/dev/null) || true

# Pull results.json if it exists.
("${scp_cmd[@]}" "${A100_HOST}:${REMOTE_BASE}/results.json" "${LOCAL_BASE}/results.json" 2>/dev/null) || true

# Pull small cache metadata files (never the memmap token blobs).
("${ssh_cmd[@]}" "${A100_HOST}" "ls -1 ${REMOTE_BASE}/cache/*.meta.json 2>/dev/null" > /tmp/a100_meta_list.txt 2>/dev/null) || true
if [[ -s /tmp/a100_meta_list.txt ]]; then
  while IFS= read -r f; do
    base="$(basename "$f")"
    ("${scp_cmd[@]}" "${A100_HOST}:${f}" "${LOCAL_BASE}/${base}" 2>/dev/null) || true
  done < /tmp/a100_meta_list.txt
fi

echo "[pull] done"

