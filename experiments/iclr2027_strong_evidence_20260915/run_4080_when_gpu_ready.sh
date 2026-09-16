#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
poll_seconds=${GPU_POLL_SECONDS:-20}
root=${plan}/official_yarn_4080_light
mkdir -p "${root}/logs"

while [[ -z "$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d;/No devices/d' | head -1 || true)" ]]; do
  sleep "${poll_seconds}"
done

cd "${repo}"
exec bash experiments/iclr2027_strong_evidence_20260915/run_4080_light_yarn_queue.sh --execute
