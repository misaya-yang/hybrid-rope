#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
upstream_pid=${UPSTREAM_PID:?set UPSTREAM_PID to the GLM queue supervisor}

while [[ -d "/proc/${upstream_pid}" ]]; do
  state=$(ps -o stat= -p "${upstream_pid}" 2>/dev/null | tr -d ' ' || true)
  command=$(ps -o cmd= -p "${upstream_pid}" 2>/dev/null || true)
  [[ -z "${state}" || "${state}" == Z* ]] && break
  [[ "${command}" == *run_glm4_after_qwen_yarn.sh* ]] || { echo "REFUSE: upstream PID reused" >&2; exit 74; }
  sleep 30
done

test -f "${plan}/glm4_9b_s4_128k/complete.json" || { echo "REFUSE: GLM queue incomplete" >&2; exit 1; }
cd "${repo}"
bash experiments/iclr2027_strong_evidence_20260915/run_four_model_yarn_quick.sh
