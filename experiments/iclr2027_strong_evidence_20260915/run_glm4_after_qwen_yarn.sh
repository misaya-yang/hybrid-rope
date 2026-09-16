#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
upstream_pid=${UPSTREAM_PID:?set UPSTREAM_PID to the official YaRN evaluator}
yarn=${plan}/four_model_128k_extreme/qwen25_3b_128k/official_yarn/run

while [[ -d "/proc/${upstream_pid}" ]]; do
  state=$(ps -o stat= -p "${upstream_pid}" 2>/dev/null | tr -d ' ' || true)
  command=$(ps -o cmd= -p "${upstream_pid}" 2>/dev/null || true)
  [[ -z "${state}" || "${state}" == Z* ]] && break
  [[ "${command}" == *qwen25_3b_s4_official_static_yarn* ]] || { echo "REFUSE: upstream PID reused" >&2; exit 74; }
  sleep 30
done

cd "${repo}"
/root/miniconda3/bin/python - "${yarn}/status.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d!={'status':'COMPLETE','rows':40,'lm_rows':5}: raise SystemExit(f'YaRN incomplete: {d}')
PY
bash experiments/iclr2027_strong_evidence_20260915/run_glm4_9b_s4_128k_queue.sh
