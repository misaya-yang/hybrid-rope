#!/usr/bin/env bash
set -euo pipefail
repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
upstream_pid=${UPSTREAM_PID:?set UPSTREAM_PID to the quick YaRN supervisor}
while [[ -d "/proc/${upstream_pid}" ]]; do
  state=$(ps -o stat= -p "${upstream_pid}" 2>/dev/null | tr -d ' ' || true);command=$(ps -o cmd= -p "${upstream_pid}" 2>/dev/null || true)
  [[ -z "${state}" || "${state}" == Z* ]] && break
  [[ "${command}" == *run_yarn_quick_after_glm.sh* ]] || { echo "REFUSE: upstream PID reused" >&2; exit 74; }
  sleep 30
done
test -f "${plan}/official_yarn_quick/complete.json" || { echo "REFUSE: quick YaRN incomplete" >&2; exit 1; }
cd "${repo}"

# Natural QA has the highest post-quick information value: it reuses completed
# T/P rows and adds only the official-static-YaRN arm.  Keep it ahead of the
# larger Full-13 completion queue.
bash experiments/iclr2027_strong_evidence_20260915/run_four_model_naturalqa_yarn.sh --execute
test -f "${plan}/official_yarn_naturalqa_four_model_complete.json" || {
  echo "REFUSE: four-model YaRN natural QA incomplete" >&2
  exit 1
}

bash experiments/iclr2027_strong_evidence_20260915/run_four_model_yarn_full13.sh
