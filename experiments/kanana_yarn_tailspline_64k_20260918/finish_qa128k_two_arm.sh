#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
root=${KANANA_EXPERIMENT_ROOT:-/root/autodl-tmp/today_rope_plan_20260914/kanana_yarn_tailspline_64k_20260918}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}
qa=${root}/qa128k

# Wait on the same lock as the active queue. This is a deterministic finalizer,
# not a polling monitor; it starts only after both active arms release the GPU.
exec 8>"${gpu_lock}"
flock 8

for arm in tailspline official_yarn; do
  "${python_bin}" - "${qa}/runs/${arm}/status.json" <<'PY'
import json,sys
if json.load(open(sys.argv[1])) != {"status":"COMPLETE","rows":118,"lm_rows":0}:
    raise SystemExit(f"incomplete arm: {sys.argv[1]}")
PY
done
[[ -s ${qa}/SKIP_MRPRO ]] || { printf 'REFUSE: missing user skip receipt\n' >&2; exit 1; }
[[ ! -e ${qa}/runs/mrpro/generations.jsonl ]] || {
  printf 'REFUSE: MrRoPE unexpectedly started\n' >&2
  exit 1
}

cd "${repo}"
export PYTHONPATH=.
"${python_bin}" -m experiments.kanana_yarn_tailspline_64k_20260918.report_qa_two_arm \
  --panel "${qa}/assets/inputs.jsonl" \
  --tailspline "${qa}/runs/tailspline/generations.jsonl" \
  --official-yarn "${qa}/runs/official_yarn/generations.jsonl" \
  --out "${qa}/reports/two_arm.json"
printf 'KANANA_128K_QA_TWO_ARM_COMPLETE %s\n' "$(date -u +%FT%TZ)"
