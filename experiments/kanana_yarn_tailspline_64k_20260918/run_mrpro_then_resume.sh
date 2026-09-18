#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
root=${KANANA_EXPERIMENT_ROOT:-${plan}/kanana_yarn_tailspline_64k_20260918}
model=${KANANA_MODEL:-/root/autodl-tmp/models/kakaocorp/kanana-1.5-8b-instruct-2505}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}

cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${KANANA_OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${KANANA_MKL_NUM_THREADS:-8}
mkdir -p "${root}/runs" "${root}/logs" "${root}/reports"

"${python_bin}" -m experiments.kanana_yarn_tailspline_64k_20260918.prepare_mrpro \
  --model "${model}" --root "${root}"

(
  exec 8>"${gpu_lock}"
  if ! flock -n 8; then
    printf 'REFUSE: another process owns %s\n' "${gpu_lock}" >&2
    exit 73
  fi
  out=${root}/runs/mrpro_pilot2
  complete=0
  if [[ -f ${out}/status.json ]]; then
    complete=$("${python_bin}" - "${out}/status.json" <<'PY'
import json,sys
print(1 if json.load(open(sys.argv[1]))=={"status":"COMPLETE","rows":20,"lm_rows":0} else 0)
PY
    )
  fi
  if [[ ${complete} != 1 ]]; then
    "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
      --data "${root}/minimal_eval_manifest.json" \
      --model "${model}" --arm Native \
      --extra-panel "${root}/assets/pilot2/inputs.jsonl" \
      --only-extra-panels --skip-lm --length-cap 65536 \
      --batch-size 1 --unmasked-unpadded-generate \
      --static-table-json "${root}/tables/mrpro.json" \
      --table-label kanana_64k_mrpro_s2 \
      --out "${out}" --execute >"${root}/logs/mrpro_pilot2.log" 2>&1
  fi
)

"${python_bin}" -m experiments.kanana_yarn_tailspline_64k_20260918.report_three_arm_pilot \
  --tailspline "${root}/runs/tailspline_pilot2/generations.jsonl" \
  --official-yarn "${root}/runs/official_yarn_pilot2/generations.jsonl" \
  --mrpro "${root}/runs/mrpro_pilot2/generations.jsonl" \
  --out "${root}/reports/pilot2_three_arm.json"

# Resume only the YaRN/TailSpline Full-13 continuation requested by the user.
PILOT_EXPAND_THRESHOLD=0.11 \
  bash experiments/kanana_yarn_tailspline_64k_20260918/run_server.sh --execute
