#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
root=${KANANA_EXPERIMENT_ROOT:-${plan}/kanana_yarn_tailspline_64k_20260918}
model=${KANANA_MODEL:-/root/autodl-tmp/models/kakaocorp/kanana-1.5-8b-instruct-2505}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
gpu_lock=${GPU_LOCK_PATH:-/tmp/hybrid-rope-gpu0.lock}
threshold=${PILOT_EXPAND_THRESHOLD:-0.10}

if [[ ${1:-} != --execute ]]; then
  printf 'PLAN_ONLY model=%s target=65536 pilot=niah_multiquery,vt rows=20 arms=official_yarn,tailspline threshold=%s\n' "${model}" "${threshold}"
  exit 0
fi
if [[ ! -f ${root}/ready.json ]]; then
  printf 'REFUSE: CPU preparation is incomplete: %s/ready.json\n' "${root}" >&2
  exit 1
fi
if [[ ! -f ${model}/model-00004-of-00004.safetensors ]]; then
  printf 'REFUSE: Kanana checkpoint is incomplete\n' >&2
  exit 1
fi

exec 8>"${gpu_lock}"
if ! flock -n 8; then
  printf 'REFUSE: another process owns %s\n' "${gpu_lock}" >&2
  exit 73
fi
total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ ${total_mib} -lt 30000 ]]; then
  printf 'REFUSE: Kanana BF16 64K requires a 32GB-class GPU; found %s MiB\n' "${total_mib}" >&2
  exit 75
fi

cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${KANANA_OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${KANANA_MKL_NUM_THREADS:-8}
mkdir -p "${root}/runs" "${root}/logs" "${root}/reports"

run_arm() {
  local arm=$1 panel_name=$2
  local panel=${root}/assets/${panel_name}/inputs.jsonl
  local out=${root}/runs/${arm}_${panel_name}
  local expected_rows=20
  if [[ ${panel_name} == rest11 ]]; then expected_rows=110; fi
  if [[ -f ${out}/status.json ]] && "${python_bin}" - "${out}/status.json" "${expected_rows}" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))=={
    "status":"COMPLETE","rows":int(sys.argv[2]),"lm_rows":0} else 1)
PY
  then
    return
  fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${root}/minimal_eval_manifest.json" \
    --model "${model}" --arm Native \
    --extra-panel "${panel}" --only-extra-panels --skip-lm \
    --length-cap 65536 --batch-size 1 --unmasked-unpadded-generate \
    --static-table-json "${root}/tables/${arm}.json" \
    --table-label "kanana_64k_${arm}" --out "${out}" --execute \
    >"${root}/logs/${arm}_${panel_name}.log" 2>&1
}

# Both pilot arms always run on exactly the same 20 frozen prompts.
run_arm official_yarn pilot2
run_arm tailspline pilot2
"${python_bin}" -m experiments.kanana_yarn_tailspline_64k_20260918.report \
  --mode pilot \
  --tailspline "${root}/runs/tailspline_pilot2/generations.jsonl" \
  --official-yarn "${root}/runs/official_yarn_pilot2/generations.jsonl" \
  --expand-threshold "${threshold}" \
  --out "${root}/reports/pilot2.json"

expand=$("${python_bin}" - "${root}/reports/pilot2.json" <<'PY'
import json,sys
print("1" if json.load(open(sys.argv[1]))["gate"]["expand_to_full13"] else "0")
PY
)
if [[ ${expand} != 1 ]]; then
  printf 'KANANA_64K_PILOT_COMPLETE_DIRECTIONALLY_LARGE; Full-13 not started\n'
  exit 0
fi

# Continue only the remaining eleven tasks; the pilot rows are reused verbatim.
run_arm official_yarn rest11
run_arm tailspline rest11
"${python_bin}" -m experiments.kanana_yarn_tailspline_64k_20260918.report \
  --mode full \
  --tailspline "${root}/runs/tailspline_pilot2/generations.jsonl" \
  --tailspline "${root}/runs/tailspline_rest11/generations.jsonl" \
  --official-yarn "${root}/runs/official_yarn_pilot2/generations.jsonl" \
  --official-yarn "${root}/runs/official_yarn_rest11/generations.jsonl" \
  --expand-threshold "${threshold}" \
  --out "${root}/reports/full13x10.json"
printf 'KANANA_64K_FULL13X10_COMPLETE %s\n' "$(date -u +%FT%TZ)"
