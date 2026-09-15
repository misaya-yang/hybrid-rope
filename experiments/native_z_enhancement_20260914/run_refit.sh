#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914/olmo_native_z5_enhancement
model=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
refit=${root}/all50_refit

mkdir -p "${refit}" "${root}/logs"
cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${root}/complete.txt" ]]; then
  printf 'Original Native-Z5 must complete before the all50 refit\n' >&2
  exit 1
fi

/root/miniconda3/bin/python -m experiments.native_z_enhancement_20260914.refit \
  --assets "${root}/assets" --model "${model}" \
  --original-optimization "${root}/optimization" --out "${refit}" \
  >"${root}/logs/all50_refit.log" 2>&1

advance=$(/root/miniconda3/bin/python - "${refit}/refit_result.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print("1" if value["decision"]["advance_to_fresh_task_confirmation"] else "0")
PY
)
if [[ "${advance}" == 1 ]]; then
  printf 'NATIVE_Z5_REFIT_READY_FOR_FRESH_TASKS %s\n' "$(date -u +%FT%TZ)" | tee "${refit}/complete.txt"
else
  printf 'NATIVE_Z5_REFIT_STOPPED_BEFORE_TASKS %s\n' "$(date -u +%FT%TZ)" | tee "${refit}/complete.txt"
fi
