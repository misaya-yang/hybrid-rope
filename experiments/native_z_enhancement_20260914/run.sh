#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914/olmo_native_z5_enhancement
model=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
olmo_classic=/root/autodl-tmp/today_rope_plan_20260914/tailspline_olmo_s4_classic

mkdir -p "${root}/assets" "${root}/optimization" "${root}/runs" "${root}/logs" "${root}/reports"
cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/miniconda3/bin/python -m experiments.native_z_enhancement_20260914.prepare \
  --model "${model}" \
  --pg19-validation /root/autodl-tmp/pg19_raw/data/validation-00000-of-00001.parquet \
  --ppl-manifest "${olmo_classic}/assets/ppl46/manifest.json" \
  --ppl-array "${olmo_classic}/assets/ppl46/lm.npy" \
  --ruler-panel "${olmo_classic}/assets/full13/rows.jsonl" \
  --natural-dir /root/autodl-tmp/olmo_fast_screen_20260908/prepared_natural_01 \
  --natural-dir /root/autodl-tmp/olmo_fast_screen_20260908/prepared_natural_02 \
  --natural-dir /root/autodl-tmp/olmo_fast_screen_20260908/prepared_natural_extra_01 \
  --longbench-archive /root/autodl-tmp/hybrid-rope-target-free-real-data-v3/longbench/data.zip \
  --out "${root}/assets"

if [[ ! -f "${root}/optimization/status.json" ]]; then
  /root/miniconda3/bin/python -m experiments.native_z_enhancement_20260914.optimize \
    --assets "${root}/assets" \
    --model "${model}" \
    --out "${root}/optimization" \
    >"${root}/logs/optimize.log" 2>&1
fi

run_arm() {
  local arm=$1
  local output=${root}/runs/${arm}
  if [[ -f "${output}/status.json" ]] && /root/miniconda3/bin/python - "${output}/status.json" <<'PY'
import json
import sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 229, "lm_rows": 0})
PY
  then
    return
  fi
  args=(
    --data "${olmo_classic}/assets/ppl46/manifest.json"
    --model "${model}"
    --arm Native
    --extra-panel "${root}/assets/ruler4k_full13x10.jsonl"
    --extra-panel "${root}/assets/natural4k_three_task.jsonl"
    --only-extra-panels --skip-lm --length-cap 4096
    --prefill-chunk-size 4096 --batch-size 4
    --out "${output}" --execute
  )
  if [[ "${arm}" == native_z5 ]]; then
    args+=(--static-table-json "${root}/optimization/table.json" --table-label olmo2_1b_native_z5)
  fi
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval "${args[@]}" \
    >"${root}/logs/${arm}.log" 2>&1
}

run_arm native
run_arm native_z5

/root/miniconda3/bin/python -m experiments.native_z_enhancement_20260914.report \
  --assets "${root}/assets" \
  --optimization "${root}/optimization" \
  --native-run "${root}/runs/native" \
  --candidate-run "${root}/runs/native_z5" \
  --out "${root}/reports/native_vs_z5.json"

printf 'OLMO_NATIVE_Z5_QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/complete.txt"
