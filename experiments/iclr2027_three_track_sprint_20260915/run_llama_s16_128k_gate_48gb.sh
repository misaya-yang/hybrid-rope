#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/tailspline_llama_s16_128k_gate
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
python_bin=/root/miniconda3/bin/python

mkdir -p "${root}/runs" "${root}/logs" "${root}/reports"
cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ "${total_mib}" -lt 45000 ]]; then
  printf 'REFUSE: Llama S16 128K gate requires >=45,000 MiB VRAM; found %s MiB\n' "${total_mib}" >&2
  exit 75
fi
if [[ ! -f "${root}/assets/ready.json" ]]; then
  printf 'REFUSE: run prepare_llama_s16_128k_assets.sh before GPU execution\n' >&2
  exit 1
fi

run_arm() {
  local arm=$1 run=${root}/runs/$1
  if [[ -f "${run}/status.json" ]] && "${python_bin}" - "${run}/status.json" <<'PY'
import json,sys
raise SystemExit(json.load(open(sys.argv[1]))!={"status":"COMPLETE","rows":130,"lm_rows":10})
PY
  then return; fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${root}/assets/ppl10/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${root}/assets/full13/inputs.jsonl" --only-extra-panels \
    --length-cap 131072 --lm-length-cap 131072 --prefill-chunk-size 8192 --batch-size 1 \
    --static-table-json "${root}/tables/${arm}.json" \
    --table-label "llama3_8b_s16_128k_${arm}" --out "${run}" --execute \
    >"${root}/logs/${arm}.log" 2>&1
}
run_arm tailspline
run_arm mrpro

"${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.llama_s16_128k_report \
  --run "tailspline=${root}/runs/tailspline" --run "mrpro=${root}/runs/mrpro" \
  --out "${root}/reports/tailspline_vs_mrpro_s16_128k_gate.json"
printf 'LLAMA_S16_128K_GATE_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/complete.txt"
