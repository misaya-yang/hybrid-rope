#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/tailspline_llama_s4_mrrope_niah_heatmap_full20
classic=${plan}/tailspline_llama_s4_classic
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
upstream=/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
python_bin=/root/miniconda3/bin/python

mkdir -p "${root}/assets" "${root}/runs" "${root}/logs" "${root}/reports"
cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.prepare_mrrope_niah_heatmap \
  --model "${model}" \
  --paul-graham-json "${upstream}/scripts/data/synthetic/json/PaulGrahamEssays.json" \
  --repeats-per-cell 20 --seed 20261015 --out "${root}/assets"

run_arm() {
  local arm=$1 run=${root}/runs/$1
  if [[ -f "${run}/status.json" ]]; then
    if "${python_bin}" - "${run}/status.json" "${run}/contract.json" \
      "${classic}/tables/${arm}.json" <<'PY'
import json,sys
status=json.load(open(sys.argv[1]));contract=json.load(open(sys.argv[2]));receipt=json.load(open(sys.argv[3]))
table=receipt.get('table',receipt);active=contract.get('static_table') or {}
valid=(status=={"status":"COMPLETE","rows":720,"lm_rows":0}
       and contract.get('batch_size')==1
       and contract.get('prefill_chunk_size')==0
       and contract.get('generation_length_caps')==[8192,16384,24576,32768]
       and active.get('values_float32')==table.get('values_float32')
       and active.get('gain')==table.get('gain'))
raise SystemExit(0 if valid else 1)
PY
    then return; fi
    printf 'REFUSE: completed %s NIAH arm has a different runtime/table contract\n' "${arm}" >&2
    exit 1
  fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${root}/assets/inputs.jsonl" --only-extra-panels --skip-lm \
    --length-cap 8192 --length-cap 16384 --length-cap 24576 --length-cap 32768 \
    --batch-size 1 --static-table-json "${classic}/tables/${arm}.json" \
    --table-label "llama3_8b_s4_mrrope_niah_heatmap_full20_${arm}" \
    --out "${run}" --execute >"${root}/logs/${arm}.log" 2>&1
}

run_arm tailspline
run_arm mrpro

"${python_bin}" -m experiments.iclr2027_three_track_sprint_20260915.mrrope_niah_heatmap_report \
  --manifest "${root}/assets/manifest.json" --panel "${root}/assets/inputs.jsonl" \
  --run "tailspline=${root}/runs/tailspline" --run "mrpro=${root}/runs/mrpro" \
  --out "${root}/reports/tailspline_vs_mrpro_niah_heatmap_full20.json" \
  --plot-prefix "${root}/reports/tailspline_vs_mrpro_niah_heatmap_full20"

printf 'MRROPE_NIAH_HEATMAP_FULL20_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/complete.txt"
