#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/tailspline_llama_s4_16k_ruler50_clean
classic=${plan}/tailspline_llama_s4_classic
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
panel=${root}/assets/inputs.jsonl
python_bin=/root/miniconda3/bin/python

mkdir -p "${root}/runs" "${root}/logs" "${root}/reports"
cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

experiments/iclr2027_three_track_sprint_20260915/prepare_clean16k_ruler50.sh \
  >"${root}/logs/prepare.log" 2>&1

for arm in tailspline mrpro; do
  run=${root}/runs/${arm}
  if [[ -f "${run}/status.json" ]] && "${python_bin}" - "${run}/status.json" <<'PY'
import json, sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 650, "lm_rows": 0})
PY
  then
    continue
  fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${panel}" --only-extra-panels --skip-lm --length-cap 16384 \
    --prefill-chunk-size 8192 --batch-size 1 \
    --static-table-json "${classic}/tables/${arm}.json" \
    --table-label "llama3_8b_s4_16k_ruler50_clean_${arm}" \
    --out "${run}" --execute >"${root}/logs/${arm}.log" 2>&1
done

report=${root}/reports/tailspline_vs_mrpro_full13_16k_50_per_task_clean.json
if [[ ! -f "${report}" ]]; then
  "${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source "tailspline=${root}/runs/tailspline/generations.jsonl" \
    --source "mrpro=${root}/runs/mrpro/generations.jsonl" \
    --candidate tailspline --baseline mrpro --length 16384 --out "${report}"
fi

printf 'CLEAN16K_RULER50_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/complete.txt"
