#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_naturalqa631
base_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
panel=${root}/assets/inputs.jsonl

mkdir -p "${root}/assets" "${root}/runs" "${root}/logs" "${root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.prepare_tailspline_llama_naturalqa631 \
  --frozen-root /root/autodl-tmp/olmo_fast_screen_20260908 \
  --download-root /root/autodl-tmp/hybrid-rope-target-free-real-data-v3 \
  --model "${model}" \
  --out "${root}/assets"

for arm in tailspline mrpro; do
  run_dir=${root}/runs/${arm}
  if [[ -f "${run_dir}/status.json" ]] && /root/miniconda3/bin/python - "${run_dir}/status.json" <<'PY'
import json
import sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 631, "lm_rows": 0})
PY
  then
    printf 'SKIP_COMPLETE %s\n' "${arm}"
    continue
  fi
  printf 'START %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${base_root}/assets/ppl46/manifest.json" \
    --model "${model}" \
    --arm Native \
    --extra-panel "${panel}" \
    --only-extra-panels \
    --skip-lm \
    --length-cap 32768 \
    --prefill-chunk-size 8192 \
    --batch-size 1 \
    --static-table-json "${base_root}/tables/${arm}.json" \
    --table-label "llama3_8b_s4_naturalqa631_${arm}" \
    --out "${run_dir}" \
    --execute >"${root}/logs/${arm}.log" 2>&1
  printf 'COMPLETE %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
done

/root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.matched_naturalqa_report \
  --panel "${panel}" \
  --candidate "${root}/runs/tailspline/generations.jsonl" \
  --baseline "${root}/runs/mrpro/generations.jsonl" \
  --out "${root}/reports/tailspline_vs_mrpro_naturalqa631.json"

printf 'NATURAL_QA_COMPLETE %s\n' "$(date -u +%FT%TZ)"
