#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
classic=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic
root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic_strong_baselines
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct

mkdir -p "${root}/tables" "${root}/runs" "${root}/logs" "${root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for arm in yarn bm; do
  table=${root}/tables/${arm}.json
  if [[ ! -f "${table}" ]]; then
    /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
      --config "${model}/config.json" \
      --method "${arm}" \
      --scale 4 \
      --candidate-id "llama3_8b_s4_classic_${arm}" \
      --model-id meta_llama3_8b_instruct \
      --role baseline \
      --changed-variable static_frequency_allocation \
      --out "${table}"
  fi
  run=${root}/runs/${arm}
  if [[ -f "${run}/status.json" ]] && /root/miniconda3/bin/python - "${run}/status.json" <<'PY'
import json
import sys
raise SystemExit(json.load(open(sys.argv[1])) != {"status": "COMPLETE", "rows": 390, "lm_rows": 138})
PY
  then
    printf 'SKIP_COMPLETE %s\n' "${arm}"
    continue
  fi
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" \
    --model "${model}" \
    --arm Native \
    --extra-panel "${classic}/assets/full13/rows.jsonl" \
    --only-extra-panels \
    --length-cap 8192 --length-cap 16384 --length-cap 32768 \
    --lm-length-cap 8192 --lm-length-cap 16384 --lm-length-cap 32768 \
    --prefill-chunk-size 8192 \
    --batch-size 2 \
    --static-table-json "${table}" \
    --table-label "llama3_8b_s4_classic_${arm}" \
    --out "${run}" \
    --execute >"${root}/logs/${arm}.log" 2>&1
done

report=${root}/reports/tailspline_vs_mrpro_yarn_bm_classic.json
if [[ ! -f "${report}" ]]; then
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.tailspline_llama_classic_report \
    --run "tailspline=${classic}/runs/tailspline" \
    --run "mrpro=${classic}/runs/mrpro" \
    --run "yarn=${root}/runs/yarn" \
    --run "bm=${root}/runs/bm" \
    --receipt "tailspline=${classic}/tables/tailspline.json" \
    --receipt "mrpro=${classic}/tables/mrpro.json" \
    --receipt "yarn=${root}/tables/yarn.json" \
    --receipt "bm=${root}/tables/bm.json" \
    --ppl-manifest "${classic}/assets/ppl46/manifest.json" \
    --candidate tailspline \
    --baseline mrpro --baseline yarn --baseline bm \
    --out "${report}"
fi
printf 'CLASSIC_STRONG_BASELINES_COMPLETE %s\n' "$(date -u +%FT%TZ)"
