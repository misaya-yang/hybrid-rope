#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_olmo_s4_classic
model_dir=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
full13_dir="${experiment_root}/assets/full13"
ppl_dir="${experiment_root}/assets/ppl46"
table_dir="${experiment_root}/tables"

mkdir -p "${experiment_root}/runs" "${experiment_root}/logs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/prepare_tailspline_olmo_classic_assets.sh"

run_arm() {
  local arm=$1
  local output_dir="${experiment_root}/runs/${arm}"
  local status_path="${output_dir}/status.json"
  if [[ -f "${status_path}" ]] && /root/miniconda3/bin/python - "${status_path}" <<'PY'
import json
import sys
status = json.load(open(sys.argv[1]))
if status != {"status": "COMPLETE", "rows": 390, "lm_rows": 138}:
    raise SystemExit(1)
PY
  then
    printf 'SKIP_COMPLETE %s\n' "${arm}"
    return
  fi

  printf 'START %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${ppl_dir}/manifest.json" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${full13_dir}/rows.jsonl" \
    --only-extra-panels \
    --length-cap 4096 \
    --length-cap 8192 \
    --length-cap 16384 \
    --lm-length-cap 4096 \
    --lm-length-cap 8192 \
    --lm-length-cap 16384 \
    --batch-size 4 \
    --static-table-json "${table_dir}/${arm}.json" \
    --table-label "olmo2_1b_s4_classic_${arm}" \
    --out "${output_dir}" \
    --execute >"${experiment_root}/logs/${arm}.log" 2>&1
  printf 'COMPLETE %s %s\n' "${arm}" "$(date -u +%FT%TZ)"
}

run_arm tailspline
run_arm mrpro

report="${experiment_root}/reports/tailspline_vs_mrpro_classic.json"
if [[ ! -e "${report}" ]]; then
  /root/miniconda3/bin/python -m \
    experiments.fixed_rope_three_interfaces_20260913.tailspline_olmo_classic_report \
    --run tailspline="${experiment_root}/runs/tailspline" \
    --run mrpro="${experiment_root}/runs/mrpro" \
    --receipt tailspline="${table_dir}/tailspline.json" \
    --receipt mrpro="${table_dir}/mrpro.json" \
    --ppl-manifest "${ppl_dir}/manifest.json" \
    --candidate tailspline \
    --baseline mrpro \
    --out "${report}"
fi

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
