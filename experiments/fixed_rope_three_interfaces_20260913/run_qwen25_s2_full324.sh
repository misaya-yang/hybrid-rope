#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/qwen25_s2_full324
data_manifest=/root/autodl-tmp/rope_qwen_baseline_20260907/prepared_v2/manifest.json
model_dir=/root/autodl-tmp/rope_qwen_baseline_20260907/model
panel=/root/autodl-tmp/band_mini_20260913/qwen_s2/frozen_324/screen.jsonl
table_dir=/root/autodl-tmp/cross_model_mix075_20260914/tables
batch_size=2

mkdir -p "${experiment_root}/logs" "${experiment_root}/runs"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_arm() {
  local arm_name=$1
  local table_path=$2
  local output_dir="${experiment_root}/runs/${arm_name}_b2"
  local log_path="${experiment_root}/logs/${arm_name}_b2.log"

  if [[ -f "${output_dir}/status.json" ]] && grep -q '"status": "COMPLETE"' "${output_dir}/status.json"; then
    printf 'SKIP_COMPLETE %s\n' "${arm_name}"
    return
  fi

  printf 'START %s %s\n' "${arm_name}" "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data_manifest}" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${panel}" \
    --only-extra-panels \
    --skip-lm \
    --length-cap 32768 \
    --length-cap 49152 \
    --length-cap 65536 \
    --prefill-chunk-size 8192 \
    --batch-size "${batch_size}" \
    --static-table-json "${table_path}" \
    --table-label "qwen25_3b_s2_${arm_name}_b2_full324" \
    --out "${output_dir}" \
    --execute >"${log_path}" 2>&1
  printf 'COMPLETE %s %s\n' "${arm_name}" "$(date -u +%FT%TZ)"
}

run_arm mix075 "${table_dir}/mix075.json"
run_arm mrpro "${table_dir}/mrpro.json"
run_arm yarn "${table_dir}/yarn.json"
run_arm bm "${table_dir}/bm_band22_39_midgain.json"

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
nohup bash "${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/analyze_qwen25_s2_full324.sh" \
  >"${experiment_root}/logs/analysis.log" 2>&1 </dev/null &
