#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/qwen25_s4_full
data_manifest=/root/autodl-tmp/rope_qwen_baseline_20260907/prepared_v2/manifest.json
model_dir=/root/autodl-tmp/rope_qwen_baseline_20260907/model
near_panel=/root/autodl-tmp/band_mini_20260913/qwen_s2/frozen_324/screen.jsonl
far_panel=/root/autodl-tmp/nongeometric_screen_20260909/heldout_mixed_s20260910/screen.jsonl
table_dir="${experiment_root}/tables"

mkdir -p "${experiment_root}/logs" "${experiment_root}/runs" "${experiment_root}/reports"
cd "${repo_dir}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_part() {
  local arm_name=$1
  local table_path=$2
  local part=$3
  local panel=$4
  local batch_size=$5
  shift 5
  local output_dir="${experiment_root}/runs/${arm_name}_${part}"
  local log_path="${experiment_root}/logs/${arm_name}_${part}.log"

  if [[ -f "${output_dir}/status.json" ]] && grep -q '"status": "COMPLETE"' "${output_dir}/status.json"; then
    printf 'SKIP_COMPLETE %s_%s\n' "${arm_name}" "${part}"
    return
  fi

  printf 'START %s_%s %s\n' "${arm_name}" "${part}" "$(date -u +%FT%TZ)"
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data_manifest}" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${panel}" \
    --only-extra-panels \
    --skip-lm \
    "$@" \
    --prefill-chunk-size 8192 \
    --batch-size "${batch_size}" \
    --static-table-json "${table_path}" \
    --table-label "qwen25_3b_s4_${arm_name}_${part}" \
    --out "${output_dir}" \
    --execute >"${log_path}" 2>&1
  printf 'COMPLETE %s_%s %s\n' "${arm_name}" "${part}" "$(date -u +%FT%TZ)"
}

run_arm() {
  local arm_name=$1
  local table_path=$2
  run_part "${arm_name}" "${table_path}" near "${near_panel}" 2 \
    --length-cap 32768 --length-cap 65536
  run_part "${arm_name}" "${table_path}" far "${far_panel}" 1 \
    --length-cap 131072
}

run_arm mix075 "${table_dir}/mix075_candidate.json"
run_arm mrpro "${table_dir}/mrpro.json"
run_arm yarn "${table_dir}/yarn.json"
run_arm bm "${table_dir}/bm.json"

report="${experiment_root}/reports/mix075_vs_mrpro_yarn_bm_32k64k128k.json"
if [[ ! -e "${report}" ]]; then
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source mix075="${experiment_root}/runs/mix075_near/generations.jsonl" \
    --source mix075="${experiment_root}/runs/mix075_far/generations.jsonl" \
    --source mrpro="${experiment_root}/runs/mrpro_near/generations.jsonl" \
    --source mrpro="${experiment_root}/runs/mrpro_far/generations.jsonl" \
    --source yarn="${experiment_root}/runs/yarn_near/generations.jsonl" \
    --source yarn="${experiment_root}/runs/yarn_far/generations.jsonl" \
    --source bm="${experiment_root}/runs/bm_near/generations.jsonl" \
    --source bm="${experiment_root}/runs/bm_far/generations.jsonl" \
    --candidate mix075 \
    --baseline mrpro \
    --baseline yarn \
    --baseline bm \
    --out "${report}"
fi

printf 'QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)"
