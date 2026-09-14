#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914
queue_root="${root}/three_model_full50_overnight"

mkdir -p "${queue_root}/logs"
cd "${repo_dir}"

wait_for_job() {
  local label=$1
  local pid_file=$2
  local result_file=$3
  local expected_command=$4
  while [[ ! -f "${result_file}" ]]; do
    if [[ ! -f "${pid_file}" ]]; then
      printf 'FAILED %s missing pid file %s\n' "${label}" "${pid_file}" >&2
      return 1
    fi
    local pid
    pid=$(cat "${pid_file}")
    if ! ps -p "${pid}" -o args= | grep -q "${expected_command}"; then
      printf 'FAILED %s process %s ended before %s\n' "${label}" "${pid}" "${result_file}" >&2
      return 1
    fi
    sleep 15
  done
  printf 'READY %s %s\n' "${label}" "$(date -u +%FT%TZ)"
}

wait_for_job \
  qwen_core6 \
  "${root}/tailspline_qwen25_s2_32k64k/supervisor.pid" \
  "${root}/tailspline_qwen25_s2_32k64k/reports/tailspline_vs_mrpro_32k64k.json" \
  run_tailspline_qwen25_s2_32k64k.sh

wait_for_job \
  olmo_assets \
  "${root}/tailspline_olmo_s4_full50/prepare.pid" \
  "${root}/tailspline_olmo_s4_full50/assets/full13_extra40/manifest.json" \
  prepare_tailspline_olmo_full50_assets.sh

"${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_olmo_s4_full50.sh"

wait_for_job \
  llama_assets \
  "${root}/tailspline_llama_s4_full50/prepare.pid" \
  "${root}/tailspline_llama_s4_full50/assets/full13_extra40/manifest.json" \
  prepare_tailspline_llama_full50_assets.sh

"${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_full50.sh"

wait_for_job \
  qwen_assets \
  "${root}/tailspline_qwen25_s2_full50/prepare.pid" \
  "${root}/tailspline_qwen25_s2_full50/assets/full13_50/manifest.json" \
  prepare_tailspline_qwen_full50_assets.sh

"${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_qwen25_s2_full50.sh"

printf 'THREE_MODEL_FULL50_COMPLETE %s\n' "$(date -u +%FT%TZ)"
