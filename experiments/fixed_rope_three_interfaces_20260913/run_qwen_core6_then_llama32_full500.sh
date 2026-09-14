#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914
qwen_root="${root}/tailspline_qwen25_s2_32k64k"
llama_root="${root}/tailspline_llama_s4_32k_full500"

cd "${repo_dir}"

wait_for_result() {
  local label=$1
  local pid_file=$2
  local result=$3
  local expected_command=$4
  while [[ ! -f "${result}" ]]; do
    if [[ ! -f "${pid_file}" ]]; then
      printf 'FAILED %s missing pid file\n' "${label}" >&2
      return 1
    fi
    local pid
    pid=$(cat "${pid_file}")
    if ! ps -p "${pid}" -o args= | grep -q "${expected_command}"; then
      printf 'FAILED %s process ended before result\n' "${label}" >&2
      return 1
    fi
    sleep 15
  done
  printf 'READY %s %s\n' "${label}" "$(date -u +%FT%TZ)"
}

wait_for_result \
  qwen_core6 \
  "${qwen_root}/supervisor.pid" \
  "${qwen_root}/reports/tailspline_vs_mrpro_32k64k.json" \
  run_tailspline_qwen25_s2_32k64k.sh

wait_for_result \
  llama32_assets \
  "${llama_root}/prepare.pid" \
  "${llama_root}/assets/full13_32k_extra490_parallel/rows.jsonl" \
  prepare_tailspline_llama_32k_full500_assets.sh

prepare_pid=$(cat "${llama_root}/prepare.pid")
while ps -p "${prepare_pid}" -o args= | grep -q prepare_tailspline_llama_32k_full500_assets.sh; do
  sleep 2
done
if ! grep -q '"status": "COMPLETE"' \
    "${llama_root}/assets/full13_32k_extra490_parallel/manifest.json"; then
  printf 'FAILED llama32 asset manifest is not complete\n' >&2
  exit 1
fi

batch_size=1
if "${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_llama32_batch2_canary.sh" \
    >"${llama_root}/logs/batch2_canary.log" 2>&1; then
  batch_size=2
fi
printf 'LLAMA32_SELECTED_BATCH %s\n' "${batch_size}"

LLAMA_FULL500_BATCH_SIZE="${batch_size}" \
  "${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_32k_full500.sh"

printf 'QWEN_CORE6_AND_LLAMA32_FULL500_COMPLETE %s\n' "$(date -u +%FT%TZ)"
