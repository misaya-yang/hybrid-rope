#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
root=/root/autodl-tmp/today_rope_plan_20260914
llama_root="${root}/tailspline_llama_s4_full50"

mkdir -p "${root}/qwen_then_llama/logs" "${llama_root}/logs"
cd "${repo_dir}"

"${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/prepare_tailspline_llama_full50_assets.sh" \
  >"${llama_root}/logs/prepare_supervisor.log" 2>&1 &
prepare_pid=$!
printf '%s\n' "${prepare_pid}" >"${llama_root}/prepare.pid"

"${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_qwen25_s2_32k64k.sh"

wait "${prepare_pid}"
"${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_full50.sh"

printf 'ALL_EXPERIMENTS_COMPLETE %s\n' "$(date -u +%FT%TZ)"
