#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
ruler_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_ruler200
natural_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_naturalqa631
ruler_report=${ruler_root}/reports/tailspline_vs_mrpro_full13_32k_200_per_task.json
natural_report=${natural_root}/reports/tailspline_vs_mrpro_naturalqa631.json

ruler_pid=$(cat "${ruler_root}/supervisor.pid")
while ps -p "${ruler_pid}" -o args= | grep -q run_tailspline_llama_s4_32k_ruler200_staged.sh; do
  sleep 5
done
if [[ ! -f "${ruler_report}" ]] || ! grep -q '"status": "MATCHED_GENERATION_RANGE_REPORT_V1"' "${ruler_report}"; then
  printf 'FAILED RULER-200 supervisor ended without a complete report\n' >&2
  exit 1
fi
printf 'RULER200_VERIFIED %s\n' "$(date -u +%FT%TZ)"

cd "${repo_dir}"
chmod +x experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_naturalqa631.sh
experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_naturalqa631.sh

if [[ ! -f "${natural_report}" ]] || ! grep -q '"status": "COMPLETE"' "${natural_report}"; then
  printf 'FAILED natural-QA runner ended without a complete report\n' >&2
  exit 1
fi
sha256sum "${ruler_report}" "${natural_report}" | tee \
  /root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_final_benchmark_sha256.txt
printf 'FINAL_BENCHMARK_CHAIN_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee \
  /root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_final_benchmark_complete.txt
