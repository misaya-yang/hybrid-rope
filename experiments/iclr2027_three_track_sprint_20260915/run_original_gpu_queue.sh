#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=/root/autodl-tmp/iclr2027_three_track_sprint_20260915
clean_report=${plan}/tailspline_llama_s4_32k_ruler200_clean/reports/tailspline_vs_mrpro_full13_32k_200_per_task_clean.json
natural_report=${plan}/tailspline_llama_s4_naturalqa631/reports/tailspline_vs_mrpro_naturalqa631.json
native_z_complete=${plan}/olmo_native_z5_enhancement/complete.txt

mkdir -p "${root}/logs" "${root}/status"
cd "${repo}"
export PYTHONPATH=.

if [[ ! -f "${clean_report}" ]]; then
  experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_32k_ruler200_clean.sh \
    >"${root}/logs/original_clean_ruler.log" 2>&1
fi
if [[ ! -f "${natural_report}" ]]; then
  experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_naturalqa631.sh \
    >"${root}/logs/original_naturalqa.log" 2>&1
fi
if [[ ! -f "${native_z_complete}" ]]; then
  experiments/native_z_enhancement_20260914/run.sh \
    >"${root}/logs/original_native_z5.log" 2>&1
fi

sha256sum "${clean_report}" "${natural_report}" \
  "${plan}/olmo_native_z5_enhancement/reports/native_vs_z5.json" \
  >"${root}/status/original_reports.sha256"
printf 'ICLR2027_ORIGINAL_GPU_QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/status/original_complete.txt"
