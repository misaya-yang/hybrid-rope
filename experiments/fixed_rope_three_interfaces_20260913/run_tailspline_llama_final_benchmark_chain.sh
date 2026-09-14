#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
ruler_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_ruler200_clean
natural_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_naturalqa631
dose_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_matched_dose_c
strong_root=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic_strong_baselines
ruler_report=${ruler_root}/reports/tailspline_vs_mrpro_full13_32k_200_per_task_clean.json
natural_report=${natural_root}/reports/tailspline_vs_mrpro_naturalqa631.json
dose_report=${dose_root}/reports/tailspline_vs_dose_control_c_classic.json
strong_report=${strong_root}/reports/tailspline_vs_mrpro_yarn_bm_classic.json

mkdir -p "${dose_root}/logs" "${ruler_root}/logs" "${natural_root}/logs" "${strong_root}/logs"

# Finish the already-running padded pair only as a diagnostic, then retire it.
diagnostic_pid_file=/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_32k_ruler200/diagnostic_last_gpu.pid
if [[ -f "${diagnostic_pid_file}" ]]; then
  diagnostic_pid=$(cat "${diagnostic_pid_file}")
  while ps -p "${diagnostic_pid}" -o args= | grep -q recovery_v2_eval; do
    sleep 5
  done
fi

dose_script=${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_matched_dose_c.sh
chmod +x "${dose_script}"
"${dose_script}" >"${dose_root}/logs/supervisor.log" 2>&1 &
dose_pid=$!
printf '%s\n' "${dose_pid}" >"${dose_root}/supervisor.pid"
wait "${dose_pid}"
if [[ ! -f "${dose_report}" ]] || ! grep -q '"status": "TAILSPLINE_LLAMA_CLASSIC_REPORT_V1"' "${dose_report}"; then
  printf 'FAILED matched-dose C ended without a complete report\n' >&2
  exit 1
fi
printf 'MATCHED_DOSE_C_VERIFIED %s\n' "$(date -u +%FT%TZ)"

clean_script=${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_32k_ruler200_clean.sh
chmod +x "${clean_script}"
"${clean_script}" >"${ruler_root}/logs/clean_supervisor.log" 2>&1 &
ruler_pid=$!
printf '%s\n' "${ruler_pid}" >"${ruler_root}/supervisor.pid"
wait "${ruler_pid}"
if [[ ! -f "${ruler_report}" ]] || ! grep -q '"status": "MATCHED_GENERATION_RANGE_REPORT_V1"' "${ruler_report}"; then
  printf 'FAILED RULER-200 supervisor ended without a complete report\n' >&2
  exit 1
fi
printf 'RULER200_VERIFIED %s\n' "$(date -u +%FT%TZ)"

cd "${repo_dir}"
chmod +x experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_naturalqa631.sh
experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_naturalqa631.sh \
  >"${natural_root}/logs/natural_supervisor.log" 2>&1 &
natural_pid=$!
printf '%s\n' "${natural_pid}" >"${natural_root}/supervisor.pid"
wait "${natural_pid}"

if [[ ! -f "${natural_report}" ]] || ! grep -q '"status": "COMPLETE"' "${natural_report}"; then
  printf 'FAILED natural-QA runner ended without a complete report\n' >&2
  exit 1
fi

strong_script=${repo_dir}/experiments/fixed_rope_three_interfaces_20260913/run_tailspline_llama_s4_classic_strong_baselines.sh
chmod +x "${strong_script}"
"${strong_script}" >"${strong_root}/logs/supervisor.log" 2>&1 &
strong_pid=$!
printf '%s\n' "${strong_pid}" >"${strong_root}/supervisor.pid"
wait "${strong_pid}"
if [[ ! -f "${strong_report}" ]] || ! grep -q '"status": "TAILSPLINE_LLAMA_CLASSIC_REPORT_V1"' "${strong_report}"; then
  printf 'FAILED classic strong baselines ended without a complete report\n' >&2
  exit 1
fi

sha256sum "${dose_report}" "${ruler_report}" "${natural_report}" "${strong_report}" | tee \
  /root/autodl-tmp/today_rope_plan_20260914/stable_accept_ready_queue_sha256.txt
printf 'STABLE_ACCEPT_READY_QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee \
  /root/autodl-tmp/today_rope_plan_20260914/stable_accept_ready_queue_complete.txt
