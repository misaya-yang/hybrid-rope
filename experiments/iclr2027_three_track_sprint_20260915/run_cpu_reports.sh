#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=/root/autodl-tmp/iclr2027_three_track_sprint_20260915
classic=${plan}/tailspline_llama_s4_classic
dose=${plan}/tailspline_llama_s4_matched_dose_c
python_bin=/root/miniconda3/bin/python

mkdir -p "${root}/assets" "${root}/reports" "${root}/status"
cd "${repo}"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""

${python_bin} -m experiments.iclr2027_three_track_sprint_20260915.verify_sprint_math \
  --out "${root}/reports/sprint_math_checks.json"
${python_bin} -m experiments.iclr2027_three_track_sprint_20260915.prepare_sprint_cpu \
  --plan-root "${plan}" --out "${root}"
${python_bin} -m experiments.iclr2027_three_track_sprint_20260915.e1_experimental_audit \
  --report "${dose}/reports/tailspline_vs_dose_control_c_classic.json" \
  --candidate-table "${classic}/tables/tailspline.json" \
  --control-table "${dose}/tables/llama_s4_tailspline_dose_control.json" \
  --candidate-run "${classic}/runs/tailspline" \
  --control-run "${dose}/runs/dose_control_c" \
  --out "${root}/reports/e1_matched_displacement_audit.json"
${python_bin} -m experiments.iclr2027_three_track_sprint_20260915.native_ppl_summary \
  --run "native=${classic}/runs/native_ppl_8k/lm_rows.jsonl" \
  --run "tailspline=${classic}/runs/tailspline/lm_rows.jsonl" \
  --run "mrpro=${classic}/runs/mrpro/lm_rows.jsonl" \
  --out "${root}/reports/native8k_ppl_comparison.json"

sha256sum "${root}"/reports/*.json >"${root}/status/cpu_reports.sha256"
printf 'ICLR2027_CPU_REPORTS_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/status/cpu_reports_complete.txt"
