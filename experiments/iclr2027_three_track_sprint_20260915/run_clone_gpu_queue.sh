#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=/root/autodl-tmp/iclr2027_three_track_sprint_20260915
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
classic=${plan}/tailspline_llama_s4_classic
clean=${plan}/tailspline_llama_s4_32k_ruler200_clean
strong=${plan}/tailspline_llama_s4_classic_strong_baselines

mkdir -p "${root}/runs" "${root}/logs" "${root}/reports" "${root}/status"
cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python_bin=/root/miniconda3/bin/python
probe=${root}/assets/classic_runtime_probe39.jsonl
yarn_table=${strong}/tables/yarn.json

${python_bin} -m experiments.iclr2027_three_track_sprint_20260915.prepare_sprint_cpu \
  --plan-root "${plan}" --out "${root}"

probe_run=${root}/runs/tailspline_classic_batch2_probe39
if [[ ! -f "${probe_run}/status.json" ]]; then
  ${python_bin} -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${probe}" --only-extra-panels --skip-lm \
    --length-cap 8192 --length-cap 16384 --length-cap 32768 \
    --prefill-chunk-size 8192 --batch-size 2 \
    --static-table-json "${classic}/tables/tailspline.json" \
    --table-label llama3_8b_s4_tailspline_classic_batch2_probe39 \
    --out "${probe_run}" --execute >"${root}/logs/tailspline_batch2_probe39.log" 2>&1
fi
${python_bin} -m experiments.iclr2027_three_track_sprint_20260915.runtime_probe_report \
  --panel "${probe}" \
  --batch1 "${classic}/runs/tailspline/generations.jsonl" \
  --batch2 "${probe_run}/generations.jsonl" \
  --out "${root}/reports/tailspline_batch1_vs_batch2_probe39.json"

clean_yarn=${root}/runs/yarn_clean32k_ruler200
if [[ ! -f "${clean_yarn}/status.json" ]]; then
  ${python_bin} -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${clean}/assets/inputs.jsonl" --only-extra-panels --skip-lm \
    --length-cap 32768 --prefill-chunk-size 8192 --batch-size 1 \
    --static-table-json "${yarn_table}" --table-label llama3_8b_s4_yarn_clean32k_ruler200 \
    --out "${clean_yarn}" --execute >"${root}/logs/yarn_clean32k_ruler200.log" 2>&1
fi
clean_report=${root}/reports/clean32k_tailspline_vs_mrpro_yarn.json
if [[ ! -f "${clean_report}" ]]; then
  ${python_bin} -m experiments.fixed_rope_three_interfaces_20260913.matched_generation_report \
    --source "tailspline=${clean}/runs/tailspline/generations.jsonl" \
    --source "mrpro=${clean}/runs/mrpro/generations.jsonl" \
    --source "yarn=${clean_yarn}/generations.jsonl" \
    --candidate tailspline --baseline mrpro --baseline yarn --length 32768 \
    --out "${clean_report}"
fi

classic_yarn=${root}/runs/yarn_classic_batch1
if [[ ! -f "${classic_yarn}/status.json" ]]; then
  ${python_bin} -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${classic}/assets/full13/rows.jsonl" --only-extra-panels \
    --length-cap 8192 --length-cap 16384 --length-cap 32768 \
    --lm-length-cap 8192 --lm-length-cap 16384 --lm-length-cap 32768 \
    --prefill-chunk-size 8192 --batch-size 1 \
    --static-table-json "${yarn_table}" --table-label llama3_8b_s4_yarn_classic_batch1 \
    --out "${classic_yarn}" --execute >"${root}/logs/yarn_classic_batch1.log" 2>&1
fi
classic_report=${root}/reports/classic_batch1_tailspline_vs_mrpro_yarn.json
if [[ ! -f "${classic_report}" ]]; then
  ${python_bin} -m experiments.fixed_rope_three_interfaces_20260913.tailspline_llama_classic_report \
    --run "tailspline=${classic}/runs/tailspline" \
    --run "mrpro=${classic}/runs/mrpro" \
    --run "yarn=${classic_yarn}" \
    --receipt "tailspline=${classic}/tables/tailspline.json" \
    --receipt "mrpro=${classic}/tables/mrpro.json" \
    --receipt "yarn=${yarn_table}" \
    --ppl-manifest "${classic}/assets/ppl46/manifest.json" \
    --candidate tailspline --baseline mrpro --baseline yarn \
    --out "${classic_report}"
fi

sha256sum "${root}"/reports/*.json >"${root}/status/clone_reports.sha256"
printf 'ICLR2027_CLONE_GPU_QUEUE_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/status/clone_complete.txt"
