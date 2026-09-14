#!/usr/bin/env bash
set -euo pipefail

repo_dir=/root/autodl-tmp/hybrid-rope
experiment_root=/root/autodl-tmp/today_rope_plan_20260914/llama_s4_factorial_gap48
legacy_root=/root/autodl-tmp/fixed_rope_three_interfaces_20260913
data_manifest=/root/autodl-tmp/rope_qwen_baseline_20260907/prepared_v2/manifest.json
model_dir=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
table_path="${experiment_root}/tables/c42_g1_midgain.json"
output_dir="${experiment_root}/runs/c42_g1_midgain"

mkdir -p "${experiment_root}/logs" "${experiment_root}/reports" "${experiment_root}/runs"
cd "${repo_dir}"
export PYTHONPATH=.

if ! [[ -f "${output_dir}/status.json" ]] || ! grep -q '"status": "COMPLETE"' "${output_dir}/status.json"; then
  /root/miniconda3/bin/python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${data_manifest}" \
    --model "${model_dir}" \
    --arm Native \
    --extra-panel "${legacy_root}/panels/llama_low108/screen.jsonl" \
    --extra-panel "${legacy_root}/panels/llama_s4_core6_fresh6_8k16k_seed20260919/screen.jsonl" \
    --extra-panel "${legacy_root}/panels/llama_s4_core6_fresh6_16k32k_seed20260917_v2/screen.jsonl" \
    --only-extra-panels \
    --skip-lm \
    --length-cap 8192 \
    --length-cap 32768 \
    --task fwe \
    --task vt \
    --prefill-chunk-size 8192 \
    --batch-size 1 \
    --static-table-json "${table_path}" \
    --table-label llama_s4_c42_band16_34_g1_midgain_gap48 \
    --out "${output_dir}" \
    --execute >"${experiment_root}/logs/c42_g1_midgain.log" 2>&1
fi

report="${experiment_root}/reports/c42_mix075_gain_factorial_vtfwe_8k32k12.json"
if [[ -e "${report}" ]]; then
  printf 'REPORT_EXISTS %s\n' "${report}"
else
  /root/miniconda3/bin/python -m experiments.fixed_rope_three_interfaces_20260913.factorial_report \
    --source Y00="${legacy_root}/runs/llama_s4_c42_band16_34_core6_8k32k6/generations.jsonl" \
    --source Y00="${legacy_root}/runs/llama_s4_c42_band16_34_fresh6_8k16k/generations.jsonl" \
    --source Y00="${legacy_root}/runs/llama_s4_c42_band16_34_fresh6_16k32k/generations.jsonl" \
    --source Y01="${output_dir}/generations.jsonl" \
    --source Y10="${legacy_root}/runs/llama_s4_transition_mix075_core6_8k32k6/generations.jsonl" \
    --source Y10="${legacy_root}/runs/llama_s4_transition_mix075_fresh6_8k16k/generations.jsonl" \
    --source Y10="${legacy_root}/runs/llama_s4_transition_mix075_fresh6_16k32k/generations.jsonl" \
    --source Y11="${legacy_root}/runs/llama_s4_mix075_band16_34_loggain_mid_core6_screen6_8k32k/generations.jsonl" \
    --source Y11="${legacy_root}/runs/llama_s4_mix075_loggain_mid_block2_8k16k/generations.jsonl" \
    --source Y11="${legacy_root}/runs/llama_s4_mix075_loggain_mid_block2_32k/generations.jsonl" \
    --task fwe \
    --task vt \
    --length 8192 \
    --length 32768 \
    --rows-per-cell 12 \
    --out "${report}"
fi

exec /root/autodl-tmp/hybrid-rope/experiments/fixed_rope_three_interfaces_20260913/run_qwen25_s4_full.sh
