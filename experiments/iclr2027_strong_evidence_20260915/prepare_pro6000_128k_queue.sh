#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
qwen_model=${QWEN_MODEL:-/root/autodl-tmp/rope_qwen_baseline_20260907/model}
upstream=${RULER_UPSTREAM:-/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a}
qwen_root=${plan}/tailspline_qwen25_s4_64k128k_clean
ready=${plan}/pro6000_128k_queue/assets_ready.json

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

# This stage is deliberately CPU-only.  It is safe to run while a different
# experiment owns GPU0 and creates no model generations.
"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer \
  --model "${qwen_model}" --model-id qwen25_3b --data-root "${upstream}" \
  --out "${qwen_root}/assets" --scale 4 --lengths 65536,131072 \
  --rows-per-task 50 --seed 20261101 --qa-offset 5600

mkdir -p "${qwen_root}/tables" "${qwen_root}/logs" "${qwen_root}/reports"
make_table() {
  local arm=$1 method=$2 role=$3 output=${qwen_root}/tables/$1.json
  if [[ -f "${output}" ]]; then return; fi
  "${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
    --config "${qwen_model}/config.json" --method "${method}" --scale 4 \
    --candidate-id "strong_qwen25_3b_s4_${arm}" --model-id qwen25_3b \
    --role "${role}" --changed-variable internal_frequency_allocation --out "${output}"
}
make_table tailspline tailspline candidate
make_table mrpro mrpro baseline

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.pro6000_128k_preflight \
  --plan-root "${plan}" --qwen-model "${qwen_model}" --out "${ready}"

# The generic runner performs a second independent read-only contract check.
"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.run_clean_matrix \
  --model "${qwen_model}" --model-id qwen25_3b \
  --data-root "${qwen_root}/assets" --out "${qwen_root}" --scale 4 \
  --lengths 65536 131072 --rows-per-task 50 \
  --data-manifest "${plan}/tailspline_llama_s4_classic/assets/ppl46/manifest.json" \
  --python "${python_bin}" --batch-size 1 --prefill-chunk-size 8192 --longest-first \
  >"${qwen_root}/logs/plan_only.json"

printf 'PRO6000_128K_ASSETS_READY %s\n' "$(date -u +%FT%TZ)"
