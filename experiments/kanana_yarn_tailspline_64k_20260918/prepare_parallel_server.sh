#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
plan=${HYBRID_ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
root=${KANANA_EXPERIMENT_ROOT:-${plan}/kanana_yarn_tailspline_64k_20260918}
model=${KANANA_MODEL:-/root/autodl-tmp/models/kakaocorp/kanana-1.5-8b-instruct-2505}
ruler=${RULER_ROOT:-/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a}
python_bin=${PYTHON_BIN:-/root/miniconda3/bin/python}
workers=${CPU_PREP_WORKERS:-12}

if [[ ! -f ${model}/tokenizer.json || ! -f ${model}/config.json ]]; then
  printf 'REFUSE: Kanana tokenizer/config is incomplete\n' >&2
  exit 1
fi
if [[ ! -f ${ruler}/scripts/synthetic.yaml ]]; then
  printf 'REFUSE: pinned RULER source is absent\n' >&2
  exit 1
fi
if [[ ${workers} -lt 1 || ${workers} -gt 12 ]]; then
  printf 'REFUSE: CPU_PREP_WORKERS must be in [1,12]\n' >&2
  exit 1
fi

cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "${root}/logs/cpu_tasks"

tasks=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)

run_task() {
  local index=$1 task=$2
  "${python_bin}" -m experiments.llama3_60dir_20260911.prepare_planb_panel \
    --model "${model}" --model-contract generic \
    --upstream "${ruler}" \
    --out "${root}/assets/full13/source_parts/${task}" \
    --stage H --contract planb --tasks "${task}" \
    --caps 65536 --counts-by-cap 65536:10 \
    --selection-mode source-order --source-only \
    --qa-base-offset 5600 --seed "$((20260918 + index * 100))" \
    >"${root}/logs/cpu_tasks/${task}.log" 2>&1
}
export -f run_task
export python_bin model ruler root

for index in "${!tasks[@]}"; do
  printf '%s %s\n' "${index}" "${tasks[${index}]}"
done | xargs -n 2 -P "${workers}" bash -c 'run_task "$1" "$2"' _

"${python_bin}" -m experiments.kanana_yarn_tailspline_64k_20260918.prepare \
  --model "${model}" --ruler "${ruler}" --root "${root}"
printf 'KANANA_CPU_ASSETS_READY %s\n' "$(date -u +%FT%TZ)"
