#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/tailspline_llama_s4_16k_ruler50_clean
model=/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct
upstream=/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
parts=${root}/source_parts

tasks=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)

mkdir -p "${parts}" "${root}/logs" "${root}/assets"
cd "${repo}"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

pids=()
for index in "${!tasks[@]}"; do
  task=${tasks[$index]}
  part=${parts}/${task}
  if [[ -f "${part}/manifest.json" ]] && grep -q '"status": "COMPLETE"' "${part}/manifest.json"; then
    continue
  fi
  /root/miniconda3/bin/python -m experiments.llama3_60dir_20260911.prepare_planb_panel \
    --model "${model}" --upstream "${upstream}" --out "${part}" \
    --stage H --contract planb --tasks "${task}" --caps 16384 \
    --counts-by-cap 16384:50 --selection-mode source-order \
    --qa-base-offset 8000 --seed "$((20261001 + index))" \
    >"${root}/logs/prepare_16k_${task}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" != 0 ]]; then
  printf '16K source preparation failed\n' >&2
  exit 1
fi

/root/miniconda3/bin/python -m \
  experiments.fixed_rope_three_interfaces_20260913.prepare_tailspline_llama_32k_ruler200_clean \
  --source-parts "${parts}" --model "${model}" --out "${root}/assets" \
  --length 16384 --rows-per-task 50

printf 'CLEAN16K_RULER50_ASSETS_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/assets_complete.txt"
