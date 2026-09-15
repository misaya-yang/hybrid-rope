#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
python_bin=/root/miniconda3/bin/python
model=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
classic=${plan}/tailspline_olmo_s4_classic
ruler_root=${plan}/tailspline_olmo_s4_16k_ruler200_clean

if [[ "${1:-}" != "--execute" ]]; then
  printf '%s\n' 'PLAN_ONLY: 1) OLMo Natural-QA631 T/P; 2) OLMo clean16K Full13x200 T/P last.'
  printf '%s\n' 'No GPU task is launched without --execute.'
  exit 0
fi

cd "${repo}"
export PYTHONPATH=.

ruler_manifest=${ruler_root}/assets/manifest.json
if [[ ! -f "${ruler_manifest}" ]]; then
  "${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer \
    --model "${model}" --model-id olmo2_1b \
    --data-root /root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a \
    --out "${ruler_root}/assets" --scale 4 --lengths 16384 --rows-per-task 200 \
    --seed 20261101 --qa-offset 5600
fi

full20=${plan}/tailspline_llama_s4_mrrope_niah_heatmap_full20
if [[ ! -f "${full20}/complete.txt" ]]; then
  printf 'REFUSE: current NIAH Full20 has not produced complete.txt\n' >&2
  exit 1
fi

bash experiments/iclr2027_strong_evidence_20260915/run_olmo_naturalqa631.sh --execute

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.run_clean_matrix \
  --model "${model}" --model-id olmo2_1b --scale 4 \
  --data-root "${ruler_root}/assets" --out "${ruler_root}" \
  --lengths 16384 --rows-per-task 200 \
  --data-manifest "${classic}/assets/ppl46/manifest.json" \
  --python "${python_bin}" --batch-size 1 --prefill-chunk-size 8192 --execute

printf 'OLMO_QA_THEN_RULER200_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${ruler_root}/queue_complete.txt"
