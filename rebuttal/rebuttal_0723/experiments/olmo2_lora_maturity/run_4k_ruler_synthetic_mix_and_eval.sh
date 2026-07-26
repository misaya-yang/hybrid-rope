#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT="${ASSET_ROOT:-/root/autodl-tmp/olmo2_1b_longalign_assets}"
CODE_ROOT="${CODE_ROOT:-$ASSET_ROOT/code}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ASSET_ROOT/models/OLMo-2-0425-1B-Instruct}"
PARENT_ADAPTER="${PARENT_ADAPTER:-$ASSET_ROOT/runs/instruct_evq_counterfactual_routing_4k_300_s20260725/adapter.pt}"
SYNTHETIC_SOURCE="${SYNTHETIC_SOURCE:-$ASSET_ROOT/data/ruler_synth_vt_cwe_fwe_qa_4k_n100_s20420726}"
EVAL_DATA="${EVAL_DATA:-$ASSET_ROOT/data/ruler_full_merged_n20_s20260802}"
ROUTING_TRAIN="${ROUTING_TRAIN:-$ASSET_ROOT/data/routing_pairs_4k_v1/train}"
NATURAL_VIEW="${NATURAL_VIEW:-$ASSET_ROOT/data/prepared_maturity_v1/longalign_paired_L4096}"
TRAINING_VIEW="${TRAINING_VIEW:-$ASSET_ROOT/data/ruler_synthetic_mix_fixed_4k_s20420726}"
OUTPUT="${OUTPUT:-$ASSET_ROOT/runs/instruct_evq_ruler_synthetic_mix_s20420726}"
EVAL_OUTPUT="${EVAL_OUTPUT:-$ASSET_ROOT/runs/full_ruler_evq_synthetic_mix_n20_s20420726}"
READY_RECEIPT="${READY_RECEIPT:-$ASSET_ROOT/receipts/instruct_4k_conversion_ready.json}"
LOG_ROOT="${LOG_ROOT:-$ASSET_ROOT/logs/evq_ruler_synthetic_mix_s20420726}"

for path in \
  "$CHECKPOINT" "$PARENT_ADAPTER" "$SYNTHETIC_SOURCE" "$EVAL_DATA" \
  "$ROUTING_TRAIN" "$NATURAL_VIEW" "$READY_RECEIPT"; do
  [[ -e "$path" ]] || {
    echo "missing input: $path" >&2
    exit 2
  }
done
[[ ! -e "$OUTPUT" && ! -e "$OUTPUT.incomplete" ]] || {
  echo "training output already exists: $OUTPUT" >&2
  exit 2
}
[[ ! -e "$EVAL_OUTPUT" ]] || {
  echo "evaluation output already exists: $EVAL_OUTPUT" >&2
  exit 2
}

mkdir -p "$LOG_ROOT"
export PYTHONPATH="$CODE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ASSET_ROOT/cache/torchinductor_5090}"

if [[ ! -e "$TRAINING_VIEW" ]]; then
  "$PYTHON_BIN" -m \
    rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_4k_ruler_synthetic_mix \
    --checkpoint "$CHECKPOINT" \
    --synthetic-root "$SYNTHETIC_SOURCE" \
    --eval-root "$EVAL_DATA" \
    --routing-train "$ROUTING_TRAIN" \
    --natural-view "$NATURAL_VIEW" \
    --output "$TRAINING_VIEW" \
    --seed 20420726 >"$LOG_ROOT/prepare.log" 2>&1
fi

"$PYTHON_BIN" -m \
  rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_ruler_synthetic_mix \
  --checkpoint "$CHECKPOINT" \
  --parent-adapter "$PARENT_ADAPTER" \
  --training-view "$TRAINING_VIEW" \
  --output "$OUTPUT" \
  --epochs 3 \
  --micro-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.05 \
  --compile-mode max-autotune-no-cudagraphs \
  --seed 20420726 >"$LOG_ROOT/train.log" 2>&1

tasks=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery
  vt cwe fwe qa_1 qa_2
)
"$PYTHON_BIN" -m \
  rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer \
  --checkpoint "$CHECKPOINT" \
  --ready-receipt "$READY_RECEIPT" \
  --data-root "$EVAL_DATA" \
  --output "$EVAL_OUTPUT" \
  --frequency evq \
  --adapter "$OUTPUT/adapter.pt" \
  --rank 64 \
  --alpha 128 \
  --tasks "${tasks[@]}" \
  --lengths 4096 8192 16384 \
  --limit-per-cell 20 >"$LOG_ROOT/eval.log" 2>&1

touch "$EVAL_OUTPUT/ALL_DONE"
