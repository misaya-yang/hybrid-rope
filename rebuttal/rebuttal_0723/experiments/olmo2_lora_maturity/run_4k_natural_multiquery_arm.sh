#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-all}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
STAGE_READY_RECEIPT="${STAGE_READY_RECEIPT:?set STAGE_READY_RECEIPT}"
PARENT_ADAPTER="${PARENT_ADAPTER:?set PARENT_ADAPTER}"
PREPARED_DATA="${PREPARED_DATA:?set PREPARED_DATA}"
ROUTING_DATA="${ROUTING_DATA:?set ROUTING_DATA}"
BACKGROUND_DIR="${BACKGROUND_DIR:?set BACKGROUND_DIR}"
RUN_OUTPUT="${RUN_OUTPUT:?set RUN_OUTPUT}"
EXPERIMENT_READY_RECEIPT="${EXPERIMENT_READY_RECEIPT:?set EXPERIMENT_READY_RECEIPT}"
FREQUENCY="${FREQUENCY:?set FREQUENCY to native or evq}"
SEED="${SEED:-20260801}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CODE_ROOT="${CODE_ROOT:?set CODE_ROOT}"
COMPILE_CACHE="${COMPILE_CACHE:?set COMPILE_CACHE}"

if [[ "$MODE" != "preflight" && "$MODE" != "train" && "$MODE" != "all" ]]; then
  echo "MODE must be preflight, train, or all" >&2
  exit 2
fi
if [[ "$FREQUENCY" != "native" && "$FREQUENCY" != "evq" ]]; then
  echo "FREQUENCY must be native or evq" >&2
  exit 2
fi

for path in \
  "$CHECKPOINT" \
  "$STAGE_READY_RECEIPT" \
  "$PARENT_ADAPTER" \
  "$PREPARED_DATA" \
  "$ROUTING_DATA" \
  "$BACKGROUND_DIR"; do
  [[ -e "$path" ]] || {
    echo "missing required input: $path" >&2
    exit 2
  }
done

export PYTHONPATH="$CODE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE"
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

module="rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity"
trainer="$CODE_ROOT/rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/train_4k_counterfactual_routing.py"

preflight() {
  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m "$module.preflight_4k_natural_multiquery" \
    --checkpoint "$CHECKPOINT" \
    --stage-ready-receipt "$STAGE_READY_RECEIPT" \
    --parent-adapter "$PARENT_ADAPTER" \
    --prepared-data "$PREPARED_DATA" \
    --routing-data "$ROUTING_DATA" \
    --background-dir "$BACKGROUND_DIR" \
    --trainer "$trainer" \
    --run-output "$RUN_OUTPUT" \
    --receipt-output "$EXPERIMENT_READY_RECEIPT" \
    --frequency "$FREQUENCY" \
    --steps 300 \
    --micro-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --rank 64 \
    --alpha 128 \
    --learning-rate 5e-5 \
    --warmup-steps 20 \
    --counterfactual-margin 1 \
    --counterfactual-margin-weight 0.5 \
    --compile-mode max-autotune-no-cudagraphs \
    --natural-eval-rows 16 \
    --seed "$SEED"
}

train() {
  [[ -f "$EXPERIMENT_READY_RECEIPT" ]] || {
    echo "missing READY receipt: $EXPERIMENT_READY_RECEIPT" >&2
    exit 2
  }
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m "$module.train_4k_counterfactual_routing" \
    --checkpoint "$CHECKPOINT" \
    --parent-adapter "$PARENT_ADAPTER" \
    --prepared-data "$PREPARED_DATA" \
    --routing-data "$ROUTING_DATA" \
    --background-dir "$BACKGROUND_DIR" \
    --ready-receipt "$STAGE_READY_RECEIPT" \
    --experiment-ready-receipt "$EXPERIMENT_READY_RECEIPT" \
    --output "$RUN_OUTPUT" \
    --frequency "$FREQUENCY" \
    --steps 300 \
    --micro-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --rank 64 \
    --alpha 128 \
    --learning-rate 5e-5 \
    --warmup-steps 20 \
    --counterfactual-margin 1 \
    --counterfactual-margin-weight 0.5 \
    --compile-mode max-autotune-no-cudagraphs \
    --natural-eval-rows 16 \
    --seed "$SEED"
}

case "$MODE" in
  preflight)
    preflight
    ;;
  train)
    train
    ;;
  all)
    preflight
    train
    ;;
esac
