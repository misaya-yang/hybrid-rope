#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/video_temporal/generated/moving_mnist_medium}"
PROFILE="${PROFILE:-blackwell96}"
EPOCHS="${EPOCHS:-16}"
EVAL_CHUNKS="${EVAL_CHUNKS:-16}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"
EXTRA_ARGS_ARR=()
if [[ -n "$EXTRA_ARGS_STR" ]]; then
  read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS_STR"
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_WORK_DIR="${WORK_DIR:-$ROOT_DIR/results/supporting_video/phase23_video_temporal_blackwell/$STAMP}"
mkdir -p "$ROOT_WORK_DIR"

echo "[phase23-10h] root=$ROOT_DIR"
echo "[phase23-10h] data_dir=$DATA_DIR"
echo "[phase23-10h] profile=$PROFILE epochs=$EPOCHS eval_chunks=$EVAL_CHUNKS"
echo "[phase23-10h] work_dir=$ROOT_WORK_DIR"
echo "[phase23-10h] extra_args=$EXTRA_ARGS_STR"

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --profile "$PROFILE"
  --epochs "$EPOCHS"
  --eval-chunks "$EVAL_CHUNKS"
  "${EXTRA_ARGS_ARR[@]}"
)

echo
echo "[phase23-10h] pass 1/2: baseline + allocation + EVQ, seed 42"
python3 "$ROOT_DIR/scripts/video_temporal/run_video_temporal_allocation_sweep.py" \
  "${COMMON_ARGS[@]}" \
  --variants "geo_k8,geo_k16,evq_k16" \
  --seeds "42" \
  --work-dir "$ROOT_WORK_DIR/pass1_seed42" \
  2>&1 | tee "$ROOT_WORK_DIR/pass1_seed42.log"

echo
echo "[phase23-10h] pass 2/2: replicate decisive pair, seed 137"
python3 "$ROOT_DIR/scripts/video_temporal/run_video_temporal_allocation_sweep.py" \
  "${COMMON_ARGS[@]}" \
  --variants "geo_k16,evq_k16" \
  --seeds "137" \
  --work-dir "$ROOT_WORK_DIR/pass2_seed137" \
  2>&1 | tee "$ROOT_WORK_DIR/pass2_seed137.log"

echo
echo "[phase23-10h] complete"
echo "[phase23-10h] inspect:"
echo "  $ROOT_WORK_DIR/pass1_seed42/summary.json"
echo "  $ROOT_WORK_DIR/pass2_seed137/summary.json"
