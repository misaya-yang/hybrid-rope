#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/video_temporal/generated/moving_mnist_medium}"
VARIANTS="${VARIANTS:-geo_k8,geo_k12,geo_k16,evq_k12,evq_k16}"
SEEDS="${SEEDS:-42,137}"
PROFILE="${PROFILE:-blackwell96}"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"
EXTRA_ARGS_ARR=()
if [[ -n "$EXTRA_ARGS_STR" ]]; then
  read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS_STR"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/results/supporting_video/phase23_video_temporal_blackwell/$STAMP}"
mkdir -p "$WORK_DIR"

echo "[phase23] root=$ROOT_DIR"
echo "[phase23] data_dir=$DATA_DIR"
echo "[phase23] work_dir=$WORK_DIR"
echo "[phase23] profile=$PROFILE variants=$VARIANTS seeds=$SEEDS"
echo "[phase23] extra_args=$EXTRA_ARGS_STR"

python3 "$ROOT_DIR/scripts/video_temporal/run_video_temporal_allocation_sweep.py" \
  --data-dir "$DATA_DIR" \
  --profile "$PROFILE" \
  --variants "$VARIANTS" \
  --seeds "$SEEDS" \
  --work-dir "$WORK_DIR" \
  "${EXTRA_ARGS_ARR[@]}" \
  2>&1 | tee "$WORK_DIR/run.log"
