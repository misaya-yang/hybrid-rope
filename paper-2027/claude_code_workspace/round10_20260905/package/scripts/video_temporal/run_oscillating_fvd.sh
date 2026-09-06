#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
# Oscillating MNIST: Fine-tune + Temporal Quality Evaluation
#
# Fine-tunes existing Moving MNIST checkpoints on oscillating data,
# then evaluates with temporal-specific metrics (FVMD, NoRepeat, FFT).
#
# Time budget: ~2h on Blackwell 96GB
#   - Fine-tuning: ~1h (2 arms × 4 epochs × ~15min)
#   - Generation + eval: ~1h (256 videos × 128f × 2 arms)
#
# Usage:
#   bash scripts/video_temporal/run_oscillating_fvd.sh
#
#   # Eval-only (after training is done):
#   EVAL_ONLY=1 WORK_DIR=results/.../existing \
#     bash scripts/video_temporal/run_oscillating_fvd.sh
# ==========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Oscillating MNIST data
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/video_temporal/generated/oscillating_mnist}"
# Base checkpoints from standard Moving MNIST training
BASE_CKPT_DIR="${BASE_CKPT_DIR:-$ROOT_DIR/results/supporting_video/phase23_fvd_verify/20260314_231255}"

PROFILE="${PROFILE:-blackwell96}"
FT_EPOCHS="${FT_EPOCHS:-4}"
N_GENERATE="${N_GENERATE:-256}"
GEN_BATCH="${GEN_BATCH:-48}"
SEED="${SEED:-42}"
FT_LR_FACTOR="${FT_LR_FACTOR:-0.3}"

STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/results/supporting_video/oscillating_fvd/$STAMP}"
mkdir -p "$WORK_DIR"

echo "================================================="
echo " Oscillating MNIST: Fine-tune + Temporal Eval"
echo "================================================="
echo "  data=$DATA_DIR"
echo "  base_ckpt=$BASE_CKPT_DIR"
echo "  epochs=$FT_EPOCHS  lr_factor=$FT_LR_FACTOR"
echo "  n_generate=$N_GENERATE  gen_batch=$GEN_BATCH"
echo "  work_dir=$WORK_DIR"
echo ""

# Step 0: Verify data + base checkpoints
if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
    echo "[step 0] Generating oscillating MNIST data..."
    python3 "$ROOT_DIR/scripts/data_prep/prepare_oscillating_mnist_video.py" \
        --out-dir "$DATA_DIR"
fi
echo "[step 0] Data: $DATA_DIR/manifest.json"

for variant in geo_k16 evq_k16; do
    ckpt="$BASE_CKPT_DIR/${variant}_seed${SEED}.pt"
    if [[ -f "$ckpt" ]]; then
        echo "[step 0] Base checkpoint: $ckpt ($(du -h "$ckpt" | cut -f1))"
    else
        echo "[step 0] WARNING: base checkpoint not found: $ckpt"
        echo "         Will train from scratch (slower)"
    fi
done

# Step 1: I3D model
I3D_PATH="$ROOT_DIR/data/video_temporal/external/i3d_torchscript.pt"
[[ -f "$I3D_PATH" ]] && echo "[step 1] I3D: $I3D_PATH" || echo "[step 1] I3D not found (pixel FVD only)"

# Step 2: Run
EXTRA_ARGS=()
if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=("--eval-only")
else
    EXTRA_ARGS+=("--finetune-from" "$BASE_CKPT_DIR")
    EXTRA_ARGS+=("--finetune-lr-factor" "$FT_LR_FACTOR")
fi

echo ""
echo "[step 2] Starting fine-tune + eval..."
python3 "$ROOT_DIR/scripts/video_temporal/run_phase23_fvd_verify.py" \
    --data-dir "$DATA_DIR" \
    --profile "$PROFILE" \
    --variants "geo_k16,evq_k16" \
    --seed "$SEED" \
    --epochs "$FT_EPOCHS" \
    --n-generate "$N_GENERATE" \
    --gen-batch-size "$GEN_BATCH" \
    --i3d-path "$I3D_PATH" \
    --work-dir "$WORK_DIR" \
    --eval-frames "128" \
    --batch-size 36 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$WORK_DIR/oscillating_fvd.log"

echo ""
echo "================================================="
echo " Done! Results: $WORK_DIR/fvd_verify_summary.json"
echo "================================================="
