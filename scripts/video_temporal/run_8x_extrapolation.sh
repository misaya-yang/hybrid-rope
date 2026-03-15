#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
# 8x Extrapolation Test
#
# Uses existing trained checkpoints, just generates longer test data
# and evaluates at 8x (256 frames). No retraining needed.
#
# Usage:
#   # Evaluate on oscillating MNIST checkpoints:
#   CKPT_DIR=results/supporting_video/oscillating_fvd/<timestamp> \
#     bash scripts/video_temporal/run_8x_extrapolation.sh
#
#   # With VideoMAE FVD:
#   WITH_VIDEOMAE=1 CKPT_DIR=... \
#     bash scripts/video_temporal/run_8x_extrapolation.sh
#
# Time: ~1-2h (data gen + generation + FVD, no training)
# ==========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CKPT_DIR="${CKPT_DIR:?'Set CKPT_DIR to directory with .pt checkpoints'}"
DATASET="${DATASET:-oscillating}"  # oscillating or linear
N_GENERATE="${N_GENERATE:-256}"
SEED="${SEED:-42}"

STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/results/supporting_video/8x_extrapolation/$STAMP}"
mkdir -p "$WORK_DIR"

echo "=========================================="
echo " 8x Extrapolation Evaluation"
echo "=========================================="
echo "  ckpt_dir=$CKPT_DIR"
echo "  dataset=$DATASET"
echo "  n_generate=$N_GENERATE"
echo "  work_dir=$WORK_DIR"
echo ""

# Step 1: Generate 256-frame test data (if needed)
# We need test/val with 256 frames for 8x eval on 32f-trained models
DATA_8X="$ROOT_DIR/data/video_temporal/generated/moving_mnist_8x"

if [[ ! -f "$DATA_8X/manifest.json" ]]; then
    echo "[step 1] Generating 256-frame Moving MNIST data..."

    if [[ "$DATASET" == "oscillating" ]]; then
        # Use oscillating variant if available, otherwise generate
        python3 -c "
import sys, json
sys.path.insert(0, 'scripts/data_prep')
from prepare_moving_mnist_video import *
import numpy as np

# Check if oscillating data prep exists
# For now, generate standard moving MNIST with 256 frames
# The user can replace this with oscillating variant

out_dir = Path('$DATA_8X')
out_dir.mkdir(parents=True, exist_ok=True)

# We only need val and test splits (longer), train is the same
raw_dir = Path('data/video_temporal/external/mnist_raw')
mnist = load_mnist(raw_dir)
test_images, test_labels = mnist['test']

print('Building 256-frame val split...')
val_tokens, val_labels = build_split_tokens(
    test_images, test_labels,
    num_videos=2000, frames=256, image_size=64,
    num_digits=2, patch_size=8, vocab_size=256, seed=43,
)

print('Building 256-frame test split...')
test_tokens, test_labels = build_split_tokens(
    test_images, test_labels,
    num_videos=2000, frames=256, image_size=64,
    num_digits=2, patch_size=8, vocab_size=256, seed=44,
)

# Copy train tokens from original dataset (32 frames, unchanged)
import shutil
orig = Path('data/video_temporal/generated/moving_mnist_medium')
shutil.copy(orig / 'train_tokens.npy', out_dir / 'train_tokens.npy')
shutil.copy(orig / 'train_labels.npy', out_dir / 'train_labels.npy')

np.save(out_dir / 'val_tokens.npy', val_tokens)
np.save(out_dir / 'val_labels.npy', val_labels)
np.save(out_dir / 'test_tokens.npy', test_tokens)
np.save(out_dir / 'test_labels.npy', test_labels)

manifest = {
    'dataset': 'moving_mnist_8x',
    'train_videos': 16000,
    'val_videos': 2000,
    'test_videos': 2000,
    'train_frames': 32,
    'eval_frames': 256,
    'image_size': 64,
    'num_digits': 2,
    'patch_size': 8,
    'vocab_size': 256,
    'patches_per_frame': 64,
    'train_tokens_per_video': 2048,
    'eval_tokens_per_video': 16384,
    'seed': 42,
    'files': {
        'train_tokens': 'train_tokens.npy',
        'train_labels': 'train_labels.npy',
        'val_tokens': 'val_tokens.npy',
        'val_labels': 'val_labels.npy',
        'test_tokens': 'test_tokens.npy',
        'test_labels': 'test_labels.npy',
    },
}
json.dump(manifest, open(out_dir / 'manifest.json', 'w'), indent=2)
print(f'Saved 256-frame dataset to {out_dir}')
print(f'  val: {val_tokens.shape}, test: {test_tokens.shape}')
"
    fi
else
    echo "[step 1] 8x data exists: $DATA_8X"
fi

# Step 2: Evaluate at 8x using existing checkpoints
echo ""
echo "[step 2] Running 8x FVD evaluation..."
python3 "$ROOT_DIR/scripts/video_temporal/run_phase23_fvd_verify.py" \
    --data-dir "$DATA_8X" \
    --eval-only \
    --variants "geo_k16,evq_k16" \
    --seed "$SEED" \
    --n-generate "$N_GENERATE" \
    --eval-frames "32,64,128,192,256" \
    --work-dir "$WORK_DIR" \
    "$@" \
    2>&1 | tee "$WORK_DIR/8x_eval.log"

echo ""
echo "=========================================="
echo " 8x Extrapolation Complete"
echo "=========================================="
echo "  Results: $WORK_DIR/fvd_verify_summary.json"
