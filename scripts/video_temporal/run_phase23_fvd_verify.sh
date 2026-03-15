#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
# Phase 23 FVD Verification: Single-seed, decisive pair only
#
# Strategy: verify FVD(EVQ) < FVD(Geo) with ONE seed FIRST,
# then scale to multi-seed if positive.
#
# Hardware: R6000 Blackwell 96GB
# Time budget: ~7h (within 12h buffer)
#   - Training: ~5h (2 arms × 16 epochs)
#   - Generation: ~1.5h (1024 videos × 4 frame counts × 2 arms)
#   - FVD evaluation: ~0.5h
#
# Usage:
#   bash scripts/video_temporal/run_phase23_fvd_verify.sh
#
#   # Quick pipeline test first (strongly recommended):
#   bash scripts/video_temporal/run_phase23_fvd_verify.sh --quick
#
#   # Eval-only on existing checkpoints:
#   WORK_DIR=results/.../existing_dir \
#     bash scripts/video_temporal/run_phase23_fvd_verify.sh --eval-only
# ==========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/video_temporal/generated/moving_mnist_medium}"
PROFILE="${PROFILE:-blackwell96}"
EPOCHS="${EPOCHS:-16}"
N_GENERATE="${N_GENERATE:-1024}"
GEN_BATCH="${GEN_BATCH:-16}"
SEED="${SEED:-42}"

STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/results/supporting_video/phase23_fvd_verify/$STAMP}"
mkdir -p "$WORK_DIR"

echo "=========================================="
echo " Phase 23: FVD Verification (Single Seed)"
echo "=========================================="
echo "  root=$ROOT_DIR"
echo "  data_dir=$DATA_DIR"
echo "  profile=$PROFILE epochs=$EPOCHS"
echo "  n_generate=$N_GENERATE gen_batch=$GEN_BATCH"
echo "  seed=$SEED"
echo "  work_dir=$WORK_DIR"
echo ""

# Step 0: Verify data exists
if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
    echo "[step 0] Preparing Moving MNIST data..."
    python3 "$ROOT_DIR/scripts/data_prep/prepare_moving_mnist_video.py"
fi
echo "[step 0] Data verified: $DATA_DIR/manifest.json"

# Step 1: Try to download I3D (optional, fallback to pixel FVD)
I3D_DIR="$ROOT_DIR/data/video_temporal/external"
I3D_PATH="$I3D_DIR/i3d_torchscript.pt"
if [[ ! -f "$I3D_PATH" ]]; then
    echo "[step 1] Attempting I3D download (optional)..."
    mkdir -p "$I3D_DIR"
    timeout 15 python3 -c "
import urllib.request, os, sys, socket
socket.setdefaulttimeout(10)
url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
dst = '$I3D_PATH'
try:
    urllib.request.urlretrieve(url, dst)
    print(f'  Downloaded I3D: {os.path.getsize(dst)/1e6:.1f}MB')
except Exception as e:
    print(f'  I3D download failed: {e}')
    print('  Will use pixel-space FVD (still valid for comparison)')
" 2>&1 || echo "  I3D download skipped (network unavailable or timeout)"
else
    echo "[step 1] I3D model found: $I3D_PATH"
fi

# Step 2: Run verification
echo ""
echo "[step 2] Starting FVD verification run..."
python3 "$ROOT_DIR/scripts/video_temporal/run_phase23_fvd_verify.py" \
    --data-dir "$DATA_DIR" \
    --profile "$PROFILE" \
    --variants "geo_k16,evq_k16" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --n-generate "$N_GENERATE" \
    --gen-batch-size "$GEN_BATCH" \
    --i3d-path "$I3D_PATH" \
    --work-dir "$WORK_DIR" \
    "$@" \
    2>&1 | tee "$WORK_DIR/fvd_verify.log"

echo ""
echo "=========================================="
echo " FVD Verification Complete"
echo "=========================================="
echo "  Results: $WORK_DIR/fvd_verify_summary.json"
echo "  Log:     $WORK_DIR/fvd_verify.log"
echo ""
echo "  Next steps:"
echo "    If EVQ wins -> replicate with seed 137:"
echo "      SEED=137 WORK_DIR=$WORK_DIR/../pass2_seed137 \\"
echo "        bash scripts/video_temporal/run_phase23_fvd_verify.sh"
echo ""
echo "    Quick inspection:"
echo "      python3 -c \"import json; d=json.load(open('$WORK_DIR/fvd_verify_summary.json')); print('EVQ wins:', d['evq_fvd_wins'], '/', d['evq_fvd_wins']+d['geo_fvd_wins'])\""
