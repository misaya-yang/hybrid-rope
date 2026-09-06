#!/bin/bash
# ============================================================================
# Video DiT Temporal Extrapolation — Overnight Launcher
#
# WHY DiT:
#   AR VideoGPT shows 27% PPL advantage but only 1.5% FVD advantage because
#   top-k sampling compresses distributional differences. DiT uses the full
#   learned distribution during denoising, so EVQ's frequency allocation
#   advantage should directly translate to generation quality.
#
# STRATEGY:
#   1. Quick sanity check (10min): verify model trains, generates, FVD computes
#   2. Full single-seed verification (2-4h): geo vs evq, 128 generated videos
#   3. Multi-seed (optional, 8-12h): 3 seeds for statistical significance
#
# Usage:
#   bash scripts/video_temporal/run_dit_overnight.sh          # full single-seed
#   bash scripts/video_temporal/run_dit_overnight.sh quick     # quick sanity
#   bash scripts/video_temporal/run_dit_overnight.sh multi     # 3 seeds
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-single}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="results/video_dit/${TIMESTAMP}"

echo "============================================"
echo "  Video DiT Temporal Extrapolation"
echo "  Mode: ${MODE}"
echo "  Work dir: ${WORK_DIR}"
echo "  Started: $(date)"
echo "============================================"

# Ensure cd-fvd is installed
echo ""
echo "Checking cd-fvd..."
python -c "import cdfvd" 2>/dev/null || {
    echo "Installing cd-fvd..."
    pip install cd-fvd
}

# GPU info
echo ""
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected!')
"

case "$MODE" in
    quick)
        echo ""
        echo "=== QUICK SANITY CHECK (10min) ==="
        python scripts/video_temporal/run_dit_temporal.py \
            --quick \
            --seed 42 \
            --n_gen 16 \
            --work_dir "${WORK_DIR}"
        ;;

    single)
        echo ""
        echo "=== SINGLE-SEED VERIFICATION (~2-4h) ==="
        python scripts/video_temporal/run_dit_temporal.py \
            --seed 42 \
            --n_gen 128 \
            --work_dir "${WORK_DIR}"
        ;;

    multi)
        echo ""
        echo "=== MULTI-SEED EXPERIMENT (~8-12h) ==="
        python scripts/video_temporal/run_dit_temporal.py \
            --seeds 42,137,256 \
            --n_gen 128 \
            --work_dir "${WORK_DIR}"
        ;;

    evq_only)
        echo ""
        echo "=== EVQ ONLY (1-2h) ==="
        python scripts/video_temporal/run_dit_temporal.py \
            --method evq \
            --seed 42 \
            --n_gen 128 \
            --work_dir "${WORK_DIR}"
        ;;

    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: $0 [quick|single|multi|evq_only]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  COMPLETED: $(date)"
echo "  Results: ${WORK_DIR}"
echo "============================================"

# Print summary if exists
if [ -f "${WORK_DIR}/summary.json" ]; then
    echo ""
    echo "=== SUMMARY ==="
    python -c "
import json
with open('${WORK_DIR}/summary.json') as f:
    data = json.load(f)
for r in data:
    method = r['method']
    seed = r['seed']
    loss = r.get('final_loss', 'N/A')
    vmae_128 = r.get('fvd_128f_yarn', {}).get('videomae_fvd', 'N/A')
    i3d_128 = r.get('fvd_128f_yarn', {}).get('i3d_fvd', 'N/A')
    print(f'  {method} seed={seed}: loss={loss:.6f}  VideoMAE_FVD(128f)={vmae_128}  I3D_FVD(128f)={i3d_128}')
"
fi
