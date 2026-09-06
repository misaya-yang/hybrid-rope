#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
# Phase Orbits overnight experiment
#
# Strategy:
#   1. Generate Phase Orbits dataset (medium difficulty)
#   2. Train geo_k16 vs evq_k16 from scratch (seed 42)
#   3. Evaluate: PPL + FVD (I3D + VideoMAE) + temporal metrics
#   4. If EVQ wins → replicate with seed 137
#
# Time budget (R6000 Blackwell 96GB):
#   Data gen:   ~15 min
#   Training:   ~5h (2 arms × 16 epochs)
#   FVD eval:   ~2h (256 videos × 4 frame counts × 2 arms)
#   Total:      ~7-8h
#
# Usage:
#   # Standard overnight run:
#   nohup bash scripts/video_temporal/run_phase_orbits_overnight.sh \
#       > logs/phase_orbits_$(date +%Y%m%d).log 2>&1 &
#
#   # Quick pipeline test first:
#   bash scripts/video_temporal/run_phase_orbits_overnight.sh --quick
#
#   # With 8x extrapolation:
#   EVAL_FRAMES="32,64,96,128,192,256" \
#       bash scripts/video_temporal/run_phase_orbits_overnight.sh
# ==========================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DIFFICULTY="${DIFFICULTY:-medium}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/video_temporal/generated/phase_orbits_${DIFFICULTY}}"
PROFILE="${PROFILE:-blackwell96}"
EPOCHS="${EPOCHS:-16}"
N_GENERATE="${N_GENERATE:-256}"
GEN_BATCH="${GEN_BATCH:-16}"
SEED="${SEED:-42}"
EVAL_FRAMES="${EVAL_FRAMES:-}"
EXTRA_EVAL_FRAMES="${EXTRA_EVAL_FRAMES:-256}"

STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/results/supporting_video/phase_orbits_${DIFFICULTY}/$STAMP}"
mkdir -p "$WORK_DIR"

echo "=========================================="
echo " Phase Orbits: Overnight Experiment"
echo "=========================================="
echo "  difficulty=$DIFFICULTY"
echo "  profile=$PROFILE epochs=$EPOCHS"
echo "  n_generate=$N_GENERATE seed=$SEED"
echo "  work_dir=$WORK_DIR"
echo ""

# Step 0: Generate data
if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
    echo "[step 0] Generating Phase Orbits dataset..."
    python3 "$ROOT_DIR/scripts/data_prep/prepare_phase_orbits.py" \
        --difficulty "$DIFFICULTY" \
        --out-dir "$DATA_DIR" \
        --extra-eval-frames "$EXTRA_EVAL_FRAMES" \
        --overwrite
else
    echo "[step 0] Dataset exists: $DATA_DIR"
fi

# Step 1: Install cd-fvd for VideoMAE FVD (if not already)
echo ""
echo "[step 1] Ensuring cd-fvd is installed..."
pip install cd-fvd 2>/dev/null || pip install cd-fvd --break-system-packages 2>/dev/null || echo "  cd-fvd install skipped"

# Step 2: Train + FVD evaluation
echo ""
echo "[step 2] Starting training + FVD evaluation..."

EVAL_FRAMES_ARG=""
if [[ -n "$EVAL_FRAMES" ]]; then
    EVAL_FRAMES_ARG="--eval-frames $EVAL_FRAMES"
fi

python3 "$ROOT_DIR/scripts/video_temporal/run_phase23_fvd_verify.py" \
    --data-dir "$DATA_DIR" \
    --profile "$PROFILE" \
    --variants "geo_k16,evq_k16" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --n-generate "$N_GENERATE" \
    --gen-batch-size "$GEN_BATCH" \
    --work-dir "$WORK_DIR" \
    $EVAL_FRAMES_ARG \
    "$@" \
    2>&1 | tee "$WORK_DIR/phase_orbits.log"

# Step 3: Run VideoMAE FVD on generated videos (if cd-fvd available)
echo ""
echo "[step 3] Computing VideoMAE FVD..."
python3 -c "
import json, sys
from pathlib import Path

work_dir = Path('$WORK_DIR')
summary_path = work_dir / 'fvd_verify_summary.json'
if not summary_path.exists():
    print('  No summary found, skipping VideoMAE FVD')
    sys.exit(0)

summary = json.loads(summary_path.read_text())
print(f'  EVQ FVD wins: {summary.get(\"evq_fvd_wins\", \"?\")}/{summary.get(\"evq_fvd_wins\", 0) + summary.get(\"geo_fvd_wins\", 0)}')
print(f'  Results: {summary_path}')
" 2>&1 || true

echo ""
echo "=========================================="
echo " Phase Orbits Complete"
echo "=========================================="
echo "  Results: $WORK_DIR"
echo "  Log:     $WORK_DIR/phase_orbits.log"
echo ""
echo "  If EVQ wins:"
echo "    SEED=137 WORK_DIR=$WORK_DIR/../pass2_seed137 \\"
echo "      bash scripts/video_temporal/run_phase_orbits_overnight.sh"
