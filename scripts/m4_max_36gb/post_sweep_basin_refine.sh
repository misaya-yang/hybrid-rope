#!/usr/bin/env bash
# Post-sweep basin refinement — denser τ sampling at L=1024 where the Basin
# is most theoretically interesting (τ*=2.0, mid-range for L-exponent fit).
#
# Purpose: after main weekend sweep completes, add 5 finer τ values to L=1024
# so the basin curve has enough resolution for publication plots.
#
# Runs:  5 new τ × 3 seeds = 15 runs at L=1024, 25M tokens each
# Time:  ~15 × 27 min = ~7 hours
#
# The main sweep gave 7 τ values spanning [0, 1.7]×τ*; this refinement adds
# 5 more in the [0.5, 1.3]×τ* "zoom" region for a 12-point basin curve.
#
# Usage:
#   bash scripts/m4_max_36gb/post_sweep_basin_refine.sh

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/scripts/supporting_eval:$REPO_ROOT/scripts/core_text_phases:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# L=1024, τ*=2.0
# Main sweep covered r ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.3, 1.7}
# Add r ∈ {0.375, 0.625, 0.875, 1.15, 1.5} for denser basin
# τ values = r × 2.0
TAU_LIST="0.7500,1.2500,1.7500,2.3000,3.0000"

echo "=============================================================="
echo " Post-sweep basin refinement at L=1024"
echo " Started: $(date)"
echo " Additional τ values: ${TAU_LIST}"
echo " Seeds: 42,43,44"
echo " Expected: 15 runs × ~27min = ~7 hours"
echo "=============================================================="

conda run -n aidemo --no-capture-output python -u \
    scripts/core_text_phases/run_evq_sweep.py \
    --tier 50m \
    --taus "${TAU_LIST}" \
    --seeds "42,43,44" \
    --dataset tinystories \
    --base 500000 \
    --train_tokens 25000000 \
    --seq_len 1024 \
    --batch_size 8 \
    --work_dir "results/weekend_sweep/L1024" \
    --resume \
    2>&1 | tee -a "results/weekend_sweep/logs/L1024_refine.log"

echo ""
echo "=============================================================="
echo " Post-sweep refinement complete: $(date)"
echo "=============================================================="
