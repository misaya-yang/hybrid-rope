#!/bin/bash
# 500M EVQ from-scratch experiment — server execution script
# Hardware: RTX PRO 6000 Blackwell 96GB
# Expected time: ~12-15 hours total (2 training runs + passkey eval)
#
# Usage:
#   cd /root/autodl-tmp/dfrope/hybrid-rope
#   bash scripts/m4_evq_sweep/run_500m_server.sh 2>&1 | tee ~/500m_experiment.log

set -euo pipefail

WORK_DIR="${HOME}/evq_sweep_500m"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"

echo "============================================================"
echo "  500M EVQ FROM-SCRATCH EXPERIMENT"
echo "  Work dir: ${WORK_DIR}"
echo "  Project:  ${PROJECT_DIR}"
echo "  Started:  $(date)"
echo "============================================================"

# ---- Phase 1: Train 500M models (τ=0.0 geometric baseline, τ=1.5 EVQ) ----
echo ""
echo "[Phase 1] Training 500M models — τ=0.0, τ=1.5, seed=42"
echo "Expected: ~10-14 hours"
echo ""

python scripts/m4_evq_sweep/run_evq_sweep.py \
    --tier 500m \
    --taus 0.0,1.5 \
    --seeds 42 \
    --work_dir "${WORK_DIR}" \
    --resume

echo ""
echo "[Phase 1] DONE. Results: ${WORK_DIR}/results_final.json"

# ---- Phase 2: Passkey retrieval on 500M checkpoints ----
echo ""
echo "[Phase 2] Passkey retrieval evaluation"
echo "Context lengths: 1024, 2048, 4096, 8192, 16384"
echo ""

python scripts/m4_evq_sweep/eval_passkey.py \
    --work_dir "${WORK_DIR}" \
    --tiers 500m \
    --taus 0.0,1.5 \
    --seeds 42 \
    --context_lengths 1024,2048,4096,8192,16384 \
    --depth_ratios 0.1,0.3,0.5,0.7,0.9 \
    --num_trials 25

echo ""
echo "============================================================"
echo "  ALL DONE"
echo "  PPL results:     ${WORK_DIR}/results_final.json"
echo "  Passkey results: ${WORK_DIR}/passkey_results.json"
echo "  Finished:        $(date)"
echo "============================================================"
