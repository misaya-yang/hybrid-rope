#!/bin/bash
# 500M Minimal Experiment: geometric vs EVQ-Cosh (+ PI inference-time baseline)
# 训练 2 runs，评测 3 methods
# 预计耗时: ~16-20 小时，成本 ~100-120 元 (6元/小时)
#
# Usage:
#   cd /root/autodl-tmp/dfrope/hybrid-rope
#   bash scripts/m4_evq_sweep/run_500m_minimal.sh 2>&1 | tee ~/500m_experiment.log

set -euo pipefail

WORK_DIR="${HOME}/evq_500m_sweep"
TIER="500m"
SEED=42
BASE=500000.0
DATASET="fineweb-edu"

echo "=========================================="
echo "  500M Minimal Experiment"
echo "  Train: geometric (τ=0.0) + EVQ-Cosh (τ=1.0)"
echo "  Eval:  geometric + PI (inference-time) + EVQ-Cosh"
echo "  Dataset: ${DATASET}"
echo "  Work dir: ${WORK_DIR}"
echo "  Started: $(date)"
echo "=========================================="

python scripts/m4_evq_sweep/run_evq_sweep.py \
    --tier ${TIER} --taus 0.0,1.0 \
    --seeds ${SEED} --base ${BASE} \
    --dataset ${DATASET} --work_dir ${WORK_DIR} --resume

echo ""
echo "=========================================="
echo "  ALL DONE"
echo "  Results: ${WORK_DIR}/results_final.json"
echo "  Finished: $(date)"
echo "=========================================="
