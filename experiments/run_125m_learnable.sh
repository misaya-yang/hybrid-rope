#!/bin/bash
# Launch 125M learnable-τ experiment matrix on server.
# Usage: bash experiments/run_125m_learnable.sh

set -euo pipefail

WORK_DIR="${1:-${HOME}/evq_125m_learnable}"

echo "Starting 125M learnable-τ experiment matrix"
echo "Work dir: ${WORK_DIR}"
echo "Time: $(date)"

PYTHONUNBUFFERED=1 python experiments/run_125m_learnable.py \
    --work_dir "${WORK_DIR}" \
    --dataset fineweb-edu \
    --resume \
    2>&1 | tee "${WORK_DIR}/run.log"

echo "Done: $(date)"
echo "Plot: python experiments/plot_tau_trajectory.py --work_dir ${WORK_DIR}"
