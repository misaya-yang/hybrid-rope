#!/usr/bin/env bash
# Auto-queue post-main-sweep tasks:
# 1. Wait for main sweep to finish (polls for master process)
# 2. Run analysis immediately (zero GPU)
# 3. Run basin refinement (adds 5 more τ at L=1024, ~7h)
# 4. Re-run analysis with refined data
#
# Usage:
#   nohup bash scripts/m4_max_36gb/queue_after_main_sweep.sh > results/weekend_sweep/logs/queue.log 2>&1 &

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=============================================================="
echo " Queue after main sweep — started $(date)"
echo " Waiting for main sweep to finish..."
echo "=============================================================="

# Wait for master bash of main sweep to end
while pgrep -f "weekend_tau_theory_sweep.sh" > /dev/null; do
    sleep 300  # check every 5 min
done
echo " Main sweep ended at $(date)"

# Also wait for python children to finish (just in case)
while pgrep -f "run_evq_sweep.py" > /dev/null; do
    sleep 60
done

echo ""
echo "=============================================================="
echo " Running initial analysis..."
echo "=============================================================="
conda run -n aidemo --no-capture-output python -u scripts/m4_max_36gb/analyze_weekend_sweep.py --plot 2>&1 | tee -a "results/weekend_sweep/logs/analysis_initial.log"

echo ""
echo "=============================================================="
echo " Launching basin refinement (L=1024 denser sampling)..."
echo "=============================================================="
bash scripts/m4_max_36gb/post_sweep_basin_refine.sh

echo ""
echo "=============================================================="
echo " Re-running analysis with refined data..."
echo "=============================================================="
conda run -n aidemo --no-capture-output python -u scripts/m4_max_36gb/analyze_weekend_sweep.py --plot 2>&1 | tee -a "results/weekend_sweep/logs/analysis_final.log"

echo ""
echo "=============================================================="
echo " All done: $(date)"
echo "=============================================================="
