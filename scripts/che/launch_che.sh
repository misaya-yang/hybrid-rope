#!/bin/bash
# CHE Benchmark launcher for 5090 server
# Usage: bash launch_che.sh [pilot|phase1|phase1_full|ablation]

set -e
export PATH="/root/miniconda3/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="/root/autodl-tmp/che_benchmark"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$WORK_DIR" "$LOG_DIR"

SCRIPT="$SCRIPT_DIR/run_che.py"

case "${1:-pilot}" in
  pilot)
    echo "=== PILOT: Even Pairs, all 6 methods, seed=42, 2K steps ==="
    for method in nope rope_geo rope_evq kerple dape evq_kerple; do
      echo ">>> $method"
      python "$SCRIPT" --task even_pairs --method "$method" --seed 42 \
        --work_dir "$WORK_DIR" --pilot --resume 2>&1 | tee "$LOG_DIR/pilot_${method}.log"
    done
    echo "=== PILOT COMPLETE ==="
    ;;

  phase1)
    echo "=== PHASE 1: Geo vs EVQ, all 15 tasks, 3 seeds, 200K steps ==="
    for seed in 42 123 7; do
      for method in rope_geo rope_evq; do
        for task in parity_check even_pairs modular_arithmetic cycle_navigation \
          modular_arithmetic_brackets reverse_string solve_equation stack_manipulation \
          binary_addition binary_multiplication bucket_sort compute_sqrt \
          duplicate_string missing_duplicate odds_first; do
          echo ">>> ${task} / ${method} / seed=${seed}"
          python "$SCRIPT" --task "$task" --method "$method" --seed "$seed" \
            --work_dir "$WORK_DIR" --resume 2>&1 | tee -a "$LOG_DIR/phase1_s${seed}.log"
        done
      done
    done
    echo "=== PHASE 1 COMPLETE ==="
    ;;

  phase1_full)
    echo "=== PHASE 1 FULL: all 6 methods, all 15 tasks, seed=42, 200K steps ==="
    for method in nope rope_geo rope_evq kerple dape evq_kerple; do
      for task in parity_check even_pairs modular_arithmetic cycle_navigation \
        modular_arithmetic_brackets reverse_string solve_equation stack_manipulation \
        binary_addition binary_multiplication bucket_sort compute_sqrt \
        duplicate_string missing_duplicate odds_first; do
        echo ">>> ${task} / ${method}"
        python "$SCRIPT" --task "$task" --method "$method" --seed 42 \
          --work_dir "$WORK_DIR" --resume 2>&1 | tee -a "$LOG_DIR/phase1_full.log"
      done
    done
    echo "=== PHASE 1 FULL COMPLETE ==="
    ;;

  ablation)
    echo "=== ABLATION: all 6 methods, 3 seeds, all 15 tasks ==="
    for seed in 42 123 7; do
      for method in nope rope_geo rope_evq kerple dape evq_kerple; do
        python "$SCRIPT" --task all --method "$method" --seed "$seed" \
          --work_dir "$WORK_DIR" --resume 2>&1 | tee -a "$LOG_DIR/ablation_s${seed}.log"
      done
    done
    echo "=== ABLATION COMPLETE ==="
    ;;

  *)
    echo "Usage: $0 [pilot|phase1|phase1_full|ablation]"
    exit 1
    ;;
esac
