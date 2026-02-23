#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <train_log_path>" >&2
  exit 1
fi

train_log="$1"

while pgrep -f run_llama8b_fair_suite.py >/dev/null; do
  echo "=== WATCHDOG ==="
  date
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,pstate --format=csv,noheader
  tail -n 5 "$train_log" || true
  echo
  sleep 60
done

echo "watchdog_exit: no run_llama8b_fair_suite.py process"
date
