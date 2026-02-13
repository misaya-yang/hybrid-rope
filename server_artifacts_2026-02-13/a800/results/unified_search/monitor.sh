#!/usr/bin/env bash
set -euo pipefail
split="B"
log="/opt/dfrope/results/unified_search/log_${split}.txt"
mon="/opt/dfrope/results/unified_search/monitor_${split}.log"
res="/opt/dfrope/results/unified_search/results_${split}.json"
echo "[monitor] start $(date -Is)" >> "$mon"
while true; do
  pid="$(pgrep -f "/opt/dfrope/unified_search.py ${split}" || true)"
  echo "--- $(date -Is) pid=${pid:-NONE}" >> "$mon"
  nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | head -n 1 >> "$mon" || true
  if [ -f "$res" ]; then
    echo "results_bytes $(wc -c < "$res")" >> "$mon"
  else
    echo "results_bytes MISSING" >> "$mon"
  fi
  if [ -f "$log" ]; then
    tail -n 25 "$log" >> "$mon" || true
    if grep -q "Traceback (most recent call last)" "$log"; then
      echo "[monitor] TRACEBACK detected, stopping." >> "$mon"
      break
    fi
    if grep -qi "out of memory\|CUDA out of memory\|killed" "$log"; then
      echo "[monitor] OOM/KILLED detected, stopping." >> "$mon"
      break
    fi
  else
    echo "log MISSING" >> "$mon"
  fi
  if [ -z "$pid" ]; then
    echo "[monitor] PROCESS EXITED, stopping." >> "$mon"
    break
  fi
  sleep 100
done
echo "[monitor] end $(date -Is)" >> "$mon"
