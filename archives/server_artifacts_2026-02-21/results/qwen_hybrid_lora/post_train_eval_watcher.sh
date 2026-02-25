#!/usr/bin/env bash
set -euo pipefail
PID="$1"
LOG="/opt/dfrope/results/qwen_hybrid_lora/eval_watcher.log"
echo "[Watcher] start $(date '+%F %T') pid=${PID}" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
done
for i in $(seq 1 120); do
  if [ -d /opt/dfrope/results/qwen_hybrid_lora/final_lora ]; then
    break
  fi
  sleep 30
done
if [ ! -d /opt/dfrope/results/qwen_hybrid_lora/final_lora ]; then
  echo "[Watcher] final_lora missing after wait" >> "$LOG"
  exit 1
fi
rm -f /opt/dfrope/results/qwen_hybrid_lora/eval_suite.json || true
echo "[Watcher] launch eval $(date '+%F %T')" >> "$LOG"
HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python -u /opt/dfrope/run_qwen_eval_suite.py >> "$LOG" 2>&1
echo "[Watcher] done $(date '+%F %T')" >> "$LOG"
