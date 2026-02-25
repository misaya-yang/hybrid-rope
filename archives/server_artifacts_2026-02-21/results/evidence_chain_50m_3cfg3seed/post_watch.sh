#!/usr/bin/env bash
set -euo pipefail
PIDF=/root/autodl-tmp/dfrope/hybrid-rope/results/evidence_chain_50m_3cfg3seed/run.pid
LOG=/root/autodl-tmp/dfrope/hybrid-rope/results/evidence_chain_50m_3cfg3seed/post_watch.log
if [ ! -f "$PIDF" ]; then
  echo "missing pid file" >> "$LOG"
  exit 1
fi
PID=$(cat "$PIDF")
echo "watching pid=$PID" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  sleep 30
done
echo "train finished, generating summary" >> "$LOG"
/root/miniconda3/bin/python /root/autodl-tmp/dfrope/hybrid-rope/scripts/summarize_50m_3cfg3seed.py >> "$LOG" 2>&1 || true
echo "done" >> "$LOG"
