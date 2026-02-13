#!/usr/bin/env bash
set -euo pipefail

# Guard script: if the main 350M run crashes after finishing geo_500k but before
# running hybrid, automatically resume with hybrid-only.
#
# This does NOT interrupt a running process.
#
# Logs: /opt/dfrope/results/350m_final/guard.log

PY=/usr/local/miniconda3/envs/py312/bin/python
WORK_DIR=/opt/dfrope/results/350m_final
LOG="$WORK_DIR/guard.log"
MAIN_SCRIPT=/opt/dfrope/run_350m_final.py

mkdir -p "$WORK_DIR"

echo "[guard] start $(date -Is)" >> "$LOG"

while true; do
  if pgrep -f "/opt/dfrope/run_350m_final.py" >/dev/null 2>&1; then
    echo "[guard] $(date -Is) main running" >> "$LOG"
    sleep 600
    continue
  fi

  # Main not running.
  echo "[guard] $(date -Is) main NOT running" >> "$LOG"

  RES="$WORK_DIR/results.json"
  if [[ ! -f "$RES" ]]; then
    echo "[guard] results.json missing, nothing to resume" >> "$LOG"
    sleep 600
    continue
  fi

  # Check which experiments are present.
  status="$($PY - <<PY
import json
p="$RES"
obj=json.load(open(p))
exp=obj.get('experiments',{})
print(('geo_500k' in exp), ('hybrid_a0.2_t100k' in exp))
PY
)"

  if [[ "$status" == "True True" ]]; then
    echo "[guard] both experiments present; done" >> "$LOG"
    break
  fi

  if [[ "$status" == "True False" ]]; then
    echo "[guard] geo done, hybrid missing -> launching hybrid-only resume" >> "$LOG"
    # Launch a new invocation that will run hybrid only.
    nohup "$PY" -u "$MAIN_SCRIPT" --only_hybrid >> "$WORK_DIR/run_hybrid_resume.log" 2>&1 &
    echo "[guard] launched pid=$!" >> "$LOG"
    sleep 60
    continue
  fi

  echo "[guard] geo not done yet (or results incomplete): status=$status" >> "$LOG"
  sleep 600
done

echo "[guard] end $(date -Is)" >> "$LOG"
