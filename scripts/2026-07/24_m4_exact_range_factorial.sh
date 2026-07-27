#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELF="$ROOT/scripts/2026-07/24_m4_exact_range_factorial.sh"
WORK_ROOT="${M4_EXACT_RANGE_ROOT:-$ROOT/results/theory/phase16_exact_range_factorial_m4_20260724}"
PYTHON="${M4_EXACT_RANGE_PYTHON:-$HOME/miniconda3/envs/aidemo/bin/python}"
RUNNER="$ROOT/scripts/core_text_phases/phase16_exact_range_factorial_m4.py"
LOG="$WORK_ROOT/launcher.log"
PID_FILE="$WORK_ROOT/launcher.pid"
LABEL="com.evq.phase16-exact-range-m4"
SERVICE="gui/$(id -u)/$LABEL"
CAPTURE_LABEL="com.evq.phase16-exact-range-capture"
CAPTURE_SERVICE="gui/$(id -u)/$CAPTURE_LABEL"
FOLLOWUP_LABEL="com.evq.phase16-exact-range-followup"
FOLLOWUP_SERVICE="gui/$(id -u)/$FOLLOWUP_LABEL"
FOLLOWUP="$ROOT/scripts/core_text_phases/phase16_exact_range_followup_m4.py"

mkdir -p "$WORK_ROOT"

case "${1:-status}" in
  start)
    if launchctl print "$SERVICE" 2>/dev/null | grep -q 'state = running'; then
      echo "already running: $SERVICE"
      exit 0
    fi
    launchctl remove "$LABEL" 2>/dev/null || true
    cd "$ROOT"
    launchctl submit -l "$LABEL" -o "$LOG" -e "$LOG" -- \
      /usr/bin/caffeinate -ims "$PYTHON" "$RUNNER" --mode run --work-root "$WORK_ROOT"
    sleep 1
    launchctl print "$SERVICE" | awk '/pid = / {print $3; exit}' >"$PID_FILE"
    echo "started: service=$SERVICE pid=$(cat "$PID_FILE") log=$LOG"
    ;;
  status)
    service="$(launchctl print "$SERVICE" 2>/dev/null || true)"
    pid="$(printf '%s\n' "$service" | awk '/pid = / {print $3; exit}')"
    completed="$(find "$WORK_ROOT/runs" -name result.json 2>/dev/null | wc -l | tr -d ' ')"
    if printf '%s\n' "$service" | grep -q 'state = running'; then state=running; else state=stopped; fi
    echo "state=$state pid=${pid:-none} completed=$completed/180"
    tail -n 12 "$LOG" 2>/dev/null || true
    ;;
  stop)
    lock_pid="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' \
      "$WORK_ROOT/sweep.lock" 2>/dev/null || true)"
    [[ -n "$lock_pid" ]] && kill -TERM "$lock_pid" 2>/dev/null || true
    [[ -z "$lock_pid" ]] && launchctl remove "$LABEL" 2>/dev/null || true
    echo "stop requested for pid=${lock_pid:-none}; current run will checkpoint"
    ;;
  report)
    cd "$ROOT"
    "$PYTHON" "$RUNNER" --mode report --work-root "$WORK_ROOT"
    ;;
  start-capture)
    if launchctl print "$CAPTURE_SERVICE" 2>/dev/null | grep -q 'state = running'; then
      echo "already running: $CAPTURE_SERVICE"
      exit 0
    fi
    launchctl remove "$CAPTURE_LABEL" 2>/dev/null || true
    launchctl submit -l "$CAPTURE_LABEL" -o "$WORK_ROOT/milestone_capture.log" \
      -e "$WORK_ROOT/milestone_capture.log" -- \
      "$PYTHON" "$FOLLOWUP" --mode capture --work-root "$WORK_ROOT"
    echo "started: service=$CAPTURE_SERVICE"
    ;;
  start-followup)
    if launchctl print "$FOLLOWUP_SERVICE" 2>/dev/null | grep -q 'state = running'; then
      echo "already running: $FOLLOWUP_SERVICE"
      exit 0
    fi
    launchctl remove "$FOLLOWUP_LABEL" 2>/dev/null || true
    launchctl submit -l "$FOLLOWUP_LABEL" -o "$WORK_ROOT/followup.log" \
      -e "$WORK_ROOT/followup.log" -- /bin/bash "$SELF" followup-worker
    echo "started: service=$FOLLOWUP_SERVICE"
    ;;
  followup-worker)
    while [[ "$(find "$WORK_ROOT/runs" -name result.json 2>/dev/null | wc -l | tr -d ' ')" -lt 180 ]]; do
      if ! launchctl print "$SERVICE" 2>/dev/null | grep -q 'state = running'; then
        echo "main sweep stopped before 180 results" >&2
        exit 1
      fi
      sleep 300
    done
    cd "$ROOT"
    exec "$PYTHON" "$FOLLOWUP" --mode augment --work-root "$WORK_ROOT"
    ;;
  followup-status)
    launchctl print "$FOLLOWUP_SERVICE" 2>/dev/null | rg 'state =|pid =' | head || true
    tail -n 20 "$WORK_ROOT/followup.log" 2>/dev/null || true
    ;;
  *)
    echo "usage: $0 {start|status|stop|report|start-capture|start-followup|followup-status}" >&2
    exit 2
    ;;
esac
