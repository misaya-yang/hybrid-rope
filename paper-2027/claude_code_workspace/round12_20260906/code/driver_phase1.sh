#!/bin/bash
# Round 12 Phase 1 driver — Track A 1B static matrix {N,Z,Y,M}.
# Sequential, single GPU process. Round12 tasks are split=train rows:
# NOT byte-identical to round-11 receipts; every cell runs fresh.
B12=/root/autodl-tmp/claude_round12_20260906
M1=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
PY=/root/miniconda3/bin/python
LOG=$B12/track_a/driver_phase1.log
SUM=$B12/track_a/driver_phase1_summary.txt
mkdir -p $B12/track_a
cd $B12/code || exit 91

run_step() {
  local name=$1; shift
  echo "=== STEP $name START $(date -u +%FT%TZ)" >> $LOG
  "$@" >> $LOG 2>&1
  local rc=$?
  echo "=== STEP $name EXIT=$rc $(date -u +%FT%TZ)" >> $LOG
  echo "$name EXIT=$rc" >> $SUM
  return $rc
}

echo "PHASE1_START $(date -u +%FT%TZ)" >> $LOG
: > $SUM

for A in N Z Y M; do
  run_step track_a_$A $PY track_a_eval.py --model $M1 --model-id olmo1b \
    --tables $B12/tables --arm $A --tasks $B12/tasks/round12_tasks.jsonl \
    --output $B12/track_a/olmo1b_$A
done

echo "PHASE1_END $(date -u +%FT%TZ)" >> $LOG
