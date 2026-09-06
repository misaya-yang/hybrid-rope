#!/bin/bash
# Canonical YaRN control (Y2) — the unfine-tuned SOTA baseline, run under the
# exact Track A matrix (same tasks, lengths, scorers) at 1B and 7B.
set -u
B12=/root/autodl-tmp/claude_round12_20260906
M1=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
M7=/root/autodl-tmp/models/OLMo-2-1124-7B-Instruct
PY=/root/miniconda3/bin/python
LOG=$B12/track_a/driver_y2.log
SUM=$B12/track_a/driver_y2_summary.txt
mkdir -p $B12/track_a
cd $B12/code || exit 91
echo "Y2_START $(date -u +%FT%TZ)" >> $LOG
: > $SUM
run_step() {
  local name=$1; shift
  echo "=== STEP $name START $(date -u +%FT%TZ)" >> $LOG
  "$@" >> $LOG 2>&1
  local rc=$?
  echo "=== STEP $name EXIT=$rc $(date -u +%FT%TZ)" >> $LOG
  echo "$name EXIT=$rc" >> $SUM
  return $rc
}
run_step track_a_1b_Y2 $PY track_a_eval.py --model $M1 --model-id olmo1b \
  --tables $B12/tables_canon --arm Y2 --tasks $B12/tasks/round12_tasks.jsonl \
  --output $B12/track_a/olmo1b_Y2
run_step track_a_7b_Y2 $PY track_a_eval.py --model $M7 --model-id olmo7b \
  --tables $B12/tables_canon --arm Y2 --tasks $B12/tasks/round12_tasks.jsonl \
  --output $B12/track_a/olmo7b_Y2
echo "Y2_END $(date -u +%FT%TZ)" >> $LOG
