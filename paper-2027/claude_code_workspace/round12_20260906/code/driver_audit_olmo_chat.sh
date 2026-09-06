#!/bin/bash
# Round 12 OLMo eval-mode audit (user directive 2026-09-06 #3): Track A was
# run entirely in raw completion mode; Qwen2.5-Instruct collapses to filler
# repetition in raw mode, so audit whether OLMo-2-Instruct results are
# affected. Raw-mode receipts already exist ($B12/track_a/olmo{1b,7b}_{N,Z},
# 512 rows each); this driver runs the chat-template variants of the SAME
# cells: {1B,7B} x {N,Z}, single_evidence 2048/8192/16384 (in-distribution
# + own 2x + own 4x). Same tasks file, same scoring, same frozen tables.
# Two 2-row smokes gate the full runs (template/decoding sanity).
set -u
B12=/root/autodl-tmp/claude_round12_20260906
M1=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
M7=/root/autodl-tmp/models/OLMo-2-1124-7B-Instruct
PY=/root/miniconda3/bin/python
OUT=$B12/track_a_audit
LOG=$OUT/audit.log
SUM=$OUT/audit_summary.txt
mkdir -p $OUT
cd $B12/code || exit 91
echo "AUDIT_START $(date -u +%FT%TZ)" >> $LOG
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

# ---- smoke gates (2 rows each) --------------------------------------------
run_step smoke_1b_Z $PY track_a_eval.py --model $M1 --model-id olmo1b \
  --tables $B12/tables --arm Z --tasks $B12/tasks/round12_tasks.jsonl \
  --families single_evidence --lengths 2048 --chat-template --smoke \
  --output $OUT/smoke_1b_Z_chat || exit 1
run_step smoke_7b_Z $PY track_a_eval.py --model $M7 --model-id olmo7b \
  --tables $B12/tables --arm Z --tasks $B12/tasks/round12_tasks.jsonl \
  --families single_evidence --lengths 2048 --chat-template --smoke \
  --output $OUT/smoke_7b_Z_chat || exit 1

# ---- full audit cells ------------------------------------------------------
run_step audit_1b_N $PY track_a_eval.py --model $M1 --model-id olmo1b \
  --tables $B12/tables --arm N --tasks $B12/tasks/round12_tasks.jsonl \
  --families single_evidence --lengths 2048 8192 16384 --chat-template \
  --output $OUT/olmo1b_N_chat
run_step audit_1b_Z $PY track_a_eval.py --model $M1 --model-id olmo1b \
  --tables $B12/tables --arm Z --tasks $B12/tasks/round12_tasks.jsonl \
  --families single_evidence --lengths 2048 8192 16384 --chat-template \
  --output $OUT/olmo1b_Z_chat
run_step audit_7b_N $PY track_a_eval.py --model $M7 --model-id olmo7b \
  --tables $B12/tables --arm N --tasks $B12/tasks/round12_tasks.jsonl \
  --families single_evidence --lengths 2048 8192 16384 --chat-template \
  --output $OUT/olmo7b_N_chat
run_step audit_7b_Z $PY track_a_eval.py --model $M7 --model-id olmo7b \
  --tables $B12/tables --arm Z --tasks $B12/tasks/round12_tasks.jsonl \
  --families single_evidence --lengths 2048 8192 16384 --chat-template \
  --output $OUT/olmo7b_Z_chat

echo "AUDIT_COMPLETE $(date -u +%FT%TZ)" >> $LOG
echo "AUDIT_QUEUE_COMPLETE" >> $SUM
