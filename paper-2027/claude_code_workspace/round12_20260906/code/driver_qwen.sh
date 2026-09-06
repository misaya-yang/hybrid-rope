#!/bin/bash
# External YaRN-SOTA controls at MATCHED extrapolation factor (user ruling
# 2026-09-06): Qwen2.5 is native 32K, so its own 2x/4x are 65536/131072
# (NOT 8K/16K). Tasks: round-12 EXT ruler_single_key rows built with the
# Qwen tokenizer (native token counts, same task design as Track A ruler
# rows). YaRN is injected via HF rope_scaling = the shipped 128K recipe
# (native 32K + static YaRN factor 4, no fine-tuning). SDPA flash/efficient
# kernels are enforced inside qwen_eval.py. OOM rows are recorded, not fatal.
set -u
B12=/root/autodl-tmp/claude_round12_20260906
TASKS=$B12/tasks/round12_tasks_ext_qwen.jsonl
PY=/root/miniconda3/bin/python
LOG=$B12/track_a/driver_qwen.log
SUM=$B12/track_a/driver_qwen_summary.txt
mkdir -p $B12/track_a
cd $B12/code || exit 91
echo "QWEN2_START $(date -u +%FT%TZ)" >> $LOG
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
M7=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
M15=/root/autodl-tmp/qwen25_1p5b_32k
M05=/root/autodl-tmp/qwen25_0p5b_32k_k32
# In-distribution sanity first: 7B at its native 32K (no YaRN). If the task
# design itself were incomprehensible this fails; if only 64K/128K fail, the
# collapse is genuine extrapolation failure.
if [ -d "$M7" ] && [ -f "$B12/tasks/round12_tasks_ext_qwen_native.jsonl" ]; then
  run_step qwen25_7b_native32k $PY qwen_eval.py --model $M7 --model-id qwen25_7b \
    --tasks $B12/tasks/round12_tasks_ext_qwen_native.jsonl --native-tasks \
    --chat-template --families ruler_single_key --lengths 32768 \
    --output $B12/track_a/qwen25_7b_native32k
fi
# 7B at its own 4x next: headline benchmark.
if [ -d "$M7" ] && [ -f "$TASKS" ]; then
  run_step qwen25_7b_4x $PY qwen_eval.py --model $M7 --model-id qwen25_7b \
    --tasks $TASKS --native-tasks --chat-template --families ruler_single_key --lengths 131072 \
    --yarn-factor 4.0 --original-max-pos 32768 \
    --output $B12/track_a/qwen25_7b_ext4x
  run_step qwen25_7b_2x $PY qwen_eval.py --model $M7 --model-id qwen25_7b \
    --tasks $TASKS --native-tasks --chat-template --families ruler_single_key --lengths 65536 \
    --yarn-factor 2.0 --original-max-pos 32768 \
    --output $B12/track_a/qwen25_7b_ext2x
fi
if [ -d "$M15" ] && [ -f "$TASKS" ]; then
  run_step qwen25_1p5b_4x $PY qwen_eval.py --model $M15 --model-id qwen25_1p5b \
    --tasks $TASKS --native-tasks --chat-template --families ruler_single_key --lengths 131072 \
    --yarn-factor 4.0 --original-max-pos 32768 \
    --output $B12/track_a/qwen25_1p5b_ext4x
  run_step qwen25_1p5b_2x $PY qwen_eval.py --model $M15 --model-id qwen25_1p5b \
    --tasks $TASKS --native-tasks --chat-template --families ruler_single_key --lengths 65536 \
    --yarn-factor 2.0 --original-max-pos 32768 \
    --output $B12/track_a/qwen25_1p5b_ext2x
fi
if [ -d "$M05" ] && [ -f "$TASKS" ]; then
  run_step qwen25_0p5b_4x $PY qwen_eval.py --model $M05 --model-id qwen25_0p5b \
    --tasks $TASKS --native-tasks --chat-template --families ruler_single_key --lengths 131072 \
    --yarn-factor 4.0 --original-max-pos 32768 \
    --output $B12/track_a/qwen25_0p5b_ext4x
  run_step qwen25_0p5b_2x $PY qwen_eval.py --model $M05 --model-id qwen25_0p5b \
    --tasks $TASKS --native-tasks --chat-template --families ruler_single_key --lengths 65536 \
    --yarn-factor 2.0 --original-max-pos 32768 \
    --output $B12/track_a/qwen25_0p5b_ext2x
fi
echo "QWEN2_END $(date -u +%FT%TZ)" >> $LOG
