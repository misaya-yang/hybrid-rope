#!/bin/bash
# OLMO_ZF: cross-family analog of case Z (full-layout exposure), after ZC read.
# ZC (compact-only) far strict 1/32; lenient containment 27/64 unconverted.
# Tests: does long-input exposure convert content at 4x extrapolation?
set -u
PY=/root/miniconda3/bin/python
B=/root/autodl-tmp/ffn_review_execution_20260904
R11=/root/autodl-tmp/claude_round11_olmo_20260905
OLMO=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
OUT=$R11/out_zf
ENGINE=$B/code_release_008/scripts/train/train_single_table_native_constrained.py
TABLE=$B/fixed_controls/Z.npy
GAIN=1.102585782722872
COMMON="--checkpoint $OLMO --checkpoint-contract $B/olmo1485_contract.json --tasks $R11/olmo_tasks/manifest.json --native-pool $B/native_pool_v3/manifest.json --seed 42 --table $TABLE --gain $GAIN --authorized"
mkdir -p $OUT/logs
stage() {
  local name=$1 cap=$2; shift 2
  echo "OLMO_ZF: $name"
  timeout $((cap+120)) "$@" > $OUT/logs/$name.log 2>&1
  local code=$?
  if [ $code -ne 0 ]; then echo "STAGE_FAILED $name exit=$code"; exit $code; fi
}
stage train 7200 $PY $ENGINE train $COMMON --max-seconds 7200 --arm Z --placement all_linear --kl-budget .02 --teacher-cache /root/ffn_review_scratch_20260904/teacher_cache_olmo --stop-after-step 128 --output $OUT/train
$PY - <<EOF
import json
c=json.load(open("$OUT/train/complete.json"))
assert c["completed_step"]==128 and c["status"]=="TRAINING_COMPLETE_NOT_FEASIBILITY_OR_CAPABILITY" and c["final_adapter_reload_greedy_exact"], c
print("train receipt verified: step128, reload exact")
EOF
[ $? -ne 0 ] && { echo "TRAIN_RECEIPT_INVALID"; exit 1; }
for step in 0 32 128; do
  adp=(); [ $step -ne 0 ] && adp=(--adapter $OUT/train/step_$(printf %03d $step))
  stage native$step 1200 $PY $ENGINE native-evaluate $COMMON --max-seconds 1200 ${adp[@]+"${adp[@]}"} --split validation --output $OUT/native$step
done
stage task128 2400 $PY $ENGINE evaluate $COMMON --max-seconds 2400 --split validation --lengths 2048 16384 --adapter $OUT/train/step_128 --output $OUT/task128
stage review 600 $PY $R11/code_round11/scripts/analysis/review_native_constrained_transfer.py --protocol single_evidence_v4 --checkpoint $OLMO --tasks $R11/olmo_tasks/manifest.json --adapter $OUT/train/step_128 --native-baseline $OUT/native0 --native-candidate $OUT/native128 --task-baseline $R11/out/task0 --task-candidate $OUT/task128 --baseline-engine-source $ENGINE --candidate-engine-source $ENGINE --output $OUT/review.json
echo "OLMO_ZF_ROUND_COMPLETE"
