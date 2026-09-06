#!/bin/bash
# E1 cross cells (Pro Unified Plan §6.2): the two off-diagonal W×T cells,
# strictly reusing round-11 validation inputs/scoring (lengths 2048 16384,
# split validation). Engine release008 invoked read-only; outputs land in
# round12 dirs only. Diagonal receipts (out_zf/task128, out_on/task128,
# out/task128=T0) are the reference manifests for identity checks.
set -u
PY=/root/miniconda3/bin/python
B=/root/autodl-tmp/ffn_review_execution_20260904
R11=/root/autodl-tmp/claude_round11_olmo_20260905
OLMO=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
OUT=/root/autodl-tmp/claude_round12_20260906/e1/cross
ENGINE=$B/code_release_008/scripts/train/train_single_table_native_constrained.py
COMMON="--checkpoint $OLMO --checkpoint-contract $B/olmo1485_contract.json --tasks $R11/olmo_tasks/manifest.json --native-pool $B/native_pool_v3/manifest.json --seed 42 --authorized"
TABLE=$B/fixed_controls/Z.npy
GAIN=1.102585782722872
mkdir -p $OUT/logs
stage() {
  local name=$1 cap=$2; shift 2
  echo "E1_CROSS: $name"
  timeout $((cap+120)) "$@" > $OUT/logs/$name.log 2>&1
  local code=$?
  if [ $code -ne 0 ]; then echo "STAGE_FAILED $name exit=$code"; exit $code; fi
}
# W_ON x T_Z: adapter trained under native table, evaluated under Z table+gain.
# Completed cells (evaluation.json present) are skipped, not re-run.
if [ -f $OUT/w_on_t_z/evaluation.json ]; then
  echo "E1_CROSS: w_on_t_z already complete, skipped"
else
  stage w_on_t_z 2400 $PY $ENGINE evaluate $COMMON --table $TABLE --gain $GAIN \
    --max-seconds 2400 --split validation --lengths 2048 16384 \
    --adapter $R11/out_on/train/step_128 --output $OUT/w_on_t_z
fi
# W_ZF x T_0: adapter trained under Z table, evaluated under native table.
if [ -f $OUT/w_zf_t_0/evaluation.json ]; then
  echo "E1_CROSS: w_zf_t_0 already complete, skipped"
else
  stage w_zf_t_0 2400 $PY $ENGINE evaluate $COMMON \
    --max-seconds 2400 --split validation --lengths 2048 16384 \
    --adapter $R11/out_zf/train/step_128 --output $OUT/w_zf_t_0
fi
echo "E1_CROSS_CELLS_COMPLETE"
