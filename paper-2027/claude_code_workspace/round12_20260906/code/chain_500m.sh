#!/bin/bash
# 500M flagship chain (single GPU): probe -> KL teacher cache -> train A+B.
# Prerequisites: CPT_DATA_FROZEN_V2_500M manifest exists, GPU idle.
# Gates recorded; on FAIL the scene is preserved and training never starts.
set -u
B12=/root/autodl-tmp/claude_round12_20260906
M7=/root/autodl-tmp/models/OLMo-2-1124-7B-Instruct
POOL=/root/autodl-tmp/ffn_review_execution_20260904/native_pool_v3/manifest.json
PY=/root/miniconda3/bin/python
RUN=$B12/runs/Z_CPT_500M
LOG=$RUN/chain.log
mkdir -p $RUN
cd $B12/code || exit 91
echo "CHAIN500_START $(date -u +%FT%TZ)" >> $LOG

# Gate 0: data must be frozen first (CPU pipeline, separate process)
if [ ! -f $B12/data/cpt_500m/manifest.json ]; then
  echo "GATE_FAIL data not frozen" >> $LOG; exit 2
fi

# Gate 1: training VRAM/throughput probe (fail-fast, records boundary)
$PY probe_7b_train.py --model $M7 --tables $B12/tables \
  --out $RUN/probe_train_result.json >> $LOG 2>&1
ST=$($PY -c "import json;print(json.load(open('$RUN/probe_train_result.json'))['status'])")
echo "PROBE=$ST $(date -u +%FT%TZ)" >> $LOG
if [ "$ST" != "PASS" ]; then echo "CHAIN500_ABORT probe gate" >> $LOG; exit 3; fi

# KL teacher cache build (teacher-only on GPU, minutes)
$PY build_kl_cache.py --model $M7 --replay-manifest $POOL \
  --out $B12/data/kl_cache_olmo7b.npz >> $LOG 2>&1 \
  || { echo "CHAIN500_ABORT kl cache" >> $LOG; exit 4; }

# Main run: 7,648 CPT updates (501.2M tokens) + 64 SFT updates
$PY track_b_train_v2.py --arm Z --model $M7 --tables $B12/tables \
  --cpt-data $B12/data/cpt_500m --kl-cache $B12/data/kl_cache_olmo7b.npz \
  --sft-views $B12/data/sft/views.jsonl \
  --replay-manifest $POOL \
  --out-root $RUN --phase all >> $LOG 2>&1
RC=$?
echo "TRAIN_EXIT=$RC $(date -u +%FT%TZ)" >> $LOG
echo "CHAIN500_COMPLETE rc=$RC $(date -u +%FT%TZ)" >> $LOG
