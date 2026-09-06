#!/bin/bash
# Wait for PG19 shard download to finish, then run the 500M CPT data build.
# CPU-only; safe to run alongside the GPU audit.
set -u
B12=/root/autodl-tmp/claude_round12_20260906
LOG=$B12/data/prep500m.log
mkdir -p $B12/data
echo "PREP500_WAIT_START $(date -u +%FT%TZ)" >> $LOG
until grep -q PG19_DL_DONE $B12/dl_pg19_more.log 2>/dev/null; do sleep 60; done
echo "PREP500_DL_SEEN $(date -u +%FT%TZ)" >> $LOG
grep PG19_DL_DONE $B12/dl_pg19_more.log >> $LOG
/root/miniconda3/bin/python $B12/code/data_prep_cpt_v2.py \
  --tokenizer /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct \
  --pg19-dir $B12/datasets/pg19/data \
  --v1-data $B12/data/cpt \
  --out $B12/data/cpt_500m >> $LOG 2>&1
echo "PREP500_EXIT=$? $(date -u +%FT%TZ)" >> $LOG
