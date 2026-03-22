#!/bin/bash
# Phase 21a: 125M MLA-32 @ L=4096, 1B tokens on v1 data
# Checkpoints at 50% (500M), 75% (750M), 100% (1B)
# After this completes, run run_125m_4k_continue_v2.sh for +500M -> 1.5B total
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
DATA_DIR=/root/autodl-tmp/data/1b_diverse_4k
WORK=/root/autodl-tmp/125m_mla32_4k_1b
LOG=${WORK}/run.log

mkdir -p $WORK
ln -sf ${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt ${WORK}/train_fineweb-edu_1000000000_4096.pt
ln -sf ${DATA_DIR}/val_fineweb-edu_5000000.pt ${WORK}/val_fineweb-edu_5000000.pt

echo ">>> Phase 21a: 125M MLA-32 @ 4K, 1B tokens (v1 data)" | tee $LOG
echo ">>> Checkpoints: 50%(500M), 75%(750M), 100%(1B)" | tee -a $LOG
echo ">>> Started: $(date)" | tee -a $LOG

/root/miniconda3/bin/python -u $SCRIPT \
    --tier 125m \
    --taus 0.0,1.414 \
    --seeds 42 \
    --attn_type mla \
    --d_rope 32 \
    --batch_size 24 \
    --compile \
    --dataset fineweb-edu \
    --seq_len 4096 \
    --work_dir $WORK \
    --train_tokens 1000000000 \
    --eval_16k \
    --passkey_mix_ratio 0.0 \
    --resume 2>&1 | tee -a $LOG

echo ">>> COMPLETE: $(date)" | tee -a $LOG
echo ">>> Next: run run_125m_4k_continue_v2.sh for +500M continuation" | tee -a $LOG
