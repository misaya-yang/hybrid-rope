#!/bin/bash
# Phase 24: 125M MLA-v2 (DeepSeek-aligned) @ L=4096, 500M tokens
# Architecture: d_rope=64 (K=32), d_nope=128, v_head=128, kv_rank=512, base=10000
# tau: set from Phase 23 sweep results
# GPU: Blackwell 96GB
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
DATA_DIR=/root/autodl-tmp/data/1b_diverse_4k
WORK=/root/autodl-tmp/125m_mla_v2_500m
LOG=${WORK}/run.log

mkdir -p $WORK
ln -sf ${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt ${WORK}/train_fineweb-edu_1000000000_4096.pt
ln -sf ${DATA_DIR}/val_fineweb-edu_5000000.pt ${WORK}/val_fineweb-edu_5000000.pt

echo ">>> Phase 24: 125M MLA-v2 @ 4K, 500M tokens — GEO vs EVQ" | tee $LOG
echo ">>> d_rope=64 (K=32), d_nope=128, v_head=128, kv_rank=512, base=10000" | tee -a $LOG
echo ">>> tau: 0.0 (GEO), TAU_STAR (from Phase 23)" | tee -a $LOG
echo ">>> Started: $(date)" | tee -a $LOG

# Phase 23 sweep result: τ=2.5 optimal (4K -3.7%, 8K -6.7%, 16K -14.5%)
TAU_STAR=2.5

/root/miniconda3/bin/python -u $SCRIPT \
    --tier 125m \
    --taus 0.0,${TAU_STAR} \
    --seeds 42 \
    --attn_type mla \
    --d_rope 64 \
    --d_nope 128 \
    --v_head_dim 128 \
    --kv_lora_rank 512 \
    --base 10000 \
    --batch_size 8 \
    --compile \
    --dataset fineweb-edu \
    --seq_len 4096 \
    --work_dir $WORK \
    --train_tokens 500000000 \
    --eval_16k \
    --passkey_mix_ratio 0.0 \
    --resume 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo ">>> COMPLETE: $(date)" | tee -a $LOG
