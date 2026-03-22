#!/bin/bash
# Phase 23: 50M MLA-v2 (industry-aligned) @ L=4096, 200M tokens — tau sweep
# Architecture: d_rope=64 (K=32), d_nope=128, v_head=128, kv_rank=384, base=10000
# Matches DeepSeek-V2/V3 and Kimi K2 ratios
# RTX 5090 32GB — bs=2, each tau separate process + sleep 10s
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
DATA_DIR=/root/autodl-tmp/data/1b_diverse_4k
WORK=/root/autodl-tmp/50m_mla_v2_tau_sweep
LOG=${WORK}/run.log

mkdir -p $WORK
ln -sf ${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt ${WORK}/train_fineweb-edu_1000000000_4096.pt
ln -sf ${DATA_DIR}/val_fineweb-edu_5000000.pt ${WORK}/val_fineweb-edu_5000000.pt

echo ">>> Phase 23: 50M MLA-v2 (DeepSeek-aligned) @ 4K, 200M tokens" | tee $LOG
echo ">>> d_rope=64 (K=32), d_nope=128, v_head=128, kv_rank=384, base=10000" | tee -a $LOG
echo ">>> Started: $(date)" | tee -a $LOG

COMMON="--tier 50m --seeds 42 --attn_type mla --d_rope 64 --d_nope 128 --v_head_dim 128 --kv_lora_rank 384 --base 10000 --batch_size 2 --compile --dataset fineweb-edu --seq_len 4096 --work_dir $WORK --train_tokens 200000000 --eval_16k --passkey_mix_ratio 0.0"

for TAU in 0.0 0.5 1.0 1.414 2.0 2.5 3.0; do
    echo "" | tee -a $LOG
    echo "=== tau=${TAU} starting at $(date) ===" | tee -a $LOG
    /root/miniconda3/bin/python -u $SCRIPT --taus ${TAU} ${COMMON} >> $LOG 2>&1
    echo "=== tau=${TAU} exit=$? at $(date) ===" | tee -a $LOG
    sleep 10
done

echo "" | tee -a $LOG
echo ">>> ALL COMPLETE: $(date)" | tee -a $LOG
