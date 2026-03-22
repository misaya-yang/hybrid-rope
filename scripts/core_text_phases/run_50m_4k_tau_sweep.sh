#!/bin/bash
# Phase 22: 50M MLA-32 @ L=4096, 200M tokens — MLA tau* sweep
# reduce-overhead + 独立进程 + sleep 60s保证干净GPU
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
DATA_DIR=/root/autodl-tmp/data/1b_diverse_4k
WORK=/root/autodl-tmp/50m_mla32_4k_tau_sweep
LOG=${WORK}/run.log

mkdir -p $WORK
ln -sf ${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt ${WORK}/train_fineweb-edu_1000000000_4096.pt
ln -sf ${DATA_DIR}/val_fineweb-edu_5000000.pt ${WORK}/val_fineweb-edu_5000000.pt

echo ">>> Phase 22: 50M MLA-32 @ 4K, 200M tokens — tau sweep" | tee $LOG
echo ">>> bs=32, reduce-overhead, each tau separate process" | tee -a $LOG
echo ">>> Started: $(date)" | tee -a $LOG

for TAU in 0.0 1.8 2.0 2.2 2.5 3.0; do
    echo "" | tee -a $LOG
    echo "=== tau=${TAU} starting at $(date) ===" | tee -a $LOG

    /root/miniconda3/bin/python -u $SCRIPT \
        --tier 50m \
        --taus ${TAU} \
        --seeds 42 \
        --attn_type mla \
        --d_rope 32 \
        --batch_size 32 \
        --compile \
        --dataset fineweb-edu \
        --seq_len 4096 \
        --work_dir $WORK \
        --train_tokens 200000000 \
        --eval_16k \
        --passkey_mix_ratio 0.0 2>&1 | tee -a $LOG

    echo "=== tau=${TAU} exit=$? at $(date), releasing GPU 60s ===" | tee -a $LOG
    sleep 60
done

echo "" | tee -a $LOG
echo ">>> ALL COMPLETE: $(date)" | tee -a $LOG
