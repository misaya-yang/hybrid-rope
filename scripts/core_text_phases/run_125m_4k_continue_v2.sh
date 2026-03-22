#!/bin/bash
# Phase 21b: 125M MLA-32 @ L=4096, continue 1B -> 1.5B on v2 data
# Requires: Phase 21a complete (run_125m_4k_1b_v1.sh)
# Loads 1B checkpoints and continues for 500M more tokens
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
PATCH=/root/autodl-tmp/scripts/core_text_phases/patch_continue_pretrain.py
DATA_DIR=/root/autodl-tmp/data/1b_diverse_4k_v2
PREV_WORK=/root/autodl-tmp/125m_mla32_4k_1b
WORK=/root/autodl-tmp/125m_mla32_4k_1b5_continue
LOG=${WORK}/run.log

GEO_CKPT=${PREV_WORK}/125m_tau0.00_seed42/model.pt
EVQ_CKPT=${PREV_WORK}/125m_tau1.41_seed42/model.pt

if [ ! -f "${GEO_CKPT}" ]; then
    echo "ERROR: GEO checkpoint not found: ${GEO_CKPT}"
    echo "Run run_125m_4k_1b_v1.sh first"
    exit 1
fi
if [ ! -f "${EVQ_CKPT}" ]; then
    echo "ERROR: EVQ checkpoint not found: ${EVQ_CKPT}"
    echo "Run run_125m_4k_1b_v1.sh first"
    exit 1
fi

# Apply continue-pretrain patch (idempotent)
/root/miniconda3/bin/python $PATCH

mkdir -p $WORK
ln -sf ${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt ${WORK}/train_fineweb-edu_1000000000_4096.pt
ln -sf ${DATA_DIR}/val_fineweb-edu_5000000.pt ${WORK}/val_fineweb-edu_5000000.pt

echo ">>> Phase 21b: 125M MLA-32 @ 4K, continue 1B->1.5B (v2 data)" | tee $LOG
echo ">>> GEO ckpt: ${GEO_CKPT}" | tee -a $LOG
echo ">>> EVQ ckpt: ${EVQ_CKPT}" | tee -a $LOG
echo ">>> Started: $(date)" | tee -a $LOG

# GEO continuation
echo "" | tee -a $LOG
echo "=== GEO (tau=0.0) continuation ===" | tee -a $LOG
/root/miniconda3/bin/python -u $SCRIPT \
    --tier 125m \
    --taus 0.0 \
    --seeds 42 \
    --attn_type mla \
    --d_rope 32 \
    --batch_size 24 \
    --compile \
    --dataset fineweb-edu \
    --seq_len 4096 \
    --work_dir $WORK \
    --train_tokens 500000000 \
    --eval_16k \
    --passkey_mix_ratio 0.0 \
    --init_from ${GEO_CKPT} \
    --continue_lr 1e-5 \
    --no_intermediate_ckpts 2>&1 | tee -a $LOG

# EVQ continuation
echo "" | tee -a $LOG
echo "=== EVQ (tau=1.414) continuation ===" | tee -a $LOG
/root/miniconda3/bin/python -u $SCRIPT \
    --tier 125m \
    --taus 1.414 \
    --seeds 42 \
    --attn_type mla \
    --d_rope 32 \
    --batch_size 24 \
    --compile \
    --dataset fineweb-edu \
    --seq_len 4096 \
    --work_dir $WORK \
    --train_tokens 500000000 \
    --eval_16k \
    --passkey_mix_ratio 0.0 \
    --init_from ${EVQ_CKPT} \
    --continue_lr 1e-5 \
    --no_intermediate_ckpts 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo ">>> COMPLETE: $(date)" | tee -a $LOG
echo ">>> Results: ${WORK}/results_checkpoint.json" | tee -a $LOG
echo ">>> Compare GEO vs EVQ at 500M / 750M / 1B / 1.5B" | tee -a $LOG
