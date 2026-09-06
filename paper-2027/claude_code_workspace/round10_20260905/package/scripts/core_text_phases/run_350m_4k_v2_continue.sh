#!/bin/bash
# Phase 20b: 350M MLA-32 @ 4096, continued pretraining 1B -> 1.5B
# Continues from Phase 20 (run_350m_4k_v2_1b.sh) checkpoints
#   GEO:  350m_mla32_4k_v2/350m_tau0.00_seed42/model.pt  -> 1.5B
#   EVQ:  350m_mla32_4k_v2/350m_tau1.41_seed42/model.pt  -> 1.5B
# Extra data: 500M tokens from 1b_diverse_4k (v1 data, unseen by v2-trained models)
# LR: 1e-5 (cosine) -> 1e-6, warmup=200 steps
# No intermediate 50%/75% checkpoints needed
set -e

SCRIPT=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
PATCH=/root/autodl-tmp/scripts/core_text_phases/patch_continue_pretrain.py

# Source model checkpoints from Phase 20
V2_WORK=/root/autodl-tmp/350m_mla32_4k_v2
GEO_CKPT=${V2_WORK}/350m_tau0.00_seed42/model.pt
EVQ_CKPT=${V2_WORK}/350m_tau1.41_seed42/model.pt

# Continuation data: v1 data (500M tokens, unseen by v2-trained models)
DATA_DIR=/root/autodl-tmp/data/1b_diverse_4k
WORK=/root/autodl-tmp/350m_mla32_4k_v2_continue
LOG=${WORK}/run.log

# Verify prerequistes
if [ ! -f "${GEO_CKPT}" ]; then
    echo "ERROR: GEO checkpoint not found: ${GEO_CKPT}"
    echo "Run run_350m_4k_v2_1b.sh first"
    exit 1
fi
if [ ! -f "${EVQ_CKPT}" ]; then
    echo "ERROR: EVQ checkpoint not found: ${EVQ_CKPT}"
    echo "Run run_350m_4k_v2_1b.sh first"
    exit 1
fi

# Apply patch (idempotent - safe to run multiple times)
echo "Applying continue-pretrain patch..."
/root/miniconda3/bin/python $PATCH

mkdir -p $WORK

# Use v1 data for 500M continuation tokens (v1 is unseen by v2-trained models)
ln -sf ${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt ${WORK}/train_fineweb-edu_1000000000_4096.pt
ln -sf ${DATA_DIR}/val_fineweb-edu_5000000.pt ${WORK}/val_fineweb-edu_5000000.pt

echo ">>> Phase 20b: 350M MLA-32 @ 4K, continued pretraining 1B -> 1.5B" | tee $LOG
echo ">>> Continuing from: ${V2_WORK}" | tee -a $LOG
echo ">>> Extra data: 1b_diverse_4k (v1, 500M tokens, unseen by v2-trained models)" | tee -a $LOG
echo ">>> LR: 1e-5 -> 1e-6, no intermediate checkpoints" | tee -a $LOG
echo ">>> Started: $(date)" | tee -a $LOG

# --- GEO continuation ---
echo "" | tee -a $LOG
echo "=== GEO (tau=0.0) continuation ===" | tee -a $LOG
/root/miniconda3/bin/python -u $SCRIPT \
    --tier 350m \
    --taus 0.0 \
    --seeds 42 \
    --attn_type mla \
    --d_rope 32 \
    --batch_size 12 \
    --compile \
    --dataset fineweb-edu \
    --seq_len 4096 \
    --work_dir $WORK \
    --train_tokens 500000000 \
    --eval_16k \
    --init_from ${GEO_CKPT} \
    --continue_lr 1e-5 \
    --no_intermediate_ckpts 2>&1 | tee -a $LOG

# --- EVQ continuation ---
echo "" | tee -a $LOG
echo "=== EVQ (tau=1.414) continuation ===" | tee -a $LOG
/root/miniconda3/bin/python -u $SCRIPT \
    --tier 350m \
    --taus 1.414 \
    --seeds 42 \
    --attn_type mla \
    --d_rope 32 \
    --batch_size 12 \
    --compile \
    --dataset fineweb-edu \
    --seq_len 4096 \
    --work_dir $WORK \
    --train_tokens 500000000 \
    --eval_16k \
    --init_from ${EVQ_CKPT} \
    --continue_lr 1e-5 \
    --no_intermediate_ckpts 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo ">>> COMPLETE: $(date)" | tee -a $LOG
echo ">>> Results: ${WORK}/results_checkpoint.json" | tee -a $LOG
echo ">>> Compare: 1B (v2) vs 1.5B (v2+v1-500M) for GEO and EVQ" | tee -a $LOG
