#!/bin/bash
# Phase 20: 350M MLA-32 @ 4096 seq_len, 1B tokens on v2 data (non-overlapping)
# Purpose: Independent replication of Phase 18b with fresh data
#   - Compare GEO vs EVQ tau=1.414 at 500M (50% ckpt) and 1B
#   - Answers: "Is GEO > EVQ at 4K a consistent result, or seed/data artifact?"
# Data: 1b_diverse_4k_v2 (Pile+OWT, seed=2024, buf=100000, non-overlapping with v1)
# seed=42 only first (single seed, same as Phase 18b for direct comparison)
# After this run, extend to 1.5B with run_350m_4k_v2_continue.sh
set -e

SCRIPT=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
DATA_DIR=/root/autodl-tmp/data/1b_diverse_4k_v2
WORK=/root/autodl-tmp/350m_mla32_4k_v2
LOG=${WORK}/run.log

if [ ! -f "${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt" ]; then
    echo "ERROR: v2 training data not found: ${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt"
    exit 1
fi

mkdir -p $WORK

ln -sf ${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt ${WORK}/train_fineweb-edu_1000000000_4096.pt
ln -sf ${DATA_DIR}/val_fineweb-edu_5000000.pt ${WORK}/val_fineweb-edu_5000000.pt

echo ">>> Phase 20: 350M MLA-32 @ 4K, 1B v2 tokens (independent replication)" | tee $LOG
echo ">>> Data: 1b_diverse_4k_v2 (seed=2024, buf=100000, non-overlapping with Phase 18b)" | tee -a $LOG
echo ">>> Taus: 0.0, 1.414 x Seed: 42 | batch_size=12, compile, eval_16k" | tee -a $LOG
echo ">>> Key: 50pct checkpoint = GEO@500M vs EVQ@500M (fair comparison)" | tee -a $LOG
echo ">>> Started: $(date)" | tee -a $LOG

/root/miniconda3/bin/python -u $SCRIPT \
    --tier 350m \
    --taus 0.0,1.414 \
    --seeds 42 \
    --attn_type mla \
    --d_rope 32 \
    --batch_size 12 \
    --compile \
    --dataset fineweb-edu \
    --seq_len 4096 \
    --work_dir $WORK \
    --train_tokens 1000000000 \
    --eval_16k \
    --resume 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo ">>> COMPLETE: $(date)" | tee -a $LOG
echo ">>> Next: run run_350m_4k_v2_continue.sh to extend to 1.5B" | tee -a $LOG
