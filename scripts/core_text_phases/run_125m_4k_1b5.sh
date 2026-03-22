#!/bin/bash
# Phase 21: 125M MLA-32 @ L=4096, 1.5B tokens — inflection point test
#
# Purpose: Prove (or disprove) that GEO's advantage over EVQ grows/shrinks with training.
# At 1B tokens (350M): GEO > EVQ baseline. At 1.5B: does EVQ recover?
# 125M model = ~7x faster than 350M → ETA ~1.5h total (both runs)
#
# Checkpoints auto-saved at 50% (750M) and 75% (1.125B)
# Compare GEO vs EVQ at: 750M, 1.125B, 1.5B
set -e

SCRIPT=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
DATA_DIR=/root/autodl-tmp/data/2b_diverse_4k
WORK=/root/autodl-tmp/125m_mla32_4k_1b5
LOG=${WORK}/run.log

# Verify combined dataset exists
if [ ! -f "${DATA_DIR}/train_fineweb-edu_2000000000_4096.pt" ]; then
    echo "Combined 2B dataset not found. Running prepare_2b_4k_combined.py first..."
    /root/miniconda3/bin/python /root/autodl-tmp/scripts/core_text_phases/prepare_2b_4k_combined.py
fi

mkdir -p $WORK

ln -sf ${DATA_DIR}/train_fineweb-edu_2000000000_4096.pt ${WORK}/train_fineweb-edu_2000000000_4096.pt
ln -sf ${DATA_DIR}/val_fineweb-edu_5000000.pt ${WORK}/val_fineweb-edu_5000000.pt

echo ">>> Phase 21: 125M MLA-32 @ 4K, 1.5B tokens — inflection point test" | tee $LOG
echo ">>> Model: 125M MLA (d_rope=32, K=16 freqs, kv_rank=192)" | tee -a $LOG
echo ">>> Data: v1+v2 combined 2B, using first 1.5B tokens" | tee -a $LOG
echo ">>> Taus: 0.0 (GEO), 1.414 (EVQ) | Seed: 42" | tee -a $LOG
echo ">>> Checkpoints: 50% (750M), 75% (1.125B), 100% (1.5B)" | tee -a $LOG
echo ">>> Expected: ETA ~45min/run, ~1.5h total" | tee -a $LOG
echo ">>> Started: $(date)" | tee -a $LOG

/root/miniconda3/bin/python -u $SCRIPT \
    --tier 125m \
    --taus 0.0,1.414 \
    --seeds 42 \
    --attn_type mla \
    --d_rope 32 \
    --batch_size 32 \
    --compile \
    --dataset fineweb-edu \
    --seq_len 4096 \
    --work_dir $WORK \
    --train_tokens 1500000000 \
    --eval_16k \
    --resume 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo ">>> COMPLETE: $(date)" | tee -a $LOG
echo ">>> Key: compare GEO vs EVQ PPL at 750M/1.125B/1.5B checkpoints" | tee -a $LOG
