#!/bin/bash
# 350M MLA-32 @ 4096 seq_len, 1B diverse tokens
# 3 taus (0.0, 1.414, 2.0) x 3 seeds (42, 43, 88) = 9 runs
# batch_size=12, torch.compile, ~6h total
set -e

SCRIPT=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
DATA_DIR=/root/autodl-tmp/data/1b_diverse_4k
WORK=/root/autodl-tmp/350m_mla32_1b_4k
LOG=${WORK}/run.log

# Verify data exists
if [ ! -f "${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt" ]; then
    echo "ERROR: Training data not found at ${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt"
    echo "Run prepare_1b_4k_data.py first."
    exit 1
fi

mkdir -p $WORK

# Symlink data into work_dir (training script expects data in work_dir)
ln -sf ${DATA_DIR}/train_fineweb-edu_1000000000_4096.pt ${WORK}/train_fineweb-edu_1000000000_4096.pt
ln -sf ${DATA_DIR}/val_fineweb-edu_5000000.pt ${WORK}/val_fineweb-edu_5000000.pt

echo ">>> 350M MLA-32 @ 4K, 1B diverse tokens" | tee $LOG
echo ">>> Taus: 0.0, 1.414 x Seeds: 42, 43, 88 (tau* formula: 1.428, use 1.414)" | tee -a $LOG
echo ">>> batch_size=12, compile, eval_16k" | tee -a $LOG
echo ">>> Started: $(date)" | tee -a $LOG

/root/miniconda3/bin/python -u $SCRIPT \
    --tier 350m \
    --taus 0.0,1.414 \
    --seeds 42,43,88 \
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
