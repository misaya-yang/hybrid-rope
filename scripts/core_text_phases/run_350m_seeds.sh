#!/bin/bash
# 350M MLA-32 EVQ experiment: 3 seeds (42 done, 43 + 88 new)
# Resume-safe: skips already-completed runs
# Expected: ~4.5h per seed x 2 seeds = ~9h total

set -e

SCRIPT=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
WORK=/root/autodl-tmp/350m_mla32_500m
LOG=${WORK}/run_seeds.log

echo ">>> 350M MLA-32 multi-seed experiment" | tee $LOG
echo ">>> Seeds: 42 (done), 43, 88" | tee -a $LOG
echo ">>> Started: $(date)" | tee -a $LOG

for SEED in 43 88; do
    echo "" | tee -a $LOG
    echo "========== SEED ${SEED} ==========" | tee -a $LOG
    /root/miniconda3/bin/python -u $SCRIPT \
        --tier 350m \
        --taus 0.0,1.414 \
        --seeds ${SEED} \
        --attn_type mla \
        --d_rope 32 \
        --batch_size 5 \
        --compile \
        --dataset fineweb-edu \
        --seq_len 8192 \
        --work_dir $WORK \
        --train_tokens 500000000 \
        --eval_16k 2>&1 | tee -a $LOG
done

echo "" | tee -a $LOG
echo ">>> Training done: $(date)" | tee -a $LOG

# Extended eval across all 3 seeds
echo ">>> Running extended eval..." | tee -a $LOG
/root/miniconda3/bin/python -u /root/autodl-tmp/eval_extended_3seeds.py 2>&1 | tee -a $LOG

echo ">>> COMPLETE: $(date)" | tee -a $LOG
