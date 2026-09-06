#!/bin/bash
set -euo pipefail
# 350M MLA-32 实验：500M tokens, seq_len=8192
# 业界最相似架构：DeepSeek-V2 style MLA with decoupled RoPE
P=/root/miniconda3/bin/python
S=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
W=/root/autodl-tmp/350m_mla32_500m

echo ">>> 350M MLA-32 GEO+EVQ, 500M tokens, seq_len=8192, compile"
$P -u $S \
    --tier 350m \
    --taus 0.0,1.414 \
    --seeds 42 \
    --attn_type mla \
    --d_rope 32 \
    --batch_size 6 \
    --compile \
    --dataset fineweb-edu \
    --seq_len 8192 \
    --work_dir $W \
    --train_tokens 500000000 \
    --eval_16k \
    2>&1

echo "350M MLA-32 DONE"
