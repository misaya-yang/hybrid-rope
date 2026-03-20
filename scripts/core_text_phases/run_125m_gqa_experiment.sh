#!/bin/bash
# ============================================================
# 125M Attention Compression × EVQ Experiment
# ============================================================
# Hypothesis: EVQ's advantage grows with attention compression
#   - MHA (12 KV heads, 32 freqs) = baseline
#   - GQA-4 (4 KV heads, 32 freqs) = KV head compression
#   - GQA-2 (2 KV heads, 32 freqs) = extreme KV head compression
#   - MLA-32 (latent KV, d_rope=32, 16 freqs) = latent + decoupled RoPE
#   - MLA-16 (latent KV, d_rope=16, 8 freqs) = extreme compression
#   Each with: τ=0 (GEO) vs τ=1.414 (EVQ, fixed across all configs)
#   Fixed τ isolates the attention mechanism as the only variable.
# Total: 10 runs × ~5-8 min each ≈ 50-80 min on Blackwell
# ============================================================

set -euo pipefail
PYTHON=/root/miniconda3/bin/python

SCRIPT="/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py"
WORK="/root/autodl-tmp/gqa_125m_experiment"
TAUS="0.0,1.414"
SEEDS="42"
TIER="125m"
DATASET="fineweb-edu"
BS=16
COMPILE=""  # 125M is small enough without compile

mkdir -p "$WORK"

echo "============================================================"
echo "  125M Attention Compression × EVQ Experiment"
echo "  Work dir: $WORK"
echo "  τ values: $TAUS"
echo "  Seeds: $SEEDS"
echo "============================================================"

# Verify GPU is free
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d ' %')
if [ "$GPU_UTIL" -gt 50 ]; then
    echo "ERROR: GPU utilization is ${GPU_UTIL}%. Another job is running."
    echo "Wait for it to finish before starting this experiment."
    exit 1
fi

echo ""
echo ">>> Phase 1/5: MHA (n_kv=12, standard)"
/root/miniconda3/bin/python -u "$SCRIPT" \
    --tier "$TIER" --taus "$TAUS" --seeds "$SEEDS" \
    --n_kv_heads 12 --batch_size "$BS" \
    --dataset "$DATASET" \
    --work_dir "$WORK/mha_kv12" \
    --train_tokens 100000000 \
    2>&1 | tee "$WORK/mha_kv12.log"

echo ""
echo ">>> Phase 2/5: GQA-4 (n_kv=4, 3x compression)"
/root/miniconda3/bin/python -u "$SCRIPT" \
    --tier "$TIER" --taus "$TAUS" --seeds "$SEEDS" \
    --n_kv_heads 4 --batch_size "$BS" \
    --dataset "$DATASET" \
    --work_dir "$WORK/gqa_kv4" \
    --train_tokens 100000000 \
    2>&1 | tee "$WORK/gqa_kv4.log"

echo ""
echo ">>> Phase 3/5: GQA-2 (n_kv=2, 6x compression)"
/root/miniconda3/bin/python -u "$SCRIPT" \
    --tier "$TIER" --taus "$TAUS" --seeds "$SEEDS" \
    --n_kv_heads 2 --batch_size "$BS" \
    --dataset "$DATASET" \
    --work_dir "$WORK/gqa_kv2" \
    --train_tokens 100000000 \
    2>&1 | tee "$WORK/gqa_kv2.log"

echo ""
echo ">>> Phase 4/5: MLA-32 (d_rope=32, 16 freqs, τ=1.414)"
mkdir -p "$WORK/mla_r32"
ln -sf /root/autodl-tmp/data/train_750m_clean/train_fineweb-edu_1470000000_2048.pt "$WORK/mla_r32/"
ln -sf /root/autodl-tmp/evq_750m_clean/val_fineweb-edu_5000000.pt "$WORK/mla_r32/"
/root/miniconda3/bin/python -u "$SCRIPT" \
    --tier "$TIER" --taus "$TAUS" --seeds "$SEEDS" \
    --attn_type mla --d_rope 32 --batch_size "$BS" \
    --dataset "$DATASET" \
    --work_dir "$WORK/mla_r32" \
    --train_tokens 100000000 \
    2>&1 | tee "$WORK/mla_r32.log"

echo ""
echo ">>> Phase 5/5: MLA-16 (d_rope=16, 8 freqs, τ=1.414)"
mkdir -p "$WORK/mla_r16"
ln -sf /root/autodl-tmp/data/train_750m_clean/train_fineweb-edu_1470000000_2048.pt "$WORK/mla_r16/"
ln -sf /root/autodl-tmp/evq_750m_clean/val_fineweb-edu_5000000.pt "$WORK/mla_r16/"
/root/miniconda3/bin/python -u "$SCRIPT" \
    --tier "$TIER" --taus "$TAUS" --seeds "$SEEDS" \
    --attn_type mla --d_rope 16 --batch_size "$BS" \
    --dataset "$DATASET" \
    --work_dir "$WORK/mla_r16" \
    --train_tokens 100000000 \
    2>&1 | tee "$WORK/mla_r16.log"

echo ""
echo "============================================================"
echo "  ALL DONE — Comparing results"
echo "============================================================"

# Summary
python -c "
import json, os

configs = [
    ('MHA (kv=12)',      '$WORK/mha_kv12/results_final.json', 32),
    ('GQA-4 (kv=4)',     '$WORK/gqa_kv4/results_final.json',  32),
    ('GQA-2 (kv=2)',     '$WORK/gqa_kv2/results_final.json',  32),
    ('MLA-32 (16 freq)', '$WORK/mla_r32/results_final.json',  16),
    ('MLA-16 (8 freq)',  '$WORK/mla_r16/results_final.json',   8),
]

print()
print(f\"{'Config':<22} {'freqs':>5} {'τ':>5} {'PPL@2K':>8} {'PPL@4K':>8} {'PPL@8K':>8} {'PK_ret':>7}\")
print('-' * 80)

for label, path, nfreq in configs:
    if not os.path.exists(path):
        print(f'{label:<22} MISSING')
        continue
    with open(path) as f:
        data = json.load(f)
    for run_id, exp in data.get('experiments', {}).items():
        tau = exp.get('tau', '?')
        ppl = exp.get('ppl', {})
        pk = exp.get('passkey', {})
        pk_global = pk.get('global', {})
        pk_ret = pk_global.get('retrieval_rate', pk.get('retrieval_rate', '?'))
        p2 = ppl.get('2048', '?')
        p4 = ppl.get('4096', '?')
        p8 = ppl.get('8192', '?')
        tag = 'GEO' if tau == 0 or tau == 0.0 else 'EVQ'
        print(f'{label:<22} {nfreq:>5} {tag:>5} {p2:>8} {p4:>8} {p8:>8} {pk_ret:>7}')

print()
print('Key: EVQ improvement should INCREASE as compression grows:')
print('  MHA(32f) < GQA-4(32f) < GQA-2(32f) < MLA-32(16f) < MLA-16(8f)')
" 2>&1 | tee "$WORK/summary.txt"
