#!/bin/bash
set -euo pipefail
P=/root/miniconda3/bin/python
S=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
W=/root/autodl-tmp/gqa_125m_experiment
BS=35

echo ">>> Phase 2: GQA-4 BS=$BS compile"
$P -u $S --tier 125m --taus 0.0,1.414 --seeds 42 --n_kv_heads 4 --batch_size $BS --compile --dataset fineweb-edu --work_dir $W/gqa_kv4 --train_tokens 100000000 2>&1

echo ">>> Phase 3: GQA-2 BS=$BS compile"
$P -u $S --tier 125m --taus 0.0,1.414 --seeds 42 --n_kv_heads 2 --batch_size $BS --compile --dataset fineweb-edu --work_dir $W/gqa_kv2 --train_tokens 100000000 2>&1

echo ">>> Phase 4: MLA-32 BS=$BS compile"
$P -u $S --tier 125m --taus 0.0,1.414 --seeds 42 --attn_type mla --d_rope 32 --batch_size $BS --compile --dataset fineweb-edu --work_dir $W/mla_r32 --train_tokens 100000000 2>&1

echo ">>> Phase 5: MLA-16 BS=$BS compile"
$P -u $S --tier 125m --taus 0.0,1.414 --seeds 42 --attn_type mla --d_rope 16 --batch_size $BS --compile --dataset fineweb-edu --work_dir $W/mla_r16 --train_tokens 100000000 2>&1

echo "ALL PHASES DONE"
