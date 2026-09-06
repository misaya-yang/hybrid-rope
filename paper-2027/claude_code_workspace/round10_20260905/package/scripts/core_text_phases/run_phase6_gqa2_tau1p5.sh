#!/bin/bash
set -euo pipefail
P=/root/miniconda3/bin/python
S=/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py
W=/root/autodl-tmp/gqa_125m_experiment
BS=35

echo ">>> Phase 6: GQA-2 tau=1.5 (right-shift test) BS=$BS compile"
$P -u $S --tier 125m --taus 0.0,1.5 --seeds 42 --n_kv_heads 2 --batch_size $BS --compile --dataset fineweb-edu --work_dir $W/gqa_kv2_tau1p5 --train_tokens 100000000 --resume 2>&1

echo "PHASE 6 DONE"
