#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aidemo
export PYTHONPATH=/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/experiments/variational_sparse_attn:$PYTHONPATH
cd /Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope/experiments/neurips_strict/phase1_prior_softmax
python experiment.py > experiment.log 2>&1
