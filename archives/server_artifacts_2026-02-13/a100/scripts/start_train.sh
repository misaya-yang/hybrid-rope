#!/bin/bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312
cd /opt/dfrope
python -u run_350m_validation.py 2>&1 | tee results/350m_validation/train.log
echo 'Training complete!'
