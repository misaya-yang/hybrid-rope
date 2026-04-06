#!/bin/bash
# LoRA Stage1: EVQ tau=1.414, seed=44, ~1.5h
set -e
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LORA="/root/autodl-tmp/lora_evq_v2"
MODEL="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
DATA="/root/autodl-tmp/data/longalign_10k/longalign_10k.jsonl"
DIR="${LORA}/checkpoints/evq_s44"

if [ -f "${DIR}/adapter_model.safetensors" ]; then
    echo "[SKIP] ${DIR} already exists"; exit 0
fi

echo ">>> EVQ-LoRA seed=44 tau=1.414 | $(date)"
/root/miniconda3/bin/python -u "${LORA}/train_evq_lora.py" \
    --model_name "${MODEL}" \
    --output_dir "${DIR}" \
    --tau 1.414 \
    --local_data_path "${DATA}" \
    --seed 44 \
    --max_steps 300 \
    --lora_r 64 \
    --lora_alpha 128
echo ">>> DONE | $(date)"
