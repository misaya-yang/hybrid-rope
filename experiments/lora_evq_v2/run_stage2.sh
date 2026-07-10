#!/bin/bash
set -e
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
STAGE1="${SCRIPT_DIR}/checkpoints/evq_r64_tau1414"
STAGE2="${SCRIPT_DIR}/checkpoints/evq_r64_stage2"
RESULT="${SCRIPT_DIR}/results"
RETRIEVAL_DATA="${SCRIPT_DIR}/retrieval_mix.jsonl"
ORIGINAL_DATA="/root/autodl-tmp/data/longalign_10k/longalign_10k.jsonl"

# Pre-flight
for f in "${STAGE1}/adapter_model.safetensors" "${RETRIEVAL_DATA}" "${ORIGINAL_DATA}" "${MODEL}/config.json"; do
    if [ ! -f "$f" ]; then echo "MISSING: $f"; exit 1; fi
done
echo "Pre-flight OK"

# Stage 2 training (~5 min)
echo "============================================"
echo "STAGE 2: Retrieval Adaptation (50 steps)"
echo "============================================"
python "${SCRIPT_DIR}/train_stage2_retrieval.py" \
    --adapter_dir "${STAGE1}" \
    --model_name "${MODEL}" \
    --output_dir "${STAGE2}" \
    --retrieval_data "${RETRIEVAL_DATA}" \
    --original_data "${ORIGINAL_DATA}" \
    --max_steps 50 \
    --learning_rate 2e-5

# RULER eval on stage2 (quick, 5 trials)
echo "============================================"
echo "RULER: Stage2 EVQ-LoRA (quick)"
echo "============================================"
python "${SCRIPT_DIR}/eval_ruler.py" \
    --model_name "${MODEL}" \
    --adapter_dir "${STAGE2}" \
    --output_dir "${RESULT}" \
    --context_lengths "4096,8192,16384" \
    --n_trials 5

echo "============================================"
echo "Done! Results in ${RESULT}/"
echo "============================================"
