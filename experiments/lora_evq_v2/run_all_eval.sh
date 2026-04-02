#!/bin/bash
# 训练完成后一键跑全部评测：base vs EVQ-LoRA
# 用法: nohup bash run_all_eval.sh > logs/eval.log 2>&1 &
set -e
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="${SCRIPT_DIR}/checkpoints/evq_r64_tau1414"
RESULT_DIR="${SCRIPT_DIR}/results"
MODEL="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"

mkdir -p "${RESULT_DIR}"

# 检查 adapter 是否存在
if [ ! -f "${CKPT_DIR}/adapter_model.safetensors" ] && [ ! -f "${CKPT_DIR}/adapter_model.bin" ]; then
    echo "❌ No adapter found in ${CKPT_DIR}"
    ls -la "${CKPT_DIR}/"
    exit 1
fi

# Quick mode: 只跑 8K/16K/32K (跳4K，base本来就能做)，trials减到5
# RULER: 6 tasks × 3 lengths × 5 trials = 90 次/model → 共180次
# Probes: 跑 MDP(5depth) + KVA 两个最关键的 → ~45次/model → 共90次
# 总计 ~270次推理，预计 ~1小时

echo "============================================"
echo "EVAL 1/4: RULER Base Instruct (quick)"
echo "============================================"
python "${SCRIPT_DIR}/eval_ruler.py" \
    --model_name "${MODEL}" \
    --base_only \
    --output_dir "${RESULT_DIR}" \
    --context_lengths "8192,16384,32768" \
    --n_trials 5

echo "============================================"
echo "EVAL 2/4: RULER EVQ-LoRA (quick)"
echo "============================================"
python "${SCRIPT_DIR}/eval_ruler.py" \
    --model_name "${MODEL}" \
    --adapter_dir "${CKPT_DIR}" \
    --output_dir "${RESULT_DIR}" \
    --context_lengths "8192,16384,32768" \
    --n_trials 5

echo "============================================"
echo "EVAL 3/4: PE Probes Base (quick: MDP+KVA)"
echo "============================================"
python "${SCRIPT_DIR}/eval_pe_probes.py" \
    --model_name "${MODEL}" \
    --base_only \
    --output_dir "${RESULT_DIR}" \
    --context_lengths "8192,16384,32768" \
    --tasks "mdp,kva" \
    --quick

echo "============================================"
echo "EVAL 4/4: PE Probes EVQ-LoRA (quick: MDP+KVA)"
echo "============================================"
python "${SCRIPT_DIR}/eval_pe_probes.py" \
    --model_name "${MODEL}" \
    --adapter_dir "${CKPT_DIR}" \
    --output_dir "${RESULT_DIR}" \
    --context_lengths "8192,16384,32768" \
    --tasks "mdp,kva" \
    --quick

echo "============================================"
echo "✅ All evaluations done!"
echo "Results in: ${RESULT_DIR}/"
ls -la "${RESULT_DIR}/"
echo "============================================"
