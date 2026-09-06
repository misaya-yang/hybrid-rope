#!/bin/bash
# Stage2: 对所有Stage1 checkpoint续训50步检索能力
# 70% 合成检索(S-NIAH/MK-NIAH/KV-Retr) + 30% 原始LongAlign
# lr=2e-5 (比Stage1低5x，防遗忘)
#
# 前置: 01_*.sh 全部完成 + 02_gen_retrieval_data.sh 已跑
# 每个checkpoint ~5min, 总计 ~30min
set -e
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LORA="/root/autodl-tmp/lora_evq_v2"
MODEL="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
RETRIEVAL="${LORA}/retrieval_mix.jsonl"
ORIGINAL="/root/autodl-tmp/data/longalign_10k/longalign_10k.jsonl"

if [ ! -f "${RETRIEVAL}" ]; then
    echo "ERROR: retrieval_mix.jsonl missing. Run 02_lora_gen_retrieval_data.sh first."
    exit 1
fi

# Stage1 checkpoint -> Stage2 output 的映射
declare -A STAGE1_DIRS
STAGE1_DIRS=(
    ["geo_s42"]="${LORA}/checkpoints/geo_s42"
    ["geo_s43"]="${LORA}/checkpoints/geo_s43"
    ["geo_s44"]="${LORA}/checkpoints/geo_s44"
    ["evq_s42"]="${LORA}/checkpoints/evq_r64_tau1414"
    ["evq_s43"]="${LORA}/checkpoints/evq_s43"
    ["evq_s44"]="${LORA}/checkpoints/evq_s44"
)

for NAME in geo_s42 geo_s43 geo_s44 evq_s42 evq_s43 evq_s44; do
    S1_DIR="${STAGE1_DIRS[$NAME]}"
    S2_DIR="${LORA}/checkpoints/${NAME}_stage2"

    # 跳过已完成的
    if [ -f "${S2_DIR}/adapter_model.safetensors" ]; then
        echo "[SKIP] ${NAME}_stage2 exists"
        continue
    fi

    # 检查Stage1是否完成
    if [ ! -f "${S1_DIR}/adapter_model.safetensors" ]; then
        echo "[SKIP] ${NAME} Stage1 not found at ${S1_DIR}"
        continue
    fi

    echo ""
    echo ">>> Stage2: ${NAME} (50 steps, lr=2e-5) | $(date)"
    /root/miniconda3/bin/python -u "${LORA}/train_stage2_retrieval.py" \
        --adapter_dir "${S1_DIR}" \
        --model_name "${MODEL}" \
        --output_dir "${S2_DIR}" \
        --retrieval_data "${RETRIEVAL}" \
        --original_data "${ORIGINAL}" \
        --max_steps 50 \
        --learning_rate 2e-5
    echo ">>> ${NAME}_stage2 DONE | $(date)"
done

echo ""
echo "=== All Stage2 Done ==="
