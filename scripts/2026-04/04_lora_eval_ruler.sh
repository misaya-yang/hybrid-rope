#!/bin/bash
# RULER评估: base + GEO-LoRA(Stage2) + EVQ-LoRA(Stage2), 每个3 seeds
# 6任务 x 3长度(8K/16K/32K) x 20 trials
# 每个checkpoint ~2h, 总计 ~14h
#
# 用法:
#   bash 04_lora_eval_ruler.sh          # 评测所有
#   bash 04_lora_eval_ruler.sh quick    # 快速模式(5 trials)
set -e
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LORA="/root/autodl-tmp/lora_evq_v2"
MODEL="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
RESULT="${LORA}/results/april_ruler"
mkdir -p "${RESULT}"

LENGTHS="8192,16384,32768"
TRIALS=20
if [ "${1}" = "quick" ]; then
    TRIALS=5
    echo "[QUICK MODE] trials=5"
fi

# 评测列表: label -> adapter_dir (空=base)
declare -A EVAL_TARGETS
EVAL_TARGETS=(
    ["base"]=""
    ["geo_s42_s2"]="${LORA}/checkpoints/geo_s42_stage2"
    ["geo_s43_s2"]="${LORA}/checkpoints/geo_s43_stage2"
    ["geo_s44_s2"]="${LORA}/checkpoints/geo_s44_stage2"
    ["evq_s42_s2"]="${LORA}/checkpoints/evq_s42_stage2"
    ["evq_s43_s2"]="${LORA}/checkpoints/evq_s43_stage2"
    ["evq_s44_s2"]="${LORA}/checkpoints/evq_s44_stage2"
)

# 也评测Stage1 (无检索训练) 作为ablation
EVAL_TARGETS+=(
    ["geo_s42_s1"]="${LORA}/checkpoints/geo_s42"
    ["evq_s42_s1"]="${LORA}/checkpoints/evq_r64_tau1414"
)

for LABEL in base geo_s42_s1 evq_s42_s1 geo_s42_s2 geo_s43_s2 geo_s44_s2 evq_s42_s2 evq_s43_s2 evq_s44_s2; do
    ADAPTER="${EVAL_TARGETS[$LABEL]}"
    OUT="${RESULT}/ruler_${LABEL}.json"

    if [ -f "${OUT}" ]; then
        echo "[SKIP] ${LABEL} already evaluated"
        continue
    fi

    echo ""
    echo "========================================"
    echo " RULER: ${LABEL} | $(date)"
    echo "========================================"

    ADAPTER_ARG=""
    if [ -z "${ADAPTER}" ]; then
        ADAPTER_ARG="--base_only"
    else
        if [ ! -f "${ADAPTER}/adapter_model.safetensors" ]; then
            echo "[SKIP] ${LABEL}: adapter not found"
            continue
        fi
        ADAPTER_ARG="--adapter_dir ${ADAPTER}"
    fi

    /root/miniconda3/bin/python -u "${LORA}/eval_ruler.py" \
        --model_name "${MODEL}" \
        ${ADAPTER_ARG} \
        --output_dir "${RESULT}" \
        --context_lengths "${LENGTHS}" \
        --n_trials ${TRIALS} \
        2>&1 | tee "${RESULT}/log_ruler_${LABEL}.txt"

    echo ">>> ${LABEL} DONE | $(date)"
done

echo ""
echo "=== All RULER Evaluations Done ==="
echo "Results: ${RESULT}/"
ls -la "${RESULT}/"
