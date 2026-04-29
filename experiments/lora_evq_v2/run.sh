#!/bin/bash
# =============================================================
# EVQ-Cosh LoRA v2 - 一键实验脚本
# =============================================================
# 用法:
#   bash run.sh              # 完整流程: 验证 → 训练 → 评测base → 评测EVQ → 对比
#   bash run.sh dryrun       # 只跑无卡验证
#   bash run.sh train        # 只训练
#   bash run.sh eval         # 只评测 (需要已训练完成)
#   bash run.sh compare      # 只对比结果
#   bash run.sh probes        # 跑 PE Probing Suite (自定义测评)
#   bash run.sh ruler         # 跑 RULER 6-task (标准PE测评)
# =============================================================

set -e

# Optional: point at a conda install without hardcoding a machine path.
CONDA_BIN_DIR="${CONDA_BIN_DIR:-${CONDA_PREFIX:-/root/miniconda3}/bin}"
export PATH="${CONDA_BIN_DIR}:$PATH"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${EVQ_LORA_BASE_DIR:-${SCRIPT_DIR}/local}"
CKPT_DIR="${EVQ_LORA_CKPT:-${SCRIPT_DIR}/checkpoints/evq_r64_tau1414}"
RESULT_DIR="${EVQ_LORA_RESULT_DIR:-${SCRIPT_DIR}/results}"

# ---- 本地路径 (ModelScope下载) ----
MODEL="${EVQ_LORA_MODEL:-${BASE_DIR}/models/Meta-Llama-3-8B-Instruct}"
LOCAL_DATA="${EVQ_LORA_TRAIN_DATA:-${BASE_DIR}/data/longalign_10k/longalign_10k.jsonl}"
WIKITEXT_PATH="${EVQ_LORA_WIKITEXT:-${BASE_DIR}/data/wikitext2/wikitext2_test.txt}"

# ---- 参数 (可按需修改) ----
TAU="${EVQ_LORA_TAU:-1.414}"
LORA_R="${EVQ_LORA_R:-64}"
LORA_ALPHA="${EVQ_LORA_ALPHA:-128}"
MAX_STEPS="${EVQ_LORA_MAX_STEPS:-300}"
MAX_SEQ_LEN="${EVQ_LORA_MAX_SEQ_LEN:-8192}"
# ---------------------------

# Pre-flight check
if [ ! -f "${MODEL}/config.json" ]; then
    echo "❌ Model not found at ${MODEL}"
    echo "   Run: EVQ_LORA_BASE_DIR=${BASE_DIR} python ${SCRIPT_DIR}/download_model_data.py"
    exit 1
fi
if [ ! -f "${LOCAL_DATA}" ]; then
    echo "❌ Training data not found at ${LOCAL_DATA}"
    echo "   Run: EVQ_LORA_BASE_DIR=${BASE_DIR} python ${SCRIPT_DIR}/download_model_data.py"
    exit 1
fi

mkdir -p "${RESULT_DIR}"

run_dryrun() {
    echo "============================================"
    echo "STEP 1: 无卡验证 (dry-run)"
    echo "============================================"
    python "${SCRIPT_DIR}/dryrun_validate.py" \
        --tau ${TAU} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --max_seq_len ${MAX_SEQ_LEN} \
        --local_data_path "${LOCAL_DATA}"
    echo ""
}

run_train() {
    echo "============================================"
    echo "STEP 2: 训练 EVQ-Cosh LoRA"
    echo "============================================"
    python "${SCRIPT_DIR}/train_evq_lora.py" \
        --model_name "${MODEL}" \
        --output_dir "${CKPT_DIR}" \
        --tau ${TAU} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --max_steps ${MAX_STEPS} \
        --max_seq_len ${MAX_SEQ_LEN} \
        --local_data_path "${LOCAL_DATA}" \
        --per_device_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-4
    echo ""
}

run_eval_base() {
    echo "============================================"
    echo "STEP 3a: 评测 Base Instruct (对照组)"
    echo "============================================"
    python "${SCRIPT_DIR}/eval_evq_lora.py" \
        --model_name "${MODEL}" \
        --base_only \
        --output_dir "${RESULT_DIR}" \
        --ppl_lengths "8192,16384,32768" \
        --passkey_lengths "8192,16384,32768" \
        --longbench_max_samples 50 \
        --ppl_data_path "${WIKITEXT_PATH}"
    echo ""
}

run_eval_evq() {
    echo "============================================"
    echo "STEP 3b: 评测 EVQ-Cosh LoRA"
    echo "============================================"
    python "${SCRIPT_DIR}/eval_evq_lora.py" \
        --model_name "${MODEL}" \
        --adapter_dir "${CKPT_DIR}" \
        --output_dir "${RESULT_DIR}" \
        --ppl_lengths "8192,16384,32768" \
        --passkey_lengths "8192,16384,32768" \
        --longbench_max_samples 50 \
        --ppl_data_path "${WIKITEXT_PATH}"
    echo ""
}

run_probes_base() {
    echo "============================================"
    echo "STEP P1: PE Probes - Base Instruct"
    echo "============================================"
    python "${SCRIPT_DIR}/eval_pe_probes.py" \
        --model_name "${MODEL}" \
        --base_only \
        --output_dir "${RESULT_DIR}" \
        --context_lengths "4096,8192,16384,32768"
    echo ""
}

run_probes_evq() {
    echo "============================================"
    echo "STEP P2: PE Probes - EVQ-Cosh LoRA"
    echo "============================================"
    python "${SCRIPT_DIR}/eval_pe_probes.py" \
        --model_name "${MODEL}" \
        --adapter_dir "${CKPT_DIR}" \
        --output_dir "${RESULT_DIR}" \
        --context_lengths "4096,8192,16384,32768"
    echo ""
}

run_ruler_base() {
    echo "============================================"
    echo "STEP R1: RULER 6-task - Base Instruct"
    echo "============================================"
    python "${SCRIPT_DIR}/eval_ruler.py" \
        --model_name "${MODEL}" \
        --base_only \
        --output_dir "${RESULT_DIR}" \
        --context_lengths "4096,8192,16384,32768" \
        --n_trials 20
    echo ""
}

run_ruler_evq() {
    echo "============================================"
    echo "STEP R2: RULER 6-task - EVQ-Cosh LoRA"
    echo "============================================"
    python "${SCRIPT_DIR}/eval_ruler.py" \
        --model_name "${MODEL}" \
        --adapter_dir "${CKPT_DIR}" \
        --output_dir "${RESULT_DIR}" \
        --context_lengths "4096,8192,16384,32768" \
        --n_trials 20
    echo ""
}

run_compare() {
    echo "============================================"
    echo "STEP 4: 对比结果"
    echo "============================================"
    python "${SCRIPT_DIR}/compare_results.py" \
        --base_result "${RESULT_DIR}/eval_base_instruct.json" \
        --evq_result "${RESULT_DIR}/eval_evq_lora.json" \
        --output "${RESULT_DIR}/comparison.json"
    echo ""
}

# ---- 主逻辑 ----
case "${1:-all}" in
    dryrun)   run_dryrun ;;
    train)    run_train ;;
    eval)     run_eval_base; run_eval_evq ;;
    probes)   run_probes_base; run_probes_evq ;;
    ruler)    run_ruler_base; run_ruler_evq ;;
    compare)  run_compare ;;
    all)
        run_dryrun
        run_train
        run_eval_base
        run_eval_evq
        run_compare
        ;;
    *)
        echo "Usage: bash run.sh [dryrun|train|eval|ruler|probes|compare|all]"
        exit 1
        ;;
esac

echo "============================================"
echo "✅ 完成! 结果在: ${RESULT_DIR}/"
echo "============================================"
