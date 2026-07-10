#!/usr/bin/env bash
# Prepare the clean seed-42 positional-distillation pilot.
# This file is an operator entrypoint; repository validation must not invoke it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${EVQ_LORA_BASE_DIR:-${ROOT}/experiments/lora_evq_v2/local}"
PYTHON="${EVQ_LORA_PYTHON:-python}"
MODEL="${EVQ_LORA_MODEL:-${BASE_DIR}/models/Meta-Llama-3-8B-Instruct}"
WORK_DIR="${EVQ_POSITIONAL_DISTILL_DIR:-${BASE_DIR}/positional_distill_s42}"
DATA_DIR="${WORK_DIR}/data"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints"
RESULT_DIR="${WORK_DIR}/results"
LOG_DIR="${WORK_DIR}/logs"
WIKITEXT="${EVQ_LORA_WIKITEXT:-${BASE_DIR}/data/wikitext2/wikitext2_test.txt}"
LOCAL_TEXT_JSONL="${EVQ_POSITIONAL_TEXT_JSONL:-}"
LORA_DIR="${ROOT}/experiments/lora_evq_v2"

mkdir -p "${DATA_DIR}" "${CHECKPOINT_DIR}" "${RESULT_DIR}" "${LOG_DIR}"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "missing required file: $1" >&2
        exit 1
    fi
}

prepare_data() {
    require_file "${MODEL}/config.json"
    if [[ -f "${DATA_DIR}/manifest.json" ]]; then
        echo "frozen data already exists: ${DATA_DIR}/manifest.json"
        return
    fi

    local source_args=()
    if [[ -n "${LOCAL_TEXT_JSONL}" ]]; then
        require_file "${LOCAL_TEXT_JSONL}"
        source_args+=(--local_jsonl "${LOCAL_TEXT_JSONL}")
    fi

    "${PYTHON}" -u "${LORA_DIR}/prepare_positional_distill_data.py" \
        --tokenizer "${MODEL}" \
        --output_dir "${DATA_DIR}" \
        --dataset HuggingFaceFW/fineweb-edu \
        --dataset_config sample-10BT \
        --split train \
        --seq_len 8192 \
        --train_sequences 2400 \
        --validation_sequences 128 \
        --seed 42 \
        "${source_args[@]}" \
        2>&1 | tee "${LOG_DIR}/prepare_data.log"
}

record_hardware() {
    {
        date -u +"UTC %Y-%m-%dT%H:%M:%SZ"
        git -C "${ROOT}" rev-parse HEAD
        "${PYTHON}" --version
        nvidia-smi --query-gpu=name,uuid,memory.total,power.limit,driver_version \
            --format=csv,noheader
    } > "${WORK_DIR}/hardware_and_revision.txt"
}

train_one() {
    local label="$1"
    local method="$2"
    local output_dir="${CHECKPOINT_DIR}/${label}"
    if [[ -f "${output_dir}/adapter_model.safetensors" ]]; then
        "${PYTHON}" "${LORA_DIR}/validate_checkpoint_artifact.py" \
            --checkpoint "${output_dir}" \
            --expected-method "${method}"
        echo "validated existing checkpoint: ${label}"
        return
    fi

    "${PYTHON}" -u "${LORA_DIR}/train_positional_distill.py" \
        --model_name "${MODEL}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${output_dir}" \
        --student_method "${method}" \
        --tau 1.414 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.0 \
        --lora_targets q_proj,k_proj \
        --max_steps 300 \
        --per_device_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --warmup_steps 30 \
        --seed 42 \
        2>&1 | tee "${LOG_DIR}/train_${label}.log"
}

train_models() {
    require_file "${DATA_DIR}/manifest.json"
    record_hardware
    train_one geo_distill_s42 native_geo
    train_one evq_distill_s42 evq_cosh
}

eval_one() {
    local variant="$1"
    local method="$2"
    local adapter_dir="${3:-}"
    local adapter_args=()
    if [[ -n "${adapter_dir}" ]]; then
        require_file "${adapter_dir}/adapter_model.safetensors"
        adapter_args+=(--adapter_dir "${adapter_dir}")
    fi

    "${PYTHON}" -u "${LORA_DIR}/eval_positional_distill.py" \
        --model_name "${MODEL}" \
        --variant "${variant}" \
        --candidate_method "${method}" \
        --tau 1.414 \
        --data_dir "${DATA_DIR}" \
        --wikitext_path "${WIKITEXT}" \
        --output_dir "${RESULT_DIR}" \
        --ppl_lengths 8192,16384,32768 \
        --ppl_chunks 5 \
        --hidden_batches 8 \
        "${adapter_args[@]}" \
        2>&1 | tee "${LOG_DIR}/eval_${variant}.log"
}

evaluate_models() {
    require_file "${DATA_DIR}/manifest.json"
    require_file "${WIKITEXT}"
    eval_one base_geo native_geo
    eval_one base_evq evq_cosh
    eval_one geo_distill_s42 native_geo "${CHECKPOINT_DIR}/geo_distill_s42"
    eval_one evq_distill_s42 evq_cosh "${CHECKPOINT_DIR}/evq_distill_s42"
    "${PYTHON}" "${LORA_DIR}/summarize_positional_distill.py" \
        --result_dir "${RESULT_DIR}" \
        2>&1 | tee "${LOG_DIR}/summarize.log"
}

case "${1:-}" in
    prepare)
        prepare_data
        ;;
    train)
        train_models
        ;;
    eval)
        evaluate_models
        ;;
    all)
        prepare_data
        train_models
        evaluate_models
        ;;
    *)
        echo "usage: $0 {prepare|train|eval|all}" >&2
        exit 2
        ;;
esac
