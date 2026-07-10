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
BENCHMARK_DIR="${WORK_DIR}/benchmarks"
WIKITEXT="${EVQ_LORA_WIKITEXT:-${BASE_DIR}/data/wikitext2/wikitext2_test.txt}"
LOCAL_TEXT_JSONL="${EVQ_POSITIONAL_TEXT_JSONL:-}"
LORA_DIR="${ROOT}/experiments/lora_evq_v2"
DATASET_REVISION="${EVQ_POSITIONAL_DATASET_REVISION:-87f09149ef4734204d70ed1d046ddc9ca3f2b8f9}"
MICRO_BATCH="${EVQ_POSITIONAL_BATCH_SIZE:-2}"
GRAD_ACCUM="${EVQ_POSITIONAL_GRAD_ACCUM:-4}"
COMPILE_MODE="${EVQ_POSITIONAL_COMPILE_MODE:-default}"
ENABLE_COMPILE="${EVQ_POSITIONAL_COMPILE:-1}"
ENABLE_CHECKPOINTING="${EVQ_POSITIONAL_GRADIENT_CHECKPOINTING:-1}"
GPU_MONITOR_PID=""
GPU_MONITOR_FILE=""
CUDA_SELECTOR="${CUDA_VISIBLE_DEVICES:-}"

require_single_gpu_selector() {
    if [[ -z "${CUDA_SELECTOR}" || "${CUDA_SELECTOR}" == *,* ]]; then
        echo "set CUDA_VISIBLE_DEVICES to exactly one GPU index or UUID" >&2
        exit 2
    fi
}

if (( MICRO_BATCH * GRAD_ACCUM != 8 )); then
    echo "batch_size * gradient_accumulation must equal 8" >&2
    exit 2
fi

mkdir -p \
    "${DATA_DIR}" \
    "${CHECKPOINT_DIR}" \
    "${RESULT_DIR}" \
    "${LOG_DIR}" \
    "${BENCHMARK_DIR}"

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
        --dataset_revision "${DATASET_REVISION}" \
        --split train \
        --seq_len 8192 \
        --train_sequences 2400 \
        --validation_sequences 128 \
        --seed 42 \
        "${source_args[@]}" \
        2>&1 | tee "${LOG_DIR}/prepare_data.log"
}

record_hardware() {
    local invocation_id="$1"
    {
        date -u +"UTC %Y-%m-%dT%H:%M:%SZ"
        git -C "${ROOT}" rev-parse HEAD
        git -C "${ROOT}" status --porcelain
        git -C "${ROOT}" diff --binary HEAD | shasum -a 256
        "${PYTHON}" --version
        "${PYTHON}" - <<'PY'
import importlib.metadata
import torch

print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")
for package in ("transformers", "peft", "accelerate", "datasets"):
    print(f"{package}={importlib.metadata.version(package)}")
if torch.cuda.is_available():
    print(f"capability={torch.cuda.get_device_capability(0)}")
    print(f"arch_list={torch.cuda.get_arch_list()}")
PY
        nvidia-smi -i "${CUDA_SELECTOR}" \
            --query-gpu=name,uuid,memory.total,power.limit,driver_version \
            --format=csv,noheader
    } > "${LOG_DIR}/hardware_${invocation_id}.txt"
}

start_gpu_monitor() {
    local label="$1"
    GPU_MONITOR_FILE="${LOG_DIR}/gpu_${label}.csv"
    nvidia-smi -i "${CUDA_SELECTOR}" \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu \
        --format=csv -l 5 > "${GPU_MONITOR_FILE}" &
    GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
    if [[ -z "${GPU_MONITOR_PID}" && -z "${GPU_MONITOR_FILE}" ]]; then
        return 0
    fi
    local exited_early=0
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        if kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
            kill "${GPU_MONITOR_PID}" 2>/dev/null || true
            wait "${GPU_MONITOR_PID}" 2>/dev/null || true
        else
            wait "${GPU_MONITOR_PID}" 2>/dev/null || true
            exited_early=1
        fi
        GPU_MONITOR_PID=""
    fi
    if [[ -z "${GPU_MONITOR_FILE}" || ! -s "${GPU_MONITOR_FILE}" ]]; then
        echo "GPU telemetry is missing" >&2
        return 1
    fi
    local line_count
    line_count="$(wc -l < "${GPU_MONITOR_FILE}")"
    if (( exited_early != 0 || line_count < 2 )); then
        echo "GPU telemetry did not retain a header and sample" >&2
        return 1
    fi
    GPU_MONITOR_FILE=""
}

trap 'stop_gpu_monitor || true' EXIT INT TERM

train_one() {
    local label="$1"
    local method="$2"
    local max_steps="$3"
    local warmup_steps="$4"
    local output_dir="${CHECKPOINT_DIR}/${label}"
    local invocation_id="${label}_$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
    if [[ -f "${output_dir}/adapter_model.safetensors" ]]; then
        "${PYTHON}" "${LORA_DIR}/validate_checkpoint_artifact.py" \
            --checkpoint "${output_dir}" \
            --expected-method "${method}" \
            --expected-objective positional_hidden_distillation \
            --expected-data-manifest "${DATA_DIR}/manifest.json" \
            --expected-model "${MODEL}" \
            --claim-log-dir "${LOG_DIR}" \
            --require-claim-ready
        echo "validated existing checkpoint: ${label}"
        {
            date -u +"UTC %Y-%m-%dT%H:%M:%SZ"
            git -C "${ROOT}" rev-parse HEAD
            echo "reused_checkpoint=${output_dir}"
        } > "${LOG_DIR}/reuse_${invocation_id}.txt"
        return
    fi

    local performance_args=()
    if [[ "${ENABLE_COMPILE}" == "1" ]]; then
        performance_args+=(--compile --compile_mode "${COMPILE_MODE}")
    else
        performance_args+=(--no-compile)
    fi
    if [[ "${ENABLE_CHECKPOINTING}" == "1" ]]; then
        performance_args+=(--gradient_checkpointing)
    else
        performance_args+=(--no-gradient_checkpointing)
    fi

    record_hardware "${invocation_id}"
    start_gpu_monitor "${invocation_id}"
    local train_log="${LOG_DIR}/train_${invocation_id}.log"
    set +e
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
        --max_steps "${max_steps}" \
        --per_device_batch_size "${MICRO_BATCH}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate 2e-5 \
        --warmup_steps "${warmup_steps}" \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --seed 42 \
        "${performance_args[@]}" \
        2>&1 | tee "${train_log}"
    local status=${PIPESTATUS[0]}
    set -e
    local monitor_status=0
    stop_gpu_monitor || monitor_status=$?
    local ledger_status=0
    if (( monitor_status == 0 )); then
        "${PYTHON}" "${LORA_DIR}/validate_checkpoint_artifact.py" \
            --checkpoint "${output_dir}" \
            --expected-method "${method}" \
            --claim-log-dir "${LOG_DIR}" \
            --append-invocation-ledger \
            --process-status "${status}" \
            --telemetry "${LOG_DIR}/gpu_${invocation_id}.csv" \
            --hardware-record "${LOG_DIR}/hardware_${invocation_id}.txt" \
            --train-log "${train_log}" || ledger_status=$?
    fi
    if (( status != 0 )); then
        return "${status}"
    fi
    if (( monitor_status != 0 )); then
        return "${monitor_status}"
    fi
    if (( ledger_status != 0 )); then
        return "${ledger_status}"
    fi
    "${PYTHON}" "${LORA_DIR}/validate_checkpoint_artifact.py" \
        --checkpoint "${output_dir}" \
        --expected-method "${method}" \
        --expected-objective positional_hidden_distillation \
        --expected-data-manifest "${DATA_DIR}/manifest.json" \
        --expected-model "${MODEL}" \
        --claim-log-dir "${LOG_DIR}" \
        --finalize-claim-ready \
        --telemetry "${LOG_DIR}/gpu_${invocation_id}.csv" \
        --hardware-record "${LOG_DIR}/hardware_${invocation_id}.txt" \
        --train-log "${train_log}"
}

train_models() {
    require_single_gpu_selector
    require_file "${DATA_DIR}/manifest.json"
    train_one evq_distill_s42 evq_cosh 300 30
    train_one geo_distill_s42 native_geo 1 0
}

run_performance_probe() {
    local suite_dir="$1"
    local label="$2"
    local batch_size="$3"
    local grad_accum="$4"
    local compile_enabled="$5"
    local checkpointing_enabled="$6"
    local output_dir="${suite_dir}/${label}"
    local invocation_id="probe_${label}_$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
    local compile_args=(--no-compile)
    if [[ "${compile_enabled}" == "1" ]]; then
        compile_args=(--compile --compile_mode "${COMPILE_MODE}")
    fi
    local checkpointing_args=(--no-gradient_checkpointing)
    if [[ "${checkpointing_enabled}" == "1" ]]; then
        checkpointing_args=(--gradient_checkpointing)
    fi

    mkdir -p "${output_dir}"
    record_hardware "${invocation_id}"
    start_gpu_monitor "${invocation_id}"
    local train_log="${LOG_DIR}/train_${invocation_id}.log"
    set +e
    "${PYTHON}" -u "${LORA_DIR}/train_positional_distill.py" \
        --model_name "${MODEL}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${output_dir}" \
        --student_method evq_cosh \
        --tau 1.414 \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.0 \
        --lora_targets q_proj,k_proj \
        --max_steps 300 \
        --performance_probe_steps 12 \
        --per_device_batch_size "${batch_size}" \
        --gradient_accumulation_steps "${grad_accum}" \
        --learning_rate 2e-5 \
        --warmup_steps 30 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        "${checkpointing_args[@]}" \
        --seed 42 \
        "${compile_args[@]}" \
        2>&1 | tee "${train_log}"
    local status=${PIPESTATUS[0]}
    set -e
    local monitor_status=0
    stop_gpu_monitor || monitor_status=$?
    if (( status != 0 || monitor_status != 0 )); then
        if [[ -f "${output_dir}/performance_probe.json" ]]; then
            mv \
                "${output_dir}/performance_probe.json" \
                "${output_dir}/performance_probe.invalid.json"
        fi
        {
            echo "process_status=${status}"
            echo "monitor_status=${monitor_status}"
        } > "${output_dir}/probe_failed.txt"
        echo "probe failed and was excluded: ${label}" >&2
    fi
    return 0
}

benchmark_models() {
    require_single_gpu_selector
    require_file "${DATA_DIR}/manifest.json"
    local suite_id="suite_$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
    local suite_dir="${BENCHMARK_DIR}/${suite_id}"
    mkdir -p "${suite_dir}"

    run_performance_probe "${suite_dir}" eager_b2_ga4_gc 2 4 0 1
    run_performance_probe "${suite_dir}" compile_b2_ga4_gc 2 4 1 1
    run_performance_probe "${suite_dir}" compile_b4_ga2_gc 4 2 1 1
    run_performance_probe "${suite_dir}" compile_b8_ga1_gc 8 1 1 1
    run_performance_probe "${suite_dir}" compile_b2_ga4_no_gc 2 4 1 0

    "${PYTHON}" - "${suite_dir}" <<'PY'
import json
import sys
from pathlib import Path

suite = Path(sys.argv[1])
rows = []
for path in sorted(suite.glob("*/performance_probe.json")):
    row = json.loads(path.read_text(encoding="utf-8"))
    total_memory_gb = row["runtime"]["device_total_memory_gb"]
    reserved_gb = row["peak_cuda_reserved_gb"]
    required_headroom_gb = max(4.0, total_memory_gb * 0.05)
    rows.append(
        {
            "candidate": path.parent.name,
            "steady_state_nominal_tokens_per_second": row[
                "steady_state_nominal_tokens_per_second"
            ],
            "peak_cuda_allocated_gb": row["peak_cuda_allocated_gb"],
            "peak_cuda_reserved_gb": row["peak_cuda_reserved_gb"],
            "device_total_memory_gb": total_memory_gb,
            "headroom_gb": total_memory_gb - reserved_gb,
            "required_headroom_gb": required_headroom_gb,
            "headroom_eligible": total_memory_gb - reserved_gb >= required_headroom_gb,
            "performance": row["performance"],
            "run_protocol_sha256": row["run_protocol_sha256"],
        }
    )
if not rows:
    raise SystemExit("all performance probes failed")
rows.sort(key=lambda row: row["steady_state_nominal_tokens_per_second"], reverse=True)
eligible = [row for row in rows if row["headroom_eligible"]]
if not eligible:
    raise SystemExit("no successful probe retained the required VRAM headroom")
summary = {
    "format_version": 1,
    "scope": "non-claim target-GPU performance probes",
    "fastest_eligible_candidate": eligible[0]["candidate"],
    "candidates": rows,
}
output = suite / "benchmark_summary.json"
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

eval_one() {
    local variant="$1"
    local method="$2"
    local adapter_dir="${3:-}"
    local adapter_args=()
    if [[ -n "${adapter_dir}" ]]; then
        require_file "${adapter_dir}/adapter_model.safetensors"
        adapter_args+=(
            --adapter_dir "${adapter_dir}"
            --claim_log_dir "${LOG_DIR}"
        )
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
        --lm_head_chunk_tokens 2048 \
        --hidden_sequences 128 \
        --hidden_batch_size 4 \
        --seed 42 \
        "${adapter_args[@]}" \
        2>&1 | tee "${LOG_DIR}/eval_${variant}.log"
}

evaluate_models() {
    require_single_gpu_selector
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
    benchmark)
        benchmark_models
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
        echo "usage: $0 {prepare|benchmark|train|eval|all}" >&2
        exit 2
        ;;
esac
