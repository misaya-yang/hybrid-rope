#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train_cross_model_lora.py}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/wikitext_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/cross_model}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/root/autodl-tmp/dfrope/ms_models}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/_logs}"

SMART_BOOTSTRAP="${SMART_BOOTSTRAP:-1}"      # 1: run already-local models first.
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"    # 1: force local loading for training.
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"            # Keep fixed across all six runs.
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"

# Fixed shared training hyper-parameters (must be identical across all six tasks).
MAX_STEPS=600
LEARNING_RATE=2e-5
WARMUP_STEPS=50
PER_DEVICE_BATCH=1
GRAD_ACCUM=8
MAX_SEQ_LEN=16384
LORA_RANK=16
LORA_ALPHA=32
LORA_TARGETS="q_proj,k_proj,v_proj,o_proj"
LR_SCHEDULER="cosine"
BF16=1

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

log() {
  echo "[$(date '+%F %T %Z')] $*"
}

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[fatal] missing train script: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${DATA_DIR}/train.txt" ]]; then
  echo "[fatal] training data missing: ${DATA_DIR}/train.txt" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[fatal] no python interpreter found" >&2
    exit 1
  fi
fi

declare -A MODEL_PATHS=()
declare -A DOWNLOAD_PIDS=()
declare -A DOWNLOAD_PATH_FILES=()
declare -A DOWNLOAD_LOG_FILES=()

LLAMA_CANDIDATES=(
  "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
  "/root/autodl-tmp/dfrope/ms_models/Meta-Llama-3-8B-Instruct"
)
MISTRAL_CANDIDATES=(
  "/root/autodl-tmp/dfrope/ms_models/MistralAI/Mistral-7B-Instruct-v0.3"
  "/root/autodl-tmp/dfrope/ms_models/mistralai/Mistral-7B-Instruct-v0.3"
  "/root/autodl-tmp/dfrope/ms_models/Mistral-7B-Instruct-v0.3"
)
QWEN_CANDIDATES=(
  "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2-7B-Instruct"
  "/root/autodl-tmp/dfrope/ms_models/qwen/Qwen2-7B-Instruct"
)

find_existing_model_path() {
  local key="$1"
  local p
  case "${key}" in
    llama)
      for p in "${LLAMA_CANDIDATES[@]}"; do
        [[ -f "${p}/config.json" ]] && echo "${p}" && return 0
      done
      ;;
    mistral)
      for p in "${MISTRAL_CANDIDATES[@]}"; do
        [[ -f "${p}/config.json" ]] && echo "${p}" && return 0
      done
      ;;
    qwen)
      for p in "${QWEN_CANDIDATES[@]}"; do
        [[ -f "${p}/config.json" ]] && echo "${p}" && return 0
      done
      ;;
    *)
      echo "[fatal] unknown model key: ${key}" >&2
      return 2
      ;;
  esac
  return 1
}

start_modelscope_download_bg() {
  local key="$1"
  local path_file="${LOG_ROOT}/download_${key}.path"
  local log_file="${LOG_ROOT}/download_${key}.log"
  local -a repos=()
  case "${key}" in
    llama)
      repos=("LLM-Research/Meta-Llama-3-8B-Instruct")
      ;;
    mistral)
      repos=("MistralAI/Mistral-7B-Instruct-v0.3" "mistralai/Mistral-7B-Instruct-v0.3")
      ;;
    qwen)
      repos=("Qwen/Qwen2-7B-Instruct" "qwen/Qwen2-7B-Instruct")
      ;;
    *)
      echo "[fatal] unknown model key for download: ${key}" >&2
      return 2
      ;;
  esac

  DOWNLOAD_PATH_FILES["${key}"]="${path_file}"
  DOWNLOAD_LOG_FILES["${key}"]="${log_file}"

  (
    set -euo pipefail
    "${PYTHON_BIN}" - "${MODEL_CACHE_DIR}" "${path_file}" "${repos[@]}" <<'PY'
import sys
from pathlib import Path

cache_dir = sys.argv[1]
path_file = Path(sys.argv[2])
repo_candidates = sys.argv[3:]

try:
    from modelscope import snapshot_download
except Exception as exc:
    print(f"[download] modelscope import failed: {type(exc).__name__}: {exc}", flush=True)
    raise

errors = []
for repo in repo_candidates:
    try:
        path = snapshot_download(
            repo,
            cache_dir=cache_dir,
            ignore_patterns=["original/*", "*.pth"],
        )
        path_file.write_text(str(path), encoding="utf-8")
        print(f"[download] ok repo={repo} path={path}", flush=True)
        raise SystemExit(0)
    except Exception as exc:
        errors.append(f"{repo}: {type(exc).__name__}: {exc}")

print("[download] all candidates failed", flush=True)
for line in errors:
    print(f"  - {line}", flush=True)
raise SystemExit(1)
PY
  ) >"${log_file}" 2>&1 &

  DOWNLOAD_PIDS["${key}"]="$!"
  log "background download started: key=${key} pid=${DOWNLOAD_PIDS[$key]} log=${log_file}"
}

ensure_model_ready() {
  local key="$1"
  local current="${MODEL_PATHS[$key]:-}"
  local local_found=""
  if [[ -n "${current}" && -f "${current}/config.json" ]]; then
    echo "${current}"
    return 0
  fi

  if [[ -n "${DOWNLOAD_PIDS[$key]:-}" ]]; then
    local pid="${DOWNLOAD_PIDS[$key]}"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      log "waiting for download: key=${key} pid=${pid}"
      wait "${pid}"
    else
      wait "${pid}" || true
    fi
  fi

  if [[ -n "${DOWNLOAD_PATH_FILES[$key]:-}" && -f "${DOWNLOAD_PATH_FILES[$key]}" ]]; then
    local downloaded
    downloaded="$(tr -d '[:space:]' < "${DOWNLOAD_PATH_FILES[$key]}")"
    if [[ -n "${downloaded}" && -f "${downloaded}/config.json" ]]; then
      MODEL_PATHS["${key}"]="${downloaded}"
      echo "${downloaded}"
      return 0
    fi
  fi

  if local_found="$(find_existing_model_path "${key}")"; then
    MODEL_PATHS["${key}"]="${local_found}"
    echo "${local_found}"
    return 0
  fi

  echo "[fatal] model unavailable after download attempts: ${key}" >&2
  if [[ -n "${DOWNLOAD_LOG_FILES[$key]:-}" ]]; then
    echo "[fatal] download log: ${DOWNLOAD_LOG_FILES[$key]}" >&2
  fi
  return 1
}

cleanup_bg_downloads() {
  local pid
  for pid in "${DOWNLOAD_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup_bg_downloads INT TERM EXIT

ALL_TASKS=(
  "mistral|mistral_7b_instruct_v0_3|baseline|42"
  "mistral|mistral_7b_instruct_v0_3|anchored_sigmoid|42"
  "qwen|qwen2_7b_instruct|baseline|42"
  "qwen|qwen2_7b_instruct|anchored_sigmoid|42"
  "llama|llama_3_8b_instruct|baseline|1337"
  "llama|llama_3_8b_instruct|anchored_sigmoid|1337"
)

append_tasks_for_key() {
  local key="$1"
  local row
  for row in "${ALL_TASKS[@]}"; do
    IFS='|' read -r t_key _ _ _ <<< "${row}"
    if [[ "${t_key}" == "${key}" ]]; then
      ORDERED_TASKS+=("${row}")
    fi
  done
}

EXISTING_KEYS=()
MISSING_KEYS=()
for key in llama mistral qwen; do
  if local_path="$(find_existing_model_path "${key}")"; then
    MODEL_PATHS["${key}"]="${local_path}"
    EXISTING_KEYS+=("${key}")
    log "local model found: key=${key} path=${local_path}"
  else
    MISSING_KEYS+=("${key}")
    start_modelscope_download_bg "${key}"
  fi
done

ORDERED_TASKS=()
if [[ "${SMART_BOOTSTRAP}" == "1" ]]; then
  log "SMART_BOOTSTRAP=1: running already-local models first."
  for key in "${EXISTING_KEYS[@]}"; do
    append_tasks_for_key "${key}"
  done
  for key in "${MISSING_KEYS[@]}"; do
    append_tasks_for_key "${key}"
  done
else
  ORDERED_TASKS=("${ALL_TASKS[@]}")
fi

log "task order:"
for row in "${ORDERED_TASKS[@]}"; do
  log "  ${row}"
done

run_task() {
  local model_key="$1"
  local model_tag="$2"
  local method="$3"
  local seed="$4"

  local model_path
  model_path="$(ensure_model_ready "${model_key}")"

  local run_dir="${OUTPUT_ROOT}/${model_tag}_${method}_${seed}"
  local run_log="${LOG_ROOT}/${model_tag}_${method}_${seed}.log"
  mkdir -p "${run_dir}"

  log "start training: model=${model_tag} method=${method} seed=${seed}"
  log "  model_path=${model_path}"
  log "  run_dir=${run_dir}"

  local -a cmd=(
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
    --method "${method}"
    --base_model_path "${model_path}"
    --output_dir "${run_dir}"
    --run_name "${model_tag}_${method}_${seed}"
    --data_dir "${DATA_DIR}"
    --seed "${seed}"
    --max_seq_len "${MAX_SEQ_LEN}"
    --max_steps "${MAX_STEPS}"
    --per_device_train_batch_size "${PER_DEVICE_BATCH}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --learning_rate "${LEARNING_RATE}"
    --warmup_steps "${WARMUP_STEPS}"
    --logging_steps 10
    --save_steps 200
    --lr_scheduler_type "${LR_SCHEDULER}"
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_target_modules "${LORA_TARGETS}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
    --model_cache_dir "${MODEL_CACHE_DIR}"
    --optim paged_adamw_8bit
  )

  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    cmd+=(--trust_remote_code)
  else
    cmd+=(--no-trust_remote_code)
  fi
  if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
    cmd+=(--local_files_only)
  else
    cmd+=(--no-local_files_only)
  fi
  if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
    cmd+=(--load_in_4bit)
  else
    cmd+=(--no-load_in_4bit)
  fi
  if [[ "${BF16}" == "1" ]]; then
    cmd+=(--bf16)
  else
    cmd+=(--no-bf16)
  fi

  "${cmd[@]}" 2>&1 | tee "${run_log}"
  log "done training: model=${model_tag} method=${method} seed=${seed}"
}

for row in "${ORDERED_TASKS[@]}"; do
  IFS='|' read -r model_key model_tag method seed <<< "${row}"
  run_task "${model_key}" "${model_tag}" "${method}" "${seed}"
done

log "all six cross-model runs completed."
