#!/usr/bin/env bash
set -euo pipefail

# New queue for: attention correction + new dataset + rank32
# Flow:
# 1) Build local long/short instruction dataset artifact.
# 2) Run short smoke (dynamic_penalty) to validate no OOM.
# 3) Launch full run (dynamic_penalty, rank32).
#
# Safety note:
# - This is an experimental dynamic-penalty queue.
# - Default production path is next_joint_attn_opt_queue.sh.
# - To avoid accidental high-cost runs, this script requires an explicit opt-in.

if [[ "${ALLOW_EXPERIMENTAL_DYNAMIC:-0}" != "1" ]]; then
  echo "[BLOCKED] Experimental queue is disabled by default."
  echo "Set ALLOW_EXPERIMENTAL_DYNAMIC=1 to run intentionally."
  exit 2
fi

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/dfrope/hybrid-rope}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-base}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct}"
SEED="${SEED:-42}"
MAX_STEPS_FULL="${MAX_STEPS_FULL:-800}"
MAX_STEPS_SMOKE="${MAX_STEPS_SMOKE:-20}"
LOG_FILE="${LOG_FILE:-$REPO_DIR/artifacts/next_attn_rank32_newdata_queue.log}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/artifacts/datasets/newdata_rank32_v1}"

LONG_JSONL="${LONG_JSONL:-$DATASET_ROOT/long_instruct.jsonl}"
SHORT_JSONL="${SHORT_JSONL:-$DATASET_ROOT/short_instruct.jsonl}"
WIKITEXT_TRAIN="${WIKITEXT_TRAIN:-/root/autodl-tmp/wikitext_data/train.txt}"

RUN_SMOKE="${RUN_SMOKE:-llama3_dynamic_rank32_newdata_smoke_seed${SEED}}"
RUN_FULL="${RUN_FULL:-llama3_dynamic_rank32_newdata_seed${SEED}}"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$DATASET_ROOT"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

run_dataset_build() {
  if [[ ! -f "$WIKITEXT_TRAIN" ]]; then
    if [[ -f "/root/autodl-tmp/dfrope/datasets/train.txt" ]]; then
      WIKITEXT_TRAIN="/root/autodl-tmp/dfrope/datasets/train.txt"
    elif [[ -f "/root/autodl-tmp/wikitext_data/train.txt" ]]; then
      WIKITEXT_TRAIN="/root/autodl-tmp/wikitext_data/train.txt"
    fi
  fi
  if [[ ! -f "$WIKITEXT_TRAIN" ]]; then
    log "ERROR: cannot find WikiText train file."
    return 2
  fi
  log "Building new dataset: $LONG_JSONL + $SHORT_JSONL"
  "$CONDA_BIN" run -n "$ENV_NAME" python "$REPO_DIR/scripts/isolated/attn/build_newdata_long_short.py" \
    --wikitext_train "$WIKITEXT_TRAIN" \
    --out_long "$LONG_JSONL" \
    --out_short "$SHORT_JSONL" \
    --n_long 6200 \
    --n_short 2800 \
    --seed "$SEED" \
    | tee -a "$LOG_FILE"
}

run_train() {
  local run_name="$1"
  local max_steps="$2"
  local train_log="$REPO_DIR/artifacts/${run_name}_train.log"
  log "Start train: run=$run_name steps=$max_steps mode=dynamic_penalty r=32"
  "$CONDA_BIN" run -n "$ENV_NAME" python "$REPO_DIR/train.py" \
    --base_model "$BASE_MODEL" \
    --run_name "$run_name" \
    --output_dir "$REPO_DIR/artifacts/attn_integrated_lora_runs" \
    --seed "$SEED" \
    --attention_mode dynamic_penalty \
    --attn_implementation eager \
    --lambda_weight 1e-3 \
    --max_steps "$max_steps" \
    --lora_r 32 \
    --lora_alpha 64 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 8192 \
    --learning_rate 2e-5 \
    --warmup_steps 30 \
    --long_data_path "$LONG_JSONL" \
    --short_data_path "$SHORT_JSONL" \
    --long_ratio 0.7 \
    --max_total_samples 9000 \
    --min_long_samples 1500 \
    --min_short_samples 700 \
    --disable_wandb \
    >>"$train_log" 2>&1
  log "Train done: run=$run_name"
}

run_eval() {
  local run_name="$1"
  local eval_log="$REPO_DIR/artifacts/${run_name}_eval.log"
  local adapter_path="$REPO_DIR/artifacts/attn_integrated_lora_runs/${run_name}/adapter"
  local inv_path="$REPO_DIR/artifacts/attn_integrated_lora_runs/${run_name}/artifacts/custom_inv_freq.pt"
  local eval_root="$REPO_DIR/artifacts/attn_integrated_eval/${run_name}"
  log "Start eval: run=$run_name"
  "$CONDA_BIN" run -n "$ENV_NAME" python "$REPO_DIR/scripts/isolated/attn/new_eval_longbench_attnbias.py" \
    --base_model_path "$BASE_MODEL" \
    --adapter_path "$adapter_path" \
    --custom_inv_freq_path "$inv_path" \
    --output_root "$eval_root" \
    --seed "$SEED" \
    --attn_bias_mode off \
    >>"$eval_log" 2>&1
  log "Eval done: run=$run_name"
}

main() {
  cd "$REPO_DIR"
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false

  run_dataset_build
  run_train "$RUN_SMOKE" "$MAX_STEPS_SMOKE"
  run_eval "$RUN_SMOKE"

  run_train "$RUN_FULL" "$MAX_STEPS_FULL"
  run_eval "$RUN_FULL"

  log "Queue complete: dynamic_penalty + rank32 + new dataset."
}

main "$@"
