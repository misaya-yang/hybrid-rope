#!/usr/bin/env bash
set -euo pipefail

# Isolated queue launcher:
# - never interrupts sacred running Qwen jobs
# - starts new LLaMA-3-8B run immediately after sacred job exits

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/dfrope/hybrid-rope}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-aidemo}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct}"
SEEDS="${SEEDS:-42,1337}"
MAX_STEPS="${MAX_STEPS:-400}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LOG_FILE="${LOG_FILE:-$REPO_DIR/artifacts/next_attn_lora_queue.log}"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

resolve_env() {
  if "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    return
  fi
  if "$CONDA_BIN" env list | awk '{print $1}' | grep -Fxq "base"; then
    log "Conda env '$ENV_NAME' not found, fallback to 'base'."
    ENV_NAME="base"
    return
  fi
  log "ERROR: no usable conda env found (wanted '$ENV_NAME')."
  exit 1
}

wait_sacred_done() {
  log "Waiting sacred Qwen eval job to finish..."
  while pgrep -af "scripts/eval_longbench.py.*Qwen|queue_qwen400_from_baseline.sh" >/dev/null 2>&1; do
    sleep 20
  done
  log "Sacred Qwen job ended. Starting next run."
}

run_train_eval() {
  local mode="$1"
  local seed="$2"
  local run_name="llama3_8b_${mode}_seed${seed}"
  local train_log="$REPO_DIR/artifacts/${run_name}_train.log"
  local eval_log="$REPO_DIR/artifacts/${run_name}_eval.log"

  log "Training start: mode=${mode} seed=${seed}"
  "$CONDA_BIN" run -n "$ENV_NAME" python "$REPO_DIR/train.py" \
    --base_model "$BASE_MODEL" \
    --run_name "$run_name" \
    --output_dir "$REPO_DIR/artifacts/attn_integrated_lora_runs" \
    --seed "$seed" \
    --max_steps "$MAX_STEPS" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --attention_mode "$mode" \
    --lambda_weight 1e-3 \
    --attn_implementation auto \
    --disable_wandb \
    >>"$train_log" 2>&1
  log "Training done: mode=${mode} seed=${seed}"

  local adapter_path="$REPO_DIR/artifacts/attn_integrated_lora_runs/${run_name}/adapter"
  local inv_path="$REPO_DIR/artifacts/attn_integrated_lora_runs/${run_name}/artifacts/custom_inv_freq.pt"
  local eval_root="$REPO_DIR/artifacts/attn_integrated_eval/${run_name}"

  log "Eval start: mode=${mode} seed=${seed}"
  "$CONDA_BIN" run -n "$ENV_NAME" python "$REPO_DIR/scripts/isolated/attn/new_eval_longbench_attnbias.py" \
    --base_model_path "$BASE_MODEL" \
    --adapter_path "$adapter_path" \
    --custom_inv_freq_path "$inv_path" \
    --output_root "$eval_root" \
    --seed "$seed" \
    --attn_bias_mode off \
    >>"$eval_log" 2>&1
  log "Eval done: mode=${mode} seed=${seed}"
}

main() {
  cd "$REPO_DIR"
  resolve_env
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  wait_sacred_done

  IFS=',' read -r -a seed_arr <<<"$SEEDS"
  for s in "${seed_arr[@]}"; do
    run_train_eval static "$s"
    run_train_eval dynamic_penalty "$s"
  done
  log "All queued runs finished."
}

main "$@"
