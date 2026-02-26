#!/usr/bin/env bash
set -euo pipefail

# NeurIPS-oriented joint optimization queue:
# - long-task dataset first
# - attention integrated LoRA (bias + macro/micro KL)
# - smoke gate before full run

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/dfrope/hybrid-rope}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-base}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct}"
SEED="${SEED:-42}"
STEPS_SMOKE="${STEPS_SMOKE:-40}"
STEPS_FULL="${STEPS_FULL:-800}"
RANK="${RANK:-32}"
ALPHA="${ALPHA:-64}"
RUN_TAG="${RUN_TAG:-v2_bias}"
LOG_FILE="${LOG_FILE:-$REPO_DIR/artifacts/next_joint_attn_opt_${RUN_TAG}.log}"
ATTN_BIAS_MODE="${ATTN_BIAS_MODE:-bias}"      # bias | bias+gate
USE_MACRO_MICRO_KL="${USE_MACRO_MICRO_KL:-0}" # 0/1
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MAX_SEQ_LEN_SMOKE="${MAX_SEQ_LEN_SMOKE:-$MAX_SEQ_LEN}"
MAX_SEQ_LEN_FULL="${MAX_SEQ_LEN_FULL:-8192}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
ATTN_IMPL_SMOKE="${ATTN_IMPL_SMOKE:-sdpa}"
ATTN_IMPL_FULL="${ATTN_IMPL_FULL:-sdpa}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCH_INPUT_TOKENS="${EVAL_MAX_BATCH_INPUT_TOKENS:-8192}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"              # 1=offline cache only, 0=allow mirror/DashScope

DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/artifacts/datasets/newdata_joint_opt_v1}"
LONG_JSONL="${LONG_JSONL:-$DATASET_ROOT/long_instruct.jsonl}"
SHORT_JSONL="${SHORT_JSONL:-$DATASET_ROOT/short_instruct.jsonl}"
WIKITEXT_TRAIN="${WIKITEXT_TRAIN:-/root/autodl-tmp/wikitext_data/train.txt}"
LONGALPACA_MIRROR_URL="${LONGALPACA_MIRROR_URL:-}"

RUN_SMOKE="llama3_jointopt_${RUN_TAG}_smoke_s${SEED}"
RUN_FULL="llama3_jointopt_${RUN_TAG}_s${SEED}"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$DATASET_ROOT"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

build_dataset() {
  log "Building long-task dataset artifact."
  "$CONDA_BIN" run -n "$ENV_NAME" python "$REPO_DIR/scripts/isolated/attn/build_newdata_long_short.py" \
    --wikitext_train "$WIKITEXT_TRAIN" \
    --out_long "$LONG_JSONL" \
    --out_short "$SHORT_JSONL" \
    --n_long 2400 \
    --n_short 1000 \
    --seed "$SEED" \
    >>"$LOG_FILE" 2>&1
}

run_train() {
  local run_name="$1"
  local steps="$2"
  local max_seq_len="$3"
  local attn_impl="$4"
  local train_log="$REPO_DIR/artifacts/${run_name}_train.log"
  log "Train start: $run_name steps=$steps rank=$RANK max_seq_len=$max_seq_len attn_impl=$attn_impl"
  local macro_flags=()
  if [[ "$USE_MACRO_MICRO_KL" == "1" ]]; then
    macro_flags=(
      --use_macro_micro_kl
      --lambda_micro 0.005
      --lambda_macro 0.003
      --regularizer_warmup_steps 80
      --pref_warmup_batches 4
    )
  fi

  "$CONDA_BIN" run -n "$ENV_NAME" python "$REPO_DIR/scripts/isolated/attn/new_lora_longalpaca_attnbias_train.py" \
    --base_model_path "$BASE_MODEL" \
    --output_dir "$REPO_DIR/artifacts/new_attnbias_v1/train" \
    --run_name "$run_name" \
    --seed "$SEED" \
    --longalpaca_path "$LONG_JSONL" \
    --longqa_path "" \
    --longalpaca_mirror_url "$LONGALPACA_MIRROR_URL" \
    --allow_wikitext_as_long_fallback \
    --wikitext_train_path "$WIKITEXT_TRAIN" \
    --max_steps "$steps" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate 2e-5 \
    --warmup_steps 50 \
    --max_seq_len "$max_seq_len" \
    --lora_rank "$RANK" \
    --lora_alpha "$ALPHA" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --attn_implementation "$attn_impl" \
    --attn_bias_mode "$ATTN_BIAS_MODE" \
    --gamma_mode constant \
    --gamma 3e-4 \
    --gate_tau 0.0 \
    --gate_tg 1.0 \
    "${macro_flags[@]}" \
    >>"$train_log" 2>&1
  log "Train done: $run_name"
}

run_eval() {
  local run_name="$1"
  local eval_log="$REPO_DIR/artifacts/${run_name}_eval.log"
  local run_root="$REPO_DIR/artifacts/new_attnbias_v1/train/$run_name"
  local adapter_path="$run_root/adapter"
  local inv_path="$run_root/artifacts/custom_inv_freq.pt"
  local eval_root="$REPO_DIR/artifacts/new_attnbias_v1/eval/$run_name"
  log "Eval start: $run_name"
  "$CONDA_BIN" run -n "$ENV_NAME" python "$REPO_DIR/scripts/isolated/attn/new_eval_longbench_attnbias.py" \
    --base_model_path "$BASE_MODEL" \
    --adapter_path "$adapter_path" \
    --custom_inv_freq_path "$inv_path" \
    --output_root "$eval_root" \
    --seed "$SEED" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --max_batch_input_tokens "$EVAL_MAX_BATCH_INPUT_TOKENS" \
    --attn_bias_mode off \
    >>"$eval_log" 2>&1
  log "Eval done: $run_name"
}

main() {
  cd "$REPO_DIR"
  if [[ "$OFFLINE_MODE" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
  else
    unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE || true
  fi
  export TOKENIZERS_PARALLELISM=false
  log "Config: attn_bias_mode=$ATTN_BIAS_MODE use_macro_micro_kl=$USE_MACRO_MICRO_KL seq_smoke=$MAX_SEQ_LEN_SMOKE seq_full=$MAX_SEQ_LEN_FULL batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS attn_smoke=$ATTN_IMPL_SMOKE attn_full=$ATTN_IMPL_FULL lora_targets=$LORA_TARGET_MODULES eval_batch=$EVAL_BATCH_SIZE eval_token_budget=$EVAL_MAX_BATCH_INPUT_TOKENS"

  build_dataset
  run_train "$RUN_SMOKE" "$STEPS_SMOKE" "$MAX_SEQ_LEN_SMOKE" "$ATTN_IMPL_SMOKE"
  run_eval "$RUN_SMOKE"
  run_train "$RUN_FULL" "$STEPS_FULL" "$MAX_SEQ_LEN_FULL" "$ATTN_IMPL_FULL"
  run_eval "$RUN_FULL"
  log "Joint attention optimization queue finished."
}

main "$@"
