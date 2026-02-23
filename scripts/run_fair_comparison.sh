#!/usr/bin/env bash
set -euo pipefail

# Fair comparison: all methods share exactly the same hyperparameters.
# Only RoPE frequency allocation method changes.

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/llama8b_fair_v2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
MAX_STEPS="${MAX_STEPS:-400}"
SEED="${SEED:-42}"
SCRIPT_PATH="${SCRIPT_PATH:-2026-02-22/scripts/run_llama8b_fair_suite.py}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[fatal] no python interpreter found. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

echo "[env] python_bin=$PYTHON_BIN"
echo "[env] script_path=$SCRIPT_PATH"

COMMON_ARGS=(
  --base_model_path "$BASE_MODEL"
  --output_root "$OUTPUT_ROOT"
  --max_seq_len "$MAX_SEQ_LEN"
  --max_steps "$MAX_STEPS"
  --seed "$SEED"
  --per_device_train_batch_size 2
  --gradient_accumulation_steps 2
  --learning_rate 2e-4
  --bf16
  --lora_rank 64
  --lora_alpha 128
  --logging_steps 10
  --save_steps 200
  --warmup_steps 20
)

echo "====== [1/5] Baseline ======"
"$PYTHON_BIN" "$SCRIPT_PATH" --method baseline --run_name baseline "${COMMON_ARGS[@]}"

echo "====== [2/5] PI ======"
"$PYTHON_BIN" "$SCRIPT_PATH" --method pi --run_name pi "${COMMON_ARGS[@]}"

echo "====== [3/5] YaRN ======"
"$PYTHON_BIN" "$SCRIPT_PATH" --method yarn --run_name yarn "${COMMON_ARGS[@]}"

echo "====== [4/5] Sigmoid ======"
"$PYTHON_BIN" "$SCRIPT_PATH" --method sigmoid --run_name sigmoid "${COMMON_ARGS[@]}"

echo "====== [5/5] Anchored Sigmoid ======"
"$PYTHON_BIN" "$SCRIPT_PATH" --method anchored_sigmoid --run_name anchored_sigmoid "${COMMON_ARGS[@]}"

echo "====== ALL DONE ======"
echo "Results in: $OUTPUT_ROOT"
