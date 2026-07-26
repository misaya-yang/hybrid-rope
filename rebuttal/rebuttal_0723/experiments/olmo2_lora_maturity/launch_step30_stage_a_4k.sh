#!/usr/bin/env bash
set -euo pipefail

: "${ASSET_ROOT:?Set ASSET_ROOT to the OLMo-2 maturity asset directory}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the prepared Python executable}"

CODE_ROOT="${CODE_ROOT:-${ASSET_ROOT}/code}"
CHECKPOINT="${ASSET_ROOT}/models/step30000_63B"
PREPARED="${ASSET_ROOT}/data/prepared_maturity_v1"
BACKGROUND="${ASSET_ROOT}/data/probe_background_v1"
RUN_NAME="${RUN_NAME:-step30_evq_stage_a_4k_20m_s20260725}"
OUTPUT="${ASSET_ROOT}/runs/${RUN_NAME}"
TARGET_SUPERVISED_TOKENS="${TARGET_SUPERVISED_TOKENS:-20000000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NATURAL_EVAL_ROWS="${NATURAL_EVAL_ROWS:-16}"
SKIP_NATURAL_EVAL="${SKIP_NATURAL_EVAL:-0}"

test -f "${CHECKPOINT}/model-00001-of-00002.safetensors"
test -f "${CHECKPOINT}/model-00002-of-00002.safetensors"
test -f "${PREPARED}/longalign_paired_L4096/input_ids.npy"
test -f "${PREPARED}/longalign_paired_L4096/manifest.json"
test -f "${BACKGROUND}/documents_L16384.npy"
test ! -e "${OUTPUT}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "${GPU_NAME}" in
  *"RTX 5090"*) ;;
  *) echo "Expected RTX 5090, found: ${GPU_NAME}" >&2; exit 1 ;;
esac

export PYTHONPATH="${CODE_ROOT}"
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="${ASSET_ROOT}/torchinductor_cache/rtx5090"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

EXTRA_ARGS=()
if [[ "${SKIP_NATURAL_EVAL}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-natural-eval)
fi

exec "${PYTHON_BIN}" -m \
  rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_stage_a \
  --checkpoint "${CHECKPOINT}" \
  --prepared-data "${PREPARED}" \
  --background-dir "${BACKGROUND}" \
  --output "${OUTPUT}" \
  --mode train \
  --frequency evq \
  --target-supervised-tokens "${TARGET_SUPERVISED_TOKENS}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --rank 64 \
  --alpha 128 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --compile-mode max-autotune-no-cudagraphs \
  --natural-eval-rows "${NATURAL_EVAL_ROWS}" \
  --natural-tail-tokens 1024 \
  --seed 20260725 \
  "${EXTRA_ARGS[@]}"
