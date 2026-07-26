#!/usr/bin/env bash
set -euo pipefail

: "${ASSET_ROOT:?Set ASSET_ROOT to the OLMo-2 maturity asset directory}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the prepared Python executable}"

CODE_ROOT="${CODE_ROOT:-${ASSET_ROOT}/code}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ASSET_ROOT}/runs}"
CHECKPOINT="${ASSET_ROOT}/models/step30000_63B"
BACKGROUND="${ASSET_ROOT}/data/probe_background_v1"
RUN_NAME="step30_dense_position_answer_s20260725"
OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${ASSET_ROOT}/logs"
PID_DIR="${ASSET_ROOT}/pids"

test -f "${CHECKPOINT}/model-00001-of-00002.safetensors"
test -f "${CHECKPOINT}/model-00002-of-00002.safetensors"
test -f "${BACKGROUND}/documents_L16384.npy"
test -f "${BACKGROUND}/documents_L16384.metadata.json"
test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${PID_DIR}"

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

exec "${PYTHON_BIN}" -m \
  rebuttal.rebuttal_0723.experiments.olmo2_lora_generalization \
  --checkpoint "${CHECKPOINT}" \
  --documents-16k "${BACKGROUND}/documents_L16384.npy" \
  --documents-16k-metadata \
    "${BACKGROUND}/documents_L16384.metadata.json" \
  --output "${OUTPUT}" \
  --schedule evq \
  --adaptation qkvo_answer \
  --train-length 16384 \
  --eval-lengths 4096 8192 16384 \
  --steps 300 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --no-gradient-checkpointing \
  --compile-mode max-autotune-no-cudagraphs \
  --train-examples 1024 \
  --eval-examples-per-cell 15 \
  --rank 64 \
  --alpha 128 \
  --learning-rate 1e-4 \
  --warmup-steps 60 \
  --seed 20260725 \
  --eval-batch-size 4 \
  --canary-count 8 \
  --train-value-pool registered \
  --train-template-count 1 \
  --train-source-fractions \
    0.05 0.13 0.21 0.29 0.37 0.45 \
    0.53 0.61 0.69 0.77 0.85 0.93 \
  --margin-loss-weight 0.25 \
  --top1-margin 1.0 \
  --counterfactual-margin 1.0
