#!/usr/bin/env bash
set -euo pipefail

: "${ASSET_ROOT:?Set ASSET_ROOT to the OLMo-2 maturity asset directory}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the prepared Python executable}"

CODE_ROOT="${CODE_ROOT:-${ASSET_ROOT}/code}"
CHECKPOINT="${ASSET_ROOT}/models/step30000_63B"
BACKGROUND="${ASSET_ROOT}/data/probe_background_v1"
ADAPTER="${ASSET_ROOT}/runs/step30_dense_position_answer_s20260725/adapter.pt"
OUTPUT="${ASSET_ROOT}/runs/step30_dense_position_ood_factorial_s20260725"

test -f "${CHECKPOINT}/model-00001-of-00002.safetensors"
test -f "${CHECKPOINT}/model-00002-of-00002.safetensors"
test -f "${BACKGROUND}/documents_L16384.npy"
test -f "${BACKGROUND}/documents_L16384.metadata.json"
test -f "${ADAPTER}"
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

exec "${PYTHON_BIN}" -m \
  rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial \
  --checkpoint "${CHECKPOINT}" \
  --adapter "${ADAPTER}" \
  --documents-16k "${BACKGROUND}/documents_L16384.npy" \
  --documents-16k-metadata \
    "${BACKGROUND}/documents_L16384.metadata.json" \
  --output "${OUTPUT}" \
  --schedule evq \
  --adaptation qkvo_answer \
  --rank 64 \
  --alpha 128 \
  --examples-per-cell 72 \
  --eval-batch-size 4 \
  --seed 20260725 \
  --train-source-fractions \
    0.05 0.13 0.21 0.29 0.37 0.45 \
    0.53 0.61 0.69 0.77 0.85 0.93
