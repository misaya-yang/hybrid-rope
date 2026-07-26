#!/usr/bin/env bash
set -euo pipefail

: "${ASSET_ROOT:?Set ASSET_ROOT to the OLMo-2 maturity asset directory}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the prepared Python executable}"

CODE_ROOT="${CODE_ROOT:-${ASSET_ROOT}/code}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ASSET_ROOT}/runs}"
CHECKPOINT="${ASSET_ROOT}/models/step30000_63B"
BACKGROUND="${ASSET_ROOT}/data/probe_background_v1"
PARENT="${OUTPUT_ROOT}/step30_dense_position_answer_s20260725/adapter.pt"
OUTPUT="${OUTPUT_ROOT}/step30_dense_position_answer_continue600_s20260725"

test -f "${PARENT}"
test ! -e "${OUTPUT}"
mkdir -p "${OUTPUT_ROOT}"

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
  --resume-adapter "${PARENT}" \
  --train-length 16384 \
  --eval-lengths 4096 8192 16384 \
  --steps 600 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --no-gradient-checkpointing \
  --compile-mode max-autotune-no-cudagraphs \
  --train-examples 1024 \
  --eval-examples-per-cell 15 \
  --rank 64 \
  --alpha 128 \
  --learning-rate 5e-5 \
  --warmup-steps 20 \
  --seed 20260725 \
  --eval-batch-size 4 \
  --canary-count 8 \
  --train-value-pool registered \
  --train-template-count 1 \
  --train-source-fractions \
    0.05 0.13 0.21 0.29 0.37 0.45 \
    0.53 0.61 0.69 0.77 0.85 0.93
