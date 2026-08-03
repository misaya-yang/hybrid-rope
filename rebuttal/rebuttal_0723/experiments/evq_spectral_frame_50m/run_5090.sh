#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${REPO_ROOT:-$(cd "${script_dir}/../../../.." && pwd)}"
: "${WORK_DIR:?set WORK_DIR to a data-disk output directory}"
: "${TRAIN_DATA:?set TRAIN_DATA to the verified 15M-token tensor}"
: "${VALIDATION_DATA:?set VALIDATION_DATA to the verified validation tensor}"
work_dir="${WORK_DIR}"
python_bin="${PYTHON_BIN:-python3}"
runner="${repo_root}/rebuttal/rebuttal_0723/experiments/evq_spectral_frame_50m/run_spectral_frame_50m.py"
train_data="${TRAIN_DATA}"
validation_data="${VALIDATION_DATA}"
prepared="${work_dir}/prepared.json"
cpu_ready="${work_dir}/cpu_ready.json"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${work_dir}/torchinductor_cache}"

mkdir -p "${work_dir}" "${TORCHINDUCTOR_CACHE_DIR}"

common=(
  "${runner}"
  --train-data "${train_data}"
  --validation-data "${validation_data}"
  --prepared-receipt "${prepared}"
)

case "${1:-}" in
  prepare)
    "${python_bin}" "${common[@]}" --mode prepare --output "${prepared}"
    ;;
  cpu-ready)
    "${python_bin}" "${common[@]}" --mode cpu-preflight --output "${cpu_ready}"
    ;;
  runtime-probe)
    "${python_bin}" "${common[@]}" \
      --mode runtime-probe \
      --cpu-ready-receipt "${cpu_ready}" \
      --output "${work_dir}/runtime_probe" \
      --device cuda \
      --micro-batch "${MICRO_BATCH:-128}" \
      --compile-mode "${COMPILE_MODE:-max-autotune-no-cudagraphs}"
    ;;
  train)
    runtime_ready="${RUNTIME_READY_RECEIPT:-${work_dir}/runtime_probe/runtime_ready.json}"
    "${python_bin}" "${common[@]}" \
      --mode train-suite \
      --cpu-ready-receipt "${cpu_ready}" \
      --runtime-ready-receipt "${runtime_ready}" \
      --output "${work_dir}/runs" \
      --device cuda \
      --micro-batch "${MICRO_BATCH:-128}" \
      --compile-mode "${COMPILE_MODE:-max-autotune-no-cudagraphs}"
    ;;
  *)
    echo "usage: $0 {prepare|cpu-ready|runtime-probe|train}" >&2
    exit 2
    ;;
esac
