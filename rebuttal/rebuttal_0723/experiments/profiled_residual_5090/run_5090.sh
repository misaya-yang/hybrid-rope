#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SOURCE_WORK_DIR="${SOURCE_WORK_DIR:-/root/autodl-tmp/fmrope_l256_500m}"
SOURCE_DATA_MANIFEST="${SOURCE_DATA_MANIFEST:-$SOURCE_WORK_DIR/data/data_manifest.json}"
ANCHOR_MANIFEST="${ANCHOR_MANIFEST:-/root/autodl-tmp/reviewer27be_shape_base/data/data_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/profiled_residual_5090}"

export OMP_NUM_THREADS="${PROFILE_CPU_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_python() {
  cd "$REPO_ROOT"
  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
    -m rebuttal.rebuttal_0723.experiments.profiled_residual_5090.run_experiment "$@"
}

case "$MODE" in
  preflight)
    run_python preflight \
      --source-work-dir "$SOURCE_WORK_DIR" \
      --source-data-manifest "$SOURCE_DATA_MANIFEST" \
      --anchor-manifest "$ANCHOR_MANIFEST" \
      --output-dir "$OUTPUT_DIR"
    ;;
  run-primary)
    [[ -f "$OUTPUT_DIR/READY.json" ]] || { echo "missing READY.json" >&2; exit 2; }
    mkdir -p "$OUTPUT_DIR/logs"
    run_python profile --split selection --arms fmrope_base256 \
      --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/logs/selection_primary.log"
    run_python select --output-dir "$OUTPUT_DIR"
    run_python profile --split test --arms fmrope_base256 \
      --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/logs/test_primary.log"
    run_python summarize --output-dir "$OUTPUT_DIR"
    ;;
  run-full)
    [[ -f "$OUTPUT_DIR/READY.json" ]] || { echo "missing READY.json" >&2; exit 2; }
    mkdir -p "$OUTPUT_DIR/logs"
    run_python profile --split selection --output-dir "$OUTPUT_DIR" \
      2>&1 | tee "$OUTPUT_DIR/logs/selection_full.log"
    run_python select --output-dir "$OUTPUT_DIR"
    run_python profile --split test --output-dir "$OUTPUT_DIR" \
      2>&1 | tee "$OUTPUT_DIR/logs/test_full.log"
    run_python summarize --output-dir "$OUTPUT_DIR"
    ;;
  *)
    echo "usage: $0 {preflight|run-primary|run-full}" >&2
    exit 2
    ;;
esac
