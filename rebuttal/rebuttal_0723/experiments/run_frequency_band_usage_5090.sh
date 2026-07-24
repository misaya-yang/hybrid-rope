#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SOURCE_READY_DIR="${SOURCE_READY_DIR:-/root/autodl-tmp/profiled_residual_5090}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/frequency_band_usage_5090}"

cd "$REPO_ROOT"
case "$MODE" in
  preflight)
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.frequency_band_usage_5090 \
      preflight --source-ready-dir "$SOURCE_READY_DIR" --output-dir "$OUTPUT_DIR"
    ;;
  run)
    test -f "$OUTPUT_DIR/READY.json"
    mkdir -p "$OUTPUT_DIR/logs"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.frequency_band_usage_5090 \
      run --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/logs/run.log"
    ;;
  *)
    echo "usage: $0 {preflight|run}" >&2
    exit 2
    ;;
esac
