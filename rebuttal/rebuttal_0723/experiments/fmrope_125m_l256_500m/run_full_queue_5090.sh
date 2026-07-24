#!/usr/bin/env bash
set -euo pipefail

: "${SMALL_REPO:?}"
: "${SMALL_WORK:?}"
: "${SMALL_DATA:?}"
: "${LARGE_REPO:?}"
: "${LARGE_WORK:?}"
: "${SEED42_COMPARISON:?}"

export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

cd "$SMALL_REPO"
FMR_WORK_DIR="$SMALL_WORK" \
FMR_DATA_DIR="$SMALL_DATA" \
  bash rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/run_5090.sh \
  run-multiseed

FMR_WORK_DIR="$SMALL_WORK" \
FMR_SEED42_COMPARISON="$SEED42_COMPARISON" \
  bash rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/run_5090.sh \
  aggregate-multiseed

while [[ ! -s "$LARGE_WORK/READY" ]]; do
  sleep 30
done

cd "$LARGE_REPO"
FMR_WORK_DIR="$LARGE_WORK" \
  bash rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/run_5090.sh \
  run-350m
