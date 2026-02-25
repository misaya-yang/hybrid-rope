#!/usr/bin/env bash
set -euo pipefail

# Example launcher for single H100.
# Update model paths before running.

OUT_DIR=${OUT_DIR:-/opt/dfrope/results/passkey_h100}
PY=${PY:-python}

mkdir -p "${OUT_DIR}"

${PY} h100_advanced_experiments/scripts/run_passkey_h100.py \
  --model geo_500k:/opt/dfrope/checkpoints/geo_500k \
  --model hybrid_a0.2_t100k:/opt/dfrope/checkpoints/hybrid_a0.2_t100k \
  --output_dir "${OUT_DIR}" \
  --lengths 2048,4096,8192,12288,16384 \
  --depths 0.1,0.3,0.5,0.7,0.9 \
  --trials 3 \
  --dtype bf16 \
  --mcq \
  --generate

echo "Done. Results: ${OUT_DIR}/passkey_results.json"

