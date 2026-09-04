#!/usr/bin/env bash
# Unified entrypoint for the Pro scale-covariance theorem and behaviour program.
set -euo pipefail

MODE="${1:-}"
REPO="${PRO_SCALE_REPO_ROOT:?set PRO_SCALE_REPO_ROOT}"
PYTHON="${PRO_SCALE_PYTHON:-python3}"
OUTPUT_ROOT="${PRO_SCALE_OUTPUT_ROOT:?set PRO_SCALE_OUTPUT_ROOT}"
TABLE_MANIFEST="${PRO_SCALE_TABLE_MANIFEST:-}"

run_cpu() {
  if [[ -n "$TABLE_MANIFEST" ]]; then
    CUDA_VISIBLE_DEVICES=-1 "$PYTHON" \
      "$REPO/scripts/analysis/finite_scale_covariance.py" \
      "$@" --table-manifest "$TABLE_MANIFEST"
  else
    CUDA_VISIBLE_DEVICES=-1 "$PYTHON" \
      "$REPO/scripts/analysis/finite_scale_covariance.py" "$@"
  fi
}

run_operator() {
  if [[ -z "$TABLE_MANIFEST" ]]; then
    echo "PRO_SCALE_TABLE_MANIFEST is required for operator modes" >&2
    exit 2
  fi
  "$PYTHON" "$REPO/scripts/analysis/optimize_scale_conjugacy.py" \
    --table-manifest "$TABLE_MANIFEST" "$@"
}

case "$MODE" in
  cpu-preflight)
    run_cpu --preflight-only
    ;;
  cpu-smoke)
    run_cpu \
      --output "$OUTPUT_ROOT/cpu_smoke.json" --random-cases 4 \
      --factors 2 --levels 1 --horizons 32 --max-discrete-dimension 4
    ;;
  cpu-full)
    run_cpu --output "$OUTPUT_ROOT/cpu_full.json"
    ;;
  operator-preflight)
    run_operator --preflight-only
    ;;
  operator-smoke)
    run_operator \
      --output "$OUTPUT_ROOT/operator_smoke.json" \
      --tables native legacy_u_p2_log_s4 min_q_exact_chain_s4 \
      --scales 4 --levels 1 --horizons 256 --condition-caps 1 8 \
      --steps 20 --restarts 1 --train-positions 16 --eval-positions 32 \
      --power-iterations 4
    ;;
  operator-primary)
    run_operator \
      --output "$OUTPUT_ROOT/operator_primary.json" \
      --scales 4 --levels 1 --horizons 4096 --condition-caps 1 8 64 \
      --steps 300 --restarts 3 --train-positions 64 --eval-positions 256
    ;;
  operator-multilevel)
    run_operator \
      --output "$OUTPUT_ROOT/operator_multilevel.json" \
      --tables native legacy_u_p2_log_s4 min_q_exact_chain_s4 \
      --scales 2 4 8 --levels 1 2 4 --horizons 4096 16384 \
      --condition-caps 1 8 64 --steps 300 --restarts 3 \
      --train-positions 64 --eval-positions 256
    ;;
  gpu-preflight)
    "$REPO/scripts/eval/run_scale_orbit_validation_5090.sh" preflight
    ;;
  gpu-smoke)
    "$REPO/scripts/eval/run_scale_orbit_validation_5090.sh" gpu-smoke
    ;;
  gpu-primary)
    "$REPO/scripts/eval/run_scale_orbit_validation_5090.sh" gpu-primary-formal
    "$REPO/scripts/eval/run_scale_orbit_validation_5090.sh" gpu-primary-ruler
    ;;
  gpu-breadth)
    "$REPO/scripts/eval/run_scale_orbit_validation_5090.sh" gpu-breadth-formal
    "$REPO/scripts/eval/run_scale_orbit_validation_5090.sh" gpu-breadth-ruler
    ;;
  summarize)
    "$REPO/scripts/eval/run_scale_orbit_validation_5090.sh" summarize
    ;;
  *)
    echo "usage: $0 {cpu-preflight|cpu-smoke|cpu-full|operator-preflight|operator-smoke|operator-primary|operator-multilevel|gpu-preflight|gpu-smoke|gpu-primary|gpu-breadth|summarize}" >&2
    exit 2
    ;;
esac
