#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
ROOT="${EVQ_REPO_ROOT:?set EVQ_REPO_ROOT}"
PYTHON="${EVQ_PYTHON:-python}"
RUN_ROOT="${EVQ_ALLOCATION_ORACLE_ROOT:?set EVQ_ALLOCATION_ORACLE_ROOT}"
CHECKPOINT="${EVQ_OLMO_CHECKPOINT:?set EVQ_OLMO_CHECKPOINT}"
CHECKPOINT_READY="${EVQ_OLMO_CHECKPOINT_READY:?set EVQ_OLMO_CHECKPOINT_READY}"
PHASE_VIEW="${EVQ_OLMO_PHASE_VIEW:?set EVQ_OLMO_PHASE_VIEW}"
DATA_V3="${EVQ_OLMO_DATA_V3:?set EVQ_OLMO_DATA_V3}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"

case "$MODE" in
  preflight)
    READY="$RUN_ROOT/full_ready.json"
    OUTPUT="$RUN_ROOT/full"
    ;;
  smoke)
    READY="$RUN_ROOT/smoke_ready.json"
    OUTPUT="$RUN_ROOT/smoke"
    ;;
  run)
    READY="$RUN_ROOT/full_ready.json"
    OUTPUT="$RUN_ROOT/full"
    ;;
  *)
    echo "usage: $0 {preflight|smoke|run}" >&2
    exit 2
    ;;
esac

COMMON=(
  --checkpoint "$CHECKPOINT"
  --checkpoint-ready-receipt "$CHECKPOINT_READY"
  --phase-view "$PHASE_VIEW"
  --raw-replay "$DATA_V3/warmup4k_clm"
  --retention-view "$DATA_V3/retention4k_raw"
  --ready-receipt "$READY"
  --output "$OUTPUT"
  --steps 300
  --validation-rows 32
  --compile-mode max-autotune-no-cudagraphs
  --seed 20260825
)

if [[ "$MODE" == "preflight" || "$MODE" == "smoke" ]]; then
  EXTRA=()
  [[ "$MODE" == "smoke" ]] && EXTRA+=(--smoke)
  "$PYTHON" -m rebuttal.rebuttal_0723.experiments.olmo2_allocation_oracle_5090.preflight "${COMMON[@]}" "${EXTRA[@]}"
fi

if [[ "$MODE" == "smoke" || "$MODE" == "run" ]]; then
  [[ "${OLMO_ALLOCATION_ORACLE_GPU_AUTHORIZED:-}" == "YES" ]] || {
    echo "set OLMO_ALLOCATION_ORACLE_GPU_AUTHORIZED=YES after explicit GPU authorization" >&2
    exit 3
  }
  EXTRA=(--authorize)
  [[ "$MODE" == "smoke" ]] && EXTRA+=(--smoke)
  exec "$PYTHON" -m rebuttal.rebuttal_0723.experiments.olmo2_allocation_oracle_5090.train "${COMMON[@]}" "${EXTRA[@]}"
fi
