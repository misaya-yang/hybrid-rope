#!/usr/bin/env bash
# Driver for the fixed-support allocation dose-response study.
#
#   build   CPU only, no authorization: freeze the table family and the
#           falsifiable static prediction.
#   screen  GPU: 32 documents over the whole grid, for a fast Pareto shape.
#   full    GPU: every document over the whole grid, the reportable endpoint.
#   ruler   GPU: capability confirmation for named tables, through the
#           existing target-free harnesses.  No new evaluation code.
#
# Every GPU stage needs BOTH --authorize and the environment gate, so no stage
# here can start paid compute by accident.
set -euo pipefail

MODE="${1:-build}"; shift || true

ROOT="${EVQ_REPO_ROOT:?set EVQ_REPO_ROOT}"
PYTHON="${EVQ_PYTHON:-/root/miniconda3/bin/python}"
OUT_ROOT="${EVQ_DOSE_ROOT:?set EVQ_DOSE_ROOT}"
CHECKPOINT="${EVQ_OLMO_CHECKPOINT:?set EVQ_OLMO_CHECKPOINT}"
LONG_ROWS="${EVQ_LONG_ROWS:?set EVQ_LONG_ROWS}"
LEARNED_TABLE="${EVQ_LEARNED_TABLE:-}"
RULER_DATA="${EVQ_RULER_DATA:-}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
GRID="$OUT_ROOT/grid"
mkdir -p "$OUT_ROOT"

case "$MODE" in
  build)
    ARGS=(--output "$GRID")
    [[ -n "$LEARNED_TABLE" ]] && ARGS+=(--learned-table "$LEARNED_TABLE")
    exec "$PYTHON" "$ROOT/scripts/analysis/allocation_dose_grid.py" "${ARGS[@]}"
    ;;

  screen|full)
    [[ "${ALLOCATION_DOSE_GRID_GPU_AUTHORIZED:-}" == "YES" ]] || {
      echo "set ALLOCATION_DOSE_GRID_GPU_AUTHORIZED=YES after explicit GPU authorization" >&2
      exit 3
    }
    [[ -f "$GRID/dose_grid_manifest.json" ]] || { echo "run '$0 build' first" >&2; exit 4; }
    if [[ "$MODE" == "screen" ]]; then LIMIT=32; OUT="$OUT_ROOT/screen"; else LIMIT=0; OUT="$OUT_ROOT/full"; fi
    exec "$PYTHON" "$ROOT/scripts/eval/eval_allocation_dose_grid.py" \
      --checkpoint "$CHECKPOINT" \
      --dose-manifest "$GRID/dose_grid_manifest.json" \
      --long-rows "$LONG_ROWS" \
      --multipliers 1 2 4 \
      --limit-rows "$LIMIT" \
      --output "$OUT" \
      --authorize "$@"
    ;;

  ruler)
    # usage: run_allocation_dose_grid_5090.sh ruler <table-name> [<table-name> ...]
    [[ -n "$RULER_DATA" ]] || { echo "set EVQ_RULER_DATA" >&2; exit 5; }
    [[ $# -ge 1 ]] || { echo "usage: $0 ruler <table-name> [...]" >&2; exit 2; }
    for NAME in "$@"; do
      TABLE="$GRID/tables/$NAME.npy"
      [[ -f "$TABLE" ]] || { echo "missing table $TABLE" >&2; exit 6; }
      for LENGTH in 8192 16384; do
        "$PYTHON" "$ROOT/scripts/eval/target_free_ruler_smoke.py" \
          --checkpoint "$CHECKPOINT" \
          --data-root "$RULER_DATA" \
          --method external_table_static \
          --table "$TABLE" \
          --table-name "$NAME" \
          --table-support native \
          --table-factor 4.0 \
          --long-attention-scaling 1.0 \
          --lengths "$LENGTH" \
          --limit-per-cell 20 \
          --native-context-length 4096 \
          --output "$OUT_ROOT/ruler_${NAME}_${LENGTH}"
      done
    done
    ;;

  *)
    echo "usage: $0 {build|screen|full|ruler}" >&2
    exit 2
    ;;
esac
