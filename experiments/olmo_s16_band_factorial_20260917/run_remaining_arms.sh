#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/autodl-tmp/today_rope_plan_20260914/olmo_s16_band_factorial_20260917}"
PROJECT="${PROJECT:-/root/autodl-tmp/hybrid-rope}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct}"
DATA="${DATA:-/root/autodl-tmp/today_rope_plan_20260914/tailspline_olmo_s4_classic/assets/ppl46/manifest.json}"
PANEL="${PANEL:-$ROOT/assets/panels/65536/inputs.jsonl}"
WAIT_PID="${WAIT_PID:-}"

mkdir -p "$ROOT/logs" "$ROOT/runs"
cd "$PROJECT"

arm_complete() {
  local arm="$1"
  "$PYTHON" - "$ROOT/runs/$arm/status.json" "$ROOT/runs/$arm/generations.jsonl" <<'PY'
import json
import pathlib
import sys

status_path = pathlib.Path(sys.argv[1])
rows_path = pathlib.Path(sys.argv[2])
if not status_path.exists() or not rows_path.exists():
    raise SystemExit(1)
status = json.loads(status_path.read_text())
rows = sum(1 for line in rows_path.open() if line.strip())
raise SystemExit(0 if status.get("status") == "COMPLETE" and rows == 16 else 1)
PY
}

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 15
  done
  if ! arm_complete band18_32; then
    echo "band18_32 did not complete; preserving its raw output and stopping the queue" >&2
    exit 1
  fi
fi

for arm in band14_35 band18_35; do
  if arm_complete "$arm"; then
    echo "$arm already complete; skipping"
    continue
  fi

  echo "starting $arm"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
      --data "$DATA" \
      --model "$MODEL" \
      --arm Native \
      --extra-panel "$PANEL" \
      --only-extra-panels \
      --skip-lm \
      --length-cap 65536 \
      --prefill-chunk-size 32768 \
      --batch-size 1 \
      --static-table-json "$ROOT/tables/$arm.json" \
      --table-label "olmo_s16_tailspline_$arm" \
      --out "$ROOT/runs/$arm" \
      --execute \
      >"$ROOT/logs/$arm.out" 2>&1

  if ! arm_complete "$arm"; then
    echo "$arm exited without a complete 16-row result" >&2
    exit 1
  fi
  echo "$arm complete"
done

echo "ALL_REMAINING_ARMS_COMPLETE"
