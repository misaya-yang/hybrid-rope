#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT="${ASSET_ROOT:-/root/autodl-tmp/olmo2_1b_longalign_assets}"
CHECKPOINT="${CHECKPOINT:-$ASSET_ROOT/models/OLMo-2-0425-1B-Instruct}"
READY_RECEIPT="${READY_RECEIPT:-$ASSET_ROOT/receipts/instruct_4k_conversion_ready.json}"
DATA_ROOT="${DATA_ROOT:-$ASSET_ROOT/data/ruler_full_merged_n20_s20260802}"
NATIVE_ADAPTER="${NATIVE_ADAPTER:?set NATIVE_ADAPTER}"
EVQ_ADAPTER="${EVQ_ADAPTER:?set EVQ_ADAPTER}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh result directory}"
LOG_ROOT="${LOG_ROOT:-$ASSET_ROOT/logs/$(basename "$OUTPUT_ROOT")}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LIMIT_PER_CELL="${LIMIT_PER_CELL:-20}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CODE_ROOT="${CODE_ROOT:-$ASSET_ROOT/code}"

if [[ "$MAX_PARALLEL" -lt 1 || "$MAX_PARALLEL" -gt 2 ]]; then
  echo "MAX_PARALLEL must be 1 or 2 on the RTX 5090" >&2
  exit 2
fi
if [[ "$LIMIT_PER_CELL" -lt 1 || "$LIMIT_PER_CELL" -gt 20 ]]; then
  echo "LIMIT_PER_CELL must be in [1, 20]" >&2
  exit 2
fi
for path in \
  "$CHECKPOINT" \
  "$READY_RECEIPT" \
  "$DATA_ROOT" \
  "$NATIVE_ADAPTER" \
  "$EVQ_ADAPTER"; do
  [[ -e "$path" ]] || {
    echo "missing required input: $path" >&2
    exit 2
  }
done

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$OUTPUT_ROOT/.status"
export PYTHONPATH="$CODE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

tasks=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery
  vt cwe fwe qa_1 qa_2
)

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL" ]]; do
    sleep 2
  done
}

run_arm() {
  local label="$1"
  local frequency="$2"
  local factor="$3"
  local adapter="$4"
  local length="$5"
  local output="$OUTPUT_ROOT/$label"
  local log="$LOG_ROOT/$label.log"
  local status="$OUTPUT_ROOT/.status/${label//\//_}.status"
  local adapter_args=()

  if [[ "$adapter" != "none" ]]; then
    adapter_args=(
      --adapter "$adapter"
      --rank 64
      --alpha 128
    )
  fi

  if [[ -f "$output/results.json" ]]; then
    echo "complete" >"$status"
    return
  fi
  if [[ -e "$output" ]]; then
    echo "failed:pre-existing-output" >"$status"
    return
  fi
  mkdir -p "$(dirname "$output")" "$(dirname "$log")"

  if "$PYTHON_BIN" -m \
    rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer \
    --checkpoint "$CHECKPOINT" \
    --ready-receipt "$READY_RECEIPT" \
    --data-root "$DATA_ROOT" \
    --output "$output" \
    --frequency "$frequency" \
    --yarn-factor "$factor" \
    "${adapter_args[@]}" \
    --tasks "${tasks[@]}" \
    --lengths "$length" \
    --limit-per-cell "$LIMIT_PER_CELL" >"$log" 2>&1; then
    echo "complete" >"$status"
  else
    echo "failed:$?" >"$status"
  fi
}

wait_for_slot
run_arm \
  "untouched_repo_ramp2" \
  repo_fixed_ramp 2 none 8192 &
wait_for_slot
run_arm \
  "untouched_repo_ramp4" \
  repo_fixed_ramp 4 none 16384 &
wait_for_slot
run_arm \
  "native_lora_repo_ramp2" \
  repo_fixed_ramp 2 "$NATIVE_ADAPTER" 8192 &
wait_for_slot
run_arm \
  "native_lora_repo_ramp4" \
  repo_fixed_ramp 4 "$NATIVE_ADAPTER" 16384 &
wait_for_slot
run_arm \
  "evq_lora_repo_ramp2" \
  evq_repo_fixed_ramp 2 "$EVQ_ADAPTER" 8192 &
wait_for_slot
run_arm \
  "evq_lora_repo_ramp4" \
  evq_repo_fixed_ramp 4 "$EVQ_ADAPTER" 16384 &

wait

if grep -nH '^failed:' "$OUTPUT_ROOT"/.status/*.status >/dev/null 2>&1; then
  grep -nH '^failed:' "$OUTPUT_ROOT"/.status/*.status >&2
  exit 1
fi

expected=6
completed="$(
  find "$OUTPUT_ROOT/.status" -type f -name '*.status' \
    -exec grep -l '^complete$' {} + | wc -l | tr -d ' '
)"
if [[ "$completed" -ne "$expected" ]]; then
  echo "result-count drift: completed=$completed expected=$expected" >&2
  exit 1
fi

touch "$OUTPUT_ROOT/ALL_DONE"
echo "adapted repository fixed-ramp evaluation complete: $OUTPUT_ROOT"
