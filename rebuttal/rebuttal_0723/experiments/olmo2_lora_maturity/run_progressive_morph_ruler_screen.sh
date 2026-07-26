#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
READY_RECEIPT="${READY_RECEIPT:?set READY_RECEIPT}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
FWE_ROOT="${FWE_ROOT:?set FWE_ROOT}"
QA2_ROOT="${QA2_ROOT:?set QA2_ROOT}"
ADAPTER="${ADAPTER:?set ADAPTER to the completed progressive-morph adapter.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh result directory}"
LOG_ROOT="${LOG_ROOT:-$(dirname "$OUTPUT_ROOT")/logs/$(basename "$OUTPUT_ROOT")}"
PYTHON_BIN="${PYTHON_BIN:?set PYTHON_BIN}"
CODE_ROOT="${CODE_ROOT:?set CODE_ROOT}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LIMIT_PER_CELL="${LIMIT_PER_CELL:-5}"
FREQUENCY="${FREQUENCY:-evq}"

if [[ "$MAX_PARALLEL" -lt 1 || "$MAX_PARALLEL" -gt 2 ]]; then
  echo "MAX_PARALLEL must be 1 or 2" >&2
  exit 2
fi
if [[ "$FREQUENCY" != "native" \
  && "$FREQUENCY" != "evq" \
  && "$FREQUENCY" != "hybrid_evq_low4" \
  && "$FREQUENCY" != "hybrid_evq_low8" ]]; then
  echo "unsupported FREQUENCY: $FREQUENCY" >&2
  exit 2
fi

for path in \
  "$CHECKPOINT" \
  "$READY_RECEIPT" \
  "$DATA_ROOT" \
  "$FWE_ROOT" \
  "$QA2_ROOT" \
  "$ADAPTER"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$OUTPUT_ROOT/.status"
export PYTHONPATH="$CODE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

tasks=(niah_single_1 niah_multikey_1 niah_multiquery vt cwe fwe qa_1)

task_root() {
  local task="$1"
  if [[ "$task" == "fwe" ]]; then
    printf '%s\n' "$FWE_ROOT"
  else
    printf '%s\n' "$DATA_ROOT/$task"
  fi
}

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL" ]]; do
    sleep 2
  done
}

run_group() {
  local label="$1"
  local prepared_root="$2"
  local task="$3"
  shift 3
  local lengths=("$@")
  local output="$OUTPUT_ROOT/$label"
  local log="$LOG_ROOT/$label.log"
  local status="$OUTPUT_ROOT/.status/$label.status"

  if [[ -f "$output/results.json" ]]; then
    echo "complete" >"$status"
    return
  fi
  if [[ -e "$output" ]]; then
    echo "failed:pre-existing-output" >"$status"
    return
  fi

  if "$PYTHON_BIN" -m \
    rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer \
    --checkpoint "$CHECKPOINT" \
    --ready-receipt "$READY_RECEIPT" \
    --data-root "$prepared_root" \
    --output "$output" \
    --frequency "$FREQUENCY" \
    --adapter "$ADAPTER" \
    --rank 64 \
    --alpha 128 \
    --tasks "$task" \
    --lengths "${lengths[@]}" \
    --limit-per-cell "$LIMIT_PER_CELL" >"$log" 2>&1; then
    echo "complete" >"$status"
  else
    echo "failed:$?" >"$status"
  fi
}

for task in "${tasks[@]}"; do
  wait_for_slot
  run_group \
    "$task" \
    "$(task_root "$task")" \
    "$task" \
    4096 8192 16384 &
done

for length in 4096 8192 16384; do
  wait_for_slot
  run_group "qa_2_L${length}" "$QA2_ROOT/qa_2_L${length}$(
    [[ "$length" == "4096" ]] && printf '%s' '_fixed'
  )" qa_2 "$length" &
done

wait

if grep -nH '^failed:' "$OUTPUT_ROOT"/.status/*.status >/dev/null 2>&1; then
  grep -nH '^failed:' "$OUTPUT_ROOT"/.status/*.status >&2
  exit 1
fi

expected=10
completed="$(
  find "$OUTPUT_ROOT/.status" -type f -name '*.status' \
    -exec grep -l '^complete$' {} + | wc -l | tr -d ' '
)"
if [[ "$completed" -ne "$expected" ]]; then
  echo "result-count drift: completed=$completed expected=$expected" >&2
  exit 1
fi

touch "$OUTPUT_ROOT/ALL_DONE"
echo "progressive-morph RULER screen complete: $OUTPUT_ROOT"
