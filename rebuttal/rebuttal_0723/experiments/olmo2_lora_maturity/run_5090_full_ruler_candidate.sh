#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT="${ASSET_ROOT:-/root/autodl-tmp/olmo2_1b_longalign_assets}"
CHECKPOINT="${CHECKPOINT:-$ASSET_ROOT/models/OLMo-2-0425-1B-Instruct}"
READY_RECEIPT="${READY_RECEIPT:-$ASSET_ROOT/receipts/instruct_4k_conversion_ready.json}"
DATA_ROOT="${DATA_ROOT:-$ASSET_ROOT/data/ruler_full_n20_s20260728}"
FWE_ROOT="${FWE_ROOT:-$DATA_ROOT/fwe}"
QA2_ROOT="${QA2_ROOT:-$ASSET_ROOT/data/ruler_qa2_fixed_n20_s20260728}"
ADAPTER="${ADAPTER:?set ADAPTER to the completed candidate adapter.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh result directory}"
LOG_ROOT="${LOG_ROOT:-$ASSET_ROOT/logs/$(basename "$OUTPUT_ROOT")}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CODE_ROOT="${CODE_ROOT:-$ASSET_ROOT/code}"
FREQUENCY="${FREQUENCY:-evq}"

if [[ "$MAX_PARALLEL" -lt 1 || "$MAX_PARALLEL" -gt 2 ]]; then
  echo "MAX_PARALLEL must be 1 or 2 on the RTX 5090" >&2
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
  "$ADAPTER" \
  "$DATA_ROOT" \
  "$FWE_ROOT" \
  "$QA2_ROOT"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$OUTPUT_ROOT/.status"
export PYTHONPATH="$CODE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

tasks=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery
  vt cwe fwe qa_1
)

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

run_cell_group() {
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
    echo "refusing incomplete pre-existing output: $output" >&2
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
    --limit-per-cell 20 >"$log" 2>&1; then
    echo "complete" >"$status"
  else
    local rc=$?
    echo "failed:$rc" >"$status"
  fi
}

for task in "${tasks[@]}"; do
  wait_for_slot
  run_cell_group "$task" "$(task_root "$task")" "$task" 4096 8192 16384 &
done

qa2_roots=(
  "$QA2_ROOT/qa_2_L4096_fixed"
  "$QA2_ROOT/qa_2_L8192"
  "$QA2_ROOT/qa_2_L16384"
)
qa2_lengths=(4096 8192 16384)
for index in 0 1 2; do
  wait_for_slot
  run_cell_group \
    "qa_2_L${qa2_lengths[$index]}" \
    "${qa2_roots[$index]}" \
    qa_2 \
    "${qa2_lengths[$index]}" &
done

wait

if grep -nH '^failed:' "$OUTPUT_ROOT"/.status/*.status >/dev/null 2>&1; then
  grep -nH '^failed:' "$OUTPUT_ROOT"/.status/*.status >&2
  exit 1
fi

expected=15
completed="$(
  find "$OUTPUT_ROOT/.status" -type f -name '*.status' \
    -exec grep -l '^complete$' {} + | wc -l | tr -d ' '
)"
if [[ "$completed" -ne "$expected" ]]; then
  echo "result-count drift: completed=$completed expected=$expected" >&2
  exit 1
fi

touch "$OUTPUT_ROOT/ALL_DONE"
echo "full RULER candidate evaluation complete: $OUTPUT_ROOT"
