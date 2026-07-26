#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT="${ASSET_ROOT:-/root/autodl-tmp/olmo2_1b_longalign_assets}"
CODE_ROOT="${CODE_ROOT:-$ASSET_ROOT/code}"
CHECKPOINT="${CHECKPOINT:-$ASSET_ROOT/models/OLMo-2-0425-1B-Instruct}"
READY_RECEIPT="${READY_RECEIPT:-$ASSET_ROOT/receipts/instruct_4k_conversion_ready.json}"
DATA_ROOT="${DATA_ROOT:-$ASSET_ROOT/data/ruler_full_merged_n20_s20260802}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh directory}"
LOG_ROOT="${LOG_ROOT:-$ASSET_ROOT/logs/$(basename "$OUTPUT_ROOT")}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

EVQ_SEED25_ADAPTER="${EVQ_SEED25_ADAPTER:-$ASSET_ROOT/runs/instruct_evq_counterfactual_routing_4k_300_s20260725/adapter.pt}"
EVQ_SEED26_ADAPTER="${EVQ_SEED26_ADAPTER:-$ASSET_ROOT/runs/instruct_evq_counterfactual_routing_4k_300_s20260726/adapter.pt}"

if [[ "$MAX_PARALLEL" -lt 1 || "$MAX_PARALLEL" -gt 2 ]]; then
  echo "MAX_PARALLEL must be 1 or 2 on RTX 5090" >&2
  exit 2
fi
for path in \
  "$CHECKPOINT" "$READY_RECEIPT" "$DATA_ROOT" \
  "$EVQ_SEED25_ADAPTER" "$EVQ_SEED26_ADAPTER"; do
  [[ -e "$path" ]] || {
    echo "missing input: $path" >&2
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
  local adapter="$3"
  local output="$OUTPUT_ROOT/$label"
  local log="$LOG_ROOT/$label.log"
  local status="$OUTPUT_ROOT/.status/$label.status"

  if [[ -e "$output" ]]; then
    echo "failed:pre-existing-output" >"$status"
    return
  fi
  if "$PYTHON_BIN" -m \
    rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer \
    --checkpoint "$CHECKPOINT" \
    --ready-receipt "$READY_RECEIPT" \
    --data-root "$DATA_ROOT" \
    --output "$output" \
    --frequency "$frequency" \
    --adapter "$adapter" \
    --rank 64 \
    --alpha 128 \
    --tasks "${tasks[@]}" \
    --lengths 4096 8192 16384 \
    --limit-per-cell 20 >"$log" 2>&1; then
    echo complete >"$status"
  else
    echo "failed:$?" >"$status"
  fi
}

wait_for_slot
run_arm evq_seed20260725 evq "$EVQ_SEED25_ADAPTER" &
wait_for_slot
run_arm evq_seed20260726 evq "$EVQ_SEED26_ADAPTER" &
wait

if grep -nH '^failed:' "$OUTPUT_ROOT"/.status/*.status >/dev/null 2>&1; then
  grep -nH '^failed:' "$OUTPUT_ROOT"/.status/*.status >&2
  exit 1
fi
touch "$OUTPUT_ROOT/ALL_DONE"
