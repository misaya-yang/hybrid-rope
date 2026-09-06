#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${CHECKPOINT:?set CHECKPOINT}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${INDEX_TABLE:?set INDEX_TABLE}"
: "${YARN_TABLE:?set YARN_TABLE}"
: "${RUN_ROOT:?set fresh RUN_ROOT}"
: "${RMB_PER_HOUR:?set RMB_PER_HOUR}"
: "${BILLING_QUANTUM_SECONDS:?set BILLING_QUANTUM_SECONDS}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -e "$RUN_ROOT" ]]; then
  echo "RUN_ROOT must be fresh: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"

export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_BIN" "$REPO/scripts/data/audit_qwen_k32_packed_natural_targets.py" \
  --data-root "$DATA_ROOT" \
  --output "$RUN_ROOT/boundary_receipt.json"

"$PYTHON_BIN" -u "$REPO/scripts/eval/eval_qwen_k32_natural_nll.py" \
  --stage canary \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --index-table "$INDEX_TABLE" \
  --yarn-table "$YARN_TABLE" \
  --boundary-receipt "$RUN_ROOT/boundary_receipt.json" \
  --output "$RUN_ROOT/canary"

"$PYTHON_BIN" "$REPO/scripts/analysis/estimate_qwen_k32_staged_nll_cost.py" \
  --canary-result "$RUN_ROOT/canary/results.json" \
  --rmb-per-hour "$RMB_PER_HOUR" \
  --billing-quantum-seconds "$BILLING_QUANTUM_SECONDS" \
  --budget-rmb 8 \
  --buffer-fraction .2 \
  --output "$RUN_ROOT/cost_receipt.json"

"$PYTHON_BIN" -u "$REPO/scripts/eval/eval_qwen_k32_natural_nll.py" \
  --stage primary \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --index-table "$INDEX_TABLE" \
  --yarn-table "$YARN_TABLE" \
  --boundary-receipt "$RUN_ROOT/boundary_receipt.json" \
  --output "$RUN_ROOT/primary"

"$PYTHON_BIN" "$REPO/scripts/analysis/summarize_qwen_k32_natural_nll_staged.py" \
  --stage primary \
  --primary-root "$RUN_ROOT/primary" \
  --output "$RUN_ROOT/primary_receipt.json"

if ! "$PYTHON_BIN" - "$RUN_ROOT/primary_receipt.json" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
raise SystemExit(0 if receipt.get("stage_b_authorized") is True else 3)
PY
then
  "$PYTHON_BIN" -u "$REPO/scripts/eval/eval_qwen_k32_natural_nll.py" \
    --stage control \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --index-table "$INDEX_TABLE" \
    --yarn-table "$YARN_TABLE" \
    --boundary-receipt "$RUN_ROOT/boundary_receipt.json" \
    --primary-receipt "$RUN_ROOT/primary_receipt.json" \
    --output "$RUN_ROOT/control"
  "$PYTHON_BIN" "$REPO/scripts/analysis/summarize_qwen_k32_natural_nll_staged.py" \
    --stage failure \
    --primary-root "$RUN_ROOT/primary" \
    --baseline-root "$RUN_ROOT/control" \
    --primary-receipt "$RUN_ROOT/primary_receipt.json" \
    --output "$RUN_ROOT/failure_diagnostic_receipt.json"
  echo "Primary resolver did not pass; YaRN positive-control diagnosis completed."
  exit 0
fi

"$PYTHON_BIN" -u "$REPO/scripts/eval/eval_qwen_k32_natural_nll.py" \
  --stage baseline \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --index-table "$INDEX_TABLE" \
  --yarn-table "$YARN_TABLE" \
  --boundary-receipt "$RUN_ROOT/boundary_receipt.json" \
  --primary-receipt "$RUN_ROOT/primary_receipt.json" \
  --output "$RUN_ROOT/baseline"

"$PYTHON_BIN" "$REPO/scripts/analysis/summarize_qwen_k32_natural_nll_staged.py" \
  --stage final \
  --primary-root "$RUN_ROOT/primary" \
  --baseline-root "$RUN_ROOT/baseline" \
  --primary-receipt "$RUN_ROOT/primary_receipt.json" \
  --output "$RUN_ROOT/final_receipt.json"
