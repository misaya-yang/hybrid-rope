#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${CHECKPOINT:?set CHECKPOINT}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${FINEWEB_SOURCE:?set FINEWEB_SOURCE}"
: "${INDEX_TABLE:?set INDEX_TABLE}"
: "${YARN_TABLE:?set YARN_TABLE}"
: "${LONGBENCH_MAIN_ZIP:?set LONGBENCH_MAIN_ZIP}"
: "${LONGBENCH_DATA_ZIP:?set LONGBENCH_DATA_ZIP}"
: "${QUESTION_PANEL_ROOT:?set QUESTION_PANEL_ROOT}"
: "${INVALID_QA_RUN_ROOT:?set INVALID_QA_RUN_ROOT}"
: "${NLL_RECEIPT:?set passing NLL_RECEIPT}"
: "${NLL_COST_RECEIPT:?set NLL_COST_RECEIPT}"
: "${QA_ROOT:?set fresh QA_ROOT}"
: "${RMB_PER_HOUR:?set RMB_PER_HOUR}"
: "${BILLING_QUANTUM_SECONDS:?set BILLING_QUANTUM_SECONDS}"

PYTHON_BIN="${PYTHON_BIN:-python}"
[[ ! -e "$QA_ROOT" ]] || { echo "QA_ROOT must be fresh: $QA_ROOT" >&2; exit 2; }
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_BIN" - "$NLL_RECEIPT" <<'PY'
import sys
from pathlib import Path
from scripts.eval.eval_qwen_k32_far_evidence_qa import load_nll_receipt
load_nll_receipt(Path(sys.argv[1]))
PY

"$PYTHON_BIN" "$REPO/scripts/data/prepare_qwen_k32_far_evidence_qa.py" \
  --checkpoint "$CHECKPOINT" \
  --packed-root "$DATA_ROOT" \
  --source-parquet "$FINEWEB_SOURCE" \
  --longbench-main-zip "$LONGBENCH_MAIN_ZIP" \
  --longbench-data-zip "$LONGBENCH_DATA_ZIP" \
  --question-panel-root "$QUESTION_PANEL_ROOT" \
  --invalid-run-root "$INVALID_QA_RUN_ROOT" \
  --output "$QA_ROOT/data"

"$PYTHON_BIN" -u "$REPO/scripts/eval/eval_qwen_k32_far_evidence_qa.py" \
  --stage canary \
  --checkpoint "$CHECKPOINT" \
  --data-root "$QA_ROOT/data" \
  --index-table "$INDEX_TABLE" \
  --yarn-table "$YARN_TABLE" \
  --nll-receipt "$NLL_RECEIPT" \
  --output "$QA_ROOT/canary"

"$PYTHON_BIN" "$REPO/scripts/analysis/estimate_qwen_k32_qa_cost.py" \
  --canary-result "$QA_ROOT/canary/results.json" \
  --nll-cost-receipt "$NLL_COST_RECEIPT" \
  --rmb-per-hour "$RMB_PER_HOUR" \
  --billing-quantum-seconds "$BILLING_QUANTUM_SECONDS" \
  --output "$QA_ROOT/cost_receipt.json"

"$PYTHON_BIN" -u "$REPO/scripts/eval/eval_qwen_k32_far_evidence_qa.py" \
  --stage full \
  --checkpoint "$CHECKPOINT" \
  --data-root "$QA_ROOT/data" \
  --index-table "$INDEX_TABLE" \
  --yarn-table "$YARN_TABLE" \
  --nll-receipt "$NLL_RECEIPT" \
  --output "$QA_ROOT/full"

"$PYTHON_BIN" "$REPO/scripts/analysis/summarize_qwen_k32_far_evidence_qa.py" \
  --root "$QA_ROOT/full" \
  --output "$QA_ROOT/final_receipt.json"
