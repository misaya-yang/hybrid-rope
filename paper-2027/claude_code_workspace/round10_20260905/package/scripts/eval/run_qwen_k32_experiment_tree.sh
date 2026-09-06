#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${TREE_ROOT:?set fresh TREE_ROOT}"
[[ ! -e "$TREE_ROOT" ]] || { echo "TREE_ROOT must be fresh: $TREE_ROOT" >&2; exit 2; }

RUN_ROOT="$TREE_ROOT/nll" \
  "$REPO/scripts/eval/run_qwen_k32_packed_nll_staged.sh"

if [[ -f "$TREE_ROOT/nll/failure_diagnostic_receipt.json" ]]; then
  echo "NLL primary failed; the YaRN control diagnosis is final. No QA is authorized."
  exit 0
fi
[[ -f "$TREE_ROOT/nll/final_receipt.json" ]] || { echo "NLL final receipt missing" >&2; exit 2; }

NLL_RECEIPT="$TREE_ROOT/nll/final_receipt.json" \
NLL_COST_RECEIPT="$TREE_ROOT/nll/cost_receipt.json" \
QA_ROOT="$TREE_ROOT/qa" \
  "$REPO/scripts/eval/run_qwen_k32_far_evidence_qa.sh"
