#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/root/autodl-tmp/hybrid-rope}
OUT_ROOT=${OUT_ROOT:-/root/autodl-tmp/today_rope_plan_20260914/tc_distance_competition_confirm_v2_20260918}
MODEL=${MODEL:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}
TAIL_TABLE=${TAIL_TABLE:-/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic/tables/tailspline.json}
CONTROL_TABLE=${CONTROL_TABLE:-/root/autodl-tmp/today_rope_plan_20260914/strong_evidence/llama_s4_clean_matched_dose_c/tables/tailspline_dose_control.json}
STAGE1_REPORT=${STAGE1_REPORT:-/root/autodl-tmp/today_rope_plan_20260914/tc_distance_competition_20260918/reports/existing_multikey_audit_v2.json}
RECONCILIATION_REPORT=${RECONCILIATION_REPORT:-/root/autodl-tmp/today_rope_plan_20260914/tc_distance_competition_20260918/reports/stage1_official_reconciliation_v1.json}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}

ASSET_ROOT=$OUT_ROOT/assets
MANIFEST=$ASSET_ROOT/manifest_v2.json
PARITY_PANEL=$ASSET_ROOT/inputs_parity_v2.jsonl
CONFIRM_PANEL=$ASSET_ROOT/inputs_confirm_v2.jsonl

mkdir -p "$OUT_ROOT"/{logs,parity,runs,reports}
cd "$CODE_ROOT"

run_parity() {
  local name=$1 table=$2 label=$3
  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false "$PYTHON_BIN" -m \
    experiments.tc_distance_competition_20260918.run_position_gap_v2 \
    --mode parity \
    --model "$MODEL" \
    --panel "$PARITY_PANEL" --panel-kind parity \
    --manifest "$MANIFEST" \
    --stage1-report "$STAGE1_REPORT" \
    --reconciliation-report "$RECONCILIATION_REPORT" \
    --static-table-json "$table" --table-label "$label" \
    --out "$OUT_ROOT/parity/$name" \
    > "$OUT_ROOT/logs/parity_$name.log" 2>&1
}

run_arm() {
  local name=$1 table=$2 label=$3
  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false "$PYTHON_BIN" -m \
    experiments.tc_distance_competition_20260918.run_position_gap_v2 \
    --mode execute \
    --model "$MODEL" \
    --panel "$CONFIRM_PANEL" --panel-kind confirmation \
    --manifest "$MANIFEST" \
    --stage1-report "$STAGE1_REPORT" \
    --reconciliation-report "$RECONCILIATION_REPORT" \
    --static-table-json "$table" --table-label "$label" \
    --parity-receipt "$OUT_ROOT/parity/$name/parity_receipt.json" \
    --out "$OUT_ROOT/runs/$name" \
    > "$OUT_ROOT/logs/$name.log" 2>&1
}

run_parity tailspline "$TAIL_TABLE" tc_dc_confirm_v2_tailspline_s4
run_parity dose_control_c "$CONTROL_TABLE" tc_dc_confirm_v2_equal_displacement_c_s4

run_arm tailspline "$TAIL_TABLE" tc_dc_confirm_v2_tailspline_s4
run_arm dose_control_c "$CONTROL_TABLE" tc_dc_confirm_v2_equal_displacement_c_s4

CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -m \
  experiments.tc_distance_competition_20260918.report_distance_competition_v2 \
  --inputs "$CONFIRM_PANEL" \
  --tailspline "$OUT_ROOT/runs/tailspline/generations.jsonl" \
  --control "$OUT_ROOT/runs/dose_control_c/generations.jsonl" \
  --out "$OUT_ROOT/reports/distance_competition_confirm_v2.json" \
  > "$OUT_ROOT/logs/report.log" 2>&1

printf 'COMPLETE\n' > "$OUT_ROOT/status"
