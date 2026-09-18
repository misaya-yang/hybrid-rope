#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/root/autodl-tmp/hybrid-rope}
OUT_ROOT=${OUT_ROOT:-/root/autodl-tmp/today_rope_plan_20260914/tc_distance_competition_20260918}
MODEL=${MODEL:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}
TAIL_TABLE=${TAIL_TABLE:-/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_classic/tables/tailspline.json}
CONTROL_TABLE=${CONTROL_TABLE:-/root/autodl-tmp/today_rope_plan_20260914/strong_evidence/llama_s4_clean_matched_dose_c/tables/tailspline_dose_control.json}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}
BATCH_SIZE=${BATCH_SIZE:-2}

mkdir -p "$OUT_ROOT"/{assets,runs,reports,logs}
cd "$CODE_ROOT"

CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false "$PYTHON_BIN" \
  experiments/tc_distance_competition_20260918/prepare_distance_competition.py \
  --model "$MODEL" --out-root "$OUT_ROOT/assets"

printf '{}\n' > "$OUT_ROOT/minimal_eval_manifest.json"

run_arm() {
  local name=$1 table=$2 label=$3
  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false "$PYTHON_BIN" -m \
    experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "$OUT_ROOT/minimal_eval_manifest.json" \
    --model "$MODEL" --arm Native \
    --extra-panel "$OUT_ROOT/assets/inputs.jsonl" \
    --only-extra-panels --skip-lm --length-cap 32768 \
    --batch-size "$BATCH_SIZE" --unmasked-unpadded-generate \
    --static-table-json "$table" --table-label "$label" \
    --out "$OUT_ROOT/runs/$name" --execute \
    > "$OUT_ROOT/logs/$name.log" 2>&1
}

run_arm tailspline "$TAIL_TABLE" tc_dc_tailspline_s4
run_arm dose_control_c "$CONTROL_TABLE" tc_dc_equal_displacement_c_s4

"$PYTHON_BIN" -m experiments.tc_distance_competition_20260918.report_distance_competition \
  --inputs "$OUT_ROOT/assets/inputs.jsonl" \
  --tailspline "$OUT_ROOT/runs/tailspline/generations.jsonl" \
  --control "$OUT_ROOT/runs/dose_control_c/generations.jsonl" \
  --out "$OUT_ROOT/reports/distance_competition.json"

printf 'COMPLETE\n' > "$OUT_ROOT/status"

