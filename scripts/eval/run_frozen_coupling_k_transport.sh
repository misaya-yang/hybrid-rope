#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${TABLE_ROOT:?set TABLE_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${EXPECTED_WEIGHT_SHA256:?set EXPECTED_WEIGHT_SHA256}"
: "${EXPECTED_DATA_MANIFEST_SHA256:?set EXPECTED_DATA_MANIFEST_SHA256}"
: "${NATIVE_LENGTH:?set NATIVE_LENGTH}"
: "${SCALE:?set SCALE}"
: "${EXPECTED_MODEL_TYPE:?set EXPECTED_MODEL_TYPE}"
: "${EXPECTED_PAIRS:?set EXPECTED_PAIRS}"

PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
RUN_WRONG_CORTH="${RUN_WRONG_CORTH:-0}"
GAIN_COEFFICIENT="${GAIN_COEFFICIENT:-0.074}"
TASKS=(niah_single_1 niah_multikey_2 niah_multikey_3 vt)
LONG_LENGTH=$((NATIVE_LENGTH * SCALE))
SCALE_TAG=$(printf '%g' "$SCALE")
MANIFEST="$TABLE_ROOT/manifest.json"

"$PYTHON_BIN" - "$CHECKPOINT/config.json" "$MANIFEST" \
  "$DATA_ROOT/manifest.json" "$EXPECTED_DATA_MANIFEST_SHA256" \
  "$EXPECTED_MODEL_TYPE" "$EXPECTED_PAIRS" <<'PY'
import hashlib, json, sys
config_path, table_path, data_path, expected_data, model_type, pairs = sys.argv[1:]
config = json.load(open(config_path))
table = json.load(open(table_path))
if hashlib.sha256(open(config_path, "rb").read()).hexdigest() != table["checkpoint"]["config_sha256"]:
    raise RuntimeError("checkpoint config hash drift")
if config["model_type"] != model_type or table["checkpoint"]["model_type"] != model_type:
    raise RuntimeError("checkpoint model type drift")
if int(table["checkpoint"]["pairs"]) != int(pairs):
    raise RuntimeError("checkpoint rotary-pair budget drift")
if config.get("rope_scaling") not in (None, {}) or float(config.get("partial_rotary_factor", 1.0)) != 1.0:
    raise RuntimeError("checkpoint is not unscaled full-head RoPE")
if config.get("sliding_window") is not None or config.get("sliding_window_pattern") is not None:
    raise RuntimeError("sliding/local attention is outside this transport protocol")
if hashlib.sha256(open(data_path, "rb").read()).hexdigest() != expected_data:
    raise RuntimeError("RULER data manifest hash drift")
PY

read -r NATIVE_HASH PHYSICAL_HASH INDEX_HASH WRONG_HASH ATTENTION_SCALE < <(
  "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
tables = p["tables"]
print(
    p["checkpoint"]["native_sha256_float32"],
    tables["dimensionless_x"]["tensor_sha256"],
    tables["normalized_raw_index"]["tensor_sha256"],
    tables.get("wrong_source_c_orth", {}).get("tensor_sha256", "NONE"),
    p["gain"]["attention_scaling"],
)
PY
)

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"

COMMON=(
  --checkpoint "$CHECKPOINT"
  --data-root "$DATA_ROOT"
  --native-context-length "$NATIVE_LENGTH"
  --expected-weight-sha256 "$EXPECTED_WEIGHT_SHA256"
  --expected-native-sha256 "$NATIVE_HASH"
  --expected-data-manifest-sha256 "$EXPECTED_DATA_MANIFEST_SHA256"
  --table-factor "$SCALE"
  --tasks "${TASKS[@]}"
  --lengths "$NATIVE_LENGTH" "$LONG_LENGTH"
  --limit-per-cell 20
)

"$PYTHON_BIN" scripts/eval/target_free_ruler_smoke.py \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --method native \
  --tasks niah_single_1 \
  --lengths "$NATIVE_LENGTH" \
  --limit-per-cell 1 \
  --native-context-length "$NATIVE_LENGTH" \
  --expected-weight-sha256 "$EXPECTED_WEIGHT_SHA256" \
  --expected-native-sha256 "$NATIVE_HASH" \
  --expected-data-manifest-sha256 "$EXPECTED_DATA_MANIFEST_SHA256" \
  --table-factor "$SCALE" \
  --output "$OUTPUT_ROOT/preflight_native"

"$PYTHON_BIN" scripts/eval/target_free_ruler_smoke.py \
  "${COMMON[@]}" \
  --method native \
  --output "$OUTPUT_ROOT/native"

run_external() {
  local label=$1
  local file=$2
  local hash=$3
  "$PYTHON_BIN" scripts/eval/target_free_ruler_smoke.py \
    "${COMMON[@]}" \
    --method external_table_static \
    --expected-active-sha256 "$hash" \
    --table "$TABLE_ROOT/${file}_s${SCALE_TAG}.npy" \
    --table-name "frozen_${label}_s${SCALE_TAG}_c${GAIN_COEFFICIENT}" \
    --table-support native_div_factor \
    --long-attention-scaling "$ATTENTION_SCALE" \
    --output "$OUTPUT_ROOT/$label"
}

run_external physical_x dimensionless_x "$PHYSICAL_HASH"
run_external normalized_index normalized_raw_index "$INDEX_HASH"

if [[ "$RUN_WRONG_CORTH" == 1 ]]; then
  [[ "$WRONG_HASH" != NONE ]]
  run_external wrong_source_c_orth wrong_source_c_orth "$WRONG_HASH"
fi
