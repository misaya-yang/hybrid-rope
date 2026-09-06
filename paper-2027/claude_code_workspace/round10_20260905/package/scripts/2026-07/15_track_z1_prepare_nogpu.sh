#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${REPO_DIR:?set REPO_DIR}"
: "${MODEL_NAME:?set MODEL_NAME}"
: "${MODEL_MANIFEST:?set MODEL_MANIFEST}"
: "${TRAINING_MANIFEST:?set TRAINING_MANIFEST}"
: "${GEO_ADAPTER:?set GEO_ADAPTER}"
: "${EVQ_ADAPTER:?set EVQ_ADAPTER}"
: "${PASSKEY_ROOT:?set PASSKEY_ROOT}"
: "${SUITE_ROOT:?set SUITE_ROOT}"
: "${RESULT_ROOT:?set RESULT_ROOT}"

PYTHON_BIN=$(command -v "$PYTHON_BIN") || {
  echo "PYTHON_BIN is not executable" >&2
  exit 1
}

entry="$REPO_DIR/experiments/lora_evq_v2/eval_sparse_conversion.py"
analysis="$REPO_DIR/scripts/analysis/readout_conversion.py"
test_file="$REPO_DIR/tests/test_readout_conversion.py"
cases_root="${CASES_ROOT:-$RESULT_ROOT/inputs/association_swap_cases}"
ready_file="${READY_FILE:-$RESULT_ROOT/track_z1.ready.env}"
preflight_root="$RESULT_ROOT/preflight"
causal_geo="$RESULT_ROOT/raw/causal_native_geo/manifest.json"
causal_evq="$RESULT_ROOT/raw/causal_evq_cosh/manifest.json"
swap_geo="$RESULT_ROOT/raw/swap_native_geo"
swap_evq="$RESULT_ROOT/raw/swap_evq_cosh"
linchpin="$RESULT_ROOT/linchpin"
gpu_complete="$RESULT_ROOT/track_z1.gpu_complete"

if "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "refusing Track Z preparation while CUDA is available; switch to no-GPU mode" >&2
  exit 1
fi

required=(
  "$PYTHON_BIN"
  "$entry"
  "$analysis"
  "$test_file"
  "$MODEL_NAME"
  "$MODEL_MANIFEST"
  "$TRAINING_MANIFEST"
  "$GEO_ADAPTER/adapter_model.safetensors"
  "$EVQ_ADAPTER/adapter_model.safetensors"
  "$PASSKEY_ROOT/passkey.jsonl"
  "$SUITE_ROOT/manifest.json"
  "$(dirname "$TRAINING_MANIFEST")/tokens.pt"
  "$(dirname "$TRAINING_MANIFEST")/offsets.pt"
  "$(dirname "$TRAINING_MANIFEST")/train_indices.pt"
  "$(dirname "$TRAINING_MANIFEST")/validation_indices.pt"
  "$causal_geo"
  "$causal_evq"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "missing no-GPU prerequisite: $path" >&2
    exit 1
  fi
done
for path in \
  "$ready_file" \
  "$ready_file.incomplete" \
  "$preflight_root/dry_run.json" \
  "$preflight_root/dry_run.json.incomplete" \
  "$cases_root" \
  "$cases_root.incomplete" \
  "$swap_geo" \
  "$swap_geo.incomplete" \
  "$swap_evq" \
  "$swap_evq.incomplete" \
  "$linchpin" \
  "$linchpin.incomplete" \
  "$gpu_complete" \
  "$gpu_complete.incomplete"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite Track Z artifact: $path" >&2
    exit 1
  fi
done

available_kib=$(df -Pk "$RESULT_ROOT" 2>/dev/null | awk 'NR == 2 {print $4}')
if [[ -z "$available_kib" || "$available_kib" -lt $((24 * 1024 * 1024)) ]]; then
  echo "Track Z requires at least 24 GiB free at RESULT_ROOT before GPU startup" >&2
  exit 1
fi

check_sha() {
  local expected=$1 path=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch: $path" >&2
    exit 1
  fi
}

check_sha 0e7efa6e83e74166a3ae5a0db6997124f881badcd7ddbdcae49cb6d211ebec9a "$GEO_ADAPTER/adapter_model.safetensors"
check_sha 8ea0423473793bb8f2ccfed46f25e67c44cd75011620542664246fa598d0c780 "$EVQ_ADAPTER/adapter_model.safetensors"
check_sha 0196fe3f3dcd932e337a7a0e91625fd12667cecbcca221020ea39428c6178210 "$MODEL_MANIFEST"
check_sha 1a610863e6602091a524e45c3ebc5907b6d0deae78851d3b0fd23ef9585e587f "$TRAINING_MANIFEST"
check_sha 21f365daed1b77e06b0a870ccb20f4965cba454c802f48dd48825c8f7ff2990d "$PASSKEY_ROOT/passkey.jsonl"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$preflight_root" "$RESULT_ROOT/inputs" "$RESULT_ROOT/logs"
"$PYTHON_BIN" -m py_compile "$entry" "$analysis" "$test_file"
"$PYTHON_BIN" -m pytest "$test_file" -q
"$PYTHON_BIN" -u "$entry" dry-run \
  --model-name "$MODEL_NAME" \
  --model-manifest "$MODEL_MANIFEST" \
  --training-data-manifest "$TRAINING_MANIFEST" \
  --geo-adapter "$GEO_ADAPTER" \
  --evq-adapter "$EVQ_ADAPTER" \
  --passkey-root "$PASSKEY_ROOT" \
  --suite-root "$SUITE_ROOT" \
  --output "$preflight_root/dry_run.json"
"$PYTHON_BIN" -u "$entry" prepare-association-swap \
  --model-name "$MODEL_NAME" \
  --training-data-manifest "$TRAINING_MANIFEST" \
  --output-dir "$cases_root"

"$PYTHON_BIN" - "$causal_geo" "$causal_evq" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

documents = []
for name, path_text in zip(("native_geo", "evq_cosh"), sys.argv[1:]):
    path = Path(path_text)
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema") != "evq_cosh.readout_conversion_trace_manifest.v1"
        or document.get("status") != "complete"
        or document.get("measurement_label") != "oracle-diagnostic"
        or document.get("substrate") != name
        or len(document.get("records", [])) != 10
    ):
        raise SystemExit(f"invalid causal manifest: {path}")
    for record in document["records"]:
        record_path = path.parent / record["file"]
        digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
        if digest != record.get("sha256"):
            raise SystemExit(f"causal record receipt mismatch: {record_path}")
    documents.append(document)
geo_keys = {(row["prompt_sha256"], row["depth_percent"]) for row in documents[0]["records"]}
evq_keys = {(row["prompt_sha256"], row["depth_percent"]) for row in documents[1]["records"]}
if geo_keys != evq_keys:
    raise SystemExit("causal Geo/EVQ records are not matched")
PY

temporary="$ready_file.incomplete"
{
  printf 'TRACK_Z_READY_VERSION=%q\n' 1
  printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
  printf 'REPO_DIR=%q\n' "$REPO_DIR"
  printf 'MODEL_NAME=%q\n' "$MODEL_NAME"
  printf 'MODEL_MANIFEST=%q\n' "$MODEL_MANIFEST"
  printf 'TRAINING_MANIFEST=%q\n' "$TRAINING_MANIFEST"
  printf 'GEO_ADAPTER=%q\n' "$GEO_ADAPTER"
  printf 'EVQ_ADAPTER=%q\n' "$EVQ_ADAPTER"
  printf 'CASES_ROOT=%q\n' "$cases_root"
  printf 'CAUSAL_GEO_MANIFEST=%q\n' "$causal_geo"
  printf 'CAUSAL_EVQ_MANIFEST=%q\n' "$causal_evq"
  printf 'SWAP_GEO_OUTPUT=%q\n' "$swap_geo"
  printf 'SWAP_EVQ_OUTPUT=%q\n' "$swap_evq"
  printf 'LINCHPIN_OUTPUT=%q\n' "$linchpin"
  printf 'GPU_COMPLETE_FILE=%q\n' "$gpu_complete"
} > "$temporary"
chmod a-w "$temporary"
mv "$temporary" "$ready_file"
printf 'READY: %s\n' "$ready_file"
