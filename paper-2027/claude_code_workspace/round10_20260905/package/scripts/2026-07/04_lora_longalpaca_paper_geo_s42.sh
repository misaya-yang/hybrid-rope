#!/usr/bin/env bash
# Shared paper-lineage seed-42 driver for the recovered LongAlpaca-12k bytes.
# Geo is the default. The EVQ wrapper may change only EVQ_PAPER_ROPE_METHOD.
set -Eeuo pipefail

PHASE="${1:-}"
case "$PHASE" in
  preflight|baseline|train|eval) ;;
  *)
    echo "usage: $0 {preflight|baseline|train|eval}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${EVQ_LORA_PYTHON:?set EVQ_LORA_PYTHON}"
MODEL="${EVQ_LORA_MODEL:?set EVQ_LORA_MODEL}"
RAW="${EVQ_PAPER_LONGALPACA_JSON:?set EVQ_PAPER_LONGALPACA_JSON}"
RAW_SHA256="${EVQ_PAPER_LONGALPACA_SHA256:?set EVQ_PAPER_LONGALPACA_SHA256}"
ROOT="${EVQ_PAPER_LONGALPACA_ROOT:?set EVQ_PAPER_LONGALPACA_ROOT}"
ROPE_METHOD="${EVQ_PAPER_ROPE_METHOD:-native_geo}"
case "$ROPE_METHOD" in
  native_geo)
    ARM_LABEL="geo_longalpaca_s42"
    UNLOCK_VALUE="${EVQ_PAPER_UNLOCK_GEO42:-NO}"
    UNLOCK_HINT="EVQ_PAPER_UNLOCK_GEO42=YES"
    ;;
  evq_cosh)
    ARM_LABEL="evq_longalpaca_tau1414_s42"
    UNLOCK_VALUE="${EVQ_PAPER_UNLOCK_EVQ42:-NO}"
    UNLOCK_HINT="EVQ_PAPER_UNLOCK_EVQ42=YES"
    case "$PHASE" in
      baseline|eval)
        echo "EVQ uses the independent 2026 temporal-holdout evaluator; this driver only supports preflight/train" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "EVQ_PAPER_ROPE_METHOD must be native_geo or evq_cosh" >&2
    exit 2
    ;;
esac
DATA_MANIFEST="$ROOT/data/longalpaca/manifest.json"
DATA_MANIFEST_SHA256="${EVQ_PAPER_LONGALPACA_MANIFEST_SHA256:?set EVQ_PAPER_LONGALPACA_MANIFEST_SHA256}"
MODEL_MANIFEST="${EVQ_PAPER_MODEL_MANIFEST:?set EVQ_PAPER_MODEL_MANIFEST}"
MODEL_MANIFEST_SHA256="${EVQ_PAPER_MODEL_MANIFEST_SHA256:?set EVQ_PAPER_MODEL_MANIFEST_SHA256}"
EVAL_MANIFEST="${EVQ_PAPER_EVAL_MANIFEST:?set EVQ_PAPER_EVAL_MANIFEST}"
EVAL_MANIFEST_SHA256="${EVQ_PAPER_EVAL_MANIFEST_SHA256:?set EVQ_PAPER_EVAL_MANIFEST_SHA256}"
CHECKPOINT="$ROOT/checkpoints/$ARM_LABEL"
RESULTS="$ROOT/results"
LOGS="$ROOT/logs"
TELEMETRY="$ROOT/telemetry"
PREFLIGHT_DIR="$ROOT/preflight"
MODEL_HASH_RECEIPT="$PREFLIGHT_DIR/model_hash_verified.json"
GPU_LOCK_FILE="${EVQ_GLOBAL_GPU_LOCK_FILE:-/tmp/evq-lora-single-gpu.lock}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/compile_cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/compile_cache/triton}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1

mkdir -p "$RESULTS" "$LOGS" "$TELEMETRY" "$PREFLIGHT_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

TRAIN_ARGS=(
  --model_name "$MODEL"
  --output_dir "$CHECKPOINT"
  --rope_method "$ROPE_METHOD"
  --tau 1.414
  --lora_r 64 --lora_alpha 128 --lora_dropout 0.05
  --lora_targets q_proj,k_proj,v_proj,o_proj
  --max_steps 300 --per_device_batch_size 2 --gradient_accumulation_steps 4
  --learning_rate 1e-4 --warmup_steps 60 --weight_decay 0.01 --max_grad_norm 1.0
  --max_seq_len 8192 --max_samples 8000 --save_steps 100
  --seed 42 --bf16 --no_4bit --compile --compile_mode default
  --strict_legacy_protocol
  --prepared_data_manifest "$DATA_MANIFEST"
  --model_manifest "$MODEL_MANIFEST"
  --resume_from_checkpoint auto
)

require_file() {
  test -s "$1" || { echo "required file missing: $1" >&2; exit 1; }
}

preflight() {
  command -v "$PYTHON" >/dev/null
  case "${TORCHDYNAMO_DISABLE:-0}" in
    1|true|TRUE|yes|YES) echo "TORCHDYNAMO_DISABLE would bypass torch.compile" >&2; exit 1 ;;
  esac
  case "${TORCH_COMPILE_DISABLE:-0}" in
    1|true|TRUE|yes|YES) echo "TORCH_COMPILE_DISABLE would bypass torch.compile" >&2; exit 1 ;;
  esac
  require_file "$MODEL/config.json"
  require_file "$RAW"
  require_file "$DATA_MANIFEST"
  require_file "$MODEL_MANIFEST"
  require_file "$EVAL_MANIFEST"
  printf '%s  %s\n' "$RAW_SHA256" "$RAW" | sha256sum -c -
  printf '%s  %s\n' "$DATA_MANIFEST_SHA256" "$DATA_MANIFEST" | sha256sum -c -
  printf '%s  %s\n' "$MODEL_MANIFEST_SHA256" "$MODEL_MANIFEST" | sha256sum -c -
  printf '%s  %s\n' "$EVAL_MANIFEST_SHA256" "$EVAL_MANIFEST" | sha256sum -c -
  "$PYTHON" - "$DATA_MANIFEST" "$EVAL_MANIFEST" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path
from experiments.lora_evq_v2.legacy_lora_protocol import (
    sha256_file,
    validate_legacy_runtime_packages,
    validate_paper_longalpaca_receipt,
)
from experiments.lora_evq_v2.eval_legacy_lora_matched import _load_eval_manifest
from experiments.lora_evq_v2.train_evq_lora import _load_strict_legacy_data
import torch._dynamo

if torch._dynamo.config.suppress_errors:
    raise SystemExit("torch._dynamo.config.suppress_errors must be false")

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
validate_paper_longalpaca_receipt(manifest.get("source", {}))
normalized = manifest.get("normalized_source", {})
normalized_path = manifest_path.parent / str(normalized.get("name", ""))
if not normalized_path.is_file() or sha256_file(normalized_path) != normalized.get("sha256"):
    raise SystemExit("normalized LongAlpaca JSONL hash mismatch")
_load_strict_legacy_data(manifest_path)
_load_eval_manifest(Path(sys.argv[2]))
packages = {
    name: importlib.metadata.version(name)
    for name in ("torch", "transformers", "peft", "accelerate", "datasets", "triton")
}
validate_legacy_runtime_packages(packages)
print(json.dumps({
    "status": "valid",
    "source": manifest["source"],
    "statistics": manifest["statistics"],
    "manifest_sha256": sha256_file(manifest_path),
    "runtime_packages": packages,
}, indent=2))
PY
  "$PYTHON" - "$MODEL" "$MODEL_MANIFEST" "$MODEL_HASH_RECEIPT" <<'PY'
import json
import os
import sys
from pathlib import Path
from experiments.lora_evq_v2.legacy_lora_protocol import sha256_file
from experiments.lora_evq_v2.prepare_legacy_model_manifest import validate_model_manifest

model_dir, manifest_path, receipt_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest_sha256 = sha256_file(manifest_path)
validate_model_manifest(model_dir, manifest, verify_hashes=False)
model_inventory = [
    {
        "name": record["name"],
        "device": (model_dir / record["name"]).stat().st_dev,
        "inode": (model_dir / record["name"]).stat().st_ino,
        "size": (model_dir / record["name"]).stat().st_size,
        "mtime_ns": (model_dir / record["name"]).stat().st_mtime_ns,
        "ctime_ns": (model_dir / record["name"]).stat().st_ctime_ns,
    }
    for record in manifest["files"]
]
receipt = {}
if receipt_path.is_file():
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
receipt_is_current = (
    receipt.get("status") == "full_sha256_verified"
    and receipt.get("model") == model_dir.name
    and receipt.get("model_dir") == str(model_dir.resolve())
    and receipt.get("model_manifest_sha256") == manifest_sha256
    and receipt.get("model_inventory") == model_inventory
)
if not receipt_is_current:
    validate_model_manifest(model_dir, manifest, verify_hashes=True)
    receipt = {
        "status": "full_sha256_verified",
        "model": model_dir.name,
        "model_dir": str(model_dir.resolve()),
        "model_manifest_sha256": manifest_sha256,
        "model_inventory": model_inventory,
    }
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
  "$PYTHON" -u "$REPO_ROOT/experiments/lora_evq_v2/train_evq_lora.py" \
    "${TRAIN_ARGS[@]}" --dry_run
}

gpu_preflight() {
  command -v nvidia-smi >/dev/null
  "$PYTHON" - <<'PY'
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("visible GPU does not support BF16")
properties = torch.cuda.get_device_properties(0)
if "Blackwell" not in properties.name or properties.total_memory < 90 * 1024**3:
    raise SystemExit(
        f"expected >=90 GiB RTX PRO 6000 Blackwell, found {properties.name} "
        f"with {properties.total_memory / 1024**3:.1f} GiB"
    )
if torch.cuda.get_device_capability(0) < (10, 0):
    raise SystemExit("visible GPU is not a Blackwell-class CUDA device")
free_bytes, _ = torch.cuda.mem_get_info(0)
if free_bytes < 90 * 1024**3:
    raise SystemExit(f"expected at least 90 GiB free GPU memory, found {free_bytes / 1024**3:.1f} GiB")
print(json.dumps({
    "name": properties.name,
    "memory_gib": properties.total_memory / 1024**3,
    "free_memory_gib": free_bytes / 1024**3,
    "capability": torch.cuda.get_device_capability(0),
    "bf16": True,
}, indent=2))
PY
  local available_kib
  available_kib="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  (( available_kib >= 10 * 1024 * 1024 )) || {
    echo "less than 10 GiB free under $ROOT" >&2
    exit 1
  }
}

record_gpu() {
  local label="$1"
  local invocation_id
  invocation_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu \
    --format=csv,noheader > "$TELEMETRY/${label}_preflight_${invocation_id}.csv"
}

eval_variant() {
  local variant="$1" adapter="${2:-}" args=()
  if [[ -n "$adapter" ]]; then
    args+=(--adapter_dir "$adapter" --training_data_manifest "$DATA_MANIFEST")
  fi
  if "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/eval_legacy_lora_matched.py" \
      --model_name "$MODEL" --model_manifest "$MODEL_MANIFEST" \
      --eval_manifest "$EVAL_MANIFEST" --variant "$variant" --output_dir "$RESULTS" \
      "${args[@]}" --validate_only >/dev/null 2>&1; then
    echo "[skip] validated evaluation: $variant"
    return
  fi
  command -v flock >/dev/null
  exec 8>"$GPU_LOCK_FILE"
  flock -n 8 || { echo "the host GPU is already leased by another experiment" >&2; exit 1; }
  gpu_preflight
  record_gpu "eval_$variant"
  "$PYTHON" -u "$REPO_ROOT/experiments/lora_evq_v2/eval_legacy_lora_matched.py" \
    --model_name "$MODEL" \
    --model_manifest "$MODEL_MANIFEST" \
    --eval_manifest "$EVAL_MANIFEST" \
    --variant "$variant" \
    --output_dir "$RESULTS" \
    "${args[@]}" 2>&1 | tee -a "$LOGS/eval_$variant.log"
}

train_arm() {
  [[ "$UNLOCK_VALUE" == YES ]] || {
    echo "$ARM_LABEL is cost-gated; set $UNLOCK_HINT explicitly" >&2
    exit 1
  }
  preflight
  command -v flock >/dev/null
  exec 9>"$GPU_LOCK_FILE"
  flock -n 9 || { echo "the host GPU is already leased by another experiment" >&2; exit 1; }
  local manifest_sha256 telemetry_pid="" training_pid="" invocation_id telemetry_path
  manifest_sha256="$(sha256sum "$DATA_MANIFEST" | awk '{print $1}')"
  if [[ -f "$CHECKPOINT/adapter_model.safetensors" ]]; then
    "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/validate_legacy_lora_artifact.py" \
      --adapter_dir "$CHECKPOINT" --expected_method "$ROPE_METHOD" --expected_seed 42 \
      --expected_data_manifest_sha256 "$manifest_sha256"
    echo "[skip] validated completed adapter: $ARM_LABEL"
    return
  fi
  gpu_preflight
  record_gpu "$ARM_LABEL"
  invocation_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  telemetry_path="$TELEMETRY/${ARM_LABEL}_${invocation_id}.csv"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv,noheader --loop-ms=5000 > "$telemetry_path" &
  telemetry_pid=$!
  trap 'if [[ -n "${training_pid:-}" ]]; then kill "$training_pid" 2>/dev/null || true; fi; if [[ -n "${telemetry_pid:-}" ]]; then kill "$telemetry_pid" 2>/dev/null || true; fi' RETURN
  for _ in 1 2 3 4 5; do
    test -s "$telemetry_path" && break
    kill -0 "$telemetry_pid" 2>/dev/null || {
      echo "telemetry monitor failed before its first sample" >&2
      exit 1
    }
    sleep 1
  done
  test -s "$telemetry_path" || {
    echo "telemetry monitor produced no samples" >&2
    exit 1
  }
  "$PYTHON" -u "$REPO_ROOT/experiments/lora_evq_v2/train_evq_lora.py" \
    "${TRAIN_ARGS[@]}" > >(tee -a "$LOGS/train_${ARM_LABEL}.log") 2>&1 &
  training_pid=$!
  while kill -0 "$training_pid" 2>/dev/null; do
    if ! kill -0 "$telemetry_pid" 2>/dev/null; then
      kill "$training_pid" 2>/dev/null || true
      wait "$training_pid" 2>/dev/null || true
      training_pid=""
      echo "telemetry monitor stopped; training was terminated" >&2
      exit 1
    fi
    sleep 5
  done
  if ! wait "$training_pid"; then
    training_pid=""
    echo "$ARM_LABEL training failed" >&2
    exit 1
  fi
  training_pid=""
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=""
  test -s "$telemetry_path" || {
    echo "telemetry monitor produced no samples" >&2
    exit 1
  }
  "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/validate_legacy_lora_artifact.py" \
    --adapter_dir "$CHECKPOINT" --expected_method "$ROPE_METHOD" --expected_seed 42 \
    --expected_data_manifest_sha256 "$manifest_sha256"
  require_file "$CHECKPOINT/checkpoint-300/adapter_model.safetensors"
  rm -rf -- "$CHECKPOINT/checkpoint-100" "$CHECKPOINT/checkpoint-200"
}

case "$PHASE" in
  preflight)
    preflight
    ;;
  baseline)
    preflight
    eval_variant base_geo_longalpaca
    ;;
  train)
    train_arm
    ;;
  eval)
    preflight
    require_file "$CHECKPOINT/adapter_model.safetensors"
    eval_variant geo_longalpaca_s42 "$CHECKPOINT"
    ;;
esac
