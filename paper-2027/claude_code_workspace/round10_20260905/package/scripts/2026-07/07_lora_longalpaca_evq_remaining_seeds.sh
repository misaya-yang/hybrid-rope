#!/usr/bin/env bash
# Train only the two remaining EVQ+LoRA seeds. The completed Geo+LoRA seed 42
# is reused as a fixed temporal-evaluation reference and is never retrained.
set -Eeuo pipefail

PHASE="${1:-}"
REQUESTED_SEED="${2:-}"
case "$PHASE" in
  preflight|train|eval) ;;
  run-remaining) test -z "$REQUESTED_SEED" || { echo "run-remaining takes no seed" >&2; exit 2; } ;;
  *)
    echo "usage: $0 {preflight|train|eval} {43|44} | run-remaining" >&2
    exit 2
    ;;
esac

validate_seed() {
  local seed="$1"
  case "$seed" in
    43|44) ;;
    *) echo "seed must be 43 or 44" >&2; exit 2 ;;
  esac
}
if [[ "$PHASE" != run-remaining ]]; then
  validate_seed "$REQUESTED_SEED"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/scripts/2026-07"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${EVQ_LORA_PYTHON:?set EVQ_LORA_PYTHON to the locked 20260711 runtime}"
MODEL="${EVQ_LORA_MODEL:?set EVQ_LORA_MODEL}"
ROOT="${EVQ_PAPER_LONGALPACA_ROOT:?set EVQ_PAPER_LONGALPACA_ROOT}"
DATA_MANIFEST="$ROOT/data/longalpaca/manifest.json"
DATA_MANIFEST_SHA256="${EVQ_PAPER_LONGALPACA_MANIFEST_SHA256:?set EVQ_PAPER_LONGALPACA_MANIFEST_SHA256}"
MODEL_MANIFEST="${EVQ_PAPER_MODEL_MANIFEST:?set EVQ_PAPER_MODEL_MANIFEST}"
MODEL_MANIFEST_SHA256="${EVQ_PAPER_MODEL_MANIFEST_SHA256:?set EVQ_PAPER_MODEL_MANIFEST_SHA256}"
DATASET_ROOT="${EVQ_TEMPORAL_HOLDOUT_ROOT:?set EVQ_TEMPORAL_HOLDOUT_ROOT}"
GPU_LOCK_FILE="${EVQ_GLOBAL_GPU_LOCK_FILE:-/tmp/evq-lora-single-gpu.lock}"
RESULTS="${EVQ_LONGALPACA_MULTISEED_RESULTS:-$ROOT/results}"
LOGS="$ROOT/logs"
TELEMETRY="$ROOT/telemetry"
PREFLIGHT_DIR="$ROOT/preflight"
MODEL_HASH_RECEIPT="$PREFLIGHT_DIR/model_hash_verified.json"
GEO_REFERENCE="$ROOT/checkpoints/geo_longalpaca_s42"

# Cache directories live outside individual adapter outputs so seeds 43 and 44
# can reuse the first compiled graph instead of paying the compile cost twice.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/compile_cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/compile_cache/triton}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p \
  "$RESULTS" "$LOGS" "$TELEMETRY" "$PREFLIGHT_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

require_file() {
  test -s "$1" || { echo "required artifact missing: $1" >&2; exit 1; }
}

evq_label() {
  local seed="$1"
  echo "evq_longalpaca_tau1414_s${seed}"
}

common_train_args() {
  local seed="$1"
  TRAIN_ARGS=(
    --model_name "$MODEL"
    --rope_method evq_cosh
    --tau 1.414
    --lora_r 64 --lora_alpha 128 --lora_dropout 0.05
    --lora_targets q_proj,k_proj,v_proj,o_proj
    --max_steps 300 --per_device_batch_size 2 --gradient_accumulation_steps 4
    --learning_rate 1e-4 --warmup_steps 60 --weight_decay 0.01 --max_grad_norm 1.0
    --max_seq_len 8192 --max_samples 8000 --save_steps 100
    --seed "$seed" --bf16 --no_4bit --compile --compile_mode default
    --strict_legacy_protocol
    --prepared_data_manifest "$DATA_MANIFEST"
    --model_manifest "$MODEL_MANIFEST"
    --resume_from_checkpoint auto
    --packed_free_causal_sdpa
  )
}

cpu_preflight() {
  local seed="$1" label checkpoint
  validate_seed "$seed"
  test -x "$PYTHON" || { echo "locked Python is not executable: $PYTHON" >&2; exit 1; }
  case "${TORCHDYNAMO_DISABLE:-0}:${TORCH_COMPILE_DISABLE:-0}" in
    *1*|*true*|*TRUE*|*yes*|*YES*) echo "torch.compile is disabled by the environment" >&2; exit 1 ;;
  esac
  require_file "$MODEL/config.json"
  require_file "$MODEL_MANIFEST"
  require_file "$DATA_MANIFEST"
  require_file "$DATASET_ROOT/collection_manifest.json"
  require_file "$GEO_REFERENCE/adapter_model.safetensors"
  printf '%s  %s\n' "$DATA_MANIFEST_SHA256" "$DATA_MANIFEST" | sha256sum -c -
  printf '%s  %s\n' "$MODEL_MANIFEST_SHA256" "$MODEL_MANIFEST" | sha256sum -c -

  "$PYTHON" - "$MODEL" "$MODEL_MANIFEST" "$MODEL_HASH_RECEIPT" "$DATA_MANIFEST" "$DATASET_ROOT" <<'PY'
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from experiments.lora_evq_v2.legacy_lora_protocol import (
    sha256_file,
    validate_legacy_runtime_packages,
    validate_paper_longalpaca_receipt,
)
from experiments.lora_evq_v2.eval_temporal_holdout_matched import load_domain_artifacts
from experiments.lora_evq_v2.prepare_legacy_model_manifest import validate_model_manifest
from experiments.lora_evq_v2.train_evq_lora import _load_strict_legacy_data
import torch._dynamo

model_dir, model_manifest_path, receipt_path, data_manifest_path, dataset_root = map(
    Path, sys.argv[1:]
)
if torch._dynamo.config.suppress_errors:
    raise SystemExit("torch._dynamo.config.suppress_errors must remain false")
packages = {
    name: importlib.metadata.version(name)
    for name in ("torch", "transformers", "peft", "accelerate", "datasets", "triton")
}
validate_legacy_runtime_packages(packages)

data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
validate_paper_longalpaca_receipt(data_manifest.get("source", {}))
_load_strict_legacy_data(data_manifest_path)

collection_path = dataset_root / "collection_manifest.json"
collection = json.loads(collection_path.read_text(encoding="utf-8"))
if collection.get("schema") != "evq_cosh.temporal_holdout_2026.collection.v1":
    raise SystemExit("temporal collection schema mismatch")
if len(collection.get("domains", {})) != 3:
    raise SystemExit("temporal collection must contain exactly three domains")
for record in collection["domains"].values():
    manifest_path = dataset_root / str(record["manifest"])
    if sha256_file(manifest_path) != record.get("manifest_sha256"):
        raise SystemExit(f"temporal domain manifest hash mismatch: {manifest_path}")
    load_domain_artifacts(manifest_path.parent)

model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
manifest_sha256 = sha256_file(model_manifest_path)
validate_model_manifest(model_dir, model_manifest, verify_hashes=False)
inventory = [
    {
        "name": record["name"],
        "device": (model_dir / record["name"]).stat().st_dev,
        "inode": (model_dir / record["name"]).stat().st_ino,
        "size": (model_dir / record["name"]).stat().st_size,
        "mtime_ns": (model_dir / record["name"]).stat().st_mtime_ns,
        "ctime_ns": (model_dir / record["name"]).stat().st_ctime_ns,
    }
    for record in model_manifest["files"]
]
receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
current = (
    receipt.get("status") == "full_sha256_verified"
    and receipt.get("model_dir") == str(model_dir.resolve())
    and receipt.get("model_manifest_sha256") == manifest_sha256
    and receipt.get("model_inventory") == inventory
)
if not current:
    validate_model_manifest(model_dir, model_manifest, verify_hashes=True)
    receipt = {
        "status": "full_sha256_verified",
        "model": model_dir.name,
        "model_dir": str(model_dir.resolve()),
        "model_manifest_sha256": manifest_sha256,
        "model_inventory": inventory,
    }
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
print(json.dumps({"status": "valid", "runtime_packages": packages}, indent=2))
PY

  common_train_args "$seed"
  label="$(evq_label "$seed")"
  checkpoint="$ROOT/checkpoints/$label"
  "$PYTHON" -u "$REPO_ROOT/experiments/lora_evq_v2/train_evq_lora.py" \
    "${TRAIN_ARGS[@]}" --output_dir "$checkpoint" --dry_run
}

gpu_preflight() {
  command -v nvidia-smi >/dev/null
  "$PYTHON" - <<'PY'
import json
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from experiments.lora_evq_v2.train_evq_lora import packed_free_causal_sdpa_forward

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one CUDA GPU must be visible")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("visible GPU does not support BF16")
properties = torch.cuda.get_device_properties(0)
free_bytes, _ = torch.cuda.mem_get_info(0)
if properties.total_memory < 90 * 1024**3 or free_bytes < 80 * 1024**3:
    raise SystemExit(
        f"insufficient RTX PRO 6000 memory: total={properties.total_memory/1024**3:.1f} GiB, "
        f"free={free_bytes/1024**3:.1f} GiB"
    )
if torch.cuda.get_device_capability(0) < (10, 0):
    raise SystemExit("the visible GPU is not Blackwell-class")

module = type("Attention", (), {"num_key_value_groups": 4})()
q = torch.randn(1, 32, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, 8, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, 8, 256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output, _ = packed_free_causal_sdpa_forward(module, q, k, v, None)
    output.float().square().mean().backward()
torch.cuda.synchronize()
if not all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (q, k, v)):
    raise SystemExit("Flash/GQA forward-backward smoke produced invalid gradients")
print(json.dumps({
    "status": "flash_gqa_ready",
    "device": properties.name,
    "capability": torch.cuda.get_device_capability(0),
    "free_gib": free_bytes / 1024**3,
}, indent=2))
PY
}

validate_adapter() {
  local method="$1" seed="$2" checkpoint="$3" manifest_sha256
  manifest_sha256="$(sha256sum "$DATA_MANIFEST" | awk '{print $1}')"
  "$PYTHON" "$REPO_ROOT/experiments/lora_evq_v2/validate_legacy_lora_artifact.py" \
    --adapter_dir "$checkpoint" \
    --expected_method "$method" \
    --expected_seed "$seed" \
    --expected_data_manifest_sha256 "$manifest_sha256"
}

train_evq() {
  local seed="$1" label checkpoint telemetry_path telemetry_pid status
  cpu_preflight "$seed"
  validate_adapter native_geo 42 "$GEO_REFERENCE"
  label="$(evq_label "$seed")"
  checkpoint="$ROOT/checkpoints/$label"
  if [[ -s "$checkpoint/adapter_model.safetensors" ]]; then
    validate_adapter evq_cosh "$seed" "$checkpoint"
    echo "[skip] validated completed adapter: $label"
    return
  fi
  command -v flock >/dev/null
  exec 9>"$GPU_LOCK_FILE"
  flock -n 9 || { echo "the GPU is already leased by another process" >&2; exit 1; }
  gpu_preflight
  common_train_args "$seed"
  telemetry_path="$TELEMETRY/${label}_$(date -u +%Y%m%dT%H%M%SZ)_$$.csv"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv,noheader --loop-ms=5000 > "$telemetry_path" &
  telemetry_pid=$!
  trap 'kill "${telemetry_pid:-}" 2>/dev/null || true' EXIT INT TERM
  for _ in 1 2 3 4 5; do
    test -s "$telemetry_path" && break
    kill -0 "$telemetry_pid" 2>/dev/null || {
      echo "GPU telemetry failed before training launch" >&2
      exit 1
    }
    sleep 1
  done
  test -s "$telemetry_path" || { echo "GPU telemetry produced no sample" >&2; exit 1; }
  set +e
  "$PYTHON" -u "$REPO_ROOT/experiments/lora_evq_v2/train_evq_lora.py" \
    "${TRAIN_ARGS[@]}" --output_dir "$checkpoint" \
    2>&1 | tee -a "$LOGS/train_${label}.log"
  status=${PIPESTATUS[0]}
  set -e
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  trap - EXIT INT TERM
  (( status == 0 )) || { echo "$label training failed" >&2; exit "$status"; }
  validate_adapter evq_cosh "$seed" "$checkpoint"
  require_file "$checkpoint/checkpoint-300/adapter_model.safetensors"
  rm -rf -- "$checkpoint/checkpoint-100" "$checkpoint/checkpoint-200"
  flock -u 9
}

eval_evq() {
  local seed="$1" evq_adapter output
  validate_seed "$seed"
  evq_adapter="$ROOT/checkpoints/evq_longalpaca_tau1414_s${seed}"
  validate_adapter native_geo 42 "$GEO_REFERENCE"
  validate_adapter evq_cosh "$seed" "$evq_adapter"
  output="$RESULTS/temporal_geo42_evq${seed}_2026.json"
  if [[ -s "$output" ]]; then
    "$PYTHON" - \
      "$output" "$seed" "$DATA_MANIFEST" "$MODEL_MANIFEST" \
      "$DATASET_ROOT/collection_manifest.json" "$GEO_REFERENCE" "$evq_adapter" <<'PY'
import json
import sys
from pathlib import Path
from experiments.lora_evq_v2.eval_temporal_holdout_three_arm import temporal_arm_contract
from experiments.lora_evq_v2.legacy_lora_protocol import sha256_file

path = Path(sys.argv[1])
seed = int(sys.argv[2])
data_manifest, model_manifest, collection_manifest, geo_adapter, evq_adapter = map(
    Path, sys.argv[3:]
)
record = json.loads(path.read_text(encoding="utf-8"))
expected = temporal_arm_contract(geo_seed=42, evq_seed=seed)
if record.get("schema") != "evq_cosh.temporal_holdout_2026.three_arm_eval.v1":
    raise SystemExit("existing temporal result has the wrong schema")
if record.get("arm_contract") != expected:
    raise SystemExit("existing temporal result has the wrong adapter seeds")
expected_hashes = {
    "training_data_manifest_sha256": sha256_file(data_manifest),
    "model_manifest_sha256": sha256_file(model_manifest),
    "collection_manifest_sha256": sha256_file(collection_manifest),
}
for field, digest in expected_hashes.items():
    if record.get(field) != digest:
        raise SystemExit(f"existing temporal result has the wrong {field}")
adapter_hashes = record.get("adapter_sha256", {})
if adapter_hashes.get("geo_lora") != sha256_file(geo_adapter / "adapter_model.safetensors"):
    raise SystemExit("existing temporal result has the wrong Geo adapter")
if adapter_hashes.get("evq_lora") != sha256_file(evq_adapter / "adapter_model.safetensors"):
    raise SystemExit("existing temporal result has the wrong EVQ adapter")
print(f"[skip] validated completed temporal result: {path}")
PY
    return
  fi
  EVQ_GEO_LONGALPACA_ADAPTER="$GEO_REFERENCE" \
  EVQ_EVQ_LONGALPACA_ADAPTER="$evq_adapter" \
  EVQ_PAPER_LONGALPACA_MANIFEST="$DATA_MANIFEST" \
  EVQ_TEMPORAL_THREE_ARM_OUTPUT="$output" \
  EVQ_GEO_LONGALPACA_EXPECTED_SEED=42 \
  EVQ_EVQ_LONGALPACA_EXPECTED_SEED="$seed" \
    "$SCRIPT_DIR/06_lora_temporal_three_arm_eval.sh"
}

case "$PHASE" in
  preflight) cpu_preflight "$REQUESTED_SEED" ;;
  train) train_evq "$REQUESTED_SEED" ;;
  eval) eval_evq "$REQUESTED_SEED" ;;
  run-remaining)
    for seed in 43 44; do
      train_evq "$seed"
      eval_evq "$seed"
    done
    ;;
esac
