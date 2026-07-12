#!/usr/bin/env bash
# Run the frozen 2026 temporal PPL/NLL comparison after EVQ-42 completes.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${EVQ_LORA_PYTHON:?set EVQ_LORA_PYTHON}"
MODEL="${EVQ_LORA_MODEL:?set EVQ_LORA_MODEL}"
MODEL_MANIFEST="${EVQ_PAPER_MODEL_MANIFEST:?set EVQ_PAPER_MODEL_MANIFEST}"
TRAINING_DATA_MANIFEST="${EVQ_PAPER_LONGALPACA_MANIFEST:?set EVQ_PAPER_LONGALPACA_MANIFEST}"
GEO_ADAPTER="${EVQ_GEO_LONGALPACA_ADAPTER:?set EVQ_GEO_LONGALPACA_ADAPTER}"
EVQ_ADAPTER="${EVQ_EVQ_LONGALPACA_ADAPTER:?set EVQ_EVQ_LONGALPACA_ADAPTER}"
DATASET_ROOT="${EVQ_TEMPORAL_HOLDOUT_ROOT:?set EVQ_TEMPORAL_HOLDOUT_ROOT}"
OUTPUT="${EVQ_TEMPORAL_THREE_ARM_OUTPUT:?set EVQ_TEMPORAL_THREE_ARM_OUTPUT}"
GPU_LOCK_FILE="${EVQ_GLOBAL_GPU_LOCK_FILE:-/tmp/evq-lora-single-gpu.lock}"

for path in \
  "$MODEL/config.json" \
  "$MODEL_MANIFEST" \
  "$TRAINING_DATA_MANIFEST" \
  "$GEO_ADAPTER/adapter_model.safetensors" \
  "$EVQ_ADAPTER/adapter_model.safetensors" \
  "$DATASET_ROOT/collection_manifest.json"; do
  test -s "$path" || { echo "required evaluation artifact missing: $path" >&2; exit 1; }
done
test ! -e "$OUTPUT" || { echo "refusing to overwrite evaluation: $OUTPUT" >&2; exit 1; }
command -v flock >/dev/null
exec 9>"$GPU_LOCK_FILE"
flock -n 9 || { echo "GPU is still leased by training or another evaluation" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT")"
"$PYTHON" -u "$REPO_ROOT/experiments/lora_evq_v2/eval_temporal_holdout_three_arm.py" \
  --model_name "$MODEL" \
  --model_manifest "$MODEL_MANIFEST" \
  --dataset_root "$DATASET_ROOT" \
  --training_data_manifest "$TRAINING_DATA_MANIFEST" \
  --geo_adapter_dir "$GEO_ADAPTER" \
  --evq_adapter_dir "$EVQ_ADAPTER" \
  --output "$OUTPUT"
