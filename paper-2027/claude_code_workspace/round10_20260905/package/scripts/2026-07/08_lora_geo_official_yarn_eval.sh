#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
case "$MODE" in
  pilot|full) ;;
  *) echo "usage: $0 {pilot|full}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${EVQ_LORA_PYTHON:?set EVQ_LORA_PYTHON to the prepared runtime interpreter}"
MODEL="${EVQ_LORA_MODEL:?set EVQ_LORA_MODEL to the local manifested model}"
MODEL_MANIFEST="${EVQ_LORA_MODEL_MANIFEST:?set EVQ_LORA_MODEL_MANIFEST}"
ADAPTER="${EVQ_GEO_LONGALPACA_ADAPTER:?set EVQ_GEO_LONGALPACA_ADAPTER}"
TRAINING_MANIFEST="${EVQ_LONGALPACA_MANIFEST:?set EVQ_LONGALPACA_MANIFEST}"
DATA_ROOT="${EVQ_CAPABILITY_DATA_ROOT:?set EVQ_CAPABILITY_DATA_ROOT to the prepared seed42 capability suite}"
OUTPUT="${EVQ_OFFICIAL_YARN_OUTPUT:-/tmp/geo_lora_s42_official_yarn_${MODE}.json}"
GPU_LOCK="${EVQ_GPU_LOCK:-/tmp/evq_lora_eval_gpu.lock}"
[[ "$(basename "$ADAPTER")" == "geo_longalpaca_s42" ]] || {
  echo "registered Geo adapter must be named geo_longalpaca_s42" >&2
  exit 1
}

for path in "$PYTHON" "$MODEL/config.json" "$MODEL_MANIFEST" "$ADAPTER/adapter_model.safetensors" \
  "$ADAPTER/custom_inv_freq.pt" "$TRAINING_MANIFEST" "$DATA_ROOT/manifest.json"; do
  test -s "$path" || { echo "required evaluation artifact missing: $path" >&2; exit 1; }
done
test ! -e "$OUTPUT" || { echo "refusing to overwrite evaluation: $OUTPUT" >&2; exit 1; }

exec 9>"$GPU_LOCK"
flock -n 9 || { echo "GPU is leased by another evaluation" >&2; exit 73; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi -L >/dev/null
exec "$PYTHON" -u "$REPO_ROOT/experiments/lora_evq_v2/eval_official_yarn_capability.py" \
  --model_name "$MODEL" \
  --model_manifest "$MODEL_MANIFEST" \
  --adapter_dir "$ADAPTER" \
  --training_data_manifest "$TRAINING_MANIFEST" \
  --data_root "$DATA_ROOT" \
  --output "$OUTPUT" \
  --substrate native_geo \
  --yarn_factors 2,4 \
  --mode "$MODE"
