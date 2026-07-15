#!/usr/bin/env bash
set -euo pipefail

: "${BUNDLE:?set BUNDLE to the prepared evaluation bundle}"
PY="$BUNDLE/runtime/bin/python"
CODE="$BUNDLE/code"
MODEL="$BUNDLE/Meta-Llama-3-8B-Instruct"
MODEL_MANIFEST="$BUNDLE/manifests/model_manifest.json"
TRAINING_MANIFEST="$BUNDLE/manifests/training_manifest.json"
DATA="$BUNDLE/data/longbench_qa16k"
READY="$BUNDLE/READY_QA16K.json"
OUT="$BUNDLE/results/qa16k_three_arm_s42_20260715"
EVAL="$CODE/experiments/lora_evq_v2/eval_qa16k_three_arm.py"

test -x "$PY"
test -f "$READY"
test -f "$DATA/manifest.json"
test -f "$EVAL"
test ! -e "$OUT"
"$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)'

mkdir -p "$OUT"
cd "$CODE"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

common=(
  --model-name "$MODEL"
  --model-manifest "$MODEL_MANIFEST"
  --training-data-manifest "$TRAINING_MANIFEST"
  --data-root "$DATA"
)

"$PY" "$EVAL" run-arm \
  --arm native_lora \
  "${common[@]}" \
  --adapter-dir "$BUNDLE/adapters/geo" \
  --output "$OUT/native_lora.json" 2>&1 | tee "$OUT/native_lora.log"

"$PY" "$EVAL" run-arm \
  --arm evq_lora \
  "${common[@]}" \
  --adapter-dir "$BUNDLE/adapters/evq" \
  --output "$OUT/evq_lora.json" 2>&1 | tee "$OUT/evq_lora.log"

"$PY" "$EVAL" run-arm \
  --arm base_native \
  "${common[@]}" \
  --adapter-dir "$BUNDLE/adapters/geo" \
  --output "$OUT/base_native.json" 2>&1 | tee "$OUT/base_native.log"

"$PY" "$EVAL" summarize \
  --base-native "$OUT/base_native.json" \
  --native-lora "$OUT/native_lora.json" \
  --evq-lora "$OUT/evq_lora.json" \
  --output "$OUT/summary.json"

test -s "$OUT/summary.json"
"$PY" -m json.tool "$OUT/summary.json"
