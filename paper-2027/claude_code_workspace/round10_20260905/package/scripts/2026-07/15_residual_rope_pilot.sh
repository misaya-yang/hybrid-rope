#!/usr/bin/env bash
set -euo pipefail

: "${BUNDLE:?set BUNDLE to the prepared evaluation bundle}"
PY="$BUNDLE/runtime/bin/python"
CODE="$BUNDLE/code"
OUT="$BUNDLE/results/residual_rope_pilot_s42_20260715_v6"
RUNNER="$CODE/experiments/lora_evq_v2/run_residual_rope_pilot.py"

test -x "$PY"
test -d "$BUNDLE/Meta-Llama-3-8B-Instruct"
test -f "$BUNDLE/data/passkey_v1/manifest.json"
test -f "$BUNDLE/data/longbench_qa16k/manifest.json"
test -f "$RUNNER"
test ! -e "$OUT"
"$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)'

cd "$CODE"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" "$RUNNER" \
  --model "$BUNDLE/Meta-Llama-3-8B-Instruct" \
  --passkey-root "$BUNDLE/data/passkey_v1" \
  --qa-root "$BUNDLE/data/longbench_qa16k" \
  --output-dir "$OUT" \
  --branch-dim 8 \
  --steps 10 \
  --train-mode passkey_16k \
  --train-length 16384 \
  --active-last-layers 4 \
  --seed 42

test -s "$OUT/summary.json"
"$PY" -m json.tool "$OUT/summary.json"
