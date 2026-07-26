#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/llama8b_ruler_mix_20260726}"
CODE_ROOT="${CODE_ROOT:-$ROOT/code/hybrid-rope}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/evq_5090_eval_bundle/runtime/bin/python}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}"
MODEL_MANIFEST="${MODEL_MANIFEST:-/root/autodl-tmp/evq_5090_eval_bundle/manifests/model_manifest.json}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/data/temporal_holdout_2026_v1}"
NATIVE_OUTPUT="${NATIVE_OUTPUT:-$ROOT/runs/native_ruler_mix_s20420726}"
OUTPUT="${OUTPUT:-$ROOT/runs/temporal_ppl_native_base_vs_native_ruler_mix.json}"
QUEUE_PID_FILE="$ROOT/pids/queue_native_matched.pid"
MODULE="rebuttal.rebuttal_0723.experiments.llama8b_ruler_mix.evaluate_temporal_ppl"

export PYTHONPATH="$CODE_ROOT:/root/autodl-tmp/evq_5090_eval_bundle/code"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

queue_alive() {
  local queue_pid
  queue_pid="$(cat "$QUEUE_PID_FILE")"
  kill -0 "$queue_pid" 2>/dev/null
}

while [[ ! -f "$NATIVE_OUTPUT/result.json" ]]; do
  queue_alive
  sleep 30
done

for shard in 1 2 3 4; do
  pid_file="$ROOT/pids/native_matched_ruler_shard_${shard}.pid"
  while [[ ! -f "$pid_file" ]]; do
    queue_alive
    sleep 15
  done
  pid="$(cat "$pid_file")"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
  test -f "$ROOT/runs/native_matched_ruler_shards/shard_${shard}/result.json"
done

test ! -e "$OUTPUT"
test ! -e "$OUTPUT.incomplete"
"$PYTHON_BIN" -m "$MODULE" \
  --checkpoint "$CHECKPOINT" \
  --model-manifest "$MODEL_MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --adapter "$NATIVE_OUTPUT" \
  --method native_geo \
  --output "$OUTPUT"
