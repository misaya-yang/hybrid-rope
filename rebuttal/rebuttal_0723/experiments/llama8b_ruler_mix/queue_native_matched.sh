#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/llama8b_ruler_mix_20260726}"
CODE_ROOT="${CODE_ROOT:-$ROOT/code/hybrid-rope}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/evq_5090_eval_bundle/runtime/bin/python}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}"
SOURCES="${SOURCES:-$ROOT/data/ruler_official_train_eval_s20420726}"
NATIVE_OUTPUT="${NATIVE_OUTPUT:-$ROOT/runs/native_ruler_mix_s20420726}"
RUNNER="$CODE_ROOT/rebuttal/rebuttal_0723/experiments/llama8b_ruler_mix/run.sh"
MODULE="rebuttal.rebuttal_0723.experiments.llama8b_ruler_mix.evaluate"

export PYTHONPATH="$CODE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/cache/torchinductor_pro6000}"

for shard in 1 2 3 4; do
  pid_file="$ROOT/pids/native_base_ruler_shard_${shard}.pid"
  test -f "$pid_file"
  pid="$(cat "$pid_file")"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
  test -f "$ROOT/runs/native_base_ruler_shards/shard_${shard}/result.json"
done

test ! -e "$NATIVE_OUTPUT"
test ! -e "$NATIVE_OUTPUT.incomplete"
MICRO_BATCH_SIZE=2 bash "$RUNNER" train-native \
  >"$ROOT/logs/train_native.log" 2>&1
test -f "$NATIVE_OUTPUT/result.json"

launch_shard() {
  local shard="$1"
  shift
  local output="$ROOT/runs/native_matched_ruler_shards/shard_${shard}"
  test ! -e "$output"
  test ! -e "$output.incomplete"
  "$PYTHON_BIN" -m "$MODULE" \
    --checkpoint "$CHECKPOINT" \
    --adapter "$NATIVE_OUTPUT" \
    --sources "$SOURCES" \
    --output "$output" \
    --method native_geo \
    --limit-per-cell 20 \
    --tasks "$@" \
    >"$ROOT/logs/native_matched_ruler_shard_${shard}.log" 2>&1 &
  echo "$!" >"$ROOT/pids/native_matched_ruler_shard_${shard}.pid"
}

mkdir -p "$ROOT/runs/native_matched_ruler_shards"
launch_shard 1 cwe niah_single_1 niah_multikey_2
launch_shard 2 niah_multiquery niah_multikey_3 niah_single_2
launch_shard 3 niah_multivalue niah_multikey_1 niah_single_3
launch_shard 4 fwe qa_1 qa_2 vt
wait

for shard in 1 2 3 4; do
  test -f "$ROOT/runs/native_matched_ruler_shards/shard_${shard}/result.json"
done
printf 'NATIVE_MATCHED_TRAIN_AND_SHARDED_EVAL_COMPLETE\n'
