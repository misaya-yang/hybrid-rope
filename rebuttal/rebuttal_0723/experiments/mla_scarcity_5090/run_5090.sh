#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(git rev-parse --show-toplevel)}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
PACKAGE="$ROOT/rebuttal/rebuttal_0723/mla_scarcity_5090"
RUN_ROOT="${RUN_ROOT:-$(dirname "$ROOT")/mla_scarcity_5090}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$(dirname "$ROOT")/reviewer27be_shape_base/data/data_manifest.json}"
DATA_DIR="${DATA_DIR:-$RUN_ROOT/data}"
DATA_MANIFEST="${DATA_MANIFEST:-$DATA_DIR/data_manifest.json}"
WORK_DIR="${WORK_DIR:-$RUN_ROOT/work}"
COMPILE_MODE="${COMPILE_MODE:-default}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORK_DIR/torchinductor_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$DATA_DIR" "$WORK_DIR" "$TORCHINDUCTOR_CACHE_DIR"

runner=("$PYTHON" "$PACKAGE/run_experiment.py")

json_field_is() {
  local path="$1"
  local field="$2"
  local expected="$3"
  "$PYTHON" - "$path" "$field" "$expected" <<'PY'
import json
import sys
path, field, expected = sys.argv[1:]
try:
    value = json.load(open(path))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
current = value
for part in field.split("."):
    if not isinstance(current, dict) or part not in current:
        raise SystemExit(1)
    current = current[part]
raise SystemExit(0 if str(current) == expected else 1)
PY
}

require_lock() {
  exec 9>"$WORK_DIR/run.lock"
  flock -n 9 || {
    echo "another MLA scarcity launcher owns $WORK_DIR/run.lock" >&2
    exit 1
  }
}

ensure_ready() {
  "${runner[@]}" validate-ready \
    --data-manifest "$DATA_MANIFEST" \
    --work-dir "$WORK_DIR" \
    --require-disk
}

train_if_missing() {
  local pairs="$1"
  local arm="$2"
  local seed="$3"
  local run_dir="$WORK_DIR/runs/k$pairs/$arm/seed$seed"
  local result="$run_dir/train_result.json"
  if json_field_is "$result" status PASS; then
    echo "SKIP completed train K=$pairs arm=$arm seed=$seed"
    return
  fi
  if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "partial/non-PASS run exists; inspect before retry: $run_dir" >&2
    exit 1
  fi
  "${runner[@]}" train \
    --frequency-pairs "$pairs" \
    --arm "$arm" \
    --seed "$seed" \
    --data-manifest "$DATA_MANIFEST" \
    --work-dir "$WORK_DIR" \
    --num-workers "$NUM_WORKERS" \
    --compile-mode "$COMPILE_MODE"
}

eval_if_missing() {
  local pairs="$1"
  local arm="$2"
  local seed="$3"
  local split="$4"
  local stage="$5"
  local operator="${6:-raw}"
  local result="$WORK_DIR/runs/k$pairs/$arm/seed$seed/eval_${split}_${stage}_${operator}.json"
  if json_field_is "$result" status PASS; then
    echo "SKIP completed eval K=$pairs arm=$arm seed=$seed split=$split stage=$stage operator=$operator"
    return
  fi
  "${runner[@]}" evaluate \
    --frequency-pairs "$pairs" \
    --arm "$arm" \
    --seed "$seed" \
    --stage "$stage" \
    --split "$split" \
    --operator "$operator" \
    --data-manifest "$DATA_MANIFEST" \
    --work-dir "$WORK_DIR"
}

prune_100m_if_needed() {
  local pairs="$1"
  local arm="$2"
  local seed="$3"
  local receipt="$WORK_DIR/runs/k$pairs/$arm/seed$seed/cleanup_prune100.json"
  if json_field_is "$receipt" status PASS; then
    return
  fi
  "${runner[@]}" cleanup-checkpoints \
    --frequency-pairs "$pairs" \
    --arm "$arm" \
    --seed "$seed" \
    --work-dir "$WORK_DIR" \
    --drop-unclaimed-100m
}

cleanup_proven_runs() {
  local split="$1"
  shift
  local seed
  for seed in "$@"; do
    local pairs arm receipt
    for pairs in 8 32; do
      for arm in native_geo range_matched_uniform evq_cosh; do
        receipt="$WORK_DIR/runs/k$pairs/$arm/seed$seed/cleanup_${split}_raw.json"
        if json_field_is "$receipt" status PASS; then
          continue
        fi
        "${runner[@]}" cleanup-checkpoints \
          --frequency-pairs "$pairs" \
          --arm "$arm" \
          --seed "$seed" \
          --work-dir "$WORK_DIR" \
          --proof-split "$split" \
          --operator raw \
          --proof-stages 200m 300m
      done
    done
  done
}

cleanup_one_proven_run() {
  local split="$1"
  local pairs="$2"
  local arm="$3"
  local seed="$4"
  local receipt="$WORK_DIR/runs/k$pairs/$arm/seed$seed/cleanup_${split}_raw.json"
  if json_field_is "$receipt" status PASS; then
    return
  fi
  "${runner[@]}" cleanup-checkpoints \
    --frequency-pairs "$pairs" \
    --arm "$arm" \
    --seed "$seed" \
    --work-dir "$WORK_DIR" \
    --proof-split "$split" \
    --operator raw \
    --proof-stages 200m 300m
}

run_preflight() {
  if [[ ! -f "$DATA_MANIFEST" ]]; then
    "$PYTHON" "$PACKAGE/prepare.py" \
      --source-manifest "$SOURCE_MANIFEST" \
      --output-dir "$DATA_DIR"
  fi
  if [[ ! -f "$WORK_DIR/schedule_diagnostics.json" ]]; then
    "$PYTHON" "$PACKAGE/analyze_schedules.py" \
      --output "$WORK_DIR/schedule_diagnostics.json"
  fi
  if [[ -f "$WORK_DIR/ready_receipt.json" ]]; then
    ensure_ready
  else
    "${runner[@]}" preflight \
      --data-manifest "$DATA_MANIFEST" \
      --work-dir "$WORK_DIR" \
      --full-hash-check \
      --prefix-hash-check \
      --verify-full-initialization
  fi
  "${runner[@]}" audit-disk --work-dir "$WORK_DIR"
}

run_gate() {
  require_lock
  ensure_ready
  local pairs arm
  for pairs in 8 32; do
    local probe="$WORK_DIR/gpu_probe_k$pairs.json"
    if ! json_field_is "$probe" status PASS; then
      "${runner[@]}" probe-gpu \
        --frequency-pairs "$pairs" \
        --data-manifest "$DATA_MANIFEST" \
        --work-dir "$WORK_DIR" \
        --compile-mode "$COMPILE_MODE"
    fi
    for arm in native_geo range_matched_uniform evq_cosh; do
      train_if_missing "$pairs" "$arm" 42
      prune_100m_if_needed "$pairs" "$arm" 42
      eval_if_missing "$pairs" "$arm" 42 selection 200m raw
      eval_if_missing "$pairs" "$arm" 42 selection 300m raw
    done
  done
  if ! [[ -f "$WORK_DIR/gate_receipt.json" ]]; then
    "${runner[@]}" gate --work-dir "$WORK_DIR"
  fi
  if json_field_is "$WORK_DIR/gate_receipt.json" status STOP; then
    cleanup_proven_runs selection 42
  fi
  "${runner[@]}" audit-disk --work-dir "$WORK_DIR"
}

run_confirm() {
  require_lock
  ensure_ready
  json_field_is "$WORK_DIR/gate_receipt.json" status PASS || {
    echo "confirmatory runs require a PASS seed-42 gate" >&2
    exit 1
  }
  local seed pairs arm
  for seed in 43 88; do
    for pairs in 8 32; do
      for arm in native_geo range_matched_uniform evq_cosh; do
        train_if_missing "$pairs" "$arm" "$seed"
        prune_100m_if_needed "$pairs" "$arm" "$seed"
        eval_if_missing "$pairs" "$arm" "$seed" test 200m raw
        eval_if_missing "$pairs" "$arm" "$seed" test 300m raw
        if [[ "$arm" != "range_matched_uniform" ]]; then
          eval_if_missing "$pairs" "$arm" "$seed" test 300m yarn_full
        fi
        cleanup_one_proven_run test "$pairs" "$arm" "$seed"
      done
    done
  done
  for pairs in 8 32; do
    for arm in native_geo range_matched_uniform evq_cosh; do
      eval_if_missing "$pairs" "$arm" 42 test 200m raw
      eval_if_missing "$pairs" "$arm" 42 test 300m raw
      if [[ "$arm" != "range_matched_uniform" ]]; then
        eval_if_missing "$pairs" "$arm" 42 test 300m yarn_full
      fi
      cleanup_one_proven_run test "$pairs" "$arm" 42
    done
  done
  if [[ ! -f "$WORK_DIR/summary_mla_scarcity.json" ]]; then
    "${runner[@]}" summarize --work-dir "$WORK_DIR"
  fi
  if [[ ! -f "$WORK_DIR/summary_mla_yarn_secondary.json" ]]; then
    "${runner[@]}" summarize-yarn --work-dir "$WORK_DIR"
  fi
  "${runner[@]}" audit-disk --work-dir "$WORK_DIR"
}

case "${1:-}" in
  preflight)
    run_preflight
    ;;
  gate)
    run_gate
    ;;
  confirm)
    run_confirm
    ;;
  disk)
    "${runner[@]}" audit-disk --work-dir "$WORK_DIR"
    ;;
  cleanup-cache)
    "${runner[@]}" cleanup-compile-cache --work-dir "$WORK_DIR"
    ;;
  *)
    echo "usage: $0 {preflight|gate|confirm|disk|cleanup-cache}" >&2
    exit 2
    ;;
esac
