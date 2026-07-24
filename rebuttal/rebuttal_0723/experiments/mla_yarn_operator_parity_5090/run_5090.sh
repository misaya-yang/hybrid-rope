#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(git rev-parse --show-toplevel)}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
PACKAGE="$ROOT/rebuttal/rebuttal_0723/mla_yarn_operator_parity_5090"
BASE_PACKAGE="$ROOT/rebuttal/rebuttal_0723/mla_scarcity_5090"
RUN_ROOT="${RUN_ROOT:-$(dirname "$ROOT")/mla_yarn_operator_parity_5090}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$(dirname "$ROOT")/mla_scarcity_5090/data/data_manifest.json}"
DATA_DIR="${DATA_DIR:-$RUN_ROOT/data}"
DATA_MANIFEST="${DATA_MANIFEST:-$DATA_DIR/data_manifest.json}"
WORK_DIR="${WORK_DIR:-$RUN_ROOT/work}"
REPORT_PATH="${REPORT_PATH:-$WORK_DIR/MLA_YARN_OPERATOR_PARITY_REPORT.md}"
COMPILE_MODE="${COMPILE_MODE:-default}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-300}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORK_DIR/torchinductor_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$DATA_DIR" "$WORK_DIR" "$TORCHINDUCTOR_CACHE_DIR"

base_runner=("$PYTHON" "$BASE_PACKAGE/run_experiment.py")
parity_runner=("$PYTHON" "$PACKAGE/run_experiment.py")
parity_operators=(
  position_interpolation
  shared_index_freq_only
  mscale_only
  shared_index_full
  virtual_coordinate_full
)

json_field_is() {
  local path="$1"
  local field="$2"
  local expected="$3"
  "$PYTHON" - "$path" "$field" "$expected" <<'PY'
import json
import sys

path, field, expected = sys.argv[1:]
try:
    current = json.load(open(path))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
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
    echo "another operator-parity launcher owns $WORK_DIR/run.lock" >&2
    exit 1
  }
}

ensure_ready() {
  "${base_runner[@]}" validate-ready \
    --data-manifest "$DATA_MANIFEST" \
    --work-dir "$WORK_DIR"
  "${parity_runner[@]}" validate-ready \
    --data-manifest "$DATA_MANIFEST" \
    --work-dir "$WORK_DIR"
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
  "${base_runner[@]}" train \
    --frequency-pairs "$pairs" \
    --arm "$arm" \
    --seed "$seed" \
    --data-manifest "$DATA_MANIFEST" \
    --work-dir "$WORK_DIR" \
    --num-workers "$NUM_WORKERS" \
    --compile-mode "$COMPILE_MODE"
}

prune_100m_if_needed() {
  local pairs="$1"
  local arm="$2"
  local seed="$3"
  local receipt="$WORK_DIR/runs/k$pairs/$arm/seed$seed/cleanup_prune100.json"
  if json_field_is "$receipt" status PASS; then
    return
  fi
  "${base_runner[@]}" cleanup-checkpoints \
    --frequency-pairs "$pairs" \
    --arm "$arm" \
    --seed "$seed" \
    --work-dir "$WORK_DIR" \
    --drop-unclaimed-100m
}

eval_raw_if_missing() {
  local pairs="$1"
  local arm="$2"
  local seed="$3"
  local split="$4"
  local stage="$5"
  local result="$WORK_DIR/runs/k$pairs/$arm/seed$seed/eval_${split}_${stage}_raw.json"
  if json_field_is "$result" status PASS; then
    echo "SKIP completed raw eval K=$pairs arm=$arm seed=$seed split=$split stage=$stage"
    return
  fi
  "${base_runner[@]}" evaluate \
    --frequency-pairs "$pairs" \
    --arm "$arm" \
    --seed "$seed" \
    --stage "$stage" \
    --split "$split" \
    --operator raw \
    --data-manifest "$DATA_MANIFEST" \
    --work-dir "$WORK_DIR"
}

eval_parity_if_missing() {
  local pairs="$1"
  local arm="$2"
  local seed="$3"
  local split="$4"
  local stage="$5"
  local operator="$6"
  local result="$WORK_DIR/runs/k$pairs/$arm/seed$seed/eval_parity_${split}_${stage}_${operator}.json"
  if json_field_is "$result" status PASS; then
    echo "SKIP parity eval K=$pairs arm=$arm seed=$seed split=$split stage=$stage operator=$operator"
    return
  fi
  "${parity_runner[@]}" evaluate \
    --frequency-pairs "$pairs" \
    --arm "$arm" \
    --seed "$seed" \
    --stage "$stage" \
    --split "$split" \
    --operator "$operator" \
    --data-manifest "$DATA_MANIFEST" \
    --work-dir "$WORK_DIR"
}

eval_registered_suite() {
  local pairs="$1"
  local arm="$2"
  local seed="$3"
  local split="$4"
  local stage="$5"
  local operator
  eval_raw_if_missing "$pairs" "$arm" "$seed" "$split" "$stage"
  if [[ "$stage" == "200m" ]]; then
    eval_parity_if_missing \
      "$pairs" "$arm" "$seed" "$split" "$stage" shared_index_full
    return
  fi
  for operator in "${parity_operators[@]}"; do
    eval_parity_if_missing \
      "$pairs" "$arm" "$seed" "$split" "$stage" "$operator"
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
  "${base_runner[@]}" cleanup-checkpoints \
    --frequency-pairs "$pairs" \
    --arm "$arm" \
    --seed "$seed" \
    --work-dir "$WORK_DIR" \
    --proof-split "$split" \
    --operator raw \
    --proof-stages 200m 300m
}

ensure_report() {
  local receipt="${REPORT_PATH}.receipt.json"
  if json_field_is "$receipt" status PASS; then
    echo "SKIP completed report $REPORT_PATH"
    return
  fi
  if [[ -e "$REPORT_PATH" || -e "$receipt" ]]; then
    echo "partial/non-PASS report exists; inspect before retry: $REPORT_PATH" >&2
    exit 1
  fi
  "$PYTHON" "$PACKAGE/report.py" \
    --work-dir "$WORK_DIR" \
    --output "$REPORT_PATH"
}

run_preflight() {
  [[ -f "$SOURCE_MANIFEST" ]] || {
    echo "completed scarcity data manifest is missing: $SOURCE_MANIFEST" >&2
    exit 1
  }
  if [[ ! -f "$DATA_MANIFEST" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PYTHON" "$PACKAGE/prepare.py" \
      --source-manifest "$SOURCE_MANIFEST" \
      --output-dir "$DATA_DIR"
  fi
  if [[ ! -f "$WORK_DIR/schedule_diagnostics.json" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PYTHON" "$BASE_PACKAGE/analyze_schedules.py" \
      --output "$WORK_DIR/schedule_diagnostics.json"
  fi
  if [[ -f "$WORK_DIR/ready_receipt.json" ]]; then
    CUDA_VISIBLE_DEVICES="" "${base_runner[@]}" validate-ready \
      --data-manifest "$DATA_MANIFEST" \
      --work-dir "$WORK_DIR" \
      --require-disk
  else
    CUDA_VISIBLE_DEVICES="" "${base_runner[@]}" preflight \
      --data-manifest "$DATA_MANIFEST" \
      --work-dir "$WORK_DIR" \
      --full-hash-check \
      --prefix-hash-check \
      --verify-full-initialization
  fi
  if [[ -f "$WORK_DIR/operator_parity_ready.json" ]]; then
    CUDA_VISIBLE_DEVICES="" "${parity_runner[@]}" validate-ready \
      --data-manifest "$DATA_MANIFEST" \
      --work-dir "$WORK_DIR"
  else
    CUDA_VISIBLE_DEVICES="" "${parity_runner[@]}" preflight \
      --data-manifest "$DATA_MANIFEST" \
      --work-dir "$WORK_DIR"
  fi
  "${base_runner[@]}" audit-disk --work-dir "$WORK_DIR"
}

run_gate() {
  require_lock
  ensure_ready
  local pairs arm stage
  for pairs in 8 32; do
    local probe="$WORK_DIR/gpu_probe_k$pairs.json"
    if ! json_field_is "$probe" status PASS; then
      "${base_runner[@]}" probe-gpu \
        --frequency-pairs "$pairs" \
        --data-manifest "$DATA_MANIFEST" \
        --work-dir "$WORK_DIR" \
        --compile-mode "$COMPILE_MODE"
    fi
    for arm in native_geo evq_cosh; do
      train_if_missing "$pairs" "$arm" 42
      prune_100m_if_needed "$pairs" "$arm" 42
      for stage in 200m 300m; do
        eval_registered_suite "$pairs" "$arm" 42 selection "$stage"
      done
    done
  done
  if [[ -f "$WORK_DIR/operator_parity_gate.json" ]]; then
    "${parity_runner[@]}" validate-gate --work-dir "$WORK_DIR"
  else
    "${parity_runner[@]}" gate --work-dir "$WORK_DIR"
  fi
  if json_field_is "$WORK_DIR/operator_parity_gate.json" status STOP; then
    for pairs in 8 32; do
      for arm in native_geo evq_cosh; do
        cleanup_one_proven_run selection "$pairs" "$arm" 42
      done
    done
    ensure_report
  fi
  "${base_runner[@]}" audit-disk --work-dir "$WORK_DIR"
}

run_confirm() {
  require_lock
  ensure_ready
  "${parity_runner[@]}" validate-gate \
    --work-dir "$WORK_DIR" \
    --require-pass
  local seed pairs arm stage
  for seed in 43 88; do
    for pairs in 8 32; do
      for arm in native_geo evq_cosh; do
        train_if_missing "$pairs" "$arm" "$seed"
        prune_100m_if_needed "$pairs" "$arm" "$seed"
        for stage in 200m 300m; do
          eval_registered_suite "$pairs" "$arm" "$seed" test "$stage"
        done
        cleanup_one_proven_run test "$pairs" "$arm" "$seed"
      done
    done
  done
  for pairs in 8 32; do
    for arm in native_geo evq_cosh; do
      for stage in 200m 300m; do
        eval_registered_suite "$pairs" "$arm" 42 test "$stage"
      done
      cleanup_one_proven_run test "$pairs" "$arm" 42
    done
  done
  if [[ -f "$WORK_DIR/summary_mla_yarn_operator_parity.json" ]]; then
    json_field_is \
      "$WORK_DIR/summary_mla_yarn_operator_parity.json" status PASS
  else
    "${parity_runner[@]}" summarize --work-dir "$WORK_DIR"
  fi
  ensure_report
  "${base_runner[@]}" audit-disk --work-dir "$WORK_DIR"
}

run_monitor() {
  local rc
  while true; do
    if "$PYTHON" "$PACKAGE/status.py" \
      --work-dir "$WORK_DIR" \
      --exit-if-terminal; then
      break
    else
      rc=$?
      if [[ "$rc" -ne 1 ]]; then
        return "$rc"
      fi
    fi
    sleep "$MONITOR_INTERVAL_SECONDS"
  done
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
    "${base_runner[@]}" audit-disk --work-dir "$WORK_DIR"
    ;;
  cleanup-cache)
    "${parity_runner[@]}" cleanup-compile-cache --work-dir "$WORK_DIR"
    ;;
  status)
    "$PYTHON" "$PACKAGE/status.py" --work-dir "$WORK_DIR"
    ;;
  monitor)
    run_monitor
    ;;
  report)
    ensure_report
    ;;
  *)
    echo "usage: $0 {preflight|gate|confirm|status|monitor|report|disk|cleanup-cache}" >&2
    exit 2
    ;;
esac
