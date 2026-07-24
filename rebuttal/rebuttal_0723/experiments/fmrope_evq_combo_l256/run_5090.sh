#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_repo_root() {
  local candidate="$SCRIPT_DIR"
  while [[ "$candidate" != "/" ]]; do
    if [[ -f "$candidate/AGENTS.md" && -f "$candidate/scripts/lib/rope/schedules.py" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    candidate="$(dirname "$candidate")"
  done
  return 1
}
REPO_ROOT="$(find_repo_root)" || {
  echo "Repository root not found from $SCRIPT_DIR" >&2
  exit 2
}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif [[ -x /root/miniconda3/bin/python ]]; then
    PYTHON_BIN=/root/miniconda3/bin/python
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Python interpreter not found" >&2
    exit 2
  fi
fi

# Shared no-card hosts often expose the host's full CPU count while granting
# only a small cgroup quota. Bound library thread pools unless explicitly
# overridden so preflight does not spend minutes oversubscribing the container.
CPU_THREADS="${FMR_CPU_THREADS:-8}"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
}

assert_cpu_only() {
  "$PYTHON_BIN" - <<'PY'
import torch
if torch.cuda.is_available():
    raise SystemExit("this mode must run before the paid GPU is enabled")
print("CPU-only gate: PASS")
PY
}

case "$MODE" in
  prepare)
    require_var FMR_WORK_DIR
    assert_cpu_only
    FMR_DATA_DIR="${FMR_DATA_DIR:-$FMR_WORK_DIR/data}"
    mkdir -p "$FMR_DATA_DIR"
    cd "$REPO_ROOT"
    if [[ -n "${FMR_SOURCE_MANIFEST:-}" ]]; then
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
        -m rebuttal.rebuttal_0723.experiments.fmrope_evq_combo_l256.prepare \
        --source_manifest "$FMR_SOURCE_MANIFEST" \
        --output_dir "$FMR_DATA_DIR"
    else
      FMR_DOWNLOAD_DIR="${FMR_DOWNLOAD_DIR:-$FMR_WORK_DIR/downloads}"
      PREPARE_ARGS=(
        --fresh
        --download_dir "$FMR_DOWNLOAD_DIR"
        --output_dir "$FMR_DATA_DIR"
        --endpoint "${FMR_HF_ENDPOINT:-https://hf-mirror.com}"
      )
      if [[ -n "${FMR_TOKENIZER_DIR:-}" ]]; then
        PREPARE_ARGS+=(--tokenizer_dir "$FMR_TOKENIZER_DIR")
      fi
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
        -m rebuttal.rebuttal_0723.experiments.fmrope_evq_combo_l256.prepare "${PREPARE_ARGS[@]}"
    fi
    ;;

  preflight)
    require_var FMR_WORK_DIR
    assert_cpu_only
    FMR_DATA_DIR="${FMR_DATA_DIR:-$FMR_WORK_DIR/data}"
    DATA_MANIFEST="$FMR_DATA_DIR/data_manifest.json"
    [[ -f "$DATA_MANIFEST" ]] || {
      echo "missing prepared manifest: $DATA_MANIFEST" >&2
      exit 2
    }
    mkdir -p "$FMR_WORK_DIR/logs"
    cd "$REPO_ROOT"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" tests/test_fmrope_evq_combo_l256.py
    "$PYTHON_BIN" -m py_compile \
      "$SCRIPT_DIR/protocol.py" \
      "$SCRIPT_DIR/prepare.py" \
      "$SCRIPT_DIR/run_experiment.py"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
      -m rebuttal.rebuttal_0723.experiments.fmrope_evq_combo_l256.prepare \
      --validate "$DATA_MANIFEST" \
      --full_hash_check
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
      -m rebuttal.rebuttal_0723.experiments.fmrope_evq_combo_l256.run_experiment preflight \
      --data_manifest "$DATA_MANIFEST" \
      --full_hash_check \
      --verify_full_initialization \
      | tee "$FMR_WORK_DIR/preflight.json"
    ;;

  run)
    require_var FMR_WORK_DIR
    FMR_DATA_DIR="${FMR_DATA_DIR:-$FMR_WORK_DIR/data}"
    DATA_MANIFEST="$FMR_DATA_DIR/data_manifest.json"
    PREFLIGHT="$FMR_WORK_DIR/preflight.json"
    [[ -f "$DATA_MANIFEST" ]] || {
      echo "missing prepared manifest: $DATA_MANIFEST" >&2
      exit 2
    }
    [[ -f "$PREFLIGHT" ]] || {
      echo "missing CPU preflight receipt: $PREFLIGHT" >&2
      exit 2
    }
    cd "$REPO_ROOT"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - "$PREFLIGHT" "$DATA_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

from rebuttal.rebuttal_0723.experiments.fmrope_evq_combo_l256.prepare import sha256_file
from rebuttal.rebuttal_0723.experiments.fmrope_evq_combo_l256.run_experiment import code_fingerprint

receipt = json.loads(Path(sys.argv[1]).read_text())
manifest = Path(sys.argv[2])
if receipt.get("status") != "PASS":
    raise SystemExit("preflight receipt is not PASS")
if receipt.get("code_sha256") != code_fingerprint():
    raise SystemExit("code changed after CPU preflight")
if receipt.get("data_manifest_sha256") != sha256_file(manifest):
    raise SystemExit("data manifest changed after CPU preflight")
print("frozen preflight receipt: PASS")
PY
    "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is required"
assert torch.cuda.is_bf16_supported(), "BF16 support is required"
props = torch.cuda.get_device_properties(0)
memory = getattr(props, "total_memory", getattr(props, "total_mem", 0))
assert memory >= 30 * 2**30, f"need >=30 GiB, found {memory / 2**30:.1f}"
print(f"GPU gate: {props.name}, {memory / 2**30:.1f} GiB")
PY
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$FMR_WORK_DIR/torchinductor_cache}"
    export TORCHINDUCTOR_FX_GRAPH_CACHE=1
    export TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$FMR_WORK_DIR/logs"
    COMPILE_MODE="${FMR_COMPILE_MODE:-max-autotune-no-cudagraphs}"
    NUM_WORKERS="${FMR_NUM_WORKERS:-8}"
    ARMS="${FMR_ARMS:-evq_cosh_tau4_fmrope_base256}"
    for arm in $ARMS; do
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
        -m rebuttal.rebuttal_0723.experiments.fmrope_evq_combo_l256.run_experiment train \
        --arm "$arm" \
        --data_manifest "$DATA_MANIFEST" \
        --work_dir "$FMR_WORK_DIR" \
        --num_workers "$NUM_WORKERS" \
        --compile_mode "$COMPILE_MODE" \
        2>&1 | tee "$FMR_WORK_DIR/logs/train_${arm}.log"
    done
    if [[ "${FMR_EVALUATE:-1}" == "1" ]]; then
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
        -m rebuttal.rebuttal_0723.experiments.fmrope_evq_combo_l256.run_experiment evaluate \
        --data_manifest "$DATA_MANIFEST" \
        --work_dir "$FMR_WORK_DIR" \
        --eval_batch_size "${FMR_EVAL_BATCH_SIZE:-2}" \
        2>&1 | tee "$FMR_WORK_DIR/logs/evaluate.log"
    fi
    ;;

  evaluate)
    require_var FMR_WORK_DIR
    FMR_DATA_DIR="${FMR_DATA_DIR:-$FMR_WORK_DIR/data}"
    DATA_MANIFEST="$FMR_DATA_DIR/data_manifest.json"
    cd "$REPO_ROOT"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
      -m rebuttal.rebuttal_0723.experiments.fmrope_evq_combo_l256.run_experiment evaluate \
      --data_manifest "$DATA_MANIFEST" \
      --work_dir "$FMR_WORK_DIR" \
      --eval_batch_size "${FMR_EVAL_BATCH_SIZE:-2}"
    ;;

  *)
    echo "usage: $0 {prepare|preflight|run|evaluate}" >&2
    exit 2
    ;;
esac
