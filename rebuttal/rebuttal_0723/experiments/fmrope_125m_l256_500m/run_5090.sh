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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FMR_MODEL_TIER="${FMR_MODEL_TIER:-151m}"
case "$FMR_MODEL_TIER" in
  151m|350m) ;;
  *)
    echo "FMR_MODEL_TIER must be 151m or 350m" >&2
    exit 2
    ;;
esac
export FMR_SEED="${FMR_SEED:-42}"
case "$FMR_SEED" in
  42|137|256) ;;
  *)
    echo "FMR_SEED must be one of 42, 137, or 256" >&2
    exit 2
    ;;
esac

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
    if [[ -n "${FMR_RETARGET_MANIFEST:-}" ]]; then
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
        -m rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.prepare \
        --retarget_manifest "$FMR_RETARGET_MANIFEST" \
        --output_dir "$FMR_DATA_DIR"
    elif [[ -n "${FMR_SOURCE_MANIFEST:-}" ]]; then
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
        -m rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.prepare \
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
        -m rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.prepare "${PREPARE_ARGS[@]}"
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
    ARMS="${FMR_ARMS:-paper_geo_base500k fmrope_base256 evq_cosh_tau4_paper_grid_base500k}"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" tests/test_fmrope_125m_l256_500m.py
    "$PYTHON_BIN" -m py_compile \
      "$SCRIPT_DIR/protocol.py" \
      "$SCRIPT_DIR/prepare.py" \
      "$SCRIPT_DIR/run_experiment.py"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
      -m rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.prepare \
      --validate "$DATA_MANIFEST" \
      --full_hash_check
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
      -m rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment preflight \
      --data_manifest "$DATA_MANIFEST" \
      --full_hash_check \
      --verify_full_initialization \
      --arms $ARMS \
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

from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.prepare import sha256_file
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.protocol import SPEC
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment import code_fingerprint

receipt = json.loads(Path(sys.argv[1]).read_text())
manifest = Path(sys.argv[2])
if receipt.get("status") != "PASS":
    raise SystemExit("preflight receipt is not PASS")
if receipt.get("code_sha256") != code_fingerprint():
    raise SystemExit("code changed after CPU preflight")
if receipt.get("data_manifest_sha256") != sha256_file(manifest):
    raise SystemExit("data manifest changed after CPU preflight")
if receipt.get("protocol_sha256") != SPEC.fingerprint():
    raise SystemExit("seed/protocol changed after CPU preflight")
if receipt.get("seed") != SPEC.seed:
    raise SystemExit("seed changed after CPU preflight")
if receipt.get("model_tier") != SPEC.model_tier:
    raise SystemExit("model tier changed after CPU preflight")
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
    COMPILE_MODE="${FMR_COMPILE_MODE:-default}"
    NUM_WORKERS="${FMR_NUM_WORKERS:-8}"
    ARMS="${FMR_ARMS:-paper_geo_base500k fmrope_base256 evq_cosh_tau4_paper_grid_base500k}"
    set -- $ARMS
    PROBE_ARM="$1"
    PROBE_PATH="$FMR_WORK_DIR/gpu_probe_${PROBE_ARM}.json"
    if [[ ! -s "$PROBE_PATH" ]]; then
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
        -m rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment probe-gpu \
        --arm "$PROBE_ARM" \
        --data_manifest "$DATA_MANIFEST" \
        --work_dir "$FMR_WORK_DIR" \
        --compile_mode "$COMPILE_MODE" \
        --timed_steps 5 \
        2>&1 | tee "$FMR_WORK_DIR/logs/gpu_probe_${PROBE_ARM}.log"
    fi
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - "$PROBE_PATH" "$PROBE_ARM" "$COMPILE_MODE" <<'PY'
import json
import sys
from pathlib import Path

from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.protocol import SPEC
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment import code_fingerprint

receipt = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "status": "PASS",
    "discarded_probe": True,
    "arm": sys.argv[2],
    "compile_mode": sys.argv[3],
    "protocol_sha256": SPEC.fingerprint(),
    "code_sha256": code_fingerprint(),
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(f"GPU probe receipt mismatch: {key}")
print("discarded GPU probe: PASS")
PY
    for arm in $ARMS; do
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
        -m rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment train \
        --arm "$arm" \
        --data_manifest "$DATA_MANIFEST" \
        --work_dir "$FMR_WORK_DIR" \
        --num_workers "$NUM_WORKERS" \
        --compile_mode "$COMPILE_MODE" \
        2>&1 | tee "$FMR_WORK_DIR/logs/train_${arm}.log"
    done
    if [[ "${FMR_EVALUATE:-1}" == "1" ]]; then
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
        -m rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment evaluate \
        --data_manifest "$DATA_MANIFEST" \
        --work_dir "$FMR_WORK_DIR" \
        --arms $ARMS \
        --eval_batch_size "${FMR_EVAL_BATCH_SIZE:-2}" \
        2>&1 | tee "$FMR_WORK_DIR/logs/evaluate.log"
      if [[ -n "${FMR_BASELINE_RESULTS:-}" ]]; then
        PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
          -m rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment compare \
          --baseline_results "$FMR_BASELINE_RESULTS" \
          --new_results "$FMR_WORK_DIR/evaluation/results.json" \
          --output_dir "$FMR_WORK_DIR/comparison"
      fi
    fi
    ;;

  evq2)
    export FMR_ARMS="anchored_cosh_tau4_fmrope_range"
    exec "$0" run
    ;;

  exact-range)
    export FMR_ARMS="fmrope_base256 anchored_cosh_tau4_fmrope_range"
    exec "$0" run
    ;;

  prepare-350m)
    export FMR_MODEL_TIER=350m
    export FMR_SEED=42
    exec "$0" prepare
    ;;

  preflight-350m)
    export FMR_MODEL_TIER=350m
    export FMR_SEED=42
    export FMR_ARMS="fmrope_base256 anchored_cosh_tau4_fmrope_range"
    exec "$0" preflight
    ;;

  run-350m)
    export FMR_MODEL_TIER=350m
    export FMR_SEED=42
    exec "$0" exact-range
    ;;

  preflight-multiseed)
    require_var FMR_WORK_DIR
    assert_cpu_only
    ROOT_WORK_DIR="$FMR_WORK_DIR"
    SHARED_DATA_DIR="${FMR_DATA_DIR:-$ROOT_WORK_DIR/data}"
    for seed in ${FMR_SEEDS:-137 256}; do
      FMR_SEED="$seed" \
      FMR_MODEL_TIER=151m \
      FMR_WORK_DIR="$ROOT_WORK_DIR/seed_${seed}" \
      FMR_DATA_DIR="$SHARED_DATA_DIR" \
      FMR_ARMS="fmrope_base256 anchored_cosh_tau4_fmrope_range" \
        "$0" preflight
    done
    ;;

  run-multiseed)
    require_var FMR_WORK_DIR
    ROOT_WORK_DIR="$FMR_WORK_DIR"
    SHARED_DATA_DIR="${FMR_DATA_DIR:-$ROOT_WORK_DIR/data}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_WORK_DIR/torchinductor_cache}"
    for seed in ${FMR_SEEDS:-137 256}; do
      FMR_SEED="$seed" \
      FMR_MODEL_TIER=151m \
      FMR_WORK_DIR="$ROOT_WORK_DIR/seed_${seed}" \
      FMR_DATA_DIR="$SHARED_DATA_DIR" \
        "$0" exact-range
    done
    ;;

  aggregate-multiseed)
    require_var FMR_WORK_DIR
    require_var FMR_SEED42_COMPARISON
    cd "$REPO_ROOT"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
      -m rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment \
      aggregate-exact-range \
      --inputs \
        "$FMR_SEED42_COMPARISON" \
        "$FMR_WORK_DIR/seed_137/evaluation/results.json" \
        "$FMR_WORK_DIR/seed_256/evaluation/results.json" \
      --output_dir "$FMR_WORK_DIR/multiseed"
    ;;

  evaluate)
    require_var FMR_WORK_DIR
    FMR_DATA_DIR="${FMR_DATA_DIR:-$FMR_WORK_DIR/data}"
    DATA_MANIFEST="$FMR_DATA_DIR/data_manifest.json"
    cd "$REPO_ROOT"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
      -m rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment evaluate \
      --data_manifest "$DATA_MANIFEST" \
      --work_dir "$FMR_WORK_DIR" \
      ${FMR_ARMS:+--arms $FMR_ARMS} \
      --eval_batch_size "${FMR_EVAL_BATCH_SIZE:-2}"
    ;;

  *)
    echo "usage: $0 {prepare|prepare-350m|preflight|run|evq2|exact-range|preflight-350m|run-350m|preflight-multiseed|run-multiseed|aggregate-multiseed|evaluate}" >&2
    exit 2
    ;;
esac
