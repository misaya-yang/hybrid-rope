#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PACKAGE_DIR="$REPO_ROOT/experiments/native_rope_evq_150m"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "required environment variable is unset: $name" >&2
    exit 2
  fi
}

file_sha256() {
  "$PYTHON_BIN" -c 'import hashlib,sys; p=sys.argv[1]; h=hashlib.sha256(); f=open(p,"rb"); [h.update(x) for x in iter(lambda:f.read(8<<20),b"")]; print(h.hexdigest())' "$1"
}

code_fingerprint() {
  REPO_ROOT="$REPO_ROOT" "$PYTHON_BIN" - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ["REPO_ROOT"])
paths = sorted((root / "experiments/native_rope_evq_150m").glob("*.py"))
paths += [
    root / "experiments/native_rope_evq_150m/run_seed42.sh",
    root / "scripts/lib/rope/official_yarn.py",
    root / "scripts/lib/rope/schedules.py",
    root / "scripts/supporting_eval/eval_passkey_scratch.py",
    root / "tests/test_native_rope_evq_150m.py",
    root / "tests/test_official_yarn_parity.py",
]
digest = hashlib.sha256()
for path in paths:
    if not path.is_file():
        raise SystemExit(f"missing code file: {path}")
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
}

case "$MODE" in
  prepare)
    require_var EVQ_150M_WORK_DIR
    require_var EVQ_150M_TRAIN_NPY
    require_var EVQ_150M_TRAIN_MANIFEST
    require_var EVQ_150M_TOKENIZER
    require_var EVQ_150M_PARQUET_DIR
    EVQ_150M_DATA_DIR="${EVQ_150M_DATA_DIR:-$EVQ_150M_WORK_DIR/data}"
    mkdir -p "$EVQ_150M_DATA_DIR" "$EVQ_150M_WORK_DIR/logs"
    "$PYTHON_BIN" -c 'import torch; raise SystemExit("prepare mode requires CPU-only instance") if torch.cuda.is_available() else None'
    if [[ -f "$EVQ_150M_DATA_DIR/data_manifest.json" ]]; then
      echo "reuse existing prepared manifest: $EVQ_150M_DATA_DIR/data_manifest.json"
    else
      cd "$REPO_ROOT"
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -m experiments.native_rope_evq_150m.prepare_data \
        --train_npy "$EVQ_150M_TRAIN_NPY" \
        --train_manifest "$EVQ_150M_TRAIN_MANIFEST" \
        --tokenizer "$EVQ_150M_TOKENIZER" \
        --parquet_dir "$EVQ_150M_PARQUET_DIR" \
        --output_dir "$EVQ_150M_DATA_DIR" \
        2>&1 | tee "$EVQ_150M_WORK_DIR/logs/prepare.log"
    fi
    ;;

  preflight)
    require_var EVQ_150M_WORK_DIR
    EVQ_150M_DATA_DIR="${EVQ_150M_DATA_DIR:-$EVQ_150M_WORK_DIR/data}"
    DATA_MANIFEST="$EVQ_150M_DATA_DIR/data_manifest.json"
    [[ -f "$DATA_MANIFEST" ]] || { echo "missing prepared data manifest: $DATA_MANIFEST" >&2; exit 2; }
    mkdir -p "$EVQ_150M_WORK_DIR"
    cd "$REPO_ROOT"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" tests/test_native_rope_evq_150m.py
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" tests/test_official_yarn_parity.py
    "$PYTHON_BIN" -m py_compile \
      "$PACKAGE_DIR/protocol.py" \
      "$PACKAGE_DIR/model.py" \
      "$PACKAGE_DIR/prepare_data.py" \
      "$PACKAGE_DIR/train.py" \
      "$PACKAGE_DIR/evaluate.py"
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -m experiments.native_rope_evq_150m.prepare_data --self_test
    for arm in native_rope endpoint_evq_tau1p5; do
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -m experiments.native_rope_evq_150m.train \
        --arm "$arm" \
        --data_manifest "$DATA_MANIFEST" \
        --work_dir "$EVQ_150M_WORK_DIR" \
        --dry_run
    done
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -m experiments.native_rope_evq_150m.evaluate \
      --data_manifest "$DATA_MANIFEST" \
      --dry_run
    DATA_SHA="$(file_sha256 "$DATA_MANIFEST")"
    CODE_SHA="$(code_fingerprint)"
    printf 'DATA_SHA=%q\nCODE_SHA=%q\n' "$DATA_SHA" "$CODE_SHA" > "$EVQ_150M_WORK_DIR/preflight.env"
    echo "preflight PASS data_sha=$DATA_SHA code_sha=$CODE_SHA"
    ;;

  run)
    require_var EVQ_150M_WORK_DIR
    EVQ_150M_DATA_DIR="${EVQ_150M_DATA_DIR:-$EVQ_150M_WORK_DIR/data}"
    DATA_MANIFEST="$EVQ_150M_DATA_DIR/data_manifest.json"
    PREFLIGHT="$EVQ_150M_WORK_DIR/preflight.env"
    [[ -f "$DATA_MANIFEST" ]] || { echo "missing prepared data manifest: $DATA_MANIFEST" >&2; exit 2; }
    [[ -f "$PREFLIGHT" ]] || { echo "missing CPU preflight: $PREFLIGHT" >&2; exit 2; }
    # shellcheck disable=SC1090
    source "$PREFLIGHT"
    [[ "$DATA_SHA" == "$(file_sha256 "$DATA_MANIFEST")" ]] || { echo "data changed after preflight" >&2; exit 2; }
    [[ "$CODE_SHA" == "$(code_fingerprint)" ]] || { echo "code changed after preflight" >&2; exit 2; }
    "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; assert torch.cuda.is_bf16_supported(), "BF16 is required"; p=torch.cuda.get_device_properties(0); m=getattr(p,"total_memory",getattr(p,"total_mem",0)); assert m >= 30*2**30, f"registered micro-batch 12 requires >=30GiB, found {m/2**30:.1f}GiB"; print(p.name, m/2**30)'
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$EVQ_150M_WORK_DIR/torchinductor_cache}"
    export TORCHINDUCTOR_FX_GRAPH_CACHE=1
    export TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$EVQ_150M_WORK_DIR/logs"
    COMPILE_MODE="${EVQ_150M_COMPILE_MODE:-max-autotune-no-cudagraphs}"
    NUM_WORKERS="${EVQ_150M_NUM_WORKERS:-8}"
    cd "$REPO_ROOT"
    for arm in native_rope endpoint_evq_tau1p5; do
      PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -m experiments.native_rope_evq_150m.train \
        --arm "$arm" \
        --data_manifest "$DATA_MANIFEST" \
        --work_dir "$EVQ_150M_WORK_DIR" \
        --num_workers "$NUM_WORKERS" \
        --compile_mode "$COMPILE_MODE" \
        2>&1 | tee "$EVQ_150M_WORK_DIR/logs/train_${arm}.log"
    done
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -m experiments.native_rope_evq_150m.evaluate \
      --work_dir "$EVQ_150M_WORK_DIR" \
      --data_manifest "$DATA_MANIFEST" \
      --output_dir "$EVQ_150M_WORK_DIR/evaluation" \
      2>&1 | tee "$EVQ_150M_WORK_DIR/logs/evaluate.log"
    ;;

  *)
    echo "usage: $0 {prepare|preflight|run}" >&2
    exit 2
    ;;
esac
