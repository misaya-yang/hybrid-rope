#!/usr/bin/env bash
# RTX 5090 released Geo baseline: PPL + RULER for step-1000/2000/5000.
# Pro 6000 training is intentionally out of scope for this launcher.
set -euo pipefail

REPO_ROOT="${OLMO2_REPO_ROOT:-/root/autodl-tmp/hybrid-rope}"
PYTHON_BIN="${OLMO2_PYTHON:-/root/olmo2_venv/bin/python}"
ASSET_ROOT="${OLMO2_ASSET_ROOT:-/root/autodl-tmp/olmo2_1b_released}"
DATA_ROOT="${OLMO2_DATA_ROOT:-/root/autodl-tmp/olmo2_1b_evq_data}"
EVAL_ROOT="${OLMO2_EVAL_ROOT:-/root/autodl-tmp/olmo2_1b_evq_eval}"
EVAL32K_ROOT="${OLMO2_EVAL32K_ROOT:-/root/autodl-tmp/olmo2_1b_evq_eval32k}"
RULER_DATA_ROOT="${OLMO2_RULER_DATA_ROOT:-/root/autodl-tmp/olmo2_1b_ruler_data}"
RULER_SWEEP_DATA_ROOT="${OLMO2_RULER_SWEEP_DATA_ROOT:-/root/autodl-tmp/olmo2_1b_ruler_4k10k}"
LM_EVAL_ROOT="${OLMO2_LM_EVAL_ROOT:-/root/autodl-tmp/lm-evaluation-harness}"
OUTPUT_ROOT="${OLMO2_RELEASED_EVAL_ROOT:-/root/autodl-tmp/olmo2_1b_released_eval}"
PACKAGE="rebuttal.rebuttal_0723.experiments.olmo2_1b_evq"
RECEIPT="$OUTPUT_ROOT/READY.json"
RULER_RECEIPT="$OUTPUT_ROOT/RULER_READY.json"
LM_EVAL_COMMIT="f4d4b3de3ee6741a7151a9fe74945ee515262f4c"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/autodl-tmp/torchinductor_cache/olmo2_sm120}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export NLTK_DATA="${NLTK_DATA:-/root/autodl-tmp/nltk_data}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip_cache}"

cpu_preflight() {
  local snapshot="${1:-}"
  local receipt="$RECEIPT"
  local snapshot_args=()
  if [[ -n "$snapshot" ]]; then
    receipt="$OUTPUT_ROOT/READY_${snapshot}.json"
    snapshot_args=(--snapshots "$snapshot")
  fi
  mkdir -p "$OUTPUT_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
  "$PYTHON_BIN" -m "$PACKAGE.preflight_released_eval" \
    --asset-root "$ASSET_ROOT" \
    "${snapshot_args[@]}" \
    --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
    --data-manifest "$DATA_ROOT/dataset_manifest.json" \
    --output-root "$OUTPUT_ROOT" \
    --receipt "$receipt"
}

require_ready() {
  local snapshot="$1"
  local receipt="$OUTPUT_ROOT/READY_${snapshot}.json"
  cpu_preflight "$snapshot" >/dev/null
  "$PYTHON_BIN" - "$receipt" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "RELEASED_BASELINE_EVAL_CPU_READY":
    raise SystemExit("released baseline evaluation is not CPU-ready")
PY
}

require_ruler_data() {
  "$PYTHON_BIN" - "$RULER_DATA_ROOT/ruler_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing RULER manifest: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "RULER_DATA_VERIFIED":
    raise SystemExit(f"RULER data status={payload.get('status')}")
print(f"RULER_DATA_OK tasks={len(payload.get('files', {}))}")
PY
}

require_gpu() {
  "$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
if "5090" not in name or (major, minor) != (12, 0):
    raise SystemExit(f"expected RTX 5090 sm_120, got {name} sm_{major}{minor}")
print(f"GPU_READY {name} sm_{major}{minor}")
PY
}

snapshot_path() {
  local name="$1"
  local path="$ASSET_ROOT/$name"
  if [[ ! -d "$path" ]]; then
    echo "missing snapshot directory: $path" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

run_ppl() {
  local name="$1"
  local step="$2"
  local checkpoint
  checkpoint="$(snapshot_path "$name")"
  local output="$OUTPUT_ROOT/released_native_step${step}"
  require_ready "$name"
  require_gpu
  if [[ -e "$output" ]]; then
    echo "PPL output already exists (resume/manual clean): $output" >&2
    exit 1
  fi
  "$PYTHON_BIN" -m "$PACKAGE.evaluate" \
    --base-model "$checkpoint" \
    --checkpoint "$checkpoint" \
    --data-manifest "$DATA_ROOT/dataset_manifest.json" \
    --schedule geo \
    --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
    --output "$output" \
    --batch-size-4k 2 \
    --batch-size-16k 1 \
    --lm-head-chunk-tokens 256 \
    --compile \
    --compile-mode max-autotune-no-cudagraphs
}

run_ppl32k() {
  local name="$1"
  local step="$2"
  local checkpoint
  checkpoint="$(snapshot_path "$name")"
  local output="$OUTPUT_ROOT/released_native_step${step}_32k"
  require_ready "$name"
  require_gpu
  if [[ -e "$output" ]]; then
    echo "32K PPL output already exists (resume/manual clean): $output" >&2
    exit 1
  fi
  "$PYTHON_BIN" -m "$PACKAGE.evaluate" \
    --base-model "$checkpoint" \
    --checkpoint "$checkpoint" \
    --data-manifest "$DATA_ROOT/dataset_manifest.json" \
    --schedule geo \
    --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
    --extra-32k-manifest "$EVAL32K_ROOT/eval32k_manifest.json" \
    --only-extra-32k \
    --output "$output" \
    --lm-head-chunk-tokens 256 \
    --compile \
    --compile-mode max-autotune-no-cudagraphs
}

run_ruler() {
  local name="$1"
  local step="$2"
  local suite="${3:-quick}"
  local limit="${4:-100}"
  local checkpoint
  checkpoint="$(snapshot_path "$name")"
  local output="$OUTPUT_ROOT/ruler_${name}_${suite}_n${limit}"
  require_ruler_data
  require_gpu
  mkdir -p "$output"
  "$PYTHON_BIN" -m "$PACKAGE.evaluate_ruler" \
    --checkpoint "$checkpoint" \
    --data-root "$RULER_DATA_ROOT" \
    --output "$output" \
    --suite "$suite" \
    --limit-per-cell "$limit"
}

prepare_lm_eval() {
  local actual_commit
  if [[ -d "$LM_EVAL_ROOT/.git" ]]; then
    actual_commit="$(git -C "$LM_EVAL_ROOT" rev-parse HEAD)"
    if [[ -n "$(git -C "$LM_EVAL_ROOT" status --porcelain --untracked-files=no)" ]]; then
      echo "lm-eval tracked sources are dirty: $LM_EVAL_ROOT" >&2
      exit 1
    fi
  elif [[ -f "$LM_EVAL_ROOT/PINNED_COMMIT" ]]; then
    actual_commit="$(tr -d '[:space:]' <"$LM_EVAL_ROOT/PINNED_COMMIT")"
  else
    echo "missing offline lm-eval source receipt: $LM_EVAL_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$LM_EVAL_ROOT/lm_eval/tasks/ruler" ]]; then
    echo "missing offline lm-eval RULER source: $LM_EVAL_ROOT" >&2
    exit 1
  fi
  if [[ "$actual_commit" != "$LM_EVAL_COMMIT" ]]; then
    echo "lm-eval commit drift: expected $LM_EVAL_COMMIT, got ${actual_commit:-missing}" >&2
    exit 1
  fi
  # prepare_ruler_data.py records SHA-256 for every imported generator module,
  # so a portable source-only bundle is independently represented in the data
  # manifest even when the full Git object store is not copied to the GPU host.
  printf '%s\n' "$LM_EVAL_COMMIT" >"$LM_EVAL_ROOT/PINNED_COMMIT"
}

prepare_ruler_data() {
  prepare_lm_eval
  local tokenizer
  tokenizer="$(snapshot_path geo5000)"
  # Prefer any complete released snapshot for the shared Dolma2 tokenizer.
  if [[ ! -f "$tokenizer/tokenizer.json" ]]; then
    tokenizer="$(snapshot_path geo1000)"
  fi
  mkdir -p "$RULER_DATA_ROOT"
  "$PYTHON_BIN" -m "$PACKAGE.prepare_ruler_data" \
    --lm-eval-root "$LM_EVAL_ROOT" \
    --tokenizer "$tokenizer" \
    --qa-source-root "$RULER_DATA_ROOT/source_cache" \
    --output "$RULER_DATA_ROOT" \
    --resume
  require_ruler_data | tee "$RULER_RECEIPT"
}

prepare_ruler_sweep_data() {
  prepare_lm_eval
  local tokenizer
  tokenizer="$(snapshot_path geo1000)"
  mkdir -p "$RULER_SWEEP_DATA_ROOT"
  "$PYTHON_BIN" -m "$PACKAGE.prepare_ruler_data" \
    --lm-eval-root "$LM_EVAL_ROOT" \
    --tokenizer "$tokenizer" \
    --qa-source-root "$RULER_DATA_ROOT/source_cache" \
    --output "$RULER_SWEEP_DATA_ROOT" \
    --tasks niah_single_1 niah_multikey_2 ruler_vt ruler_fwe \
    --lengths 4096 5120 6144 7168 8192 9216 10240 \
    --resume
}

run_ruler_sweep() {
  local name="$1"
  local step="$2"
  local checkpoint
  checkpoint="$(snapshot_path "$name")"
  local output="$OUTPUT_ROOT/ruler_${name}_quick_4k10k_n100"
  require_gpu
  mkdir -p "$output"
  "$PYTHON_BIN" -m "$PACKAGE.evaluate_ruler" \
    --checkpoint "$checkpoint" \
    --data-root "$RULER_SWEEP_DATA_ROOT" \
    --output "$output" \
    --suite quick \
    --lengths 4096 5120 6144 7168 8192 9216 10240 \
    --limit-per-cell 100
}

require_ruler_sweep_data() {
  "$PYTHON_BIN" - "$RULER_SWEEP_DATA_ROOT/ruler_manifest.json" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
expected_tasks = [
    "niah_single_1",
    "niah_multikey_2",
    "ruler_vt",
    "ruler_fwe",
]
expected_lengths = [4096, 5120, 6144, 7168, 8192, 9216, 10240]
if payload.get("status") != "RULER_DATA_VERIFIED":
    raise SystemExit(f"RULER sweep data status={payload.get('status')}")
if payload["implementation"]["tasks"] != expected_tasks:
    raise SystemExit("RULER sweep task drift")
if payload["protocol"]["lengths"] != expected_lengths:
    raise SystemExit("RULER sweep length drift")
if set(payload["files"]) != set(expected_tasks):
    raise SystemExit("RULER sweep file set drift")
print("RULER_SWEEP_DATA_OK")
PY
}

case "${1:-}" in
  preflight)
    cpu_preflight
    ;;
  prepare-ruler)
    prepare_ruler_data
    ;;
  prepare-ruler-sweep)
    prepare_ruler_sweep_data
    ;;
  eval1000)
    run_ppl geo1000 1000
    ;;
  eval2000)
    run_ppl geo2000 2000
    ;;
  eval5000)
    run_ppl geo5000 5000
    ;;
  eval1000-32k)
    run_ppl32k geo1000 1000
    ;;
  eval2000-32k)
    run_ppl32k geo2000 2000
    ;;
  eval5000-32k)
    run_ppl32k geo5000 5000
    ;;
  ruler1000)
    run_ruler geo1000 1000 "${2:-quick}" "${3:-100}"
    ;;
  ruler2000)
    run_ruler geo2000 2000 "${2:-quick}" "${3:-100}"
    ;;
  ruler5000)
    run_ruler geo5000 5000 "${2:-quick}" "${3:-100}"
    ;;
  ruler1000-sweep)
    run_ruler_sweep geo1000 1000
    ;;
  ruler2000-sweep)
    run_ruler_sweep geo2000 2000
    ;;
  ruler-sweep-sequence)
    prepare_ruler_sweep_data
    require_ruler_sweep_data
    run_ruler_sweep geo1000 1000
    run_ruler_sweep geo2000 2000
    ;;
  ppl-sequence)
    run_ppl geo1000 1000
    run_ppl geo2000 2000
    run_ppl geo5000 5000
    ;;
  ppl32k-sequence)
    run_ppl32k geo1000 1000
    run_ppl32k geo2000 2000
    run_ppl32k geo5000 5000
    ;;
  ruler-sequence)
    run_ruler geo1000 1000 "${2:-quick}" "${3:-100}"
    run_ruler geo2000 2000 "${2:-quick}" "${3:-100}"
    run_ruler geo5000 5000 "${2:-quick}" "${3:-100}"
    ;;
  full-sequence)
    run_ppl geo1000 1000
    run_ppl geo2000 2000
    run_ppl geo5000 5000
    run_ruler geo1000 1000 "${2:-quick}" "${3:-100}"
    run_ruler geo2000 2000 "${2:-quick}" "${3:-100}"
    run_ruler geo5000 5000 "${2:-quick}" "${3:-100}"
    ;;
  status)
    test -f "$RECEIPT" && cat "$RECEIPT" || true
    test -f "$RULER_RECEIPT" && cat "$RULER_RECEIPT" || true
    find "$OUTPUT_ROOT" -maxdepth 3 -type f \
      \( -name "results.json" -o -name "per_token_nll.npz" -o -name "examples.jsonl" \) \
      -printf "%p %s\n" 2>/dev/null | sort
    ls -lah "$ASSET_ROOT" 2>/dev/null || true
    ls -lah "$RULER_DATA_ROOT" 2>/dev/null || true
    ;;
  *)
    cat >&2 <<'EOF'
usage: run_5090_released_eval.sh {
  preflight|prepare-ruler|prepare-ruler-sweep|
  eval1000|eval2000|eval5000|ppl-sequence|
  eval1000-32k|eval2000-32k|eval5000-32k|ppl32k-sequence|
  ruler1000|ruler2000|ruler5000|ruler1000-sweep|ruler2000-sweep|
  ruler-sweep-sequence|ruler-sequence|
  full-sequence|status
} [ruler_suite=quick|full] [limit_per_cell=100]
EOF
    exit 2
    ;;
esac
