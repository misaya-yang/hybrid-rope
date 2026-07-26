#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VENV_ROOT="${OLMO2_VENV_ROOT:-/root/olmo2_venv}"
if [[ -x "$VENV_ROOT/bin/python" ]]; then
  DEFAULT_PYTHON="$VENV_ROOT/bin/python"
else
  DEFAULT_PYTHON="/root/miniconda3/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
ASSET_ROOT="${OLMO2_ASSET_ROOT:-/root/olmo2_assets}"
DATA_ROOT="${OLMO2_DATA_ROOT:-/root/autodl-tmp/olmo2_1b_evq_data}"
WORK_ROOT="${OLMO2_WORK_ROOT:-/root/autodl-tmp/olmo2_1b_evq_run}"
CHECKPOINT_ROOT="${OLMO2_CHECKPOINT_ROOT:-$WORK_ROOT}"
COMPILE_CACHE="${OLMO2_COMPILE_CACHE:-/root/autodl-tmp/torchinductor_cache/olmo2_sm120}"
DATA_MANIFEST="$DATA_ROOT/dataset_manifest.json"
EVAL_ROOT="${OLMO2_EVAL_ROOT:-/root/autodl-tmp/olmo2_1b_evq_eval}"
RETRIEVAL_ROOT="$EVAL_ROOT/retrieval"
EVAL_WORK="$WORK_ROOT/evaluation"
CPU_RECEIPT="$WORK_ROOT/cpu_preflight.json"
PORTABLE_RECEIPT="${OLMO2_PORTABLE_RECEIPT:-$WORK_ROOT/portable_preflight.json}"
PROBE_DIR="$WORK_ROOT/probes"
BACKEND_RECEIPT="$WORK_ROOT/selected_backend.json"
RUNTIME_CANDIDATE="$WORK_ROOT/runtime_candidate.json"
RUNTIME_RECEIPT="$WORK_ROOT/selected_runtime.json"
GEO_SENTINEL="$WORK_ROOT/geo_sentinel_20"
EVQ_OUTPUT="$CHECKPOINT_ROOT/evq_1000"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE"

require_cpu_receipt() {
  "$PYTHON_BIN" - \
    "$CPU_RECEIPT" \
    "$SCRIPT_DIR" \
    "$DATA_MANIFEST" \
    "$EVAL_ROOT/eval_manifest.json" \
    "$RETRIEVAL_ROOT/retrieval_manifest.json" \
    "$ASSET_ROOT/asset_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import sha256_file
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.preflight import validate_code

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "CPU_PREFLIGHT_PASS":
    raise SystemExit(f"CPU preflight is not PASS: {path}")
expected_code = payload["checks"]["code"]["evidence"]["code_sha256"]
actual_code = validate_code(Path(sys.argv[2]))["code_sha256"]
if expected_code != actual_code:
    raise SystemExit("experiment code changed after CPU preflight")
expected_data = payload["checks"]["dataset"]["evidence"]["manifest_sha256"]
if expected_data != sha256_file(Path(sys.argv[3])):
    raise SystemExit("dataset manifest changed after CPU preflight")
expected_eval = payload["checks"]["evaluation_dataset"]["evidence"]["manifest_sha256"]
if expected_eval != sha256_file(Path(sys.argv[4])):
    raise SystemExit("evaluation manifest changed after CPU preflight")
expected_retrieval = payload["checks"]["retrieval_dataset"]["evidence"]["manifest_sha256"]
if expected_retrieval != sha256_file(Path(sys.argv[5])):
    raise SystemExit("retrieval manifest changed after CPU preflight")
expected_assets = payload["checks"]["assets"]["evidence"]["asset_manifest_sha256"]
if expected_assets != sha256_file(Path(sys.argv[6])):
    raise SystemExit("asset manifest changed after CPU preflight")
PY
}

require_launch_storage() {
  "$PYTHON_BIN" - "$CPU_RECEIPT" "$EVQ_OUTPUT" <<'PY'
import json
import os
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
train_output = Path(sys.argv[2])
checkpoint_root = train_output.parent
minimum = int(
    receipt["checks"]["storage"]["evidence"]["minimum_free_bytes"]
)
statistics = os.statvfs(checkpoint_root)
available = statistics.f_bavail * statistics.f_frsize
if available < minimum:
    raise SystemExit(
        f"launch storage drift: {available / 2**30:.1f} GiB free, "
        f"need {minimum / 2**30:.1f} GiB"
    )
if train_output.exists() and any(train_output.iterdir()):
    raise SystemExit(f"EVQ output is non-empty before launch: {train_output}")
print(
    f"launch storage: PASS ({available / 2**30:.1f} GiB free)",
    flush=True,
)
PY
}

require_runtime_receipt() {
  "$PYTHON_BIN" - "$RUNTIME_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "GPU_RUNTIME_SELECTED":
    raise SystemExit(f"GPU runtime is not selected: {path}")
PY
}

selected_backend() {
  "$PYTHON_BIN" - "$RUNTIME_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["selected_loss_backend"])
PY
}

candidate_backend() {
  "$PYTHON_BIN" - "$RUNTIME_CANDIDATE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["selected_loss_backend"])
PY
}

candidate_microbatch() {
  "$PYTHON_BIN" - "$RUNTIME_CANDIDATE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["selected_microbatch_sequences"])
PY
}

selected_microbatch() {
  "$PYTHON_BIN" - "$RUNTIME_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["selected_microbatch_sequences"])
PY
}

require_geo_sentinel() {
  "$PYTHON_BIN" - "$GEO_SENTINEL/completed.json" "$GEO_SENTINEL/train.jsonl" <<'PY'
import json
import math
import sys
from pathlib import Path

complete_path = Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if complete_path.is_file():
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        complete.get("status") != "TRAINING_COMPLETE"
        or complete.get("stop_step") != 20
        or len(rows) != 20
    ):
        raise SystemExit("completed Geo sentinel receipt is inconsistent")
    status = "complete_20"
elif len(rows) == 18:
    # The registered one-hour host shutdown interrupted the sentinel after
    # step 18. Preserve that fact; do not manufacture a completion receipt.
    status = "scheduled_shutdown_after_18"
else:
    raise SystemExit(
        f"Geo sentinel has {len(rows)} rows without a completion receipt"
    )
if [int(row["step"]) for row in rows] != list(range(1, len(rows) + 1)):
    raise SystemExit("Geo sentinel step sequence drift")
if not all(row.get("finite") and math.isfinite(row["train_ce_loss"]) for row in rows):
    raise SystemExit("Geo sentinel contains a non-finite loss")
if rows[-1]["train_ce_loss"] >= rows[0]["train_ce_loss"]:
    raise SystemExit("Geo sentinel loss did not decrease")
print(
    f"Geo sentinel: PASS ({status}, {len(rows)} steps, "
    f"CE {rows[0]['train_ce_loss']:.4f}->{rows[-1]['train_ce_loss']:.4f})"
)
PY
}

require_evq_500() {
  "$PYTHON_BIN" - \
    "$EVQ_OUTPUT/completed.json" \
    "$EVQ_OUTPUT/evq_smoke_step20.json" \
    "$EVQ_OUTPUT/step-000500-full/trainer_state.json" <<'PY'
import json
import sys
from pathlib import Path

completed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
smoke = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
state = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if completed.get("status") != "TRAINING_COMPLETE" or completed.get("stop_step") != 500:
    raise SystemExit("EVQ phase 0->500 is incomplete")
if smoke.get("status") != "EVQ_SMOKE_PASS":
    raise SystemExit("EVQ step-20 smoke did not pass")
if state.get("step") != 500:
    raise SystemExit("EVQ step-500 full state is invalid")
PY
}

run_probe() {
  local backend="$1"
  local microbatch="${2:-4}"
  local suffix="${3:-}"
  local warmup="${4:-5}"
  local measured="${5:-20}"
  mkdir -p "$PROBE_DIR" "$COMPILE_CACHE"
  "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.train \
    probe \
    --model-path "$ASSET_ROOT/step0" \
    --data-manifest "$DATA_MANIFEST" \
    --output "$PROBE_DIR" \
    --schedule evq \
    --loss-backend "$backend" \
    --microbatch "$microbatch" \
    --data-workers 8 \
    --compile \
    --compile-mode max-autotune-no-cudagraphs \
    --probe-warmup "$warmup" \
    --probe-steps "$measured" \
    --probe-suffix "$suffix"
}

case "$MODE" in
  prepare-env)
    /root/miniconda3/bin/python -m venv --system-site-packages "$VENV_ROOT"
    "$VENV_ROOT/bin/python" -m pip install \
      "transformers==4.57.6" \
      "safetensors==0.8.0" \
      "liger-kernel==0.7.0" \
      "accelerate==1.14.0" \
      "pyarrow==23.0.1" \
      "pytest==8.4.2"
    "$VENV_ROOT/bin/python" - <<'PY'
import liger_kernel
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("liger_kernel", getattr(liger_kernel, "__version__", "installed"))
PY
    ;;
  prepare-assets)
    mkdir -p "$ASSET_ROOT"
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
      "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_assets \
      --asset-root "$ASSET_ROOT" \
      --include-geo1000 \
      --max-workers 8
    ;;
  prepare-data)
    mkdir -p "$DATA_ROOT"
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_official_stream \
      --official-config "$ASSET_ROOT/upstream/OLMo2-1B-stage1.yaml" \
      --output-dir "$DATA_ROOT" \
      --tokenizer-path "$ASSET_ROOT/step0" \
      --source-workers 64 \
      --download-workers 128 \
      --steps 1000
    ;;
  validate-data)
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_official_stream \
      --official-config "$ASSET_ROOT/upstream/OLMo2-1B-stage1.yaml" \
      --output-dir "$DATA_ROOT" \
      --tokenizer-path "$ASSET_ROOT/step0" \
      --steps 1000 \
      --validate-only
    ;;
  prove-order)
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_official_stream \
      --official-config "$ASSET_ROOT/upstream/OLMo2-1B-stage1.yaml" \
      --output-dir "$DATA_ROOT" \
      --tokenizer-path "$ASSET_ROOT/step0" \
      --steps 1000 \
      --prove-order-only
    ;;
  validate-assets)
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_assets \
      --asset-root "$ASSET_ROOT" \
      --include-geo1000 \
      --validate-only
    ;;
  prepare-eval)
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_eval_data \
      --output-dir "$EVAL_ROOT" \
      --tokenizer-path "$ASSET_ROOT/step0" \
      --long-count 128 \
      --short-count 256
    ;;
  validate-eval)
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_eval_data \
      --output-dir "$EVAL_ROOT" \
      --tokenizer-path "$ASSET_ROOT/step0" \
      --long-count 128 \
      --short-count 256 \
      --validate-only
    ;;
  prepare-retrieval)
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_retrieval_data \
      --tokenizer-path "$ASSET_ROOT/step0" \
      --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
      --output-dir "$RETRIEVAL_ROOT" \
      --examples-per-cell 4
    ;;
  validate-retrieval)
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_retrieval_data \
      --tokenizer-path "$ASSET_ROOT/step0" \
      --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
      --output-dir "$RETRIEVAL_ROOT" \
      --examples-per-cell 4 \
      --validate-only
    ;;
  cpu-preflight)
    mkdir -p "$WORK_ROOT" "$CHECKPOINT_ROOT" "$COMPILE_CACHE"
    CUDA_VISIBLE_DEVICES="" \
      "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.preflight \
      --asset-root "$ASSET_ROOT" \
      --data-manifest "$DATA_MANIFEST" \
      --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
      --retrieval-manifest "$RETRIEVAL_ROOT/retrieval_manifest.json" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --train-output "$EVQ_OUTPUT" \
      --compile-cache "$COMPILE_CACHE" \
      --receipt "$CPU_RECEIPT" \
      --gpu-command "cd $REPO_ROOT && bash $SCRIPT_DIR/run_pro6000.sh gpu-sequence"
    ;;
  portable-preflight)
    mkdir -p "$WORK_ROOT" "$COMPILE_CACHE"
    CUDA_VISIBLE_DEVICES="" \
      "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.preflight \
      --asset-root "$ASSET_ROOT" \
      --data-manifest "$DATA_MANIFEST" \
      --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
      --retrieval-manifest "$RETRIEVAL_ROOT/retrieval_manifest.json" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --train-output "$EVQ_OUTPUT" \
      --compile-cache "$COMPILE_CACHE" \
      --receipt "$PORTABLE_RECEIPT" \
      --gpu-command "cd $REPO_ROOT && bash $SCRIPT_DIR/run_pro6000.sh gpu-sequence" \
      --portable-only
    ;;
  cpu-preflight-from-portable)
    test -f "$PORTABLE_RECEIPT"
    mkdir -p "$WORK_ROOT" "$CHECKPOINT_ROOT" "$COMPILE_CACHE"
    CUDA_VISIBLE_DEVICES="" \
      "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.preflight \
      --asset-root "$ASSET_ROOT" \
      --data-manifest "$DATA_MANIFEST" \
      --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
      --retrieval-manifest "$RETRIEVAL_ROOT/retrieval_manifest.json" \
      --checkpoint-root "$CHECKPOINT_ROOT" \
      --train-output "$EVQ_OUTPUT" \
      --compile-cache "$COMPILE_CACHE" \
      --receipt "$CPU_RECEIPT" \
      --gpu-command "cd $REPO_ROOT && bash $SCRIPT_DIR/run_pro6000.sh gpu-sequence" \
      --portable-receipt "$PORTABLE_RECEIPT"
    ;;
  probe-native)
    require_cpu_receipt
    run_probe native
    ;;
  probe-liger)
    require_cpu_receipt
    run_probe liger
    ;;
  select-probe)
    require_cpu_receipt
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.select_probe \
      --probe-dir "$PROBE_DIR" \
      --output "$BACKEND_RECEIPT"
    ;;
  probe-microbatch8)
    require_cpu_receipt
    backend="$("$PYTHON_BIN" - "$BACKEND_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["selected_loss_backend"])
PY
)"
    "$PYTHON_BIN" - "$PROBE_DIR/probe_${backend}_mb8.json" <<'PY'
import sys
from pathlib import Path
Path(sys.argv[1]).unlink(missing_ok=True)
PY
    if run_probe "$backend" 8 "_mb8" 5 20; then
      :
    else
      echo "microbatch=8 probe failed; retaining microbatch=4 candidate" >&2
    fi
    ;;
  choose-runtime)
    backend="$("$PYTHON_BIN" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["selected_loss_backend"])' \
      "$BACKEND_RECEIPT")"
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.select_runtime \
      choose \
      --backend-receipt "$BACKEND_RECEIPT" \
      --microbatch8-probe "$PROBE_DIR/probe_${backend}_mb8.json" \
      --output "$RUNTIME_CANDIDATE"
    ;;
  probe-sustained)
    require_cpu_receipt
    backend="$(candidate_backend)"
    microbatch="$(candidate_microbatch)"
    run_probe "$backend" "$microbatch" "_sustained" 100 300
    ;;
  finalize-runtime)
    backend="$(candidate_backend)"
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.select_runtime \
      finalize \
      --candidate-receipt "$RUNTIME_CANDIDATE" \
      --sustained-probe "$PROBE_DIR/probe_${backend}_sustained.json" \
      --output "$RUNTIME_RECEIPT"
    ;;
  geo-sentinel)
    require_cpu_receipt
    require_runtime_receipt
    backend="$(selected_backend)"
    microbatch="$(selected_microbatch)"
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.train \
      train \
      --model-path "$ASSET_ROOT/step0" \
      --data-manifest "$DATA_MANIFEST" \
      --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
      --output "$GEO_SENTINEL" \
      --schedule geo \
      --loss-backend "$backend" \
      --microbatch "$microbatch" \
      --data-workers 8 \
      --compile \
      --compile-mode max-autotune-no-cudagraphs \
      --stop-step 20 \
      --validation-interval 20 \
      --long-validation-interval 250 \
      --full-save-steps "" \
      --verify-data-hashes
    ;;
  train-evq-500)
    require_cpu_receipt
    require_runtime_receipt
    require_geo_sentinel
    backend="$(selected_backend)"
    microbatch="$(selected_microbatch)"
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.train \
      train \
      --model-path "$ASSET_ROOT/step0" \
      --data-manifest "$DATA_MANIFEST" \
      --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
      --output "$EVQ_OUTPUT" \
      --schedule evq \
      --loss-backend "$backend" \
      --microbatch "$microbatch" \
      --data-workers 8 \
      --compile \
      --compile-mode max-autotune-no-cudagraphs \
      --stop-step 500 \
      --full-save-steps "500" \
      --reference-sentinel-log "$GEO_SENTINEL/train.jsonl" \
      --verify-data-hashes
    ;;
  train-evq-1000)
    require_cpu_receipt
    require_runtime_receipt
    require_geo_sentinel
    require_evq_500
    backend="$(selected_backend)"
    microbatch="$(selected_microbatch)"
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.train \
      train \
      --model-path "$ASSET_ROOT/step0" \
      --data-manifest "$DATA_MANIFEST" \
      --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
      --output "$EVQ_OUTPUT" \
      --schedule evq \
      --loss-backend "$backend" \
      --microbatch "$microbatch" \
      --data-workers 8 \
      --compile \
      --compile-mode max-autotune-no-cudagraphs \
      --start-step 500 \
      --stop-step 1000 \
      --resume-from "$EVQ_OUTPUT/step-000500-full" \
      --full-save-steps "1000" \
      --verify-data-hashes
    ;;
  train-evq-1000-direct)
    require_cpu_receipt
    require_runtime_receipt
    require_geo_sentinel
    backend="$(selected_backend)"
    microbatch="$(selected_microbatch)"
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.train \
      train \
      --model-path "$ASSET_ROOT/step0" \
      --data-manifest "$DATA_MANIFEST" \
      --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
      --output "$EVQ_OUTPUT" \
      --schedule evq \
      --loss-backend "$backend" \
      --microbatch "$microbatch" \
      --data-workers 8 \
      --compile \
      --compile-mode max-autotune-no-cudagraphs \
      --stop-step 1000 \
      --full-save-steps "1000" \
      --reference-sentinel-log "$GEO_SENTINEL/train.jsonl" \
      --verify-data-hashes
    ;;
  train-evq)
    "$0" train-evq-500
    "$0" train-evq-1000
    ;;
  eval-geo)
    require_cpu_receipt
    mkdir -p "$EVAL_WORK"
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate \
      --base-model "$ASSET_ROOT/step0" \
      --checkpoint "$ASSET_ROOT/geo1000" \
      --data-manifest "$DATA_MANIFEST" \
      --schedule geo \
      --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
      --output "$EVAL_WORK/geo1000" \
      --compile \
      --compile-mode max-autotune-no-cudagraphs
    ;;
  eval-evq)
    require_cpu_receipt
    test -f "$EVQ_OUTPUT/step-001000-full/trainer_state.json"
    mkdir -p "$EVAL_WORK"
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate \
      --base-model "$ASSET_ROOT/step0" \
      --checkpoint "$EVQ_OUTPUT/step-001000-full" \
      --data-manifest "$DATA_MANIFEST" \
      --schedule evq \
      --eval-manifest "$EVAL_ROOT/eval_manifest.json" \
      --output "$EVAL_WORK/evq1000" \
      --compile \
      --compile-mode max-autotune-no-cudagraphs
    ;;
  compare-eval)
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.compare_eval \
      --geo "$EVAL_WORK/geo1000/results.json" \
      --evq "$EVAL_WORK/evq1000/results.json" \
      --output "$EVAL_WORK/comparison.json"
    ;;
  eval-retrieval-geo)
    require_cpu_receipt
    mkdir -p "$EVAL_WORK"
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_retrieval \
      --base-model "$ASSET_ROOT/step0" \
      --checkpoint "$ASSET_ROOT/geo1000" \
      --data-manifest "$DATA_MANIFEST" \
      --schedule geo \
      --retrieval-manifest "$RETRIEVAL_ROOT/retrieval_manifest.json" \
      --output "$EVAL_WORK/retrieval_geo1000" \
      --compile \
      --compile-mode max-autotune-no-cudagraphs
    ;;
  eval-retrieval-evq)
    require_cpu_receipt
    test -f "$EVQ_OUTPUT/step-001000-full/trainer_state.json"
    mkdir -p "$EVAL_WORK"
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_retrieval \
      --base-model "$ASSET_ROOT/step0" \
      --checkpoint "$EVQ_OUTPUT/step-001000-full" \
      --data-manifest "$DATA_MANIFEST" \
      --schedule evq \
      --retrieval-manifest "$RETRIEVAL_ROOT/retrieval_manifest.json" \
      --output "$EVAL_WORK/retrieval_evq1000" \
      --compile \
      --compile-mode max-autotune-no-cudagraphs
    ;;
  compare-retrieval)
    "$PYTHON_BIN" -m rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.compare_retrieval \
      --geo "$EVAL_WORK/retrieval_geo1000/results.json" \
      --evq "$EVAL_WORK/retrieval_evq1000/results.json" \
      --output "$EVAL_WORK/retrieval_comparison.json"
    ;;
  gpu-sequence)
    "$0" validate-assets
    require_cpu_receipt
    require_launch_storage
    run_probe native
    run_probe liger
    "$0" select-probe
    "$0" probe-microbatch8
    "$0" choose-runtime
    "$0" probe-sustained
    "$0" finalize-runtime
    "$0" geo-sentinel
    "$0" train-evq-1000-direct
    "$0" eval-geo
    "$0" eval-evq
    "$0" compare-eval
    "$0" eval-retrieval-geo
    "$0" eval-retrieval-evq
    "$0" compare-retrieval
    ;;
  formal-sequence)
    "$0" validate-assets
    require_cpu_receipt
    require_launch_storage
    require_runtime_receipt
    require_geo_sentinel
    "$0" train-evq-1000-direct
    "$0" eval-geo
    "$0" eval-evq
    "$0" compare-eval
    "$0" eval-retrieval-geo
    "$0" eval-retrieval-evq
    "$0" compare-retrieval
    ;;
  *)
    echo "usage: $0 {prepare-env|prepare-assets|validate-assets|prepare-data|prove-order|validate-data|prepare-eval|validate-eval|prepare-retrieval|validate-retrieval|portable-preflight|cpu-preflight|cpu-preflight-from-portable|probe-native|probe-liger|select-probe|probe-microbatch8|choose-runtime|probe-sustained|finalize-runtime|geo-sentinel|train-evq-500|train-evq-1000|train-evq-1000-direct|train-evq|eval-geo|eval-evq|compare-eval|eval-retrieval-geo|eval-retrieval-evq|compare-retrieval|gpu-sequence|formal-sequence}" >&2
    exit 2
    ;;
esac
