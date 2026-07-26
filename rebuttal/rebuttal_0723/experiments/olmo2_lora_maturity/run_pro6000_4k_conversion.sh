#!/usr/bin/env bash
set -euo pipefail

ASSET_ROOT="${ASSET_ROOT:-/root/autodl-tmp/olmo2_lora_maturity_assets}"
CODE_ROOT="${CODE_ROOT:-$ASSET_ROOT/code}"
MODEL="${MODEL:-$ASSET_ROOT/models/step30000_63B}"
RAW_ROOT="${RAW_ROOT:-$ASSET_ROOT/data/raw}"
PREPARED="${PREPARED:-$ASSET_ROOT/data/prepared_4k_conversion}"
BACKGROUND="${BACKGROUND:-$ASSET_ROOT/data/probe_background_16k}"
BINDING="${BINDING:-$ASSET_ROOT/data/binding_4k}"
CAUSAL="${CAUSAL:-$ASSET_ROOT/data/causal_16k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ASSET_ROOT/run_4k_conversion}"
READY="${READY:-$ASSET_ROOT/OLMO2_STEP30_4K_CONVERSION_READY.json}"
RULER_ROOT="${RULER_ROOT:-/root/autodl-tmp/olmo2_1b_evq_eval/ruler}"
PYTHON="${PYTHON:-/root/olmo2_venv/bin/python}"
SEED="${SEED:-20260725}"
HARD_END_EPOCH="${HARD_END_EPOCH:-1785030300}"

export PYTHONPATH="$CODE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="$ASSET_ROOT/torchinductor_cache"
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

module_name="rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity"

remaining_seconds() {
  local now
  now="$(date +%s)"
  echo $((HARD_END_EPOCH - now))
}

require_time() {
  local required="$1"
  local label="$2"
  local remaining
  remaining="$(remaining_seconds)"
  if (( remaining < required )); then
    echo "STOP: only ${remaining}s remain; ${label} requires ${required}s" >&2
    return 20
  fi
}

report() {
  if [[ -d "$OUTPUT_ROOT" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PYTHON" \
      -m "$module_name.summarize_4k_conversion" \
      --run-root "$OUTPUT_ROOT" \
      --output "$OUTPUT_ROOT/REPORT.md" || true
  fi
}

verify_ready_immutability() {
  CUDA_VISIBLE_DEVICES="" "$PYTHON" - "$READY" "$CODE_ROOT" "$MODEL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1]).resolve()
code_root = Path(sys.argv[2]).resolve()
model = Path(sys.argv[3]).resolve()
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if receipt.get("status") != "OLMO2_STEP30_4K_CONVERSION_READY":
    raise SystemExit("invalid READY status")
checkpoint = receipt["checkpoint"]
if Path(checkpoint["checkpoint_path"]).resolve() != model:
    raise SystemExit("READY checkpoint path drift")
for name, expected in checkpoint["files"].items():
    stat = (model / name).stat()
    if stat.st_size != int(expected["bytes"]) or stat.st_mtime_ns != int(expected["mtime_ns"]):
        raise SystemExit(f"checkpoint changed after READY: {name}")
for relative, expected in receipt["code"]["files"].items():
    digest = hashlib.sha256()
    with (code_root / relative).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit(f"code changed after READY: {relative}")
print("READY immutability verified")
PY
}

prepare_assets() {
  mkdir -p "$ASSET_ROOT/data"
  if [[ -d "$PREPARED" ]]; then
    CUDA_VISIBLE_DEVICES="" ionice -c 3 nice -n 19 "$PYTHON" \
      -m "$module_name.prepare_data" \
      --output-root "$PREPARED" \
      --verify-only
  else
    CUDA_VISIBLE_DEVICES="" ionice -c 3 nice -n 19 "$PYTHON" \
      -m "$module_name.prepare_data" \
      --checkpoint "$MODEL" \
      --longalign-jsonl "$RAW_ROOT/longalign/long.jsonl" \
      --tulu-parquet-dir "$RAW_ROOT/tulu3/data" \
      --output-root "$PREPARED"
  fi

  if [[ -d "$BACKGROUND" ]]; then
    CUDA_VISIBLE_DEVICES="" ionice -c 3 nice -n 19 "$PYTHON" \
      -m "$module_name.prepare_probe_background" \
      --output-dir "$BACKGROUND" \
      --verify-only
  else
    CUDA_VISIBLE_DEVICES="" ionice -c 3 nice -n 19 "$PYTHON" \
      -m "$module_name.prepare_probe_background" \
      --checkpoint "$MODEL" \
      --longalign-jsonl "$RAW_ROOT/longalign/long.jsonl" \
      --exclude-rows-jsonl "$PREPARED/longalign_paired_L4096/rows.jsonl" \
      --output-dir "$BACKGROUND" \
      --evaluation-rows 128
  fi

  if [[ -d "$BINDING" ]]; then
    CUDA_VISIBLE_DEVICES="" ionice -c 3 nice -n 19 "$PYTHON" \
      - "$BINDING" <<'PY'
import json
import sys
from pathlib import Path
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.preflight_4k_conversion import _verify_binding_collection
print(json.dumps(_verify_binding_collection(Path(sys.argv[1]).resolve()), indent=2))
PY
  else
    CUDA_VISIBLE_DEVICES="" ionice -c 3 nice -n 19 "$PYTHON" \
      -m "$module_name.prepare_4k_binding_data" \
      --checkpoint "$MODEL" \
      --background-dir "$BACKGROUND" \
      --output "$BINDING"
  fi

  if [[ -d "$CAUSAL" ]]; then
    CUDA_VISIBLE_DEVICES="" ionice -c 3 nice -n 19 "$PYTHON" \
      -m "$module_name.prepare_causal_data" \
      --output-root "$CAUSAL" \
      --verify-only
  else
    CUDA_VISIBLE_DEVICES="" ionice -c 3 nice -n 19 "$PYTHON" \
      -m "$module_name.prepare_causal_data" \
      --checkpoint "$MODEL" \
      --background-dir "$BACKGROUND" \
      --output-root "$CAUSAL" \
      --training-examples 16
  fi
}

preflight() {
  if [[ -e "$READY" ]]; then
    echo "READY receipt already exists: $READY" >&2
    return 20
  fi
  mkdir -p "$OUTPUT_ROOT"
  CUDA_VISIBLE_DEVICES="" ionice -c 3 nice -n 19 "$PYTHON" \
    -m "$module_name.preflight_4k_conversion" \
    --checkpoint "$MODEL" \
    --prepared-data "$PREPARED" \
    --background-dir "$BACKGROUND" \
    --binding-data "$BINDING" \
    --causal-data "$CAUSAL" \
    --code-root "$CODE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --receipt "$READY"
  verify_ready_immutability
}

run_gate() {
  local gate="$1"
  shift
  mkdir -p "$OUTPUT_ROOT/gates"
  "$PYTHON" -m "$module_name.gate_4k_conversion" \
    "$gate" "$@"
}

formal() {
  require_time 5400 "single conversion arm"
  verify_ready_immutability
  if [[ -e "$OUTPUT_ROOT/base_evq" ]]; then
    echo "formal output already exists; refusing an ambiguous rerun" >&2
    return 20
  fi
  mkdir -p "$OUTPUT_ROOT"
  trap report EXIT
  export CUDA_VISIBLE_DEVICES=0

  "$PYTHON" -m "$module_name.train_4k_stage_a" \
    --checkpoint "$MODEL" \
    --prepared-data "$PREPARED" \
    --background-dir "$BACKGROUND" \
    --ready-receipt "$READY" \
    --output "$OUTPUT_ROOT/base_evq" \
    --mode base \
    --frequency evq \
    --target-supervised-tokens 0 \
    --natural-eval-rows 16 \
    --natural-tail-tokens 1024

  "$PYTHON" -m "$module_name.train_4k_stage_a" \
    --checkpoint "$MODEL" \
    --prepared-data "$PREPARED" \
    --background-dir "$BACKGROUND" \
    --ready-receipt "$READY" \
    --output "$OUTPUT_ROOT/stage_a" \
    --mode train \
    --frequency evq \
    --target-supervised-tokens 20000000 \
    --micro-batch-size 8 \
    --gradient-accumulation-steps 1 \
    --rank 64 \
    --alpha 128 \
    --learning-rate 1e-4 \
    --warmup-ratio 0.05 \
    --compile-mode max-autotune-no-cudagraphs \
    --natural-eval-rows 16 \
    --natural-tail-tokens 1024 \
    --seed "$SEED"

  if ! run_gate stage-a \
    --base "$OUTPUT_ROOT/base_evq/results.json" \
    --candidate "$OUTPUT_ROOT/stage_a/results.json" \
    --output "$OUTPUT_ROOT/gates/stage_a.json"; then
    echo "STOP: Stage A failed its predeclared value gate" >&2
    return 20
  fi

  require_time 3600 "binding stages and 16K canary"
  "$PYTHON" -m "$module_name.train_4k_stage_b" \
    --checkpoint "$MODEL" \
    --parent-adapter "$OUTPUT_ROOT/stage_a/adapter.pt" \
    --prepared-data "$PREPARED" \
    --binding-data "$BINDING" \
    --background-dir "$BACKGROUND" \
    --ready-receipt "$READY" \
    --output "$OUTPUT_ROOT/stage_b1" \
    --stage b1 \
    --steps 100 \
    --micro-batch-size 8 \
    --gradient-accumulation-steps 1 \
    --rank 64 \
    --alpha 128 \
    --learning-rate 5e-5 \
    --warmup-steps 20 \
    --compile-mode max-autotune-no-cudagraphs \
    --seed "$SEED"

  if ! run_gate stage-b1 \
    --candidate "$OUTPUT_ROOT/stage_b1/results.json" \
    --output "$OUTPUT_ROOT/gates/stage_b1.json"; then
    echo "STOP: Stage B1 did not learn binding; Stage B2 is not admitted" >&2
    return 20
  fi

  "$PYTHON" -m "$module_name.train_4k_stage_b" \
    --checkpoint "$MODEL" \
    --parent-adapter "$OUTPUT_ROOT/stage_b1/adapter.pt" \
    --prepared-data "$PREPARED" \
    --binding-data "$BINDING" \
    --background-dir "$BACKGROUND" \
    --ready-receipt "$READY" \
    --output "$OUTPUT_ROOT/stage_b2" \
    --stage b2 \
    --steps 300 \
    --micro-batch-size 8 \
    --gradient-accumulation-steps 1 \
    --rank 64 \
    --alpha 128 \
    --learning-rate 5e-5 \
    --warmup-steps 20 \
    --compile-mode max-autotune-no-cudagraphs \
    --natural-eval-rows 16 \
    --seed "$SEED"

  if ! run_gate stage-b2 \
    --stage-a "$OUTPUT_ROOT/stage_a/results.json" \
    --candidate "$OUTPUT_ROOT/stage_b2/results.json" \
    --output "$OUTPUT_ROOT/gates/stage_b2.json"; then
    echo "STOP: Stage B2 failed held-out 4K binding/no-harm gate" >&2
    return 20
  fi

  "$PYTHON" -m "$module_name.evaluate_4k_conversion" \
    --checkpoint "$MODEL" \
    --ready-receipt "$READY" \
    --causal-data "$CAUSAL" \
    --output "$OUTPUT_ROOT/causal_base_canary" \
    --sets canary_full_heldout \
    --batch-size 1 \
    --seed "$SEED"

  "$PYTHON" -m "$module_name.evaluate_4k_conversion" \
    --checkpoint "$MODEL" \
    --adapter "$OUTPUT_ROOT/stage_b2/adapter.pt" \
    --ready-receipt "$READY" \
    --causal-data "$CAUSAL" \
    --output "$OUTPUT_ROOT/causal_stage_b2_canary" \
    --sets canary_full_heldout \
    --batch-size 1 \
    --seed "$SEED"

  if ! run_gate causal \
    --base "$OUTPUT_ROOT/causal_base_canary/results.json" \
    --candidate "$OUTPUT_ROOT/causal_stage_b2_canary/results.json" \
    --set canary_full_heldout \
    --output "$OUTPUT_ROOT/gates/causal_16k.json"; then
    echo "STOP: no 16K capability conversion; broad eval is not admitted" >&2
    return 20
  fi

  if require_time 1800 "full 16K causal evaluation"; then
    "$PYTHON" -m "$module_name.evaluate_4k_conversion" \
      --checkpoint "$MODEL" \
      --adapter "$OUTPUT_ROOT/stage_b2/adapter.pt" \
      --ready-receipt "$READY" \
      --causal-data "$CAUSAL" \
      --output "$OUTPUT_ROOT/causal_stage_b2_full" \
      --sets \
        train_position_train_template \
        train_position_eval_template \
        eval_position_train_template \
        eval_position_eval_template \
        eval_dense_positions \
      --batch-size 1 \
      --seed "$SEED"
  fi

  if [[ -d "$RULER_ROOT" ]] && require_time 3600 "RULER 3-cell evaluation"; then
    "$PYTHON" \
      -m rebuttal.rebuttal_0723.experiments.evaluate_ruler_lora \
      --checkpoint "$MODEL" \
      --adapter "$OUTPUT_ROOT/stage_b2/adapter.pt" \
      --ready-receipt "$READY" \
      --schedule evq \
      --data-root "$RULER_ROOT" \
      --output "$OUTPUT_ROOT/ruler_stage_b2" \
      --lengths 4096 8192 16384 \
      --limit-per-cell 100
  fi

  report
  trap - EXIT
}

wait_for_identity() {
  local pid="$1"
  local expected_start="$2"
  while [[ -r "/proc/$pid/stat" ]]; do
    local actual_start
    actual_start="$(awk '{print $22}' "/proc/$pid/stat")"
    if [[ "$actual_start" != "$expected_start" ]]; then
      return 0
    fi
    sleep 20
  done
}

wait_and_run() {
  local formal_pid="${CURRENT_FORMAL_PID:?CURRENT_FORMAL_PID is required}"
  local formal_start="${CURRENT_FORMAL_START:?CURRENT_FORMAL_START is required}"
  local watcher_pid="${CURRENT_WATCHER_PID:?CURRENT_WATCHER_PID is required}"
  local watcher_start="${CURRENT_WATCHER_START:?CURRENT_WATCHER_START is required}"
  local formal_cmdline
  local watcher_cmdline

  formal_cmdline="$(tr '\0' ' ' <"/proc/$formal_pid/cmdline")"
  watcher_cmdline="$(tr '\0' ' ' <"/proc/$watcher_pid/cmdline")"
  [[ "$formal_cmdline" == *"run_pro6000.sh formal-sequence"* ]]
  [[ "$watcher_cmdline" == *"while kill -0 $formal_pid"* ]]
  [[ "$(awk '{print $22}' "/proc/$formal_pid/stat")" == "$formal_start" ]]
  [[ "$(awk '{print $22}' "/proc/$watcher_pid/stat")" == "$watcher_start" ]]

  wait_for_identity "$formal_pid" "$formal_start"
  wait_for_identity "$watcher_pid" "$watcher_start"

  test -f /root/autodl-tmp/olmo2_1b_evq_run/evq_1000/step-001000-full/trainer_state.json
  test -f /root/autodl-tmp/olmo2_1b_evq_run/evaluation/evq1000/results.json
  test -f /root/autodl-tmp/olmo2_1b_evq_run/evaluation/retrieval_evq1000/results.json
  verify_ready_immutability

  local gpu_pids
  gpu_pids="$(
    nvidia-smi --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
  )"
  if [[ -n "$gpu_pids" ]]; then
    echo "STOP: GPU still has compute process(es): $gpu_pids" >&2
    return 20
  fi
  formal
}

case "${1:-}" in
  prepare-assets)
    prepare_assets
    ;;
  preflight)
    preflight
    ;;
  formal)
    formal
    ;;
  report)
    report
    ;;
  wait-and-run)
    wait_and_run
    ;;
  *)
    echo "usage: $0 {prepare-assets|preflight|formal|report|wait-and-run}" >&2
    exit 2
    ;;
esac
