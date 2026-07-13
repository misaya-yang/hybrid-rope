#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

required=(
  PYTHON_BIN
  EVQ_REPAIR_MODEL
  EVQ_REPAIR_MODEL_MANIFEST
  EVQ_REPAIR_LONGALPACA_MANIFEST
  EVQ_REPAIR_PARENT_ADAPTER
  EVQ_REPAIR_FILLER_DIR
  EVQ_REPAIR_TEMPORAL_ROOT
  EVQ_REPAIR_CAPABILITY_DIR
  EVQ_REPAIR_WORK_DIR
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "missing required environment variable: $variable" >&2
    exit 2
  fi
done

DATA_DIR="$EVQ_REPAIR_WORK_DIR/data"
CHECKPOINT_DIR="$EVQ_REPAIR_WORK_DIR/checkpoints"
RESULT_DIR="$EVQ_REPAIR_WORK_DIR/results"
GPU_LOCK="$EVQ_REPAIR_WORK_DIR/seed42-retrieval-repair.gpu.lock"
R8_SELECTED_SEGMENT="${EVQ_REPAIR_R8_SELECTED_SEGMENT:-1}"
R16_SELECTED_SEGMENT="${EVQ_REPAIR_R16_SELECTED_SEGMENT:-1}"

case "$R8_SELECTED_SEGMENT:$R16_SELECTED_SEGMENT" in
  1:1|1:2|2:1|2:2) ;;
  *) echo "selected segments must each be 1 or 2" >&2; exit 2 ;;
esac

mkdir -p "$EVQ_REPAIR_WORK_DIR" "$CHECKPOINT_DIR" "$RESULT_DIR"

require_file() {
  [[ -f "$1" ]] || { echo "required file is missing: $1" >&2; exit 3; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "required directory is missing: $1" >&2; exit 3; }
}

require_static_inputs() {
  require_file "$PYTHON_BIN"
  require_dir "$EVQ_REPAIR_MODEL"
  require_file "$EVQ_REPAIR_MODEL_MANIFEST"
  require_file "$EVQ_REPAIR_LONGALPACA_MANIFEST"
  require_dir "$EVQ_REPAIR_PARENT_ADAPTER"
  require_dir "$EVQ_REPAIR_FILLER_DIR"
  require_dir "$EVQ_REPAIR_TEMPORAL_ROOT"
  require_dir "$EVQ_REPAIR_CAPABILITY_DIR"
}

require_gpu_lock() {
  command -v flock >/dev/null || { echo "flock is required" >&2; exit 4; }
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 4; }
  nvidia-smi -L >/dev/null
  "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch reports CUDA unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("the registered run requires BF16-capable CUDA")
print({"cuda": torch.cuda.get_device_name(0), "bf16": True})
PY
  exec 9>"$GPU_LOCK"
  flock -n 9 || { echo "another EVQ repair GPU command holds $GPU_LOCK" >&2; exit 4; }
}

checkpoint_path() {
  printf '%s/%s_segment%s' "$CHECKPOINT_DIR" "$1" "$2"
}

gate_path() {
  printf '%s/%s_segment%s_gate.json' "$RESULT_DIR" "$1" "$2"
}

check_gate() {
  local stage="$1" segment="$2" status="$3" adapter="$4" gate="$5"
  "$PYTHON_BIN" -m rebuttal.evq_seed42_retrieval_repair.evaluate check-gate \
    --gate "$gate" \
    --status "$status" \
    --stage "$stage" \
    --segment "$segment" \
    --checkpoint-adapter "$adapter" \
    --model-manifest "$EVQ_REPAIR_MODEL_MANIFEST" \
    --longalpaca-manifest "$EVQ_REPAIR_LONGALPACA_MANIFEST" \
    --data-dir "$DATA_DIR"
}

run_controlled_eval() {
  local stage="$1" split="$2" adapter="$3" kind="$4" output="$5"
  if [[ -f "$output" ]]; then
    echo "reuse completed controlled result: $output"
    return
  fi
  "$PYTHON_BIN" -u -m rebuttal.evq_seed42_retrieval_repair.evaluate controlled \
    --model-name "$EVQ_REPAIR_MODEL" \
    --model-manifest "$EVQ_REPAIR_MODEL_MANIFEST" \
    --longalpaca-manifest "$EVQ_REPAIR_LONGALPACA_MANIFEST" \
    --adapter-dir "$adapter" \
    --adapter-kind "$kind" \
    --data-dir "$DATA_DIR" \
    --stage "$stage" \
    --split "$split" \
    --output "$output"
}

run_passkey_eval() {
  local length="$1" adapter="$2" kind="$3" output="$4"
  if [[ -f "$output" ]]; then
    echo "reuse completed passkey result: $output"
    return
  fi
  "$PYTHON_BIN" -u -m rebuttal.evq_seed42_retrieval_repair.evaluate passkey \
    --model-name "$EVQ_REPAIR_MODEL" \
    --model-manifest "$EVQ_REPAIR_MODEL_MANIFEST" \
    --longalpaca-manifest "$EVQ_REPAIR_LONGALPACA_MANIFEST" \
    --adapter-dir "$adapter" \
    --adapter-kind "$kind" \
    --data-dir "$DATA_DIR" \
    --target-length "$length" \
    --output "$output"
}

run_temporal_eval() {
  local stage="$1" adapter="$2" kind="$3" max_packs="$4" output="$5"
  if [[ -f "$output" ]]; then
    echo "reuse completed temporal result: $output"
    return
  fi
  "$PYTHON_BIN" -u -m rebuttal.evq_seed42_retrieval_repair.evaluate temporal \
    --model-name "$EVQ_REPAIR_MODEL" \
    --model-manifest "$EVQ_REPAIR_MODEL_MANIFEST" \
    --longalpaca-manifest "$EVQ_REPAIR_LONGALPACA_MANIFEST" \
    --adapter-dir "$adapter" \
    --adapter-kind "$kind" \
    --data-dir "$DATA_DIR" \
    --stage "$stage" \
    --temporal-root "$EVQ_REPAIR_TEMPORAL_ROOT" \
    --max-packs-per-domain "$max_packs" \
    --output "$output"
}

run_capability_eval() {
  local length="$1" adapter="$2" output="$3"
  if [[ -f "$output" ]]; then
    echo "reuse completed capability result: $output"
    return
  fi
  "$PYTHON_BIN" -u -m rebuttal.evq_seed42_retrieval_repair.evaluate capability \
    --model-name "$EVQ_REPAIR_MODEL" \
    --model-manifest "$EVQ_REPAIR_MODEL_MANIFEST" \
    --longalpaca-manifest "$EVQ_REPAIR_LONGALPACA_MANIFEST" \
    --adapter-dir "$adapter" \
    --adapter-kind repair \
    --data-dir "$DATA_DIR" \
    --capability-dir "$EVQ_REPAIR_CAPABILITY_DIR" \
    --target-length "$length" \
    --mode full \
    --output "$output"
}

run_baseline() {
  local stage="$1" adapter kind
  case "$stage" in
    r8)
      adapter="$EVQ_REPAIR_PARENT_ADAPTER"
      kind="legacy"
      ;;
    r16)
      adapter="$(checkpoint_path r8 "$R8_SELECTED_SEGMENT")"
      check_gate r8 "$R8_SELECTED_SEGMENT" pass "$adapter" "$(gate_path r8 "$R8_SELECTED_SEGMENT")"
      kind="repair"
      ;;
    *) echo "baseline stage must be r8 or r16" >&2; exit 2 ;;
  esac
  require_gpu_lock
  run_controlled_eval "$stage" validation "$adapter" "$kind" "$RESULT_DIR/baseline_${stage}_controlled.json"
  run_temporal_eval "$stage" "$adapter" "$kind" 1 "$RESULT_DIR/baseline_${stage}_temporal.json"
}

run_training() {
  local stage="$1" segment="$2" parent gate=() output
  [[ "$segment" == "1" || "$segment" == "2" ]] || { echo "segment must be 1 or 2" >&2; exit 2; }
  output="$(checkpoint_path "$stage" "$segment")"
  [[ ! -e "$output" ]] || { echo "checkpoint output already exists: $output" >&2; exit 5; }
  case "$stage:$segment" in
    r8:1)
      parent="$EVQ_REPAIR_PARENT_ADAPTER"
      ;;
    r8:2)
      parent="$(checkpoint_path r8 1)"
      check_gate r8 1 rescue_allowed "$parent" "$(gate_path r8 1)"
      gate=(--gate "$(gate_path r8 1)")
      ;;
    r16:1)
      parent="$(checkpoint_path r8 "$R8_SELECTED_SEGMENT")"
      check_gate r8 "$R8_SELECTED_SEGMENT" pass "$parent" "$(gate_path r8 "$R8_SELECTED_SEGMENT")"
      gate=(--gate "$(gate_path r8 "$R8_SELECTED_SEGMENT")")
      ;;
    r16:2)
      parent="$(checkpoint_path r16 1)"
      check_gate r16 1 rescue_allowed "$parent" "$(gate_path r16 1)"
      gate=(--gate "$(gate_path r16 1)")
      ;;
    *) echo "unsupported stage/segment: $stage/$segment" >&2; exit 2 ;;
  esac
  require_gpu_lock
  "$PYTHON_BIN" -u -m rebuttal.evq_seed42_retrieval_repair.train \
    --model-name "$EVQ_REPAIR_MODEL" \
    --model-manifest "$EVQ_REPAIR_MODEL_MANIFEST" \
    --longalpaca-manifest "$EVQ_REPAIR_LONGALPACA_MANIFEST" \
    --parent-adapter "$parent" \
    --data-dir "$DATA_DIR" \
    --stage "$stage" \
    --segment "$segment" \
    --output-dir "$output" \
    "${gate[@]}"
}

run_stage_gate() {
  local stage="$1" segment="$2" checkpoint parent_controlled parent_temporal output_gate
  checkpoint="$(checkpoint_path "$stage" "$segment")"
  output_gate="$(gate_path "$stage" "$segment")"
  require_dir "$checkpoint"
  [[ ! -e "$output_gate" ]] || {
    echo "gate output already exists; refusing any repeated GPU evaluation: $output_gate" >&2
    exit 5
  }
  if [[ "$segment" == "1" ]]; then
    parent_controlled="$RESULT_DIR/baseline_${stage}_controlled.json"
    parent_temporal="$RESULT_DIR/baseline_${stage}_temporal.json"
  elif [[ "$segment" == "2" ]]; then
    parent_controlled="$RESULT_DIR/${stage}_segment1_controlled.json"
    parent_temporal="$RESULT_DIR/${stage}_segment1_temporal.json"
  else
    echo "segment must be 1 or 2" >&2
    exit 2
  fi
  require_file "$parent_controlled"
  require_file "$parent_temporal"
  require_gpu_lock
  run_controlled_eval "$stage" validation "$checkpoint" repair "$RESULT_DIR/${stage}_segment${segment}_controlled.json"
  if [[ "$stage" == "r8" ]]; then
    run_passkey_eval 8192 "$checkpoint" repair "$RESULT_DIR/${stage}_segment${segment}_passkey.json"
  else
    run_passkey_eval 16384 "$checkpoint" repair "$RESULT_DIR/${stage}_segment${segment}_passkey.json"
  fi
  run_temporal_eval "$stage" "$checkpoint" repair 1 "$RESULT_DIR/${stage}_segment${segment}_temporal.json"
  "$PYTHON_BIN" -m rebuttal.evq_seed42_retrieval_repair.evaluate gate \
    --stage "$stage" \
    --segment "$segment" \
    --parent-controlled "$parent_controlled" \
    --checkpoint-controlled "$RESULT_DIR/${stage}_segment${segment}_controlled.json" \
    --passkey-result "$RESULT_DIR/${stage}_segment${segment}_passkey.json" \
    --parent-temporal "$parent_temporal" \
    --checkpoint-temporal "$RESULT_DIR/${stage}_segment${segment}_temporal.json" \
    --output "$output_gate"
}

run_final() {
  local checkpoint r8_parent
  checkpoint="$(checkpoint_path r16 "$R16_SELECTED_SEGMENT")"
  check_gate r16 "$R16_SELECTED_SEGMENT" pass "$checkpoint" "$(gate_path r16 "$R16_SELECTED_SEGMENT")"
  r8_parent="$(checkpoint_path r8 "$R8_SELECTED_SEGMENT")"
  check_gate r8 "$R8_SELECTED_SEGMENT" pass "$r8_parent" "$(gate_path r8 "$R8_SELECTED_SEGMENT")"
  require_gpu_lock
  run_controlled_eval r8 test "$checkpoint" repair "$RESULT_DIR/final_test_r8.json"
  run_controlled_eval r16 test "$checkpoint" repair "$RESULT_DIR/final_test_r16.json"
  run_passkey_eval 32768 "$checkpoint" repair "$RESULT_DIR/final_passkey_32768.json"
  for length in 8192 16384 32768; do
    run_capability_eval "$length" "$checkpoint" "$RESULT_DIR/final_capability_${length}.json"
  done
  run_temporal_eval r16 "$r8_parent" repair 0 "$RESULT_DIR/final_parent_temporal_full_r16.json"
  run_temporal_eval r16 "$checkpoint" repair 0 "$RESULT_DIR/final_checkpoint_temporal_full_r16.json"
  echo "final artifacts complete; no paper claim or metric has been changed"
}

usage() {
  cat <<'EOF'
Usage:
  run_seed42.sh prepare
  run_seed42.sh preflight
  run_seed42.sh baseline r8
  run_seed42.sh train-r8 1|2
  run_seed42.sh gate-r8 1|2
  run_seed42.sh baseline r16
  run_seed42.sh train-r16 1|2
  run_seed42.sh gate-r16 1|2
  run_seed42.sh final

Each invocation performs only the named phase. A gate never launches training,
and training never launches the next stage.
EOF
}

require_static_inputs
command_name="${1:-help}"
shift || true
case "$command_name" in
  prepare)
    if [[ -d "$DATA_DIR" ]]; then
      "$PYTHON_BIN" -m rebuttal.evq_seed42_retrieval_repair.prepare_data \
        --output-dir "$DATA_DIR" --validate-only
    else
      "$PYTHON_BIN" -u -m rebuttal.evq_seed42_retrieval_repair.prepare_data \
        --tokenizer "$EVQ_REPAIR_MODEL" \
        --filler-dir "$EVQ_REPAIR_FILLER_DIR" \
        --output-dir "$DATA_DIR" \
        --seed 42
    fi
    ;;
  preflight)
    require_dir "$DATA_DIR"
    "$PYTHON_BIN" -m py_compile \
      "$REPO_ROOT/rebuttal/evq_seed42_retrieval_repair/protocol.py" \
      "$REPO_ROOT/rebuttal/evq_seed42_retrieval_repair/prepare_data.py" \
      "$REPO_ROOT/rebuttal/evq_seed42_retrieval_repair/train.py" \
      "$REPO_ROOT/rebuttal/evq_seed42_retrieval_repair/evaluate.py" \
      "$REPO_ROOT/experiments/lora_evq_v2/eval_official_yarn_capability.py" \
      "$REPO_ROOT/scripts/lib/rope/official_yarn.py"
    bash -n "$REPO_ROOT/rebuttal/evq_seed42_retrieval_repair/run_seed42.sh"
    "$PYTHON_BIN" -m pytest \
      "$REPO_ROOT/tests/test_evq_seed42_retrieval_repair.py" \
      "$REPO_ROOT/tests/test_frequency_adaptation_8b.py" \
      "$REPO_ROOT/tests/test_official_yarn_parity.py" \
      "$REPO_ROOT/tests/test_official_yarn_capability_eval.py" -q
    "$PYTHON_BIN" -m rebuttal.evq_seed42_retrieval_repair.prepare_data \
      --output-dir "$DATA_DIR" --validate-only
    "$PYTHON_BIN" -m rebuttal.evq_seed42_retrieval_repair.train \
      --model-name "$EVQ_REPAIR_MODEL" \
      --model-manifest "$EVQ_REPAIR_MODEL_MANIFEST" \
      --longalpaca-manifest "$EVQ_REPAIR_LONGALPACA_MANIFEST" \
      --parent-adapter "$EVQ_REPAIR_PARENT_ADAPTER" \
      --data-dir "$DATA_DIR" \
      --stage r8 --segment 1 \
      --output-dir "$EVQ_REPAIR_WORK_DIR/.dry-run-r8-unused" \
      --dry-run
    for stage in r8 r16; do
      "$PYTHON_BIN" -m rebuttal.evq_seed42_retrieval_repair.evaluate preflight \
        --model-name "$EVQ_REPAIR_MODEL" \
        --model-manifest "$EVQ_REPAIR_MODEL_MANIFEST" \
        --longalpaca-manifest "$EVQ_REPAIR_LONGALPACA_MANIFEST" \
        --adapter-dir "$EVQ_REPAIR_PARENT_ADAPTER" \
        --adapter-kind legacy \
        --data-dir "$DATA_DIR" \
        --stage "$stage" \
        --temporal-root "$EVQ_REPAIR_TEMPORAL_ROOT"
    done
    "$PYTHON_BIN" - <<'PY'
import os
from experiments.lora_evq_v2.eval_official_yarn_capability import load_capability_suite
manifest, rows = load_capability_suite(os.environ["EVQ_REPAIR_CAPABILITY_DIR"])
print({"capability_rows": len(rows), "manifest_rows": manifest["row_count"]})
PY
    ;;
  baseline)
    run_baseline "${1:?baseline requires r8 or r16}"
    ;;
  train-r8)
    run_training r8 "${1:?train-r8 requires segment 1 or 2}"
    ;;
  gate-r8)
    run_stage_gate r8 "${1:?gate-r8 requires segment 1 or 2}"
    ;;
  train-r16)
    run_training r16 "${1:?train-r16 requires segment 1 or 2}"
    ;;
  gate-r16)
    run_stage_gate r16 "${1:?gate-r16 requires segment 1 or 2}"
    ;;
  final)
    run_final
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
