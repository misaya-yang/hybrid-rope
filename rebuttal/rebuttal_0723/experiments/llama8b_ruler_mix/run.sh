#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
ROOT="${ROOT:-/root/autodl-tmp/llama8b_ruler_mix_20260726}"
CODE_ROOT="${CODE_ROOT:-$ROOT/code/hybrid-rope}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/evq_5090_eval_bundle/runtime/bin/python}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}"
MODEL_MANIFEST="${MODEL_MANIFEST:-/root/autodl-tmp/evq_5090_eval_bundle/manifests/model_manifest.json}"
TEMPORAL_DATASET="${TEMPORAL_DATASET:-/root/autodl-tmp/data/temporal_holdout_2026_v1}"
EVQ_PARENT="${EVQ_PARENT:-/root/autodl-tmp/evq_5090_eval_bundle/adapters/evq}"
NATIVE_PARENT="${NATIVE_PARENT:-/root/autodl-tmp/evq_5090_eval_bundle/adapters/geo}"
RULER_ROOT="${RULER_ROOT:-/root/autodl-tmp/data/seed42_capability_sources/RULER-38da79d}"
NLTK_DATA="${NLTK_DATA:-/root/autodl-tmp/data/seed42_capability_sources/nltk_data}"
LONGALPACA="${LONGALPACA:-/root/autodl-tmp/data/longalpaca_12k_paper_candidate/LongAlpaca-12k_raw.json}"
SOURCES="${SOURCES:-$ROOT/data/ruler_official_train_eval_s20420726}"
TRAINING_VIEW="${TRAINING_VIEW:-$ROOT/data/physical8k_ruler_mix_s20420726}"
READY="${READY:-$ROOT/receipts/physical8k_ruler_mix_ready.json}"
EVQ_OUTPUT="${EVQ_OUTPUT:-$ROOT/runs/evq_ruler_mix_s20420726}"
NATIVE_OUTPUT="${NATIVE_OUTPUT:-$ROOT/runs/native_ruler_mix_s20420726}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
TASKS_OVERRIDE="${TASKS_OVERRIDE:-}"
LENGTHS_OVERRIDE="${LENGTHS_OVERRIDE:-}"

export PYTHONPATH="$CODE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/cache/torchinductor_pro6000}"
mkdir -p "$ROOT/logs" "$ROOT/receipts" "$ROOT/runs" "$ROOT/cache"

case "$MICRO_BATCH_SIZE" in
  1|2|4) ;;
  *)
    echo "MICRO_BATCH_SIZE must be 1, 2, or 4" >&2
    exit 2
    ;;
esac
GRADIENT_ACCUMULATION_STEPS=$((8 / MICRO_BATCH_SIZE))

prepare_sources() {
  local scope="${1:-all}"
  "$PYTHON_BIN" -m \
    rebuttal.rebuttal_0723.experiments.llama8b_ruler_mix.prepare_ruler_sources \
    --ruler-root "$RULER_ROOT" \
    --tokenizer "$CHECKPOINT" \
    --nltk-data "$NLTK_DATA" \
    --output "$SOURCES" \
    --scope "$scope" \
    --workers 4
}

prepare_view() {
  "$PYTHON_BIN" -m \
    rebuttal.rebuttal_0723.experiments.llama8b_ruler_mix.prepare_training_view \
    --tokenizer "$CHECKPOINT" \
    --sources "$SOURCES" \
    --longalpaca-json "$LONGALPACA" \
    --output "$TRAINING_VIEW"
}

preflight() {
  "$PYTHON_BIN" -m \
    rebuttal.rebuttal_0723.experiments.llama8b_ruler_mix.preflight \
    --code-root "$CODE_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --model-manifest "$MODEL_MANIFEST" \
    --training-view "$TRAINING_VIEW" \
    --evq-parent "$EVQ_PARENT" \
    --native-parent "$NATIVE_PARENT" \
    --output "$READY"
}

probe_arm() {
  local method="$1"
  local parent="$2"
  local micro="$3"
  local accumulation=$((8 / micro))
  local result="$ROOT/receipts/${method}_mb${micro}_probe.json"
  "$PYTHON_BIN" -m \
    rebuttal.rebuttal_0723.experiments.llama8b_ruler_mix.train \
    --checkpoint "$CHECKPOINT" \
    --parent-adapter "$parent" \
    --training-view "$TRAINING_VIEW" \
    --ready-receipt "$READY" \
    --output "$result" \
    --method "$method" \
    --micro-batch-size "$micro" \
    --gradient-accumulation-steps "$accumulation" \
    --probe-only
}

train_arm() {
  local method="$1"
  local parent="$2"
  local destination="$3"
  shift 3
  local -a extra=("$@")
  "$PYTHON_BIN" -m \
    rebuttal.rebuttal_0723.experiments.llama8b_ruler_mix.train \
    --checkpoint "$CHECKPOINT" \
    --parent-adapter "$parent" \
    --training-view "$TRAINING_VIEW" \
    --ready-receipt "$READY" \
    --output "$destination" \
    --method "$method" \
    --epochs 3 \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning-rate 2e-5 \
    --warmup-ratio 0.05 \
    --compile-mode max-autotune-no-cudagraphs \
    "${extra[@]}"
}

eval_arm() {
  local method="$1"
  local adapter="$2"
  local destination="$3"
  local count="$4"
  local -a extra=()
  local -a adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args+=(--adapter "$adapter")
  fi
  if [[ -n "$TASKS_OVERRIDE" ]]; then
    read -r -a selected_tasks <<<"$TASKS_OVERRIDE"
    extra+=(--tasks "${selected_tasks[@]}")
  fi
  if [[ -n "$LENGTHS_OVERRIDE" ]]; then
    read -r -a selected_lengths <<<"$LENGTHS_OVERRIDE"
    extra+=(--lengths "${selected_lengths[@]}")
  fi
  "$PYTHON_BIN" -m \
    rebuttal.rebuttal_0723.experiments.llama8b_ruler_mix.evaluate \
    --checkpoint "$CHECKPOINT" \
    "${adapter_args[@]}" \
    --sources "$SOURCES" \
    --output "$destination" \
    --method "$method" \
    --limit-per-cell "$count" \
    "${extra[@]}"
}

eval_temporal_ppl() {
  local method="$1"
  local adapter="$2"
  local destination="$3"
  "$PYTHON_BIN" -m \
    rebuttal.rebuttal_0723.experiments.llama8b_ruler_mix.evaluate_temporal_ppl \
    --checkpoint "$CHECKPOINT" \
    --model-manifest "$MODEL_MANIFEST" \
    --dataset-root "$TEMPORAL_DATASET" \
    --adapter "$adapter" \
    --method "$method" \
    --output "$destination"
}

case "$ACTION" in
  prepare-sources) prepare_sources all ;;
  prepare-train-sources) prepare_sources train ;;
  prepare-eval-sources) prepare_sources eval ;;
  prepare-view) prepare_view ;;
  preflight) preflight ;;
  prepare-all)
    prepare_sources all
    prepare_view
    preflight
    ;;
  probe-evq-mb1) probe_arm evq_cosh "$EVQ_PARENT" 1 ;;
  probe-evq-mb2) probe_arm evq_cosh "$EVQ_PARENT" 2 ;;
  train-evq) train_arm evq_cosh "$EVQ_PARENT" "$EVQ_OUTPUT" ;;
  resume-evq)
    train_arm evq_cosh "$EVQ_PARENT" "$EVQ_OUTPUT" --resume
    ;;
  eval-evq-screen)
    eval_arm evq_cosh "$EVQ_OUTPUT" \
      "$ROOT/runs/eval_evq_ruler_mix_n20" 20
    ;;
  eval-evq-formal)
    eval_arm evq_cosh "$EVQ_OUTPUT" \
      "$ROOT/runs/eval_evq_ruler_mix_n100" 100
    ;;
  eval-evq-temporal-ppl)
    eval_temporal_ppl evq_cosh "$EVQ_OUTPUT" \
      "$ROOT/runs/temporal_ppl_native_base_vs_evq_ruler_mix.json"
    ;;
  train-native)
    train_arm native_geo "$NATIVE_PARENT" "$NATIVE_OUTPUT"
    ;;
  eval-native-screen)
    eval_arm native_geo "$NATIVE_OUTPUT" \
      "$ROOT/runs/eval_native_ruler_mix_n20" 20
    ;;
  eval-native-temporal-ppl)
    eval_temporal_ppl native_geo "$NATIVE_OUTPUT" \
      "$ROOT/runs/temporal_ppl_native_base_vs_native_ruler_mix.json"
    ;;
  eval-base-screen)
    eval_arm native_base "" \
      "$ROOT/runs/eval_native_base_n20" 20
    ;;
  *)
    echo "usage: $0 {prepare-sources|prepare-train-sources|prepare-eval-sources|prepare-view|preflight|prepare-all|probe-evq-mb1|probe-evq-mb2|train-evq|eval-evq-screen|eval-evq-formal|eval-evq-temporal-ppl|train-native|eval-native-screen|eval-native-temporal-ppl|eval-base-screen}" >&2
    exit 2
    ;;
esac
