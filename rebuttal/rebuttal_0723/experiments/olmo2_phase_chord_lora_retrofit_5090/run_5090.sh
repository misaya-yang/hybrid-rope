#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
root="${PHASE_CHORD_LORA_ROOT:-/root/autodl-tmp/iclr_next_runs/phase_chord_lora_retrofit_20260822}"
code="${CODE_ROOT:-/root/autodl-tmp/hybrid-rope}"
python="${PYTHON:-/root/miniconda3/bin/python}"
asset_root="${PHASE_CHORD_ASSET_ROOT:-$root/assets}"
checkpoint="${CHECKPOINT:-$asset_root/checkpoint}"
checkpoint_ready="${CHECKPOINT_READY:-$asset_root/checkpoint_ready.json}"
frequency_manifest="${FREQUENCY_MANIFEST:-$asset_root/target_manifest.json}"
source_tensor="${SOURCE_TENSOR:-$asset_root/source_4k.pt}"
source_receipt="${SOURCE_TOKEN_RECEIPT:-$asset_root/source_4k_receipt.json}"
train_data="${TRAIN_DATA:-$root/train8k}"
validation_data_8k="${VALIDATION_DATA_8K:-$root/val8k}"
validation_data_16k="${VALIDATION_DATA_16K:-$root/val16k}"
ruler_data="${RULER_DATA:-$root/ruler_full13}"
ruler_source="${RULER_SOURCE:-/root/autodl-tmp/iclr_next_runs/far_pass_chord_20260821/sources/RULER}"
teacher="$root/teacher_native"
module="rebuttal.rebuttal_0723.experiments.olmo2_phase_chord_lora_retrofit_5090"

export PYTHONPATH="$code"
export TOKENIZERS_PARALLELISM=false

case "$action" in
  install-peft|check-peft|prepare-ruler|"") ;;
  *)
    echo "REVOKED: the frozen training views contain no input cue identifying the selected source block." >&2
    echo "Teacher, training, and evaluation actions are disabled before CUDA initialization." >&2
    exit 64
    ;;
esac

gpu_environment() {
  export CUDA_VISIBLE_DEVICES=0
  export OLMO2_PHASE_CHORD_LORA_GPU_AUTHORIZED=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCHINDUCTOR_CACHE_DIR="$root/torchinductor_cache"
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
}

arm_output() {
  printf '%s/%s' "$root" "$1"
}

common_args() {
  local arm="$1"
  printf '%s\n' \
    --arm "$arm" \
    --checkpoint "$checkpoint" \
    --checkpoint-ready-receipt "$checkpoint_ready" \
    --frequency-manifest "$frequency_manifest" \
    --source-tensor "$source_tensor" \
    --source-token-receipt "$source_receipt" \
    --train-data "$train_data" \
    --teacher-assets "$teacher" \
    --steps 300 --morph-steps 60 --rank 64 --alpha 128 \
    --micro-pairs "${MICRO_PAIRS:-1}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --learning-rate "${LEARNING_RATE:-5e-5}" \
    --warmup-steps "${WARMUP_STEPS:-20}" \
    --compile-mode "${COMPILE_MODE:-max-autotune-no-cudagraphs}" \
    --seed "${TRAIN_SEED:-20260822}"
}

preflight_arm() {
  local arm="$1"
  local run_dir smoke_dir
  run_dir="$(arm_output "$arm")"
  smoke_dir="$(arm_output "smoke_$arm")"
  mapfile -t args < <(common_args "$arm")
  CUDA_VISIBLE_DEVICES="" "$python" -m "$module.train_static_table_lora" \
    --mode preflight --ready-receipt "$root/ready_${arm}.json" \
    --output "$run_dir" "${args[@]}"
  CUDA_VISIBLE_DEVICES="" "$python" -m "$module.train_static_table_lora" \
    --mode preflight --ready-receipt "$root/ready_smoke_${arm}.json" \
    --output "$smoke_dir" "${args[@]}"
}

smoke_arm() {
  local arm="$1"
  local smoke_dir
  smoke_dir="$(arm_output "smoke_$arm")"
  gpu_environment
  mapfile -t args < <(common_args "$arm")
  "$python" -m "$module.train_static_table_lora" \
    --mode smoke --authorize \
    --ready-receipt "$root/ready_smoke_${arm}.json" \
    --output "$smoke_dir" "${args[@]}"
}

run_arm() {
  local arm="$1"
  local run_dir smoke_dir
  run_dir="$(arm_output "$arm")"
  smoke_dir="$(arm_output "smoke_$arm")"
  gpu_environment
  mapfile -t args < <(common_args "$arm")
  "$python" -m "$module.train_static_table_lora" \
    --mode run --authorize \
    --ready-receipt "$root/ready_${arm}.json" \
    --gpu-ready-receipt "$smoke_dir/results.json" \
    --output "$run_dir" "${args[@]}"
}

eval_arm() {
  local arm="$1"
  local run_dir
  run_dir="$(arm_output "$arm")"
  gpu_environment
  mapfile -t args < <(common_args "$arm")
  "$python" -m "$module.train_static_table_lora" \
    --mode eval-natural --authorize \
    --adapter-bundle "$run_dir/artifacts" \
    --validation-data-8k "$validation_data_8k" \
    --validation-data-16k "$validation_data_16k" \
    --eval-rows "${EVAL_ROWS:-64}" \
    --output "${run_dir}_natural_eval" "${args[@]}"
}

eval_ruler_arm() {
  local arm="$1"
  local run_dir
  run_dir="$(arm_output "$arm")"
  gpu_environment
  "$python" -m "$module.evaluate_static_peft_ruler" \
    --authorize --arm "$arm" \
    --checkpoint "$checkpoint" \
    --checkpoint-ready-receipt "$checkpoint_ready" \
    --frequency-manifest "$frequency_manifest" \
    --adapter-bundle "$run_dir/artifacts" \
    --data-root "$ruler_data" \
    --lengths 4096 8192 16384 --limit-per-cell 20 \
    --output "${run_dir}_ruler_full13"
}

case "$action" in
  install-peft)
    CUDA_VISIBLE_DEVICES="" "$python" -m pip install --no-input \
      "peft==${PEFT_VERSION:-0.20.0}" \
      "accelerate==${ACCELERATE_VERSION:-1.14.0}"
    ;;
  check-peft)
    CUDA_VISIBLE_DEVICES="" "$python" -c \
      'import accelerate, peft, torch, transformers; assert torch.__version__.startswith("2.8."); assert transformers.__version__.startswith("5.15."); assert peft.__version__ == "0.20.0"; assert accelerate.__version__ == "1.14.0"; assert not torch.cuda.is_initialized(); print(torch.__version__, transformers.__version__, peft.__version__, accelerate.__version__)'
    ;;
  prepare-ruler)
    CUDA_VISIBLE_DEVICES="" "$python" -m \
      rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_instruct_ruler_transfer \
      --ruler-root "$ruler_source" \
      --checkpoint "$checkpoint" \
      --output "$ruler_data" \
      --lengths 4096 8192 16384 \
      --samples-per-cell 20 \
      --seed 20260822
    ;;
  teacher)
    gpu_environment
    "$python" -m "$module.train_static_table_lora" \
      --mode teacher --authorize \
      --checkpoint "$checkpoint" \
      --checkpoint-ready-receipt "$checkpoint_ready" \
      --source-tensor "$source_tensor" \
      --source-token-receipt "$source_receipt" \
      --train-data "$train_data" \
      --teacher-batch-pairs "${TEACHER_BATCH_PAIRS:-16}" \
      --short-pool-rows 16 --short-nll-tokens 64 \
      --short-hidden-positions 8 \
      --output "$teacher"
    ;;
  preflight-native) preflight_arm native ;;
  smoke-native) smoke_arm native ;;
  run-native) run_arm native ;;
  eval-native) eval_arm native ;;
  eval-ruler-native) eval_ruler_arm native ;;
  preflight-evq) preflight_arm anchored_evq_cosh_tau_2 ;;
  smoke-evq) smoke_arm anchored_evq_cosh_tau_2 ;;
  run-evq) run_arm anchored_evq_cosh_tau_2 ;;
  eval-evq) eval_arm anchored_evq_cosh_tau_2 ;;
  eval-ruler-evq) eval_ruler_arm anchored_evq_cosh_tau_2 ;;
  preflight-phase) preflight_arm phase_chord_olmo_r0_lambda_0p1 ;;
  smoke-phase) smoke_arm phase_chord_olmo_r0_lambda_0p1 ;;
  run-phase) run_arm phase_chord_olmo_r0_lambda_0p1 ;;
  eval-phase) eval_arm phase_chord_olmo_r0_lambda_0p1 ;;
  eval-ruler-phase) eval_ruler_arm phase_chord_olmo_r0_lambda_0p1 ;;
  eval-ruler-2wiki)
    echo "RULER/2Wiki PLACEHOLDER: inspect all three natural-eval receipts first." >&2
    echo "No downstream evaluator is launched automatically." >&2
    exit 3
    ;;
  *)
    echo "usage: $0 {install-peft|check-peft|prepare-ruler|teacher|preflight-native|smoke-native|run-native|eval-native|eval-ruler-native|preflight-evq|smoke-evq|run-evq|eval-evq|eval-ruler-evq|preflight-phase|smoke-phase|run-phase|eval-phase|eval-ruler-phase|eval-ruler-2wiki}" >&2
    exit 2
    ;;
esac
