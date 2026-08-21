#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
root="${FAR_PASS_ROOT:-/root/autodl-tmp/iclr_next_runs/far_pass_chord_20260821}"
code="${CODE_ROOT:-/root/autodl-tmp/hybrid-rope}"
python="${PYTHON:-/root/miniconda3/bin/python}"
checkpoint="${CHECKPOINT:-/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct}"
checkpoint_ready="$root/checkpoint_ready.json"
training_view="$root/data/natural_span_continuous_8k_s20260821"
prepared="$root/prepared_continuous_8k_v1.json"
smoke="$root/smoke_continuous_8k_v1"
run="$root/run_continuous_8k_v1"
module="rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity"

export PYTHONPATH="$code"
export TOKENIZERS_PARALLELISM=false

gpu_environment() {
  export CUDA_VISIBLE_DEVICES=0
  export OLMO_FAR_PASS_CHORD_GPU_AUTHORIZED=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCHINDUCTOR_CACHE_DIR="$root/torchinductor_cache"
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
}

common_train_args() {
  printf '%s\n' \
    --training-mode continuous_8k \
    --steps 300 --micro-batch-size 2 \
    --gradient-accumulation-steps 4 \
    --projection-rank 64 --residual-pairs 8 \
    --initial-logit-gain 0.1 \
    --content-value-dim 32 \
    --content-projection-rank 64 \
    --initial-content-gain 0.1 \
    --first-token-loss-weight 0.5 \
    --learning-rate 5e-5 --warmup-steps 20 \
    --compile-mode max-autotune-no-cudagraphs \
    --validation-rows 16 --seed 20260821
}

run_mode() {
  local selected="$1"
  local output="$smoke"
  local ready=()
  if [[ "$selected" == train ]]; then
    output="$run"
    ready=(--gpu-ready-receipt "$smoke/gpu_ready.json")
  fi
  gpu_environment
  mapfile -t extra < <(common_train_args)
  "$python" -m "$module.train_4k_far_only_evq_residual" \
    --mode "$selected" --authorize \
    --checkpoint "$checkpoint" \
    --checkpoint-ready-receipt "$checkpoint_ready" \
    --training-view "$training_view" \
    --prepared-receipt "$prepared" \
    "${ready[@]}" --output "$output" "${extra[@]}"
}

case "$mode" in
  smoke|train) run_mode "$mode" ;;
  *)
    echo "usage: $0 {smoke|train}" >&2
    exit 2
    ;;
esac
