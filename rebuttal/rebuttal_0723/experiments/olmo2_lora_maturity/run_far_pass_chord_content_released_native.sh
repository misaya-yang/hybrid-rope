#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
root="${FAR_PASS_ROOT:-/root/autodl-tmp/iclr_next_runs/far_pass_chord_20260821}"
code="${CODE_ROOT:-/root/autodl-tmp/hybrid-rope}"
python="${PYTHON:-/root/miniconda3/bin/python}"
checkpoint="${CHECKPOINT:-/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct}"
checkpoint_ready="$root/checkpoint_ready.json"
training_view="$root/data/natural_span_phase_s20260821"
prepared="$root/prepared_content_v1.json"
smoke="$root/smoke_content_v1"
run="$root/run_content_v1"
core_eval="$root/data/ruler_eval_core4_v3_s20260822_n20"
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
    --steps 300 --micro-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --projection-rank 64 --residual-pairs 8 \
    --initial-logit-gain 0.1 \
    --content-value-dim 32 \
    --content-projection-rank 64 \
    --initial-content-gain 0.1 \
    --learning-rate 5e-5 --warmup-steps 20 \
    --compile-mode max-autotune-no-cudagraphs \
    --validation-rows 16 --seed 20260821
}

smoke_run() {
  gpu_environment
  mapfile -t extra < <(common_train_args)
  "$python" -m "$module.train_4k_far_only_evq_residual" \
    --mode smoke --authorize \
    --checkpoint "$checkpoint" \
    --checkpoint-ready-receipt "$checkpoint_ready" \
    --training-view "$training_view" \
    --prepared-receipt "$prepared" \
    --output "$smoke" "${extra[@]}"
}

train_run() {
  gpu_environment
  mapfile -t extra < <(common_train_args)
  "$python" -m "$module.train_4k_far_only_evq_residual" \
    --mode train --authorize \
    --checkpoint "$checkpoint" \
    --checkpoint-ready-receipt "$checkpoint_ready" \
    --training-view "$training_view" \
    --prepared-receipt "$prepared" \
    --gpu-ready-receipt "$smoke/gpu_ready.json" \
    --output "$run" "${extra[@]}"
}

eval_core4() {
  gpu_environment
  "$python" -m "$module.evaluate_instruct_ruler_transfer" \
    --checkpoint "$checkpoint" \
    --ready-receipt "$checkpoint_ready" \
    --data-root "$core_eval" \
    --output "$root/eval_core4_content_v1" \
    --frequency native \
    --adapter "$run/adapter.pt" \
    --adaptation far_pass_chord_residual \
    --lengths 8192 16384 \
    --limit-per-cell 20
}

case "$mode" in
  smoke) smoke_run ;;
  train) train_run ;;
  eval-core4) eval_core4 ;;
  *)
    echo "usage: $0 {smoke|train|eval-core4}" >&2
    exit 2
    ;;
esac
