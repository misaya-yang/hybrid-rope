#!/bin/bash
# Wan2.1-T2V-1.3B EVQ Fine-tuning Overnight Script
# Target: RTX 6000 Pro 96GB
#
# Usage:
#   bash run_wan21_overnight.sh               # full experiment (GEO + EVQ, 500 steps each)
#   bash run_wan21_overnight.sh pilot          # 5-step pilot to verify setup
#   bash run_wan21_overnight.sh download       # download model only
#   bash run_wan21_overnight.sh tau_sweep      # τ sweep: 1.5, 2.0, 3.2
#
# Before running:
#   1. Set MODEL_LOCAL to your downloaded Wan2.1-T2V-1.3B-Diffusers path
#   2. Set DATA_DIR to your video directory (optional, will self-distill if empty)

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- User config: CHANGE THESE ----
MODEL_ID="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_LOCAL="models/wan21-t2v-1.3b"          # Set to your local path
DATA_DIR=""                                    # Set to video dir, or leave empty for self-distill
# ------------------------------------

SCRIPT="scripts/video_temporal/wan21_evq_finetune.py"
OUTPUT="results/wan21_evq"
CACHE="data/wan21_cache"

# ---- Environment ----
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "============================================"
echo "  Wan2.1-T2V-1.3B EVQ Experiment"
echo "  $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# ---- Download model if needed ----
download_model() {
    if [ -d "$MODEL_LOCAL" ]; then
        echo "[Download] Model already exists at $MODEL_LOCAL"
    else
        echo "[Download] Downloading Wan2.1-T2V-1.3B to $MODEL_LOCAL ..."
        pip install -q huggingface_hub --break-system-packages 2>/dev/null || true
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$MODEL_LOCAL',
                  ignore_patterns=['*.msgpack', '*.h5', '*.ot'])
print('Download complete.')
"
    fi
}

# ---- Install dependencies ----
install_deps() {
    echo "[Setup] Installing dependencies..."
    pip install -q diffusers transformers accelerate peft sentencepiece --break-system-packages 2>/dev/null || true
    pip install -q decord imageio imageio-ffmpeg --break-system-packages 2>/dev/null || true
}

# ---- Pilot: 5 steps to verify everything works ----
run_pilot() {
    echo ""
    echo "=== PILOT: 5 steps to verify setup ==="
    python3 "$SCRIPT" \
        --model_path "$MODEL_LOCAL" \
        --method both \
        --tau 3.2 \
        --num_steps 5 \
        --batch_size 1 \
        --grad_accum 1 \
        --output_dir "$OUTPUT/pilot" \
        --cache_dir "$CACHE/pilot" \
        --pilot \
        --no_compile
    echo "=== PILOT COMPLETE ==="
}

# ---- Full experiment: head-to-head GEO vs EVQ(τ=3.2) ----
run_full() {
    echo ""
    echo "=== EXPERIMENT: GEO vs EVQ(τ=3.2), base=10000 ==="
    echo "  Theory: τ*_DiT = 0.53 × K_t/√T = 0.53 × 22/√13 ≈ 3.23"
    echo "  Dead channels: 10/22 (45.5%) with base=10000"
    echo ""

    DATA_ARG=""
    if [ -n "$DATA_DIR" ]; then
        DATA_ARG="--data_dir $DATA_DIR"
    fi

    python3 "$SCRIPT" \
        --model_path "$MODEL_LOCAL" \
        --method both \
        --tau 3.2 \
        --theta_t 10000 \
        --num_steps 500 \
        --batch_size 1 \
        --grad_accum 4 \
        --lora_rank 16 \
        --lr 1e-4 \
        --output_dir "$OUTPUT" \
        --cache_dir "$CACHE" \
        --seed 42 \
        $DATA_ARG

    echo ""
    echo "============================================"
    echo "  EXPERIMENT COMPLETE"
    echo "  $(date)"
    echo "  Results in: $OUTPUT/"
    echo "============================================"
}

# ---- τ sweep: test multiple τ values ----
run_tau_sweep() {
    echo ""
    echo "=== τ SWEEP: testing τ = 1.5, 2.0, 3.2 ==="

    DATA_ARG=""
    if [ -n "$DATA_DIR" ]; then
        DATA_ARG="--data_dir $DATA_DIR"
    fi

    for TAU in 1.5 2.0 3.2; do
        echo ""
        echo "--- τ = $TAU ---"
        python3 "$SCRIPT" \
            --model_path "$MODEL_LOCAL" \
            --method both \
            --tau $TAU \
            --theta_t 10000 \
            --num_steps 500 \
            --batch_size 1 \
            --grad_accum 4 \
            --lora_rank 16 \
            --lr 1e-4 \
            --output_dir "$OUTPUT/tau_sweep_$TAU" \
            --cache_dir "$CACHE" \
            --seed 42 \
            --skip_eval \
            $DATA_ARG
    done

    echo ""
    echo "============================================"
    echo "  τ SWEEP COMPLETE"
    echo "  $(date)"
    echo "  Results in: $OUTPUT/tau_sweep_*/"
    echo "============================================"
}

# ---- Main ----
MODE="${1:-full}"

install_deps
download_model

case "$MODE" in
    pilot)
        run_pilot
        ;;
    download)
        echo "Model downloaded. Ready to run."
        ;;
    tau_sweep)
        run_tau_sweep
        ;;
    full|*)
        run_full
        ;;
esac
