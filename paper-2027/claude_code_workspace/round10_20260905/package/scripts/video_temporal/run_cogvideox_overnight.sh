#!/bin/bash
# CogVideoX-2B EVQ Fine-tuning Overnight Script
# Target: RTX 6000 Pro 96GB
#
# Usage:
#   bash run_cogvideox_overnight.sh               # full experiment (GEO + EVQ, 500 steps each)
#   bash run_cogvideox_overnight.sh pilot          # 5-step pilot to verify setup
#   bash run_cogvideox_overnight.sh download       # download model only
#   bash run_cogvideox_overnight.sh geo_only       # GEO baseline only
#   bash run_cogvideox_overnight.sh evq_only       # EVQ only

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL_ID="THUDM/CogVideoX-2b"
MODEL_LOCAL="models/cogvideox-2b"
SCRIPT="scripts/video_temporal/cogvideox_evq_finetune.py"
OUTPUT="results/cogvideox_evq"
CACHE="data/cogvideox_cache"

# ---- Environment ----
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "============================================"
echo "  CogVideoX-2B EVQ Experiment"
echo "  $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# ---- Download model if needed ----
download_model() {
    if [ -d "$MODEL_LOCAL" ]; then
        echo "[Download] Model already exists at $MODEL_LOCAL"
    else
        echo "[Download] Downloading CogVideoX-2B to $MODEL_LOCAL ..."
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
        --tau 1.2 \
        --num_steps 5 \
        --batch_size 1 \
        --grad_accum 1 \
        --output_dir "$OUTPUT/pilot" \
        --cache_dir "$CACHE" \
        --pilot \
        --no_compile
    echo "=== PILOT COMPLETE ==="
}

# ---- Full experiment ----
run_full() {
    echo ""
    echo "=== EXPERIMENT 1: GEO baseline + EVQ(tau=1.2), base=10000 ==="
    python3 "$SCRIPT" \
        --model_path "$MODEL_LOCAL" \
        --method both \
        --tau 1.2 \
        --theta_t 10000 \
        --num_steps 500 \
        --batch_size 1 \
        --grad_accum 4 \
        --lora_rank 16 \
        --lr 1e-4 \
        --output_dir "$OUTPUT" \
        --cache_dir "$CACHE" \
        --seed 42

    echo ""
    echo "=== EXPERIMENT 2: EVQ(tau=1.2), base=1000 ==="
    python3 "$SCRIPT" \
        --model_path "$MODEL_LOCAL" \
        --method evq \
        --tau 1.2 \
        --theta_t 1000 \
        --num_steps 500 \
        --batch_size 1 \
        --grad_accum 4 \
        --lora_rank 16 \
        --lr 1e-4 \
        --output_dir "$OUTPUT" \
        --cache_dir "$CACHE" \
        --seed 42 \
        --skip_eval

    echo ""
    echo "============================================"
    echo "  ALL EXPERIMENTS COMPLETE"
    echo "  $(date)"
    echo "  Results in: $OUTPUT/"
    echo "============================================"
}

run_geo_only() {
    python3 "$SCRIPT" \
        --model_path "$MODEL_LOCAL" \
        --method geo \
        --theta_t 10000 \
        --num_steps 500 \
        --batch_size 1 \
        --grad_accum 4 \
        --output_dir "$OUTPUT" \
        --cache_dir "$CACHE" \
        --seed 42
}

run_evq_only() {
    python3 "$SCRIPT" \
        --model_path "$MODEL_LOCAL" \
        --method evq \
        --tau 1.2 \
        --theta_t 10000 \
        --num_steps 500 \
        --batch_size 1 \
        --grad_accum 4 \
        --output_dir "$OUTPUT" \
        --cache_dir "$CACHE" \
        --seed 42
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
    geo_only)
        run_geo_only
        ;;
    evq_only)
        run_evq_only
        ;;
    full|*)
        run_full
        ;;
esac
