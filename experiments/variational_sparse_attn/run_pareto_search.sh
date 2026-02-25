#!/bin/bash
# Full Pareto Frontier Search on A100
# This runs multiple fine-tuning jobs to find the optimal sparse config

set -e

echo "============================================"
echo "Pareto Frontier Search on A100"
echo "============================================"

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This script is for CUDA GPUs only."
    echo "For M4 Mac, use: python search_pareto.py with smaller search space"
    exit 1
fi

# Configuration
MODEL="gpt2"  # Start with Small, can scale up
METHOD="lora"  # Recommended for grid search
EPOCHS=3
BATCH_SIZE=8
MAX_LENGTH=512
USE_AMP=true

# Search grid - finer grid for better Pareto coverage
# You can adjust these based on initial results
LAMBDAS=(0.0 0.001 0.005 0.01 0.02 0.05)
GAMMAS=(0.5 1.0 2.0 3.0 5.0 10.0)

echo ""
echo "Search Configuration:"
echo "  Model: $MODEL"
echo "  Method: $METHOD"
echo "  Epochs per config: $EPOCHS"
echo "  Lambda values: ${LAMBDAS[@]}"
echo "  Gamma values: ${GAMMAS[@]}"
echo "  Total configs: $((${#LAMBDAS[@]} * ${#GAMMAS[@]}))"
echo "  Est. time: ~$((${#LAMBDAS[@]} * ${#GAMMAS[@]} * EPOCHS * 15)) min (approx)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Run Pareto search
python search_pareto.py \
    --model $MODEL \
    --method $METHOD \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lambdas "${LAMBDAS[@]}" \
    --gammas "${GAMMAS[@]}" \
    --max_length $MAX_LENGTH \
    --use_amp \
    --output_dir outputs/pareto_search

echo ""
echo "Pareto search complete!"
echo "Check outputs/pareto_search/ for results and plots"
