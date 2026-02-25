#!/bin/bash
# Run Prior-guided Variational Sparse Attention Experiment
# =========================================================

set -e  # Exit on error

echo "============================================================"
echo "Prior-guided Variational Sparse Attention - Full Experiment"
echo "============================================================"
echo ""

# Configuration
OUTPUT_DIR="outputs/variational_sparse_attn"
MODEL="gpt2"
SEQ_LEN=1024
STRIDE=512
MAX_TOKENS=100000  # Reduce to 50000 for faster testing
LAM=8.0
ALPHA=1.5
SEED=42

# Gamma values for sweep
GAMMAS=(0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.0 3.0 5.0)

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Seq Length: $SEQ_LEN"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Lambda: $LAM"
echo "  Alpha: $ALPHA"
echo "  Gammas: ${GAMMAS[*]}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run main experiment
echo "Starting experiment..."
python main_experiment.py \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --seq_len $SEQ_LEN \
    --stride $STRIDE \
    --max_tokens $MAX_TOKENS \
    --lam $LAM \
    --alpha $ALPHA \
    --gammas "${GAMMAS[@]}" \
    --seed $SEED

echo ""
echo "============================================================"
echo "Experiment complete!"
echo "Check outputs in: $OUTPUT_DIR"
echo "============================================================"
