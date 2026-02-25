#!/bin/bash
# Fine-tuning on M4 Max (36GB)
# Suitable for GPT-2 Small/Medium

set -e

echo "============================================"
echo "Fine-tuning Sparse Attention on M4 Max"
echo "============================================"

# Configuration
MODEL="gpt2"  # or gpt2-medium for ~6GB model
METHOD="lora"  # LoRA saves memory
EPOCHS=3
BATCH_SIZE=2  # Smaller for M4
MAX_LENGTH=256  # Shorter sequences

# Sparse attention config
LAMBDA=0.01
GAMMA=1.0

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Method: $METHOD"
echo "  Epochs: $EPOCHS"
echo "  Lambda: $LAMBDA"
echo "  Gamma: $GAMMA"
echo ""

# Run fine-tuning
python finetune_sparse.py \
    --model $MODEL \
    --method $METHOD \
    --variant prior_sparse \
    --lam $LAMBDA \
    --gamma $GAMMA \
    --prior_mode centered \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --lr 5e-5 \
    --output_dir outputs/finetune_m4

echo ""
echo "Done! Check outputs/finetune_m4/ for results"
