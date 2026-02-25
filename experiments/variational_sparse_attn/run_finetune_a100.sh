#!/bin/bash
# Fine-tuning on A100 (40GB or 80GB)
# Can handle GPT-2 Large/XL with LoRA

set -e

echo "============================================"
echo "Fine-tuning Sparse Attention on A100"
echo "============================================"

# Detect GPU memory
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "Detected GPU Memory: ${GPU_MEM} MiB"
    
    if [ "$GPU_MEM" -gt 70000 ]; then
        echo "A100 80GB detected - can run full fine-tuning"
        MODEL="gpt2-large"
        METHOD="full"
        BATCH_SIZE=8
    elif [ "$GPU_MEM" -gt 40000 ]; then
        echo "A100 40GB detected - using LoRA for larger models"
        MODEL="gpt2-large"
        METHOD="lora"
        BATCH_SIZE=8
    else
        echo "Smaller GPU detected - using conservative settings"
        MODEL="gpt2"
        METHOD="lora"
        BATCH_SIZE=4
    fi
else
    echo "nvidia-smi not found, using conservative defaults"
    MODEL="gpt2"
    METHOD="lora"
    BATCH_SIZE=4
fi

# Configuration
EPOCHS=3
MAX_LENGTH=512
USE_AMP=true  # Enable mixed precision on CUDA

# Sparse attention config - these will be swept in Pareto search
LAMBDA=0.01
GAMMA=1.0

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Method: $METHOD"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Length: $MAX_LENGTH"
echo "  Lambda: $LAMBDA"
echo "  Gamma: $GAMMA"
echo "  AMP: $USE_AMP"
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
    --use_amp \
    --output_dir outputs/finetune_a100

echo ""
echo "Done! Check outputs/finetune_a100/ for results"
