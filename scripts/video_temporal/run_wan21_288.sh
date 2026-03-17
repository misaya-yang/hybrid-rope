#!/bin/bash
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/wan21/Wan2.1:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== System check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
python -c "from wan.modules.vae import WanVAE; print('wan OK')"

echo "=== Encode 288x288 ==="
python -u /root/autodl-tmp/wan21/workspace/wan21_prepare_data.py \
  --model_dir /root/autodl-tmp/wan21 \
  --video_dir /root/autodl-tmp/data/wan21_encoded/raw_videos \
  --output_dir /root/autodl-tmp/data/wan21_288 \
  --height 288 --width 288 --num_videos 20 --device cuda

python -c "
import torch
lat=torch.load('/root/autodl-tmp/data/wan21_288/latents.pt',weights_only=True)
txt=torch.load('/root/autodl-tmp/data/wan21_288/text_embeds.pt',weights_only=True)
print(f'latents:{lat.shape} text:{txt.shape}')
"
echo "=== Done ==="
