#!/bin/bash
# Kill all GPU and training processes
pkill -f run_overnight 2>/dev/null
pkill -f run_llama 2>/dev/null
sleep 3

# Verify GPU is clear
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
    echo "Killing GPU process $pid"
    kill -9 $pid 2>/dev/null
done
sleep 2

# Verify clean
echo "GPU after cleanup:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Clean old results  
rm -rf /root/autodl-tmp/dfrope/hybrid-rope/results/overnight_8h
mkdir -p /root/autodl-tmp/dfrope/hybrid-rope/results/overnight_8h

# Launch via pure shell nohup (no Python wrapper that imports torch!)
cd /root/autodl-tmp/dfrope/hybrid-rope
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup /root/miniconda3/bin/python -u 2026-02-22/scripts/run_overnight_8h.py \
    > results/overnight_8h/console.log 2>&1 &
echo "LAUNCHED PID=$!"
