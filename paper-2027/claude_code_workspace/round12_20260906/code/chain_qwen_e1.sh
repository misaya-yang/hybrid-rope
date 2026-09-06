#!/bin/bash
# Overnight queue v2 (2026-09-06, after user re-framing to matched 2x/4x):
#   1. Qwen2.5 external controls at their own 2x/4x (65536/131072, yarn 2/4)
#   2. E1 cross cells + training-fit readouts
# Launched only when the GPU is idle.
set -u
B12=/root/autodl-tmp/claude_round12_20260906
echo "CHAIN2_ON: start $(date -u +%FT%TZ)"
sleep 10
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  sleep 120
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "CHAIN2_ON: ABORT GPU_NOT_IDLE $(date -u +%FT%TZ)"; exit 2
  fi
fi
echo "CHAIN2_ON: GPU idle, launching queue $(date -u +%FT%TZ)"
cd $B12/code || exit 91
bash driver_qwen.sh
echo "CHAIN2_ON: driver_qwen rc=$? $(date -u +%FT%TZ)"
bash e1/driver_e1.sh
echo "CHAIN2_ON: driver_e1 rc=$? $(date -u +%FT%TZ)"
echo "CHAIN2_ON: QUEUE_COMPLETE $(date -u +%FT%TZ)"
