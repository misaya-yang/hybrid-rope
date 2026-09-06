#!/bin/bash
# Overnight queue after Phase 2 (7B Track A) exits. Order fixed 2026-09-06 by
# new success criterion (beat unfine-tuned YaRN SOTA at 4x):
#   1. Y2 canonical YaRN control at 1B + 7B (same Track A matrix)
#   2. Qwen2.5 external controls (1.5B/0.5B/7B-if-downloaded, single_evidence)
#   3. E1 cross cells + training-fit readouts + summary
set -u
B12=/root/autodl-tmp/claude_round12_20260906
echo "CHAIN_ON: waiting for driver_phase2 to exit $(date -u +%FT%TZ)"
while pgrep -f driver_phase2.sh > /dev/null; do sleep 60; done
echo "CHAIN_ON: phase2 driver exited $(date -u +%FT%TZ)"
sleep 30
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  sleep 120
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "CHAIN_ON: ABORT GPU_NOT_IDLE $(date -u +%FT%TZ)"; exit 2
  fi
fi
echo "CHAIN_ON: GPU idle, launching overnight queue $(date -u +%FT%TZ)"
cd $B12/code
bash driver_y2.sh
echo "CHAIN_ON: driver_y2 rc=$? $(date -u +%FT%TZ)"
bash driver_qwen.sh
echo "CHAIN_ON: driver_qwen rc=$? $(date -u +%FT%TZ)"
bash e1/driver_e1.sh
echo "CHAIN_ON: driver_e1 rc=$? $(date -u +%FT%TZ)"
echo "CHAIN_ON: OVERNIGHT_QUEUE_COMPLETE $(date -u +%FT%TZ)"
