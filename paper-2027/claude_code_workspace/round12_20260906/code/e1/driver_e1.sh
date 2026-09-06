#!/bin/bash
# E1 orchestrator (Pro Unified Plan §6.2, <=6 GPU-h). GPU-exclusive: only
# start after Phase 2 (7B Track A) has exited. Stages:
#   1. cross cells W_ONxT_Z and W_ZF x T_0 (engine, validation 2048+16384)
#   2. fit readout ZF steps 0/32/64/96/128 on train-far + validation-far
#      (built-in diagonal contract check: ZF step128 val-far == 2/128, 1/32)
#   3. fit readout ON steps 0/32/128 (control arm; batch trajectory already
#      in out_on/train/training.jsonl)
# Missing native KL at steps 64/96 is recorded as such, not re-run here.
set -u
cd "$(dirname "$0")"
OUT=/root/autodl-tmp/claude_round12_20260906/e1
LOG=$OUT/driver_e1.log
mkdir -p $OUT
exec >> $LOG 2>&1
echo "=== E1 START $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
bash e1_cross_cells.sh
echo "cross rc=$?"
PY=/root/miniconda3/bin/python
$PY e1_fit_readout.py --arm ZF --steps 0 32 64 96 128 \
  --output $OUT/fit/ZF --max-seconds 14400
echo "fit_ZF rc=$?"
$PY e1_fit_readout.py --arm ON --steps 0 32 128 \
  --output $OUT/fit/ON --max-seconds 8600
echo "fit_ON rc=$?"
/root/miniconda3/bin/python e1_summarize.py
echo "=== E1 END $(date -u +%FT%TZ) E1_DRIVER_COMPLETE"
