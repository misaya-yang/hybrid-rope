#!/bin/bash
# Wait for (1) 500M data frozen and (2) GPU audit finished, then run the
# gated 500M chain (probe -> KL cache -> train). Fully detached; single GPU
# process discipline enforced by the waits.
set -u
B12=/root/autodl-tmp/claude_round12_20260906
LOG=$B12/runs/Z_CPT_500M/chain_wait.log
mkdir -p $B12/runs/Z_CPT_500M
echo "WAIT_START $(date -u +%FT%TZ)" >> $LOG
until [ -f $B12/data/cpt_500m/manifest.json ]; do sleep 120; done
echo "DATA_FROZEN $(date -u +%FT%TZ)" >> $LOG
until ! pgrep -f "track_a_eval\.py|driver_audit_olmo_chat" >/dev/null 2>&1; do
  sleep 120
done
sleep 30
echo "AUDIT_DONE_GPU_FREE $(date -u +%FT%TZ)" >> $LOG
bash $B12/code/chain_500m.sh >> $LOG 2>&1
echo "CHAIN_WAITER_DONE rc=$? $(date -u +%FT%TZ)" >> $LOG
