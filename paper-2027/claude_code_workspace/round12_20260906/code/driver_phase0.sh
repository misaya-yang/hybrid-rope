#!/bin/bash
# Round 12 Phase 0 driver (2026-09-06). Sequential, single GPU process at a time.
# Each step logs START/EXIT to driver_phase0.log; exit codes to driver_phase0_summary.txt.
# Failure policy: preserve scene; wd dependents skip if ref-dump step failed; fork steps independent.
B12=/root/autodl-tmp/claude_round12_20260906
R11=/root/autodl-tmp/claude_round11_olmo_20260905
M1=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
M7=/root/autodl-tmp/models/OLMo-2-1124-7B-Instruct
PY=/root/miniconda3/bin/python
LOG=$B12/diag/driver_phase0.log
SUM=$B12/diag/driver_phase0_summary.txt
SID1=0ee492a7351f230cc7aac34a6970df2e00f7aa838fb5569e3fbd8cc9208a9a2d
SID2=50b88ec1b6454878c549aa1e5cd39cb8ebdde7979aeddb98d9bbd97350b7afa0
mkdir -p $B12/diag
cd $B12/code || exit 91

run_step() {
  local name=$1; shift
  echo "=== STEP $name START $(date -u +%FT%TZ)" >> $LOG
  "$@" >> $LOG 2>&1
  local rc=$?
  echo "=== STEP $name EXIT=$rc $(date -u +%FT%TZ)" >> $LOG
  echo "$name EXIT=$rc" >> $SUM
  return $rc
}

echo "PHASE0_START $(date -u +%FT%TZ)" >> $LOG
: > $SUM

# ---- P0a: content-fork E/B (synthesis §11.2), five systems ----------------
run_step fork_eb_T0 $PY diag_fork_eb.py --model $M1 --tables $B12/tables \
  --system-id T0 --arm N --views $R11/olmo_tasks/transport_views.jsonl \
  --family single_evidence --split validation --layouts compact near far \
  --out $B12/diag/fork_eb_T0

run_step fork_eb_Z0 $PY diag_fork_eb.py --model $M1 --tables $B12/tables \
  --system-id Z0 --arm Z --views $R11/olmo_tasks/transport_views.jsonl \
  --receipts $R11/out/z0/examples.jsonl \
  --family single_evidence --split validation --layouts compact near far \
  --out $B12/diag/fork_eb_Z0

run_step fork_eb_ZC $PY diag_fork_eb.py --model $M1 --tables $B12/tables \
  --system-id ZC --arm Z --adapter $R11/out/train/step_128 \
  --views $R11/olmo_tasks/transport_views.jsonl \
  --receipts $R11/out/task128/examples.jsonl \
  --family single_evidence --split validation --layouts compact near far \
  --out $B12/diag/fork_eb_ZC

run_step fork_eb_ZF $PY diag_fork_eb.py --model $M1 --tables $B12/tables \
  --system-id ZF --arm Z --adapter $R11/out_zf/train/step_128 \
  --views $R11/olmo_tasks/transport_views.jsonl \
  --receipts $R11/out_zf/task128/examples.jsonl \
  --family single_evidence --split validation --layouts compact near far \
  --out $B12/diag/fork_eb_ZF

run_step fork_eb_ON $PY diag_fork_eb.py --model $M1 --tables $B12/tables \
  --system-id ON --arm N --adapter $R11/out_on/train/step_128 \
  --views $R11/olmo_tasks/transport_views.jsonl \
  --receipts $R11/out_on/task128/examples.jsonl \
  --family single_evidence --split validation --layouts compact near far \
  --out $B12/diag/fork_eb_ON

# ---- P0b: write/logit decomposition (synthesis §11.1), T0 dumps reference --
if run_step wd_T0 $PY diag_write_decomp.py --mode system --system-id T0 --arm N \
  --model $M1 --tables $B12/tables --views $R11/olmo_tasks/transport_views.jsonl \
  --proofs $R11/olmo_tasks/source_proofs.jsonl --instances $SID1 $SID2 \
  --out $B12/diag/wd; then
  run_step wd_Z0 $PY diag_write_decomp.py --mode system --system-id Z0 --arm Z \
    --model $M1 --tables $B12/tables --views $R11/olmo_tasks/transport_views.jsonl \
    --proofs $R11/olmo_tasks/source_proofs.jsonl --instances $SID1 $SID2 \
    --ref-dump $B12/diag/wd/T0/ref_dump.npz --out $B12/diag/wd
  run_step wd_ZF $PY diag_write_decomp.py --mode system --system-id ZF --arm Z \
    --adapter $R11/out_zf/train/step_128 \
    --model $M1 --tables $B12/tables --views $R11/olmo_tasks/transport_views.jsonl \
    --proofs $R11/olmo_tasks/source_proofs.jsonl --instances $SID1 $SID2 \
    --ref-dump $B12/diag/wd/T0/ref_dump.npz --out $B12/diag/wd
  run_step wd_ON $PY diag_write_decomp.py --mode system --system-id ON --arm N \
    --adapter $R11/out_on/train/step_128 \
    --model $M1 --tables $B12/tables --views $R11/olmo_tasks/transport_views.jsonl \
    --proofs $R11/olmo_tasks/source_proofs.jsonl --instances $SID1 $SID2 \
    --ref-dump $B12/diag/wd/T0/ref_dump.npz --out $B12/diag/wd
  run_step wd_report $PY diag_write_decomp.py --mode report --out $B12/diag/wd
else
  echo "wd_T0 FAILED: dependents skipped, scene preserved" >> $SUM
fi

# ---- P2 gate: 4080/7B capability probe -------------------------------------
run_step probe $PY probe_capability.py --model-1b $M1 --model-7b $M7 \
  --out $B12/probe4080

echo "PHASE0_END $(date -u +%FT%TZ)" >> $LOG
