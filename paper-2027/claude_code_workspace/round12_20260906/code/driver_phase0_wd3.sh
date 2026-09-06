#!/bin/bash
# Round 12 Phase 0b re-run #3 (2026-09-06) — §2.3 relative-frame fix.
# Bug fixed: §2.3 direct/inherited terms must use the RELATIVE operator
# M(j)=g^2 R((j-qpos) theta), not the absolute key-position rotation.
# Prior Z0/ZF/ON outputs were renamed to *_s23bug_20260906 and preserved.
# T0 is untouched: its ref_dump.npz is the reference, T0 has no §2.3 block
# (ref is None for T0) and its §5.2/outer recon passed gates.
B12=/root/autodl-tmp/claude_round12_20260906
R11=/root/autodl-tmp/claude_round11_olmo_20260905
M1=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
PY=/root/miniconda3/bin/python
LOG=$B12/diag/driver_phase0_wd3.log
SUM=$B12/diag/driver_phase0_wd3_summary.txt
SID1=0ee492a7351f230cc7aac34a6970df2e00f7aa838fb5569e3fbd8cc9208a9a2d
SID2=50b88ec1b6454878c549aa1e5cd39cb8ebdde7979aeddb98d9bbd97350b7afa0
REF=$B12/diag/wd/T0/ref_dump.npz
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

echo "WD_RERUN3_START $(date -u +%FT%TZ)" >> $LOG
: > $SUM

if [ ! -f "$REF" ]; then
  echo "ABORT: T0 ref_dump.npz missing" | tee -a $LOG $SUM; exit 92
fi

run_step wd_Z0 $PY diag_write_decomp.py --mode system --system-id Z0 --arm Z \
  --model $M1 --tables $B12/tables --views $R11/olmo_tasks/transport_views.jsonl \
  --proofs $R11/olmo_tasks/source_proofs.jsonl --instances $SID1 $SID2 \
  --ref-dump $REF --out $B12/diag/wd
run_step wd_ZF $PY diag_write_decomp.py --mode system --system-id ZF --arm Z \
  --adapter $R11/out_zf/train/step_128 \
  --model $M1 --tables $B12/tables --views $R11/olmo_tasks/transport_views.jsonl \
  --proofs $R11/olmo_tasks/source_proofs.jsonl --instances $SID1 $SID2 \
  --ref-dump $REF --out $B12/diag/wd
run_step wd_ON $PY diag_write_decomp.py --mode system --system-id ON --arm N \
  --adapter $R11/out_on/train/step_128 \
  --model $M1 --tables $B12/tables --views $R11/olmo_tasks/transport_views.jsonl \
  --proofs $R11/olmo_tasks/source_proofs.jsonl --instances $SID1 $SID2 \
  --ref-dump $REF --out $B12/diag/wd
run_step wd_report $PY diag_write_decomp.py --mode report --out $B12/diag/wd

echo "WD_RERUN3_END $(date -u +%FT%TZ)" >> $LOG
