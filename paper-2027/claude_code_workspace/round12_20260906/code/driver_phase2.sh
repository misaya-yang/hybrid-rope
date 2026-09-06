#!/bin/bash
# Round 12 Phase 2 driver — 7B Track A, gated by probe_result.json.
# Precondition (checked here): probe stage2 + stage3(16K) both PASS.
# Structure: arm-N qualification (single_evidence 2048, 64 rows/cell) first;
# only on success run the full {N,Z,Y,M} matrix.
B12=/root/autodl-tmp/claude_round12_20260906
M7=/root/autodl-tmp/models/OLMo-2-1124-7B-Instruct
PY=/root/miniconda3/bin/python
LOG=$B12/track_a/driver_phase2.log
SUM=$B12/track_a/driver_phase2_summary.txt
mkdir -p $B12/track_a
cd $B12/code || exit 91

echo "PHASE2_START $(date -u +%FT%TZ)" >> $LOG
: > $SUM

# ---- gate: probe verdict ----------------------------------------------------
PROBE=$B12/probe4080/probe_result.json
if [ ! -f $PROBE ]; then
  echo "GATE_FAIL probe_result.json missing" | tee -a $LOG $SUM; exit 2
fi
GATE=$($PY - <<'EOF'
import json
r = json.load(open("/root/autodl-tmp/claude_round12_20260906/probe4080/probe_result.json"))
s2 = r.get("stage2_7b_load", {})
s3 = r.get("stage3_7b_forward", {})
ok2 = s2.get("to_cuda", {}).get("status") == "PASS" if isinstance(s2.get("to_cuda"), dict) else False
ok3 = ok2 and s3.get("forward_16384", {}).get("status") == "PASS"
print("PASS" if (ok2 and ok3) else "FAIL")
EOF
)
echo "GATE=$GATE" | tee -a $LOG $SUM
if [ "$GATE" != "PASS" ]; then
  echo "PHASE2_SKIPPED (probe gate); boundary recorded in probe_result.json" >> $LOG
  exit 0
fi

run_step() {
  local name=$1; shift
  echo "=== STEP $name START $(date -u +%FT%TZ)" >> $LOG
  "$@" >> $LOG 2>&1
  local rc=$?
  echo "=== STEP $name EXIT=$rc $(date -u +%FT%TZ)" >> $LOG
  echo "$name EXIT=$rc" >> $SUM
  return $rc
}

# ---- qualification: arm N, single_evidence compact 2048, 64 rows/cell -------
if run_step qualif_N $PY track_a_eval.py --model $M7 --model-id olmo7b \
  --tables $B12/tables --arm N --tasks $B12/tasks/round12_tasks.jsonl \
  --families single_evidence --lengths 2048 --max-rows-per-cell 64 \
  --output $B12/track_a/olmo7b_N_qualif; then
  for A in N Z Y M; do
    run_step track_a_7b_$A $PY track_a_eval.py --model $M7 --model-id olmo7b \
      --tables $B12/tables --arm $A --tasks $B12/tasks/round12_tasks.jsonl \
      --output $B12/track_a/olmo7b_$A
  done
else
  echo "QUALIF_FAILED: full 7B matrix not started; scene preserved" >> $SUM
fi

echo "PHASE2_END $(date -u +%FT%TZ)" >> $LOG
