#!/bin/bash
# Out-of-sample confirmation of the walk's interior point.
#
# WHY.  The walk (180-row panel) returns PURE TRADE -- every a>0 loses in-window
# and gains long range -- but the LINEARITY null is rejected (chi2/dof 2.50 at
# 16384, max residual 6.54pp).  The long-range gain saturates almost immediately
# (+9.82 at a=0.25, best +10.87 at a=0.50) while the in-window cost keeps
# accruing, so a=0.50 dominates BOTH endpoints:
#
#     a=0.50   d4096 = -2.69pp (t=-1.51, not significant)   d16384 = +10.87pp (t=+3.28)
#
# But that was located on rows already used to choose the plateau members, which
# is precisely the shape of the +12..14pp illusion the held-out panel destroyed.
# WALK_PREREG section 6 says a located interior point must reproduce on data this
# segment was not fitted on before it can be called a result.
#
# This runs walk_a0p5 -- a NEW table -- on prepared_natural_union: 391 rows, five
# natural reading-comprehension families, never run in this campaign, zero
# overlap with the selection panel.  The BM arm (nat_bm) comes from
# chain_natural.  Prediction, recorded in the pre-registration: +1..+4pp, i.e.
# it will NOT clear the +5pp bar, because 346 of the 391 rows are single-reference
# and the binarized reading of this interior point is only +3.33pp (t=1.07).
#
# Does NOT wait: the card has ~25 GB free while only chain_pro runs, so this goes
# now rather than queueing behind chain_natural.
cd /root/autodl-tmp/phase1_20260910
export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
PANEL=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_natural_union/screen.jsonl
EMPTY=/root/autodl-tmp/phase1_20260910/empty_archive
OUT=/root/autodl-tmp/phase1_20260910/natural_out
mkdir -p "$EMPTY" "$OUT"

echo "=== STAGE walk_a0p5 START $(date -Is) ==="
"$PY" olmo_beta.py --root "$OUT" --model "$MODEL" --panel "$PANEL" \
    --archive "$EMPTY" --betas "" --turns "" --walk 0.5 > walkconf.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "=== STAGE walk_a0p5 FAILED rc=$rc $(date -Is) ==="; tail -25 walkconf.log
else
  echo "=== STAGE walk_a0p5 OK $(date -Is) ==="
fi
"$PY" walkconf_read.py 2>&1 | tee walkconf_read.txt
echo "=== WALKCONF COMPLETE $(date -Is) ==="
