#!/bin/bash
# The gain sweep: is YaRN's analytic mscale value actually the optimum?
#
# WHY THIS IS A THEORY TEST AND NOT TUNING.  The gain is the dominant design face
# in this whole campaign -- on the 350-row panel the SAME table moves from 0.0371
# to 0.4190 when the gain goes from 1.0 to YaRN's value, i.e. +38.19pp (t=+15.05),
# while the entire frequency-allocation programme moves things by +-5pp.  And
# YaRN's value is not fitted, it is DERIVED:
#
#     mscale = 0.1 * ln(s) + 1,   s = L / L_train = 16384 / 4096 = 4
#            = 1.138629436111989
#
# So measuring the response around it tests an analytic derivation against data,
# which is exactly the goal's first item -- finding the optimal strategy rather
# than searching for it.  If the peak sits at 1.1386 the derivation is validated;
# if it sits elsewhere, the derivation is systematically off for this model.
#
# DESIGN.  1.0 and 1.1386 already exist in olmo_gain/.  Three new points bracket
# the peak: 1.05 and 1.10 map the steep rise from 1.0, and 1.20 checks for
# overshoot.  BM at all of them, 350 rows, ~1.2 h at full speed.
#
# QUEUES BEHIND THE `pro` GROUP so the card is not oversubscribed; the gain group
# owns the GPU after that and this runs at full speed.
cd /root/autodl-tmp/phase1_20260910
export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
NEWT=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl
ARCH=/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01
OUT=/root/autodl-tmp/phase1_20260910/olmo_gsweep
mkdir -p "$OUT"

# wait on the log, not on pgrep: a pgrep pattern would also match this command
# line and spin forever (lesson 4b, hit once today already)
for g in pro natural gain; do
  until ! pgrep -f "supervise\.sh $g" >/dev/null 2>&1; do sleep 30; done
done
sleep 15

tries=0
while :; do
  have=0
  [ -f "$OUT/gain_bm_g1p2.jsonl" ] && have=$(wc -l < "$OUT/gain_bm_g1p2.jsonl")
  if [ "$have" -ge 350 ]; then
    echo "=== GSWEEP COMPLETE $have/350 $(date -Is) ==="
    break
  fi
  tries=$((tries+1))
  if [ "$tries" -gt 6 ]; then echo "=== GSWEEP GAVE UP $(date -Is) ==="; break; fi
  echo "=== GSWEEP START try=$tries have=$have/350 $(date -Is) ==="
  "$PY" olmo_beta.py --root "$OUT" --model "$MODEL" --panel "$NEWT" \
      --archive "$ARCH" --betas "" --turns "" \
      --gain-tables bm --gains 1.05,1.10,1.20 >> gsweep.log 2>&1
  echo "=== GSWEEP EXIT rc=$? $(date -Is) ==="
  sleep 5
done
"$PY" gsweep_read.py 2>&1 | tee gsweep_read.txt
echo "=== GSWEEP DONE $(date -Is) ==="
