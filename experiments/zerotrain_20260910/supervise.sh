#!/bin/bash
# Self-healing supervisor for the remaining stage list.
#
# WHY THIS EXISTS.  At 11:29:13 on 2026-09-11 every running runner was killed
# within 2 seconds of the others -- no OOM, no disk pressure, GPU healthy, and
# processes started earlier in the day survived.  From inside the container the
# cause is unattributable, and it idled the card for 43 minutes, which is the one
# thing the standing goal forbids.  The chain scripts had no retry, so a killed
# runner took its whole chain down silently: they wrote a STAGE line and nothing
# else, no FAILED, because the *script* died and not the runner.
#
# So: each stage now declares the row count its output file must reach.  The
# supervisor re-runs the stage until that count is met, up to a retry cap.  The
# runners already skip rows present in their jsonl, so a retry resumes rather than
# repeating work and costs nothing.
#
# USAGE:  ./supervise.sh <group>      group in {natural, pro, gain}
# Groups are disjoint so three supervisors can run concurrently inside 32 GB.

set -u
cd /root/autodl-tmp/phase1_20260910
export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
ARCH=/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01
NAT=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_natural_union/screen.jsonl
NEWT=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl
HOLD=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_holdout_union/screen.jsonl
EMPTY=/root/autodl-tmp/phase1_20260910/empty_archive
mkdir -p "$EMPTY"

# stage <name> <want_rows> <watch_file> <root> <panel> <archive> <extra args...>
stage () {
  local name="$1" want="$2" watch="$3" root="$4" panel="$5" arch="$6"; shift 6
  local tries=0
  while :; do
    local have=0
    [ -f "$watch" ] && have=$(wc -l < "$watch")
    if [ "$have" -ge "$want" ]; then
      echo "=== STAGE $name COMPLETE $have/$want $(date -Is) ==="
      return 0
    fi
    tries=$((tries+1))
    if [ "$tries" -gt 6 ]; then
      echo "=== STAGE $name GAVE UP after 6 tries, at $have/$want $(date -Is) ==="
      return 1
    fi
    echo "=== STAGE $name START try=$tries have=$have/$want $(date -Is) ==="
    "$PY" olmo_beta.py --root "$root" --model "$MODEL" --panel "$panel" \
        --archive "$arch" "$@" >> "sup_${name}.log" 2>&1
    local rc=$?
    echo "=== STAGE $name EXIT rc=$rc $(date -Is) ==="
    [ $rc -ne 0 ] && tail -4 "sup_${name}.log"
    sleep 5
  done
}

group="${1:-natural}"
echo "### supervisor group=$group pid=$$ start $(date -Is)"

case "$group" in
  natural)
    # 391 rows each.  walk_a0p5 is the out-of-sample confirmation of the walk's
    # interior point; the other four are the generalisation panel.
    stage nat_bm    391 natural_out/beta_b1p0.jsonl  natural_out "$NAT" "$EMPTY" --betas 1.0 --turns ""
    stage nat_b3    391 natural_out/beta_b3p0.jsonl  natural_out "$NAT" "$EMPTY" --betas 3.0 --turns ""
    stage nat_a1b64 391 natural_out/wide_b1p0.jsonl  natural_out "$NAT" "$EMPTY" --wide-betas 1.0 --turns ""
    stage nat_b4w   391 natural_out/wide_b4p0.jsonl  natural_out "$NAT" "$EMPTY" --wide-betas 4.0 --turns ""
    stage walkconf  391 natural_out/walk_a0p5.jsonl  natural_out "$NAT" "$EMPTY" --walk 0.5 --turns ""
    ;;
  pro)
    # step42 out of sample (180 rows), then the native x gain 2x2 on two panels.
    stage s42h       180 s42_out/pro_step42.jsonl   s42_out  "$HOLD" "$EMPTY" --pro-tables step42 --turns ""
    stage ngain_h    180 olmo_ngain_h/gain_native_g1p138629436111989.jsonl olmo_ngain_h "$HOLD" "$EMPTY" --gain-tables native --gains 1.0,1.138629436111989 --turns ""
    stage ngain_newt 350 olmo_ngain/gain_native_g1p138629436111989.jsonl   olmo_ngain   "$NEWT" "$ARCH" --gain-tables native --gains 1.0,1.138629436111989 --turns ""
    ;;
  gain)
    # Pro section 5's gain x table interaction: three tables at two gains.
    stage g2x2 350 olmo_gain2x2/gain_b3_g1p138629436111989.jsonl olmo_gain2x2 "$NEWT" "$ARCH" \
        --gain-tables a1_b64,mrpro,b3 --gains 1.0,1.138629436111989 --turns ""
    ;;
  *)
    echo "unknown group $group" >&2; exit 2
    ;;
esac
echo "### supervisor group=$group DONE $(date -Is)"
