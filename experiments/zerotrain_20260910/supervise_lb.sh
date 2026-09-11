#!/bin/bash
# LongBridge signed pair, ported to OLMo.  Governed by LONGBRIDGE_PREREG_20260911.md.
#
# WHY.  The library's only real signed control is LongBridge: same slots, same
# magnitude, opposite sign on nu.  On Qwen (36 rows) the pair differs by
# +6.11pp at 128K with SE 8.14 -- t=+0.75, i.e. direction right, power absent.
# The other two "mirror controls" in the library are byte-identical to their
# tested arms, so they are not controls at all.
#
# The slots are not free: they are the slots whose effective period 2*pi*4^m/omega
# lies in [W, 4W].  That formula reproduces the historical Qwen choice [36,37,38,39]
# exactly, and gives [28,29,30,31] on OLMo.  The magnitude follows the ground
# truth's stated semantics, "public phase -1 rad at the eval length", i.e.
# delta = 1/(4W): 7.629e-06 on Qwen, 6.103515625e-05 on OLMo -- same relative
# size (+4..12% vs +6..14% per slot).
#
# A signed pair is the one design that stays valid at low SNR: the two arms
# differ only in sign, so any panel-level bias, selection effect or task
# difficulty cancels in the difference.
#
# Runs BOTH panels.  The 350-row panel gives the power; the 180-row held-out
# confirms, and its pair difference is uncontaminated by selection.
cd /root/autodl-tmp/phase1_20260910
export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
NEWT=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl
HOLD=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_holdout_union/screen.jsonl
ARCH=/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01
EMPTY=/root/autodl-tmp/phase1_20260910/empty_archive
SHIFT='6.103515625e-05:28,29,30,31;-6.103515625e-05:28,29,30,31'

# wait for the pro group by FILE condition (pgrep patterns self-match; lesson 4b)
until ! pgrep -f "supervise\.sh pro" >/dev/null 2>&1; do sleep 30; done
sleep 15

# stage <name> <want> <watch> <root> <panel> <archive>
stage () {
  local name="$1" want="$2" watch="$3" root="$4" panel="$5" arch="$6"
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
      echo "=== STAGE $name GAVE UP at $have/$want $(date -Is) ==="; return 1
    fi
    echo "=== STAGE $name START try=$tries have=$have/$want $(date -Is) ==="
    "$PY" olmo_beta.py --root "$root" --model "$MODEL" --panel "$panel" \
        --archive "$arch" --betas "" --turns "" --nu-shift "$SHIFT" \
        >> "sup_${name}.log" 2>&1
    echo "=== STAGE $name EXIT rc=$? $(date -Is) ==="
    sleep 5
  done
}

# both arms write to the same root; the LAST arm is the watch file
stage lb_newt 350 olmo_lb/nu_p6p104em05.jsonl olmo_lb "$NEWT" "$ARCH"
stage lb_hold 180 olmo_lb_h/nu_p6p104em05.jsonl olmo_lb_h "$HOLD" "$EMPTY"
"$PY" lb_read.py 2>&1 | tee lb_read.txt
echo "=== LB COMPLETE $(date -Is) ==="
