#!/bin/bash
# native x gain 2x2 on RULER, on TWO panels.
#
# WHY TWO PANELS.  The continuous instrument says the gain's NLL effect flips
# sign with the table at 4096:
#     native     gain 1.0 = 2.834894   gain YaRN = 2.984121  -> gain 1.0 better
#     beta_b1_BM gain 1.0 = 3.245447   gain YaRN = 2.955406  -> gain YaRN better
# and the task side so far only has the BM column (0.0371 vs 0.4190, +38.19pp,
# t=+15.05).  The native row is the interesting one: it is the only setting where
# the NLL ordering points AWAY from the deployed configuration.
#
#   * 350-row newtasks (all 16384): directly comparable to the existing BM
#     column.  But `native` does no extrapolation handling, so it may floor at
#     both gains and say nothing.
#   * holdout_union (60 rows @4096 + 120 @16384): the 4096 rows are where the NLL
#     comparison actually lives, and 4096 is inside the native window, so
#     `native` has real signal there.
cd /root/autodl-tmp/phase1_20260910
export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
ARCH=/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01
GAINS=1.0,1.138629436111989

while pgrep -f "chain_natur[a]l" >/dev/null 2>&1; do sleep 30; done
while pgrep -f "chain_wal[k]" >/dev/null 2>&1; do sleep 30; done
sleep 20

run () {
  local name="$1" root="$2" panel="$3" log="$4"
  echo "=== STAGE $name START $(date -Is) ==="
  "$PY" olmo_beta.py --root "$root" --model "$MODEL" --panel "$panel" \
      --archive "$ARCH" --betas "" --turns "" --gain-tables native \
      --gains "$GAINS" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== STAGE $name FAILED rc=$rc $(date -Is) ==="; tail -25 "$log"
  else
    echo "=== STAGE $name OK $(date -Is) ==="
  fi
}

mkdir -p olmo_ngain olmo_ngain_h
run ngain_newtasks /root/autodl-tmp/phase1_20260910/olmo_ngain \
    /root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl \
    olmo_ngain.log
run ngain_holdout /root/autodl-tmp/phase1_20260910/olmo_ngain_h \
    /root/autodl-tmp/olmo_fast_screen_20260908/prepared_holdout_union/screen.jsonl \
    olmo_ngain_h.log

echo "=== NGAIN COMPLETE $(date -Is) ==="
