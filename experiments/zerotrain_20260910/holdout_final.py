#!/usr/bin/env python3
"""THE FINAL TEST: the champion on a HELD-OUT task set it was never selected on.

WHY.  The campaign's headline is +14.20pp over the deployed table on the 350-row
panel (b3_lo14, 80W/24L, t=+6.64).  But that panel has seven tasks and the entire
gain sits on ONE of them:

    niah_single_3  +46.0pp   (all other tasks +0.6 ... +14.5pp)

and every table in the campaign was selected by looking at that panel.  A result
that lives on one task of the panel used to select it is not yet a result.

`prepared_ruler_holdout_01` is 72 rows over SIX tasks the selection panel never
used -- niah_single_2, niah_multikey_2, niah_multiquery, vt, fwe, qa_1 -- with
capabilities 4096 and 16384.  This chain runs the deployed table and the three
champions on it, so the comparison is paired within the holdout and nothing in
it was ever read while choosing the tables.

WHAT IT DOES NOT CLAIM.  72 rows is few.  The point is not a tight confidence
interval; it is whether the DIRECTION survives on unseen tasks.  If the champion
loses or ties there, the +14.20pp was task-selection, and that is the finding.

The archive is deliberately NOT passed: the holdout has no archived endpoints, so
the deployed table is re-run here as `--betas 1.0` and the pairing is done
afterwards from the per-row jsonl.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = "/root/autodl-tmp/phase1_20260910"

CHAIN = r'''#!/bin/bash
cd /root/autodl-tmp/phase1_20260910
export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
HD=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_holdout_01/screen.jsonl
EMPTY=/root/autodl-tmp/phase1_20260910/empty_archive
mkdir -p "$EMPTY"

run () {
  local name="$1" log="$2"; shift 2
  echo "=== STAGE $name START $(date -Is) ==="
  "$@" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== STAGE $name FAILED rc=$rc $(date -Is) ==="; tail -25 "$log"
  else
    echo "=== STAGE $name OK $(date -Is) ==="
  fi
}

# wait for the current b4_wide RULER stage so the holdout does not start a fourth
# arm on a 32 GB card that also hosts the 131k-token Qwen job
while pgrep -f "olmo_beta.py --root /root/autodl-tmp/phase1_20260910/olmo_wide" >/dev/null 2>&1; do sleep 30; done
sleep 25

# the deployed table, re-run here because the holdout has no archived endpoint
run hold_bm hold_bm.log $PY olmo_beta.py --root "$ROOT/holdout" \
    --model "$M" --panel "$HD" --archive "$EMPTY" --betas 1.0 --turns ""

# the three champions, in the geometry each one was selected in
run hold_b3 hold_b3.log $PY olmo_beta.py --root "$ROOT/holdout" \
    --model "$M" --panel "$HD" --archive "$EMPTY" --betas 3.0 --turns ""
run hold_a1b64 hold_a1b64.log $PY olmo_beta.py --root "$ROOT/holdout" \
    --model "$M" --panel "$HD" --archive "$EMPTY" --wide-betas 1.0 --turns ""
run hold_b4wide hold_b4wide.log $PY olmo_beta.py --root "$ROOT/holdout" \
    --model "$M" --panel "$HD" --archive "$EMPTY" --wide-betas 4.0 --turns ""

echo "=== HOLDOUT COMPLETE $(date -Is) ==="
'''


def main():
    path = pathlib.Path(ROOT) / "chain_holdout.sh"
    path.write_text(CHAIN)
    path.chmod(0o755)
    subprocess.run(["bash", "-lc",
                    f"cd {ROOT} && setsid nohup ./chain_holdout.sh > chain_holdout.log "
                    "2>&1 < /dev/null & disown ; sleep 4 ; "
                    "ps -eo pid,args | grep 'chain_holdou[t]' | head -1"])
    print("armed chain_holdout.sh: BM -> b3_lo14 -> a1_b64 -> b4_wide on 72 held-out rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
