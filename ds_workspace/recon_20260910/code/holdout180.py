#!/usr/bin/env python3
"""THE UNDERPOWERED-QUESTION EXPERIMENT: 180 held-out rows instead of 72.

WHY THIS EXISTS.  The 72-row held-out panel gave the plateau members
  b3_lo14  -0.00pp | turns_a1_b64 +4.40pp | wide_b4 +6.94pp   (pooled +3.78, t=+1.22)
against a selection panel that had said +12..14pp for all three.  That is enough
to kill the selection-panel reading but NOT enough to say whether a smaller
effect is real: the largest arm's SE is 4.36pp, so 72 rows cannot exclude
anything inside roughly +/-10pp.

72 rows is also not a number anyone chose -- it is what one prepared panel
happened to contain.  Three panels exist over the SAME six tasks the selection
panel never used (niah_single_2, niah_multikey_2, niah_multiquery, vt, fwe,
qa_1), and their union is 180 rows with zero id collisions after disambiguation.
Scaling the measured SE by 1/sqrt(N) gives ~2.8pp, i.e. t=2 resolves ~5.5pp.

WHAT IT MEASURES.  The deployed table and the three plateau members on those 180
rows, BM measured in-run as before (the union panel has no archived endpoint).
Four arms, roughly 13 min each on this card.

PRE-REGISTERED DECISION RULE -- written before any row of this panel was read:

  primary statistic  pooled delta = per-row mean over the three members of
                     (member - BM), with its paired SE.  Pooling is legitimate
                     because the hypothesis under test ("the plateau members
                     beat the deployed table") was stated before the pool.

    pooled delta >= +5pp AND t >= 2   -> the effect is real; the 72-row reading
                                         was underpowered, not negative.
    pooled delta <= +1.5pp with SE <= 3pp
                                      -> the effect is inside noise.  "The
                                         tuning does not generalise" becomes an
                                         established result rather than a
                                         failure to measure.
    otherwise                         -> still unresolved; report the interval
                                         and stop.  Do NOT re-run with more rows
                                         in the hope of moving it -- at that
                                         point the honest answer is the bound.

  secondary (reported, not decisive)  each member separately, and the member-vs-
                     member contrasts.  These are what distinguish "one effect
                     shared by all three" from "one member carrying it".

WHY 180 AND NOT MORE.  The three panels are what exists; building more would mean
generating fresh RULER prompts, which changes the task draw and breaks the
"same six tasks" comparability.  If 180 rows lands in the unresolved band, the
right move is to report the bound, not to shop for a sample size that decides.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = "/root/autodl-tmp/phase1_20260910"
UNION = "/root/autodl-tmp/olmo_fast_screen_20260908/prepared_holdout_union/screen.jsonl"

CHAIN = r'''#!/bin/bash
cd /root/autodl-tmp/phase1_20260910
export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
PANEL=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_holdout_union/screen.jsonl
EMPTY=/root/autodl-tmp/phase1_20260910/empty_archive
OUT=/root/autodl-tmp/phase1_20260910/holdout180
mkdir -p "$EMPTY" "$OUT"

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

# the deployed table first: everything is paired against it, and it is the only
# arm whose number is needed before the others mean anything
run h180_bm   h180_bm.log   $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --betas 1.0 --turns ""
run h180_b3   h180_b3.log   $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --betas 3.0 --turns ""
run h180_a1b64 h180_a1b64.log $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --wide-betas 1.0 --turns ""
run h180_b4wide h180_b4wide.log $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --wide-betas 4.0 --turns ""

echo "=== HOLDOUT180 COMPLETE $(date -Is) ==="
'''


def main():
    p = pathlib.Path(ROOT) / "chain_holdout180.sh"
    p.write_text(CHAIN)
    p.chmod(0o755)
    subprocess.run(["bash", "-lc",
                    f"cd {ROOT} && setsid nohup ./chain_holdout180.sh > chain_holdout180.log "
                    "2>&1 < /dev/null & disown ; sleep 5 ; "
                    "ps -eo pid,args | grep 'chain_holdout18[0]' | head -1"])
    print("armed chain_holdout180.sh: BM -> b3_lo14 -> a1_b64 -> wide_b4 on 180 held-out rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
