#!/usr/bin/env python3
"""Arm the out-of-sample test of pro_step42 on the 180 held-out rows.

RUN ON THE SERVER.  Governed by STEP42_PREREG_20260911.md, committed before any
row was read.

WHY.  step42 is the first table in this campaign to beat the deployed BM
significantly in BOTH readings on the 350-row selection panel:

    fractional +7.26pp (t=+3.39)   whole-row +6.00pp (t=+2.48)

But +32 of those points come from `niah_single_3` alone -- the very task the
earlier held-out test identified as the source of the plateau members' illusory
+12..14pp.  That makes "same selection effect" the strong prior, and the only way
to settle it is a panel of tasks the selection panel never used.

The comparison table already exists on that panel (`holdout180/beta_b1p0.jsonl`),
so this is ONE arm, about 8 minutes.

Queues behind chain_walk to avoid OOM: three chains already own the card.
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
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
PANEL=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_holdout_union/screen.jsonl
EMPTY=/root/autodl-tmp/phase1_20260910/empty_archive
OUT=/root/autodl-tmp/phase1_20260910/s42_out
mkdir -p "$EMPTY" "$OUT"

# condition on the FILE, not pgrep: a pgrep pattern also matches this command
# line and spins forever (lesson 4b, hit once already today)
until [ "$(cat walk_out/walk_a1.jsonl 2>/dev/null | wc -l)" -ge 180 ]; do sleep 30; done
sleep 20

echo "=== STAGE s42_h START $(date -Is) ==="
"$PY" olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" \
    --betas "" --turns "" --pro-tables step42 > s42_h.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "=== STAGE s42_h FAILED rc=$rc $(date -Is) ==="; tail -25 s42_h.log
else
  echo "=== STAGE s42_h OK $(date -Is) ==="
fi
"$PY" step42_read.py 2>&1 | tee step42_read.txt
echo "=== S42 COMPLETE $(date -Is) ==="
'''


def main():
    p = pathlib.Path(ROOT) / "chain_step42.sh"
    p.write_text(CHAIN)
    p.chmod(0o755)
    chk = subprocess.run(["bash", "-lc",
                          f"bash -n {p} && test -s {p} && wc -l < {p}"],
                         capture_output=True, text=True)
    print("chain landed, lines:", chk.stdout.strip() or chk.stderr.strip())
    subprocess.run(["bash", "-lc",
                    f"cd {ROOT} && setsid nohup ./chain_step42.sh > chain_step42.log "
                    "2>&1 < /dev/null & disown ; sleep 5 ; "
                    "ps -eo pid,args | grep 'chain_step4[2]' | head -2"], check=False)
    print("armed chain_step42.sh: step42 on the 180 held-out rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
