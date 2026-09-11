#!/usr/bin/env python3
"""Arm the frontier walk: five doses between the deployed table and a1_b64.

RUN ON THE SERVER.  Governed by WALK_PREREG_20260911.md (git 82dc5e9), committed
before any row of this run was read.

WHY IT QUEUES BEHIND chain_s8.  The card is 32 GB and already runs three
processes at 100% utilisation; a fourth would not add throughput, it would only
add OOM risk on the 32768 rows chain_s8 is pushing.  Queueing costs nothing.

All five arms run in ONE process so the whole curve is measured under identical
conditions.  a=0 and a=1 rebuild tables that already exist as `beta_b1p0` and
`wide_b1p0`; walk_read.py checks them row-by-row against those files and refuses
to interpret the curve if they differ (that check is what makes the interior
points comparable to the stored endpoints).
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
OUT=/root/autodl-tmp/phase1_20260910/walk_out
mkdir -p "$EMPTY" "$OUT"

# wait for the 32768 run: the card is already at 100% and a fourth process
# would only add OOM risk on the long rows
while pgrep -f "chain_s[8]" >/dev/null 2>&1; do sleep 30; done
sleep 20

echo "=== STAGE walk START $(date -Is) ==="
"$PY" olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" \
    --betas "" --turns "" --walk 0,0.25,0.5,0.75,1 > walk.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "=== STAGE walk FAILED rc=$rc $(date -Is) ==="; tail -25 walk.log
else
  echo "=== STAGE walk OK $(date -Is) ==="
fi
"$PY" walk_read.py 2>&1 | tee walk_read.txt
echo "=== WALK COMPLETE $(date -Is) ==="
'''


def main():
    p = pathlib.Path(ROOT) / "chain_walk.sh"
    p.write_text(CHAIN)
    p.chmod(0o755)
    chk = subprocess.run(["bash", "-lc", f"test -s {p} && wc -l < {p}"],
                         capture_output=True, text=True)
    print("script landed, lines:", chk.stdout.strip() or "MISSING")
    subprocess.run(["bash", "-lc",
                    f"cd {ROOT} && setsid nohup ./chain_walk.sh > chain_walk.log "
                    "2>&1 < /dev/null & disown ; sleep 5 ; "
                    "ps -eo pid,args | grep 'chain_wal[k]' | head -2"], check=False)
    print("armed chain_walk.sh: a = 0, .25, .5, .75, 1 on the 180-row panel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
