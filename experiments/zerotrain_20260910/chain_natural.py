#!/usr/bin/env python3
"""Arm the natural-QA generalisation chain (391 rows, 5 families, all @16384).

RUN ON THE SERVER.  Governed by NATURAL_PREREG_20260911.md, committed before any
row of this panel was read.

WHY IT QUEUES LAST.  Three chains already own the card; a fourth would add OOM
risk without adding throughput.  This waits for chain_walk, by which point the
card should be free, so it runs at full speed.

Four arms, ALL pre-existing tables -- no new table is introduced, so a positive
result here is a statement about the plateau family rather than about a new
construction fitted to this panel.
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
PANEL=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_natural_union/screen.jsonl
EMPTY=/root/autodl-tmp/phase1_20260910/empty_archive
OUT=/root/autodl-tmp/phase1_20260910/natural_out
mkdir -p "$EMPTY" "$OUT"

while pgrep -f "chain_wal[k]" >/dev/null 2>&1; do sleep 30; done
sleep 20

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

run nat_bm    nat_bm.log    $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --betas 1.0 --turns ""
run nat_b3    nat_b3.log    $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --betas 3.0 --turns ""
run nat_a1b64 nat_a1b64.log $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --wide-betas 1.0 --turns ""
run nat_b4w   nat_b4w.log   $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --wide-betas 4.0 --turns ""

"$PY" natural_read.py 2>&1 | tee natural_read.txt
echo "=== NATURAL COMPLETE $(date -Is) ==="
'''


def main():
    p = pathlib.Path(ROOT) / "chain_natural.sh"
    p.write_text(CHAIN)
    p.chmod(0o755)
    chk = subprocess.run(["bash", "-lc", f"test -s {p} && wc -l < {p}"],
                         capture_output=True, text=True)
    print("script landed, lines:", chk.stdout.strip() or "MISSING")
    subprocess.run(["bash", "-lc",
                    f"cd {ROOT} && setsid nohup ./chain_natural.sh > chain_natural.log "
                    "2>&1 < /dev/null & disown ; sleep 5 ; "
                    "ps -eo pid,args | grep 'chain_natur[al]' | head -2"], check=False)
    print("armed chain_natural.sh: BM -> b3 -> a1b64 -> b4wide on 391 natural rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
