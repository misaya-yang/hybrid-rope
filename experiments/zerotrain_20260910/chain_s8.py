#!/usr/bin/env python3
"""Arm the length-axis chain: the four arms on prepared_s8_01.

RUN ON THE SERVER.  Governed by S8_PREREG_20260911.md (git a909d1b), which was
committed before any row of this panel was read.

Panel: 24 rows at 4096 (identical row ids to holdout180 -- a built-in
reproduction gate) + 48 rows at 32768 (never read by anything).  Same six tasks
the selection panel never used, so the length axis is measured WITHIN task.

Four arms, BM measured in-run (this panel has no archived endpoint).
Smoke-tested: 32768 prefill is fine under flash SDPA, ~15 s/row at 32768 and
~1 s at 4096, so ~12 min/arm.
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
PANEL=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_s8_01/screen.jsonl
EMPTY=/root/autodl-tmp/phase1_20260910/empty_archive
OUT=/root/autodl-tmp/phase1_20260910/s8_out
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

run s8_bm    s8_bm.log    $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --betas 1.0 --turns ""
run s8_b3    s8_b3.log    $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --betas 3.0 --turns ""
run s8_a1b64 s8_a1b64.log $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --wide-betas 1.0 --turns ""
run s8_b4w   s8_b4w.log   $PY olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$EMPTY" --wide-betas 4.0 --turns ""

echo "=== S8 COMPLETE $(date -Is) ==="
'''


def main():
    p = pathlib.Path(ROOT) / "chain_s8.sh"
    p.write_text(CHAIN)
    p.chmod(0o755)
    chk = subprocess.run(["bash", "-lc", f"test -s {p} && wc -l < {p}"],
                         capture_output=True, text=True)
    print("script landed, lines:", chk.stdout.strip() or "MISSING")
    subprocess.run(["bash", "-lc",
                    f"cd {ROOT} && setsid nohup ./chain_s8.sh > chain_s8.log "
                    "2>&1 < /dev/null & disown ; sleep 6 ; "
                    "ps -eo pid,args | grep 'chain_s[8]' | head -2"], check=False)
    print("armed chain_s8.sh: BM -> b3 -> a1_b64 -> b4wide on 72 rows (24@4096, 48@32768)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
