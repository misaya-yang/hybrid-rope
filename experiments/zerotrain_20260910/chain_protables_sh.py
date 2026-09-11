#!/usr/bin/env python3
"""Deploy chain_protables.sh properly (the heredoc attempt silently produced nothing).

WHAT IT RUNS.  The Pro model's two tables on the 350-row RULER panel -- the only
instrument this campaign treats as an arbiter.  Their continuous-instrument
numbers are already in: pro_step42 +0.017 and pro_condEVQ +0.036 against the
deployed table, i.e. both worse.  The RULER run is the confirmation.

WHY IT HAS TO WAIT.  The card is 32 GB and already hosts the serial chain's gain
stage; this waits for chain_serial to finish rather than racing it.

The previous attempt embedded the script in a nested heredoc inside an ssh
single-quoted command.  The outer shell consumed the terminator and the file was
never written -- the same failure this session has now hit three times, always
when a heredoc is nested inside another quoting context.  Writing the file
locally and copying it is the pattern that works.
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
PANEL=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl
ARCH=/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01
OUT=/root/autodl-tmp/phase1_20260910/olmo_pro

while pgrep -f "chain_seria[l]" >/dev/null 2>&1; do sleep 30; done
while pgrep -f "chain_pr[o]" >/dev/null 2>&1; do sleep 30; done
sleep 25

echo "=== STAGE pro_tables START $(date -Is) ==="
"$PY" olmo_beta.py --root "$OUT" \
    --model "$MODEL" --panel "$PANEL" --archive "$ARCH" \
    --betas "" --turns "" --pro-tables condEVQ,step42 > olmo_pro.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "=== STAGE pro_tables FAILED rc=$rc $(date -Is) ==="
  tail -25 olmo_pro.log
else
  echo "=== STAGE pro_tables OK $(date -Is) ==="
fi
echo "=== PRO TABLES COMPLETE $(date -Is) ==="
'''


def main():
    p = pathlib.Path(ROOT) / "chain_protables.sh"
    p.write_text(CHAIN)
    p.chmod(0o755)
    # verify it landed before arming it -- the previous attempt did not
    chk = subprocess.run(["bash", "-lc",
                          f"test -s {p} && echo present && wc -l < {p}"],
                         capture_output=True, text=True)
    print("file check:", chk.stdout.strip() or chk.stderr.strip())
    subprocess.run(["bash", "-lc",
                    f"cd {ROOT} && setsid nohup ./chain_protables.sh > chain_protables.log "
                    "2>&1 < /dev/null & disown ; sleep 4 ; "
                    "ps -eo pid,args | grep 'chain_protable[s]' | head -1"])
    print("armed chain_protables.sh (waits for chain_serial and chain_pro)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
