#!/usr/bin/env python3
"""Rebuild the RULER queue as ONE strictly serial chain.

WHY.  Three RULER stages matter and they were launched as three independent
watchers, each waiting on the same upstream condition:

  1. b4_wide    -- the continuous instrument's nominal leader (2.8216)
  2. evq tau=0.5/1.0/2.0 -- EVQ's FIRST long-range measurement anywhere
  3. gain 1.0 vs 1.138629436111989 -- the design face worth 70x the table

Because they waited on the same condition they would all wake at the same
instant, and four concurrent arms on a 32 GB card that already has a 131k-token
Qwen job resident has already OOMed once today.  This writes a single chain
with the stages strictly in series, and kills the duplicate watchers.

The stage ORDER is by decision value, so that if the card runs out of patience
the most informative stages have already run:

  b4_wide   -- the only continuously-leading arm with no panel number yet
  evq       -- the second road, never measured at long range at all
  gain      -- a single fixed table, answering whether gain is a confound
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-tmp/phase1_20260910")

CHAIN = r'''#!/bin/bash
cd /root/autodl-tmp/phase1_20260910
export PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:/root/autodl-tmp/phase1_20260910/repoharness:/root/autodl-tmp/phase1_20260910
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
P=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl
A=/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01

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
  return $rc
}

# 1. wait for the C42/C42V24 stage (which is already running) to release the card
while pgrep -f "olmo_beta.py --root /root/autodl-tmp/phase1_20260910/olmo_c42" >/dev/null 2>&1; do sleep 30; done
sleep 25

# 2. b4_wide -- the continuous leader, the only top-5 arm with no panel number
run b4_wide olmo_wide.log $PY olmo_beta.py --root /root/autodl-tmp/phase1_20260910/olmo_wide \
    --model "$M" --panel "$P" --archive "$A" --wide-betas 4.0 --turns ""

# 3. EVQ tau = 0.5 / 1.0 / 2.0 -- the FIRST long-range measurement of EVQ anywhere
run evq olmo_evq.log $PY olmo_beta.py --root /root/autodl-tmp/phase1_20260910/olmo_evq \
    --model "$M" --panel "$P" --archive "$A" --betas "" --turns "" --evq 0.5,1.0,2.0

# 4. gain 1.0 vs the inherited 1.138629436111989 -- table identical, one scalar
run gain olmo_gain.log $PY olmo_beta.py --root /root/autodl-tmp/phase1_20260910/olmo_gain \
    --model "$M" --panel "$P" --archive "$A" --betas "" --turns "" --gains 1.0,1.138629436111989

echo "=== SERIAL QUEUE COMPLETE $(date -Is) ==="
'''


def main():
    # kill the duplicate watchers; the already-running arm process is untouched
    subprocess.run(["bash", "-lc",
                    "pkill -9 -f 'chain_ev[q].sh' ; pkill -9 -f 'chain_wid[e].sh' ; true"])
    path = ROOT / "chain_serial.sh"
    path.write_text(CHAIN)
    path.chmod(0o755)
    subprocess.run(["bash", "-lc",
                    f"cd {ROOT} && setsid nohup ./chain_serial.sh > chain_serial.log "
                    "2>&1 < /dev/null & disown ; sleep 4 ; "
                    "ps -eo pid,args | grep 'chain_seria[l]' | head -1"])
    print("armed chain_serial.sh -- b4_wide -> evq -> gain, strictly in series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
