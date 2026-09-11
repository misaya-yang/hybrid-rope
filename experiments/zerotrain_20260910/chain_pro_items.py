#!/usr/bin/env python3
"""Queue the two things the Pro model's plan asks for by name.

§9.3 -- A FIXED TO A ONE-POINT DIAGNOSTIC.  Our increment A derived a table from
a local Fisher cost and it collapsed (step_hi25: RULER 0.1121, continuous
3.0452).  The plan's correction is that the corner-point theorem was derived for
a DIFFERENT cost (C_j = 4t_j(1-4^-m), concave) than the one derive_tstar
substituted (0.5 (ln4)^2 sum F_jj m_j^2, convex with zero slope at native), so
the theorem never applied to what we tested.  Its prescription is a single
diagnostic: take the winners' budget S=42 and build the UNIQUE same-budget step,
h=22, i.e. m_j = 1[j>=22].  Verified: m_step(22, lo=14) has S = 42.000000
exactly.

The point of h=22 rather than h=25 is that it is the SAME BUDGET as the four
plateau members, so the comparison is budget-matched -- something no previous
step arm was.  If the same-budget step also loses, that kills the step family at
this budget only (the plan is explicit that it does not kill all steps at all
budgets, and does not localise which task term causes the difference).

§5 / §9.2 -- THE GAIN x TABLE 2x2.  Our campaign found gain is worth 0.0502 nats
in-window while the whole frequency table is worth 0.0007, and the plan's
corrections are that (i) the ratio depends on the denominator and is not a RULER
effect ratio, (ii) gain is an effect MODIFIER rather than a confounder, and
(iii) it is too strong to say the literature never handled gain jointly.  Its
prescription is the interaction

    I = [R(new,g1) - R(Mr,g1)] - [R(new,gY) - R(Mr,gY)]

with tables {a1_b64, MrRoPE} x gain {1.0, 1.138629436111989}.  The existing
`--gains` flag could only reach the deployed BM; `patch_gain2x2.py` adds
`--gain-tables` so the interaction is actually reachable.  This chain runs the
two tables the plan names, plus b3 for completeness; BM is left to the already
queued stage so nothing is duplicated.

Both stages wait for the currently running serial chain, so at most one new arm
is resident at a time on the 32 GB card.
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
RUNROOT=/root/autodl-tmp/phase1_20260910

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

# wait for the serial chain (evq -> gain) to drain before starting
while pgrep -f "chain_seria[l].sh" >/dev/null 2>&1; do sleep 30; done
sleep 25

# Pro plan §9.3: the budget-matched step.  m_step(22) has S = 42 exactly.
run h22 olmo_h22.log $PY olmo_beta.py --root "$RUNROOT/olmo_h22" \
    --model "$MODEL" --panel "$PANEL" --archive "$ARCH" \
    --betas "" --turns "" --steps 22

# Pro plan §5/§9.2: the gain x table interaction, two tables x two gains
run gain2x2 olmo_gain2x2.log $PY olmo_beta.py --root "$RUNROOT/olmo_gain2x2" \
    --model "$MODEL" --panel "$PANEL" --archive "$ARCH" \
    --betas "" --turns "" --gains 1.0,1.138629436111989 \
    --gain-tables a1_b64,mrpro,b3

echo "=== PRO ITEMS COMPLETE $(date -Is) ==="
'''


def main():
    p = pathlib.Path(ROOT) / "chain_pro.sh"
    p.write_text(CHAIN)
    p.chmod(0o755)
    subprocess.run(["bash", "-lc",
                    f"cd {ROOT} && setsid nohup ./chain_pro.sh > chain_pro.log "
                    "2>&1 < /dev/null & disown ; sleep 4 ; "
                    "ps -eo pid,args | grep 'chain_pr[o]' | head -1"])
    print("armed chain_pro.sh: --steps 22 (§9.3) then --gain-tables 2x2 (§5/§9.2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
