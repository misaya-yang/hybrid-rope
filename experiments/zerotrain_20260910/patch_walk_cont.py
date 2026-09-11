#!/usr/bin/env python3
"""Add the frontier-walk doses to the CONTINUOUS instrument's arm bank.

RUN ON THE SERVER (edits a server file).

WHY.  The walk is the decisive RULER experiment, but RULER only reports accuracy
on 60 in-window rows, where a handful of rows breaking and a smooth average shift
look identical.  The continuous instrument (teacher-forced tail-512 NLL at 16384,
16 documents, native measured in-process) sees the SAME question as a real number
and costs ~25 s per arm, so all five doses cost about two minutes.

WHAT THE PAIR DECIDES.  CONSTRAINT_IS_SLACK_20260911.md measures the per-slot
in-window gradient of the continuous instrument at the deployed table:
sign-alternating, magnitude 0.002-0.010, mean +0.0015 -- i.e. in-band
reallocation is free ON AVERAGE.  The RULER panel nevertheless loses 4-6pp
in-window, and the break counts say only 12 of 60 rows ever break.  Those are
compatible if the damage is a SPARSE TAIL rather than a shift of the mean -- but
they are incompatible if the continuous instrument also moves smoothly.

  * continuous NLL flat in a  AND  RULER 4096 breaking a few rows
        -> sparse tail; "in-band free on average" survives, and the in-window
           cost is a distributional effect invisible to a mean
  * continuous NLL rises smoothly in a
        -> the average DOES move; "in-band free" was a low-power artefact

Either way the walk adjudicates a claim the campaign currently holds at three
stars.  Uses the RAW m_incr_beta tables, not the release() variants, so the doses
are bit-identical to what olmo_beta.py builds for the RULER walk.

Idempotent; verifies the arms are importable before returning.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

SRC = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_longnll.py")

DEFS = '''    # ---- THE FRONTIER WALK: the two endpoints of one family -----------------
    # BM = m_incr_beta(1.0, n, low) has increments on slots [15,32]; the wide
    # table reaches three slots further to the fast end.  Same ramp exponent,
    # same m=1 plateau from slot 32, so the segment between them is one
    # parameter and its exchange rate is measurable.
    _bm = np.asarray(m_incr_beta(1.0, n=n, low=lo), float)
    _wid = np.asarray(m_incr_beta(1.0, n=21, low=11), float)

'''

ENTRIES = '''        # ---- the frontier walk, doses identical to the RULER walk ------------
        *[(f"walk_a{str(a).replace('.', 'p')}", (1.0 - a) * _bm + a * _wid)
          for a in (0.0, 0.25, 0.5, 0.75, 1.0)],

'''

ANCHOR_RETURN = "    return [\n"
ANCHOR_ENTRY = '        ("pro_step42", _pro_table("step42")),\n'


def main():
    s = SRC.read_text()
    if "walk_a0p25" in s:
        print("already patched")
    else:
        assert s.count(ANCHOR_RETURN) == 1, "return anchor not unique"
        assert s.count(ANCHOR_ENTRY) == 1, "entry anchor not unique"
        s = s.replace(ANCHOR_RETURN, DEFS + ANCHOR_RETURN, 1)
        s = s.replace(ANCHOR_ENTRY, ANCHOR_ENTRY + ENTRIES, 1)
        SRC.write_text(s)
        print("patched")
    r = subprocess.run(["/root/miniconda3/bin/python", "-c", "import ast,sys;"
                        "ast.parse(open(sys.argv[1]).read())", str(SRC)],
                       capture_output=True, text=True)
    print("syntax:", "OK" if r.returncode == 0 else r.stderr[-400:])
    # verify the arms exist and the endpoints reproduce the RULER runner exactly
    chk = subprocess.run(
        ["/root/miniconda3/bin/python", "-c", """
import sys, numpy as np
sys.path.insert(0, "/root/autodl-tmp/nongeometric_screen_20260909/code")
sys.path.insert(0, "/root/autodl-tmp/phase1_20260910/repoharness")
sys.path.insert(0, "/root/autodl-tmp/phase1_20260910")
import olmo_longnll as L
from experiments.curvature_20260910.tables import m_incr_beta
arms = dict(L.build_arms())
want = {f"walk_a{str(a).replace('.','p')}" for a in (0.0, 0.25, 0.5, 0.75, 1.0)}
missing = want - set(arms)
print("missing:", sorted(missing) if missing else "none")
bm = m_incr_beta(1.0, n=18, low=14); wid = m_incr_beta(1.0, n=21, low=11)
print("a=0 identical to --betas 1.0     :", np.array_equal(arms["walk_a0p0"], bm))
print("a=1 identical to --wide-betas 1.0:", np.array_equal(arms["walk_a1p0"], wid))
print("sum_m:", {k: round(float(v.sum()), 4) for k, v in sorted(arms.items())
               if k.startswith("walk_")})
"""], capture_output=True, text=True)
    print(chk.stdout.strip())
    if chk.stderr.strip():
        print("STDERR:", chk.stderr[-500:])
    return 0 if ("missing: none" in chk.stdout and "syntax: OK" in
                 (r.stderr or "syntax: OK")) else 1


if __name__ == "__main__":
    sys.exit(main())
