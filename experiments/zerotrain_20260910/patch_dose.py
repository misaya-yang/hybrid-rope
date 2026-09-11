#!/usr/bin/env python3
"""Add the MrRoPE<->BM compression-dose family to the continuous instrument.

RUN ON THE SERVER (edits olmo_longnll.py).

WHY -- this is the experiment that turns goal item 1 from an open direction into
a measured rule.  The standing claim after today is:

    the optimal compression depends on how much rescue the base model needs
    (OLMo 4x needs 4.35 nats -> BM's extra compression wins by 0.826).

!! CORRECTION 2026-09-11.  This docstring used to continue "...Qwen 4x needs 0.18
-> MrRoPE's lighter compression wins".  That clause came from reading the SIGN of
`BM - MrRoPE` backwards (NLL is lower-is-better).  Recomputed, Qwen 4x also
favours BM (-0.0074, t=-3.34); there is no flip.  See CORRECTION_SIGN_20260911.md.

So the rule is NOT "the optimum flips with the model".  What the dose family
established is narrower and is what this instrument measures: the optimum sits at
the maximum-compression END at every length tested on OLMo (a*(L) == 1).  The
open question this patch serves is therefore the SHAPE of the curve, not which
end wins.

That is a DIRECTION, not a strategy.  To make it a strategy we need the optimum
as a FUNCTION of length on one model, which is cheap on this instrument
(~15-45 s per arm).

THE FAMILY.  m(a) = (1-a)*m_MrRoPE + a*m_BM, a in {0,.25,.5,.75,1}.  Both
endpoints already exist as measured arms (mrpro, beta_b1_BM), so every interior
point is bracketed by two known tables rather than being a fresh guess -- and
this segment spans the two methods the project actually compares, so every
interior point is bracketed by two measured endpoints rather than being a fresh
guess.

WHAT IT ANSWERS.  Whether the curve's optimum MOVES with length.  Measured
answer on OLMo at 1x/2x/4x: it does not (a*(L) == 1, interior points rejected at
|t| >= 2.2) -- so this instrument's remaining use is to give the curve's SHAPE
(a continuous NLL readout of all five dose levels), not to locate a moving
optimum.  NOTE: it does NOT answer the cross-model question in the old title of
this paragraph -- see the correction above.

Idempotent; verified by an import check.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

SRC = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_longnll.py")

ENTRY = '''        # ---- THE COMPRESSION DOSE: MrRoPE -> BM -----------------------------
        # Endpoints are the two already-measured tables; the interior is
        # bracketed.  a*(L) is the question (see patch_dose.py).
        *[(f"dose_a{str(a).replace('.', 'p')}",
           (1.0 - a) * np.asarray(m_mrpro(n=n, low=lo), float)
           + a * np.asarray(m_incr_beta(1.0, n=n, low=lo), float))
          for a in (0.0, 0.25, 0.5, 0.75, 1.0)],

'''

ANCHOR = '        ("pro_step42", _pro_table("step42")),\n'


def main():
    s = SRC.read_text()
    if "dose_a0p25" in s:
        print("already patched")
    else:
        assert s.count(ANCHOR) == 1, "anchor not unique"
        s = s.replace(ANCHOR, ANCHOR + ENTRY, 1)
        SRC.write_text(s)
        print("patched")
    r = subprocess.run(["/root/miniconda3/bin/python", "-c", "import ast,sys;"
                        "ast.parse(open(sys.argv[1]).read())", str(SRC)],
                       capture_output=True, text=True)
    print("syntax:", "OK" if r.returncode == 0 else r.stderr[-400:])
    c = subprocess.run(["/root/miniconda3/bin/python", "-c", """
import sys, numpy as np
sys.path.insert(0, "/root/autodl-tmp/nongeometric_screen_20260909/code")
sys.path.insert(0, "/root/autodl-tmp/phase1_20260910/repoharness")
sys.path.insert(0, "/root/autodl-tmp/phase1_20260910")
import olmo_longnll as L
from experiments.curvature_20260910.tables import m_mrpro, m_incr_beta
A = dict(L.build_arms())
want = {f"dose_a{str(a).replace('.','p')}" for a in (0.0,0.25,0.5,0.75,1.0)}
print("missing:", sorted(want - set(A)) or "none")
print("a=0 identical to mrpro:", np.array_equal(A["dose_a0p0"], A["mrpro"]))
print("a=1 identical to beta_b1_BM:", np.array_equal(A["dose_a1p0"], A["beta_b1_BM"]))
print("sum_m:", {k: round(float(v.sum()),3) for k,v in sorted(A.items()) if k.startswith("dose_")})
"""], capture_output=True, text=True)
    print(c.stdout.strip())
    if c.stderr.strip():
        print("STDERR:", c.stderr[-300:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
