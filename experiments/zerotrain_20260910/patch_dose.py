#!/usr/bin/env python3
"""Add the MrRoPE<->BM compression-dose family to the continuous instrument.

RUN ON THE SERVER (edits olmo_longnll.py).

WHY -- this is the experiment that turns goal item 1 from an open direction into
a measured rule.  The standing claim after today is:

    the optimal compression depends on how much rescue the base model needs
    (OLMo 4x needs 4.35 nats -> BM's extra compression wins by 0.826;
     Qwen 4x needs 0.18     -> MrRoPE's lighter compression wins)

That is a DIRECTION, not a strategy.  To make it a strategy we need the optimum
as a FUNCTION of length on one model, which is cheap on this instrument
(~15-45 s per arm).

THE FAMILY.  m(a) = (1-a)*m_MrRoPE + a*m_BM, a in {0,.25,.5,.75,1}.  Both
endpoints already exist as measured arms (mrpro, beta_b1_BM), so every interior
point is bracketed by two known tables rather than being a fresh guess -- and
this is the one segment where the sign is known to flip between models, which is
why it is the right segment for the question.

WHAT IT ANSWERS.  If a*(L) increases with length, "compress more when the base
is more broken" becomes a measured rule and the adaptive strategy is concrete.
If a*(L) is constant, the optimum does not move with length on this model and the
leverage story needs re-examination on the length axis.

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
