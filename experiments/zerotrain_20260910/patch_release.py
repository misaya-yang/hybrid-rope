#!/usr/bin/env python3
"""Add the RELEASE arms to olmo_longnll.py.

THE AXIS.  `m_incr_beta` (and every table in this campaign) pins m = 1 on every
slot past the band.  That is not a physical requirement -- it is a hard-coded
one.  Its cost is measurable: at 4x the window, slot j is readable iff

    4 * t_W(j) * 4^(-m_j) >= 0.25     <=>     m_j <= log4(16 * t_W(j)) =: m_max(j)

and on OLMo

    slot   38    39     40     41     42     43     44     45     46
    m_max 1.05  0.906  0.758  0.610  0.462  0.314  0.167  0.019  -0.129

so slots 39-45 are READABLE at m = 0 and are pushed out the BOTTOM of the window
by the m = 1 plateau.  Seven slots, thrown away.

WHY THE OBVIOUS OBJECTION IS WRONG.  The campaign already tested "cut the
plateau": the KNIFE taper, delta = 0.0063/slot, scored -2.30pp and was read as
"the m=1 plateau is load-bearing".  But 0.0063 is ~23x too small.  On slots
39-45 the KNIFE taper leaves m at 0.956 ... 0.918, ALL ABOVE m_max, so it
recovers ZERO slots and pays pure cost.  The release rate that actually works is

    rate = 0.98 * ln(1/ratio) / ln4 = 0.98 * 0.147903 = 0.14495 per slot

where the bound 0.147903 = ln(theta)/(2*K*ln2) is exactly the slope at which nu
stops being strictly decreasing -- the same number that appears as the readable
band's slope, because sliding along the band IS holding nu constant.  Verified:
at that rate every slot 39-45 lands below m_max AND nu stays strictly decreasing.

WHAT IT TESTS.  These arms hold the shape fixed and only move the slow tail, so
they separate the two candidate mechanisms cleanly:

    N (readable-slot count):  BM 17 -> 24,  b3 18 -> 25,  b4_wide 19 -> 26
    S (sum of m):             BM 40.5 -> 18.5, b3 43.6 -> 21.6, b4_wide 46.7 -> 24.6

The N mechanism predicts a large gain; the S mechanism predicts the floor (S is
far outside the measured 36.3-42.4 range).  Both were pre-registered by the two
analyses that derived the axis independently.

NOT A SCAN.  The rate is pinned by two inequalities (window membership and nu
monotonicity) to a narrow interval; there is no free parameter to sweep.
"""
from __future__ import annotations

import math
import pathlib
import sys

P = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_longnll.py")

BLOCK = None

NEW_BLOCK = '''    # ---- THE RELEASE AXIS ------------------------------------------------
    # m = 1 past the band is hard-coded, not required.  Slots 39-45 are
    # readable at m = 0 and get pushed out the bottom of the 4x window by the
    # plateau.  The rate below is pinned by two inequalities -- stay inside the
    # window (m_j <= log4(16 t_W(j))) and keep nu strictly decreasing
    # (m_j - m_{j+1} < ln(1/ratio)/ln4 = 0.147903) -- so there is nothing to
    # sweep.  0.98 of the bound satisfies both at every slot.
    LN4 = math.log(4.0)
    _ratio = OLMO["theta"] ** (1.0 / 64.0)
    RELEASE_RATE = 0.98 * math.log(_ratio) / LN4

    def release(m, start=38, rate=RELEASE_RATE):
        m = np.asarray(m, dtype=float).copy()
        for j in range(start + 1, 64):
            m[j] = max(0.0, 1.0 - rate * (j - start))
        return m

    return [
        ("native", np.zeros(64)),
        ("rel_bm", release(np.asarray(m_incr_beta(1.0, n=n, low=lo), float))),
        ("rel_b3", release(np.asarray(m_incr_beta(3.0, n=n, low=lo), float))),
        ("rel_b4w", release(np.asarray(m_incr_beta(4.0, n=21, low=11), float))),'''

MARKER = "    return [\n        (\"native\",    np.zeros(64)),"


def main():
    s = P.read_text()
    if "rel_bm" in s:
        print("already patched")
        return 0
    if MARKER not in s:
        print("REFUSING: anchor not found", file=sys.stderr)
        return 2
    s = s.replace(MARKER, NEW_BLOCK, 1)
    if "import math" not in s.split("\n\n")[0]:
        s = s.replace("import json\n", "import json\nimport math\n", 1)
    P.write_text(s)
    print("patched olmo_longnll.py with rel_bm / rel_b3 / rel_b4w")
    import numpy as np
    return 0


if __name__ == "__main__":
    sys.exit(main())
