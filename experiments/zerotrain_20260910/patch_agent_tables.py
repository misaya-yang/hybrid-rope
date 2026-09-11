#!/usr/bin/env python3
"""Add the two tables the sub-agents actually PREDICTED, not variants of them.

The release arms I built (rel_bm / rel_b3 / rel_b4w) are BM/b3/b4_wide with the
whole slow tail released.  They did not help: N rose from 17 to 24-26 and the
continuous long-range NLL did not improve (rel_bm was 0.034 WORSE than BM).
That falsifies MY variant, but it is not a fair test of either agent's claim --
neither of them proposed those tables.

What they proposed:

  minimal-params' T*  -- m = 1 on slots 24-38, then a release ramp on 39-44,
                          zero elsewhere.  S = 17.96, N = 22.  Its band starts
                          at 24, not 15, so it is NOT a variant of BM.
                          Prediction: RULER in [0.50, 0.86]; the S mechanism
                          predicts ~0.

  evq-limit's C_slow -- the deployed BM with ONLY slots 39-45 dropped from m=1
                          back to the band's upper edge log4(16 t_W(j)); the
                          other 57 slots bit-identical.  Its forcing puts a
                          reversed spike at the slow end, cancelling the
                          terminal monopole -- the "fifth vertex" (dipole).

Both are tested here on the same continuous instrument, in the same process as
the deployed BM, so the deltas are paired.
"""
from __future__ import annotations

import pathlib
import sys

P = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_longnll.py")

ANCHOR = '        ("rel_b4w", release(np.asarray(m_incr_beta(4.0, n=21, low=11), float))),'

ADD = '''        ("rel_b4w", release(np.asarray(m_incr_beta(4.0, n=21, low=11), float))),
        # ---- the two tables the sub-agents actually PREDICTED, not variants --
        # minimal-params' T*: m = 1 on 24-38, release ramp on 39-44, zero else.
        # Band starts at 24, so it is NOT a variant of the deployed BM.
        ("Tstar", _tstar()),
        # evq-limit's C_slow: deployed BM with slots 39-45 dropped back to the
        # band edge log4(16 t_W); the other 57 slots bit-identical.
        ("Cslow", _cslow()),'''

HELPERS = '''    def _tstar():
        m = np.zeros(64)
        m[24:39] = 1.0
        for j in range(39, 45):
            m[j] = max(0.0, 1.0 - RELEASE_RATE * (j - 38))
        return m

    def _cslow():
        m = np.asarray(m_incr_beta(1.0, n=n, low=lo), dtype=float).copy()
        # evq-limit used a uniform 0.1479/slot slope, which sits EXACTLY on the
        # nu-monotonicity bound (log4(1/ratio) = 0.147905) and therefore makes nu
        # constant, not strictly decreasing.  Verified: that version fails the
        # monotonicity check.  This is the same construction at 0.98 of the
        # bound, which is legal at every slot.
        for j in range(39, 46):
            m[j] = max(math.log(16.0 * tW_all[j], 4.0),
                       1.0 - RELEASE_RATE * (j - 38))
        return m

'''


def main():
    s = P.read_text()
    if "Tstar" in s:
        print("already patched")
        return 0
    if ANCHOR not in s:
        print("REFUSING: anchor not found", file=sys.stderr)
        return 2
    s = s.replace(ANCHOR, ADD, 1)
    # helpers go right after the release() definition
    marker = "        return m\n\n    return ["
    if marker not in s:
        print("REFUSING: helper anchor not found", file=sys.stderr)
        return 2
    # t_W for every slot, in the checkpoint's own geometry
    pre = ('    tW_all = OLMO["theta"] ** (-np.arange(64) / 64.0) * OLMO["window"] '
           '/ (2.0 * math.pi)\n\n')
    s = s.replace(marker, pre + HELPERS + "    return [", 1)
    P.write_text(s)
    print("patched olmo_longnll.py with Tstar and Cslow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
