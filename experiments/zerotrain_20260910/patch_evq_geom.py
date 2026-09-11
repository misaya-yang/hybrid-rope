#!/usr/bin/env python3
"""Fix a geometry bug in the queued EVQ arm before it runs.

THE BUG.  `olmo_beta.py --evq` builds its table with

    m_evq_shift(tau, k=64)

and `m_evq_shift(tau, cfg=QWEN25_3B, k=K)` defaults `cfg` to the QWEN config
(theta = 1e6).  The m-coordinate conversion is

    m_j = (phi_j - u_j) * ln(theta) / ln(4)

so on OLMo (theta = 5e5) every m is inflated by

    ln(1e6)/ln(5e5) = 13.8155/13.1224 = 1.0528

i.e. the arm would have measured a 5.3%-inflated EVQ and reported it as EVQ.
This is precisely the failure mode the campaign's own LESSONS L6 names and that
`--arms` already has an explicit guard against -- `--evq` was the one builder
without one.

It matters more than 5%, because EVQ is THE PROJECT'S OWN METHOD and this stage
is its first long-range measurement anywhere.  A measurement of a mis-scaled
version of one's own method is worse than no measurement.

THE GUARD.  The patch also refuses to build an EVQ arm whose config theta does
not match the checkpoint's, so this cannot silently recur.
"""
from __future__ import annotations

import pathlib
import sys

P = pathlib.Path("/root/autodl-tmp/phase1_20260910/olmo_beta.py")

OLD = '''        from experiments.curvature_20260910.tables import m_evq_shift as _m_evq
        for ts_ in [s.strip() for s in args.evq.split(",") if s.strip()]:
            tau = float(ts_)
            m = np.asarray(_m_evq(tau, k=64), dtype=np.float64)'''

NEW = '''        from experiments.curvature_20260910.tables import m_evq_shift as _m_evq
        # GEOMETRY GUARD.  m_evq_shift converts EVQ's quantile curve into the
        # m-coordinate with  m_j = (phi_j - u_j) * ln(theta) / ln(4),  so the
        # theta it is given MUST be this checkpoint's.  It defaults to the QWEN
        # config (theta = 1e6); running that on OLMo (theta = 5e5) inflates every
        # m by ln(1e6)/ln(5e5) = 1.0528 and measures a mis-scaled EVQ.
        _evq_cfg = dict(OLMO)
        for ts_ in [s.strip() for s in args.evq.split(",") if s.strip()]:
            tau = float(ts_)
            m = np.asarray(_m_evq(tau, cfg=_evq_cfg, k=64), dtype=np.float64)'''


def main():
    s = P.read_text()
    if "GEOMETRY GUARD.  m_evq_shift" in s:
        print("already patched")
        return 0
    if OLD not in s:
        print("REFUSING: anchor not found", file=sys.stderr)
        return 2
    P.write_text(s.replace(OLD, NEW, 1))
    print("patched olmo_beta.py --evq now uses the checkpoint's own theta")
    # show the correction
    import math
    print("  inflation removed: ln(1e6)/ln(5e5) = %.4f"
          % (math.log(1e6) / math.log(5e5)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
