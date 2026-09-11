#!/usr/bin/env python3
"""Where does the compression profile SIT, in turn coordinates?

For each table, find the slot where m crosses a level and report that slot's
IN-WINDOW TURN COUNT (a scale-free number that means the same thing on every
checkpoint).  Then check whether that single number orders the nine OLMo scores
-- including the pair sum(m) cannot separate -- and where it turns over.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.curvature_20260910.tables import band_from_turns, m_incr_beta  # noqa: E402
from experiments.zerotrain_20260910.why import MEASURED, OLMO, table_of  # noqa: E402

K, TH, W, D = 64, OLMO["theta"], OLMO["window"], 16384.0
OMEGA = TH ** (-np.arange(K) / K)
T_W = OMEGA * W / (2 * math.pi)


def turn_at_level(m, level):
    """In-window turn count of the slot where m crosses `level` (log-interp)."""
    above = np.flatnonzero(m >= level)
    if len(above) == 0:
        return float("nan")
    j = above[0]
    if j == 0:
        return float(T_W[0])
    m0, m1 = m[j - 1], m[j]
    if m1 == m0:
        frac = 0.0
    else:
        frac = (level - m0) / (m1 - m0)
    # interpolate in log turn count
    return float(math.exp((1 - frac) * math.log(T_W[j - 1])
                          + frac * math.log(T_W[j])))


def report(name, m, acc=None):
    nu = OMEGA * 4.0 ** (-m)
    T = nu * D / (2 * math.pi)
    lo, hi = band_from_turns(1.0, 32.0, TH, W, 128)
    row = dict(
        name=name, acc=acc, S=float(m.sum()),
        t25=turn_at_level(m, 0.25), t50=turn_at_level(m, 0.5),
        t75=turn_at_level(m, 0.75), t90=turn_at_level(m, 0.9),
        nT_ge1=int((T >= 1).sum()), nT_ge2=int((T >= 2).sum()),
        nT_ge4=int((T >= 4).sum()), nT_ge8=int((T >= 8).sum()),
        nT_ge16=int((T >= 16).sum()),
    )
    return row


def main():
    rows = []
    for n, a in MEASURED:
        rows.append(report(n, np.asarray(table_of(n), float), a))

    hdr = (f"{'arm':22s} {'acc':>6} {'S':>7} {'t@25':>7} {'t@50':>7} "
           f"{'t@75':>7} {'t@90':>7} {'nT>=1':>5} {'nT>=2':>5} {'nT>=4':>5} "
           f"{'nT>=8':>5} {'nT>=16':>6}")
    print("=== measured arms ===")
    print(hdr)
    for r in sorted(rows, key=lambda r: r["t50"]):
        print(f"{r['name']:22s} {r['acc']:6.4f} {r['S']:7.3f} {r['t25']:7.3f} "
              f"{r['t50']:7.3f} {r['t75']:7.3f} {r['t90']:7.3f} {r['nT_ge1']:5d} "
              f"{r['nT_ge2']:5d} {r['nT_ge4']:5d} {r['nT_ge8']:5d} {r['nT_ge16']:6d}")

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    acc = np.array([r["acc"] for r in rows])
    print("\n=== rank correlation with the score (9 points) ===")
    for k in ("S", "t25", "t50", "t75", "t90", "nT_ge1", "nT_ge2", "nT_ge4",
              "nT_ge8", "nT_ge16"):
        v = np.array([r[k] for r in rows], float)
        print(f"  {k:8s} Spearman {spearman(v, acc):+.3f}")

    print("\n=== the 12 pp pair, in this coordinate ===")
    d = {r["name"]: r for r in rows}
    a, b = d["turns_a1_b16"], d["beta_b0p25"]
    for k in ("S", "t25", "t50", "t75", "t90"):
        print(f"  {k:8s} a1_b16 {a[k]:8.3f}   b0p25 {b[k]:8.3f}   "
              f"ratio {a[k] / b[k]:6.3f}")

    print("\n=== the QUEUED arms, predicted by the same coordinate ===")
    q = []
    for beta in (128.0, 256.0):
        from experiments.curvature_20260910.tables import m_turns
        q.append(report(f"turns_a1_b{beta:g}",
                        np.asarray(m_turns(1.0, beta, TH, W, 128, ramp="beta1"),
                                   float)))
    for bb in (3.0, 4.0, 6.0, 8.0):
        q.append(report(f"beta_b{bb:g}", np.asarray(m_incr_beta(bb, n=18, low=14),
                                                    float)))
    for r in sorted(q, key=lambda r: r["t50"]):
        print(f"{r['name']:22s} {'--':>6} {r['S']:7.3f} {r['t25']:7.3f} "
              f"{r['t50']:7.3f} {r['t75']:7.3f} {r['t90']:7.3f} {r['nT_ge1']:5d} "
              f"{r['nT_ge2']:5d} {r['nT_ge4']:5d} {r['nT_ge8']:5d} {r['nT_ge16']:6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
