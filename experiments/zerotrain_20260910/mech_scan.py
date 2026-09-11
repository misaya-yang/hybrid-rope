#!/usr/bin/env python3
"""Scan mechanism-parameterised functionals of the table against the 9 OLMo
scores, with the paired-decision test as the filter.  Every functional here is
written as a PHYSICAL sentence first; the arithmetic is the translation.

Count of candidates tried is printed, because a scan that does not say how many
things it looked at is a scan that has already lied (LESSONS L1).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.zerotrain_20260910.why import MEASURED, OLMO, table_of  # noqa: E402

K = 64
TH = OLMO["theta"]
W = OLMO["window"]
D = 16384.0
LN4 = math.log(4.0)
OMEGA = TH ** (-np.arange(K) / K)
T_W = OMEGA * W / (2 * math.pi)          # turns completed in the trained window
NATIVE_LGAP = math.log(TH) / K           # native log-frequency gap
RUNG = LN4 / NATIVE_LGAP                 # rungs of the native ladder per unit m


def m_of(name):
    return np.asarray(table_of(name), float)


def nov_frac(m, nsamp=4001):
    """Fraction of the TEST range on which each slot's phase is OUTSIDE the arc
    the model actually visited in training.

    Slot j visits phase 2*pi*t_W[j] in-window.  If t_W >= 1 that is the whole
    circle and nothing is novel.  Otherwise the trained arc is [0, 2*pi*t_W] and
    the test trajectory is 2*pi*t_W*(Delta/W)*4^-m, which leaves that arc -- and,
    past 2*pi, re-enters the circle at phases never trained.
    """
    d = np.linspace(0.0, D, nsamp)
    out = np.zeros(K)
    for j in range(K):
        if T_W[j] >= 1.0:
            continue
        turns = T_W[j] * (d / W) * 4.0 ** (-m[j])       # turns completed by Delta
        phase = (turns % 1.0) * 2 * math.pi
        inside = phase <= 2 * math.pi * T_W[j]
        out[j] = 1.0 - inside.mean()
    return out


def bands(m):
    """The occupied rungs j + RUNG*m_j; report the gap structure of that ladder."""
    rung = np.arange(K) + RUNG * m
    gaps = np.diff(rung)
    return rung, gaps


def feats(m):
    nu = OMEGA * 4.0 ** (-m)
    T = nu * D / (2 * math.pi)
    u = nov_frac(m)
    rung, gaps = bands(m)
    f = {}
    # --- phase-velocity / aliasing family -------------------------------
    f["sum_nu"] = nu.sum()
    f["sum_nu2"] = (nu ** 2).sum()
    f["sum_nu_over_1"] = (nu / (1.0 + nu)).sum()
    # --- test-turn-count family -----------------------------------------
    for c in (0.5, 1.0, 2.0, 4.0, 8.0):
        f[f"nT_lt_{c}"] = float((T <= c).sum())
        f[f"logT_lt_{c}"] = float(np.log1p(T[T <= c]).sum())
    f["sum_invT"] = float((1.0 / T).sum())
    f["sum_T"] = float(T.sum())
    f["sum_sqrtT"] = float(np.sqrt(T).sum())
    f["entropy_T"] = float(-(np.log(T / T.sum()) * (T / T.sum())).sum())
    # --- novelty / coverage family --------------------------------------
    f["nov_unw"] = float(u.sum())
    f["nov_w_om"] = float((u * OMEGA).sum())
    f["nov_w_nu"] = float((u * nu).sum())
    f["nov_top_nu"] = float((u * nu).max())
    f["cov_w_om"] = float(((1 - u) * OMEGA).sum())
    # --- rung-ladder family ---------------------------------------------
    f["rung_maxgap"] = float(gaps.max())
    f["rung_mingap"] = float(gaps.min())
    f["rung_span"] = float(rung[-1] - rung[0])
    f["rung_holes"] = float(np.maximum(gaps - 1.0, 0).sum())
    f["rung_crowd"] = float(np.maximum(1.0 - gaps, 0).sum())
    f["rung_gapvar"] = float(gaps.var())
    # --- a budget-sensitive composite -----------------------------------
    f["sum_m"] = float(m.sum())
    return f


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    names = [n for n, _ in MEASURED]
    acc = {n: a for n, a in MEASURED}
    F = {n: feats(m_of(n)) for n in names}
    keys = sorted(F[names[0]])

    # all 36 ordered pairs, as an unordered set
    pairs = [(names[i], names[j]) for i in range(len(names))
             for j in range(i + 1, len(names))]

    rows = []
    for k in keys:
        v = np.array([F[n][k] for n in names], float)
        sp = spearman(v, np.array([acc[n] for n in names], float))
        vote = 0
        for a, b in pairs:
            va, vb = F[a][k], F[b][k]
            scale = max(abs(va), abs(vb), 1e-12)
            same_feat = abs(va - vb) / scale < 0.05
            same_acc = abs(acc[a] - acc[b]) < 0.05
            vote += 1 if (same_feat == same_acc) else -1
        rows.append((k, sp, vote))

    print(f"{len(keys)} candidates tried, 9 points, {len(pairs)} pairs")
    print(f"{'feature':18s} {'Spearman':>9s} {'net votes':>10s}  "
          f"({'of ' + str(len(pairs))})")
    for k, sp, vote in sorted(rows, key=lambda r: -r[2]):
        print(f"{k:18s} {sp:+9.3f} {vote:10d}")

    print("\n=== raw values for the top 6 ===")
    for k, sp, vote in sorted(rows, key=lambda r: -r[2])[:6]:
        print(f"{k}: " + " ".join(f"{F[n][k]:.4g}" for n in names))
    print("\narm order: " + " ".join(n[:10] for n in names))
    print("accuracy : " + " ".join(f"{acc[n]:.4f}" for n in names))


if __name__ == "__main__":
    main()
