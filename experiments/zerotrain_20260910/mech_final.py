#!/usr/bin/env python3
"""The numbers the mechanism report quotes.  Every one is arithmetic here."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.curvature_20260910.tables import (band_from_turns, m_incr_beta,  # noqa: E402
                                                   m_turns)
from experiments.zerotrain_20260910.why import MEASURED, OLMO, table_of  # noqa: E402

K, TH, W, D = 64, OLMO["theta"], OLMO["window"], 16384.0
OMEGA = TH ** (-np.arange(K) / K)
T_W = OMEGA * W / (2 * math.pi)
LN4 = math.log(4.0)


def unconstrained_fraction(m, j, nsamp=200001):
    """Fraction of the TEST range on which slot j's phase is on a part of the
    circle the trained loss never constrained.  Only possible when t_j < 1."""
    t = T_W[j]
    if t >= 1.0:
        return 0.0
    d = np.linspace(0.0, D, nsamp)
    turns = t * (d / W) * 4.0 ** (-m[j])
    return float(((turns % 1.0) > t).mean())


def main():
    print("### 1. the alpha = 1 boundary, exactly")
    lo, hi = band_from_turns(1.0, 32.0, TH, W, 128)
    print(f"    band [alpha=1, beta=32] on OLMo = slots [{lo}, {hi}], n = {hi - lo}")
    print(f"    t_W[{hi - 1}] = {T_W[hi - 1]:.4f} (>1, last slot the model saw wrap)")
    print(f"    t_W[{hi}]   = {T_W[hi]:.4f} (<1, first slot it never saw wrap)")
    print(f"    so 'fully compress iff t_W < 1' == 'fully compress iff j >= {hi}'")

    print("\n### 2. unconstrained-phase fraction of the arms that violate it")
    for nm in ("turns_a0p5_b32", "turns_a1_b32" if False else "MrProBM b=1 (arch)"):
        m = np.asarray(table_of(nm), float)
        bad = [(j, m[j], unconstrained_fraction(m, j))
               for j in range(K) if T_W[j] < 1.0 and m[j] < 1 - 1e-12]
        print(f"    {nm}: {len(bad)} slots with t_W<1 and m<1")
        for j, mj, u in bad[:8]:
            print(f"       slot {j:2d} t_W={T_W[j]:.4f} m={mj:.4f} "
                  f"-> {u * 100:5.1f}% of the test range in unconstrained phase")

    print("\n### 3. drift  d_j(D) = 4 t_j (1 - 4^-m_j)  turns, at m = 1")
    for j in (0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 63):
        print(f"    slot {j:2d}  t_W={T_W[j]:9.3f}   d(m=1)={3 * T_W[j]:10.3f} turns"
              f"   marginal d/dm at m=0 = {4 * T_W[j] * LN4:10.3f}")

    print("\n### 4. beta=64 is mostly a RE-SHAPING, not an edge move")
    m32 = np.asarray(m_turns(1.0, 32.0, TH, W, 128, ramp="beta1"), float)
    m64 = np.asarray(m_turns(1.0, 64.0, TH, W, 128, ramp="beta1"), float)
    d = m64 - m32
    print(f"    slots 12,13,14: dSum(m) = {d[12:15].sum():.4f}  "
          f"(d = {np.round(d[12:15], 4).tolist()})")
    print(f"    slots 15..31  : dSum(m) = {d[15:32].sum():.4f}")
    print(f"    total dSum(m) = {d.sum():.4f};  "
          f"share from the three edge slots = {d[12:15].sum() / d.sum():.1%}")

    print("\n### 5. the 12 pp pair, concretely at D = 16384")
    a = np.asarray(table_of("turns_a1_b16"), float)
    b = np.asarray(table_of("beta_b0p25"), float)
    for nm, m in (("a1_b16", a), ("b0p25", b)):
        nu = OMEGA * 4.0 ** (-m)
        T = nu * D / (2 * math.pi)
        print(f"    {nm}: T_j = " + " ".join(f"{T[j]:.2f}" for j in range(16, 33)))
    dm = a - b
    print("    d m  = " + " ".join(f"{dm[j]:+.4f}" for j in range(16, 33)))
    print(f"    dSum(m) = {dm.sum():+.4f};  "
          f"sum over the fast half (15..23) = {dm[15:24].sum():+.4f};  "
          f"over the slow half (24..31) = {dm[24:32].sum():+.4f}")

    print("\n### 6. the STEP family the mechanism says is optimal")
    acc = {n: v for n, v in MEASURED}
    print(f"    {'t*':>6} {'slots m=1':>10} {'Sum m':>7} {'linear-fit pred':>16}")
    for tstar in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        m = (T_W < tstar).astype(float)
        S = m.sum()
        print(f"    {tstar:6.1f} {int(S):10d} {S:7.1f} {0.09595 * S - 3.51983:16.4f}")
    print("    (linear fit acc = 0.09595*Sum m - 3.51983, from the nine points)")

    print("\n### 7. compression front t@75 for the queued arms")
    print(f"    {'arm':16s} {'Sum m':>7} {'t @ m=0.75':>11}")
    for nm, m in (("beta_b2 (meas)", np.asarray(m_incr_beta(2.0, n=18, low=14), float)),
                  ("beta_b3", np.asarray(m_incr_beta(3.0, n=18, low=14), float)),
                  ("beta_b4", np.asarray(m_incr_beta(4.0, n=18, low=14), float)),
                  ("beta_b6", np.asarray(m_incr_beta(6.0, n=18, low=14), float)),
                  ("beta_b8", np.asarray(m_incr_beta(8.0, n=18, low=14), float)),
                  ("turns_b128", np.asarray(m_turns(1.0, 128.0, TH, W, 128,
                                                    ramp="beta1"), float)),
                  ("turns_b256", np.asarray(m_turns(1.0, 256.0, TH, W, 128,
                                                    ramp="beta1"), float))):
        above = np.flatnonzero(m >= 0.75)
        j = int(above[0])
        frac = (0.75 - m[j - 1]) / (m[j] - m[j - 1])
        t75 = math.exp((1 - frac) * math.log(T_W[j - 1]) + frac * math.log(T_W[j]))
        print(f"    {nm:16s} {m.sum():7.3f} {t75:11.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
