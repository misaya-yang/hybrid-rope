#!/usr/bin/env python3
"""Invert the nine OLMo scores for the per-slot gradient they imply.

If the long-range loss is (to first order) a linear functional of the table,

    score_i  =  c  +  sum_j g_j * m_ij

then nine arms give eight independent equations in g.  The system is
underdetermined, so the object to look at is the MINIMUM-NORM solution: it is
the unique g the data actually constrains (any component orthogonal to all
Delta m is invisible and is returned as zero).  That is a feature -- what comes
back is exactly what the measurements say, no more.

THIS IS A DIAGNOSTIC, NOT A CAPABILITY CLAIM.  It says "the nine points are
consistent with this profile"; it does not say compressing a slot causes a score
change.  LESSONS L1 is about exactly that jump.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.zerotrain_20260910.why import MEASURED, table_of  # noqa: E402

K = 64


def main():
    names = [n for n, _ in MEASURED]
    acc = np.array([a for _, a in MEASURED])
    M = np.array([np.asarray(table_of(n), float) for n in names])   # 9 x 64

    Mc = M - M.mean(0)
    yc = acc - acc.mean()
    # minimum-norm least squares
    g, *_ = np.linalg.lstsq(Mc, yc, rcond=None)
    pred = acc.mean() + Mc @ g
    res = acc - pred
    print("=== minimum-norm per-slot gradient implied by the 9 points ===")
    print(f"{'arm':22s} {'acc':>7} {'fit':>7} {'resid':>8}")
    for n, a, p, r in zip(names, acc, pred, res):
        print(f"{n:22s} {a:7.4f} {p:7.4f} {r:+8.4f}")
    print(f"\nRMSE = {np.sqrt((res ** 2).mean()):.4f}   "
          f"(2.8pp is the naive 350-row standard error)")

    print("\n=== g by slot (pp per unit m) ===")
    for j in range(K):
        if abs(g[j]) > 1e-12 or 8 <= j <= 40:
            print(f"  slot {j:2d}  g = {100 * g[j]:+8.3f} pp")

    print("\n=== what the fit says about the key pairs ===")
    idx = {n: i for i, n in enumerate(names)}
    for a, b in [("turns_a1_b16", "beta_b0p25"), ("turns_a1_b64", "MrProBM b=1 (arch)"),
                 ("beta_b2", "MrProBM b=1 (arch)"), ("turns_a2_b32", "MrProBM b=1 (arch)")]:
        dm = M[idx[a]] - M[idx[b]]
        print(f"{a} vs {b}: d_acc = {100 * (acc[idx[a]] - acc[idx[b]]):+.2f} pp, "
              f"d_sum_m = {dm.sum():+.3f}, "
              f"model <g,dm> = {100 * float(g @ dm):+.2f} pp")
        nz = np.flatnonzero(np.abs(dm) > 1e-9)
        print(f"    differing slots: {nz.min()}..{nz.max()}  ({len(nz)} slots)")

    # --- rank / identifiability -------------------------------------------
    s = np.linalg.svd(Mc, compute_uv=False)
    print(f"\nsingular values of the centred 9x64 design: "
          f"{np.array2string(s, precision=3)}")
    print("effective rank (s > 1e-6 * s0) =", int((s > 1e-6 * s[0]).sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
