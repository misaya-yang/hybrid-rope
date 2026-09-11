#!/usr/bin/env python3
"""Is S the cause, or a proxy for something deeper?  Separate the collinear family.

WHY THIS FILE EXISTS.  `why.py` found that the only table-only quantities that
track the 9 OLMo scores are GLOBAL INTEGRALS (S = sum(m), sum(nu), crossing
slot).  But S is collinear with at least three other integrals by construction,
and 9 points -- 4 of them one family -- cannot separate them.  Before spending
GPU on a discriminating experiment we need to know exactly HOW MUCH the existing
data can and cannot see.  That is a rank/conditioning question, so it has a
numerical answer, not an opinion.

THE STRUCTURAL FACTS THIS FILE ESTABLISHES (all verified, not asserted):

  (1) S = L_ramp + n_plateau  EXACTLY.  These three are not three directions;
      they span a 2-plane.  So "S vs L_ramp vs n_plateau" is really "does the
      score see the SUM, or the SPLIT?", i.e. a single 1-dimensional question.

  (2) sum(nu) is NOT a function of S.  nu_j = omega_j * 4^-m_j is a sum of
      exponentials dominated by the LEAST compressed slots, so it weights the
      held end.  Whether it is collinear with S on THESE 9 arms is an empirical
      question -- and it is the one that decides whether the aliasing story and
      the budget story are distinguishable at all.

  (3) The designs that actually separate are the ones where the split
      (S vs n_plateau) or the held-end (S vs sum(nu)) is moved while the other
      is pinned.  This file CONSTRUCTS them and reports the exact geometry, so
      the experiment is pre-registered with numbers rather than hopes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.zerotrain_20260910.why import MEASURED, table_of, spearman  # noqa: E402

OLMO = dict(theta=500_000.0, window=4096, head_dim=128, K=64, low=14, n=18)
D_TEST = 16384.0
TH = OLMO["theta"]
K = OLMO["K"]

# native frequencies, shared by every arm (the spectrum is only re-allocated)
OMEGA = TH ** (-np.arange(K, dtype=float) / K)


def candidates(m, D=D_TEST):
    """The four candidates under test, plus the pieces that separate them."""
    nu = OMEGA * np.power(4.0, -m)
    S = float(m.sum())
    n_pl = int(np.sum(m > 1 - 1e-12))
    n_held = int(np.sum(m < 1e-12))
    n_ramp = K - n_pl - n_held
    # ramp integral: sum of m over slots strictly between the plateaus
    L_ramp = float(m[(m > 1e-12) & (m < 1 - 1e-12)].sum())
    return dict(
        S=S,
        mean_m=S / K,
        n_plateau=float(n_pl),
        n_held=float(n_held),
        n_ramp=float(n_ramp),
        L_ramp=L_ramp,
        sum_nu=float(nu.sum()),
        # the held-end / plateau-end decomposition of sum(nu): makes explicit
        # which end sum(nu) actually reads
        nu_held=float(nu[m < 1e-12].sum()),
        nu_plateau=float(nu[m > 1 - 1e-12].sum()),
        nu_ramp=float(nu[(m > 1e-12) & (m < 1 - 1e-12)].sum()),
        # aliasing reading at the test distance
        turns_max=float((nu * D / (2 * math.pi)).max()),
    )


def cond_number(M):
    """Condition number of a design matrix after standardising its columns.

    Standardising first is the point: the raw scales differ by orders of
    magnitude (S ~ 40, sum(nu) ~ 1), so an unstandardised condition number would
    just report the units.  The standardised one reports REDUNDANCY.
    """
    M = np.asarray(M, float)
    sd = M.std(axis=0)
    keep = sd > 1e-12
    Z = (M[:, keep] - M[:, keep].mean(axis=0)) / sd[keep]
    return float(np.linalg.cond(Z)), int(keep.sum())


def build_arm(name):
    return np.asarray(table_of(name), dtype=float)


def design_probe():
    """CONSTRUCT the separating arms from the table constructions already in the
    repo, and report their exact geometry.  No GPU, no guessing: if a design
    cannot be built with the required contrast, we find that out here."""
    from experiments.curvature_20260910.tables import m_turns, band_from_turns

    out = []
    # --- probe 1: same S, different n_plateau -----------------------------
    # n_plateau is set by the band's upper edge hi (slots hi+1..63 are all m=1).
    # Moving hi by one slot moves S by ~1 and n_plateau by 1, but we can buy the
    # slot back by widening the ramp.  Sweep (alpha, beta) and look for pairs at
    # matched S.
    grid = []
    for alpha in (0.5, 0.7, 1.0, 1.4, 2.0, 2.8):
        for beta in (12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256):
            try:
                m = np.asarray(m_turns(alpha, beta, TH, OLMO["window"],
                                       OLMO["head_dim"], ramp="beta1"),
                               dtype=float)
            except ValueError:
                continue
            lo, hi = band_from_turns(alpha, beta, TH, OLMO["window"],
                                     OLMO["head_dim"], k=K)
            c = candidates(m)
            c.update(alpha=alpha, beta=beta, lo=int(lo), hi=int(hi))
            grid.append(c)
    out.append(("turns_grid", grid))

    # --- probe 2: same S, different sum(nu) -------------------------------
    # sum(nu) is dominated by the held end, so it is moved by how many slots stay
    # at m=0.  The ramp shapes available here all start at slot lo, so holding
    # more slots means a later band.  That is the alpha knob at fixed beta.
    return out


def pair_scan(grid, key_x, key_y, tol_rel=0.002):
    """Arms that agree on key_x within tol but differ on key_y -- the separator
    candidates.  Reports the max contrast available at matched key_x."""
    grid = sorted(grid, key=lambda c: c[key_x])
    best = []
    for i in range(len(grid)):
        for j in range(i + 1, len(grid)):
            a, b = grid[i], grid[j]
            dx = abs(a[key_x] - b[key_x]) / max(abs(a[key_x]), 1e-12)
            if dx > tol_rel:
                continue
            dy = abs(a[key_y] - b[key_y])
            best.append((dy, a, b, dx))
    best.sort(key=lambda t: -t[0])
    return best


def main():
    print("=" * 78)
    print("PART 1.  THE 9 MEASURED ARMS: candidate values, exact")
    print("=" * 78)
    names = [n for n, _ in MEASURED]
    acc = np.array([a for _, a in MEASURED])
    C = []
    for n in names:
        C.append(candidates(build_arm(n)))
    keys = ["S", "mean_m", "n_plateau", "n_held", "n_ramp", "L_ramp", "sum_nu",
            "nu_held", "nu_plateau", "nu_ramp", "turns_max"]
    print(f"\n{'arm':22s} " + " ".join(f"{k:>10s}" for k in keys) + "   RULER")
    for n, c, a in zip(names, C, acc):
        print(f"{n:22s} " + " ".join(f"{c[k]:10.4f}" for k in keys)
              + f"   {a:.4f}")

    # --- the exact identity -------------------------------------------------
    print("\n--- EXACT IDENTITY CHECK:  S = L_ramp + n_plateau ---")
    for n, c in zip(names, C):
        resid = c["S"] - (c["L_ramp"] + c["n_plateau"])
        print(f"  {n:22s} S={c['S']:9.4f}  L_ramp+n_plateau="
              f"{c['L_ramp'] + c['n_plateau']:9.4f}  residual={resid:+.2e}")
    print("  (exact => S, L_ramp, n_plateau are 3 labels for 2 dimensions)")

    print("\n--- EXACT IDENTITY CHECK:  sum_nu = nu_held + nu_ramp + nu_plateau ---")
    for n, c in zip(names, C):
        resid = c["sum_nu"] - (c["nu_held"] + c["nu_ramp"] + c["nu_plateau"])
        print(f"  {n:22s} sum_nu={c['sum_nu']:.6f}  parts="
              f"{c['nu_held'] + c['nu_ramp'] + c['nu_plateau']:.6f}"
              f"  residual={resid:+.2e}")

    print("\n" + "=" * 78)
    print("PART 2.  COLLINEARITY OF THE CANDIDATES ON THE 9 POINTS")
    print("=" * 78)
    M = np.array([[c[k] for k in keys] for c in C], dtype=float)
    print(f"\n{'':12s} " + " ".join(f"{k[:9]:>9s}" for k in keys))
    for i, ki in enumerate(keys):
        row = []
        for j in range(len(keys)):
            v = M[:, i], M[:, j]
            r = (float(np.corrcoef(v[0], v[1])[0, 1]) if v[0].std() > 0
                 and v[1].std() > 0 else float("nan"))
            row.append(r)
        print(f"{ki:12s} " + " ".join(f"{x:+9.3f}" for x in row))

    # --- the question that matters: which candidates are collinear with S ---
    print("\n--- Pearson r with S, and with the SCORE ---")
    print(f"{'candidate':12s} {'r(S)':>8s} {'r(acc)':>8s} {'Spearman(acc)':>14s}"
          f" {'sd':>10s}")
    for k in keys:
        v = M[:, keys.index(k)]
        rs = (float(np.corrcoef(v, M[:, 0])[0, 1]) if v.std() > 0 else float("nan"))
        ra = (float(np.corrcoef(v, acc)[0, 1]) if v.std() > 0 else float("nan"))
        print(f"{k:12s} {rs:+8.3f} {ra:+8.3f} {spearman(v, acc):+14.3f}"
              f" {v.std():10.5f}")

    # --- condition number of the actual separating designs ----------------
    print("\n--- CONDITION NUMBER of standardised design matrices ---")
    tests = [
        ("S vs n_plateau", ["S", "n_plateau"]),
        ("S vs L_ramp", ["S", "L_ramp"]),
        ("S vs sum_nu", ["S", "sum_nu"]),
        ("S vs mean_m", ["S", "mean_m"]),
        ("S vs n_plateau vs sum_nu", ["S", "n_plateau", "sum_nu"]),
        ("S vs L_ramp vs sum_nu", ["S", "L_ramp", "sum_nu"]),
        ("all four", ["S", "n_plateau", "L_ramp", "sum_nu"]),
    ]
    for label, ks in tests:
        sub = np.array([[c[k] for k in ks] for c in C], dtype=float)
        cn, rank = cond_number(sub)
        print(f"  {label:28s} n_cols={len(ks)}  rank_eff={rank}  cond={cn:12.1f}"
              f"  {'SINGULAR' if cn > 1e6 else ('ill-posed' if cn > 100 else 'ok')}")

    print("\n" + "=" * 78)
    print("PART 3.  WHAT THE EXISTING DATA CANNOT SEE")
    print("=" * 78)
    # The honest rank statement: how many EFFECTIVE directions of variation do
    # the 9 arms actually span in the (S, n_plateau, sum_nu) space?
    sub = np.array([[c[k] for k in ("S", "n_plateau", "sum_nu")] for c in C])
    sd = sub.std(axis=0)
    Z = (sub - sub.mean(axis=0)) / sd
    u, s, vt = np.linalg.svd(Z, full_matrices=False)
    print(f"\n  singular values of standardised [S, n_plateau, sum_nu]: "
          f"{np.round(s, 4).tolist()}")
    print(f"  variance explained: {np.round(s ** 2 / (s ** 2).sum(), 4).tolist()}")
    print(f"  effective rank (s > 0.1): {int((s > 0.1).sum())} of 3")
    print("\n  principal directions (rows = components of [S, n_plateau, sum_nu]):")
    for i in range(len(s)):
        print(f"    pc{i + 1} (sv={s[i]:.4f}): {np.round(vt[i], 4).tolist()}")

    if __name__ == "__main__":
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
