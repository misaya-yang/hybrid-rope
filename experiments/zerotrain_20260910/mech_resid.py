#!/usr/bin/env python3
"""Which table functional explains the residual that sum(m) leaves?

sum(m) alone gives R^2 = 0.925 on the nine OLMo points.  The question this file
answers is whether ANY table-only functional absorbs the remaining 7.5%, and how
many were tried before one did.  The count is printed with the answer.

Nothing here is a mechanism.  It is a screen for candidate mechanisms, and its
output is a shortlist to be killed by a NEW arm, not a conclusion.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.zerotrain_20260910.why import MEASURED, OLMO, table_of  # noqa: E402

K, TH, W, D = 64, OLMO["theta"], OLMO["window"], 16384.0
OMEGA = TH ** (-np.arange(K) / K)
T_W = OMEGA * W / (2 * math.pi)


def cands(m):
    nu = OMEGA * 4.0 ** (-m)
    T = nu * D / (2 * math.pi)
    Tw = T_W                     # native in-window turns (table-independent)
    dphi = D * OMEGA * (1 - 4.0 ** (-m)) / (2 * math.pi)   # turns of phase error
    out = {}
    out["sum_m"] = m.sum()
    out["sum_m_over_ge1"] = m[T_W >= 1].sum()      # budget spent on wrapped slots
    out["sum_m_over_lt1"] = m[T_W < 1].sum()       # budget spent below one turn
    out["sum_T"] = T.sum()
    out["sum_logT_lt4"] = float(np.log1p(T[T <= 4]).sum())
    out["sum_logT_lt2"] = float(np.log1p(T[T <= 2]).sum())
    out["sum_logT_lt8"] = float(np.log1p(T[T <= 8]).sum())
    out["nT_lt2"] = float((T <= 2).sum())
    out["nT_lt4"] = float((T <= 4).sum())
    out["dphi_sum"] = float(dphi.sum())
    out["dphi_w_nu"] = float((dphi * OMEGA).sum())
    out["dphi_max"] = float(dphi.max())
    out["dphi_lt1_turns"] = float((dphi < 1.0).sum())
    out["dphi_lt0p1"] = float((dphi < 0.1).sum())
    # how much of the compression budget lands where the phase error is < 1 turn
    out["m_cheap"] = float(m[dphi < 1.0].sum())
    out["m_expensive"] = float(m[dphi >= 1.0].sum())
    # ramp steepness
    dm = np.diff(m)
    out["max_dm"] = float(dm.max())
    out["sum_dm2"] = float((dm ** 2).sum())
    out["ramp_T_span"] = float(np.log(T[T_W >= 1].max() / T.min()))
    out["n_ramp"] = float(((m > 1e-12) & (m < 1 - 1e-12)).sum())
    out["m_mid"] = float(m[(m > 0.1) & (m < 0.9)].sum())
    return out


def main():
    names = [n for n, _ in MEASURED]
    acc = np.array([a for _, a in MEASURED])
    C = {n: cands(np.asarray(table_of(n), float)) for n in names}
    keys = sorted(C[names[0]])

    def r2(x):
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, acc, rcond=None)
        res = acc - X @ beta
        return 1 - (res ** 2).sum() / ((acc - acc.mean()) ** 2).sum()

    base = r2(np.array([C[n]["sum_m"] for n in names]))
    print(f"sum(m) alone: R^2 = {base:.4f}\n")
    print(f"{len(keys)} candidates tried; single-variable R^2 and the R^2 of the "
          f"2-variable fit (sum_m, X):")
    rows = []
    for k in keys:
        x = np.array([C[n][k] for n in names])
        if x.std() == 0:
            continue
        r1 = r2(x)
        X = np.column_stack([np.array([C[n]["sum_m"] for n in names]), x])
        X = np.column_stack([np.ones(len(acc)), X])
        b, *_ = np.linalg.lstsq(X, acc, rcond=None)
        res = acc - X @ b
        r2b = 1 - (res ** 2).sum() / ((acc - acc.mean()) ** 2).sum()
        rows.append((k, r1, r2b))
    for k, r1, r2b in sorted(rows, key=lambda r: -r[2]):
        print(f"  {k:18s} alone {r1:+.4f}   with sum_m {r2b:+.4f}")

    print("\n=== does any of them fix the 12 pp pair? ===")
    i = names.index("turns_a1_b16")
    j = names.index("beta_b0p25")
    for k in keys:
        a, b = C[names[i]][k], C[names[j]][k]
        sc = max(abs(a), abs(b), 1e-12)
        print(f"  {k:18s} a1_b16={a:10.4f}  b0p25={b:10.4f}  "
              f"rel diff {(a - b) / sc:+8.3%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
