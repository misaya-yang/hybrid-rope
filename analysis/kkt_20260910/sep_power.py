#!/usr/bin/env python3
"""How much CAN 9 points separate?  Partial correlations, VIF, leave-one-out.

The task is to say, numerically, what the existing 9 OLMo points can and cannot
decide.  "They are collinear" is an opinion until it is a number.  The numbers
that decide it are:

  * partial correlations -- does S still explain the score once n_plateau and
    sum(nu) are controlled, and vice versa?  This is exactly "separation".
  * VIF -- how much does the collinearity inflate the standard error of the
    split coefficient?
  * leave-one-out -- is a correlation carried by ONE arm?  With 9 points and 4
    from one family, a single high-leverage point can manufacture a correlation.

The noise model matters and must be stated.  The RULER panel is 350 rows per
arm; the score is a mean over them.  Under a binomial reading of a mean over
350 rows the standard error is at most sqrt(.25/350)=0.027, and the panel is an
average over tasks so the effective SE is smaller, not larger.  We therefore
report coefficient SEs at sigma = 0.02 per arm, and show the sigma at which a
coefficient becomes distinguishable from zero.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.kkt_20260910.sep_cause import candidates, OLMO, OMEGA, TH, K  # noqa: E402
from experiments.zerotrain_20260910.why import MEASURED, table_of, spearman  # noqa: E402

N_ROWS = 350
# MEASURED, not assumed.  The archived per-row records for the two anchors give
# the paired contrast SE directly: MrPro vs MrProBM over the same 350 rows has
# sd(diff) = 0.4909 => SE = 0.0262, and the unpaired reading is 0.0285.  The
# per-row cross-arm correlation is only +0.19, so pairing buys almost nothing
# and 0.026 is the honest per-contrast scale.
SIGMA_PANEL = 0.026


def load():
    names = [n for n, _ in MEASURED]
    acc = np.array([a for _, a in MEASURED], dtype=float)
    C = [candidates(np.asarray(table_of(n), dtype=float)) for n in names]
    return names, acc, C


def standardise(M):
    M = np.asarray(M, float)
    mu, sd = M.mean(axis=0), M.std(axis=0)
    return (M - mu) / sd, mu, sd


def ols(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return beta, r


def partial_corr(C, acc, target, controls):
    """corr(acc, target | controls): correlate the residuals of both on the
    controls.  This IS the separation question in one number."""
    Z = np.column_stack([C[c] for c in controls] + [np.ones(len(acc))])
    _, rt = ols(Z, C[target])
    _, ra = ols(Z, acc)
    if rt.std() < 1e-12 or ra.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rt, ra)[0, 1])


def vif(C, target, others):
    Z = np.column_stack([C[o] for o in others] + [np.ones(len(C[others[0]]))])
    _, r = ols(Z, C[target])
    ss_res = float((r ** 2).sum())
    ss_tot = float(((C[target] - C[target].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    return 1.0 / max(1e-12, 1.0 - r2), r2


def main():
    names, acc, C = load()
    n = len(names)
    cols = {k: np.array([c[k] for c in C], dtype=float)
            for k in C[0]}

    print("=" * 78)
    print("A. IS mean(m) A SEPARATE CANDIDATE?")
    print("=" * 78)
    ratio = cols["mean_m"] / (cols["S"] / K)
    print(f"  max |mean(m) - S/64| over 9 arms = {np.abs(ratio - 1).max():.3e}")
    print("  => mean(m) is S divided by 64.  It is not a hypothesis; it is a")
    print("     change of units.  Any 'turning point at mean(m)=0.63' is")
    print(f"     identically 'turning point at S={0.63 * K:.1f}'.")

    print("\n" + "=" * 78)
    print("B. sum(nu): WHAT DOES IT ACTUALLY READ?")
    print("=" * 78)
    r_geo = math.exp(-math.log(TH) / K)
    print(f"  native frequency ratio omega_{{j+1}}/omega_j = {r_geo:.6f}")
    print(f"  so summing a HELD PREFIX gives a closed form:")
    print(f"    nu_held(n_held) = (1 - r^n) / (1 - r),  r = {r_geo:.6f}")
    nh = cols["n_held"]
    closed = (1 - r_geo ** nh) / (1 - r_geo)
    print(f"\n  {'n_held':>7s} {'nu_held (measured)':>20s} {'closed form':>14s}"
          f" {'rel err':>10s}")
    for v in sorted(set(nh)):
        sel = nh == v
        meas = cols["nu_held"][sel][0]
        cf = (1 - r_geo ** v) / (1 - r_geo)
        print(f"  {int(v):7d} {meas:20.6f} {cf:14.6f} {abs(meas - cf) / cf:10.2e}")
    frac = cols["nu_held"] / cols["sum_nu"]
    print(f"\n  nu_held / sum(nu): min={frac.min():.5f} max={frac.max():.5f}")
    print("  => sum(nu) is >=96% a DETERMINISTIC function of one integer,")
    print("     n_held.  It is not an independent continuous axis at all.")
    r_nu_nh = float(np.corrcoef(cols["sum_nu"], nh)[0, 1])
    print(f"  corr(sum_nu, n_held) = {r_nu_nh:+.5f}  (Spearman "
          f"{spearman(cols['sum_nu'], nh):+.4f})")

    print("\n" + "=" * 78)
    print("C. THE STRUCTURAL IDENTITY, RESTATED")
    print("=" * 78)
    print("  S = L_ramp + n_plateau                        (exact, verified)")
    print("  n_held + n_ramp + n_plateau = 64              (exact, by definition)")
    print("  => the free coordinates are (n_held, n_ramp, L_ramp);")
    print("     S is DERIVED:  S = L_ramp + 64 - n_held - n_ramp")
    print("  => 'n_plateau' and 'L_ramp' are THE SAME TEST at fixed S")
    print("     (n_plateau = S - L_ramp identically).  One hypothesis, not two.")
    for name, c in zip(names, C):
        lhs = c["S"]
        rhs = c["L_ramp"] + 64 - c["n_held"] - c["n_ramp"]
        assert abs(lhs - rhs) < 1e-9, (name, lhs, rhs)
    print("  verified on all 9 arms (assert passed).")

    print("\n" + "=" * 78)
    print("D. PARTIAL CORRELATIONS: can the 9 points separate?")
    print("=" * 78)
    tri = [("S", ["n_plateau", "sum_nu"]),
           ("n_plateau", ["S", "sum_nu"]),
           ("sum_nu", ["S", "n_plateau"]),
           ("n_held", ["S", "n_plateau"]),
           ("L_ramp", ["S", "sum_nu"])]
    print(f"\n  {'target':12s} {'controls':22s} {'r_partial':>10s} {'r_raw':>8s}")
    for t, ctl in tri:
        rp = partial_corr(cols, acc, t, ctl)
        rr = float(np.corrcoef(cols[t], acc)[0, 1])
        print(f"  {t:12s} {','.join(ctl):22s} {rp:+10.3f} {rr:+8.3f}")

    print("\n" + "=" * 78)
    print("E. VARIANCE INFLATION -- what it costs to fit the split")
    print("=" * 78)
    print(f"\n  {'target':12s} {'vs':22s} {'VIF':>10s} {'R^2':>8s} {'SE infl':>9s}")
    for t, others in [("S", ["n_plateau", "sum_nu"]),
                      ("n_plateau", ["S", "sum_nu"]),
                      ("sum_nu", ["S", "n_plateau"]),
                      ("n_held", ["S", "n_plateau"])]:
        v, r2 = vif(cols, t, others)
        print(f"  {t:12s} {','.join(others):22s} {v:10.2f} {r2:8.4f}"
              f" {math.sqrt(v):9.2f}")

    print("\n  Design [1, S, n_plateau, sum_nu] on 9 points:")
    X = np.column_stack([np.ones(n), (cols["S"] - cols["S"].mean()) / cols["S"].std(),
                         (cols["n_plateau"] - cols["n_plateau"].mean())
                         / cols["n_plateau"].std(),
                         (cols["sum_nu"] - cols["sum_nu"].mean())
                         / cols["sum_nu"].std()])
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv)) * SIGMA_PANEL
    beta, resid = ols(X, acc)
    dof = n - X.shape[1]
    print(f"  dof = {dof};  residual sd at the fitted model = "
          f"{resid.std(ddof=X.shape[1]):.4f}")
    print(f"  (residual sd >> sigma={SIGMA_PANEL} => the model is misspecified,"
          f" or noise is larger)")
    print(f"\n  {'coef':14s} {'beta':>10s} {'SE@sig=.02':>12s} {'t':>8s}")
    for lbl, b, s in zip(["intercept", "S", "n_plateau", "sum_nu"], beta, se):
        print(f"  {lbl:14s} {b:+10.4f} {s:12.4f} {b / s:8.2f}")
    print(f"\n  sigma needed for |t|>2 on n_plateau: "
          f"{abs(beta[2]) / (2 * math.sqrt(XtX_inv[2, 2])):.4f}")
    print(f"  sigma needed for |t|>2 on sum_nu:    "
          f"{abs(beta[3]) / (2 * math.sqrt(XtX_inv[3, 3])):.4f}")
    print(f"  (measured residual sd is {resid.std(ddof=X.shape[1]):.4f})")

    print("\n" + "=" * 78)
    print("F. LEAVE-ONE-OUT: is the correlation carried by ONE arm?")
    print("=" * 78)
    pairs = [("S", "sum_nu"), ("S", "n_plateau"), ("S", "n_held"),
             ("n_plateau", "sum_nu")]
    print(f"\n  {'pair':22s} {'full':>8s} " + " ".join(f"{n[:9]:>9s}" for n in names))
    for a, b in pairs:
        full = float(np.corrcoef(cols[a], cols[b])[0, 1])
        loo = []
        for i in range(n):
            keep = np.arange(n) != i
            if cols[a][keep].std() < 1e-12 or cols[b][keep].std() < 1e-12:
                loo.append(float("nan"))
            else:
                loo.append(float(np.corrcoef(cols[a][keep], cols[b][keep])[0, 1]))
        print(f"  {a + ' vs ' + b:22s} {full:+8.3f} "
              + " ".join(f"{x:+9.3f}" for x in loo))

    print("\n  Same for the SCORE correlation of each candidate:")
    print(f"\n  {'candidate':12s} {'full':>8s} " + " ".join(f"{n[:9]:>9s}"
                                                             for n in names))
    for k in ("S", "n_plateau", "sum_nu", "n_held", "L_ramp"):
        full = float(np.corrcoef(cols[k], acc)[0, 1])
        loo = []
        for i in range(n):
            keep = np.arange(n) != i
            if cols[k][keep].std() < 1e-12:
                loo.append(float("nan"))
            else:
                loo.append(float(np.corrcoef(cols[k][keep], acc[keep])[0, 1]))
        print(f"  {k:12s} {full:+8.3f} " + " ".join(f"{x:+9.3f}" for x in loo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
