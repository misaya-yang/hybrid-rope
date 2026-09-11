#!/usr/bin/env python3
"""The allocation problem, solved in closed form instead of searched.

Everything lives in the compression coordinate

    eps_j := ln(omega_j / nu_j)          native eq 0,  MrRoPE eq m_j ln S

so a table is a point eps in R^64 and a step is delta.  On the distribution the
checkpoint was trained for, a step costs output drift

    D_N(delta) = (1/2) delta^T F_N delta + O(||delta||^3),   F_N = output-KL Fisher >= 0

and buys, to first order, the long-range objective

    L(eps_b + delta) - L(eps_b) = g^T delta + O(||delta||^2)

subject to the linear budget constraints A delta = 0 (both endpoints pinned,
total log-span fixed).  The local problem is a trust-region step in the F_N
metric, and while no ordering or box constraint is active its solution is
unique and closed form:

    delta* = -sqrt( 2 eps / (g^T P_F g) ) * P_F g
    P_F    = F_N^-1 - F_N^-1 A^T (A F_N^-1 A^T)^-1 A F_N^-1

with first-order long-range gain  -g^T delta* = sqrt(2 eps g^T P_F g).  This is
Cauchy-Schwarz in the F_N inner product, not a model choice.  Read it as:
whiten every frequency direction by the catastrophic-forgetting risk the frozen
checkpoint attaches to it, then move along the whitened long-range gradient.
It keeps the cross-frequency coupling that a per-slot benefit/cost ratio drops.

THE DECISIVE NUMBER IS G = g^T P_F g.

  G ~ 0   the long gradient has no component the native metric can pay for.
          No local step improves the task loss at any affordable native price.
          That is a positive statement about MrRoPE: it is a KKT point of the
          constrained problem, and the theory EXPLAINS it rather than beating it.
  G > 0   there is a measured direction that buys long-range improvement for a
          definite, pre-computable native price.  The theory BEATS it, and the
          predicted gain sqrt(2 eps G) is a falsifiable number.

Either branch is a result; neither depends on a curve search.

  python -m experiments.curvature_20260910.solve_kkt \
      --fisher runs/fisher_mrpro_32k.json --long runs/gl_mrpro_128k.json \
      --eps 1e-3 --out runs/kkt_table.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from . import tables as T


# ---------------------------------------------------------------------------
def load_fisher(path, slots):
    """-> (F[64,64] or None, diag[64], source label)."""
    rec = json.load(open(path))
    if rec.get("fisher_matrix"):
        F = np.asarray(rec["fisher_matrix"], dtype=np.float64)
        F = 0.5 * (F + F.T)
        return F, np.diag(F).copy(), rec.get("table", "?")
    diag = np.zeros(T.K)
    seen = np.zeros(T.K, dtype=bool)
    for k, v in (rec.get("fisher_diag") or {}).items():
        diag[int(k)], seen[int(k)] = float(v), True
    missing = [j for j in slots if not seen[j]]
    if missing:
        raise ValueError(f"{path} has no Fisher diagonal for slots {missing}")
    return None, diag, rec.get("table", "?")


def load_grad(path, slots):
    """-> (g[64], seen[64], record).  Unmeasured slots are left at zero and
    flagged by `seen`; every consumer here filters on it."""
    rec = json.load(open(path))
    g = np.zeros(T.K)
    seen = np.zeros(T.K, dtype=bool)
    for k, v in rec["grad"].items():
        g[int(k)], seen[int(k)] = float(v), True
    missing = [j for j in slots if not seen[j]]
    if missing:
        raise ValueError(f"{path} has no gradient for slots {missing}")
    return g, seen, rec


# ---------------------------------------------------------------------------
def constraint_matrix(pin_ends=True, fix_sum=False, extra_rows=()):
    """Rows of A; A delta = 0 pins the fast end, the slow end and optionally the
    total log-span.  Leaving A empty means the solver is free to move the
    endpoints too, which is the PI-like direction the geometric families never
    explore -- worth running as a second arm, not as the main one."""
    rows = list(extra_rows)
    if pin_ends:
        rows.append(np.eye(T.K)[0])
        rows.append(np.eye(T.K)[-1])
    if fix_sum:
        rows.append(np.ones(T.K))
    return np.array(rows) if rows else np.zeros((0, T.K))


def price_diagonal(diag, slots, floor_frac=1e-9):
    """Make the diagonal safe to invert, and say which slots were not priced.

    A negative or near-zero F_jj is not a cheap frequency -- it is a slot whose
    native cost this probe could not measure (the perturbation response sits
    under the forward's own rounding, which is what a negative diagonal means
    since the true Fisher is PSD).  Inverting it either flips the sign of the
    cost or sends 1/F to ~1e300 and the step to infinity.

    The conservative reading, and the default here: an unpriced slot is PINNED,
    not freed.  Buying long-range budget with a slot whose in-window price is
    unknown is exactly how a table gets released that then fails the native
    check, and the whole point of this exercise is that the native side is the
    binding constraint.  Set `free` explicitly to spend it anyway -- that is a
    different, more aggressive arm and should be labelled as one.

    Pinning is done by inflating the diagonal rather than by adding a constraint
    row, so the projection stays a pure function of the constraints the caller
    declared.
    """
    d = np.asarray(diag, dtype=np.float64).copy()
    priced = np.ones(T.K, dtype=bool)
    big = float(np.max(np.abs(d[slots]))) if len(slots) else 1.0
    for j in range(T.K):
        if d[j] <= floor_frac * big:
            priced[j] = False
            d[j] = big * 1e9
    return d, priced, [int(j) for j in np.flatnonzero(~priced)]


def inverse(F, diag):
    if F is None:
        return np.diag(1.0 / np.clip(diag, 1e-300, None))
    w, V = np.linalg.eigh(F)
    w = np.clip(w, max(1e-12 * float(w.max()), 1e-300), None)
    return (V / w) @ V.T


def projected_gradient(F, diag, g, A):
    """P_F g and G = g^T P_F g.  G is the whole answer."""
    Finv = inverse(F, diag)
    Finv_g = Finv @ g
    if A.shape[0]:
        M = A @ Finv @ A.T
        PFg = Finv_g - Finv @ A.T @ np.linalg.solve(M + 1e-14 * np.eye(M.shape[0]), A @ Finv_g)
    else:
        PFg = Finv_g
    return PFg, float(g @ PFg)


def projected_step(F, diag, g, A, eps):
    PFg, G = projected_gradient(F, diag, g, A)
    if G <= 0:
        return dict(d=np.zeros(T.K), PFg=PFg, G=G, kappa=None,
                    pred_native_kl=0.0, pred_long_gain=0.0, degenerate=True)
    kappa = math.sqrt(2.0 * eps / G)
    d = -kappa * PFg
    M = F if F is not None else np.diag(diag)
    return dict(d=d, PFg=PFg, G=G, kappa=kappa, degenerate=False,
                pred_native_kl=0.5 * float(d @ M @ d),
                pred_long_gain=float(-(g @ d)))


def monotone_report(nu):
    d = np.diff(np.asarray(nu, dtype=np.float64))
    bad = np.flatnonzero(d > 0)
    return dict(ordered=bool(bad.size == 0), violations=[int(j) for j in bad],
                worst_rise=float(d.max()) if d.size else 0.0,
                nu_min=float(np.min(nu)), nu_max=float(np.max(nu)))


def lambda_spectrum(g_native, g_long, floor_frac=0.05):
    """Secondary diagnostic at a candidate table: lambda_j = -g_Lj / g_Nj.

    Reading the two LOSSES (in-window and long) rather than the KL metric.  One
    lambda across the free slots is the KKT signature of a two-loss trade-off;
    the spread is measured evidence against it.  Slots whose native gradient is
    under the floor are reported separately as free money -- movable at no
    in-window cost, which is where a uniform allocation is most likely to have
    left budget unspent.
    """
    gN, gL = np.asarray(g_native), np.asarray(g_long)
    peak = float(np.abs(gN).max())
    floor = floor_frac * peak if peak > 0 else 0.0
    free = np.abs(gN) <= floor
    lam = np.full(T.K, np.nan)
    nz = ~free
    lam[nz] = -gL[nz] / gN[nz]
    ok = np.isfinite(lam)
    return dict(lambda_j={str(j): (None if not np.isfinite(lam[j]) else float(lam[j])) for j in range(T.K)},
                free_slots=[int(j) for j in np.flatnonzero(free)],
                free_slot_long_grad={str(int(j)): float(gL[j]) for j in np.flatnonzero(free)},
                lambda_median=float(np.median(lam[ok])) if ok.any() else None,
                lambda_iqr=float(np.subtract(*np.percentile(lam[ok], [75, 25]))) if ok.any() else None,
                lambda_cv=(float(np.std(lam[ok]) / abs(np.mean(lam[ok])))
                           if ok.any() and np.mean(lam[ok]) else None),
                floor=floor)


# ---------------------------------------------------------------------------
def build_receipt(base_table, fisher_path, long_path, eps, PFg, G, step, A, slots,
                  seen, diag, F, judge=None, unpriced=(), free_unpriced=False):
    base = T.build(base_table)
    d = step["d"]
    nu_new = base["values_float32"] * np.exp(-d)          # d = +Delta eps = -Delta ln nu
    m_new = base["m"] + d / T.LN_S
    big = np.abs(d) / T.LN_S > 1e-4
    rec = dict(
        base_table=base_table, fisher_source=fisher_path, long_source=long_path,
        slots=slots, n_constraints=int(A.shape[0]),
        eps=eps, G=G, G_per_slot=G / len(slots), kappa=step["kappa"],
        degenerate=step["degenerate"],
        pred_native_kl=step["pred_native_kl"], pred_long_gain=step["pred_long_gain"],
        # the falsifiable headline: "this much long-loss improvement, this much
        # output-KL drift".  forward_check.py measures both for real.
        d={str(j): float(d[j]) for j in range(T.K)},
        PFg={str(j): float(PFg[j]) for j in range(T.K)},
        # carried so forward_check.py can budget-match its controls inside this
        # same metric; without it the controls would compare budgets, not directions
        fisher_diag={str(j): float(diag[j]) for j in range(T.K)},
        m=[float(x) for x in base["m"]], m_new=[float(x) for x in m_new],
        nu_new=[float(x) for x in nu_new], gain=base["gain"], theta=base["theta"],
        d_norm=float(np.linalg.norm(d)), d_max=float(np.abs(d).max()),
        slots_moved=[int(j) for j in np.flatnonzero(big)],
        monotone=monotone_report(nu_new),
        fisher_is_full_matrix=F is not None,
        unpriced_slots=unpriced,
        unpriced_policy=("freed at the native gap" if free_unpriced else "pinned"),
    )
    if F is not None:
        # The precise diagonality question, asked on the step that is actually
        # taken rather than on an all-ones direction: how much of this step's
        # quadratic form does the diagonal alone account for?  Low means the
        # solver's answer depends on cross-frequency structure a per-slot
        # benefit/cost reading cannot see.
        qd = float(d @ np.diag(np.diag(F)) @ d)
        qt = float(d @ F @ d)
        rec["step_diag_share"] = (qd / qt) if qt else None
    if judge:
        rec["judge"] = judge
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fisher", required=True, help="local_probe json: output KL / F_N")
    ap.add_argument("--long", required=True, help="long_grad json: g_L")
    ap.add_argument("--native-grad", default=None,
                    help="long_grad json measured on in-window inputs, for the lambda diagnostic")
    ap.add_argument("--base-table", default="mrpro_n17", choices=sorted(T.CONSTRUCTIONS))
    # choices= so a typo names the valid sets instead of dying in SLOT_SETS[...]
    # with a bare KeyError.  Same guard long_grad.py and forward_check.py carry.
    ap.add_argument("--slots", default="wide",
                    choices=["all", "bridge", "wide"])
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="pre-registered native output-KL budget, nats/token")
    ap.add_argument("--eps-sweep", action="store_true",
                    help="also report the budget curve (descriptive only; the "
                         "operating point is --eps and must be fixed in advance)")
    ap.add_argument("--no-pin-ends", action="store_true")
    ap.add_argument("--fix-sum", action="store_true")
    ap.add_argument("--free-unpriced", action="store_true",
                    help="let slots whose native cost was not measurable move anyway, "
                         "pricing them at their own native gap.  Off by default: an "
                         "unpriced slot is pinned, because spending budget you cannot "
                         "price is how a table passes the local check and fails the "
                         "native one.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from .long_grad import SLOT_SETS
    slots = SLOT_SETS[args.slots]
    F, diag, fsrc = load_fisher(args.fisher, slots)
    gL, seen, lrec = load_grad(args.long, slots)
    if F is None:
        diag, priced, unpriced = price_diagonal(diag, slots)
        if args.free_unpriced:
            for j in unpriced:
                diag[j] = 1.0 / T.LN_S                      # the unmeasured slot's own gap, as a stand-in
            priced = np.ones(T.K, dtype=bool)
    else:
        # with the full matrix the unpriced slots are handled by the spectrum,
        # but report them anyway: they are where a diagonal solver would have
        # gone wrong
        _, priced, unpriced = price_diagonal(np.diag(F), slots)
    A = constraint_matrix(pin_ends=not args.no_pin_ends, fix_sum=args.fix_sum)
    step = projected_step(F, diag, gL, A, args.eps)

    judge = None
    if args.native_grad:
        gN, _, nrec = load_grad(args.native_grad, slots)
        judge = lambda_spectrum(gN, gL)
        judge["native_grad_source"] = args.native_grad
        judge["base_native_loss"] = nrec.get("base_loss")

    rec = build_receipt(args.base_table, args.fisher, args.long, args.eps,
                        step["PFg"], step["G"], step, A, slots, seen, diag, F, judge,
                        unpriced=unpriced, free_unpriced=args.free_unpriced)

    if args.eps_sweep:
        rec["budget_curve"] = {
            f"{e:g}": dict(pred_long_gain=projected_step(F, diag, gL, A, e)["pred_long_gain"])
            for e in (1e-5, 1e-4, 1e-3, 1e-2)}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)

    verdict = ("EXPLAINS (no payable direction)" if rec["degenerate"] or rec["G"] <= 0
               else "CAN BEAT (payable direction found)")
    if unpriced:
        print(f"NOTE: {len(unpriced)} slot(s) had no measurable native cost and were "
              f"{rec['unpriced_policy']}: {unpriced[:12]}"
              f"{' ...' if len(unpriced) > 12 else ''}")
    print(f"G = g^T P_F g = {rec['G']:.6e}   ->  {verdict}")
    if not rec["degenerate"]:
        print(f"predicted long-loss gain at eps={args.eps:g}: {rec['pred_long_gain']:.6e} nats/token")
        print(f"predicted native output KL:                  {rec['pred_native_kl']:.6e}")
        print(f"step ||d||_inf = {rec['d_max']/T.LN_S:.4f} in m-units over {len(rec['slots_moved'])} slots")
        print(f"monotone after step: {rec['monotone']['ordered']}"
              + ("" if rec['monotone']['ordered']
                 else f"  VIOLATIONS at {rec['monotone']['violations'][:8]}"))
    if judge:
        print(f"lambda median {judge['lambda_median']}  IQR {judge['lambda_iqr']}  "
              f"CV {judge['lambda_cv']}  free slots {judge['free_slots']}")
    print("\n  slot      m_before -> m_after")
    for j in range(T.K):
        if abs(rec["m_new"][j] - rec["m"][j]) > 1e-5:
            print(f"  {j:4d}   {rec['m'][j]:8.4f} -> {rec['m_new'][j]:8.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
