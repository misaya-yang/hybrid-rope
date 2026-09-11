#!/usr/bin/env python3
"""Free test of the constraint model against the arms the panel already scored.

This is the one check that can falsify half of this package at zero GPU cost.

The package rests on a claim about the CONSTRAINT side: that

    D_N(d) = (1/2) d^T F_N d,     d_j = ln(omega_j / nu_j)      (eps-coordinate)

measured at the native length, is the in-window damage a table does.  The panel
already carries the in-window verdict for 16 arms: its 32K column.  So the
constraint model can be tested directly, with no forwards of our own, against
numbers that were measured before this package existed:

    D_N should order the arms the way the 32K column orders them.

If it does not, the metric is wrong and every downstream solve inherits the
error -- and that is knowable before the card is touched.

Three things this buys, in order of importance:

1.  THE VETO TEST.  `E1_s28_less` (= `m28 := m27`) is a measured pure Pareto
    point: 128K 78.1250 -> 83.3333 at 32K 87.2222 -> 87.2222, i.e. +5.21 pp at
    long range for no measured in-window cost.  A Pareto-improving direction
    that already exists is a direct threat to the `G ~= 0` branch of the solve:
    `G ~= 0` asserts there is no payable long-range direction, and this one is
    payable on the panel.  The reconciliation is checkable here -- is
    `E1_s28_less` inside the native budget `eps` or outside it?

      * cost(E1_s28_less) <= eps  and the solve returns G ~= 0
            -> the linear model missed a step the panel can see.  The solve is
               falsified, and this is the finding to write up.
      * cost(E1_s28_less) >  eps
            -> it buys its +5.21 pp by spending more native budget than the
               pre-registered eps allows; the two statements are consistent and
               the solve is asked for a cheaper direction.

2.  A measured scale for `eps`.  `eps` is pre-registered in nats/token before
    the gradient is measured, so it cannot be read off this table -- but the
    table says what the budget buys.  Reporting cost(MrPro) and cost(Native)
    alongside the panel's own verdicts is what makes the choice legible instead
    of arbitrary.

3.  The KKT residual read on the winners, in the same metric the solve uses.

Reads deployed tables from `ground_truth_tables.json` -- the values that were
actually installed, not a reconstruction -- so a construction error cannot leak
in on this path.

  python -m experiments.curvature_20260910.arms \
      --gt analysis/unify_20260910/tables/ground_truth_tables.json \
      --fisher runs/fisher_mrpro_n17_32k.json --eps 1e-3 \
      --out runs/arms_mrpro_n17.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import tables as T

# The arms whose 32K/128K the 36-row panel measured, as recorded in
# analysis/unify_20260910/tables/ground_truth_tables.json.  The parse below
# reads the recorded string rather than a copy of it, so a corrected panel
# number cannot go stale here.
PANEL_RE = re.compile(r"\(\s*([-\d.]+)\s*/\s*([-\d.]+)\s*\)")

# Arms the derivation line has already judged, with the reading it gave them.
# Used only to label the output; nothing here depends on the label being right.
KNOWN = {
    "E1_s28_less": "pure Pareto point (bank-edge unload, +5.21pp long, no 32K cost)",
    "MrUni": "violates I1 -- displaces the bank edge",
    "Smooth_MrBudget": "takes budget out of the danger zone",
    "E1_pair28_29": "hole ratio overshoot, 1.46x",
    "HighGapToLong": "EVQ's claim transplanted to the frozen model",
    "E2_tail_more": "violates I2 -- pushes the tail past m=1",
    "E8_zero51": "violates I2 in shape",
    "FullLagP2_Transfer3B": "extreme point: early saturation, short-end hole",
    "StackFrontBack": "E1_s28_less (+) LBS -- a QUEUED arm, not yet measured",
    "MrProN16": "N'=16 of MrPro's own family -- a QUEUED arm, not yet measured",
    "MrProN15": "N'=15 -- a QUEUED arm, not yet measured",
}


def load_fisher(path):
    """F around the native point: the full matrix when the probe produced one,
    otherwise the diagonal.  Same rule solve_kkt.py uses, so the two cannot
    disagree about what F is."""
    rec = json.loads(Path(path).read_text())
    if rec.get("fisher_matrix"):
        F = np.asarray(rec["fisher_matrix"], dtype=np.float64)
        return 0.5 * (F + F.T), "full"
    diag = rec.get("fisher_diag")
    if not diag:
        raise ValueError(f"{path}: no fisher_matrix and no fisher_diag")
    d = np.array([float(diag[str(j)]) for j in range(T.K)])
    return np.diag(d), "diagonal"


def panel_of(entry):
    m = PANEL_RE.search(entry.get("panel") or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def eps_of(nu, native):
    """eps_j = ln(omega_j / nu_j): the compression coordinate the solver steps in.

    Native has eps = 0 by construction, which is the point -- F is measured at
    the native table, so D_N is a quadratic about eps = 0 and an arm's cost is
    its displacement from that origin.
    """
    return np.log(native / np.asarray(nu, dtype=np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fisher", required=True, help="local_probe.py receipt at the native length")
    ap.add_argument("--base", default="MrPro", help="arm the solve starts from")
    ap.add_argument("--eps", type=float, default=1e-3, help="the pre-registered native budget")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    methods = gt["methods"]

    F, kind = load_fisher(args.fisher)
    if F.shape != (T.K, T.K):
        raise ValueError(f"F is {F.shape}, expected {(T.K, T.K)}")

    native = np.asarray(methods["Native"]["nu_j"], dtype=np.float64)
    base_nu = np.asarray(methods[args.base]["nu_j"], dtype=np.float64)

    def cost(nu):
        d = eps_of(nu, native)
        return 0.5 * float(d @ F @ d), d

    c_base, d_base = cost(base_nu)
    rows = []
    for name, entry in sorted(methods.items()):
        if "nu_j" not in entry:
            continue
        nu = np.asarray(entry["nu_j"], dtype=np.float64)
        if nu.shape != (T.K,) or not (nu > 0).all() or not np.isfinite(nu).all():
            continue
        c, d = cost(nu)
        pan = panel_of(entry)
        rows.append(dict(
            arm=name, cost=c, cost_over_eps=c / args.eps,
            inside_budget=bool(c <= args.eps),
            cost_rel_base=c - c_base,
            # a positive d_j means this arm slowed clock j relative to native
            n_slots_moved=int((np.abs(d - eps_of(base_nu, native)) > 1e-4).sum()),
            max_abs_d=float(np.abs(d).max()),
            panel_32k=pan[0] if pan else None, panel_128k=pan[1] if pan else None,
            note=KNOWN.get(name, ""),
        ))

    scored = [r for r in rows if r["panel_32k"] is not None]
    rows.sort(key=lambda r: r["cost"])

    # --- report -------------------------------------------------------------
    print(f"F is {kind} ({args.fisher}); budget eps = {args.eps:g} nats/token")
    print(f"base = {args.base}, cost = {c_base:.6g} ({c_base / args.eps:.3g} x eps)\n")
    print(f"{'arm':24s} {'D_N':>12s} {'/eps':>8s} {'in':>4s} {'32K':>8s} {'128K':>8s}  note")
    print("-" * 96)
    for r in rows:
        p32 = f"{r['panel_32k']:.4f}" if r["panel_32k"] is not None else "-"
        p128 = f"{r['panel_128k']:.4f}" if r["panel_128k"] is not None else "-"
        print(f"{r['arm']:24s} {r['cost']:12.5g} {r['cost_over_eps']:8.3g} "
              f"{'yes' if r['inside_budget'] else 'no':>4s} {p32:>8s} {p128:>8s}  {r['note']}")

    # --- 1. the veto test ---------------------------------------------------
    print()
    veto = None
    if "E1_s28_less" in methods:
        _, d_v = cost(np.asarray(methods["E1_s28_less"]["nu_j"], dtype=np.float64))
        c_v = 0.5 * float(d_v @ F @ d_v)
        veto = dict(cost=c_v, over_eps=c_v / args.eps, inside=bool(c_v <= args.eps),
                    extra_over_base=c_v - c_base)
        print(f"VETO TEST -- E1_s28_less (m28 := m27), panel 87.2222/83.3333:")
        print(f"  cost {c_v:.6g} = {c_v / args.eps:.3g} x eps, "
              f"{'INSIDE' if veto['inside'] else 'OUTSIDE'} the budget")
        print(f"  it spends {c_v - c_base:+.6g} nats/token more than {args.base} "
              f"to buy +5.21 pp at 128K")
        if veto["inside"]:
            print("  -> a payable long-range direction EXISTS inside the budget.  If the solve")
            print("     returns G ~= 0 the linear model has missed it and the solve is")
            print("     FALSIFIED by a panel number measured before it existed.  Write that up.")
        else:
            print("  -> it is outside the budget: it buys its +5.21 pp by spending more native")
            print("     budget than eps allows.  Consistent with G ~= 0; the solve is then asked")
            print("     whether a CHEAPER direction exists.  Raise eps to price this arm in.")
        print()

    # --- 2. does D_N order the arms the way the 32K column does? ------------
    corr = None
    if len(scored) >= 4:
        c = np.array([r["cost"] for r in scored])
        p = np.array([r["panel_32k"] for r in scored])
        # Spearman by hand: this is a handful of points and scipy is not a dependency.
        rc = np.argsort(np.argsort(c)).astype(float)
        rp = np.argsort(np.argsort(p)).astype(float)
        corr = float(np.corrcoef(rc, rp)[0, 1])
        print(f"CONSTRAINT MODEL vs the panel's own 32K column, {len(scored)} scored arms:")
        print(f"  Spearman rho = {corr:+.3f}   (D_N is claimed to BE the in-window damage)")
        if corr < 0.3:
            print("  -> the metric does not order in-window damage.  The constraint side of the")
            print("     solve is not measuring what it claims and no solver output is readable.")
            print("     Check F first (fp32 rounding floor, delta too small), then the model.")
        else:
            print("  -> consistent in rank.  Not proof -- n is small, the arms are correlated")
            print("     with each other, and the panel is the development set -- but a metric")
            print("     that could not even order these could not carry a solve.")
        print()

    # --- 3. what the budget buys -------------------------------------------
    inside = [r["arm"] for r in rows if r["inside_budget"] and r["arm"] != "Native"]
    print(f"arms inside eps (excluding Native): {inside if inside else 'none'}")
    print("eps is PRE-REGISTERED, not chosen from this table.  What this line is for is")
    print("making the choice legible: an eps that admits no arm at all is a budget no")
    print("measured method ever paid, and an eps that admits every arm is no constraint.")

    rec = dict(fisher=args.fisher, fisher_kind=kind, eps=args.eps, base=args.base,
               base_cost=c_base, arms=rows, veto_test=veto, spearman_vs_32k=corr,
               arms_inside_budget=inside)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
