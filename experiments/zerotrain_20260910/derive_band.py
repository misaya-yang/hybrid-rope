#!/usr/bin/env python3
"""Derive the optimal band from the two measured gradients -- no sweep.

THE STATEMENT.  For  min L_long  s.t.  D_N <= eps,  the Lagrangian is
L + lambda (D - eps) with lambda >= 0, and the marginal cost of compressing slot j
is

    Delta(j) = dL_long/dm_j  +  lambda * dD_N/dm_j

with lambda fixed by the constraint itself,  lambda = max(0, -<n,e>/<n,n>)  where
e and n are the two gradients (plan sec.5's residual diagnostic).

The band is then not a tuned window but a SET:

    the optimal band is  { j : Delta(j) < 0 },

because compressing a slot whose marginal Lagrangian cost is negative lowers the
constrained objective, and un-compressing one whose cost is positive lowers it
back.  Slots outside that set sit at their constraint-determined values: the fast
side at m = 0 because compressing costs more than it buys, the slow side at m = 1
because the cost has flattened to nothing and the plateau is free.

THIS IS THE "FIND THE OPTIMUM" PATH THE CAMPAIGN ASKS FOR, AND IT IS NOT A
SWEEP.  A sweep can only report the best grid point, and only for the model it
ran on; this reports a band for whatever checkpoint the two gradients were
measured on, and the same formula applies to the next checkpoint unchanged.  It
also makes YaRN's rule checkable rather than merely comparable: the deployed
family's band is "1 <= turns <= 32", so the question becomes whether the measured
Delta(j) changes sign at the turn counts 1 and 32, and if not, at which.

WHAT IT CANNOT SAY, STATED FIRST BECAUSE IT IS THE TRAP THIS PROJECT KEEPS
FALLING INTO.  Delta(j) is a FIRST-ORDER quantity at the point the gradients were
measured, and this repository has a documented history -- ten separate lineages,
re-committed twice within four hours on 2026-09-08 -- of promoting exactly this
kind of local quantity into a capability claim.  A sign change in Delta(j)
predicts where a band edge belongs TO FIRST ORDER.  It does not establish that
moving the edge there improves any task, and the run that would establish it is
the RULER sweep, not this file.  The receipt says so, and `kkt_residual.py`
carries the same caveat on the measurement side.

TWO DECLARED SIMPLIFICATIONS, both of which can move the answer:
  1. n is the gradient of the IN-WINDOW TAIL NLL, not of the plan's output-KL
     D_N.  They are different functionals and the KL one is the plan's.
  2. e is the long-range gradient of a teacher-forced answer cross-entropy on six
     rows, evaluated at MrRoPE's table.  Six rows is a small panel, and the
     gradient is taken at a point that is the incumbent rather than the optimum.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

LN4 = math.log(4.0)


def load_kkt(path):
    d = json.loads(Path(path).read_text())
    e = np.asarray(d["e"], dtype=np.float64)
    n = np.asarray(d["n"], dtype=np.float64)
    if e.shape != (64,) or n.shape != (64,):
        raise ValueError(f"expected two 64-vectors, got {e.shape} and {n.shape}")
    return d, e, n


def lambda_hat(e, n):
    """The plan's sec.5 multiplier: max(0, -<n,e>/<n,n>)."""
    nn = float(n @ n)
    if nn <= 0:
        return None, "native gradient is zero"
    return max(0.0, -float(n @ e) / nn), None


def delta_profile(e, n, lam):
    """Delta(j) = dL/dm_j + lambda dD/dm_j, in the m coordinate.

    e and n are dL/d(delta) with  freq = base * exp(-delta); since x = -log nu
    and m = (x - x^0)/ln 4, both need the same factor ln 4 to become d/dm.
    """
    return LN4 * (e + lam * n)


def band_from_profile(delta, tol_frac=0.02):
    """The set {j : Delta(j) < 0}, and the edges that define it.

    `tol_frac` is a dead zone as a fraction of the profile's own scale: a slot
    whose |Delta| is below it is called indifferent rather than assigned a side,
    because a sign that flips within the measurement's own noise is not a
    finding.  The threshold is reported.
    """
    scale = float(np.abs(delta).max())
    tol = tol_frac * scale if scale > 0 else 0.0
    neg = delta < -tol
    pos = delta > tol
    idx = np.flatnonzero(neg)
    return dict(
        tol=float(tol), scale=float(scale),
        negative_slots=[int(j) for j in idx],
        positive_slots=[int(j) for j in np.flatnonzero(pos)],
        indifferent_slots=[int(j) for j in np.flatnonzero(~neg & ~pos)],
        band_lo=(int(idx.min()) if idx.size else None),
        band_hi=(int(idx.max()) if idx.size else None),
        n_negative=int(idx.size),
    )


def turns_of(slot, theta, window, head_dim):
    """The in-window turn count of a slot, which is the coordinate YaRN's rule
    is actually written in."""
    omega = theta ** (-2.0 * slot / head_dim)
    return float(window * omega / (2.0 * math.pi))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kkt", required=True)
    ap.add_argument("--theta", type=float, default=1e6)
    ap.add_argument("--window", type=float, default=32768)
    ap.add_argument("--head-dim", type=float, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    d, e, n = load_kkt(args.kkt)
    lam, why = lambda_hat(e, n)
    if lam is None:
        print(f"cannot derive a band: {why}", file=sys.stderr)
        return 2
    delta = delta_profile(e, n, lam)
    prof = band_from_profile(delta)

    rows = []
    for j in range(64):
        rows.append(dict(slot=j, turns=turns_of(j, args.theta, args.window,
                                                args.head_dim),
                         dL_dm_long=float(LN4 * e[j]),
                         dD_dm_native=float(LN4 * n[j]),
                         delta=float(delta[j]),
                         sign=("compress" if delta[j] < -prof["tol"] else
                               "hold/uncompress" if delta[j] > prof["tol"] else
                               "indifferent")))
    out = dict(lambda_hat=float(lam), profile=prof, per_slot=rows,
               config=dict(theta=args.theta, window=args.window,
                           head_dim=args.head_dim),
               e_rows=d.get("e_rows"), n_docs=d.get("n_docs"),
               source=d.get("scope"), native_metric=d.get("native_metric"),
               scope=("FIRST-ORDER statement about where a band edge belongs at "
                      "the point the gradients were measured. It does not "
                      "establish that moving the edge improves any task; the "
                      "RULER sweep is the run that would."),
               simplifications=["n is the in-window tail NLL, not the plan's "
                                "output-KL D_N",
                                "e is a teacher-forced answer CE on six rows, at "
                                "MrRoPE's table rather than at an optimum"])
    if prof["band_lo"] is not None:
        out["band"] = [prof["band_lo"], prof["band_hi"]]
        out["band_turns"] = [turns_of(prof["band_lo"], args.theta, args.window,
                                      args.head_dim),
                             turns_of(prof["band_hi"], args.theta, args.window,
                                      args.head_dim)]
        out["yarn_rule_turns"] = [1.0, 32.0]
        out["band_matches_yarn_rule"] = bool(
            abs(out["band_turns"][1] - 1.0) < 0.5 * abs(out["band_turns"][1]) or
            abs(out["band_turns"][0] - 32.0) < 0.5 * abs(out["band_turns"][0]))

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"lambda_hat = {lam:.6g}   (from e, n)")
    print(f"Delta scale = {prof['scale']:.3f}, dead zone = {prof['tol']:.3f}")
    print(f"  slots where compressing is worth it (Delta < 0): "
          f"{prof['band_lo']} .. {prof['band_hi']}  ({prof['n_negative']} slots)")
    if "band_turns" in out:
        bt = out["band_turns"]
        print(f"  that band in TURN coordinates: [{bt[0]:.2f}, {bt[1]:.2f}]")
        print(f"  YaRN's rule says:              [1.00, 32.00]")
    print("\nper-slot Delta (compress = negative = the band wants this slot):")
    for r in rows[::4]:
        print(f"  slot {r['slot']:2d}  turns {r['turns']:8.2f}  "
              f"dL/dm {r['dL_dm_long']:+9.3f}  dD/dm {r['dD_dm_native']:+9.3f}  "
              f"Delta {r['delta']:+9.3f}  {r['sign']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
