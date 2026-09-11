#!/usr/bin/env python3
"""Rank the bank by the MEASURED long-range gradient, not by a task score.

THE ONE COMPUTATION THIS FILE DOES.  `kkt_residual.py` measures e = dL_long/dDelta
at MrRoPE's table and persists it as a 64-vector.  A table m has a long-range
gradient value

    <e, m>  =  sum_j e_j m_j ,

and the FIRST-ORDER claim of the whole campaign is that a table with a lower
<e, m> is the better long-range table.  So the measured vector ranks the bank
without running a single generation.  This file computes that ranking, and -- the
part that makes it a test rather than a tautology -- compares it against outcomes
that were measured BEFORE the gradient was:

    OLMo   16K RULER   MrRoPE 2.78%   vs BM 51.32%   (gap 48.5 points)
    Qwen  128K RULER   MrRoPE 78.13%  vs BM 70.83%   (gap -7.3 points)

The two models disagree about which of the pair is better.  If the ranking
reproduces that sign flip, the first-order model has predicted a cross-checkpoint
reversal from a local measurement; if it puts the same table on top on both
models, the first-order model is refuted and the campaign's central object
(`Delta(j) = dL/dm + lambda dD/dm`) loses its predictive claim.

WHAT IT DOES NOT SAY, DECLARED FIRST.  This is a first-order quantity evaluated at
ONE table on ONE checkpoint, and this repository has a documented history -- ten
separate lineages, re-committed twice in four hours on 2026-09-08 -- of promoting
exactly this kind of local quantity into a capability claim (ds_workspace/LESSONS
L1).  A ranking that happens to agree with a measured score is evidence, not a
theorem; a ranking that disagrees kills the model and is worth more.  The receipt
carries this caveat, and so does every table it prints.

THE GAP-PRICE TRANSFORM.  e is dL/dm per SLOT.  A table is not specified by m; the
deployed family is specified by its INCREMENTS

    eps_k = m_{lo+k} - m_{lo+k-1},   k = 1..n,

with m = 0 on slots <= lo and m = 1 on slots >= hi.  Changing eps_i moves every
slot from lo+i onward, so the increment-space gradient is the SUFFIX SUM

    g_eps[i] = sum_{k>=i} e[lo+k],

which is the plan's sec.5 `price_i = sum_{j>=i} v_j` read for this coordinate.
Both transforms are computed and both are reported, because a table can be scored
either way and they can disagree.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# The project's numeric import path is ABSOLUTE (`experiments.<pkg>.<mod>`), so
# the repo root goes on the path, not the package's own directory.  Getting this
# wrong raised ModuleNotFoundError in the first smoke run.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LN4 = math.log(4.0)

# The two checkpoints' geometry.  The band is not passed in: it is DERIVED from
# the turn rule, so the script cannot silently be given a band that flatters it.
CONFIGS = {
    "qwen": dict(theta=1e6, window=32768, head_dim=128, K=64),
    "olmo": dict(theta=5e5, window=4096, head_dim=128, K=64),
}

# The measured outcomes, entered from the archive.  These are the thing being
# predicted, so they are input data, not results of this script.
OUTCOMES = {
    "qwen": dict(arm="mrpro_n17", score=78.13, rival="beta_b1", rival_score=70.83,
                 metric="128K RULER macro, archived run_qwen3_01"),
    "olmo": dict(arm="mrpro_n17", score=2.78, rival="beta_b1", rival_score=51.32,
                 metric="16K RULER macro, archived run_ruler_newtasks_01"),
}


def native_inv_freq(theta, k=64):
    return theta ** (-np.arange(k, dtype=np.float64) / k)


def m_to_nu(m, theta):
    return native_inv_freq(theta) * np.power(4.0, -np.asarray(m, dtype=np.float64))


def derive_band(theta, window, head_dim, k=64):
    """The band as the deployed code defines it, not as a tuned constant.

    `lo` is the last slot with more than 32 in-window turns and `hi` is the first
    with fewer than one; that is exactly `fast[-1], slow[0]` in the archived
    harness's transform().  Returned as (lo, hi) with n = hi - lo increments.
    """
    turns = native_inv_freq(theta, k) * window / (2.0 * math.pi)
    fast = np.flatnonzero(turns > 32.0)
    slow = np.flatnonzero(turns < 1.0)
    if not fast.size or not slow.size:
        raise ValueError("checkpoint has no 32/1-turn boundaries")
    return int(fast[-1]), int(slow[0])


def increments_to_m(eps, lo, n, k=64):
    m = np.zeros(k, dtype=np.float64)
    m[lo + 1: lo + n + 1] = np.cumsum(np.asarray(eps, dtype=np.float64))
    m[lo + n + 1:] = m[lo + n]
    return m


def m_to_increments(m, lo, n):
    """eps_k = m_{lo+k} - m_{lo+k-1}, k = 1..n -- there are n of them.

    m[lo:lo+n+1] is the n+1 slots from the fast plateau to the slow one, so its
    diff IS the increment vector.  An earlier version prepended a 0 and returned
    n+1 entries, which then failed to dot against the n-entry gap prices; the
    off-by-one is worth naming because both vectors are plausible-looking.
    """
    return np.diff(np.asarray(m, dtype=np.float64)[lo: lo + n + 1])


def suffix_sum(v):
    return np.cumsum(np.asarray(v, dtype=np.float64)[::-1])[::-1]


def load_bank(cfg, lo, hi):
    """Every construction the project has, as (name, m[64])."""
    from experiments.curvature_20260910 import tables as T
    import phase1_screen as P
    out = []
    for name, build in sorted(T.CONSTRUCTIONS.items()):
        try:
            m = np.asarray(build(), dtype=np.float64)
        except Exception as exc:                       # pragma: no cover
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        if m.shape != (64,) or not np.isfinite(m).all():
            continue
        out.append((name, m))
    for name, build in (("beta_b0p25", lambda: P.m_incr_beta(0.25)),
                        ("beta_b0p5", lambda: P.m_incr_beta(0.5)),
                        ("beta_b2", lambda: P.m_incr_beta(2.0))):
        out.append((name, np.asarray(build(), dtype=np.float64)))
    return out


def lambda_hat(e, n):
    """The plan's sec.5 multiplier: max(0, -<n,e>/<n,n>)."""
    nn = float(n @ n)
    if nn <= 0:
        return None, "native gradient is zero; the constraint prices nothing"
    return max(0.0, -float(n @ e) / nn), None


def score_long(e, m):
    """<e, m>: the UNCONSTRAINED first-order long-range value. Lower is better.

    THIS STATISTIC IS NOT A RANKING AND IS NOT USED AS ONE.  If e is negative
    everywhere -- and on the measured Qwen gradient it nearly is -- then <e, m> is
    minimised by m = 1 on every slot, so the "winner" is the pure-interpolation
    table, which is YaRN taken to its limit and is measured to be WORSE than
    MrRoPE at long range.  A statistic whose optimum is a known-bad table cannot
    rank tables.  It is computed and printed because the plan asks for it and
    because its failure is itself the argument for the constrained form; the
    ranking below uses the Lagrangian.
    """
    return float(np.dot(e, m))


def score_lagrangian(e, n, lam, m):
    """<e + lambda*n, m>: the ranking statistic, and the plan's own object.

    lambda is fixed by the constraint itself rather than tuned, so this has no
    free parameter.  A table with a lower value is one the constrained problem
    prefers, which is a statement about directions at the measurement point and
    NOT a capability claim.
    """
    if lam is None:
        return float("nan")
    return float(np.dot(e + lam * np.asarray(n, dtype=np.float64), m))


def score_eps(e, m, lo, n):
    """The same thing written in the coordinate the family is built in."""
    g = suffix_sum(e[lo + 1: lo + n + 1])
    eps = m_to_increments(m, lo, n)
    return float(np.dot(g, eps))


def roughness(eps):
    ext = np.concatenate(([0.0], np.asarray(eps, dtype=np.float64), [0.0]))
    return float(np.sum(np.diff(ext) ** 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kkt", required=True, help="kkt.json from kkt_residual.py")
    ap.add_argument("--model", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)

    cfg = CONFIGS[args.model]
    theta, window, hd = cfg["theta"], cfg["window"], cfg["head_dim"]
    lo, hi = derive_band(theta, window, hd)
    n = hi - lo
    print(f"[{args.model}] theta {theta:g}  window {window}  head_dim {hd}")
    print(f"[{args.model}] derived band = [{lo}, {hi}]  n = {n} increments")

    d = json.loads(Path(args.kkt).read_text())
    e = LN4 * np.asarray(d["e"], dtype=np.float64)          # dL/dm, not dL/dDelta
    nv = LN4 * np.asarray(d["n"], dtype=np.float64)
    if e.shape != (64,) or nv.shape != (64,):
        raise ValueError(f"expected two 64-vectors, got {e.shape} and {nv.shape}")
    lam, why = lambda_hat(e, nv)
    print(f"[{args.model}] e from {d.get('e_rows')} long-range rows, "
          f"|e| = {np.linalg.norm(e):.3f}")
    print(f"[{args.model}] |e| on the band = {np.linalg.norm(e[lo:hi+1]):.3f}, "
          f"outside = {np.linalg.norm(np.delete(e, np.s_[lo:hi+1])):.3f}")
    print(f"[{args.model}] lambda_hat = "
          f"{'None (' + why + ')' if lam is None else f'{lam:.6g}'}")

    g_eps = suffix_sum(e[lo + 1: lo + n + 1])
    print(f"[{args.model}] gap prices (suffix sums of e over the band) = "
          f"{np.round(g_eps, 3).tolist()}")

    bank = load_bank(cfg, lo, hi)
    rows = []
    for name, m in bank:
        eps = m_to_increments(m, lo, n)
        rows.append(dict(
            arm=name, long=score_long(e, m),
            lagrangian=score_lagrangian(e, nv, lam, m),
            long_eps=score_eps(e, m, lo, n),
            roughness=roughness(eps), sum_m=float(m.sum()),
            held=int(np.sum(m < 1e-12)), plateau=int(np.sum(m > 1 - 1e-12))))
    rows.sort(key=lambda r: (r["lagrangian"] if np.isfinite(r["lagrangian"])
                             else r["long"]))

    print(f"\n=== ranked by the Lagrangian <e + lambda*n, m>  "
          f"(lower = the CONSTRAINED problem prefers it) ===")
    print(f"{'arm':22s} {'<e+ln,m>':>10s} {'<e,m>':>10s} {'R(eps)':>9s} "
          f"{'sum m':>7s} {'held':>5s} {'plateau':>7s}")
    for r in rows[: args.top]:
        print(f"{r['arm']:22s} {r['lagrangian']:10.3f} {r['long']:10.3f} "
              f"{r['roughness']:9.5f} {r['sum_m']:7.3f} {r['held']:5d} "
              f"{r['plateau']:7d}")
    print("\nthe pure-interpolation table is deliberately kept in the listing: it "
          "minimises <e,m> whenever e<0 everywhere, which is why <e,m> cannot rank.")

    # ---- THE TEST: does the ranking put the right one on top? -------------
    occ = OUTCOMES[args.model]
    by = {r["arm"]: r for r in rows}
    if occ["arm"] in by and occ["rival"] in by:
        a, b = by[occ["arm"]], by[occ["rival"]]
        measured_sign = ("MrRoPE better" if occ["score"] > occ["rival_score"]
                         else "BM better")
        print(f"\n=== the outcome being predicted ({occ['metric']}) ===")
        print(f"  {occ['arm']:12s} measured {occ['score']:6.2f}")
        print(f"  {occ['rival']:12s} measured {occ['rival_score']:6.2f}")
        print(f"  measured           : {measured_sign}")
        # The pair shares a band, a gain and a budget, so the difference of the
        # two statistics is exactly the shape term -- no normalisation needed.
        verdicts = {}
        for label, key in (("<e,m> (unconstrained)", "long"),
                           ("<e+ln,m> (Lagrangian)", "lagrangian")):
            if not np.isfinite(a[key]) or not np.isfinite(b[key]):
                continue
            pred = "MrRoPE better" if a[key] < b[key] else "BM better"
            verdicts[key] = dict(predicted=pred, agree=(pred == measured_sign),
                                 mrpro=a[key], bm=b[key])
            print(f"  {label:24s} MrRoPE {a[key]:+9.3f}  BM {b[key]:+9.3f}  "
                  f"-> {pred:14s} {'AGREE' if pred == measured_sign else 'REFUTED'}")
        agree = all(v["agree"] for v in verdicts.values()) if verdicts else None
        verdict = ("first-order model reproduces this checkpoint's sign"
                   if agree else
                   "FIRST-ORDER MODEL REFUTED on this checkpoint by the Lagrangian")
    else:
        verdict = "one of the two endpoint arms is missing from the bank"
        agree = None
        verdicts = {}
        print(f"\n=== cannot test: {verdict} ===")

    res = dict(model=args.model, band=[lo, hi], n=n, config=cfg,
               lambda_hat=lam, lambda_why=why, e_norm=float(np.linalg.norm(e)),
               n_norm=float(np.linalg.norm(nv)), gap_prices=g_eps.tolist(),
               ranked=rows, outcome=occ, agree=agree, verdicts=verdicts,
               verdict=verdict,
               scope=("FIRST-ORDER ranking from ONE gradient measured at ONE table "
                      "on ONE checkpoint. Not a capability claim; see "
                      "ds_workspace/LESSONS L1. The test is whether it "
                      "reproduces a sign measured before the gradient existed."))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
