#!/usr/bin/env python3
"""coverage_theory_20260911.py — the ceiling/coverage theory of RoPE tables.

A frozen model trained at window W "knows", for each slot j, phase patterns with
turn counts up to t_j(W).  A compression table m_j does not change what the slot
knows; it CHANGES WHERE IN DISTANCE that competence is delivered:

    slot j compressed by m_j presents, at test distance d, the turn count
        T_j(d) = t_j(W) * (d/W) * 4^{-m_j}.

Usable (in-distribution AND informative) at depth u = log4(d/W):

    T_j in [delta, t_j(W)]   <=>   m_j - min(kappa_j, L) <= u <= m_j
    kappa_j = log4(t_j(W)/delta)      (signal room)
    L       = over-compression allowance (calibrate: 1, 2, 3, inf)

So m_j is literally a CEILING POSITION: slot j covers a window of depths that
ends at u = m_j and hangs down by min(kappa_j, L).  The table's job is to tile
the extrapolation zone with these windows.  Everything else the campaign
measured falls out of this reading; the script below checks each claim against
every arm that has a measured score.

Pure numpy, CPU only.  Run from anywhere:
    python ds_workspace/recon_20260910/code/coverage_theory_20260911.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

K = 64
THETA = 5.0e5
W = 4096
LN4 = math.log(4.0)

tW = (W / (2 * math.pi)) * THETA ** (-np.arange(K) / K)   # training turns per window
g = np.log2(tW)                                            # affine in j: g = G - lambda*j
LAM = math.log(THETA) / (K * math.log(2.0))                # octaves per slot (0.29580...)

DELTA = 0.25                                               # informative floor, turns
kappa = np.log(tW / DELTA) / LN4
ALIVE = tW >= DELTA                                        # slots dead in training stay dead


# ---------------------------------------------------------------- tables
# Constructors copied VERBATIM from the server's phase1_screen.py so that
# every reconstructed m is the array that was actually run (verified by sum_m).
def m_incr_beta(b, n=18, lo=14):
    kk = np.arange(1, n + 1, dtype=np.float64)
    w = kk * np.power(n + 1 - kk, float(b))
    eps = w / w.sum()
    mq = np.concatenate([[0.0], np.cumsum(eps)])
    out = np.zeros(K)
    out[lo:lo + n + 1] = mq
    out[lo + n + 1:] = 1.0
    return out


def band_from_turns(alpha, beta, theta=THETA, window=W, head_dim=128):
    lo = head_dim * math.log(window / (float(beta) * 2.0 * math.pi)) / (2.0 * math.log(theta))
    hi = head_dim * math.log(window / (float(alpha) * 2.0 * math.pi)) / (2.0 * math.log(theta))
    return int(math.floor(lo)), int(math.ceil(hi))


def m_turns(alpha, beta, ramp="mrpro"):
    lo, hi = band_from_turns(alpha, beta)
    lo, hi = max(0, lo), min(K - 1, hi)
    n = hi - lo
    q = np.arange(1, n + 1, dtype=np.float64)
    if ramp == "mrpro":
        eps = q / q.sum()
    elif ramp == "beta1":
        w = q * (n + 1 - q)
        eps = w / w.sum()
    elif ramp == "linear":
        eps = np.full(n, 1.0 / n)
    m = np.zeros(K)
    m[lo + 1:hi + 1] = np.cumsum(eps)
    m[hi + 1:] = 1.0
    return m


def m_C42(lo=14, n=18):
    kk = np.arange(1, n + 1, dtype=np.float64)
    p = 6.0 * kk * (n + 1 - kk) / (n * (n + 1) * (n + 2))
    eps = p * (1.0 - (10.0 / 119.0) * (kk - 19.0 / 2.0))
    m = np.zeros(K)
    m[lo + 1:lo + n + 1] = np.cumsum(eps)
    m[lo + n + 1:] = 1.0
    return m


def m_C42V24(lo=14, n=18):
    kk = np.arange(1, n + 1, dtype=np.float64)
    p = 6.0 * kk * (n + 1 - kk) / (n * (n + 1) * (n + 2))
    eps = p * (1.0 - (10.0 / 119.0) * (kk - 19.0 / 2.0)
               + (35.0 / 1496.0) * ((kk - 19.0 / 2.0) ** 2 - 357.0 / 20.0))
    m = np.zeros(K)
    m[lo + 1:lo + n + 1] = np.cumsum(eps)
    m[lo + n + 1:] = 1.0
    return m


def m_step(hi):
    m = np.zeros(K)
    m[hi:] = 1.0
    return m


def evq_phi(tau):
    u = (np.arange(K, dtype=np.float64) + 0.5) / K
    if abs(tau) < 1e-8:
        return u
    return 1.0 - np.arcsinh((1.0 - u) * math.sinh(tau)) / tau


def m_evq_shift(tau):
    # canonical EVQ on THIS checkpoint's theta (the geometry-fixed version)
    return (evq_phi(tau) - (np.arange(K, dtype=np.float64) + 0.5) / K) * math.log(THETA) / LN4


def m_walk(a, m_a=m_incr_beta(1.0, 21, 11), m_b=m_incr_beta(1.0, 18, 14)):
    return (1 - a) * m_b + a * m_a


def shifted(base, s=0.5, lo=15):
    """Ceiling shift: add s to every ceiling at or above slot lo (8x prescription)."""
    m = base.copy()
    m[lo:] = m[lo:] + s
    return m


ARMS = {
    # name: (m, ruler350 or None, contNLL@16384 or None)
    "MrRoPE":        (m_incr_beta(0.0),            0.0709, 3.6887),
    "beta_b0.25":    (m_incr_beta(0.25),           0.1447, None),
    "beta_b0.5":     (m_incr_beta(0.5),            0.2320, None),
    "BM":            (m_incr_beta(1.0),            0.4167, 2.8627),
    "beta_b2":       (m_incr_beta(2.0),            0.5001, None),
    "b3_lo14":       (m_incr_beta(3.0),            0.5587, 2.8267),
    "b2_wide":       (m_incr_beta(2.0, 21, 11),    None,   2.8449),
    "b3_wide":       (m_incr_beta(3.0, 21, 11),    None,   2.8244),
    "b4_wide":       (m_incr_beta(4.0, 21, 11),    0.5433, 2.8216),
    "turns_a1_b64":  (m_turns(1, 64, "beta1"),     0.5384, 2.8402),
    "turns_a2_b32":  (m_turns(2, 32, "mrpro"),     0.5100, 2.8846),
    "turns_a1_b16":  (m_turns(1, 16, "beta1"),     0.2651, None),
    "turns_a05_b32": (m_turns(0.5, 32, "mrpro"),   0.1604, None),
    "ctl_C42":       (m_C42(),                     0.4367, 2.9417),
    "ctl_C42V24":    (m_C42V24(),                  0.5440, 2.8309),
    "step_hi25":     (m_step(25),                  0.1121, 3.0452),
    "evq_shift_t0p5":(m_evq_shift(0.5),            0.0,    None),
    "evq_shift_t1p0":(m_evq_shift(1.0),            0.0,    None),
    "evq_shift_t2p0":(m_evq_shift(2.0),            0.0,    None),
    "native":        (np.zeros(K),                 None,   None),   # known unusable at 4x
    "interp":        (np.ones(K),                  None,   None),   # known unusable
    "walk_a0p25":    (m_walk(0.25),                None,   None),
    "walk_a0p5":     (m_walk(0.5),                 None,   None),
    "walk_a0p75":    (m_walk(0.75),                None,   None),
    "shift8x_BM":    (shifted(m_incr_beta(1.0)),   None,   None),   # variant A: +0.5 shift
    "shift8x_wide":  (shifted(m_incr_beta(1.0, 21, 11)), None, None),
    "scale8x_BM":    (1.5 * m_incr_beta(1.0),      None,   None),   # variant B: amplitude scale
    "scale8x_wide":  (1.5 * m_incr_beta(1.0, 21, 11), None, None),
    "step42":        (m_step(22),                  None,   2.8795),   # Pro §9.3, running now
    "Tstar":         (m_step(19),                  None,   3.0372),   # hole family, archived
}


# ---------------------------------------------------------------- quantities
def coverage(m, u, L=2.0):
    """# slots usable at depth u: alive, ceiling reaches u, not over-compressed."""
    reach = np.minimum(kappa, L)
    lo = m - reach
    return int(np.sum(ALIVE & (m >= u - 1e-12) & (lo <= u + 1e-12)))


def coverage_curve(m, L=2.0, umax=1.6, n=161):
    us = np.linspace(0.0, umax, n)
    return us, np.array([coverage(m, u, L) for u in us])


def mismatch(m, u):
    """mean-abs distance (log2-turn units / 2) from the depth-u configuration to
    the training manifold {config(u') : u' <= 0}."""
    o = u - m[ALIVE]                       # offset vector of alive slots
    # best c <= 0 minimising mean|o - c|: median if median<=0 else 0
    cs = np.concatenate([[0.0], np.sort(o)[:-1]])  # candidate breakpoints <= 0
    best = np.inf
    for c in cs[o if False else None] if False else [c for c in cs if c <= 1e-12]:
        v = np.mean(np.abs(o - c))
        if v < best:
            best = v
    return float(best)


def spectrum_gaps(m):
    dlog_nu = np.diff(g - 2.0 * m)         # log2 nu spacing between consecutive slots
    return dlog_nu


def N_evq(m, D_fac=4.0, band=(0.25, 16.0)):
    tD = tW * D_fac * np.power(4.0, -m)
    return int(np.sum((tD >= band[0]) & (tD <= band[1])))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


# ---------------------------------------------------------------- main
def main():
    out = {}
    print(f"config: theta={THETA:.0f} W={W} K={K} lambda={LAM:.6f} oct/slot")
    print(f"alive slots (tW>=delta): j in [0, {np.max(np.where(ALIVE)[0])}] "
          f"({ALIVE.sum()} slots); dead = {K - ALIVE.sum()}\n")

    # verify reconstruction against recorded sum_m
    recorded = {"MrRoPE": 37.667, "BM": 40.5, "turns_a1_b64": 42.0, "b3_lo14": 43.637,
                "ctl_C42": 42.0, "ctl_C42V24": 42.0, "evq_shift_t0p5": -6.157,
                "evq_shift_t1p0": -22.952, "evq_shift_t2p0": -72.223}
    print("== reconstruction check (sum_m vs recorded) ==")
    for name, val in recorded.items():
        s = float(ARMS[name][0].sum())
        flag = "OK " if abs(s - val) < 0.01 else "FAIL"
        print(f"  {flag} {name:16s} reconstructed {s:9.3f}  recorded {val:9.3f}")

    rows = []
    for name, (m, ruler, cont) in ARMS.items():
        us, n = coverage_curve(m)
        zone = (us >= 0.02) & (us <= 0.98)
        rows.append(dict(
            name=name, sum_m=float(m.sum()), N=N_evq(m),
            n_min=float(n[zone].min()), n_int=float(n[zone].mean()),
            n_u03=float(coverage(m, 0.3)), n_u05=float(coverage(m, 0.5)),
            n_u07=float(coverage(m, 0.7)), n_u095=float(coverage(m, 0.95)),
            n_u15=float(coverage(m, 1.5)),
            mis_mid=mismatch(m, 0.5), mis_deep=mismatch(m, 0.95),
            gap_max=float(np.max(-spectrum_gaps(m)) / LAM),
            ruler=ruler, cont=cont,
        ))
    out["arms"] = rows

    print("\n== per-arm table ==")
    hdr = ("arm", "sumM", "N", "n_min", "n_int", "n(.3)", "n(.5)", "n(.7)",
           "n(.95)", "n(1.5)", "mis.5", "mis.95", "gap", "RULER", "contNLL")
    print(("  %-15s" + "%7s" * 14) % hdr)
    for r in rows:
        print("  %-15s%7.2f%5d%7.1f%7.2f%6d%6d%6d%7d%6d%6.3f%6.3f%7.2f%9s%8s"
              % (r["name"], r["sum_m"], r["N"], r["n_min"], r["n_int"], r["n_u03"],
                 r["n_u05"], r["n_u07"], r["n_u095"], r["n_u15"], r["mis_mid"],
                 r["mis_deep"], r["gap_max"],
                 "%.4f" % r["ruler"] if r["ruler"] is not None else "-",
                 "%.4f" % r["cont"] if r["cont"] is not None else "-"))

    print("\n== rank correlations vs RULER 350 (17 arms with scores) ==")
    for q in ["sum_m", "N", "n_min", "n_int", "n_u03", "n_u05", "n_u07", "n_u095"]:
        vals = [r[q] for r in rows if r["ruler"] is not None]
        sc = [r["ruler"] for r in rows if r["ruler"] is not None]
        print(f"  {q:8s} rho = {spearman(vals, sc):+.3f}")
    print("\n== rank correlations vs continuous NLL (lower=better; expect negative rho) ==")
    for q in ["sum_m", "N", "n_min", "n_int", "n_u095"]:
        vals = [r[q] for r in rows if r["cont"] is not None]
        sc = [r["cont"] for r in rows if r["cont"] is not None]
        print(f"  {q:8s} rho = {spearman(vals, sc):+.3f}")

    print("\n== the discriminating pairs ==")
    for a, b in [("ctl_C42", "ctl_C42V24"), ("BM", "step_hi25"), ("MrRoPE", "BM"),
                 ("BM", "turns_a1_b64"), ("turns_a1_b16", "turns_a1_b64"),
                 ("evq_shift_t0p5", "BM")]:
        ra = next(r for r in rows if r["name"] == a)
        rb = next(r for r in rows if r["name"] == b)
        sa = ra["ruler"] if ra["ruler"] is not None else float("nan")
        sb = rb["ruler"] if rb["ruler"] is not None else float("nan")
        print(f"  {a} ({sa:.4f}) vs {b} ({sb:.4f}): "
              f"n_min {ra['n_min']:.0f}/{rb['n_min']:.0f}  "
              f"n_int {ra['n_int']:.1f}/{rb['n_int']:.1f}  "
              f"n(.5) {ra['n_u05']}/{rb['n_u05']}  "
              f"n(.95) {ra['n_u095']}/{rb['n_u095']}  "
              f"mis.5 {ra['mis_mid']:.3f}/{rb['mis_mid']:.3f}  "
              f"gap {ra['gap_max']:.1f}/{rb['gap_max']:.1f}")

    print("\n== 8x prescription: shift (A) vs amplitude-scale (B) ==")
    for name in ["BM", "turns_a1_b64", "shift8x_BM", "shift8x_wide",
                 "scale8x_BM", "scale8x_wide"]:
        m = ARMS[name][0]
        marks = {0.5: "u=0.5", 1.0: "u=1", 1.25: "u=1.25", 1.5: "u=1.5(8x)"}
        txt = "  ".join(f"{lbl}:n={coverage(m, u)}" for u, lbl in marks.items())
        # in-window coverage loss vs native at u=-1 (d=W/4) and u=-2 (d=W/16)
        nat = ARMS["native"][0]
        inw = "  ".join(f"uw({u}):{coverage(m, u)}/{coverage(nat, u)}"
                        for u in (-1.0, -2.0))
        print(f"  {name:14s} sum_m={m.sum():6.2f} gap={np.max(-spectrum_gaps(m))/LAM:4.2f}x"
              f"  {txt}  {inw}")

    print("\n== sensitivity of the coverage story (delta, L) ==")
    global DELTA, kappa
    base_delta, base_kappa = DELTA, kappa
    scored = [r["name"] for r in rows if r["ruler"] is not None]
    for dlt in (0.125, 0.25, 0.5, 1.0):
        kline = []
        for LL in (1.0, 2.0, 3.0, 10.0):
            DELTA = dlt
            kappa = np.log(tW / DELTA) / LN4
            vals = [coverage(ARMS[n][0], 0.3, LL) for n in scored]
            sc = [next(r for r in rows if r["name"] == n)["ruler"] for n in scored]
            kline.append(f"L={LL}:{spearman(vals, sc):+.3f}")
            DELTA, kappa = base_delta, base_kappa
        print(f"  delta={dlt:6.3f}  " + "  ".join(kline))

    print("\n== prereg predictions for in-flight arms (write BEFORE reading) ==")
    print("  step42 (Pro RULER run in flight): hole table (gap "
          f"{np.max(-spectrum_gaps(ARMS['step42'][0]))/LAM:.1f}x native) -> "
          "predict RULER 0.10-0.35, far below BM 0.4167")
    print("  condEVQ (smooth, no hole): predict 0.40-0.55, within plateau of BM")
    print("  walk a=0.25/0.5/0.75: predict monotone mild, NO interior kink "
          "(linear-ish dose-response; both endpoints same mechanism)")


    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "coverage_theory_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nresults -> coverage_theory_results.json")


if __name__ == "__main__":
    main()
