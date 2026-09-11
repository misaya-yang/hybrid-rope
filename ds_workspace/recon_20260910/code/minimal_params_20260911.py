#!/usr/bin/env python3
"""Minimal parameterisation of the 12 OLMo 16K-RULER points, and the
readable-window knapsack solved exactly.

Run:  python3 ds_workspace/recon_20260910/code/minimal_params_20260911.py
      (cwd = repo root; numpy only)

Companion to ds_workspace/recon_20260910/MINIMAL_PARAMS_20260911.md.

WHAT IS NEW HERE vs midband_n.py / evq_limit_20260911.py
  * the per-slot m-INTERVAL that makes a slot readable is derived in closed
    form (sec 1), and so is the exact reachable slot range;
  * the minimal CARDINALITY of a parameter set is decided by exhaustive
    subset search with leave-one-out, not by eyeballing candidates;
  * the redundancy audit is explicit: which candidates are exact renamings
    on these 12 points (algebraic identity / rank-1 collinearity);
  * the knapsack is SOLVED in closed form (sec 5), the nu-monotonicity
    coupling is handled exactly (sec 5b), and the result is compared
    against BM and the three winners (sec 6);
  * sec 5c shows the pure-count objective is DEGENERATE and names the term
    that de-degenerates it; sec 7 turns that into a 2-arm discriminator.

PROVENANCE (LESSONS.md discipline): [PROVED] arithmetic/definition,
recomputable here; [DERIVED] follows from a named assumption; [GUESS]
pattern in the 12 points only.  No capability claim except sec 7.
"""
from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.curvature_20260910 import tables as T  # noqa: E402

K = 64
LN4 = math.log(4.0)
LN2 = math.log(2.0)
THETA = 500_000.0
W_TRAIN = 4096.0
D_TEST = 4.0 * W_TRAIN
LOW, N_INCR = 14, 18
SE_PAIRED = 0.0262
GATE = 2.0 * SE_PAIRED

LAM2 = math.log(THETA) / (K * LN2)          # log2 slope of g_j in j
LAM4 = math.log(THETA) / (K * LN4)          # == LAM2/2, the nu-monotonicity cap
OMEGA = T.native_inv_freq(THETA, K)
T_W = W_TRAIN * OMEGA / (2.0 * math.pi)
G = np.log2(T_W)
J = np.arange(K, dtype=float)


# ---------------------------------------------------------------------------
def knife(delta, a):
    m = np.asarray(T.m_taper(delta, hi=32, lo=LOW), dtype=np.float64).copy()
    if a:
        m[:LOW] = np.maximum(m[:LOW],
                             a * np.arange(1, LOW + 1, dtype=np.float64))
    return np.clip(m, 0.0, 1.0)


def turns(a, b, ramp="beta1"):
    return np.asarray(T.m_turns(a, b, THETA, W_TRAIN, 128, ramp=ramp),
                      dtype=np.float64)


ARMS = {
    "MrPro":         (T.m_incr_beta(0.0, n=N_INCR, low=LOW), 24.80 / 350),
    "beta_b0p25":    (T.m_incr_beta(0.25, n=N_INCR, low=LOW), 0.144714),
    "turns_a0p5b32": (turns(0.5, 32.0), 0.160429),
    "beta_b0p5":     (T.m_incr_beta(0.5, n=N_INCR, low=LOW), 0.232000),
    "turns_a1b16":   (turns(1.0, 16.0), 0.265143),
    "MrProBM":       (T.m_incr_beta(1.0, n=N_INCR, low=LOW), 145.85 / 350),
    "knife_taper":   (knife(0.0063, 0.0), 0.3937),
    "knife_leak":    (knife(0.0, 0.0063), 0.3969),
    "knife_both":    (knife(0.0063, 0.0063), 0.3846),
    "beta_b2":       (T.m_incr_beta(2.0, n=N_INCR, low=LOW), 0.500143),
    "turns_a2b32":   (turns(2.0, 32.0), 0.510000),
    "turns_a1b64":   (turns(1.0, 64.0), 0.538429),
}
S_LEDGER = {
    "MrPro": 37.666667, "beta_b0p25": 38.464958, "turns_a0p5b32": 39.000000,
    "beta_b0p5": 39.209117, "turns_a1b16": 38.500000, "knife_both": 38.037,
    "knife_taper": 37.375, "knife_leak": 41.162, "MrProBM": 40.500000,
    "beta_b2": 42.378947, "turns_a2b32": 42.000000, "turns_a1b64": 42.000000,
}
N_TABLE = {
    "MrPro": 15, "beta_b0p25": 16, "turns_a0p5b32": 16, "beta_b0p5": 16,
    "turns_a1b16": 16, "MrProBM": 17, "knife_taper": 17, "knife_leak": 17,
    "knife_both": 17, "beta_b2": 17, "turns_a2b32": 17, "turns_a1b64": 17,
}
NAMES = list(ARMS)
ACC = np.array([ARMS[n][1] for n in NAMES])
M = {n: np.asarray(ARMS[n][0], dtype=np.float64) for n in NAMES}


# ---------------------------------------------------------------------------
def t_at_test(m):
    return D_TEST * OMEGA * np.power(4.0, -np.asarray(m, float)) / (2 * math.pi)


def N_of(m, lo=0.25, hi=16.0):
    td = t_at_test(m)
    return int(np.sum((td >= lo) & (td <= hi)))


MU = (G - 2.0) / 2.0            # entry threshold  (readable <=> m_j >= MU_j)
NUBAR = (G + 4.0) / 2.0         # exit threshold   (readable <=> m_j <= NUBAR_j)
REACH = (MU <= 1.0) & (NUBAR >= 0.0)
J_IN = int(J[REACH][0])
J_OUT = int(J[REACH][-1])
J_CAL = int(J[NUBAR >= 1.0][-1])            # last slot with NUBAR >= 1


# ---------------------------------------------------------------------------
def eps_of(m):
    return np.diff(np.concatenate(([0.0], np.asarray(m, float))))


def jit(k, base=1e-3, slope=1e-3):
    """The knapsack optimum with k of the 6 fast-side competing slots bought.

    The cheapest readable state is m_j = MU_j, but that profile has slope
    exactly -LAM4 in j, i.e. it makes nu CONSTANT -- it sits on the boundary of
    nu-monotonicity and is not strictly legal (and slot j0 lands exactly on
    t_D = 16).  `base` lifts it off the top edge and `slope*(j - j0)` makes it
    fall slower than LAM4, which is what strict nu-monotonicity requires.
    """
    if k == 0:
        return np.zeros(K)
    j0 = 25 - k
    m = np.zeros(K)
    m[j0:25] = MU[j0:25] + base + slope * (J[j0:25] - j0)
    return m


def qhat(m, p=2.0):
    w = (T_W / T_W[LOW]) ** p
    return float((w * np.asarray(m, float) ** 2).sum())


def w_of(p=2.0):
    return (T_W / T_W[LOW]) ** p


def cover(m):
    return float(np.mean(np.minimum(1.0, np.power(4.0, np.asarray(m, float) - 1.0))))


def drift(m):
    return float(np.sum(4.0 * T_W * (1.0 - np.power(4.0, -np.asarray(m, float)))))


def sum_nu(m):
    return float(np.sum(np.power(4.0, -np.asarray(m, float)) * OMEGA))


BANK = {
    "S":         lambda m: float(np.sum(m)),
    "N":         lambda m: float(N_of(m)),
    "mu_eps":    lambda m: float((J * eps_of(m)).sum()),
    "Q_p2":      lambda m: qhat(m, 2.0),
    "Q_p1":      lambda m: qhat(m, 1.0),
    "Q_p0":      lambda m: qhat(m, 0.0),
    "C":         cover,
    "drift":     drift,
    "sum_nu":    sum_nu,
    "m63":       lambda m: float(np.asarray(m, float)[-1]),
    "n_plateau": lambda m: float(np.sum(np.asarray(m, float) > 1 - 1e-12)),
    "n_held":    lambda m: float(np.sum(np.asarray(m, float) < 1e-12)),
    "n_ramp":    lambda m: float(np.sum((np.asarray(m, float) > 1e-12)
                                        & (np.asarray(m, float) < 1 - 1e-12))),
    "eps_max":   lambda m: float(eps_of(m).max()),
    "n_lost":    lambda m: float(np.sum(REACH & ~((np.asarray(m, float) >= MU - 1e-12)
                                                  & (np.asarray(m, float) <= NUBAR + 1e-12)))),
    "med_h":     lambda m: float(np.median((2.0 + G - 2.0 * np.asarray(m, float))[
        (2.0 + G - 2.0 * np.asarray(m, float) >= -2.0)
        & (2.0 + G - 2.0 * np.asarray(m, float) <= 4.0)])),
}


def ex_needed(m):
    m = np.asarray(m, float)
    return float(np.sum(m[(MU > 0.0) & (MU <= 1.0)]))


BANK["ex_needed"] = ex_needed


# ---------------------------------------------------------------------------
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort").astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(rx @ ry / math.sqrt((rx @ rx) * (ry @ ry)))


def _dm(X, y):
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y):
        X = X.T
    return X


def fit_rms(X, y):
    X = _dm(X, y)
    A = np.hstack([np.ones((len(y), 1)), X])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.sqrt(np.mean((A @ c - y) ** 2))), c


def loo_rms(X, y):
    X = _dm(X, y)
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        A = np.hstack([np.ones((keep.sum(), 1)), X[keep]])
        c, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        out[i] = float(np.concatenate([[1.0], X[i]]) @ c)
    return float(np.sqrt(np.mean((out - y) ** 2)))


def pair_verdicts(vals, y, gate=0.0):
    order = np.argsort(-np.asarray(y))
    v = np.asarray(vals, float)[order]
    yy = np.asarray(y, float)[order]
    best = None
    for sign in (1.0, -1.0):
        p = f = t = 0
        for a, b in combinations(range(len(y)), 2):
            if yy[a] - yy[b] <= gate:
                continue
            d = sign * (v[a] - v[b])
            if abs(d) < 1e-12:
                t += 1
            elif d > 0:
                p += 1
            else:
                f += 1
        if best is None or p > best[0]:
            best = (p, f, t)
    return best


# ---------------------------------------------------------------------------
def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    sec("0  build check")
    bad = {n: (float(np.sum(M[n])), S_LEDGER[n]) for n in NAMES
           if abs(float(np.sum(M[n])) - S_LEDGER[n]) > 5e-3}
    nbad = {n: (N_of(M[n]), N_TABLE[n]) for n in NAMES if N_of(M[n]) != N_TABLE[n]}
    print(f"  S vs LEDGER        : {bad if bad else 'all 12 within 5e-3'}")
    print(f"  N vs given table   : {nbad if nbad else 'all 12 exact'}")
    assert not bad and not nbad

    sec("1  the readability geometry, closed form  [PROVED]")
    print(f"  g_j = log2 t_j = {G[0]:.4f} - {LAM2:.6f} j        (affine, exact)")
    print("  readable <=> t_D,j in [0.25,16] <=> m_j in [MU_j, NUBAR_j]"
          " = [(g_j-2)/2, (g_j+4)/2]")
    print(f"  band width = 3 exactly;  d MU/dj = -LAM2/2 = {-LAM2 / 2:.6f}")
    print(f"  nu-monotonicity cap: m_(j+1)-m_j > -LAM4 = {-LAM4:.6f}")
    print(f"  ==> the band edge and the monotonicity cap are THE SAME NUMBER"
          f"  [PROVED]")
    print(f"  reachable slots (MU<=1, NUBAR>=0) : {J_IN}..{J_OUT}"
          f"  ({int(REACH.sum())} slots)   N_max = {int(REACH.sum())}")
    print(f"  free at m=0 (MU<=0 and NUBAR>=0)  : 25..{J_OUT}"
          f"  ({int(((MU <= 0) & REACH).sum())} slots) = native N = {N_of(np.zeros(K))}")
    print(f"  last slot with NUBAR>=1 (calibratable at m=1): {J_CAL}")

    sec("2  every candidate: single-variable fit and the 66-pair verdict")
    print(f"  gate=0: all 66 pairs.  gate=2SE={GATE:.4f}: only decided pairs.")
    print(f"  {'candidate':11s} {'spearman':>9s} {'rms':>8s} {'LOO':>8s} "
          f"{'pass/66':>9s} {'pass/decided':>13s}")
    VALS = {k: np.array([f(M[n]) for n in NAMES]) for k, f in BANK.items()}
    for nm, v in sorted(VALS.items(), key=lambda kv: fit_rms(kv[1], ACC)[0]):
        r, _ = fit_rms(v, ACC)
        p0, f0, t0 = pair_verdicts(v, ACC, 0.0)
        pg, fg, tg = pair_verdicts(v, ACC, GATE)
        print(f"  {nm:11s} {spearman(v, ACC):+9.3f} {r:8.4f} {loo_rms(v, ACC):8.4f} "
              f"{p0 + t0 * 0.5:6.1f}/{66:<2d} {pg + tg * 0.5:7.1f}/{pg + fg + tg:<2d}")
    print(f"  constant model rms = {ACC.std():.4f};   noise floor = {SE_PAIRED}")

    sec("2b  the 'residual is 2.2x noise' claim: same cardinality, two bases")
    Xb = np.column_stack([VALS["S"], VALS["n_plateau"], VALS["sum_nu"]])
    r, _ = fit_rms(Xb, ACC)
    print(f"  SEP_CAUSE basis   {{S, n_plateau, sum_nu}} + intercept:"
          f"  rms {r:.4f}  LOO {loo_rms(Xb, ACC):.4f}   (4 parameters)")
    best4 = None
    keys = [k for k in BANK]
    n9 = [n for n in NAMES if not n.startswith("knife")]
    A9 = np.array([ARMS[n][1] for n in n9])
    V9 = {k: np.array([f(M[n]) for n in n9]) for k, f in BANK.items()}
    X9 = np.column_stack([V9["S"], V9["n_plateau"], V9["sum_nu"]])
    r9, _ = fit_rms(X9, A9)
    print(f"  same basis on the 9 pre-KNIFE points: rms {r9:.4f}")
    print(f"  NOT reproduced: SEP_CAUSE reports 0.0573 for its 4-parameter model,"
          f" so that model is")
    print(f"  specified differently (4 predictors, or other variables).  The"
          f" cardinality-matched")
    print(f"  comparison below is what the conclusion rests on, not the recall of"
          f" their number.")
    for combo in combinations(keys, 3):
        X = np.column_stack([VALS[c] for c in combo])
        rr, _ = fit_rms(X, ACC)
        if best4 is None or rr < best4[0]:
            best4 = (rr, loo_rms(X, ACC), combo)
    print(f"  best 3-predictor basis {list(best4[2])} + intercept:"
          f"  rms {best4[0]:.4f}  LOO {best4[1]:.4f}   (4 parameters)")
    print(f"  -> at EQUAL cardinality the residual moves {r:.4f} -> {best4[0]:.4f}."
          f"  The misfit was a basis artefact.")

    sec("3  minimum cardinality: exhaustive subset search + leave-one-out")
    found = {}
    for k in (1, 2, 3, 4):
        hits = []
        for combo in combinations(keys, k):
            X = np.column_stack([VALS[c] for c in combo])
            rr, _ = fit_rms(X, ACC)
            hits.append((rr, loo_rms(X, ACC), combo))
        hits.sort()
        found[k] = hits
        n_ok = sum(1 for h in hits if h[0] <= SE_PAIRED)
        print(f"  k={k}: {n_ok} of {len(hits)} subsets reach in-sample rms <= "
              f"{SE_PAIRED}")
        for rr, lr, c in hits[:3]:
            print(f"       rms {rr:.4f}  LOO {lr:.4f}   {list(c)}")
    kmin = min(k for k in found if found[k] and found[k][0][0] <= SE_PAIRED)
    print(f"\n  in-sample: k = {kmin} parameters (+intercept) reach the noise floor")
    for k in (2, 3, 4, 5):
        if k in found and found[k]:
            rr, lr, c = min(found[k], key=lambda t: t[1])
            print(f"  best LOO at k={k}: LOO {lr:.4f}  ({list(c)})")

    sec("4  redundancy: exact renamings vs independent content")
    e1 = max(abs(float(np.sum(M[n])) / K - float(np.mean(M[n]))) for n in NAMES)
    e2 = max(abs(float(np.sum(M[n])) - (K * M[n][-1]
                                        - float((J * eps_of(M[n])).sum())))
             for n in NAMES)
    k1 = [n for n in NAMES if abs(M[n][-1] - 1.0) < 1e-12]
    print(f"  mean(m) == S/64                        max err {e1:.1e}  [PROVED]")
    print(f"  S == 64*(m63-m0) - mu_eps              max err {e2:.1e}  [PROVED]")
    print(f"  {len(k1)}/12 arms have m63 = 1  ==>  on those S == 64 - mu_eps")
    e3 = max(abs(drift(M[n]) - (4.0 * T_W.sum()
                                 - (2.0 * W_TRAIN / math.pi) * sum_nu(M[n])))
             for n in NAMES)
    print(f"  drift == 4*sum(t_j) - (2W/pi)*sum_nu     max err {e3:.1e}  [PROVED]")
    print("      (so 'drift' and 'sum_nu' are ONE number, not two)")
    print("  rank-collinear pairs over the 12 arms:")
    seen = set()
    for a in keys:
        for b in keys:
            if a < b and abs(abs(spearman(VALS[a], VALS[b])) - 1.0) < 1e-9:
                print(f"      {a:12s} == {b:12s}  (|rho| = 1)")
                seen.add(a)
    print(f"  structural: n_lost == {J_OUT - J_IN + 1} - N exactly (reach set is fixed)")
    print("  ALSO: within the beta family, S, mu_eps and b are ONE degree of freedom")
    bs = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
    Ss = np.array([np.sum(T.m_incr_beta(b, n=N_INCR, low=LOW)) for b in bs])
    print(f"      b    = {list(bs)}")
    print(f"      S(b) = {[round(float(x), 3) for x in Ss]}   strictly monotone")

    sec("5  the readable-window knapsack, solved exactly  [DERIVED]")
    print("  max N s.t. sum_j (1/2) F_jj (ln4)^2 m_j^2 <= eps.")
    print("  Per slot the cheapest readable state is m_j = max(0, MU_j), so:")
    print("    N = 21 is free; the paid slots are j = 19..24 at m_j = MU_j, cost")
    print("    c_j = (1/2) F_jj (ln4)^2 MU_j^2.  Take them cheapest first.")
    print("  F_jj is NOT available (tstar.json is on the unreachable server, no")
    print("  local copy).  Using the analytic proxy F_jj ~ (t_j/t_14)^p validated")
    print("  on Qwen 32K at R2 = 0.992-0.996.  p=2 primary, p=1 sensitivity.")
    for p in (2.0, 1.0):
        w = w_of(p)
        idx = np.flatnonzero((MU > 0.0) & (MU <= 1.0))[::-1]
        c = 0.5 * w[idx] * LN4 ** 2 * MU[idx] ** 2
        o = np.argsort(c)
        idx, c = idx[o], c[o]
        cum = np.cumsum(c)
        print(f"\n  --- p = {p:g} ---   N = 21 + k")
        for r in range(len(idx)):
            print(f"      k={r + 1}  +slot {idx[r]:2d} (MU={MU[idx[r]]:.4f})"
                  f"   cum cost {cum[r]:.4e}   N = {22 + r}")
        qbm = qhat(M["MrProBM"], p)
        kstar = int(np.searchsorted(cum, qbm))
        # push off the nu-boundary: the JIT profile makes nu exactly constant,
        # so it needs a hair of extra compression to be strictly legal.
        m_t = jit(kstar)
        print(f"      MrProBM Q_p{p:g} = {qbm:.4e}"
              f"   ->  at BM's own budget the knapsack affords k = {kstar}"
              f"  ({'all 6' if kstar == 6 else 'partial'})")
        print(f"      that table: S = {m_t.sum():.4f}, N = {N_of(m_t)},"
              f" Q_p{p:g} = {qhat(m_t, p):.4e}")

    sec("5b  nu-monotonicity, handled exactly")
    print(f"  nu_(j+1) < nu_j  <=>  m_(j+1) - m_j > -LAM4 = {-LAM4:.6f}")
    print("  m may jump UP freely; it may fall by at most LAM4 per slot.")
    print("  Because d MU/dj = -LAM4 exactly, the just-in-time entry profile")
    print("  m_j = MU_j makes nu CONSTANT -- it sits exactly ON the boundary.")
    log4nu = np.log2(OMEGA * np.power(4.0, -MU)) / 2.0
    print(f"  check: mean d(log4 nu) along m = MU : {np.diff(log4nu).mean():+.3e}"
          f"  (0 => boundary) [PROVED]")
    print("  So any legal table is the JIT profile pushed just off the boundary.")

    sec("5c  the pure-count objective is DEGENERATE; calibration de-degenerates it")
    print("  max N alone is achieved by m_j = max(0, MU_j): a table that leaves")
    print(f"  32 slots at m = 0, has S = {float(MU[(MU > 0) & REACH].sum()):.3f}"
          f" (MrPro's is 37.667), and is otherwise native.")
    print("  N is maximised by DOING NOTHING.  That is not a solution, it is a")
    print("  proof that the pure-count objective, as posed, has no interior")
    print("  optimum on this geometry.")
    print("  MECHANISM sec 1-2 supplies the missing term: slot j is a CALIBRATED")
    print("  readout over [0,D] only if m_j >= 1.  Adding it:")
    print(f"    readable AND calibrated <=> m_j in [max(1,MU_j), NUBAR_j],")
    print(f"    non-empty only for j <= {J_CAL} (NUBAR_j >= 1) and MU_j <= 1")
    print(f"    => the block j = {J_IN}..{J_CAL} at m_j = 1, plus a RELEASE ramp on")
    print(f"    j = {J_CAL + 1}..{J_OUT} forced by nu-monotonicity (slope -LAM4).")

    def calibrated():
        """The derived optimum.  The release slope is FORCED into an interval.

        Slot j in [39,45] stays readable iff m_j <= NUBAR_j; nu-monotonicity
        demands m_j - m_(j-1) > -LAM4.  With a linear release at rate s from
        m_38 = 1, the first needs s >= LAM4 - (NUBAR_38 - 1)/7 and the second
        s < LAM4, so s lies in a non-empty interval and the ramp is unique up
        to that width.  s = LAM4*(1 - 0.02) is inside it.
        """
        m = np.zeros(K)
        m[J_IN:J_CAL + 1] = 1.0
        s = LAM4 * 0.98
        for j in range(J_CAL + 1, K):
            m[j] = max(0.0, 1.0 - s * (j - J_CAL))
        return m

    m_cal = calibrated()
    inc = np.diff(np.concatenate([[0.0], m_cal]))
    ok = inc.min() > -LAM4
    print(f"    S = {m_cal.sum():.4f}   N = {N_of(m_cal)}"
          f"   min increment = {inc.min():+.6f} (> -{LAM4:.6f}: "
          f"{'OK' if ok else 'VIOLATES'})")
    print(f"    Q_p2 = {qhat(m_cal, 2.0):.4e}   MrProBM Q_p2 = {qhat(M['MrProBM'], 2.0):.4e}")

    sec("5d  the CALIBRATED knapsack frontier (the solved answer to the task)")
    print("  With calibration the entry state is m_j = max(1, MU_j): every slot")
    print("  below the free band costs a full compression to become a calibrated")
    print("  channel.  Frontier over the reachable set:")
    for p in (2.0, 1.0):
        w = w_of(p)
        cand = [j for j in range(0, J_CAL + 1) if MU[j] <= 1.0]
        ent = np.array([max(1.0, MU[j]) for j in cand])
        c = 0.5 * w[cand] * LN4 ** 2 * ent ** 2
        o = np.argsort(c)
        cand, c, ent = [cand[i] for i in o], c[o], ent[o]
        cum = np.cumsum(c)
        print(f"  --- p = {p:g} ---   N_cal = number of FULL-COMPRESSION channels")
        print("      NOTHING is free here: at m = 0 no slot is calibrated over")
        print("      [0,D], so N_cal starts at 0 (BM's own N_cal is 6).")
        for r in range(len(cand)):
            print(f"      +slot {cand[r]:2d} (entry m={ent[r]:.3f})"
                  f"  cum cost {cum[r]:.4e}   N_cal = {r + 1}")
        qbm = qhat(M["MrProBM"], p)
        kk = int(np.searchsorted(cum, qbm))
        print(f"      MrProBM Q_p{p:g} = {qbm:.4e}  ->"
              f" at BM's budget the calibrated knapsack affords N_cal = {kk}")
        print(f"      (the table of sec 5c is the k = {len(cand)} end of this"
              f" frontier: every reachable slot bought)")

    def solved_at_budget(p=2.0):
        """argmax N_cal s.t. Q_p(m) <= Q_p(MrProBM) -- the literal task answer.

        Buy the calibrated channels cheapest-first until BM's own in-window
        budget is exhausted, then append the release ramp that nu-monotonicity
        forces.  This is the table the task asks for: same eps, maximum number
        of readable calibrated slots.
        """
        w, qbm = w_of(p), qhat(M["MrProBM"], p)
        cand = [j for j in range(0, J_CAL + 1) if MU[j] <= 1.0]
        ent = np.array([max(1.0, MU[j]) for j in cand])
        c = 0.5 * w[cand] * LN4 ** 2 * ent ** 2
        o = np.argsort(c)
        cand, ent, c = [cand[i] for i in o], ent[o], c[o]
        hi = cand[int(np.searchsorted(np.cumsum(c), qbm)) - 1]
        m = calibrated()
        m[J_IN:hi] = 0.0        # buy only the cheapest k: m = 1 on [hi, J_CAL]
        return m, hi, qbm

    sec("5e  THE SOLVED TABLE at BM's own in-window budget  [the task's answer]")
    m_star, hi_star, qbm = solved_at_budget(2.0)
    h = 2.0 + G - 2.0 * m_star
    sl = np.flatnonzero((h >= -2) & (h <= 4))
    inc = np.diff(np.concatenate([[0.0], m_star]))
    n_cal_t = int(np.sum((m_star > 1 - 1e-9) & (NUBAR >= 1.0)))
    n_cal_bm = int(np.sum((M["MrProBM"] > 1 - 1e-9) & (NUBAR >= 1.0)))
    print(f"  max N s.t. Q_p2(m) <= Q_p2(MrProBM) = {qbm:.4e}")
    print(f"  solution: m = 1 on slots {hi_star}..{J_CAL} (the {J_CAL - hi_star + 1}"
          f" cheapest calibrated")
    print(f"            channels), then the forced release ramp on"
          f" {J_CAL + 1}..{J_OUT}, 0 elsewhere")
    print(f"  {'':16s} {'S':>8s} {'N':>4s} {'N_cal':>6s} {'Q_p2':>10s} "
          f"{'m63':>6s}")
    print(f"  {'SOLVED T*':16s} {np.sum(m_star):8.3f} {N_of(m_star):4d} "
          f"{n_cal_t:6d} {qhat(m_star, 2.0):10.4e} {m_star[-1]:6.3f}")
    print(f"  {'MrProBM':16s} {np.sum(M['MrProBM']):8.3f} {N_of(M['MrProBM']):4d} "
          f"{n_cal_bm:6d} {qbm:10.4e} {M['MrProBM'][-1]:6.3f}")
    print(f"  {'best winner':16s} {42.000:8.3f} {17:4d} {n_cal_bm:6d} "
          f"{qhat(M['turns_a1b64'], 2.0):10.4e} {1.0:6.3f}")
    print(f"  legality: min increment {inc.min():+.6f} (> {-LAM4:.6f}), "
          f"in-window {sl[0]}..{sl[-1]} ({len(sl)} slots), monotone nu: "
          f"{'OK' if inc.min() > -LAM4 else 'VIOLATES'}")

    sec("6  the solved table vs BM and the three winners")
    tabs = [("KNAPSACK(cal)", m_cal),
            ("KNAPSACK(JIT)", None),
            ("MrProBM", M["MrProBM"]),
            ("turns_a2b32", M["turns_a2b32"]),
            ("turns_a1b64", M["turns_a1b64"]),
            ("beta_b2", M["beta_b2"])]
    m_jit = jit(6)
    tabs[1] = ("KNAPSACK(JIT)", m_jit)
    print(f"  {'arm':15s} {'S':>8s} {'N':>3s} {'m63':>6s} {'Q_p2':>10s} "
          f"{'in-window slots':>24s}")
    for nm, m in tabs:
        h = 2.0 + G - 2.0 * m
        sl = np.flatnonzero((h >= -2) & (h <= 4))
        rng = f"{sl[0]}..{sl[-1]} ({len(sl)})" if len(sl) else "none"
        print(f"  {nm:15s} {np.sum(m):8.3f} {N_of(m):3d} {m[-1]:6.3f} "
              f"{qhat(m, 2.0):10.4g} {rng:>24s}")

    sec("7  the discriminating pair  [the only capability claim]")
    print("  Two tables with the SAME N = 27 and Delta S = "
          f"{abs(m_cal.sum() - m_jit.sum()):.2f}:")
    for nm, m in (("P1 KNAPSACK(cal)", m_cal), ("P2 KNAPSACK(JIT)", m_jit)):
        print(f"    {nm:16s} S = {np.sum(m):7.3f}  N = {N_of(m)}  "
              f"Q_p2 = {qhat(m, 2.0):.4e}  m63 = {m[-1]:.3f}")
    qbm = qhat(M["MrProBM"], 2.0)
    print(f"  P1 Q_p2 / MrProBM Q_p2 = {qhat(m_cal, 2.0) / qbm:.2f}x")
    print("  Predictions (each theory on its own fitted scale):")
    sl = 0.1079       # d(score)/dS from the 12 points, slope of the S fit
    print(f"    N theory   : both >= the best measured winner 0.5384")
    print(f"    S theory   : P1 (S={m_cal.sum():.1f}) and P2 (S={m_jit.sum():.1f})"
          f" both far below the fitted range [37.4,42.4]")
    print(f"                 -> linear extrapolation gives"
          f" {0.5384 - sl * (42.0 - m_cal.sum()):+.2f} and"
          f" {0.5384 - sl * (42.0 - m_jit.sum()):+.2f}, i.e. clipped to ~0")
    print(f"    calibration: P1 >> P2 (P2's in-window slots are uncalibrated)")
    print(f"    pure count : P1 == P2 within noise")


if __name__ == "__main__":
    main()
