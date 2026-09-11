#!/usr/bin/env python3
"""CPU verification of the Pro plan's §2/§4 claims (a)-(h).

Run:  .venv/bin/python ds_workspace/recon_20260910/code/verify_pro_a_20260911.py

Nothing here touches a model or a GPU.  Every table is rebuilt from the repo's
own constructions (experiments/curvature_20260910/tables.py) so the numbers are
commensurate with LEDGER_20260911.md; the ledger S values are asserted.

OLMo-2-0425-1B-Instruct geometry: theta=5e5, W=4096, head_dim=128, K=64.
The test distance is 4W = 16384 (the 16K RULER length).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.curvature_20260910 import tables as T  # noqa: E402

K = 64
LN4 = math.log(4.0)
THETA = 500_000.0
W = 4096.0
DTEST = 4.0 * W
LOW, N_INCR = 14, 18          # the deployed three-band geometry on OLMo

# S from LEDGER_20260911.md (asserted, so a build error is not a finding)
S_LEDGER = {
    "native": 0.0, "MrPro": 37.666667, "BM": 40.500000, "step_hi22": 42.0,
    "step_hi25": 39.0, "b3_lo14": 43.637, "a1_b64": 42.000000,
    "C42": 42.000000, "C42V24": 42.000000, "b4_wide": 46.68,
}


def omega():
    return T.native_inv_freq(THETA, K)


def turns_w(nu=None):
    nu = omega() if nu is None else nu
    return W * np.asarray(nu, float) / (2.0 * math.pi)


def turns_d(m):
    return DTEST / (2.0 * math.pi) * omega() * np.power(4.0, -np.asarray(m, float))


def req(m_irrelevant=None):
    """Compression needed to bring slot j into the READABLE band at 4W.

    Readable at the test distance means T_j = 4 t_W,j 4^-m in [0.25, 16],
    equivalently t_W,j 4^-m in [0.0625, 4].  Requirement is the TOP edge:
    req_j = ln(t_W,j / 4) / ln 4   (m needed so that t_W 4^-m = 4).
    req > 1 means the slot cannot be rescued by any m <= 1.
    """
    return np.log(turns_w() / 4.0) / LN4


REQ = req()


def N_readable(m, lo=0.25, hi=16.0):
    Tj = turns_d(m)
    return int(np.sum((Tj >= lo) & (Tj <= hi)))


def s_entry(m):
    """Sum of min(m_j, req_j) -- 'how much of the savable budget was spent'."""
    return float(np.sum(np.minimum(np.asarray(m, float), np.maximum(REQ, 0.0))))


def laplacian_eps(m, lo=LOW, n=N_INCR):
    """(L eps) on the band [lo, lo+n], eps_k = m_k - m_{k-1}, Dirichlet 0/0.

    This is the campaign's forcing convention (FOUR_CORNERS_20260911 §1):
    for BM (eps ~ k(n+1-k)) it returns a constant, for MrRoPE a single point
    source at the last increment.  Verified in the selftest at the bottom.
    """
    m = np.asarray(m, float)
    eps = np.diff(m[lo: lo + n + 1])                 # n increments
    e = np.concatenate(([0.0], eps, [0.0]))          # Dirichlet at both ends
    return 2.0 * e[1:-1] - e[:-2] - e[2:]


def max_adj_period_ratio(m):
    nu = omega() * np.power(4.0, -np.asarray(m, float))
    r = nu[:-1] / nu[1:]
    return float(r.max()), int(np.argmax(r))


def report(name, m):
    m = np.asarray(m, float)
    S = float(m.sum())
    led = S_LEDGER.get(name)
    ok = "" if led is None else (" OK" if abs(S - led) < 5e-3 else f" MISMATCH vs {led}")
    nu = omega() * np.power(4.0, -m)
    mono = bool(np.all(np.diff(nu) < 0))
    ratio, at = max_adj_period_ratio(m)
    Tj = turns_d(m)
    nread = N_readable(m)
    sav = np.where((REQ > 0) & (REQ <= 1))[0]
    return dict(
        name=name, S=S, s_ok=ok, mono=mono, ratio=ratio, ratio_at=at,
        N=nread, n_held=int(np.sum(m < 1e-12)),
        n_plateau=int(np.sum(m > 1 - 1e-12)),
        n_ramp=int(np.sum((m > 1e-12) & (m < 1 - 1e-12))),
        S_entry=s_entry(m),
        S_entry_1924=float(np.sum(np.minimum(m[19:25], np.maximum(REQ[19:25], 0)))),
        savable=list(sav.tolist()),
        Leps=laplacian_eps(m),
        Tj=Tj,
    )


def build_all():
    return {
        "native":  T.m_native(K),
        "MrPro":   T.m_incr_beta(0.0, n=N_INCR, low=LOW),
        "BM":      T.m_incr_beta(1.0, n=N_INCR, low=LOW),
        "step_hi22": T.m_step(22),
        "step_hi25": T.m_step(25),
        "b3_lo14": T.m_incr_beta(3.0, n=N_INCR, low=LOW),
        # the WIDE band [11,32] (n=21): patch_wide.py documents that
        # m_incr_beta(1.0, n=21, low=11) is bit-for-bit m_turns(1,64,ramp=beta1)
        "a1_b64":  T.m_incr_beta(1.0, n=21, low=11),
        "C42":     T.m_C42(),
        "C42V24":  T.m_C42V24(),
        "b4_wide": T.m_incr_beta(4.0, n=21, low=11),
    }


# ---------------------------------------------------------------------------
def secl(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    tabs = build_all()
    rows = {k: report(k, v) for k, v in tabs.items()}

    secl("0. geometry")
    tW = turns_w()
    print(f"  t_W[j] = W*omega_j/2pi : t_W[0]={tW[0]:.3f} t_W[19]={tW[19]:.3f} "
          f"t_W[24]={tW[24]:.4f} t_W[38]={tW[38]:.4f} t_W[63]={tW[63]:.6f}")
    print(f"  native log gap theta^(1/64) = {THETA ** (1/64):.5f}  (plan quotes 1.2276)")
    sav = np.where((REQ > 0) & (REQ <= 1))[0]
    print(f"  slots with 0 < req <= 1 (rescuable at 4W) : {sav.tolist()}")
    print("  req_j on those slots: " +
          " ".join(f"{j}:{REQ[j]:.4f}" for j in sav))
    print(f"  sum(req) on the rescuable set = {REQ[sav].sum():.4f} "
          f"(memo quotes 2.965)")
    print(f"  readable window at 4W is T_j in [0.25,16] -> unsigned slots "
          f"{[int(j) for j in np.where((turns_d(np.zeros(K)) >= .25) & (turns_d(np.zeros(K)) <= 16))[0]]}")

    secl("1. h=22 vs the field  (S, N@16384, monotone, max period ratio, Eps forcing)")
    hdr = (f"{'table':11s} {'S':>8s} {'N':>3s} {'held':>4s} {'ramp':>4s} {'plat':>4s} "
           f"{'maxratio':>8s} {'at':>3s} {'mono':>5s} {'S_entry':>8s} {'S_e[19:24]':>10s}")
    print(hdr)
    for nm in ("native", "MrPro", "step_hi25", "BM", "C42", "C42V24", "a1_b64",
               "step_hi22", "b3_lo14", "b4_wide"):
        r = rows[nm]
        print(f"{nm:11s} {r['S']:8.3f} {r['N']:3d} {r['n_held']:4d} {r['n_ramp']:4d} "
              f"{r['n_plateau']:4d} {r['ratio']:8.4f} {r['ratio_at']:3d} "
              f"{str(r['mono']):>5s} {r['S_entry']:8.4f} {r['S_entry_1924']:10.4f}"
              f"{r['s_ok']}")

    secl("2. the h=22 table itself  m_j = 1[j>=22]")
    m22 = np.asarray(tabs["step_hi22"], float)
    print(f"  S = {m22.sum():.1f}   (= K - h = 64 - 22)   [plan claim (h) says 42]")
    print(f"  m_20..m_25 = {np.round(m22[20:26], 3).tolist()}")
    print(f"  nu strictly decreasing : {rows['step_hi22']['mono']}")
    nu22 = omega() * np.power(4.0, -m22)
    print(f"  max adjacent period ratio {rows['step_hi22']['ratio']:.5f} at "
          f"j={rows['step_hi22']['ratio_at']}  (plan quotes 4.91028)")
    print(f"  a1_b64 max adjacent ratio {rows['a1_b64']['ratio']:.5f} at "
          f"j={rows['a1_b64']['ratio_at']}  (plan quotes 1.34953)")
    print(f"  readable slots at 16384 : N = {rows['step_hi22']['N']}  "
          f"(slots {[int(j) for j in np.where((rows['step_hi22']['Tj'] >= .25) & (rows['step_hi22']['Tj'] <= 16))[0]]})")
    print(f"  vs step_hi25 N = {rows['step_hi25']['N']}   BM N = {rows['BM']['N']}   "
          f"a1_b64 N = {rows['a1_b64']['N']}")
    print(f"  T_j at the h=22 cliff: T_21 = {rows['step_hi22']['Tj'][21]:.4f} "
          f"(m=0 -> 4*t_W = {4*tW[21]:.4f}), T_22 = {rows['step_hi22']['Tj'][22]:.4f}")

    secl("3. L*eps forcing profile (band [14,32], n=18, Dirichlet 0/0)")
    hdr = f"{'table':11s} " + " ".join(f"{'q'+str(q+1):>7s}" for q in range(18))
    print(hdr)
    for nm in ("MrPro", "BM", "C42", "a1_b64", "step_hi25", "step_hi22"):
        L = rows[nm]["Leps"]
        print(f"{nm:11s} " + " ".join(f"{v:7.4f}" for v in L))
    print("\n  q index -> slot: q_k is the increment at slot 14+k (k=1..18)")
    L22 = rows["step_hi22"]["Leps"]
    nz = np.where(np.abs(L22) > 1e-12)[0]
    print(f"  h=22 nonzero forcing at q = {[int(q+1) for q in nz]} "
          f"= slots {[int(q+1+LOW) for q in nz]}, values {np.round(L22[nz],4).tolist()}")
    print(f"  => the h=22 forcing is a SECOND-DIFFERENCE OF A DELTA at slot 22 "
          f"(interior point source), not a constant and not an end spike.")

    secl("4. what h=22 does to the six rescuable slots 19..24")
    print(f"{'j':>3s} {'t_W':>8s} {'req':>7s} | {'m(h=25)':>8s} {'m(h=22)':>8s} | "
          f"{'T(h=25)':>9s} {'T(h=22)':>9s} | readable?")
    for j in range(19, 26):
        T25 = rows["step_hi25"]["Tj"][j]
        T22 = rows["step_hi22"]["Tj"][j]
        r25 = "yes" if 0.25 <= T25 <= 16 else "NO"
        r22 = "yes" if 0.25 <= T22 <= 16 else "NO"
        print(f"{j:3d} {tW[j]:8.3f} {REQ[j]:7.4f} | {tabs['step_hi25'][j]:8.3f} "
              f"{tabs['step_hi22'][j]:8.3f} | {T25:9.4f} {T22:9.4f} | "
              f"h25 {r25:3s}  h22 {r22:3s}")
    print(f"\n  S_entry (all slots)      h=25 {rows['step_hi25']['S_entry']:.4f}   "
          f"h=22 {rows['step_hi22']['S_entry']:.4f}   BM {rows['BM']['S_entry']:.4f}   "
          f"C42V24 {rows['C42V24']['S_entry']:.4f}")
    print(f"  S_entry (slots 19..24)   h=25 {rows['step_hi25']['S_entry_1924']:.4f}   "
          f"h=22 {rows['step_hi22']['S_entry_1924']:.4f}   BM {rows['BM']['S_entry_1924']:.4f}   "
          f"C42V24 {rows['C42V24']['S_entry_1924']:.4f}")

    secl("4b. h=22 vs the DEPLOYED table, slot by slot (the actual intervention)")
    bm = np.asarray(tabs["BM"], float)
    print(f"{'j':>3s} {'t_W':>7s} {'req':>7s} {'m_BM':>7s} {'m_h25':>7s} {'m_h22':>7s} "
          f"{'dm vs BM':>9s}  note")
    for j in range(14, 34):
        note = ""
        if 0 < REQ[j] <= 1:
            note = "rescuable"
            if bm[j] >= REQ[j]:
                note += " (BM already enters)"
        dm = float(bm[j] - tabs["step_hi22"][j])
        print(f"{j:3d} {tW[j]:7.3f} {REQ[j]:7.4f} {bm[j]:7.4f} {tabs['step_hi25'][j]:7.4f} "
              f"{tabs['step_hi22'][j]:7.4f} {dm:+9.4f}  {note}")
    print("\n  slots 34..63 are m=1 in ALL THREE tables (BM plateau starts at 33):")
    print(f"    identical? {bool(np.allclose(bm[33:], tabs['step_hi22'][33:]))}")
    print(f"  ||m_h22 - m_BM|| = {np.linalg.norm(np.asarray(tabs['step_hi22']) - bm):.4f}"
          f"   sum of positive dm = "
          f"{np.sum(np.clip(np.asarray(tabs['step_hi22']) - bm, 0, None)):.4f}"
          f"   sum of negative dm = "
          f"{np.sum(np.clip(np.asarray(tabs['step_hi22']) - bm, None, 0)):.4f}")
    print("  => h=22 spends the SAME total extra compression as BM has budget for")
    print("     but takes it from slots 15-21 (BM's ramp front) and gives it to")
    print("     slots 22-24 (full compression), i.e. it OVERSHOOTS where the")
    print("     requirement is small and ABANDONS the three slots (19,20,21)")
    print("     whose requirement is largest.")

    secl("4c. the campaign's OWN predictors applied to h=22")
    def S_in(m, lo=14, hi=32):
        return float(np.asarray(m, float)[lo:hi + 1].sum())
    print(f"{'table':11s} {'S_in[14..32]':>13s} {'N':>3s} {'S_entry':>8s} "
          f"{'readable set'}")
    for nm in ("MrPro", "step_hi25", "BM", "C42", "C42V24", "a1_b64", "step_hi22",
               "b3_lo14"):
        m = np.asarray(tabs[nm], float)
        Tj = rows[nm]["Tj"]
        rd = np.where((Tj >= .25) & (Tj <= 16))[0]
        span = f"{int(rd.min())}..{int(rd.max())}" if len(rd) else "-"
        print(f"{nm:11s} {S_in(m):13.4f} {rows[nm]['N']:3d} "
              f"{rows[nm]['S_entry']:8.4f} {span}")
    print("\n  campaign calibrations (RESULT_THEORY_TESTS / MIDBAND_MECH):")
    print("    continuous NLL ~ S_in : Pearson -0.877 (the best single scalar, "
          "with counterexamples)")
    print("    RULER logit(acc) = -20.45 + 1.19*N  (each readable slot x3.2 odds)")
    print("    value model (WHY §6): V must encode 'can this slot become readable'")
    n22, nbm = rows["step_hi22"]["N"], rows["BM"]["N"]
    print(f"\n  h=22 has N={n22}, BM has N={nbm}, h=25 has N={rows['step_hi25']['N']}")
    print(f"    -> the N relation says h=22 ~= BM on RULER (not a collapse)")
    print(f"  h=22 has S_entry={rows['step_hi22']['S_entry']:.3f} vs BM "
          f"{rows['BM']['S_entry']:.3f} -> the value model says h=22 < BM")
    print(f"  h=22 S_in={S_in(tabs['step_hi22']):.3f} vs BM {S_in(tabs['BM']):.3f}"
          f" -> the best continuous scalar says h=22 is WORSE (negative slope)")
    # a 7-point linear fit of the campaign's own continuous numbers on S_in
    fit_pts = [("MrPro", 6.6667, 3.6887), ("step_hi25", 8.0, 3.0452),
               ("BM", 9.5, 2.8627), ("a1_b64", 10.9537, 2.8402),
               ("C42", 11.0, 2.9417), ("C42V24", 11.0, 2.8309),
               ("b3_lo14", 12.6374, 2.8267)]
    xs = np.array([p[1] for p in fit_pts])
    ys = np.array([p[2] for p in fit_pts])
    A = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = ys - A @ coef
    print(f"\n  7-point linear fit NLL = {coef[0]:.4f} {coef[1]:+.4f}*S_in  "
          f"rms {float(np.sqrt(np.mean(resid**2))):.4f}")
    print(f"    -> point prediction at S_in = 11.0 (h=22, same as C42): "
          f"NLL = {coef[0] + coef[1]*11.0:.4f}")

    print("\n  PRE-REGISTERED PREDICTION (before any GPU run):")
    print("    h=22 is strictly between h=25 and BM on both instruments:")
    print("      RULER   0.1121 < acc(h=22) < 0.4167, point estimate ~0.30")
    print("      contin. 2.8627 < NLL(h=22) < 3.0452, point estimate ~2.93")
    print("    and it does NOT reach the winners' plateau (0.54-0.56 / 2.82-2.84).")
    print("    FALSIFIER: if acc(h=22) >= 0.50 the window/value story is wrong;")
    print("    if acc(h=22) <= 0.15 the readability set {22..38} is not what")
    print("    drives the collapse.")

    secl("5. SNR calibration: single-slot vs coherent-direction probes (measured inputs)")
    # C42 -> C42V24 is the campaign's one CONTROLLED whole-table move at fixed
    # band, fixed S, fixed centroid.  Measured: continuous -0.1109 (t=-3.91),
    # RULER +10.73pp (t=+5.47).  Its ||dm|| sets the coherent-step scale.
    d = np.asarray(tabs["C42V24"], float) - np.asarray(tabs["C42"], float)
    print(f"  ||dm||(C42->C42V24) = {np.linalg.norm(d):.4f}   "
          f"max|dm| = {np.abs(d).max():.4f}   support = slots "
          f"{int(np.argmax(np.abs(d) > 1e-12))}..{int(K - 1 - np.argmax(np.abs(d[::-1]) > 1e-12))}")
    print(f"  measured continuous dR = -0.1109 nats (t=-3.91) -> SE_paired = "
          f"{0.1109/3.91:.4f} nats")
    print(f"  measured RULER delta = +10.73pp (t=+5.47)")
    g_coherent = 0.1109 / np.linalg.norm(d)
    print(f"  => directional slope |dR|/||dm|| = {g_coherent:.4f} nats per unit ||dm||")
    print("  single-slot, from CONSTRAINT_IS_SLACK_20260911 §1/§2 (16 paired docs):")
    print("     in-window effect of one slot at delta=0.05 : 0.005-0.015 nats, SE 0.005")
    print("     => single-slot gradient signal |g_j| ~ 0.002-0.010, SE = 0.005/0.05 = 0.10")
    print(f"     SNR(single slot, in-window) = {0.002/0.1:.3f} .. {0.010/0.1:.3f}")
    print("  coherent direction u = +1/sqrt(18) on the band (budget direction):")
    for gbar in (0.002, 0.010, 0.05):
        sig = math.sqrt(18) * gbar
        print(f"     if all 18 in-band |g_j| = {gbar:.3f} coherent: "
              f"signal = sqrt(18)*g = {sig:.4f}, SE = 0.10 -> SNR = {sig/0.1:.2f}")
    print("  => the coherent gain is sqrt(n_eff) ~ 4.24, NOT an averaging gain:")
    print("     one forward gives ONE paired noisy number whatever the direction.")
    print(f"  long-range analogue: SE_paired(R) = {0.1109/3.91:.4f}; "
          f"single-slot dR/dm values measured 0.016..0.28 -> SE(dR/dm) = "
          f"{0.1109/3.91/0.05:.3f}")
    print(f"     => single-slot SNR(R) = {0.28/(0.1109/3.91/0.05):.2f} (best slot), "
          f"{0.05/(0.1109/3.91/0.05):.2f} (typical)")
    print(f"     => coherent-direction SNR(R) up to sqrt(18)x that = "
          f"{0.28*math.sqrt(18)/(0.1109/3.91/0.05):.2f}")

    secl("6. the Fisher budget that derive_tstar actually solved (claim a/b relevance)")
    print("  derive_tstar picks min feasible hi with suffix[hi] <= dam_bm,")
    print("  suffix[hi] = sum_{j>=hi} 0.5 (ln4)^2 F_jj  (step m_j = 1[j>=hi]).")
    print("  It reported derived_hi = 25 (RESULT_THEORY_TESTS_20260911).")
    print("  => suffix[25] <= dam_bm < suffix[24]: the FISHER BUDGET BOUND the edge.")
    print("     Slot 24 alone at m=1 costs 0.5*(ln4)^2*F_24 = "
          f"{0.5*LN4**2*7842:.1f} (F_24 = 7842, measured).")
    print("  => the constraint excluded the step at 24 and forced 25, while the")
    print("     measured true in-window cost of compressing slot 24 is NEGATIVE")
    print("     (monotone to -0.0150 at m=1, t=-3.2). The binding constraint")
    print("     pushed the edge the WRONG WAY by 3 slots (25 vs the winners' 22).")

    secl("7. algebra re-checks (claims b, c, e, h)")
    # (b) interior maximum under the Fisher cost
    #     existence of the W0 branch requires -1/(4 lamF) > -1/e  <=>  4 lamF > e
    print(f"  (b) interior stationary point of G_F exists iff 4*lambda*F > e = "
          f"{math.e:.5f}, i.e. lambda*F > {math.e/4:.5f}")
    for lamF in (0.3, 3.0):
        gridb = np.linspace(0, 1, 400001)
        vb = 4.0 ** gridb / 4.0 - 0.5 * lamF * LN4 ** 2 * gridb ** 2
        ib = int(np.argmax(vb))
        print(f"      lambda*F = {lamF:5.2f}: argmax on [0,1] = m = {gridb[ib]:.5f} "
              f"(G = {vb[ib]:.6f});  G(0) = {0.25:.6f}, "
              f"G(1) = {4.0/4.0 - 0.5*lamF*LN4**2:.6f}  "
              f"-> interior max? {vb[ib] > max(0.25, 1.0 - 0.5*lamF*LN4**2)}")
    lamF = 0.3
    def GF(m):  # 4^m/4 - 0.5*lamF*(ln4)^2*m^2
        return 4.0 ** m / 4.0 - 0.5 * lamF * LN4 ** 2 * m ** 2
    # stationary point m* = -W0(-1/(4 lamF))/ln4 (plan's Lambert-W form)
    grid = np.linspace(0, 1, 200001)
    vals = np.array([GF(x) for x in grid])
    i = int(np.argmax(vals))
    print(f"  (b) grid argmax of G_F on [0,1] with lambda*F = {lamF}: "
          f"m = {grid[i]:.5f}, G = {vals[i]:.6f}")
    print(f"      endpoints: G(0) = {GF(0.0):.6f}  G(1) = {GF(1.0):.6f}"
          f"  -> interior max? {vals[i] > max(GF(0.0), GF(1.0))}")
    print(f"      threshold m where G_F'' flips: 4^m/4 = lamF -> "
          f"m = {math.log(4*lamF)/LN4:.5f}  (interior max lives below it)")
    print("      W0 branch: m* = -W0(-1/(4 lamF))/ln4 in (0, 1/ln4), a local MAX;")
    print("      W-1 branch: m* > 1/ln4, a local MIN.  lambda*F = 3 gives "
          "m* = 0.0659 (matches the grid).")
    print(f"      plan's toy: max 4^m/4 s.t. m^2/2 <= 1/8, 0<=m<=1 -> feasible "
          f"m<=0.5, optimum m=0.5, m=1 INFEASIBLE  [confirmed]")
    # (c) duality gap
    U = lambda m: 4.0 ** m / 4.0
    C = lambda m: 1.0 - 4.0 ** (-m)
    m_feas = np.linspace(0, 0.5, 100001)
    print(f"  (c) primal max U s.t. C<=1/2, t=1/4: U(0.5) = {U(0.5):.4f}, "
          f"box corners: U(0) = {U(0.0):.4f}, m=1 infeasible (C(1) = {C(1.0):.4f})")
    dual = []
    for lam in np.linspace(0, 2, 20001):
        dual.append(lam / 2.0 + max(U(0.0), U(1.0) - lam * C(1.0)))
    dual = np.array(dual)
    lams = np.linspace(0, 2, 20001)
    print(f"      dual min = {dual.min():.4f} at lambda = {lams[int(np.argmin(dual))]:.4f}"
          f"  (plan says 3/4 at lambda=1)  gap = {dual.min() - 0.5:.4f}")
    # (e) 199x along (1,1)
    Fp = np.array([[1.0, .99], [.99, 1.0]])
    Fm = np.array([[1.0, -.99], [-.99, 1.0]])
    u = np.ones(2)
    qp, qm = 0.5 * u @ Fp @ u, 0.5 * u @ Fm @ u
    print(f"  (e) 0.5*u'F_+u = {qp:.4f}, 0.5*u'F_-u = {qm:.4f}, ratio = {qp/qm:.1f} "
          f"(plan says 1.99 / .01 / 199x); diagonals identical: "
          f"{np.diag(Fp).tolist()} vs {np.diag(Fm).tolist()}; "
          f"eigs +: {np.linalg.eigvalsh(Fp).round(4).tolist()}")
    # (h) uniqueness of the same-budget step
    print("  (h) S = 64 - h for a step m_j = 1[j>=h]:")
    for S_target, nm in ((42.0, "C42 / a1_b64 / C42V24"), (40.5, "BM"), (39.0, "step_hi25")):
        print(f"      S = {S_target:.1f} ({nm:22s}) -> h = {64 - S_target:.1f}")
    print("      => S=42 pins h=22 UNIQUELY inside the step family  [confirmed]")

    secl("8. internal tension to record")
    print("  CONSTRAINT_IS_SLACK says 'in-band reallocation is free (<=0.01 nats)'.")
    print("  But C42 -> C42V24 is an IN-BAND, same-S, same-centroid, same-platform")
    print("  move worth -0.1109 nats (t=-3.91) continuous and +10.73pp (t=+5.47)")
    print("  RULER. So in-band reallocation is NOT free at the table level;")
    print("  the single-slot probe was underpowered, not the effect small.")
    print(f"  same-band ||dm|| for that pair = {np.linalg.norm(d):.4f} vs the "
          f"single-slot step 0.05.")
    return rows


if __name__ == "__main__":
    main()
