#!/usr/bin/env python3
"""The budget axis of the frequency-allocation KKT, re-solved.  Companion to
ds_workspace/recon_20260910/ANALYTIC_BUDGET_20260911.md.

Run:  python3 ds_workspace/recon_20260910/code/budget_20260911.py
      (cwd = repo root; needs numpy only)

WHAT THIS FILE ESTABLISHES, and which of the three provenance classes each claim
is in (LESSONS.md discipline: proved / derived-under-stated-assumptions / guessed
-- never blended):

  P1  S = sum_j (K-j) * eps_j      eps_j = m_j - m_{j-1}, m_{-1} = 0        PROVED
      so for m_{K-1} = 1:  S = K - jbar,  jbar = sum_j j*eps_j.  The budget IS
      the complement of the mean slot index of the compression: every table is
      budget-equivalent to the pure step at that mean index.
  P2  c_j = K-j is in the kernel of the Dirichlet Laplacian (exact).        PROVED
      MrRoPE's eps ~ k is anti-harmonic -> S is MINIMISED at S = K - hi.
  P3  The output-KL / Fisher damage is DIAGONAL in m-coordinates:            PROVED
      D = (ln4)^2/2 * sum_j w_j m_j^2 (no cross terms).  Water-filling gives
      the damage-minimal profile at fixed budget: m_j = clip(mu/(2 w_j)).
  P4  Flat weights (w = const) are KILLED by a matched pair.                PROVED-by-measurement
  P5  The turn-weighted coordinate sum_j (turns_j(W))^p m_j^2 fits the 8
      measured Qwen in-window costs at R^2 = 0.992-0.996 for p in [1,2], and
      FAILS by 26-50x on the out-of-box arm evq_s_t2 -> local model only.
      DERIVED (fit); see the report for the exact provenance label.
  P6  WHY_20260911's mean(m) alignment of the Qwen in-window curve against OLMo
      scores is off by 10^3 in this coordinate.                       DERIVED
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.curvature_20260910 import tables as T  # noqa: E402

K = T.K
OLMO = dict(theta=500_000.0, window=4096, head_dim=128, low=14, n=18)
QWEN = dict(theta=1_000_000.0, window=32768, head_dim=128, low=23, n=17)


# --------------------------------------------------------------------------
# P1 / P2: the budget functional
# --------------------------------------------------------------------------

def profile(m):
    """eps_j = m_j - m_{j-1} for j = 0..K-1, with m_{-1} := 0."""
    m = np.asarray(m, dtype=float)
    return np.diff(np.concatenate([[0.0], m]))


def budget(m):
    """(S, S_identity, jbar, sum eps).  S_identity is the closed form."""
    m = np.asarray(m, dtype=float)
    e = profile(m)
    j = np.arange(K, dtype=float)
    return (float(m.sum()), float(K * m[-1] - (j * e).sum()),
            float((j * e).sum()), float(e.sum()))


def omega(theta, j):
    """Native inverse frequency; theta ** (-j/64) == theta ** (-2j/head_dim)."""
    return theta ** (-np.asarray(j, dtype=float) / float(K))


def turns(theta, window, j):
    return omega(theta, j) * window / (2.0 * math.pi)


def turn_w(cfg):
    """Damage weights in TURN units, normalised to the band's fast edge."""
    t = turns(cfg["theta"], cfg["window"], np.arange(K))
    return t / t[cfg["low"]]


def arms(cfg):
    """Budget-normalised arms in this checkpoint's own geometry."""
    lo, n = cfg["low"], cfg["n"]
    out = []
    for b in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, 64.0):
        out.append((f"beta_b{b:g}", T.m_incr_beta(b, n=n, low=lo)))
    for a, bb in ((1, 16), (1, 32), (1, 64), (2, 32), (0.5, 32), (1, 128),
                  (1, 256), (4, 32)):
        out.append((f"turns_a{a:g}_b{bb}",
                    T.m_turns(a, bb, cfg["theta"], cfg["window"],
                              cfg["head_dim"], ramp="beta1")))
    for a in (0.0, 0.002, 0.005, 0.01, 0.02):
        if a * lo < 1.0:
            out.append((f"leak_a{a:g}", T.m_leak(a, base="bm", n=n, low=lo)))
    return out


def non_normalised():
    """Arms outside the box {m in [0,1], m_63 = 1} -- listed, never fitted."""
    return [("native", T.m_native()),
            ("evq_d_t1", T.m_evq_deployed(1.0)),
            ("evq_d_t2", T.m_evq_deployed(2.0)),
            ("evq_d_t4", T.m_evq_deployed(4.0)),
            ("evq_s_t2", T.m_evq_shift(2.0)),
            ("evq_s_t4", T.m_evq_shift(4.0)),
            ("power_p2", T.m_power_shift(2.0)),
            ("power_p4", T.m_power_shift(4.0))]


def step(s):
    m = np.zeros(K)
    m[s:] = 1.0
    return m


def waterfill(w, S):
    """min sum_j w_j m_j^2 s.t. sum m_j = S, 0 <= m <= 1."""
    w = np.asarray(w, dtype=float)
    a, b = 0.0, 2.0 * w.max() * max(S, 1.0) + 1.0
    for _ in range(400):
        mu = 0.5 * (a + b)
        m = np.minimum(1.0, mu / (2.0 * np.maximum(w, 1e-300)))
        if m.sum() < S:
            a = mu
        else:
            b = mu
    mu = 0.5 * (a + b)
    return np.minimum(1.0, mu / (2.0 * np.maximum(w, 1e-300))), mu


# Measured in-window costs (Qwen2.5-3B frozen, 32K column, per-length offset
# already removed).  Source: recon_20260910/CONTRACTION_SUPPORT_20260910.md §5.
MEAS = [("evq_d_t4", -0.0043), ("evq_d_t1", -0.0004), ("leak_a0", -0.0013),
        ("leak_a0.005", -0.0038), ("leak_a0.01", -0.0060),
        ("leak_a0.02", +0.0026), ("power_p2", +0.0406), ("power_p4", +0.0668)]
HOLD = ("evq_s_t2", +3.1712)


def qhat(m, cfg, p=2.0):
    return float(((turn_w(cfg) ** p) * np.asarray(m, dtype=float) ** 2).sum())


def fit(p, cfg=QWEN):
    tw = turn_w(cfg) ** p
    x = np.array([float((tw * np.asarray(dict(arms(cfg) + non_normalised())[n],
                                           dtype=float) ** 2).sum())
                  for n, _ in MEAS])
    y = np.array([c for _, c in MEAS])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = 1.0 - float(((pred - y) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    return float(coef[0]), float(coef[1]), r2, x, y, pred


def main():
    print("=" * 78)
    print("P1  S = sum_j (K-j) eps_j  and  S = K*m_{K-1} - jbar")
    print("=" * 78)
    worst, nchk = 0.0, 0
    for cfg in (OLMO, QWEN):
        for _, m in arms(cfg) + non_normalised():
            S, Sid, _, _ = budget(m)
            worst = max(worst, abs(S - Sid))
            nchk += 1
    print(f"  max |S - (K*m_63 - jbar)|  over {nchk} arms = {worst:.3e}")
    print("  proof: m_j = sum_{i<=j} eps_i => sum_j m_j = sum_i eps_i * #{j>=i}"
          " = sum_i eps_i (K-i).")

    print()
    print("=" * 78)
    print("P2  c_j = K-j is harmonic; the band's budget range")
    print("=" * 78)
    c = (K - np.arange(K)).astype(float)
    Lc = np.array([2 * c[j] - (c[j - 1] if j > 0 else 0.0)
                   - (c[j + 1] if j < K - 1 else 0.0) for j in range(1, K - 1)])
    print(f"  max |(L c)_j| over interior j = {np.abs(Lc).max():.1e}  (affine => exact 0)")
    lo, n = OLMO["low"], OLMO["n"]
    kk = np.arange(1, n + 1, dtype=float)
    for nm, eps in (("MrRoPE  eps~k", kk / kk.sum()),
                    ("BM      eps~k(n+1-k)", (kk * (n + 1 - kk)) / (kk * (n + 1 - kk)).sum()),
                    ("uniform eps=1/n", np.full(n, 1.0 / n)),
                    ("front step eps=e_1", np.eye(n)[0]),
                    ("back  step eps=e_n", np.eye(n)[-1])):
        S = float(((K - (lo + kk)) * eps).sum())
        print(f"  {nm:24s} S={S:8.4f}  mean(m)={S / K:.4f}")

    print()
    print("=" * 78)
    print("P3  water-filling: the damage-minimal profile at fixed budget")
    print("=" * 78)
    w = turn_w(OLMO) ** 2.0
    print(f"  ramp ratio of m_j ~ 1/w_j: {1.0 / (1.0 / (turn_w(OLMO)[0] / turn_w(OLMO)[1]) ** 2.0):.4f}"
          f"  per slot  (== 1.5069 for w ~ turns^2)")
    for S in (42, 46, 50):
        m, mu = waterfill(w, S)
        j1 = int(np.flatnonzero(m > 1 - 1e-12)[0])
        print(f"  S={S:3d}  mu={mu:.4g}  first slot at m=1: {j1:2d}   Qhat={qhat(m, OLMO):.5g}")
    print(f"  m == 1 (uniform 4x downshift): S={K}  Qhat={float((w * np.ones(K) ** 2).sum()):.4g}")

    print()
    print("=" * 78)
    print("P4  the matched pair that kills flat weights")
    print("=" * 78)
    tabs = dict(arms(QWEN) + non_normalised())
    for nm in ("leak_a0.02", "power_p2"):
        m = np.asarray(tabs[nm], dtype=float)
        print(f"  {nm:10s} S={m.sum():7.3f}  sum_j m_j^2={float((m * m).sum()):8.4f}"
              f"  Qhat(p=2)={qhat(m, QWEN):9.5g}")
    print("  measured costs: +0.0026 vs +0.0406 nats -> flat weights DEAD.")

    print()
    print("=" * 78)
    print("P5  turn-weighted damage fit (8 in-box points), hold-out = evq_s_t2")
    print("=" * 78)
    for p in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        a, b, r2, x, y, pred = fit(p)
        qh = qhat(tabs[HOLD[0]], QWEN, p)
        print(f"  p={p:4.2f}  slope={a:11.4g}  offset={b:+.5f}  R2={r2:.4f}"
              f"  RMSE={float(np.sqrt(((pred - y) ** 2).mean())):.5f}"
              f" | holdout pred={b + a * qh:+.4f} vs measured {HOLD[1]:+.4f}")

    print()
    print("=" * 78)
    print("P6  WHY's mean(m) alignment, in the damage coordinate")
    print("=" * 78)
    tabO, tabQ = dict(arms(OLMO)), dict(arms(QWEN))
    for mo, mq in (("beta_b2", "leak_a0.02"), ("beta_b8", "leak_a0.03")):
        if mq not in tabQ:
            continue
        qo, qq = qhat(tabO[mo], OLMO), qhat(tabQ[mq], QWEN)
        print(f"  mean(m)={tabO[mo].sum() / K:.4f}: OLMo {mo} Qhat={qo:9.5g}"
              f" | Qwen {mq} Qhat={qq:9.5g}  ratio={qq / qo:8.1f}")

    print()
    print("=" * 78)
    print("P7  queued arms in the damage coordinate (the prediction)")
    print("=" * 78)
    print(f"  anchor: every arm with Qhat <= {qhat(tabQ['leak_a0.02'], QWEN):.4g} measured"
          f" |cost| <= 0.006 nats (Qwen 32K)")
    print(f"  {'arm':18s} {'S':>8s} {'mean(m)':>8s} {'Qhat':>10s} {'anchor/Qhat':>12s}")
    qa = qhat(tabQ["leak_a0.02"], QWEN)
    for nm, b in (("beta_b2", 2), ("beta_b3", 3), ("beta_b4", 4), ("beta_b6", 6),
                  ("beta_b8", 8)):
        m = T.m_incr_beta(float(b), n=OLMO["n"], low=OLMO["low"])
        print(f"  OLMo {nm:13s} {m.sum():8.3f} {m.sum() / K:8.4f} {qhat(m, OLMO):10.5g}"
              f" {qa / qhat(m, OLMO):12.1f}x")
    for bb in (128, 256, 512):
        m = T.m_turns(1, bb, OLMO["theta"], OLMO["window"], 128, ramp="beta1")
        print(f"  OLMo turns_a1_b{bb:<4d} {m.sum():8.3f} {m.sum() / K:8.4f}"
              f" {qhat(m, OLMO):10.5g} {qa / qhat(m, OLMO):12.1f}x")

    print()
    print("=" * 78)
    print("P8  the 12 pp residual pair")
    print("=" * 78)
    for a_, b_ in (("beta_b0.25", "turns_a1_b16"), ("turns_a1_b64", "turns_a2_b32")):
        ma, mb = np.asarray(tabO[a_], float), np.asarray(tabO[b_], float)
        ea, eb = profile(ma), profile(mb)
        j = np.arange(K, dtype=float)
        sda = math.sqrt(max(0.0, float((j * j * ea).sum() - (j * ea).sum() ** 2)))
        sdb = math.sqrt(max(0.0, float((j * j * eb).sum() - (j * eb).sum() ** 2)))
        d = ma - mb
        print(f"  {a_} vs {b_}: dS={ma.sum() - mb.sum():+.4f}"
              f"  sd(j)={sda:.4f} vs {sdb:.4f}  ||dm||2={float(np.sqrt((d * d).sum())):.4f}"
              f"  support {np.flatnonzero(np.abs(d) > 1e-12)[[0, -1]]}")
    print("  scores: beta_b0.25 0.1447, turns_a1_b16 0.2651 (d = 0.1204);"
          "  a1_b64 0.5384, a2_b32 0.5100 (d = 0.0284, inside noise)")


if __name__ == "__main__":
    main()
