#!/usr/bin/env python3
"""The N-mechanism: what the readable-slot count says about the two roads.

Companion to ds_workspace/recon_20260910/EVQ_LIMIT_20260911.md.

Run:  python3 ds_workspace/recon_20260910/code/evq_limit_20260911.py
      (cwd = repo root; numpy only)

COORDINATES (all defined here, nothing imported from a claim):
  omega_j = theta**(-j/K)                 native inverse frequency, slot j
  t_j(W)  = omega_j * W / (2 pi)          in-window TURN COUNT
  t_D,j   = 4 * t_j(W) * 4**(-m_j)        turn count at the 4x TEST distance
  N(m)    = #{ j : t_D,j in [0.25, 16] }  THE READABLE-SLOT COUNT
  g_j     = log2 t_j(W)                   affine in j:  G - lam*j
  h_j     = log2 t_D,j = 2 + g_j - 2 m_j
  readable  <=>  h_j in [-2, 4]  <=>  g_j - 2 m_j in [-4, 2]

Damage model (ANALYTIC_BUDGET 20260911 sec 2.1-2.4, calibrated on Qwen 32K):
  Qhat(m) = sum_j ( turns_j(W) / turns_lo )**p  *  m_j**2 ,   p in [1,2]
  cost    ~= c0 + a * Qhat        (so matched Qhat == matched in-window cost)
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

# 11 measured OLMo 16K RULER points (RESULT_SIGMA_SHAPE / RESOLUTION_MISSING_TERM)
SCORES = [
    ("mrpro", 0.0709), ("beta_b0.25", 0.1447), ("turns_a1_b16", 0.2651),
    ("turns_a0.5_b32", 0.1604), ("beta_b0.5", 0.2320), ("bm", 0.4167),
    ("turns_a2_b32", 0.5100), ("turns_a1_b64", 0.5384), ("beta_b2", 0.5001),
    ("K_taper", 0.3937), ("K_both", 0.3846),
]


# ---------------------------------------------------------------------------
# the mechanism
# ---------------------------------------------------------------------------
def turns_w(cfg, k=K):
    return cfg["theta"] ** (-np.arange(k, dtype=float) / k) * cfg["window"] / (2 * math.pi)


def log2_turns(cfg, k=K):
    return np.log2(turns_w(cfg, k))


def N_readable(m, cfg, mult=4.0, lo=0.25, hi=16.0, k=K):
    """Readable-slot count at `mult` x the training window."""
    td = mult * turns_w(cfg, k) * np.power(4.0, -np.asarray(m, dtype=float))
    return int(((td >= lo * (1 - 1e-9)) & (td <= hi * (1 + 1e-9))).sum())


def readable_set(m, cfg, mult=4.0, lo=0.25, hi=16.0, k=K):
    td = mult * turns_w(cfg, k) * np.power(4.0, -np.asarray(m, dtype=float))
    return np.flatnonzero((td >= lo * (1 - 1e-9)) & (td <= hi * (1 + 1e-9)))


def window_in_m(cfg, k=K):
    """Per-slot m-interval that makes the slot readable.

    readable <=> t_D in [0.25,16] <=> log2 t_D = 2 + g_j - 2 m_j in [-2,4]
             <=> g_j - 2 m_j in [-4,2]  <=>  m_j in [(g_j-2)/2, (g_j+4)/2].
    Width exactly 3 (== 6 log2-units / 2), sliding DOWN at lam/2 per slot.
    """
    g = log2_turns(cfg, k)
    return (g - 2.0) / 2.0, (g + 4.0) / 2.0


def qhat(m, cfg, p=2.0, k=K):
    tw = turns_w(cfg, k)
    w = (tw / tw[cfg["low"]]) ** p
    return float((w * np.asarray(m, dtype=float) ** 2).sum())


# ---------------------------------------------------------------------------
# arm bank
# ---------------------------------------------------------------------------
def olmo_arms():
    lo, n = OLMO["low"], OLMO["n"]
    A = {}
    A["native"] = T.m_native()
    A["interp"] = T.m_interp()
    A["mrpro"] = T.m_mrpro(n=n, low=lo)
    A["bm"] = T.m_incr_beta(1.0, n=n, low=lo)
    for b in (0.0, 0.25, 0.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        A[f"beta_b{b:g}"] = T.m_incr_beta(b, n=n, low=lo)
    for a, bb in ((0.5, 32), (1.0, 16), (1.0, 32), (1.0, 64), (1.0, 128),
                  (1.0, 256), (2.0, 32), (4.0, 32), (8.0, 32), (16.0, 32)):
        A[f"turns_a{a:g}_b{bb:g}"] = T.m_turns(
            a, bb, OLMO["theta"], OLMO["window"], OLMO["head_dim"], ramp="beta1")
    for b in (0.0, 1.0, 2.0):
        A[f"turnsmr_a1_b{32 * 2 ** b:g}"] = T.m_turns(
            1.0, 32 * 2 ** b, OLMO["theta"], OLMO["window"], OLMO["head_dim"],
            ramp="mrpro")
    for hi in (18, 20, 22, 25, 28, 30, 32, 36, 40, 45):
        A[f"step_hi{hi}"] = T.m_step(hi, lo=lo)
    for tau in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0):
        A[f"evq_deploy_t{tau:g}"] = T.m_evq_deployed(tau)
        A[f"evq_shift_t{tau:g}"] = T.m_evq_shift(tau, cfg=OLMO)
    # KNIFE taper arms (RESOLUTION_MISSING_TERM): m = 1 - delta*(j-hi), hi=32
    for d in (0.008, 0.015):
        A[f"taper_d{d:g}"] = T.m_taper(d, hi=32, lo=lo)
    return A


def knife_tapers():
    """The two measured KNIFE arms, reconstructed from their reported m_63."""
    out = {}
    for nm, m63 in (("K_taper", 0.805), ("K_both", 0.74)):
        m = np.zeros(K)
        m[:14] = 0.0
        n = 18
        q = np.arange(1, n + 1, dtype=float)
        w = q * (n + 1 - q)
        eps = w / w.sum()
        m[15:33] = np.cumsum(eps)
        j = np.arange(32, K, dtype=float)
        # linear decay from 1 at j=32 to m63 at j=63
        m[32:] = 1.0 - (1.0 - m63) * (j - 32.0) / (K - 1 - 32.0)
        out[nm] = m
    return out


# ---------------------------------------------------------------------------
# the exact N-optimal frontier: DP over slots
# ---------------------------------------------------------------------------
def frontier(cfg, mus, m_lo=0.0, m_hi=1.0, grid=1201, drop_max=None, p=2.0, k=K,
             pin_last=None):
    """max  N(m) - mu * Qhat(m)   over all tables m with m_j in [m_lo, m_hi]
    and nu strictly decreasing (m_{j+1} - m_j >= -drop_max).

    Time-optimal DP: state = quantised m_j.  Returns the Pareto set
    {(Qhat, N, m)} for the swept mu.
    """
    tw = turns_w(cfg, k)
    w = (tw / tw[cfg["low"]]) ** p
    lo_m, hi_m = window_in_m(cfg, k)
    if drop_max is None:
        drop_max = math.log(cfg["theta"]) / k / T.LN_S      # nu strictly decreasing
    step = (m_hi - m_lo) / (grid - 1)
    gs = m_lo + step * np.arange(grid)                      # candidate m values

    # per-slot reward: 1[readable] - mu * w_j * m^2
    rew = np.empty((k, grid))
    for j in range(k):
        ok = (gs >= lo_m[j]) & (gs <= hi_m[j])
        rew[j] = ok.astype(float)
    if pin_last is not None:
        rew[k - 1] = np.where(np.abs(gs - pin_last) <= step / 2.0 + 1e-12, 0.0, -1e9)
    # DP: V[i] = best value over slots j..k-1 given m_j = gs[i]
    i_min = np.searchsorted(gs, gs - drop_max, side="left")
    out = []
    for mu in mus:
        R = rew - mu * (w[:, None] * gs[None, :] ** 2)
        V = R[k - 1].copy()
        nxt = np.zeros((k, grid), dtype=np.int32)
        for j in range(k - 2, -1, -1):
            best = np.empty(grid)
            bi = np.empty(grid, dtype=np.int32)
            bv, bj = -np.inf, grid - 1
            for i in range(grid - 1, -1, -1):     # suffix max + argmax
                if V[i] > bv:
                    bv, bj = V[i], i
                best[i], bi[i] = bv, bj
            take = np.minimum(i_min, grid - 1)
            V = R[j] + np.where(i_min < grid, best[take], -np.inf)
            nxt[j] = bi[take]
        i0 = int(np.argmax(V))
        m = np.empty(k)
        m[0] = gs[i0]
        for j in range(k - 1):
            m[j + 1] = gs[nxt[j, int(np.searchsorted(gs, m[j]))]]
        out.append((mu, qhat(m, cfg, p=p), N_readable(m, cfg), m))
    return out


def pareto(points, key=("q", "n")):
    """Keep the non-dominated set in (Qhat, N) -- lower Qhat, higher N."""
    pts = sorted(points, key=lambda r: (r[1], -r[2]))
    keep, best_n = [], -1
    for r in pts:
        if r[2] > best_n:
            keep.append(r)
            best_n = r[2]
    return keep


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("0  coordinate check: is N the stated statistic?  (11 OLMo 16K points)")
    print("=" * 78)
    A = olmo_arms()
    A.update(knife_tapers())
    for nm, sc in SCORES:
        if nm not in A:
            print(f"  {nm:16s} MISSING")
            continue
        m = A[nm]
        rs = readable_set(m, OLMO)
        rng = f"{rs[0]}..{rs[-1]}" if len(rs) else "--"
        print(f"  {nm:16s} N={N_readable(m, OLMO):2d}  Qhat_p2={qhat(m, OLMO, 2.0):9.4f}"
              f"  S={m.sum():7.3f}  m63={m[-1]:.3f}  readable {rng}  RULER={sc:.4f}")

    print()
    print("  controls (no compressed band at all):")
    for nm in ("native", "interp"):
        m = A[nm]
        rs = readable_set(m, OLMO)
        print(f"  {nm:16s} N={N_readable(m, OLMO):2d}  Qhat_p2={qhat(m, OLMO, 2.0):9.4f}"
              f"  S={m.sum():7.3f}  readable {rs[0]}..{rs[-1]}")

    print()
    print("=" * 78)
    print("1  the ceiling: N <= 6/lam for any table with m' >= 0")
    print("=" * 78)
    for nm, cfg in (("OLMo(5e5,64)", OLMO), ("Qwen(1e6,64)", QWEN)):
        lam = math.log(cfg["theta"]) / K / math.log(2.0)
        print(f"  {nm:14s} lam=ln(th)/(K ln2)={lam:.5f} log2/slot -> 6/lam={6 / lam:.2f}"
              f"   (m in [0,1] both plateaus sit exactly on this)")
    lam = math.log(OLMO["theta"]) / K / math.log(2.0)
    print(f"  a single 0->1 jump buys 2/lam={2 / lam:.2f} more -> hard bound 8/lam={8 / lam:.2f}")

    print()
    print("=" * 78)
    print("2  N of the two roads at matched damage  (OLMo geometry)")
    print("=" * 78)
    print(f"  {'arm':18s} {'N':>3s} {'Qhat_p2':>10s} {'Qhat_p1':>10s} {'S':>8s} "
          f"{'span_m':>7s} {'held':>5s} {'plat':>5s}")
    order = (["native", "mrpro", "bm"] +
             [f"beta_b{b:g}" for b in (0.0, 0.25, 0.5, 2.0, 3.0, 4.0, 6.0, 8.0)] +
             [f"turns_a{a:g}_b{bb:g}" for a, bb in
              ((0.5, 32), (1.0, 16), (1.0, 32), (1.0, 64), (1.0, 128), (1.0, 256),
               (2.0, 32), (4.0, 32), (8.0, 32), (16.0, 32))] +
             [f"step_hi{h}" for h in (18, 20, 22, 25, 28, 30, 32, 36, 40, 45)] +
             [f"evq_deploy_t{t:g}" for t in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)] +
             [f"evq_shift_t{t:g}" for t in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)])
    for nm in order:
        m = A[nm]
        d = np.diff(m)
        print(f"  {nm:18s} {N_readable(m, OLMO):3d} {qhat(m, OLMO, 2.0):10.4f}"
              f" {qhat(m, OLMO, 1.0):10.4f} {m.sum():8.3f} {m[-1] - m[0]:7.3f}"
              f" {(np.abs(m) < 1e-12).sum():5d} {(np.abs(m - 1) < 1e-12).sum():5d}")

    print()
    print("=" * 78)
    print("3  EVQ-deployed: N vs tau (the smooth family's own frontier)")
    print("=" * 78)
    for tau in (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0):
        m = T.m_evq_deployed(tau)
        rs = readable_set(m, OLMO)
        rng = f"{rs[0]}..{rs[-1]}" if len(rs) else "--"
        print(f"  tau={tau:5.2f}  N={N_readable(m, OLMO):2d}  Qhat_p2={qhat(m, OLMO, 2.0):8.4f}"
              f"  S={m.sum():7.3f}  mean(m)={m.sum() / K:.4f}  readable slots {rng}")

    print()
    print("=" * 78)
    print("4  EVQ-shift: N vs tau (pure redistribution, span = native)")
    print("=" * 78)
    for tau in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0):
        m = T.m_evq_shift(tau, cfg=OLMO)
        rs = readable_set(m, OLMO)
        rng = f"{rs[0]}..{rs[-1]}" if len(rs) else "--"
        print(f"  tau={tau:5.2f}  N={N_readable(m, OLMO):2d}  Qhat_p2={qhat(m, OLMO, 2.0):9.4f}"
              f"  Qhat_p1={qhat(m, OLMO, 1.0):8.4f}  span_m={m[-1] - m[0]:7.4f}"
              f"  min m={m.min():+7.4f} max m={m.max():+7.4f}  '{rng}'")

    print()
    print("=" * 78)
    print("5  the exact N-optimal frontier (DP, no family assumed)")
    print("=" * 78)
    mus = sorted({0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
                  1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0})
    for tag, kw in (("box [0,1], m63 free", dict(m_lo=0.0, m_hi=1.0)),
                    ("box [0,1], m63 PINNED = 1", dict(m_lo=0.0, m_hi=1.0, pin_last=1.0)),
                    ("box [-.5,1.5], m63 free", dict(m_lo=-0.5, m_hi=1.5))):
        pts = frontier(OLMO, mus, **kw)
        pf = pareto(pts)
        print(f"  --- {tag}")
        print(f"  {'mu':>9s} {'Qhat_p2':>10s} {'N':>3s}  structure")
        for mu, q, n, m in pf:
            j1 = int(np.flatnonzero(m > 1 - 1e-9)[0]) if (m > 1 - 1e-9).any() else -1
            j0 = int(np.flatnonzero(m > 1e-9)[0]) if (m > 1e-9).any() else -1
            print(f"  {mu:9.2g} {q:10.4f} {n:3d}  first m>0 at {j0}, first m=1 at {j1},"
                  f" S={m.sum():7.3f}, m63={m[-1]:+.3f}")

    print()
    print("=" * 78)
    print("6  the exact N-optimal table at BM's damage budget (Qhat_p2 <= 0.0637)")
    print("=" * 78)
    for tag, kw in (("A  box[0,1], m63 free", dict(m_lo=0.0, m_hi=1.0)),
                    ("B  box[0,1], m63=1", dict(m_lo=0.0, m_hi=1.0, pin_last=1.0))):
        # bisect mu to sit just under the BM budget
        lo_mu, hi_mu = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo_mu + hi_mu)
            _, q, _, _ = frontier(OLMO, [mid], grid=601, **kw)[0]
            if q > 0.0637:
                lo_mu = mid
            else:
                hi_mu = mid
        _, q, n, m = frontier(OLMO, [hi_mu], grid=1201, **kw)[0]
        print(f"  {tag}: mu={hi_mu:.3e}  N={n}  Qhat_p2={q:.4f} (BM 0.0637)  "
              f"S={m.sum():.4f}  m0={m[0]:+.4f} m63={m[-1]:+.4f}")
        print("   m = [" + ", ".join(f"{v:.3f}" for v in m) + "]")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 7  the exact decomposition and the strip-riding candidate
# ---------------------------------------------------------------------------
def theta(cfg, k=K):
    """theta_j = (g_j - 2)/2: LOWER edge of the readable strip, in m units."""
    return (log2_turns(cfg, k) - 2.0) / 2.0


def wsplit(m, cfg, k=K):
    """(free, fast-won, slow-won, fast range, slow range, lists) -- exact split.

    READABLE <=> m_j in [theta_j, theta_j+3] cap [0,1], and theta is affine
    decreasing, so every slot falls into exactly one of five classes:
        theta_j > 1            NEVER readable
        0 < theta_j <= 1       readable iff m_j >= theta_j       (FAST contested)
        -2 <= theta_j <= 0     ALWAYS readable (strip contains all of [0,1])
        -3 <= theta_j < -2     readable iff m_j <= theta_j + 3   (SLOW contested)
        theta_j < -3           NEVER readable
    Nothing is hard-coded per geometry.
    """
    th = theta(cfg, k)
    m = np.asarray(m, float)
    fast = [j for j in range(k) if 0.0 < th[j] <= 1.0]
    free = [j for j in range(k) if -2.0 <= th[j] <= 0.0]
    slow = [j for j in range(k) if -3.0 <= th[j] < -2.0]
    never = [j for j in range(k) if th[j] > 1.0] + [j for j in range(k) if th[j] < -3.0]
    assert len(never) + len(fast) + len(free) + len(slow) == k
    fw = [j for j in fast if m[j] >= th[j]]
    sw = [j for j in slow if m[j] <= th[j] + 3.0]
    return len(free), len(fw), len(sw), fast, slow, fw, sw


def wing_arm(cfg, fast=True, slow=True, jc=32, k=K, margin=0.05):
    """Strip-riding candidate.

    FAST flank: m_j = theta_j + margin * t,  t = (j - j_first)/(j_last - j_first)
                (>= theta_j so readable; descends at delta - margin/5 < delta so
                 nu strictly decreases -- the exact ride, margin = 0, is the
                 knife edge where nu is CONSTANT)
    SLOW flank: m_j = theta_j + 3 - margin * (1 - t)
                (<= theta_j + 3 so readable; same legality slack)
    PLATEAU   : m = 1 from jc on (except where a flank has overwritten it).
    """
    th = theta(cfg, k)
    m = np.zeros(k)
    m[jc:] = 1.0
    if fast:
        js = [j for j in range(k) if 0.0 < th[j] <= 1.0]
        j0, j1 = js[0], js[-1]
        for j in js:
            t = (j - j0) / float(j1 - j0)
            m[j] = th[j] + margin * t
    if slow:
        js = [j for j in range(k) if -3.0 <= th[j] < -2.0]
        j0, j1 = js[0], js[-1]
        for j in js:
            t = (j - j0) / float(j1 - j0)
            m[j] = th[j] + 3.0 - margin * (1.0 - t)
    return cone_repair(m, cfg)          # keep the nu-decreasing guard satisfied


def nu_sorted(m, cfg, k=K):
    """The L7 guard: nu must be strictly decreasing (m may decrease by < lam)."""
    nu = cfg["theta"] ** (-np.arange(k, dtype=float) / k) * np.power(4.0, -np.asarray(m, float))
    return bool((np.diff(nu) < 0).all())


def section7():
    print()
    print("=" * 78)
    print("7  exact decomposition:  N = free + #{fast contested won} + #{slow won}")
    print("=" * 78)
    A = olmo_arms()
    A.update(knife_tapers())
    for nm, sc in SCORES:
        m = A[nm]
        fr, fw, sw, fa, sl, fj, sj = wsplit(m, OLMO)
        print(f"  {nm:16s} N={N_readable(m, OLMO):2d} = {fr} + {fw} + {sw}"
              f"   fast won {fj}   slow won {sj}   RULER={sc:.4f}")

    print()
    for tag, cfg in (("OLMo", OLMO), ("Qwen", QWEN)):
        fr, fw, sw, fa, sl, _, _ = wsplit(np.zeros(K), cfg)
        print(f"  {tag}: free={fr} (slots {fa[0] + len(fa)}..{sl[0] - 1})"
              f"  fast contested={len(fa)} {fa[0]}..{fa[-1]}"
              f"  slow contested={len(sl)} {sl[0]}..{sl[-1]}"
              f"  -> ceiling {fr + len(fa) + len(sl)}")
    print("  identity: strip slope dtheta/dj = -ln(theta)/(2K ln2) = the slope at")
    print("            which nu is CONSTANT -> an exact ride holds nu fixed.")

    print()
    print("=" * 78)
    print("8  the strip-riding candidates")
    print("=" * 78)
    cand = {
        "BM (anchor)": T.m_incr_beta(1.0, n=OLMO["n"], low=OLMO["low"]),
        "A_fast (fast wing)": wing_arm(OLMO, fast=True, slow=False),
        "A_slow (slow wing)": wing_arm(OLMO, fast=False, slow=True),
        "A_both (both wings)": wing_arm(OLMO, fast=True, slow=True),
        "A_both margin0.10": wing_arm(OLMO, fast=True, slow=True, margin=0.10),
        "A_slow jc=46": wing_arm(OLMO, fast=False, slow=True, jc=46),
        "step_hi19": np.where(np.arange(K) >= 19, 1.0, 0.0),
    }
    print(f"  {'arm':22s} {'N':>3s} {'S':>8s} {'Qhat_p2':>9s} {'Qhat_p1':>9s}"
          f" {'nu<0':>6s}  readable")
    for nm, m in cand.items():
        fr, fw, sw, fa, sl, _, _ = wsplit(m, OLMO)
        rs = readable_set(m, OLMO)
        print(f"  {nm:22s} {N_readable(m, OLMO):3d} {m.sum():8.3f}"
              f" {qhat(m, OLMO, 2.0):9.4f} {qhat(m, OLMO, 1.0):9.4f}"
              f" {str(nu_sorted(m, OLMO)):>6s}"
              f"  {rs[0]}..{rs[-1]} (win {fw}+{sw})")

    print()
    print("  see section 12 for the final candidate's full m vector and nu profile.")

    print()
    print("=" * 78)
    print("9  does the N-mapping survive being read as two statistics?")
    print("=" * 78)
    xs = np.array([[N_readable(A[nm], OLMO), A[nm].sum()] for nm, _ in SCORES], float)
    ys = np.array([s for _, s in SCORES])
    Amat = np.hstack([xs, np.ones((len(ys), 1))])
    coef, *_ = np.linalg.lstsq(Amat, ys, rcond=None)
    pred = Amat @ coef
    print(f"  score ~ a*N + b*S + c :  a={coef[0]:+.4f} b={coef[1]:+.4f} c={coef[2]:+.4f}"
          f"   RMSE={float(np.sqrt(((pred - ys) ** 2).mean())):.4f}")
    for nm, x, y, p in zip([n for n, _ in SCORES], xs, ys, pred):
        print(f"    {nm:16s} N={int(x[0]):2d} S={x[1]:7.3f}  meas={y:.4f}"
              f"  fit={p:+.4f}  resid={y - p:+.4f}")
    print()
    for tag, cols in (("N only", [0]), ("S only", [1])):
        Am = np.hstack([xs[:, cols], np.ones((len(ys), 1))])
        cf, *_ = np.linalg.lstsq(Am, ys, rcond=None)
        pr = Am @ cf
        print(f"  {tag:8s} RMSE={float(np.sqrt(((pr - ys) ** 2).mean())):.4f}"
              f"   coef={np.round(cf, 4)}")


# ---------------------------------------------------------------------------
# 10  deliverable: the greedy frontier and the final candidate
# ---------------------------------------------------------------------------
def greedy_frontier(cfg, k=K):
    """Exact solution of  max N  s.t.  Qhat(m) <= eps,  m in [0,1].

    N is separable and each slot's cheapest readable point is
        m_j = theta_j                (fast contested: the strip's lower edge)
        m_j = 0                      (free: 0 is already inside the strip)
        m_j = theta_j + 3            (slow contested: the strip's upper edge)
    so the problem is a KNAPSACK with uniform value 1 and per-slot cost
    c_j = w_j * m_j^2, solved by sorting.  Returns the (cost, N, m) frontier.
    """
    th = theta(cfg, k)
    tw = turns_w(cfg, k)
    w = (tw / tw[cfg["low"]]) ** 2
    free = [j for j in range(k) if -2.0 <= th[j] <= 0.0]
    items = []
    for j in range(k):
        if 0.0 < th[j] <= 1.0:
            items.append((w[j] * th[j] ** 2, j, th[j]))
        elif -3.0 <= th[j] < -2.0:
            items.append((w[j] * (th[j] + 3.0) ** 2, j, th[j] + 3.0))
    items.sort()
    out, cost, m = [], float(sum(w[j] for j in free) * 0.0), np.zeros(k)
    N = len(free)
    out.append((cost, N, m.copy()))
    for c, j, val in items:
        cost += c
        m = m.copy()
        m[j] = val
        N += 1
        out.append((cost, N, m))
    return out, free, items


def deliverable():
    print()
    print("=" * 78)
    print("10  the exact N-frontier at fixed in-window damage (knapsack, no DP)")
    print("=" * 78)
    pts, free, items = greedy_frontier(OLMO)
    print(f"  free readable slots (any m in [0,1]): {free[0]}..{free[-1]} = {len(free)}")
    print(f"  contested slots, cheapest first: "
          f"{[(j, round(c, 6)) for c, j, _ in items]}")
    print()
    print(f"  {'cum Qhat_p2':>11s} {'N':>3s}   note")
    for c, N, m in pts:
        note = "free only" if m.sum() == 0 else (
            "newest slot " + str(int(np.flatnonzero(m > 0)[-1])))
        print(f"  {c:11.6f} {N:3d}   {note}")
    print()
    print("  three-band reference points:")
    for nm, mm in (("bm", T.m_incr_beta(1.0, n=OLMO["n"], low=OLMO["low"])),
                   ("beta_b2", T.m_incr_beta(2.0, n=OLMO["n"], low=OLMO["low"])),
                   ("a1_b64", T.m_turns(1.0, 64, OLMO["theta"], OLMO["window"],
                                        128, ramp="beta1"))):
        fr, fw, sw, _, _, _, _ = wsplit(mm, OLMO)
        print(f"    {nm:8s} N={N_readable(mm, OLMO):2d}  Qhat_p2={qhat(mm, OLMO, 2.0):.4f}"
              f"  S={mm.sum():.3f}")

    print()
    print("=" * 78)
    print("11  THE CANDIDATE (OLMo geometry), full 64-slot m vector")
    print("=" * 78)
    m = build_arm(OLMO, fast=True, slow=True, jc=32)
    fr, fw, sw, fa, sl, fj, sj = wsplit(m, OLMO)
    rs = readable_set(m, OLMO)
    nu = OLMO["theta"] ** (-np.arange(K, dtype=float) / K) * np.power(4.0, -m)
    print(f"  N={N_readable(m, OLMO)}  S={m.sum():.4f}  span={m[-1] - m[0]:.4f}"
          f"  Qhat_p2={qhat(m, OLMO, 2.0):.4f}  Qhat_p1={qhat(m, OLMO, 1.0):.4f}")
    print(f"  readable {rs[0]}..{rs[-1]} ({len(rs)} slots; fast won {fj}, slow won {sj})")
    print(f"  nu strictly decreasing: {nu_sorted(m, OLMO)}"
          f"   min gap {np.diff(np.log(nu)).max():.5f} nats"
          f"   max gap {(-np.diff(np.log(nu))).max():.5f} nats")
    print("  m = [" + ", ".join(f"{v:.4f}" for v in m) + "]")
    print("  nu/nu_native - 1 (per slot, x100) = ["
          + ", ".join(f"{100 * (v - 1):+.1f}" for v in
                      (nu / (OLMO["theta"] ** (-np.arange(K, dtype=float) / K)))) + "]")


def cone_repair(m, cfg, k=K):
    """Minimal repair that makes nu strictly decreasing (the L7 guard IS on nu).

    nu_j = omega_j 4^{-m_j} decreases  <=>  m_{j+1} - m_j > -ln(theta)/(K ln4).
    Raising a later slot is the minimal fix and never costs readability, since
    the strip's own slope is exactly that bound.
    """
    dm = math.log(cfg["theta"]) / (k * math.log(4.0)) * (1.0 - 1e-9)
    m = np.array(m, dtype=float)
    for j in range(k - 1):
        if m[j + 1] < m[j] - dm:
            m[j + 1] = m[j] - dm
    return m


def build_arm(cfg, fast=True, slow=True, jc=32, k=K, inset=1e-3):
    """The candidate: strip-riding wings + plateau, then the nu-cone repair.

    `inset` keeps the ride strictly INSIDE the strip (the exact ride sits on the
    boundary and any rounding throws the last slot out).  The ride's slope is
    the strip's own slope, which IS the nu-degeneracy slope, so these flanks are
    nu-flat by construction -- the cone repair is what keeps the table legal,
    and the residual is 1e-9 nats per slot, not a physical quantity.
    """
    th = theta(cfg, k)
    m = np.zeros(k)
    m[jc:] = 1.0
    if fast:
        for j in range(k):
            if 0.0 < th[j] <= 1.0:
                m[j] = th[j] + inset
    if slow:
        for j in range(k):
            if -3.0 <= th[j] < -2.0:
                m[j] = th[j] + 3.0 - inset
    return cone_repair(m, cfg)


def final_report():
    print()
    print("=" * 78)
    print("12  FINAL: the candidate and its decomposition (OLMo), nu-guard repaired")
    print("=" * 78)
    cands = [("CANDIDATE  (both wings)", build_arm(OLMO, True, True)),
             ("C_fast     (fast wing only)", build_arm(OLMO, True, False)),
             ("C_slow     (slow wing only)", build_arm(OLMO, False, True)),
             ("C_slow jc=46 (no mid plateau)", build_arm(OLMO, False, True, jc=46)),
             ("BM         (anchor)",
              T.m_incr_beta(1.0, n=OLMO["n"], low=OLMO["low"]))]
    for tag, m in cands:
        fr, fw, sw, fa, sl, fj, sj = wsplit(m, OLMO)
        rs = readable_set(m, OLMO)
        nu = OLMO["theta"] ** (-np.arange(K, dtype=float) / K) * np.power(4.0, -m)
        gaps = -np.diff(np.log(nu))
        print(f"  {tag:32s} N={len(rs):2d} (free {fr} + fast {fw} + slow {sw})"
              f"  S={m.sum():7.3f}  Qhat_p2={qhat(m, OLMO, 2.0):7.4f}"
              f"  Qhat_p1={qhat(m, OLMO, 1.0):6.3f}  {rs[0]}..{rs[-1]}"
              f"  nu-dec {bool((gaps > 0).all())}")
    print()
    print("  CANDIDATE m vector, full 64 entries:")
    print("   [" + ", ".join(f"{v:.4f}" for v in cands[0][1]) + "]")
    print()
    print("  CANDIDATE nu/native - 1 (%) per slot:")
    m = cands[0][1]
    nu = OLMO["theta"] ** (-np.arange(K, dtype=float) / K) * np.power(4.0, -m)
    nat = OLMO["theta"] ** (-np.arange(K, dtype=float) / K)
    print("   [" + ", ".join(f"{100 * (a / b - 1):+.1f}" for a, b in zip(nu, nat)) + "]")
    print()
    print("  Qwen geometry, same construction:")
    for tag, m in (("CANDIDATE", build_arm(QWEN, True, True, jc=40)),
                   ("C_slow   ", build_arm(QWEN, False, True, jc=40)),
                   ("BM anchor", T.m_incr_beta(1.0, n=QWEN["n"], low=QWEN["low"]))):
        rs = readable_set(m, QWEN)
        fr, fw, sw, fa, sl, fj, sj = wsplit(m, QWEN)
        print(f"    {tag} N={len(rs):2d} (free {fr} + fast {fw} + slow {sw})"
              f"  S={m.sum():7.3f}  Qhat_p2={qhat(m, QWEN, 2.0):8.4f}"
              f"  {rs[0]}..{rs[-1]}")
    print("  Qwen candidate m[23:56]: [" + ", ".join(
        f"{v:.3f}" for v in build_arm(QWEN, True, True, jc=40)[23:56]) + "]")
