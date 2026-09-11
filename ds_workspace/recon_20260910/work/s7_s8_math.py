"""§7 (cross-model transport) and §8 (conditional-EVQ reference) verification.

Pure numpy/mpmath-free CPU math.  Nothing here needs a GPU, a model or a
checkpoint; every number the Pro plan states in §7 and §8 is either reproduced
or refuted below.
"""
from __future__ import annotations

import math

import numpy as np

K = 64
LN4 = math.log(4.0)

OLMO = dict(theta=5e5, window=4096, head_dim=128)      # source
QWEN3B = dict(theta=1e6, window=32768, head_dim=128)   # target of §7
QWEN15B = dict(theta=1e6, window=32768, head_dim=128)  # our Qwen instrument


# ---------------------------------------------------------------------------
# table builders (local, self-contained; mirror experiments/curvature_20260910)
# ---------------------------------------------------------------------------
def m_native(k=K):
    return np.zeros(k)


def m_incr_beta(b, n, low, shape_a=1.0, k=K):
    kk = np.arange(1, int(n) + 1, dtype=np.float64)
    w = np.power(kk, float(shape_a)) * np.power(int(n) + 1 - kk, float(b))
    eps = w / w.sum()
    mq = np.concatenate([[0.0], np.cumsum(eps)])
    out = np.zeros(k)
    out[low:low + int(n) + 1] = mq
    out[low + int(n) + 1:] = 1.0          # the plateau tail
    return out


def band_from_turns(alpha, beta, theta, window, head_dim, k=K):
    """YaRN's find_correction_dim read as a TURN-COUNT window (server tables.py)."""
    lo = head_dim * math.log(window / (float(beta) * 2 * math.pi)) / (2 * math.log(theta))
    hi = head_dim * math.log(window / (float(alpha) * 2 * math.pi)) / (2 * math.log(theta))
    return int(math.floor(lo)), int(math.ceil(hi))


def m_turns(alpha, beta, theta, window, head_dim=128, ramp="beta1", k=K):
    lo, hi = band_from_turns(alpha, beta, theta, window, head_dim, k=k)
    lo, hi = max(0, lo), min(k - 1, hi)
    n = hi - lo
    if ramp == "beta1":
        return m_incr_beta(1.0, n, lo)
    if ramp == "mrpro":
        return m_incr_beta(0.0, n, lo)
    raise KeyError(ramp)


def m_mrpro(n, low, k=K):
    q = np.clip(np.arange(k) - low, 0, n).astype(np.float64)
    m = q * (q + 1.0) / (n * (n + 1.0))
    m[np.arange(k) >= low + n + 1] = 1.0
    return m


def eps_of(m):
    """eps_k = m_k - m_{k-1} with m_{-1} := 0, so k = 0..K-1 and sum eps = m_63.

    S = sum_j m_j = sum_k (K-k) eps_k = K - mu_eps  whenever m_{K-1} = 1, which
    is the identity tables.py quotes.  eps_0 = m_0 is zero for every table here.
    """
    return np.diff(np.concatenate([[0.0], np.asarray(m, dtype=np.float64)]))


def moments(m):
    e = eps_of(m)
    kk = np.arange(0, K, dtype=np.float64)
    mu = float((kk * e).sum())
    var = float(((kk - mu) ** 2 * e).sum())
    return mu, var


def S_of(m):
    return float(np.sum(m))


# ---------------------------------------------------------------------------
# §8
# ---------------------------------------------------------------------------
def phi_tau(u, tau):
    """Exact inverse CDF of rho_tau(y) = tau cosh[tau(y-1/2)] / (2 sinh(tau/2))."""
    return 0.5 + np.arcsinh((2.0 * u - 1.0) * math.sinh(tau / 2.0)) / tau


def solve_tau_geom(theta, n=21, k=K):
    """Solve 2 tanh(tau/2)/tau = n*d0/R, d0 = ln(theta)/K, R = n*d0 + ln4."""
    d0 = math.log(theta) / k
    R = n * d0 + LN4
    target = n * d0 / R
    lo, hi = 1e-9, 60.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        f = 2.0 * math.tanh(mid / 2.0) / mid if mid > 0 else 1.0
        if f > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), target, d0, R


def m_cond_evq(tau, theta, lo=11, n=21, k=K):
    """x_{lo+q} = x_lo + R*phi_tau(q/n); m = (x - x_native)/ln4."""
    d0 = math.log(theta) / k
    R = n * d0 + LN4
    q = np.arange(0, n + 1, dtype=np.float64)
    x = R * phi_tau(q / n, tau)              # x_lo := 0
    m = (x - q * d0) / LN4
    out = np.zeros(k)
    out[lo:lo + n + 1] = m
    out[lo + n + 1:] = 1.0
    return out, R, d0


# ---------------------------------------------------------------------------
# §7
# ---------------------------------------------------------------------------
def transport_linear(m_src, src, tgt, k=K):
    """k' = K_t/ln(theta_t) * [ln(W_t/W_s) + (ln(theta_s)/K_s)*k].

    Fractional k' is split between the two adjacent slots by centroid (linear
    interpolation weights), so sum(eps) and sum(k*eps) are both preserved.
    Returns (m_tgt, info).
    """
    Kt, Ks = k, k
    a = Kt / math.log(tgt["theta"]) * math.log(tgt["window"] / src["window"])
    b = (math.log(src["theta"]) / Ks) * Kt / math.log(tgt["theta"])
    e = eps_of(m_src)
    kk = np.arange(0, K, dtype=np.float64)   # eps index k IS the slot position
    kp = a + b * kk
    e_new = np.zeros(K)
    for kk_i, kp_i, e_i in zip(kk, kp, e):
        if e_i == 0.0:
            continue
        flo = math.floor(kp_i)
        w_hi = kp_i - flo            # weight on ceil
        w_lo = 1.0 - w_hi            # weight on floor
        # slot index j holds eps_{k=j} meaning m_j - m_{j-1}
        for idx, w in ((int(flo), w_lo), (int(flo) + 1, w_hi)):
            if w <= 0:
                continue
            if not (0 <= idx <= K - 1):
                raise ValueError(f"transported mass out of range at k'={kp_i:.4f}")
            e_new[idx] += e_i * w
    m = np.cumsum(e_new)
    return m, dict(a=a, b=b, kp=kp, eps_new=e_new)


def main():
    # ---------------------------------------------------------------
    # STEP 0: validate every arm reconstruction against its measured sum_m
    # ---------------------------------------------------------------
    print("=" * 78)
    print("STEP 0  ARM RECONSTRUCTION vs MEASURED sum_m (from *_summary.json)")
    print("=" * 78)
    measured = {
        "turns_a1_b64  (m_incr_beta(1.0,21,11))": (m_incr_beta(1.0, 21, 11), 42.0),
        "turns_a1_b16  (m_incr_beta(1.0,16,11))": (m_incr_beta(1.0, 16, 11), 38.5),
        "turns_a0p5_b32": (m_turns(0.5, 32.0, 5e5, 4096), 33.5),
        "turns_a2_b32": (m_turns(2.0, 32.0, 5e5, 4096), 33.5),
        "b3_lo14       (m_incr_beta(3.0,18,14))": (m_incr_beta(3.0, 18, 14), 43.637372802960215),
        "wide_b4p0     (m_incr_beta(4.0,21,11))": (m_incr_beta(4.0, 21, 11), 46.681784536706814),
        "step_hi25     (m_step(25,14))": (np.concatenate([np.zeros(25), np.ones(39)]), 39.0),
        "BM            (m_incr_beta(1.0,18,14))": (m_incr_beta(1.0, 18, 14), 40.5),
        "MrPro         (m_incr_beta(0.0,18,14))": (m_incr_beta(0.0, 18, 14), 37.66666666666667),
    }
    for name, (tab, sm) in measured.items():
        got = S_of(tab)
        flag = "OK " if abs(got - sm) < 5e-6 else "MISMATCH"
        print(f"  {flag} {name:42s} built S={got:12.8f}  measured={sm:12.8f}"
              f"  d={got-sm:+.2e}")
    print(f"  note: turns_a0p5_b32 / turns_a2_b32 measured sum_m are on the QWEN "
          f"panel (33.5),\n        not OLMo -- listed only to show the turn-window "
          f"band is config-derived.")

    print("\n" + "=" * 78)
    print("§8  CONDITIONAL-EVQ REFERENCE — verifying the Pro plan's numbers")
    print("=" * 78)

    tau, target, d0, R = solve_tau_geom(OLMO["theta"])
    print(f"\n  geometry: theta={OLMO['theta']:.0f} K={K} n=21")
    print(f"    d0 = ln(theta)/K          = {d0:.12f}")
    print(f"    R  = n*d0 + ln4           = {R:.12f}")
    print(f"    n*d0/R (target ratio)     = {target:.12f}")
    lhs = 2.0 * math.tanh(tau / 2.0) / tau
    print(f"    solved tau                = {tau:.10f}   (plan claims 2.0301373113)")
    print(f"    check 2 tanh(tau/2)/tau   = {lhs:.12f}  vs target {target:.12f}"
          f"   resid={lhs-target:+.3e}")

    m, R2, d0b = m_cond_evq(tau, OLMO["theta"])
    mu, var = moments(m)
    print(f"\n  table x_(lo+q) = x_lo + R*phi_tau(q/n),  nu_j = exp(-x_j), band [11,32]")
    print(f"    S = sum m                 = {S_of(m):.12f}   (plan claims 42)")
    print(f"    centroid mu = sum k eps_k = {mu:.12f}   (64 - S = {64-S_of(m):.12f})")
    print(f"    increment variance        = {var:.8f}   (plan claims ~18.78373)")
    e = eps_of(m)
    kk = np.arange(0, K, dtype=np.float64)
    # alternative readings of "increment variance"
    w = e[e > 0]
    kpos = kk[e > 0]
    print(f"      [alt] unweighted var of k over support   = "
          f"{np.var(kpos, ddof=0):.8f}")
    print(f"      [alt] unweighted var (ddof=1)            = "
          f"{np.var(kpos, ddof=1):.8f}")
    print(f"      [alt] var of eps values themselves       = {np.var(w, ddof=0):.8f}")
    print(f"      n_positive_eps = {len(w)}, support k = "
          f"{{{int(kpos.min())}..{int(kpos.max())}}}")

    print(f"\n  monotonicity / ordering / endpoints:")
    print(f"    m monotone non-decreasing : {bool(np.all(np.diff(m) >= -1e-15))}")
    print(f"    m in [0,1]                : {bool(m.min() >= -1e-15 and m.max() <= 1+1e-15)}")
    print(f"    in-band strictly increasing: {bool(np.all(np.diff(m[11:33]) > 0))}")
    print(f"    m[0..10] all zero         : {bool(np.all(m[:11] == 0))}")
    print(f"    m[33..63] all one         : {bool(np.all(m[33:] == 1))}")
    nu = np.exp(-(np.arange(K) * d0 + 0.0) * 0)  # placeholder
    print(f"    eps positive in band      : {bool(np.all(e[12:33] > 0))}  "
          f"min eps={e[12:33].min():.6f}")

    # compare against the winner's shape
    win = m_incr_beta(1.0, 21, 11)
    print(f"\n  comparison with turns_a1_b64 = m_incr_beta(1.0,n=21,low=11):")
    print(f"    winner  S={S_of(win):.6f}  mu={moments(win)[0]:.6f}  "
          f"var={moments(win)[1]:.6f}")
    print(f"    condEVQ S={S_of(m):.6f}  mu={moments(m)[0]:.6f}  var={var:.6f}")
    print(f"    max |m_condEVQ - m_win| in band = "
          f"{np.abs(m[11:33]-win[11:33]).max():.6f}")
    print(f"    identical outside band          = "
          f"{bool(np.allclose(m[:11], win[:11]) and np.allclose(m[33:], win[33:]))}")

    print("\n  FULL 64-VALUE TABLE (m_j, nu_j, and the winner for reference)")
    print(f"  {'j':>3s} {'m_condEVQ':>12s} {'nu_condEVQ':>14s} "
          f"{'m_win':>10s} {'nu_win':>14s} {'m_diff':>10s}")
    for j in range(K):
        xj = (math.log(OLMO["theta"]) / K) * j + m[j] * LN4
        nuj = math.exp(-xj)
        xw = (math.log(OLMO["theta"]) / K) * j + win[j] * LN4
        nuw = math.exp(-xw)
        print(f"  {j:3d} {m[j]:12.8f} {nuj:14.6e} {win[j]:10.6f} {nuw:14.6e} "
              f"{m[j]-win[j]:+10.3e}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("§7  CROSS-MODEL TRANSPORT — verifying the Pro plan's numbers")
    print("=" * 78)
    src, tgt = OLMO, QWEN3B
    print(f"  source: theta={src['theta']:.0f} W={src['window']} K={K}")
    print(f"  target: theta={tgt['theta']:.0f} W={tgt['window']} K={K}")
    a = K / math.log(tgt["theta"]) * math.log(tgt["window"] / src["window"])
    b = (math.log(src["theta"]) / K) * K / math.log(tgt["theta"])
    print(f"\n  k' = {a:.10f} + {b:.10f} * k")
    print(f"    plan claims  k' = 9.632959861 + 0.949828334*k")
    print(f"    delta_a = {a-9.632959861:+.3e}   delta_b = {b-0.949828334:+.3e}")

    src_tab = m_incr_beta(1.0, 21, 11)      # OLMo winner turns_a1_b64
    mu_s, var_s = moments(src_tab)
    print(f"\n  source table = OLMo turns_a1_b64: S={S_of(src_tab):.6f} "
          f"mu_s={mu_s:.6f} var_s={var_s:.6f}")
    print(f"    predicted mu_Q = a + b*mu_s = {a + b*mu_s:.10f}")
    print(f"    plan claims    mu_Q         = 30.5291832105")
    print(f"    delta = {a + b*mu_s - 30.5291832105:+.3e}")
    print(f"    plan claims    S_Q          = 33.4708167895")
    print(f"    64 - mu_Q                   = {64 - (a + b*mu_s):.10f}")
    print(f"    identity check: plan's S_Q == 64 - plan's mu_Q ? "
          f"{abs((64-30.5291832105) - 33.4708167895):.3e}")

    m_t, info = transport_linear(src_tab, src, tgt)
    mu_t, var_t = moments(m_t)
    print(f"\n  TRANSPORTED TABLE (fractional mass split by centroid):")
    print(f"    S_Q   = {S_of(m_t):.10f}")
    print(f"    mu_Q  = {mu_t:.10f}")
    print(f"    var_Q = {var_t:.10f}")
    print(f"    m_63 = {m_t[-1]:.10f}  (must be 1 for S = 64 - mu)")
    print(f"    monotone non-decreasing: {bool(np.all(np.diff(m_t) >= -1e-15))}")
    print(f"    m in [0,1]: {bool(m_t.min() >= -1e-15 and m_t.max() <= 1+1e-15)}")
    print(f"    k' range for the source support: "
          f"[{info['kp'][11]:.4f}, {info['kp'][31]:.4f}]  (K=64)")
    print(f"    transported mass total = {info['eps_new'].sum():.12f}")
    print(f"    slots touched = {int((info['eps_new']>0).sum())}")
    nu_t = np.exp(-(np.arange(K) * math.log(tgt["theta"]) / K + m_t * LN4))
    print(f"    nu strictly decreasing: {bool(np.all(np.diff(nu_t) < 0))}")

    print(f"\n  FULL TRANSPORTED 64-VALUE TABLE (m, nu, eps)")
    e_t = eps_of(m_t)
    for j in range(K):
        mark = "  <-mass" if e_t[j] > 0 else ""
        print(f"    j={j:3d}  m={m_t[j]:12.8f}  nu={nu_t[j]:14.6e}  "
              f"eps={e_t[j]:12.8f}{mark}")


if __name__ == "__main__":
    main()
