#!/usr/bin/env python3
"""Every table the Pro model's RESEARCH_PLAN.md asks for, built and self-checked.

Pure numpy, no GPU, no model.  Run from the repo root:

    python ds_workspace/recon_20260910/code/pro_tables_20260911.py

It prints each table's defining numbers and asserts the properties the plan
claims, so a mismatch is loud rather than silent.  Import `build_all()` to get
them as arrays.

THE THREE TABLES

  step42      §9.3  the budget-matched step: m_j = 1[j >= 22].  The plan's point
                    is that this is the UNIQUE step with S = 42, i.e. the same
                    budget as all four plateau members -- which no previous step
                    arm was (step_hi25 had S = 39).  It is a one-point
                    diagnostic of the step family at the winners' budget, not a
                    restart of the stopped step scan.

  condEVQ     §8    conditional EVQ.  Fix the winner's two ends and its band
                    [11,32]; inside the band maximise the EVQ functional subject
                    to unit mass AND mean 1/2.  The KKT gives the two-point
                    hyperbolic density
                        rho_tau(y) = tau cosh[tau(y-1/2)] / (2 sinh(tau/2))
                    with inverse CDF
                        phi_tau(u) = 1/2 + asinh[(2u-1) sinh(tau/2)] / tau
                    and the scale tau is pinned by ONE geometric boundary
                    condition -- the continuous edge slope matches the native
                    log-gap:
                        2 tanh(tau/2) / tau = n*d0/R ,  d0 = ln(theta)/64,
                        n = 21, R = n*d0 + ln 4 .
                    No free parameter, and nothing is fitted to a score.  The
                    plan is explicit that this is NOT claimed to be
                    performance-optimal; it is meant as a zero-budget residual
                    DIRECTION inside an already-validated table.

  transport   §7    the log-turn transport of a source table onto a target
                    checkpoint.  Move each increment's mass to the slot with the
                    same native turn count:
                        k' = K_t/ln(theta_t) * [ln(W_t/W_s) + (ln(theta_s)/K_s) k]
                    splitting fractional mass by linear interpolation between
                    adjacent slots.  This is the plan's answer to "cross-model
                    means transport the whole threshold distribution, not just S".

WHAT IS DELIBERATELY *NOT* HERE.  The plan also proposes a projected-Fisher
measurement in a small number of explicit directions (budget, same-budget shape,
gain, EVQ residual) instead of the 64 per-slot diagonals.  That needs forwards,
so it lives in `pro_projected_fisher.py`, not here.
"""
from __future__ import annotations

import math

import numpy as np

K = 64
OLMO = dict(theta=5.0e5, window=4096, head_dim=128)
QWEN = dict(theta=1.0e6, window=32768, head_dim=128)
LN4 = math.log(4.0)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def native_x(theta, k=K):
    """log-frequency of the native grid: nu_j = exp(-x_j), x_j = (j/K) ln theta."""
    return (np.arange(k, dtype=float) / k) * math.log(theta)


def m_from_x(x, theta, k=K):
    """m-coordinate: nu_j = omega_j * 4^{-m_j}  =>  m_j = (x_j - x_j^native)/ln4."""
    return (np.asarray(x, float) - native_x(theta, k)) / LN4


def nu_from_x(x):
    return np.exp(-np.asarray(x, float))


def band_of(m, tol=1e-12):
    e = np.diff(np.concatenate([[0.0], np.asarray(m, float)]))
    idx = np.flatnonzero(np.abs(e) > tol)
    return int(idx.min()), int(idx.max())


def report(name, m, theta, claims):
    m = np.asarray(m, float)
    nu = nu_from_x(native_x(theta) + LN4 * m)
    lo, hi = band_of(m)
    print(f"\n=== {name} ===")
    print(f"  S = Σm           = {m.sum():.6f}")
    print(f"  band             = [{lo},{hi}]   m_63 = {m[63]:.6f}   m_0 = {m[0]:.6f}")
    print(f"  nu strictly dec. = {bool(np.all(np.diff(nu) < 0))}")
    print(f"  span (log)       = {float(np.log(nu[0] / nu[-1])):.6f}")
    ok = True
    for label, got, want, tol in claims:
        good = abs(got - want) <= tol
        ok &= good
        print(f"  CLAIM {label:<34} got {got:+.8f}  want {want:+.8f}  "
              f"{'OK' if good else '**MISMATCH**'}")
    return ok


# --------------------------------------------------------------------------
# §9.3  the budget-matched step
# --------------------------------------------------------------------------
def step42(k=K, lo=14):
    """m_j = 1[j >= 22].  The unique step with S = 42."""
    m = np.zeros(k)
    m[22:] = 1.0
    return m


# --------------------------------------------------------------------------
# §8  conditional EVQ
# --------------------------------------------------------------------------
def solve_tau(n, d0, R, lo=1e-9, hi=50.0, iters=200):
    """Solve 2 tanh(tau/2)/tau = n*d0/R for tau > 0 (LHS decreases in tau)."""
    target = n * d0 / R

    def f(t):
        if t < 1e-9:
            return 1.0 - target
        return 2.0 * math.tanh(t / 2.0) / t - target

    a, b = lo, hi
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"no sign change bracketing tau: f({a})={fa}, f({b})={fb}")
    for _ in range(iters):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def phi_tau(u, tau):
    """Inverse CDF of rho_tau."""
    u = np.asarray(u, float)
    return 0.5 + np.arcsinh((2.0 * u - 1.0) * math.sinh(tau / 2.0)) / tau


def cond_evq(cfg=OLMO, lo=11, n=21, k=K):
    """The plan's conditional-EVQ reference on `cfg`, inside band [lo, lo+n]."""
    d0 = math.log(cfg["theta"]) / k
    R = n * d0 + LN4
    tau = solve_tau(n, d0, R)
    q = np.arange(n + 1, dtype=float) / n
    x_lo = native_x(cfg["theta"], k)[lo]
    x = native_x(cfg["theta"], k).copy()
    x[lo:lo + n + 1] = x_lo + R * phi_tau(q, tau)
    m = m_from_x(x, cfg["theta"], k)
    # BEYOND the band the winner holds m = 1 (a1_b64 = m_incr_beta(1, n=21,
    # low=11) has m = 1 from slot 33).  The plan's S = 42 is the band interior
    # PLUS that plateau; leaving it at zero gives S = 11, which is a different
    # table.  Set m, not x: a constant x would NOT be m = 1, because the native
    # x increases with the slot.
    m[lo + n + 1:] = 1.0
    x = native_x(cfg["theta"], k) + LN4 * m
    return dict(tau=tau, d0=d0, R=R, x=x, m=m, lo=lo, n=n)


# --------------------------------------------------------------------------
# §7  log-turn transport
# --------------------------------------------------------------------------
def transport(m_src, cfg_src=OLMO, cfg_tgt=QWEN, k=K):
    """Move increment mass to the target slot with the same native turn count."""
    m_src = np.asarray(m_src, float)
    eps = np.diff(np.concatenate([[0.0], m_src]))          # eps[i] = m_i - m_{i-1}
    Ks, Kt = k, k
    th_s, th_t = cfg_src["theta"], cfg_tgt["theta"]
    W_s, W_t = cfg_src["window"], cfg_tgt["window"]
    # k' = (K_t/ln th_t) * [ ln(W_t/W_s) + (ln th_s/K_s) * i ]
    a = Kt / math.log(th_t)                                 # overall scale
    C = math.log(W_t / W_s)                                 # constant part
    D = math.log(th_s) / Ks                                 # per-slot part
    out = np.zeros(k)
    for i, mass in enumerate(eps):
        if mass == 0.0:
            continue
        kp = a * (C + D * i)
        if kp < 0 or kp > k - 1:
            raise ValueError(f"transport of slot {i} lands at {kp:.4f}, outside "
                             f"[0,{k-1}] -- the physical mapping fails on this "
                             f"target; the plan says report it, do not clip")
        j = int(math.floor(kp))
        frac = kp - j
        if j + 1 <= k - 1:
            out[j] += mass * (1.0 - frac)
            out[j + 1] += mass * frac
        else:
            out[j] += mass
    return np.cumsum(out), dict(a=a, C=C, D=D, ac=a * C, aD=a * D)


# --------------------------------------------------------------------------
def build_all():
    """Return every table, plus the verification verdict."""
    out = {}

    out["step42"] = step42()

    ce = cond_evq()
    out["condEVQ"] = ce["m"]

    # source = the RULER-confirmed winner on OLMo: a1_b64 = m_incr_beta(1, n=21, low=11)
    try:
        import sys
        som = "/root/autodl-tmp/phase1_20260910"
        if som not in sys.path:
            sys.path.insert(0, som)
        from experiments.curvature_20260910.tables import m_incr_beta
        src = np.asarray(m_incr_beta(1.0, n=21, low=11), float)
    except Exception:
        # standalone fallback: eps ~ k(n+1-k) on [12,32], n=21, m=1 from 33
        q = np.arange(1, 22, dtype=float)
        w = q * (22 - q)
        e = w / w.sum()
        src = np.zeros(K)
        src[12:33] = np.cumsum(e)
        src[33:] = 1.0
    tr, meta = transport(src)
    out["transport_a1b64"] = tr
    out["_transport_meta"] = meta

    ok = True
    ok &= report("step42  (Pro §9.3)", out["step42"], OLMO["theta"], [
        ("S == 42 (winner budget)", out["step42"].sum(), 42.0, 1e-9),
    ])
    ok &= report("condEVQ (Pro §8)", out["condEVQ"], OLMO["theta"], [
        ("S == 42", out["condEVQ"].sum(), 42.0, 1e-6),
        ("tau", ce["tau"], 2.0301373113, 1e-8),
    ])
    ok &= report("transport a1_b64 -> Qwen (Pro §7)", out["transport_a1b64"],
                 QWEN["theta"], [
        ("S == 33.4708167895", out["transport_a1b64"].sum(), 33.4708167895, 1e-6),
    ])
    out["_all_ok"] = ok
    return out


if __name__ == "__main__":
    r = build_all()
    print("\n--- Pro §7 transport constants (claimed a=9.632959861, b=0.949828334) ---")
    mm = r["_transport_meta"]
    print(f"  a*C = {mm['ac']:.9f}  (claim 9.632959861)")
    print(f"  a*D = {mm['aD']:.9f}  (claim 0.949828334)")
    print(f"\nALL CLAIMS: {'OK' if r['_all_ok'] else 'SOME MISMATCHED'}")
    for name in ("step42", "condEVQ", "transport_a1b64"):
        print(f"\n{name} = " + np.array2string(r[name], precision=6, max_line_width=110))
