#!/usr/bin/env python3
"""theory_checks.py — EXPERIMENT_PLAN.md §6 math package (reconstructed).

The plan's companion files (theory_checks.py / math_checks.json) were produced
in another session and never landed in the repo; this rebuilds them from the
plan text itself.  Pure numpy, CPU only.

    Status: MATHEMATICAL CHECKS ONLY — no model scores are produced or implied.

What is verified (check ids appear in math_checks.json):

  T1  EVQ-Cosh spectral geometry: phi_tau(u) = 1 - asinh((1-u) sinh tau)/tau,
      endpoints, monotonicity, normalized log-gap phi_tau'(u) with limits
      tanh(tau)/tau (fast end, u=0) and sinh(tau)/tau (slow end, u=1);
      tau=2 -> 0.482 / 1.813 as the plan states; cosh-density identity.
  T2  condEVQ symmetric inverse-CDF (Pro §8 convention) and its table
      cross-check via pro_tables_20260911 (tau=2.0301373113, S=42).
  T3  Finite-window complex-exponential correlation:
      |G_ij| = |sin(L d/2) / (L sin(d/2))| vs direct summation.
  T4  The REAL sin/cos design Gram contains nu_i+nu_j terms: the
      difference-frequency-only formula is quantifiably wrong at low
      frequencies ("low end is not a free content channel").
  T5  Ridge linear-readout closed form
      E(f) = f'f - (Phi'f)' (Phi'Phi + eta I)^-1 (Phi'f) vs direct argmin.
  T6  Gradient flow r(t) = exp(-Phi Phi' t) f; spectral form
      ||r(t)||^2 = sum_a exp(-2 lam_a t) |u_a'f|^2 vs Euler integration;
      zero-eigenvalue components stay as unrepresentable residual.
  T7  Fixed-offset head margin: attention(correct) >= 1/[1+(N-1)e^{-beta Gamma}]
      with Gamma = min_{d != d*} [k(d*) - k(d)].
  T8  YaRN indexed ramp vs MrPro quadratic ramp:
      rotation identity R(s d w/s) = R(d w); shared endpoints; front
      perturbation O(N^-1) (YaRN) vs O(N^-2) (MrPro); mid-band saturation
      (YaRN) vs power-law in ln s (MrPro); turn count W*nu/2pi is a
      coordinate, not a joint-coverage certificate.
  T9  Power control exponent for P2: solve sum_j (j/(K-1))^p =
      sum_j phi_tau(j/(K-1)) with tau=2, K=64 -> p ~= 1.626111 (plan value);
      control table emitted to math_checks.json.
  T10 Target-free log-turn transport (P5 rule), re-implemented from the plan
      text: OLMo a1_b64 -> Qwen gives S_Q = 33.4708167895; array cross-checked
      against pro_tables_20260911.transport; out-of-range mapping must raise,
      never silently clip.

Run:
    python experiments/twotrack_20260911/theory_checks.py
Writes math_checks.json next to this file.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ds_workspace", "recon_20260910", "code"))

RESULTS: dict = {}


def record(cid: str, ok: bool, **kw):
    RESULTS[cid] = {"ok": bool(ok), **kw}
    print(f"[{'PASS' if ok else '**FAIL**'}] {cid}: " +
          ", ".join(f"{k}={v}" for k, v in kw.items()
                    if not isinstance(v, (list, np.ndarray))))
    return bool(ok)


# ---------------------------------------------------------------- T1
def phi_plan(u, tau):
    """EVQ-Cosh inverse CDF as written in the plan §1.1 (phi(0)=0 fast end,
    phi(1)=1 slow end; nu = exp(-A phi))."""
    u = np.asarray(u, float)
    return 1.0 - np.arcsinh((1.0 - u) * math.sinh(tau)) / tau


def t1_spectral_geometry():
    taus = [0.5, 1.0, 2.0, 4.0]
    u = np.linspace(0.0, 1.0, 20001)
    ok = True
    rows = {}
    for tau in taus:
        p = phi_plan(u, tau)
        end0, end1 = float(p[0]), float(p[-1])
        mono = bool(np.all(np.diff(p) > 0))
        # analytic derivative: phi'(u) = sinh(tau) / (tau * sqrt(1+((1-u) sinh tau)^2))
        sh = math.sinh(tau)
        dphi = sh / (tau * np.sqrt(1.0 + ((1.0 - u) * sh) ** 2))
        num = np.gradient(p, u)
        dev = float(np.max(np.abs(num[10:-10] - dphi[10:-10])))
        # central-difference truncation error is |f'''|h^2/6 with
        # |f'''| = sinh^3(tau)/tau * |1-2w^2|/(1+w^2)^{5/2} <= sinh^3/tau;
        # the numeric gradient cannot beat its own truncation bound, so the
        # tolerance is that bound (x1.5), not an absolute guess.
        h = float(u[1] - u[0])
        trunc = (sh ** 3 / tau) * h * h / 6.0
        gap_fast = float(dphi[0])       # u=0 -> tanh(tau)/tau
        gap_slow = float(dphi[-1])      # u=1 -> sinh(tau)/tau
        cf, cs = math.tanh(tau) / tau, sh / tau
        # cosh-density identity: dphi/du = sinh(tau) / (tau * cosh(tau*(1-phi)))
        ident = float(np.max(np.abs(dphi - sh / (tau * np.cosh(tau * (1.0 - p))))))
        rows[f"tau={tau}"] = dict(gap_fast=gap_fast, gap_slow=gap_slow,
                                  tanh_over_tau=cf, sinh_over_tau=cs,
                                  numeric_grad_dev=dev, truncation_bound=trunc)
        ok &= abs(end0) < 1e-14 and abs(end1 - 1.0) < 1e-14 and mono
        ok &= abs(gap_fast - cf) < 1e-12 and abs(gap_slow - cs) < 1e-12
        ok &= dev < max(1e-9, 1.5 * trunc) and ident < 1e-12
    # the plan quotes tau=2 -> "about 0.482 and 1.813"
    ok &= abs(rows["tau=2.0"]["gap_fast"] - 0.482) < 5e-4
    ok &= abs(rows["tau=2.0"]["gap_slow"] - 1.813) < 5e-4
    record("T1_evq_cosh_geometry", ok, tau2_gap_fast=rows["tau=2.0"]["gap_fast"],
           tau2_gap_slow=rows["tau=2.0"]["gap_slow"], per_tau=rows)


# ---------------------------------------------------------------- T2
def t2_condevq_symmetric():
    import pro_tables_20260911 as pt  # type: ignore  # runtime sys.path import
    tau = pt.solve_tau(21, math.log(5.0e5) / 64, 21 * (math.log(5.0e5) / 64) + math.log(4.0))
    u = np.linspace(0, 1, 10001)
    ph = pt.phi_tau(u, tau)
    sym = float(np.max(np.abs(pt.phi_tau(1.0 - u, tau) - (1.0 - ph))))
    ok = abs(tau - 2.0301373113) < 1e-8
    ok &= abs(float(ph[0])) < 1e-14 and abs(float(ph[-1]) - 1) < 1e-14
    ok &= abs(float(pt.phi_tau(np.array([0.5]), tau)[0]) - 0.5) < 1e-14
    ok &= sym < 1e-14
    ce = pt.cond_evq()
    m = ce["m"]
    ok &= abs(float(m.sum()) - 42.0) < 1e-6
    ok &= bool(np.all(m[33:] == 1.0))          # winner plateau beyond the band
    ok &= bool(np.all(np.diff(m) >= -1e-15))   # monotone m
    nu = pt.nu_from_x(pt.native_x(pt.OLMO["theta"]) + math.log(4.0) * m)
    ok &= bool(np.all(np.diff(nu) < 0))        # strictly decreasing frequencies
    record("T2_condevq_symmetric_phi", ok, tau=tau, symmetry_dev=sym, S=float(m.sum()))


# ---------------------------------------------------------------- T3
def dirichlet(d, L):
    return np.sin(L * d / 2.0) / (L * np.sin(d / 2.0))


def t3_complex_gram():
    rng = np.random.default_rng(0)
    ok = True
    worst = 0.0
    for L in (64, 512, 4096):
        n = np.arange(L, dtype=float)
        for _ in range(200):
            vi, vj = rng.uniform(1e-3, math.pi, 2)
            d = vi - vj
            direct = abs(np.mean(np.exp(1j * d * n)))
            closed = abs(dirichlet(d, L))
            worst = max(worst, abs(direct - closed))
            ok &= abs(direct - closed) < 1e-12
    record("T3_dirichlet_kernel", ok, max_abs_dev=worst)


# ---------------------------------------------------------------- T4
def t4_real_gram_sum_terms():
    """Plan §1.1: the REAL sin/cos design Gram contains nu_i+nu_j (sum-frequency)
    terms; use the full matrix, not the complex difference-only formula.

    Verified by direct summation over the window (no closed form assumed):
      CC'/L = (D_diff + D_sum)/2,  SS'/L = (D_diff - D_sum)/2,
      CS'/L = (S_sum - S_diff)/2,  CC'/L + SS'/L = D_diff exactly
    (the last is why the complex formula ever looked sufficient), and the
    complex-exponential Gram E^H E/L is exactly D_diff - i*S_diff.
    Then quantified: at turns<~1 frequencies the sum/cross terms are O(0.1-0.6)
    -- the low end is NOT a free content channel; at hundreds of turns they
    average out to ~1e-4."""
    L = 4096
    n = np.arange(L, dtype=float)
    out = {}
    ok = True
    for tag, ks in (("low", [0.3, 0.8, 1.4]), ("high", [301.1, 452.2, 604.4])):
        v = 2.0 * math.pi * np.array(ks) / L
        C = np.cos(np.outer(n, v))              # L x K
        S = np.sin(np.outer(n, v))
        CC = C.T @ C / L
        SS = S.T @ S / L
        CS = C.T @ S / L
        d = v[:, None] - v[None, :]
        s = v[:, None] + v[None, :]
        Dd = np.mean(np.cos(d[..., None] * n), axis=-1)   # difference, cos
        Ds = np.mean(np.cos(s[..., None] * n), axis=-1)   # sum, cos
        Sd = np.mean(np.sin(d[..., None] * n), axis=-1)   # difference, sin
        Ss = np.mean(np.sin(s[..., None] * n), axis=-1)   # sum, sin
        # exact block identities (pure trig, checked against the real Gram)
        id_cc = float(np.max(np.abs(CC - 0.5 * (Dd + Ds))))
        id_ss = float(np.max(np.abs(SS - 0.5 * (Dd - Ds))))
        id_cs = float(np.max(np.abs(CS - 0.5 * (Ss - Sd))))
        canc = float(np.max(np.abs(CC + SS - Dd)))        # sum terms cancel
        E = C + 1j * S                                    # complex design
        Gc = E.conj().T @ E / L
        id_cx = float(np.max(np.abs(Gc - (Dd - 1j * Sd))))
        ok &= max(id_cc, id_ss, id_cs, canc, id_cx) < 1e-12
        m_ds = float(np.max(np.abs(Ds)))                  # sum-frequency term
        m_cs = float(np.max(np.abs(CS)))                  # cross blocks
        out[tag] = dict(max_sum_freq_cos=m_ds, max_cross_block=m_cs,
                        identity_dev=max(id_cc, id_ss, id_cs, canc, id_cx))
        if tag == "low":
            ok &= m_ds > 0.05 and m_cs > 0.05   # both large: full matrix needed
        else:
            ok &= m_ds < 1e-3 and m_cs < 1e-3   # die out: diff-only asymptotically ok
    record("T4_sum_frequency_terms_matter", ok, **out)


# ---------------------------------------------------------------- T5
def t5_ridge_readout():
    rng = np.random.default_rng(2)
    ok = True
    worst = 0.0
    for _ in range(20):
        m_ = int(rng.integers(3, 12))
        d = int(rng.integers(2, 8))
        Phi = rng.normal(size=(m_, d))
        f = rng.normal(size=m_)
        for eta in (1e-3, 0.1, 1.0, 10.0):
            A = Phi.T @ Phi + eta * np.eye(d)
            b = Phi.T @ f
            c = np.linalg.solve(A, b)
            direct = float((Phi @ c - f) @ (Phi @ c - f) + eta * c @ c)
            closed = float(f @ f - b @ np.linalg.solve(A, b))
            worst = max(worst, abs(direct - closed))
            ok &= abs(direct - closed) < 1e-9
    record("T5_ridge_closed_form", ok, max_abs_dev=worst)


# ---------------------------------------------------------------- T6
def t6_gradient_flow():
    rng = np.random.default_rng(3)
    ok = True
    m_, d = 9, 4                     # rank-deficient: null space exists
    Phi = rng.normal(size=(m_, d))
    f = rng.normal(size=m_)
    G = Phi @ Phi.T
    lam, U = np.linalg.eigh(G)
    lam = np.clip(lam, 0.0, None)
    coeff = U.T @ f
    worst_spec = 0.0
    for t in (0.05, 0.2, 1.0, 5.0):
        r_spec = U @ (np.exp(-lam * t) * coeff)
        # Euler integration dr/dt = -G r
        steps = max(2000, int(4000 * t))
        dt = t / steps
        r = f.copy()
        for _ in range(steps):
            r = r - dt * (G @ r)
        worst_spec = max(worst_spec, float(np.max(np.abs(r_spec - r))))
        ok &= worst_spec < 2e-3
        sq = float(np.sum(np.exp(-2.0 * lam * t) * coeff ** 2))
        ok &= abs(sq - float(r_spec @ r_spec)) < 1e-10
    # t -> infinity: residual = component in the null space of Phi' (lam ~ 0)
    null_mask = lam < 1e-10
    r_inf = U[:, null_mask] @ coeff[null_mask] if null_mask.any() else np.zeros(m_)
    ok &= bool(null_mask.any())      # construction must actually be rank-deficient
    # the null component is exactly the unlearnable part
    ok &= float(np.max(np.abs(G @ r_inf))) < 1e-12
    record("T6_gradient_flow_spectral", ok, max_euler_dev=worst_spec,
           n_zero_eigs=int(null_mask.sum()),
           residual_norm=float(np.linalg.norm(r_inf)))


# ---------------------------------------------------------------- T7
def t7_gamma_margin():
    rng = np.random.default_rng(4)
    ok = True
    worst_slack = 0.0
    for _ in range(50):
        Kn = rng.integers(3, 8)
        nu = rng.uniform(0.05, math.pi, Kn)
        a = rng.uniform(0.0, 1.0, Kn)          # a_j >= 0
        dstar = float(rng.integers(0, 50))
        grid = np.arange(-60, 61, dtype=float) + dstar
        k = lambda dd: float(np.sum(a * np.cos(nu * (dd - dstar))))
        kd = k(dstar)
        others = np.array([k(x) for x in grid if x != dstar])
        Gamma = float(np.min(kd - others))
        if Gamma <= 0:
            continue                            # bound is vacuous but still true
        N = int(rng.integers(2, 12))
        distr = rng.choice(others, size=N - 1, replace=False)
        for beta in (0.5, 2.0, 8.0):
            scores = beta * np.concatenate([[kd], distr])
            p = np.exp(scores - scores.max())
            p_correct = float(p[0] / p.sum())
            bound = 1.0 / (1.0 + (N - 1) * math.exp(-beta * Gamma))
            worst_slack = max(worst_slack, bound - p_correct)
            ok &= p_correct >= bound - 1e-12
    record("T7_gamma_margin_bound", ok, max_bound_minus_actual=worst_slack)


# ---------------------------------------------------------------- T8
def rot(ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([[c, -s], [s, c]])


def yarn_nu(omega, q, N, s):
    return omega * (1.0 - q / N + (q / N) / s)


def mrpro_nu(omega, q, N, s):
    return omega * s ** (-(q * (q + 1)) / (N * (N + 1)))


def t8_yarn_vs_mrpro():
    rng = np.random.default_rng(5)
    ok = True
    # (a) rotation identity R(s d * w/s) = R(d w)
    worst = 0.0
    for _ in range(50):
        d, w, s = rng.uniform(1, 500), rng.uniform(1e-3, 1.0), rng.uniform(1.5, 8.0)
        worst = max(worst, float(np.max(np.abs(rot((s * d) * (w / s)) - rot(d * w)))))
    ok &= worst < 1e-10
    # (b) shared endpoints
    N, s = 64, 4.0
    ep = max(abs(yarn_nu(1.0, 0, N, s) - 1.0), abs(mrpro_nu(1.0, 0, N, s) - 1.0),
             abs(yarn_nu(1.0, N, N, s) - 1.0 / s), abs(mrpro_nu(1.0, N, N, s) - 1.0 / s))
    ok &= ep < 1e-14
    # (c) front-end order: fixed q, sweep N -> YaRN O(N^-1), MrPro O(N^-2)
    q, s = 1, 4.0
    Ns = np.array([32, 64, 128, 256, 512, 1024], float)
    dy = np.array([abs(math.log(yarn_nu(1.0, q, n_, s))) for n_ in Ns])
    dm = np.array([abs(math.log(mrpro_nu(1.0, q, n_, s))) for n_ in Ns])
    slope_y = float(np.polyfit(np.log(Ns), np.log(dy), 1)[0])
    slope_m = float(np.polyfit(np.log(Ns), np.log(dm), 1)[0])
    ok &= abs(slope_y + 1.0) < 0.05 and abs(slope_m + 2.0) < 0.05
    # (d) mid-band with s: YaRN's log-shift saturates at ln(1/(1-q/N)), while
    # MrPro's log-shift stays a power law in s: |ln nu| = [q(q+1)/(N(N+1))] ln s
    # -- LINEAR in ln s with the declared exponent (fit slope of |ln nu| vs ln s)
    q, N = 32, 64
    ss = 2.0 ** np.arange(1, 17)
    dy2 = np.array([abs(math.log(yarn_nu(1.0, q, N, x))) for x in ss])
    dm2 = np.array([abs(math.log(mrpro_nu(1.0, q, N, x))) for x in ss])
    yarn_sat = float(abs(dy2[-1] - math.log(2.0)))          # -> ln 2 bound
    mr_slope = float(np.polyfit(np.log(ss), dm2, 1)[0])     # linear in ln s
    mr_pred = (q * (q + 1)) / (N * (N + 1))                  # 32*33/(64*65)
    ok &= yarn_sat < 1e-3 and abs(mr_slope - mr_pred) < 1e-9
    ok &= float(np.max(dy2)) < math.log(2.0) + 1e-12         # strictly bounded
    # (e) turn count is a coordinate, not a joint-coverage certificate
    W = 8
    v1, v2 = 2 * math.pi / W, 2 * math.pi * 2 / W            # v2 = 2 v1
    n = np.arange(0, 4 * W)
    pairs = {(round((x * v1) % (2 * math.pi), 9), round((x * v2) % (2 * math.pi), 9))
             for x in n}
    turns_v1 = float(4 * W * v1 / (2 * math.pi))              # 4 full turns
    joint_cov = len(pairs)                                    # 8, not 4*W
    ok &= turns_v1 == 4.0 and joint_cov == W and joint_cov < len(n)
    record("T8_yarn_mrpro_operators", ok, front_slope_yarn=slope_y,
           front_slope_mrpro=slope_m, yarn_midband_bound_dev=yarn_sat,
           mrpro_midband_slope=mr_slope, mrpro_slope_predicted=mr_pred,
           joint_phase_distinct=joint_cov, positions_walked=len(n))


# ---------------------------------------------------------------- T9
def t9_power_control():
    K, tau = 64, 2.0
    u = np.arange(K, dtype=float) / (K - 1)
    target = float(np.sum(phi_plan(u, tau)))

    def g(p):
        return float(np.sum(u ** p)) - target

    lo, hi = 0.2, 8.0
    assert g(lo) > 0 > g(hi) or g(lo) < 0 < g(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    ok = abs(p - 1.626111) < 2e-6
    ctl = u ** p
    ok &= abs(float(ctl[0]) - phi_plan(np.array([0.0]), tau)[0]) < 1e-14   # both 0
    ok &= abs(float(ctl[-1]) - phi_plan(np.array([1.0]), tau)[0]) < 1e-14  # both 1
    ok &= abs(float(np.sum(ctl)) - target) < 1e-9                          # mean matched
    max_shape_dev = float(np.max(np.abs(ctl - phi_plan(u, tau))))
    record("T9_power_control_exponent", ok, p=p, plan_value=1.626111,
           mean_matched_dev=abs(float(np.sum(ctl)) - target),
           max_shape_dev_vs_cosh=max_shape_dev)
    RESULTS["T9_power_control_exponent"]["control_table_tau2_K64"] = \
        [round(float(x), 12) for x in ctl]
    return p


# ---------------------------------------------------------------- T10
def build_a1_b64_source(K=64):
    """a1_b64 = m_incr_beta(1.0, n=21, low=11): eps ~ k(n+1-k) on the band,
    m = 1 plateau from slot 33.  Same standalone construction as
    pro_tables_20260911's fallback (bit-identical convention)."""
    q = np.arange(1, 22, dtype=float)
    w = q * (22 - q)
    e = w / w.sum()
    src = np.zeros(K)
    src[12:33] = np.cumsum(e)
    src[33:] = 1.0
    return src


def transport_turns(m_src, theta_s, W_s, theta_t, W_t, K=64):
    """Independent implementation of the P5 rule straight from the plan text:
    put each increment's mass at the target slot with the SAME native turn
    count W*omega/2pi, i.e. solve W_s*omega_s(i) = W_t*omega_t(k') on the
    native grids omega(j) = theta^(-j/K) -> k' = (K/ln theta_t) [ln(W_t/W_s)
    + (ln theta_s / K) i]; split fractional mass over adjacent slots;
    accumulate -> m.  Out-of-range must RAISE (plan: report, never clip)."""
    eps = np.diff(np.concatenate([[0.0], np.asarray(m_src, float)]))
    a = K / math.log(theta_t)
    C = math.log(W_t / W_s)
    D = math.log(theta_s) / K
    out = np.zeros(K)
    for i, mass in enumerate(eps):
        if mass == 0.0:
            continue
        kp = a * (C + D * i)
        if kp < 0 or kp > K - 1:
            raise ValueError(f"transport maps slot {i} to {kp:.4f}, outside "
                             f"[0,{K-1}]: report, do not clip")
        j = int(math.floor(kp))
        frac = kp - j
        out[j] += mass * (1.0 - frac)
        if j + 1 <= K - 1:
            out[j + 1] += mass * frac
    return np.cumsum(out), dict(aC=a * C, aD=a * D)


def t10_qwen_transport():
    TH_S, W_S = 5.0e5, 4096          # OLMo-2
    TH_T, W_T = 1.0e6, 32768         # Qwen2.5
    src = build_a1_b64_source()
    mine, meta = transport_turns(src, TH_S, W_S, TH_T, W_T)
    S_Q = float(mine.sum())
    ok = abs(S_Q - 33.4708167895) < 1e-7
    # transport constants the Pro plan quotes
    ok &= abs(meta["aC"] - 9.632959861) < 1e-8
    ok &= abs(meta["aD"] - 0.949828334) < 1e-8
    # cross-check against the repo's pro_tables implementation
    try:
        import pro_tables_20260911 as pt  # type: ignore  # runtime sys.path import
        theirs = pt.transport(src, pt.OLMO, pt.QWEN)[0]
        ok &= float(np.max(np.abs(mine - theirs))) < 1e-12
        xcheck = "array identical to pro_tables_20260911.transport"
    except Exception as e:  # pragma: no cover
        xcheck = f"pro_tables import failed ({e}); S-only check"
    # out-of-range must raise, not clip: a target with a tiny theta window
    raised = False
    try:
        transport_turns(src, TH_S, W_S, 1.0e6, 4096 * 10 ** 9, K=64)
    except ValueError:
        raised = True
    ok &= raised
    # mass conservation: increments sum preserved
    ok &= abs(float(mine[-1]) - float(src[-1])) < 1e-12
    record("T10_qwen_transport_rule", ok, S_Q=S_Q, plan_value=33.4708167895,
           aC=meta["aC"], aD=meta["aD"], out_of_range_raises=raised,
           cross_check=xcheck)
    RESULTS["T10_qwen_transport_rule"]["qwen_transport_a1b64"] = \
        [round(float(x), 12) for x in mine]
    RESULTS["T10_qwen_transport_rule"]["a1_b64_source"] = \
        [round(float(x), 12) for x in src]
    return mine


def main():
    np.seterr(all="raise")
    t1_spectral_geometry()
    t2_condevq_symmetric()
    t3_complex_gram()
    t4_real_gram_sum_terms()
    t5_ridge_readout()
    t6_gradient_flow()
    t7_gamma_margin()
    t8_yarn_vs_mrpro()
    t9_power_control()
    t10_qwen_transport()

    all_ok = all(v["ok"] for v in RESULTS.values())
    payload = {
        "status": "MATH CHECKS ONLY - no new model scores",
        "generated": "2026-09-11",
        "source_plan": "EXPERIMENT_PLAN.md (EVQ / YaRN-MrRoPE two-track)",
        "all_ok": bool(all_ok),
        "checks": RESULTS,
    }
    out = os.path.join(HERE, "math_checks.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\n{'ALL CHECKS OK' if all_ok else 'SOME CHECKS FAILED'} -> {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
