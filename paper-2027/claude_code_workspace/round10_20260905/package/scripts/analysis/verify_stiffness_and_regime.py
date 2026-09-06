#!/usr/bin/env python3
"""
Softmax Transport: f-Divergence Stiffness Sweep & LoRA Phase Transition
========================================================================

Self-contained verification for NeurIPS 2026 appendix.
Reproduces two key results:

  1. The f-divergence exponent p ∈ [0.75, 0.80] gives L-exponent = -0.500
     for τ*, with χ² (p=1) as the first-principles choice (gap 0.035).

  2. LoRA with rank r < K = d_head/2 on a pretrained model creates a
     phase transition where τ* → 0, explaining the PPL 77.1 catastrophe.

Dependencies: numpy only.  Runtime: ~60s on M4 Max.
"""
import numpy as np
import time

# ── numpy compat ──
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ══════════════════════════════════════════════════════════════════════
# Core physics
# ══════════════════════════════════════════════════════════════════════

N_PHI = 3000
PHI = np.linspace(1e-5, 1 - 1e-5, N_PHI)

def rho_tau(phi: np.ndarray, tau: float) -> np.ndarray:
    """Normalized cosh density: ρ_τ(φ) = τ cosh(τ(1-φ))/sinh(τ)."""
    if abs(tau) < 1e-12:
        return np.ones_like(phi)
    return tau * np.cosh(tau * (1.0 - phi)) / np.sinh(tau)

def q_channel(x: np.ndarray) -> np.ndarray:
    """Single-channel softmax transport kernel q(x).
    q(x) = 1/2 + sin(2x)/(4x) - (sin(x)/x)²
    """
    out = np.zeros_like(x, dtype=float)
    m = np.abs(x) > 1e-10
    xm = x[m]
    sinx = np.sin(xm)
    out[m] = 0.5 + np.sin(2 * xm) / (4 * xm) - (sinx / xm) ** 2
    return out

# Pre-compute q values for each L to avoid redundant work.
_q_cache: dict[int, np.ndarray] = {}

def _q_at_L(L: int, base: float) -> np.ndarray:
    key = (L, base)
    if key not in _q_cache:
        _q_cache[key] = q_channel(L * base ** (-PHI))
    return _q_cache[key]

def utility(tau: float, L: int, M: int, base: float) -> float:
    """U(τ,L) = (M/L) ∫ q(Lb^{-φ}) ρ_τ(φ) dφ."""
    return (M / L) * _trapz(_q_at_L(L, base) * rho_tau(PHI, tau), PHI)


# ══════════════════════════════════════════════════════════════════════
# Stiffness functionals
# ══════════════════════════════════════════════════════════════════════

def S_L2(tau: float, M: int) -> float:
    """(1/2M) ∫(ρ-1)² dφ — closed form."""
    if abs(tau) < 1e-12:
        return 0.0
    st = np.sinh(tau)
    return (1 / (2 * M)) * (tau * (2 * tau + np.sinh(2 * tau)) / (4 * st ** 2) - 1)

def S_chi2(tau: float, M: int) -> float:
    """(1/M) ∫(ρ-1)²/ρ dφ = (1/M)[sinh(τ)arctan(sinh τ)/τ² - 1] — closed form."""
    if abs(tau) < 1e-12:
        return 0.0
    return (1 / M) * (np.sinh(tau) * np.arctan(np.sinh(tau)) / tau ** 2 - 1)

def S_power(tau: float, p: float, M: int) -> float:
    """(1/M) ∫(ρ-1)²/ρ^p dφ — numerical for general p."""
    if abs(tau) < 1e-12:
        return 0.0
    r = rho_tau(PHI, tau)
    return (1 / M) * _trapz((r - 1) ** 2 / r ** p, PHI)

def S_KL(tau: float, M: int) -> float:
    """D_KL(ρ‖1)/M = (1/M) ∫ ρ ln ρ dφ — closed form."""
    if abs(tau) < 1e-12:
        return 0.0
    return (1 / M) * (
        np.log(tau / np.tanh(tau)) - 1 + np.arctan(np.sinh(tau)) / np.sinh(tau)
    )

def S_revKL(tau: float, M: int) -> float:
    """D_KL(1‖ρ)/M = -(1/M) ∫ ln ρ dφ — numerical."""
    if abs(tau) < 1e-12:
        return 0.0
    r = rho_tau(PHI, tau)
    return -(1 / M) * _trapz(np.log(r), PHI)

def S_Jeffreys(tau: float, M: int) -> float:
    """Jeffreys divergence / M = (1/M) ∫(ρ-1)ln ρ dφ."""
    if abs(tau) < 1e-12:
        return 0.0
    r = rho_tau(PHI, tau)
    return (1 / M) * _trapz((r - 1) * np.log(r), PHI)


# ══════════════════════════════════════════════════════════════════════
# Optimizer: find τ minimizing F(τ) = S(τ) - λ U(τ,L)
# ══════════════════════════════════════════════════════════════════════

def find_tau_opt(
    S_func,  # callable(tau) -> float
    lam: float,
    L: int,
    M: int,
    base: float,
    tau_lo: float = 0.005,
    tau_hi: float = 20.0,
    n_grid: int = 2000,
) -> float:
    """Grid search + golden-section refinement."""
    taus = np.linspace(tau_lo, tau_hi, n_grid)
    Fs = np.array([S_func(t) - lam * utility(t, L, M, base) for t in taus])
    idx = int(np.argmin(Fs))
    a = taus[max(0, idx - 3)]
    c = taus[min(n_grid - 1, idx + 3)]
    gr = (np.sqrt(5) + 1) / 2
    for _ in range(80):
        b1 = c - (c - a) / gr
        b2 = a + (c - a) / gr
        f1 = S_func(b1) - lam * utility(b1, L, M, base)
        f2 = S_func(b2) - lam * utility(b2, L, M, base)
        if f1 < f2:
            c = b2
        else:
            a = b1
        if c - a < 1e-11:
            break
    return (a + c) / 2


def calibrate_lambda(
    S_func,
    target_tau: float,
    L_cal: int,
    M: int,
    base: float,
) -> float:
    """Binary search for λ so that τ_opt(L_cal) = target_tau."""
    lo, hi = 1e-4, 1e5
    for _ in range(80):
        mid = (lo + hi) / 2
        t = find_tau_opt(S_func, mid, L_cal, M, base)
        if t < target_tau:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def L_exponent(
    S_func,
    M: int,
    base: float,
    d_head: int,
    Ls: list[int] | np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Calibrate at L=2048, sweep L, fit log-log slope."""
    if Ls is None:
        Ls = np.array([128, 256, 512, 1024, 2048, 4096, 8192])
    Ls = np.asarray(Ls)
    target = d_head / np.sqrt(2048)
    lam = calibrate_lambda(S_func, target, 2048, M, base)
    taus = np.array([find_tau_opt(S_func, lam, int(L), M, base) for L in Ls])
    log_L = np.log(Ls.astype(float))
    log_t = np.log(taus)
    slope = float(np.polyfit(log_L, log_t, 1)[0])
    return slope, taus


# ══════════════════════════════════════════════════════════════════════
# PART 1 — f-Divergence stiffness sweep
# ══════════════════════════════════════════════════════════════════════

def part1_stiffness_sweep():
    print("=" * 72)
    print("PART 1: f-Divergence Stiffness Sweep")
    print("=" * 72)

    M, base, d = 32, 500_000, 64
    Ls = np.array([128, 256, 512, 1024, 2048, 4096, 8192])

    # ── 1a. Standard divergences ──
    header = f"  {'Stiffness':28s} | {'L-exp':>7s} | {'gap':>6s} | {'grade':>5s}"
    print(f"\n  --- Standard divergences (d_head={d}, K={M}, base={base}) ---\n")
    print(header)
    print("  " + "-" * len(header))

    divergences = [
        ("L² (p=0)",        lambda t: S_L2(t, M)),
        ("χ² (p=1)",        lambda t: S_chi2(t, M)),
        ("KL  ∫ρ ln ρ",     lambda t: S_KL(t, M)),
        ("rev-KL -∫ln ρ",   lambda t: S_revKL(t, M)),
        ("Jeffreys ∫(ρ-1)lnρ", lambda t: S_Jeffreys(t, M)),
    ]
    for name, sf in divergences:
        sl, _ = L_exponent(sf, M, base, d, Ls)
        gap = abs(sl + 0.5)
        g = "★★★" if gap < 0.01 else ("★★" if gap < 0.03 else ("★" if gap < 0.06 else ""))
        print(f"  {name:28s} | {sl:+7.4f} | {gap:6.4f} | {g:>5s}")

    # ── 1b. p-sweep in S_p family ──
    print(f"\n  --- S_p(τ) = (1/M) ∫(ρ-1)²/ρ^p  sweep ---\n")
    print(f"  {'p':>6s} | {'L-exp':>7s} | {'gap':>6s} | {'grade':>5s}")
    print("  " + "-" * 36)

    ps = [0.0, 0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00, 1.25, 1.50, 2.00]
    best_gap, best_p = 1.0, 0.0
    for p in ps:
        sf = (lambda t, _p=p: S_power(t, _p, M))
        sl, _ = L_exponent(sf, M, base, d, Ls)
        gap = abs(sl + 0.5)
        g = "★★★" if gap < 0.01 else ("★★" if gap < 0.03 else ("★" if gap < 0.06 else ""))
        if gap < best_gap:
            best_gap, best_p = gap, p
        print(f"  {p:6.2f} | {sl:+7.4f} | {gap:6.4f} | {g:>5s}")

    print(f"\n  Best: p = {best_p:.2f}  (gap {best_gap:.4f})")

    # ── 1c. χ² closed-form verification ──
    print(f"\n  --- χ² closed-form vs numerical ---\n")
    for tau in [0.5, 1.0, 1.5, 2.0, 3.0]:
        cf = S_chi2(tau, M)
        nm = S_power(tau, 1.0, M)
        err = abs(cf - nm) / (abs(nm) + 1e-30)
        print(f"  τ={tau:.1f}: closed={cf:.6e}  numerical={nm:.6e}  rel_err={err:.1e}")

    # ── 1d. Effective exponent at operating point ──
    print(f"\n  --- Effective exponent d ln S / d ln τ ---\n")
    print(f"  {'τ':>5s} | {'L²':>6s} | {'p=.75':>6s} | {'χ²':>6s} | {'KL':>6s} | {'τ⁴':>6s}")
    print("  " + "-" * 44)
    dt = 0.02
    for tc in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        row = []
        for sf in [
            lambda t: S_L2(t, M),
            lambda t: S_power(t, 0.75, M),
            lambda t: S_chi2(t, M),
            lambda t: S_KL(t, M),
            lambda t: tc ** 4 / (90 * M),  # placeholder; compute locally
        ]:
            s1, s2 = sf(tc - dt), sf(tc + dt)
            if s1 > 0 and s2 > 0:
                e = (np.log(s2) - np.log(s1)) / (np.log(tc + dt) - np.log(tc - dt))
            else:
                e = float("nan")
            row.append(e)
        # τ⁴ always has exponent 4
        row[-1] = 4.00
        print(f"  {tc:5.1f} | {row[0]:6.2f} | {row[1]:6.2f} | {row[2]:6.2f} | {row[3]:6.2f} | {row[4]:6.2f}")

    # ── 1e. τ_opt table for χ² ──
    print(f"\n  --- τ_opt table for χ² stiffness ---\n")
    sf_chi2 = lambda t: S_chi2(t, M)
    sl_chi2, taus_chi2 = L_exponent(sf_chi2, M, base, d, Ls)
    exps = {128: 4.0, 256: 4.0, 512: 2.8, 1024: 2.0, 2048: 1.5, 4096: 1.5, 8192: 1.414}
    print(f"  {'L':>6s} | {'τ_χ²':>7s} | {'d/√L':>7s} | {'τ_exp':>7s} | {'ratio':>7s}")
    print("  " + "-" * 46)
    for i, L in enumerate(Ls):
        dsL = d / np.sqrt(L)
        te = exps.get(int(L), float("nan"))
        print(f"  {L:6.0f} | {taus_chi2[i]:7.3f} | {dsL:7.3f} | {te:7.3f} | {taus_chi2[i]/dsL:7.3f}")
    print(f"\n  L-exponent (χ²): {sl_chi2:+.4f}  (gap {abs(sl_chi2+0.5):.4f})")


# ══════════════════════════════════════════════════════════════════════
# PART 2 — LoRA phase-transition analysis
# ══════════════════════════════════════════════════════════════════════

def part2_lora_phase_transition():
    print("\n\n" + "=" * 72)
    print("PART 2: LoRA Phase-Transition Analysis")
    print("=" * 72)

    # ── 2a. Setup ──
    d_head = 128          # LLaMA-3-8B
    K = d_head // 2       # 64 frequency channels
    M = K                 # = 64
    base = 500_000
    L = 8192              # typical SFT length
    tau_pt = d_head / np.sqrt(L)  # 1.414

    print(f"\n  Config: d_head={d_head}, K={K}, L={L}, base={base}")
    print(f"  Pre-training τ* = d/√L = {tau_pt:.3f}")

    # ── 2b. Frozen-channel analysis ──
    print(f"\n  --- Frozen-channel analysis ---\n")
    print(f"  {'r':>5s} | {'r/K':>5s} | {'frozen':>7s} | {'Δφ_mid':>7s} | {'displaced':>10s}")
    print("  " + "-" * 48)
    for r in [4, 8, 16, 32, 48, 64, 96]:
        f_frozen = max(0, 1 - r / K)
        dphi = tau_pt ** 2 / 16
        n_displaced = dphi * K
        frozen_displaced = f_frozen * n_displaced
        print(
            f"  {r:5d} | {r/K:5.2f} | {f_frozen:7.2f} | {dphi:7.4f} | "
            f"{frozen_displaced:6.1f}/{n_displaced:.1f} frozen"
        )

    # ── 2c. LoRA effective stiffness model ──
    # S_total(τ; r, Λ₀) = S_χ²(τ) + Λ₀(1-r/K)(τ²/d_head)
    #
    # We calibrate Λ₀ from the r=16 catastrophe:
    #   At τ=1.414, ΔCE = ln(77.1/11.8) = 1.877 nats
    #   ≈ Λ₀ × (1 - 16/64) × 1.414²/128
    delta_CE = np.log(77.1 / 11.8)
    frac_frozen_16 = 1 - 16 / K
    Lambda0 = delta_CE / (frac_frozen_16 * tau_pt ** 2 / d_head)

    print(f"\n  --- Coupling strength calibration ---")
    print(f"  ΔCE (r=16, τ=1.414) = ln(77.1/11.8) = {delta_CE:.3f} nats")
    print(f"  Frozen fraction at r=16: {frac_frozen_16:.2f}")
    print(f"  → Λ₀ = {Lambda0:.1f}")

    # ── 2d. F(τ) landscape for several ranks ──
    print(f"\n  --- Variational landscape F(τ) for different LoRA ranks ---\n")
    taus = np.linspace(0.01, 3.0, 600)

    # Calibrate λ from pre-training (r=∞, S=S_χ²)
    sf_pt = lambda t: S_chi2(t, M)
    lam_pt = calibrate_lambda(sf_pt, tau_pt, L, M, base)

    ranks_to_show = [16, 32, 48, 64]
    print(f"  {'r':>4s} | {'τ_opt':>7s} | {'F(τ_opt)':>10s} | {'F(0)':>10s} | {'EVQ viable?':>12s}")
    print("  " + "-" * 56)

    for r in ranks_to_show:
        ff = max(0.0, 1 - r / K)
        sf_lora = lambda t, _ff=ff: S_chi2(t, M) + Lambda0 * _ff * t ** 2 / d_head
        # Evaluate at a grid
        Fs = np.array([sf_lora(t) - lam_pt * utility(t, L, M, base) for t in taus])
        idx = int(np.argmin(Fs))
        tau_opt_r = taus[idx]
        F_opt = Fs[idx]
        F_0 = sf_lora(0.01) - lam_pt * utility(0.01, L, M, base)
        viable = "✅ YES" if tau_opt_r > 0.3 else "❌ NO"
        print(f"  {r:4d} | {tau_opt_r:7.3f} | {F_opt:10.4e} | {F_0:10.4e} | {viable}")

    # ── 2e. Full rank sweep ──
    print(f"\n  --- LoRA rank sweep: predicted τ* ---\n")
    print(f"  {'r':>5s} | {'r/K':>5s} | {'τ*_LoRA':>8s} | {'PPL_pred':>9s} | {'status':>20s}")
    print("  " + "-" * 60)

    for r in [2, 4, 8, 16, 24, 32, 48, 64, 96, 128]:
        ff = max(0.0, 1 - min(r, K) / K)
        sf_lora = lambda t, _ff=ff: S_chi2(t, M) + Lambda0 * _ff * t ** 2 / d_head
        # Find τ_opt
        Fs = np.array([sf_lora(t) - lam_pt * utility(t, L, M, base) for t in taus])
        idx = int(np.argmin(Fs))
        tau_r = taus[idx]

        # Rough PPL model: PPL ∝ exp(S_frozen(τ_applied))
        # If we naively apply τ=τ_pt regardless of rank:
        S_fro = Lambda0 * ff * tau_pt ** 2 / d_head
        ppl_naive = 11.8 * np.exp(S_fro)

        if tau_r < 0.1:
            status = "❌ EVQ impossible"
        elif tau_r < 0.5:
            status = "⚠️  marginal"
        elif tau_r < tau_pt * 0.8:
            status = f"✓  partial EVQ"
        else:
            status = "✅ full EVQ"

        print(
            f"  {r:5d} | {min(r,K)/K:5.2f} | {tau_r:8.3f} | "
            f"{ppl_naive:9.1f} | {status}"
        )

    # ── 2f. Phase transition plot data ──
    print(f"\n  --- Phase boundary ---")
    print(f"  Critical rank (τ_opt first > 0.5): ", end="")
    for r in range(1, K + 1):
        ff = 1 - r / K
        sf = lambda t, _ff=ff: S_chi2(t, M) + Lambda0 * _ff * t ** 2 / d_head
        Fs = np.array([sf(t) - lam_pt * utility(t, L, M, base) for t in taus])
        if taus[int(np.argmin(Fs))] > 0.5:
            print(f"r_c = {r}  (r_c/K = {r/K:.2f})")
            break

    # ── 2g. SFT exponential recovery ──
    print(f"\n  --- SFT coupling decay model ---")
    print(f"  τ*_SFT(T) = τ*_pt / √(1 + Λ·exp(-T·η·σ))")
    print(f"  Λ = {Lambda0:.0f},  τ*_pt = {tau_pt:.3f}\n")
    print(f"  {'T·η·σ':>8s} | {'τ*_SFT':>8s} | {'τ*/τ*_pt':>10s}")
    print("  " + "-" * 32)
    for Tes in [0, 0.5, 1, 2, 3, 5, 10, 20]:
        tau_sft = tau_pt / np.sqrt(1 + Lambda0 * np.exp(-Tes))
        print(f"  {Tes:8.1f} | {tau_sft:8.4f} | {tau_sft/tau_pt:10.4f}")


# ══════════════════════════════════════════════════════════════════════
# PART 3 — Self-consistency: derive S(τ) from τ* ∝ 1/√L
# ══════════════════════════════════════════════════════════════════════

def part3_self_consistency():
    print("\n\n" + "=" * 72)
    print("PART 3: Self-Consistency Derivation")
    print("=" * 72)
    print("  If τ* = A·M/√L holds exactly, the balance equation along")
    print("  L = A²M²/τ² uniquely determines S'(τ) = c·τ²·h(τ).\n")

    M, base, d = 32, 500_000, 64
    A = 2  # τ* = 2M/√L = d_head/√L

    tau_grid = np.linspace(0.05, 8.0, 400)

    # Compute ∂ρ/∂τ
    def drho(phi, tau):
        if abs(tau) < 1e-10:
            return 2 * tau * ((1 - phi) ** 2 / 2 - 1 / 6)
        r = rho_tau(phi, tau)
        return r * (1 / tau + (1 - phi) * np.tanh(tau * (1 - phi)) - 1 / np.tanh(tau))

    # h(τ) = ∫ q(A²M²b^{-φ}/τ²) · (∂ρ/∂τ) dφ
    h = np.zeros(len(tau_grid))
    for i, tau in enumerate(tau_grid):
        L_eff = A ** 2 * M ** 2 / tau ** 2
        qv = q_channel(L_eff * base ** (-PHI))
        h[i] = _trapz(qv * drho(PHI, tau), PHI)

    # S'(τ) ∝ τ² h(τ);  S(τ) = cumulative integral
    Sp = tau_grid ** 2 * h
    S_derived = np.zeros(len(tau_grid))
    for i in range(1, len(tau_grid)):
        S_derived[i] = S_derived[i - 1] + 0.5 * (Sp[i] + Sp[i - 1]) * (tau_grid[i] - tau_grid[i - 1])

    # Normalize at τ = 1.414
    ref = np.argmin(np.abs(tau_grid - 1.414))
    Sn = S_derived / (S_derived[ref] + 1e-30)

    # Compare shapes
    print(f"  {'τ':>5s} | {'S_derived':>10s} | {'S(p=0)':>10s} | {'S(p=0.75)':>10s} | {'S(p=1)':>10s}")
    print("  " + "-" * 56)
    for tc in [0.5, 1.0, 1.414, 2.0, 3.0, 5.0]:
        idx = np.argmin(np.abs(tau_grid - tc))
        s0 = S_L2(tc, M) / (S_L2(1.414, M) + 1e-30)
        s75 = S_power(tc, 0.75, M) / (S_power(1.414, 0.75, M) + 1e-30)
        s1 = S_chi2(tc, M) / (S_chi2(1.414, M) + 1e-30)
        print(f"  {tc:5.3f} | {Sn[idx]:10.4f} | {s0:10.4f} | {s75:10.4f} | {s1:10.4f}")

    # Best-fit p
    best_p, best_r = 0.0, 1e10
    for p in np.arange(0.0, 2.01, 0.05):
        r2 = 0.0
        for tc in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            idx = np.argmin(np.abs(tau_grid - tc))
            sd = Sn[idx]
            sp = S_power(tc, p, M) / (S_power(1.414, p, M) + 1e-30)
            if sd > 0 and sp > 0:
                r2 += (np.log(sd) - np.log(sp)) ** 2
        if r2 < best_r:
            best_r, best_p = r2, p
    print(f"\n  Best-fit p for S_derived: p = {best_p:.2f}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    part1_stiffness_sweep()
    part2_lora_phase_transition()
    part3_self_consistency()

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"{'='*72}")
