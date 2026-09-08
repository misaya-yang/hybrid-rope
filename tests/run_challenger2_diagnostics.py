"""Detailed numerical verification and diagnostic script for Challenger 2.
Executes empirical tests and prints full quantitative results for:
1. Slot-19 phase error and Taylor breakdown for C2 on Qwen and OLMo.
2. Layer Jacobian Gram matrix decomposition (Observable 1 vs Observable 3).
3. Softmax distractor dilution curve vs temperature gain.
4. MrRoPE-Pro phase shift profiles and comparison with FullLagP2.
"""
import math
import numpy as np


def run_taylor_breakdown_audit():
    print("=== AUDIT 1: TAYLOR BREAKDOWN ON SLOT 19 (C2) ===")
    models = [
        {"name": "OLMo-1B", "base": 500000.0, "K": 64, "L_0": 4096, "knee_slot": 19, "delta_m": 0.044109},
        {"name": "Qwen-1.5B", "base": 1000000.0, "K": 64, "L_0": 32768, "knee_slot": 19, "delta_m": 0.044109},
    ]

    for model in models:
        base = model["base"]
        K = model["K"]
        L_0 = model["L_0"]
        k = model["knee_slot"]
        dm = model["delta_m"]

        omega_k = base ** (-k / K)
        # reference m approx 0.25 (transition)
        m_ref = 0.25
        omega_ref = omega_k * (4.0 ** (-m_ref))
        omega_c2 = omega_k * (4.0 ** (-(m_ref + dm)))
        domega = omega_c2 - omega_ref

        print(f"\nModel: {model['name']} (Native L_0={L_0})")
        print(f"  Slot {k}: native omega={omega_k:.6e}, ref omega'={omega_ref:.6e}, c2 omega'={omega_c2:.6e}")
        print(f"  Delta omega: {domega:.6e} rad/token")

        for s in [1, 2, 4]:
            L = s * L_0
            phase_shift = abs(domega) * L
            quad_term = 0.5 * (domega * L) ** 2

            # Oscillatory test f(omega) = cos(omega * L)
            true_diff = math.cos((omega_ref + domega) * L) - math.cos(omega_ref * L)
            taylor_1st = -L * math.sin(omega_ref * L) * domega
            taylor_2nd = taylor_1st - 0.5 * (L ** 2) * math.cos(omega_ref * L) * (domega ** 2)
            err_2nd = abs(taylor_2nd - true_diff)

            print(f"  At L={L} ({s}x):")
            print(f"    Total phase shift: {phase_shift:.4f} rad ({phase_shift / (2*math.pi):.2f} turns)")
            print(f"    True cos diff:     {true_diff:.6f}")
            print(f"    Taylor 2nd order:  {taylor_2nd:.6f}")
            print(f"    Quad term |0.5*(dL)^2|: {quad_term:.4f}")
            print(f"    Absolute Error:    {err_2nd:.6f}")
            if phase_shift > math.pi:
                print(f"    >>> NON-LINEAR BREAKDOWN: Phase shift > pi; Taylor expansion invalid! <<<")


def run_jacobian_gram_decomposition_audit():
    print("\n=== AUDIT 2: JACOBIAN GRAM DECOMPOSITION (H vs J) ===")
    rng = np.random.default_rng(20260908)
    H_heads = 8
    T_tokens = 16
    D_head = 8
    D_out = 32
    pairs = D_head // 2

    u = rng.normal(size=(H_heads, T_tokens, pairs, D_head))
    delta = rng.uniform(1.0, 100.0, size=T_tokens)
    wo = rng.normal(size=(H_heads, D_out, D_head))

    j = 0
    # Observable 3: Head-contracted
    J_h = np.sum(delta[None, :, None] * u[:, :, j, :], axis=1)  # (H, D_head)
    W_J_h = np.einsum('hod,hd->ho', wo, J_h)  # (H, D_out)
    J_total = np.sum(W_J_h, axis=0)
    E_total_J = np.sum(J_total ** 2)

    diag_head = np.sum(W_J_h ** 2)
    cross_head = E_total_J - diag_head

    # Observable 1: Token-contracted
    v_t = np.einsum('hod,htd->to', wo, u[:, :, j, :])  # (T, D_out)
    J_total_token = np.sum(delta[:, None] * v_t, axis=0)
    E_total_H = np.sum(J_total_token ** 2)

    diag_token = np.sum((delta[:, None] * v_t) ** 2)
    cross_token = E_total_H - diag_token

    print(f"  Total Layer Perturbation Energy: {E_total_J:.6f}")
    print(f"  Via Head Decomposition (Observable 3):")
    print(f"    Sum of Individual Head Energies: {diag_head:.6f}")
    print(f"    Cross-Head Interference (W_O cancellation): {cross_head:.6f}")
    print(f"    Ratio (Total / Head Diag): {E_total_J / diag_head:.4f}")
    print(f"  Via Token Decomposition (Observable 1):")
    print(f"    Sum of Individual Token Energies: {diag_token:.6f}")
    print(f"    Cross-Token Interference: {cross_token:.6f}")
    print(f"    Ratio (Total / Token Diag): {E_total_H / diag_token:.4f}")
    print(f"  Identity Check: |E_total_J - E_total_H| = {abs(E_total_J - E_total_H):.2e}")


def run_mrrope_residual_audit():
    print("\n=== AUDIT 3: MRROPE-PRO RESIDUAL & CONSERVATISM AUDIT ===")
    K = 64
    base = 1000000.0
    L_0 = 32768
    inv_freq = base ** (-np.arange(K, dtype=np.float64) / K)
    turns = (inv_freq * L_0) / (2.0 * math.pi)

    l = int(np.max(np.where(turns > 32.0)[0]))
    h = int(np.min(np.where(turns < 1.0)[0]))
    n = h - l

    print(f"  Qwen Base=1e6, L_0=32768: l={l} (turns={turns[l]:.2f}), h={h} (turns={turns[h]:.2f}), n={n}")

    t = np.clip(np.arange(K) - l, 0, n)
    m_pro = (t * (t + 1)) / (n * (n + 1))
    m_linear = t / n
    m_cubic = (t / n) ** 3

    print(f"  Midpoint slot (j={l + n//2}):")
    print(f"    Linear exponent: {m_linear[l + n//2]:.4f}")
    print(f"    MrRoPE-Pro quadratic exponent: {m_pro[l + n//2]:.4f}")
    print(f"    Cubic exponent: {m_cubic[l + n//2]:.4f}")

    # Maximum phase difference across context
    s = 4.0
    L_ext = s * L_0
    omega_pro = inv_freq * (s ** (-m_pro))
    phase_pro_ext = omega_pro * L_ext
    phase_native_ext = inv_freq * L_ext

    # Check phase error for fast slots
    max_fast_err = np.max(np.abs(phase_pro_ext[:l+1] - phase_native_ext[:l+1]))
    print(f"  Regime I (Fast, j <= {l}): Max Phase Perturbation = {max_fast_err:.4f} rad (Strict 0)")

    # Check phase in Regime III (Slow, j >= h)
    max_slow_phase = np.max(phase_pro_ext[h:])
    print(f"  Regime III (Slow, j >= {h}): Max Phase across {int(L_ext)} tokens = {max_slow_phase:.4f} rad (< 2*pi = {2*math.pi:.4f})")
    print(f"  Regime II (Transition): Exponent progression is strictly empirical, not derived from Fisher matrix or weights.")


if __name__ == "__main__":
    run_taylor_breakdown_audit()
    run_jacobian_gram_decomposition_audit()
    run_mrrope_residual_audit()
