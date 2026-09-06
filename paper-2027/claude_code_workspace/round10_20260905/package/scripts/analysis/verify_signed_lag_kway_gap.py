#!/usr/bin/env python3
"""CPU verification of signed-lag, beat, affine, gap, and k-way identities.

No checkpoint, no GPU, no LM numbers.  Run:

    conda run --no-capture-output -n aidemo python \\
        scripts/analysis/verify_signed_lag_kway_gap.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

J2 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def rotation(theta: torch.Tensor) -> torch.Tensor:
    """theta [N] -> [N,2,2]."""
    c, s = torch.cos(theta), torch.sin(theta)
    out = torch.empty(theta.shape[0], 2, 2, dtype=theta.dtype)
    out[:, 0, 0] = c
    out[:, 0, 1] = -s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    return out


def native_omega(K: int = 64, base: float = 5e5) -> np.ndarray:
    k = np.arange(K, dtype=np.float64)
    return base ** (-k / K)


def anchored_cosh_z(K: int = 64, tau: float = 4.0) -> np.ndarray:
    u = (np.arange(K) + 0.5) / K
    phi = 1.0 - np.arcsinh((1.0 - u) * np.sinh(tau)) / tau
    return (phi - phi[0]) / (phi[-1] - phi[0])


def check_beat_identity(rng: np.random.Generator) -> Check:
    max_err = 0.0
    for _ in range(300):
        Xi = rng.normal(size=(2, 2))
        w, wp, d = rng.uniform(1e-4, 2.0, size=3)
        d = 1.0 + 8000.0 * d
        th, thp = w * d, wp * d
        c, s = np.cos(th), np.sin(th)
        cp, sp = np.cos(thp), np.sin(thp)
        U = np.array([[c, -s], [s, c]])
        Up = np.array([[cp, -sp], [sp, cp]])
        lhs = float(np.sum(Xi * (Up - U)))
        dw, wbar = wp - w, 0.5 * (wp + w)
        m = wbar * d
        amp = 2.0 * np.sin(dw * d / 2.0)
        A = Xi[0, 0] + Xi[1, 1]
        B = Xi[1, 0] - Xi[0, 1]
        rhs = amp * (-A * np.sin(m) + B * np.cos(m))
        max_err = max(max_err, abs(lhs - rhs))
    ok = max_err < 1e-10
    return Check("beat identity (7)", ok, f"max|lhs-rhs|={max_err:.3e}")


def check_dU_dlambda() -> Check:
    R, v, w, d = 12.917, 0.15, 3.2e-4, 5000.0
    lam = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    omega = w * torch.exp(-R * lam * v)
    U = rotation(omega * torch.tensor([d], dtype=torch.float64))[0]
    g = torch.ones(2, 2, dtype=torch.float64)
    (U * g).sum().backward()
    analytic = -R * v * w * d * (J2 @ rotation(torch.tensor([w * d], dtype=torch.float64))[0])
    # d<U,g>/dλ = <dU/dλ, g>
    numeric = float(lam.grad)
    predicted = float((analytic * g).sum())
    err = abs(numeric - predicted)
    return Check("dU/dλ = -R v ω d J U", err < 1e-10, f"err={err:.3e}")


def check_chain_rule_linear() -> Check:
    """L = sum_d,k <Ξ_{dk}, U(ω_k(λ) d)> ; (2) must hold exactly."""
    torch.manual_seed(0)
    K, D = 16, 48
    R = 8.0
    v = torch.randn(K, dtype=torch.float64)
    v = v / v.norm()
    omega0 = torch.linspace(1.0, 1e-3, K, dtype=torch.float64)
    Xi = torch.randn(D, K, 2, 2, dtype=torch.float64)
    d = torch.arange(D, dtype=torch.float64) + 1.0

    lam = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    omega = omega0 * torch.exp(-R * lam * v)
    loss = torch.zeros((), dtype=torch.float64)
    for k in range(K):
        U = rotation(omega[k] * d)
        loss = loss + (Xi[:, k] * U).sum()
    loss.backward()

    response = torch.zeros(D, dtype=torch.float64)
    for k in range(K):
        U0 = rotation(omega0[k] * d)
        dU = (-R * v[k] * omega0[k] * d).reshape(-1, 1, 1) * (J2 @ U0)
        response = response + (Xi[:, k] * dU).sum(dim=(1, 2))
    pred = float(response.sum())
    err = abs(float(lam.grad) - pred)
    return Check("chain rule (1)-(2) linear pairing", err < 1e-8, f"err={err:.3e} dL/dλ={float(lam.grad):.6e}")


def check_chain_rule_softmax_attention() -> Check:
    """Signed lag response matches autograd through a rotary attention loss."""
    torch.manual_seed(1)
    K, T = 8, 32
    R = 6.0
    v = torch.randn(K, dtype=torch.float64)
    v = v / v.norm()
    omega0 = torch.linspace(0.8, 2e-3, K, dtype=torch.float64)
    q = torch.randn(K, 2, dtype=torch.float64)
    key = torch.randn(T, K, 2, dtype=torch.float64)
    d = torch.arange(T, dtype=torch.float64)
    target = T - 1

    def scores(omega: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(T, dtype=torch.float64)
        for k in range(K):
            U = rotation(omega[k] * d)
            qk = torch.einsum("tij,j->ti", U, q[k])
            out = out + (qk * key[:, k]).sum(dim=-1)
        return out

    lam = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    omega = omega0 * torch.exp(-R * lam * v)
    logits = scores(omega)
    loss = torch.logsumexp(logits, dim=0) - logits[target]
    loss.backward()
    autograd_val = float(lam.grad)

    # Rebuild U with grad to read Ξ = ∂L/∂U per (d,k).
    U_list = []
    omega_det = omega0.detach()
    for k in range(K):
        Uk = rotation(omega_det[k] * d).detach().requires_grad_(True)
        U_list.append(Uk)
    logits2 = torch.zeros(T, dtype=torch.float64)
    for k in range(K):
        qk = torch.einsum("tij,j->ti", U_list[k], q[k])
        logits2 = logits2 + (qk * key[:, k]).sum(dim=-1)
    loss2 = torch.logsumexp(logits2, dim=0) - logits2[target]
    grads = torch.autograd.grad(loss2, U_list)
    response = torch.zeros(T, dtype=torch.float64)
    for k, Xi_k in enumerate(grads):
        U0 = rotation(omega0[k] * d)
        dU = (-R * v[k] * omega0[k] * d).reshape(-1, 1, 1) * (J2 @ U0)
        response = response + (Xi_k * dU).sum(dim=(1, 2))
    pred = float(response.sum())
    rel = abs(autograd_val - pred) / max(abs(autograd_val), 1e-12)
    return Check(
        "chain rule through softmax attention",
        rel < 1e-8,
        f"rel={rel:.3e} autograd={autograd_val:.6e} r_tab_sum={pred:.6e}",
    )


def check_weight_can_flip_sign() -> Check:
    """Same Ω, v; two random Q/K realizations; dL/dλ can change sign."""
    torch.manual_seed(2)
    K, T = 8, 40
    R = 6.0
    v = torch.randn(K, dtype=torch.float64)
    v = v / v.norm()
    omega0 = torch.linspace(0.8, 2e-3, K, dtype=torch.float64)
    d = torch.arange(T, dtype=torch.float64)
    target = T - 1
    signs = []
    for seed in range(40):
        torch.manual_seed(100 + seed)
        q = torch.randn(K, 2, dtype=torch.float64)
        key = torch.randn(T, K, 2, dtype=torch.float64)
        lam = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        omega = omega0 * torch.exp(-R * lam * v)
        logits = torch.zeros(T, dtype=torch.float64)
        for k in range(K):
            U = rotation(omega[k] * d)
            qk = torch.einsum("tij,j->ti", U, q[k])
            logits = logits + (qk * key[:, k]).sum(dim=-1)
        loss = torch.logsumexp(logits, dim=0) - logits[target]
        loss.backward()
        signs.append(float(np.sign(lam.grad.item() or 0.0)))
    npos = sum(s > 0 for s in signs)
    nneg = sum(s < 0 for s in signs)
    ok = npos >= 5 and nneg >= 5
    return Check(
        "same Ω, different weights can flip sign of dL/dλ",
        ok,
        f"40 random Q/K: +{npos} / -{nneg} / 0={40-npos-nneg}",
    )


def check_carrier_support() -> Check:
    L, K, base = 4096.0, 64, 5e5
    omega = native_omega(K, base)
    u = omega * L
    Tset = np.where((u > math.pi / 2) & (u < math.pi))[0]
    t = math.pi / u[Tset]
    expected = np.array([35, 36, 37, 38])
    ok = np.array_equal(Tset, expected)
    return Check(
        "transition carrier set T(Ω,L)",
        ok,
        f"k={Tset.tolist()}  π/(ωL)={np.round(t, 3).tolist()}",
    )


def check_envelope_not_in_window() -> Check:
    L, K, base = 4096.0, 64, 5e5
    omega = native_omega(K, base)
    z_geo = np.linspace(0.0, 1.0, K)
    z_c = anchored_cosh_z(K, 4.0)
    dz = float(np.max(np.abs(z_c - z_geo)))
    learned = 0.00130
    ratio = dz / learned
    R = ((K - 1) / K) * math.log(base)
    k = 38
    dwL_full = R * dz * (omega[k] * L)
    dwL_learned = R * learned * (omega[k] * L)
    t_env_full = 2 * math.pi / dwL_full
    t_env_learned = 2 * math.pi / dwL_learned
    ok = (t_env_learned > 100.0) and (t_env_full < 3.0)
    return Check(
        "F10 envelope zero not in (L,3L); full Cosh can be",
        ok,
        f"|Δz|_cosh={dz:.4f} ratio={ratio:.1f}  "
        f"d_env/L full={t_env_full:.2f} learned={t_env_learned:.1f}",
    )


def relative_defect(theta: np.ndarray) -> float:
    inc = np.exp(1j * np.diff(theta))
    return float(np.max(np.abs(inc - inc[0])))


def check_f11_affine() -> Check:
    L = 4096
    omega = native_omega()[40]
    p = np.arange(0, 3 * L + 1, dtype=np.float64)
    affine = relative_defect(omega * p)
    m = 0.5
    theta = np.where(p <= L, omega * p, omega * L + (1.0 - m) * omega * (p - L))
    kinked = relative_defect(theta)
    expected = abs(np.exp(1j * omega) - np.exp(1j * (1.0 - m) * omega))
    ok = affine < 1e-12 and abs(kinked - expected) < 1e-12
    return Check(
        "F11 leaves the exact relative set",
        ok,
        f"affine={affine:.3e} kinked={kinked:.6e} |e^{{iω}}-e^{{i(1-m)ω}}|={expected:.6e}",
    )


def check_gap_geometry() -> Check:
    rng = np.random.default_rng(0)
    K, n, N = 64, 63, 4000
    z_geo = np.linspace(0.0, 1.0, K)
    z_c = anchored_cosh_z(K, 4.0)
    dC = z_c - z_geo
    dC = dC / np.linalg.norm(dC)
    g = rng.normal(size=(N, n))
    e = np.exp(g - g.max(axis=1, keepdims=True))
    delta = e / e.sum(axis=1, keepdims=True)
    Z = np.concatenate(
        [np.zeros((N, 1)), np.cumsum(delta, axis=1)[:, :-1], np.ones((N, 1))],
        axis=1,
    )
    dZ = Z - z_geo
    dZ = dZ / np.linalg.norm(dZ, axis=1, keepdims=True)
    c = dZ @ dC
    p99 = float(np.quantile(c, 0.99))
    p_hi = float(np.mean(c > 0.854))
    U, S, Vt = np.linalg.svd(dZ - dZ.mean(0), full_matrices=False)
    mode1 = Vt[0]
    mode1 = mode1 / np.linalg.norm(mode1)
    if mode1 @ dC < 0:
        mode1 = -mode1
    pca_cos = float(mode1 @ dC)
    var1 = float(S[0] ** 2 / np.sum(S**2))
    # F10 claim: 0.854 is inside the null (below p99 ~ 0.965)
    ok = (0.90 < p99 < 0.99) and (pca_cos > 0.85) and (p_hi > 0.05) and (p_hi < 0.25)
    return Check(
        "softmax-gap image: PCA1~Cosh, 0.854 inside null p99",
        ok,
        f"p99={p99:.3f} P(cos>0.854)={p_hi:.3f} PCA1·Cosh={pca_cos:.3f} var1={var1:.3f}",
    )


def block_invsqrt(omega: np.ndarray, L: int) -> np.ndarray:
    d = np.arange(L, dtype=np.float64)
    p = (L - d).astype(np.float64)
    p /= p.sum()
    K = len(omega)
    S = np.zeros((2 * K, 2 * K), dtype=np.float64)
    for i, w in enumerate(omega):
        X = np.stack([np.cos(w * d), np.sin(w * d)], axis=1)
        S[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = X.T @ (p[:, None] * X)
    wS, VS = np.linalg.eigh((S + S.T) / 2)
    keep = wS > 1e-12 * max(float(wS.max()), 1.0)
    return (VS[:, keep] / np.sqrt(wS[keep])) @ VS[:, keep].T


def phi_vec(omega: np.ndarray, d: float) -> np.ndarray:
    th = omega * d
    return np.stack([np.cos(th), np.sin(th)], axis=1).reshape(-1)


def kway_eigs(omega: np.ndarray, L: int, lags: list[int], M: np.ndarray | None = None) -> np.ndarray:
    iS = block_invsqrt(omega, L)
    if M is None:
        A = iS
    else:
        wM, VM = np.linalg.eigh((M + M.T) / 2)
        keep = wM > 1e-12 * max(float(wM.max()), 1.0)
        sqrtM = (VM[:, keep] * np.sqrt(wM[keep])) @ VM[:, keep].T
        A = sqrtM @ iS
    U = np.column_stack([A @ phi_vec(omega, float(d)) for d in lags])
    G = U.T @ U
    return np.linalg.eigvalsh((G + G.T) / 2)


def check_kway_not_pairwise() -> Check:
    lam_eq = float(np.min(np.linalg.eigvalsh(np.array(
        [[1, 0.4, 0.4], [0.4, 1, 0.4], [0.4, 0.4, 1]], dtype=np.float64
    ))))
    lam_uneq = float(np.min(np.linalg.eigvalsh(np.array(
        [[1, 0.95, 0.15], [0.95, 1, 0.10], [0.15, 0.10, 1]], dtype=np.float64
    ))))
    mean_uneq = (0.95 + 0.15 + 0.10) / 3
    ok = abs(lam_eq - 0.6) < 1e-12 and lam_uneq < 0.1 and abs(mean_uneq - 0.4) < 1e-12
    return Check(
        "k=3 λ_min not fixed by mean pairwise corr",
        ok,
        f"equal ρ=0.4 → λmin={lam_eq:.3f}; (0.95,0.15,0.10) → λmin={lam_uneq:.3f}",
    )


def _pair_mask_M(K: int, which: np.ndarray) -> np.ndarray:
    dim = 2 * K
    diag = np.zeros(dim, dtype=np.float64)
    for k in which:
        diag[2 * k : 2 * k + 2] = 1.0
    return np.diag(diag)


def check_kway_M_can_flip_table_ranking() -> Check:
    """Same lags; structured M (not isotropic noise) can reverse geo vs Cosh λ_min."""
    L, K, base = 4096, 64, 5e5
    omega_g = native_omega(K, base)
    z_c = anchored_cosh_z(K, 4.0)
    x0, x1 = -np.log(omega_g[0]), -np.log(omega_g[-1])
    omega_c = np.exp(-(x0 + (x1 - x0) * z_c))
    lags = [L + 512, L + 1536, L + 2560]
    geo_I = kway_eigs(omega_g, L, lags, None)[0]
    cosh_I = kway_eigs(omega_c, L, lags, None)[0]
    i_sign = np.sign(cosh_I - geo_I)
    masks = {
        "fast 0:16": _pair_mask_M(K, np.arange(16)),
        "mid 24:40": _pair_mask_M(K, np.arange(24, 40)),
        "slow 48:64": _pair_mask_M(K, np.arange(48, 64)),
        "transition 35:39": _pair_mask_M(K, np.arange(35, 39)),
    }
    flips = []
    details = []
    for name, M in masks.items():
        ev_g = kway_eigs(omega_g, L, lags, M)[0]
        ev_c = kway_eigs(omega_c, L, lags, M)[0]
        flipped = np.sign(ev_c - ev_g) != i_sign
        if flipped:
            flips.append(name)
        details.append(f"{name} geo={ev_g:.2f} cosh={ev_c:.2f}")
    ok = len(flips) >= 1 and geo_I > 0 and cosh_I > 0
    return Check(
        "k-way: pair-band M can reverse geo vs Cosh λ_min ranking",
        ok,
        f"M=I geo={geo_I:.2f} cosh={cosh_I:.2f} (Cosh higher). flips={flips or 'none'}; "
        + "; ".join(details),
    )


def main() -> int:
    rng = np.random.default_rng(0)
    checks = [
        check_beat_identity(rng),
        check_dU_dlambda(),
        check_chain_rule_linear(),
        check_chain_rule_softmax_attention(),
        check_weight_can_flip_sign(),
        check_carrier_support(),
        check_envelope_not_in_window(),
        check_f11_affine(),
        check_gap_geometry(),
        check_kway_not_pairwise(),
        check_kway_M_can_flip_table_ranking(),
    ]
    print("signed-lag / k-way / gap geometry  CPU verification")
    print("OLMo checkpoint: absent on this machine; real Ξ not computed.")
    print()
    failed = 0
    for c in checks:
        mark = "PASS" if c.passed else "FAIL"
        if not c.passed:
            failed += 1
        print(f"[{mark}] {c.name}")
        print(f"       {c.detail}")
    print()
    print(f"{len(checks) - failed}/{len(checks)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
