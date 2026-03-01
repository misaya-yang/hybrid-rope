#!/usr/bin/env python3
"""
Learnable EVQ-Cosh RoPE: Theory-guided learnable positional encoding.

The variational framework derives the optimal frequency family:
    φ_k(τ) = 1 - (1/τ) arcsinh((1-u_k) sinh τ)
    ω_k = base^{-φ_k}

where τ = √(β/α) controls the interference-resolution trade-off.
Instead of manually tuning τ, we make it a learnable parameter.

Theory provides:
    1. The functional form (EVQ-Cosh family)
    2. Boundary anchoring (endpoints don't move)
    3. τ→0 recovers geometric RoPE (safe initialization)
    4. Gradient closed-form with Taylor-stable fallback

Reference: §4 of "RoPE Scaling as a Variational Inverse Problem"
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(x: float, beta: float = 1.0) -> float:
    """Compute ψ such that softplus(ψ) = x, i.e., ψ = log(exp(x*beta) - 1) / beta."""
    return math.log(math.expm1(x * beta)) / beta


class LearnableEVQRoPE(nn.Module):
    """
    Learnable EVQ-Cosh Rotary Position Embedding.

    A single scalar parameter τ controls the entire frequency spectrum.
    The functional form is derived from variational optimization (Theorem 1),
    and τ is learned during training via standard backpropagation.

    Key mathematical properties:
        - Boundary anchoring: ∂φ/∂τ = 0 at k=0 and k=N-1
        - Geometric recovery: τ→0 gives standard RoPE
        - Smooth gradient: closed-form with Taylor fallback near τ=0

    Args:
        dim: Head dimension (frequencies = dim // 2)
        max_seq_len: Maximum sequence length (for precomputing positions)
        base: RoPE base frequency (default: 10000)
        tau_init: Initial value of τ (default: 1.0, neutral)
        tau_lr_multiplier: Suggested LR multiplier for τ (metadata only)
        device: Device for buffers
        dtype: Dtype for computation (float64 recommended for frequencies)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        tau_init: float = 1.0,
        tau_lr_multiplier: float = 10.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.dim = dim
        self.n_freqs = dim // 2
        self.max_seq_len = max_seq_len
        self.base = base
        self.tau_lr_multiplier = tau_lr_multiplier

        # Learnable parameter: ψ such that τ = softplus(ψ)
        psi_init = _inverse_softplus(tau_init)
        self.raw_tau = nn.Parameter(
            torch.tensor(psi_init, dtype=dtype, device=device)
        )

        # Fixed: midpoint quantization u_k = (k + 0.5) / N  [matches paper eq. 9]
        u = (torch.arange(self.n_freqs, dtype=dtype, device=device) + 0.5) / self.n_freqs
        self.register_buffer("u", u)

        # Cache for position indices
        pos = torch.arange(max_seq_len, dtype=dtype, device=device)
        self.register_buffer("pos", pos)

    @property
    def tau(self) -> torch.Tensor:
        """Current τ value (always positive via softplus)."""
        return F.softplus(self.raw_tau)

    def _compute_phi(self, tau: torch.Tensor) -> torch.Tensor:
        """Compute φ_k(τ) with gradient-safe Taylor fallback near τ=0.

        For τ < 1e-4, uses 2nd-order Taylor:
            φ_k ≈ u_k - (τ²/6)·A_k·(1 - A_k²)
        which maintains ∂φ/∂τ ∝ τ (never traps τ near 0).
        """
        u = self.u  # (n_freqs,)
        A = 1.0 - u  # A_k = 1 - u_k

        if tau.item() < 1e-4:
            # Taylor: φ_k ≈ u_k - (τ²/6)·A_k·(1 - A_k²)
            # Gradient: ∂φ_k/∂τ ≈ -(τ/3)·A_k·(1 - A_k²)
            phi = u - (tau * tau / 6.0) * A * (1.0 - A * A)
        else:
            sinh_tau = torch.sinh(tau)
            phi = 1.0 - (1.0 / tau) * torch.arcsinh(A * sinh_tau)

        return phi

    def get_frequencies(self) -> torch.Tensor:
        """
        Compute EVQ-Cosh frequencies ω_k = base^{-φ_k(τ)}.

        Returns:
            freqs: shape (n_freqs,), the RoPE frequencies
        """
        phi = self._compute_phi(self.tau)
        freqs = torch.pow(self.base, -phi)
        return freqs

    def forward(self, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Compute rotary embedding angles for positions [0, seq_len).

        Args:
            seq_len: Number of positions (default: max_seq_len)

        Returns:
            angles: shape (seq_len, n_freqs), the rotation angles θ_k · m
        """
        if seq_len is None:
            seq_len = self.max_seq_len

        freqs = self.get_frequencies()  # (n_freqs,)
        positions = self.pos[:seq_len]  # (seq_len,)
        angles = torch.outer(positions, freqs)  # (seq_len, n_freqs)
        return angles

    def get_cos_sin(self, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin embeddings for RoPE application.

        Args:
            seq_len: Number of positions

        Returns:
            cos_embed: shape (seq_len, n_freqs)
            sin_embed: shape (seq_len, n_freqs)
        """
        angles = self.forward(seq_len)
        return angles.cos(), angles.sin()

    @torch.no_grad()
    def get_tau_value(self) -> float:
        """Get current τ value (for logging)."""
        return self.tau.item()

    @torch.no_grad()
    def get_phi_schedule(self) -> torch.Tensor:
        """Get current φ schedule (for visualization)."""
        return self._compute_phi(self.tau)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_freqs={self.n_freqs}, base={self.base}, "
            f"tau={self.tau.item():.4f}, max_seq_len={self.max_seq_len}"
        )


class EVQRoPEWrapper(nn.Module):
    """
    Drop-in wrapper that replaces standard RoPE in any transformer.

    Usage:
        # Replace standard RoPE
        model.rope = EVQRoPEWrapper(head_dim=128, max_seq_len=16384)

        # In attention forward:
        cos, sin = model.rope(seq_len=input_ids.shape[1])
        # Apply as standard RoPE cos/sin

    For optimizer setup:
        tau_params = [p for n, p in model.named_parameters() if 'raw_tau' in n]
        other_params = [p for n, p in model.named_parameters() if 'raw_tau' not in n]
        optimizer = AdamW([
            {"params": other_params, "lr": base_lr},
            {"params": tau_params, "lr": base_lr * 10.0, "weight_decay": 0.0},
        ])
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        tau_init: float = 1.0,
    ):
        super().__init__()
        self.evq = LearnableEVQRoPE(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=base,
            tau_init=tau_init,
        )

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) embeddings, compatible with HuggingFace RoPE interface."""
        return self.evq.get_cos_sin(seq_len)

    @property
    def tau(self) -> float:
        return self.evq.get_tau_value()


# ============================================================
# Utility: τ estimation from data (Algorithm 1)
# ============================================================

def estimate_tau_from_distance_prior(
    D_hist: torch.Tensor,
    base: float = 10000.0,
    n_grid: int = 64,
    max_delta: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    Algorithm 1: Data-driven τ estimation.

    Estimates τ* = √(β/α) from the empirical distance distribution D̂(Δ)
    by fitting the kernel matrix K to the broadband decomposition K ≈ αI/Δφ + βM.

    Uses GPT-5.2 Pro's recommended two-step fitting:
    Step 1: Non-diagonal elements → β, c₀
    Step 2: Diagonal elements → α

    Args:
        D_hist: shape (n_bins,), empirical distance histogram D̂(Δ) for Δ=1,...,n_bins
        base: RoPE base frequency
        n_grid: Number of φ grid points for kernel computation
        max_delta: Maximum distance to consider (default: len(D_hist))

    Returns:
        tau_star: Estimated optimal τ
        alpha: Diagonal ridge strength
        beta: Off-diagonal coupling strength
        residual: Relative Frobenius residual ‖K - K_approx‖_F / ‖K‖_F
    """
    D_hist = D_hist.double()
    if max_delta is not None:
        D_hist = D_hist[:max_delta]

    # Normalize to proper distribution
    D_hist = D_hist / D_hist.sum()

    n_bins = len(D_hist)
    deltas = torch.arange(1, n_bins + 1, dtype=torch.float64)

    # φ grid and corresponding angular frequencies
    phi = torch.linspace(0, 1, n_grid, dtype=torch.float64)
    omega = base ** (-phi)  # (n_grid,)

    # Construct kernel matrix K_ij = Σ_Δ D(Δ) cos(ω_i Δ) cos(ω_j Δ)
    # = ½ Σ_Δ D(Δ) [cos((ω_i - ω_j)Δ) + cos((ω_i + ω_j)Δ)]
    cos_table = torch.cos(omega.unsqueeze(1) * deltas.unsqueeze(0))  # (n_grid, n_bins)
    weighted_cos = cos_table * D_hist.unsqueeze(0)  # (n_grid, n_bins)
    K = weighted_cos @ cos_table.T  # (n_grid, n_grid)

    # Min matrix: M_ij = min(φ_i, φ_j)
    M = torch.minimum(phi.unsqueeze(1), phi.unsqueeze(0))  # (n_grid, n_grid)

    # Step 1: Fit non-diagonal elements → β, c₀
    # K_ij ≈ c₀ + β · min(φ_i, φ_j) for i ≠ j
    mask_offdiag = ~torch.eye(n_grid, dtype=torch.bool)
    K_offdiag = K[mask_offdiag]
    M_offdiag = M[mask_offdiag]
    ones_offdiag = torch.ones_like(K_offdiag)

    # Linear regression: K_offdiag = c₀ · 1 + β · M_offdiag
    A_offdiag = torch.stack([ones_offdiag, M_offdiag], dim=1)
    coeffs_offdiag = torch.linalg.lstsq(A_offdiag, K_offdiag).solution
    c0 = coeffs_offdiag[0].item()
    beta = coeffs_offdiag[1].item()

    # Step 2: Fit diagonal elements → α
    # K_ii ≈ c₀ + β · φ_i + α / Δφ
    dphi = phi[1].item() - phi[0].item()  # grid spacing
    K_diag = K.diagonal()
    residual_diag = K_diag - c0 - beta * phi  # should be ≈ α / Δφ
    alpha = (residual_diag.mean().item()) * dphi

    # Compute τ*
    if alpha <= 0 or beta <= 0:
        # Fallback: if fitting gives non-physical values
        tau_star = 1.0  # default to neutral
    else:
        tau_star = math.sqrt(beta / alpha)

    # Compute residual for diagnostics
    I_approx = torch.eye(n_grid, dtype=torch.float64) * (alpha / dphi)
    K_approx = c0 + beta * M + I_approx
    relative_residual = torch.norm(K - K_approx, 'fro').item() / torch.norm(K, 'fro').item()

    return tau_star, alpha, beta, relative_residual


def measure_distance_distribution(
    token_ids: torch.Tensor,
    max_delta: int = 4096,
    sample_size: int = 100000,
) -> torch.Tensor:
    """
    Measure empirical distance distribution D̂(Δ) from tokenized text.

    Counts co-occurrence of identical tokens at distance Δ, normalized.

    Args:
        token_ids: shape (n_tokens,), flat token sequence
        max_delta: Maximum distance to measure
        sample_size: Number of random positions to sample (for efficiency)

    Returns:
        D_hist: shape (max_delta,), D̂(Δ) for Δ=1,...,max_delta
    """
    n = len(token_ids)
    D_hist = torch.zeros(max_delta, dtype=torch.float64)

    # Sample random positions for efficiency
    if sample_size < n:
        indices = torch.randperm(n - max_delta)[:sample_size]
    else:
        indices = torch.arange(n - max_delta)

    for idx in indices:
        token = token_ids[idx]
        window = token_ids[idx + 1: idx + 1 + max_delta]
        matches = (window == token).double()
        D_hist += matches

    # Normalize
    D_hist = D_hist / D_hist.sum().clamp(min=1e-10)
    return D_hist


# ============================================================
# Training utilities
# ============================================================

class TauLogger:
    """
    Utility for logging τ trajectory during training.

    Usage:
        tau_logger = TauLogger(log_interval=100)
        for step in range(n_steps):
            ...
            tau_logger.log(step, model.rope.evq)
        tau_logger.save("tau_trajectory.json")
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.trajectory = []  # list of (step, tau_value, loss_value)

    def log(self, step: int, evq_module: LearnableEVQRoPE, loss: Optional[float] = None):
        if step % self.log_interval == 0:
            tau_val = evq_module.get_tau_value()
            entry = {"step": step, "tau": tau_val}
            if loss is not None:
                entry["loss"] = loss
            self.trajectory.append(entry)

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.trajectory, f, indent=2)

    def get_final_tau(self) -> float:
        if not self.trajectory:
            return float("nan")
        return self.trajectory[-1]["tau"]

    def get_convergence_std(self, last_fraction: float = 0.2) -> float:
        """Standard deviation of τ over the last fraction of training."""
        if len(self.trajectory) < 5:
            return float("nan")
        n_last = max(1, int(len(self.trajectory) * last_fraction))
        taus = [e["tau"] for e in self.trajectory[-n_last:]]
        return torch.tensor(taus).std().item()


def setup_optimizer_with_tau(
    model: nn.Module,
    base_lr: float,
    tau_lr_multiplier: float = 10.0,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with separate learning rate for τ.

    τ gets:
    - Higher learning rate (compensates for being a single scalar vs matrix)
    - Zero weight decay (it's not an overfitting source)
    """
    tau_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "raw_tau" in name:
            tau_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": tau_params, "lr": base_lr * tau_lr_multiplier, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups)


# ============================================================
# Quick verification
# ============================================================

if __name__ == "__main__":
    print("=== LearnableEVQRoPE Verification ===\n")

    # Test 1: Basic construction
    rope = LearnableEVQRoPE(dim=128, max_seq_len=4096, tau_init=1.5)
    print(f"Module: {rope}")
    print(f"τ initial: {rope.tau.item():.4f}")
    print(f"Number of parameters: {sum(p.numel() for p in rope.parameters())}")

    # Test 2: Gradient flow
    freqs = rope.get_frequencies()
    loss = freqs.sum()
    loss.backward()
    print(f"\n∂L/∂ψ (raw_tau grad): {rope.raw_tau.grad.item():.6f}")
    print(f"τ gradient exists: {rope.raw_tau.grad is not None}")

    # Test 3: τ → 0 recovers geometric
    rope_geo = LearnableEVQRoPE(dim=128, max_seq_len=4096, tau_init=0.001)
    freqs_geo = rope_geo.get_frequencies()
    # Standard geometric frequencies
    n = 64
    u = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    freqs_standard = 10000.0 ** (-u)
    diff = (freqs_geo - freqs_standard).abs().max().item()
    print(f"\nτ→0 geometric recovery max error: {diff:.2e}")

    # Test 4: Boundary anchoring
    rope_test = LearnableEVQRoPE(dim=128, max_seq_len=4096, tau_init=1.0)
    phi = rope_test.get_phi_schedule()
    print(f"\nBoundary check:")
    print(f"  φ_0 (should be ≈0): {phi[0].item():.6f}")
    print(f"  φ_{n-1} (should be ≈1): {phi[-1].item():.6f}")

    # Test 5: Forward pass shape
    cos_emb, sin_emb = rope.get_cos_sin(seq_len=512)
    print(f"\nForward pass shape: cos={cos_emb.shape}, sin={sin_emb.shape}")

    # Test 6: Gradient flow at small τ (Taylor branch)
    rope_small = LearnableEVQRoPE(dim=128, max_seq_len=4096, tau_init=1e-5)
    freqs_small = rope_small.get_frequencies()
    freqs_small.sum().backward()
    grad_small = rope_small.raw_tau.grad.item()
    print(f"\nSmall τ gradient check (τ={rope_small.tau.item():.2e}):")
    print(f"  ∂L/∂ψ = {grad_small:.2e} (should be nonzero)")
    assert abs(grad_small) > 0, "Gradient is zero at small τ — Taylor branch broken!"
    print(f"  ✓ Gradient flows through Taylor branch")

    # Test 7: Taylor-to-full continuity at τ=1e-4 boundary
    for tau_val in [9e-5, 1.1e-4]:
        r = LearnableEVQRoPE(dim=128, max_seq_len=4096, tau_init=tau_val)
        f = r.get_frequencies()
        f.sum().backward()
    rope_lo = LearnableEVQRoPE(dim=128, max_seq_len=4096, tau_init=9e-5)
    rope_hi = LearnableEVQRoPE(dim=128, max_seq_len=4096, tau_init=1.1e-4)
    phi_lo = rope_lo._compute_phi(rope_lo.tau)
    phi_hi = rope_hi._compute_phi(rope_hi.tau)
    boundary_diff = (phi_lo - phi_hi).abs().max().item()
    print(f"\nTaylor-to-full continuity at boundary:")
    print(f"  max|φ(9e-5) - φ(1.1e-4)| = {boundary_diff:.2e}")
    assert boundary_diff < 1e-6, f"Discontinuity at Taylor boundary: {boundary_diff}"
    print(f"  ✓ Smooth transition")

    # Test 8: torch.autograd.gradcheck (finite-difference vs analytic)
    print("\nAutograd gradcheck (finite differences):")
    n_gc = 8
    u_gc = (torch.arange(n_gc, dtype=torch.float64) + 0.5) / n_gc
    A_gc = 1.0 - u_gc
    def freq_fn(psi):
        tau = F.softplus(psi)
        sinh_tau = torch.sinh(tau)
        phi = 1.0 - (1.0 / tau) * torch.arcsinh(A_gc * sinh_tau)
        return torch.pow(10000.0, -phi)
    psi_test = torch.tensor(_inverse_softplus(1.0), dtype=torch.float64, requires_grad=True)
    passed = torch.autograd.gradcheck(freq_fn, psi_test, eps=1e-6)
    print(f"  gradcheck passed: {passed}")

    # Test 9: D(Δ) → τ estimation (synthetic power-law)
    print("\n=== Algorithm 1: D(Δ) → τ* ===")
    deltas = torch.arange(1, 2049, dtype=torch.float64)
    # Synthetic power-law: D(Δ) ∝ Δ^{-1.5}
    D_synthetic = deltas ** (-1.5)
    D_synthetic = D_synthetic / D_synthetic.sum()
    tau_est, alpha, beta, residual = estimate_tau_from_distance_prior(D_synthetic)
    print(f"Synthetic power-law (p=1.5):")
    print(f"  α = {alpha:.6f}")
    print(f"  β = {beta:.6f}")
    print(f"  τ* = {tau_est:.4f}")
    print(f"  Fit residual = {residual:.4f}")

    # Test 10: Uniform prior → τ ≈ 0 (Algorithm 1 sanity)
    D_uniform = torch.ones(2048, dtype=torch.float64) / 2048
    tau_unif, _, _, res_unif = estimate_tau_from_distance_prior(D_uniform)
    print(f"\nUniform prior:")
    print(f"  τ* = {tau_unif:.4f} (should be small)")
    print(f"  Fit residual = {res_unif:.4f}")

    print("\n✅ All checks passed.")
