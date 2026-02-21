#!/usr/bin/env python3
"""
Theoretical Validation: Broadband Diagonalization Assumption & Theorem 2
=====================================================================
Validate the theoretical claims using GPU-accelerated kernel matrix computation.

Mathematical Definitions:
- Variational kernel: K(φ_j, φ_k) = Σ_Δ D(Δ) · cos(ω_j Δ) · cos(ω_k Δ)
- Diagonal Dominance Ratio: r = Σ|diag(K)| / Σ|K|
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
d = 128
L = 16384
b = 10000.0
M = 256  # number of sampling points for φ

print(f"\nParameters: d={d}, L={L}, b={b}, M={M}")


def create_frequency_grid(b: float, M: int) -> torch.Tensor:
    """Create frequency grid ω = b^(-φ) for φ ∈ [0, 1]"""
    phi = torch.linspace(0, 1, M, dtype=torch.float64, device=device)
    omega = b ** (-phi)
    return omega


def create_distance_grid(L: int) -> torch.Tensor:
    """Create distance vector Δ ∈ [1, L]"""
    return torch.arange(1, L + 1, dtype=torch.float64, device=device)


def compute_kernel_matrix(omega: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Compute kernel matrix K(φ_j, φ_k) = Σ_Δ D(Δ) · cos(ω_j Δ) · cos(ω_k Δ)
    
    Using matrix form:
    - C[j, Δ] = cos(ω_j * Δ)
    - K = C @ diag(D) @ C^T
    """
    # Create distance vector
    Delta = create_distance_grid(L)
    
    # Compute feature matrix C: (M, L)
    # C[j, Δ] = cos(ω_j * Δ)
    omega_expanded = omega.unsqueeze(1)  # (M, 1)
    Delta_expanded = Delta.unsqueeze(0)  # (1, L)
    C = torch.cos(omega_expanded * Delta_expanded)  # (M, L)
    
    # K = C @ diag(D) @ C^T
    D_diag = torch.diag(D)
    K = C @ D_diag @ C.T
    
    return K


def diagonal_dominance_ratio(K: torch.Tensor) -> float:
    """Compute diagonal dominance ratio r = Σ|diag(K)| / Σ|K|"""
    diag = torch.diag(K).abs().sum()
    total = K.abs().sum()
    return (diag / total).item()


# Create frequency grid
omega = create_frequency_grid(b, M)
print(f"Frequency range: ω ∈ [{omega.min().item():.6f}, {omega.max().item():.6f}]")

# =============================================================================
# Prior 1: Uniform
# =============================================================================
print("\n" + "="*60)
print("Computing: Uniform Prior")
print("="*60)

D_uniform = torch.ones(L, dtype=torch.float64, device=device)
D_uniform = D_uniform / D_uniform.sum()  # Normalize

K_uniform = compute_kernel_matrix(omega, D_uniform)
r_uniform = diagonal_dominance_ratio(K_uniform)
print(f"Diagonal Dominance Ratio (Uniform): r = {r_uniform:.6f}")

# =============================================================================
# Prior 2: Power-law (1/Δ)
# =============================================================================
print("\n" + "="*60)
print("Computing: Power-law Prior (D(Δ) ∝ 1/Δ)")
print("="*60)

Delta = create_distance_grid(L)
D_powerlaw = 1.0 / Delta
D_powerlaw = D_powerlaw / D_powerlaw.sum()  # Normalize

K_powerlaw = compute_kernel_matrix(omega, D_powerlaw)
r_powerlaw = diagonal_dominance_ratio(K_powerlaw)
print(f"Diagonal Dominance Ratio (Power-law): r = {r_powerlaw:.6f}")

# =============================================================================
# Prior 3: Bimodal (two Gaussians at Δ=1 and Δ=L)
# =============================================================================
print("\n" + "="*60)
print("Computing: Bimodal Prior (two Gaussians)")
print("="*60)

sigma = 50.0  # standard deviation

# Gaussian centered at Δ=1
gauss1 = torch.exp(-((Delta - 1.0) ** 2) / (2 * sigma ** 2))

# Gaussian centered at Δ=L
gauss2 = torch.exp(-((Delta - float(L)) ** 2) / (2 * sigma ** 2))

D_bimodal = gauss1 + gauss2
D_bimodal = D_bimodal / D_bimodal.sum()  # Normalize

K_bimodal = compute_kernel_matrix(omega, D_bimodal)
r_bimodal = diagonal_dominance_ratio(K_bimodal)
print(f"Diagonal Dominance Ratio (Bimodal): r = {r_bimodal:.6f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("SUMMARY: Diagonal Dominance Ratios")
print("="*60)
print(f"Uniform:    r = {r_uniform:.6f}")
print(f"Power-law: r = {r_powerlaw:.6f}")
print(f"Bimodal:   r = {r_bimodal:.6f}")

# =============================================================================
# Plot 1: Kernel Heatmaps
# =============================================================================
print("\n" + "="*60)
print("Generating: kernel_diagonal_dominance.png")
print("="*60)

output_dir = Path("results/theoretical_validation")
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Convert to numpy for plotting
K_uniform_np = K_uniform.cpu().numpy()
K_powerlaw_np = K_powerlaw.cpu().numpy()
K_bimodal_np = K_bimodal.cpu().numpy()

# Plot heatmaps
vmin = min(K_uniform_np.min(), K_powerlaw_np.min(), K_bimodal_np.min())
vmax = max(K_uniform_np.max(), K_powerlaw_np.max(), K_bimodal_np.max())

sns.heatmap(K_uniform_np, ax=axes[0], cmap='viridis', 
            cbar=True, xticklabels=[], yticklabels=[],
            vmin=vmin, vmax=vmax)
axes[0].set_title(f'Uniform\nr = {r_uniform:.4f}')

sns.heatmap(K_powerlaw_np, ax=axes[1], cmap='viridis', 
            cbar=True, xticklabels=[], yticklabels=[],
            vmin=vmin, vmax=vmax)
axes[1].set_title(f'Power-law\nr = {r_powerlaw:.4f}')

sns.heatmap(K_bimodal_np, ax=axes[2], cmap='viridis', 
            cbar=True, xticklabels=[], yticklabels=[],
            vmin=vmin, vmax=vmax)
axes[2].set_title(f'Bimodal\nr = {r_bimodal:.4f}')

plt.tight_layout()
plt.savefig(output_dir / "kernel_diagonal_dominance.png", dpi=150, bbox_inches='tight')
print(f"Saved to {output_dir / 'kernel_diagonal_dominance.png'}")

# =============================================================================
# Plot 2: Theorem 2 Validation (Power-law diagonal)
# =============================================================================
print("\n" + "="*60)
print("Generating: theorem2_powerlaw_validation.png")
print("="*60)

# Extract diagonal: E_pow(φ) = diag(K_pow)
E_powerlaw = torch.diag(K_powerlaw).cpu().numpy()
phi = np.linspace(0, 1, M)

# Extract middle section φ ∈ [0.1, 0.9]
mask = (phi >= 0.1) & (phi <= 0.9)
phi_mid = phi[mask]
E_mid = E_powerlaw[mask]

# Linear fit: y = A + B * phi
A, B = np.polyfit(phi_mid, E_mid, 1)
print(f"Linear fit in φ ∈ [0.1, 0.9]: E = {A:.4f} + {B:.4f} * φ")
print(f"Slope B = {B:.6f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(phi, E_powerlaw, 'b-', linewidth=2, label='Numerical E_pow(φ)')
ax.plot(phi_mid, A + B * phi_mid, 'r--', linewidth=2, 
        label=f'Linear fit: y = {A:.3f} + {B:.3f}φ')

ax.set_xlabel('φ', fontsize=12)
ax.set_ylabel('E_pow(φ) = diag(K_pow)', fontsize=12)
ax.set_title(f'Theorem 2 Validation: Power-law Prior\nLinear fit slope B = {B:.4f}', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "theorem2_powerlaw_validation.png", dpi=150, bbox_inches='tight')
print(f"Saved to {output_dir / 'theorem2_powerlaw_validation.png'}")

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"1. Diagonal Dominance Ratios:")
print(f"   - Uniform:    r = {r_uniform:.6f}")
print(f"   - Power-law: r = {r_powerlaw:.6f}")
print(f"   - Bimodal:    r = {r_bimodal:.6f}")
print(f"\n2. Theorem 2 Linear Fit:")
print(f"   - Slope B = {B:.6f}")
print(f"\n3. Output files saved to: {output_dir}")
print("="*60)
