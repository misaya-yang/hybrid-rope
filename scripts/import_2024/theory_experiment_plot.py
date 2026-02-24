#!/usr/bin/env python3
"""
Theory-Experiment Comparison Plot for Hybrid-RoPE paper.

Generates a figure showing:
  - Theoretical optimal ρ*(ϕ) derived from corrected E_diag with fitted γ
  - Actual frequency allocations of Geometric, Sigmoid, Anchored-Sigmoid

This is the single most persuasive visual in the paper:
it connects theory to experiment on the same axes.

Usage:
    python theory_experiment_plot.py [--gamma 1.03] [--base 10000] [--L 16384] [--d 128]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import sici  # Ci(z) from SciPy special functions
import argparse
import json
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Euler-Mascheroni constant
GAMMA_EM = 0.5772156649


def _trapz(y, x):
    """NumPy compatibility for versions that removed np.trapz."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def cosine_integral(z):
    """Compute Ci(z) with SciPy's sici implementation."""
    if z <= 0:
        return float('-inf')
    _, ci = sici(z)
    return float(ci)


def compute_Dhat_exact(phi, b, L):
    """Compute D̂_b(2b^{-ϕ}) using exact Ci formula (Eq. 13)."""
    xi = 2.0 * b**(-phi)
    xi_L = xi * L
    
    ci_xi = cosine_integral(xi)
    ci_xiL = cosine_integral(xi_L)
    
    return (ci_xiL - ci_xi) / np.log(L)


def compute_Dhat_corrected(phi, b, L):
    """
    Compute D̂_b using CORRECTED asymptotics:
    - Ci(ξL) ≈ 0 for ξL >> 1
    - Ci(ξ) ≈ γ + ln(ξ) for ξ << 1
    """
    return (phi * np.log(b) - GAMMA_EM - np.log(2)) / np.log(L)


def compute_Ediag(phi, b, L, use_exact=True):
    """Compute E_diag(ϕ) = ½(1 + D̂_b(2b^{-ϕ}))"""
    if use_exact:
        Dhat = compute_Dhat_exact(phi, b, L)
    else:
        Dhat = compute_Dhat_corrected(phi, b, L)
    return 0.5 * (1.0 + Dhat)


def compute_rho_star_diagonal(phi_array, b, L, use_exact=True):
    """
    Compute theoretical ρ*(ϕ) = 1/E_diag(ϕ) / ∫ 1/E_diag dϕ
    (Inverse law, Lemma 1)
    """
    E_vals = np.array([compute_Ediag(phi, b, L, use_exact) for phi in phi_array])
    
    # Clamp to avoid division by zero
    E_vals = np.maximum(E_vals, 1e-6)
    
    inv_E = 1.0 / E_vals
    
    # Normalize: ∫ρ dϕ = 1 (trapezoidal)
    norm = _trapz(inv_E, phi_array)
    return inv_E / norm, E_vals


def compute_rho_cosh(phi_array):
    """
    Compute the exact full-functional solution ρ*_full ∝ cosh(1-ϕ).
    Normalized so ∫ρ dϕ = 1.
    """
    rho = np.cosh(1.0 - phi_array)
    norm = _trapz(rho, phi_array)
    return rho / norm


def geometric_allocation(phi_array, d):
    """
    Standard geometric RoPE: ρ ≡ 1 (uniform density).
    Returns normalized ρ and corresponding ω_k values.
    """
    return np.ones_like(phi_array)


def sigmoid_allocation(phi_array, base, d, k_param=None, x0_param=None):
    """
    Sigmoid RoPE schedule.
    
    g(ϕ) = ϕ + A·σ(k(ϕ - x₀)) where σ is the sigmoid function.
    ρ(ϕ) = g'(ϕ) = 1 + A·k·σ(k(ϕ-x₀))·(1-σ(k(ϕ-x₀)))
    
    Default parameters from the paper's sigmoid schedule.
    Adjust k_param and x0_param to match your actual implementation.
    """
    if k_param is None:
        k_param = 8.0   # steepness - adjust to match your code
    if x0_param is None:
        x0_param = 0.5  # midpoint - adjust to match your code
    
    sig = 1.0 / (1.0 + np.exp(-k_param * (phi_array - x0_param)))
    
    # Density is derivative of the warp function
    rho = 1.0 + k_param * sig * (1.0 - sig) * 0.5  # 0.5 = amplitude
    
    # Normalize
    norm = _trapz(rho, phi_array)
    return rho / norm


def anchored_sigmoid_allocation(phi_array, alpha=0.2, k_param=None, x0_param=None):
    """
    Anchored sigmoid: anchors high-frequency end, applies sigmoid to rest.
    
    For ϕ ∈ [0, α]: ρ(ϕ) = 1 (anchored, preserving high-freq resolution)
    For ϕ ∈ [α, 1]: ρ(ϕ) = sigmoid warp
    
    Adjust parameters to match your actual implementation.
    """
    if k_param is None:
        k_param = 8.0
    if x0_param is None:
        x0_param = 0.6
    
    rho = np.ones_like(phi_array)
    
    mask = phi_array > alpha
    phi_shifted = (phi_array[mask] - alpha) / (1.0 - alpha)  # rescale to [0,1]
    sig = 1.0 / (1.0 + np.exp(-k_param * (phi_shifted - 0.5)))
    rho[mask] = 1.0 + k_param * sig * (1.0 - sig) * 0.4
    
    norm = _trapz(rho, phi_array)
    return rho / norm


def density_from_inv_freq(inv_freq, phi_grid):
    """
    Convert discrete inv_freq values to a continuous density rho(phi).
    Uses normalized log-frequency coordinate phi in [0, 1].
    """
    omega = np.asarray(inv_freq, dtype=float).reshape(-1)
    omega = omega[np.isfinite(omega) & (omega > 0)]
    if omega.size < 2:
        return None

    phi_raw = -np.log(omega)
    phi_sorted = np.sort(phi_raw)
    span = phi_sorted[-1] - phi_sorted[0]
    if span <= 1e-12:
        return np.ones_like(phi_grid)

    # Normalize to [0, 1] so different absolute rope theta are comparable.
    phi_norm = (phi_sorted - phi_sorted[0]) / span
    u = np.linspace(0.0, 1.0, len(phi_norm))

    dphi_du = np.gradient(phi_norm, u)
    rho_samples = 1.0 / np.clip(dphi_du, 1e-8, None)

    rho = np.interp(phi_grid, phi_norm, rho_samples, left=rho_samples[0], right=rho_samples[-1])
    rho = np.maximum(rho, 1e-8)
    rho /= _trapz(rho, phi_grid)
    return rho


def _load_inv_freq_from_pt(path):
    try:
        import torch
    except Exception:
        return None
    try:
        obj = torch.load(path, map_location="cpu")
        if hasattr(obj, "detach") and hasattr(obj, "cpu"):
            return obj.detach().cpu().numpy().reshape(-1)
        if isinstance(obj, dict):
            for v in obj.values():
                if hasattr(v, "detach") and hasattr(v, "cpu"):
                    return v.detach().cpu().numpy().reshape(-1)
    except Exception:
        return None
    return None


def load_actual_frequencies_from_repo(method, d=128, base=10000, actual_invfreq_json=None, results_dir=None):
    """
    If you have the actual inv_freq tensors saved, load them here.
    
    Otherwise, this function generates schematic allocations
    based on the method's design principles.
    
    *** ADAPT THIS TO LOAD YOUR ACTUAL inv_freq BUFFERS ***
    
    To extract actual frequencies from a checkpoint:
        import torch
        ckpt = torch.load("path/to/adapter/model.safetensors")
        inv_freq = ckpt["model.layers.0.self_attn.rotary_emb.inv_freq"]
        omega = inv_freq.numpy()
        # Convert to phi: phi_k = -log(omega_k) / log(base)
        phi = -np.log(omega) / np.log(base)
    """
    script_dir = Path(__file__).resolve().parent

    if actual_invfreq_json:
        json_candidates = [Path(actual_invfreq_json)]
    else:
        json_candidates = [
            script_dir / "real_inv_freq_20260223.json",
            script_dir / "real_inv_freq.json",
        ]

    for jp in json_candidates:
        if jp.exists():
            try:
                with open(jp) as f:
                    data = json.load(f)
                vals = data.get(method)
                if isinstance(vals, list) and vals:
                    return np.asarray(vals, dtype=float)
            except Exception:
                pass

    if results_dir:
        pt_path = Path(results_dir) / method / "artifacts" / "custom_inv_freq.pt"
        if pt_path.exists():
            vals = _load_inv_freq_from_pt(str(pt_path))
            if vals is not None:
                return vals

    return None


def maybe_override_with_real_density(method, fallback_rho, phi_grid, args):
    """Load real inv_freq for a method and convert to rho(phi) when available."""
    inv = load_actual_frequencies_from_repo(
        method,
        d=args.d,
        base=args.base,
        actual_invfreq_json=args.actual_invfreq_json,
        results_dir=args.results_dir,
    )
    if inv is None:
        print(f"  Using schematic {method.replace('_', '-')} allocation (real inv_freq not found)")
        return fallback_rho

    rho_real = density_from_inv_freq(inv, phi_grid)
    if rho_real is None:
        print(f"  Loaded {method} inv_freq but failed to build density; using schematic allocation")
        return fallback_rho

    print(f"  Loaded real inv_freq for {method} ({len(inv)} values)")
    return rho_real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=1.03,
                        help="Power-law exponent from EXP_ATTN_D_DIST (default: 1.03)")
    parser.add_argument("--base", type=float, default=10000,
                        help="RoPE base (default: 10000)")
    parser.add_argument("--L", type=int, default=16384,
                        help="Context length (default: 16384)")
    parser.add_argument("--d", type=int, default=128,
                        help="Head dimension (default: 128)")
    parser.add_argument("--output", type=str, default="theory_experiment_comparison.pdf",
                        help="Output file path")
    parser.add_argument("--output_png", type=str, default="theory_experiment_comparison.png",
                        help="Output PNG file path")
    parser.add_argument("--actual_invfreq_json", type=str, default=None,
                        help="Optional JSON file containing real inv_freq arrays for methods")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Optional results root for loading method/artifacts/custom_inv_freq.pt")
    args = parser.parse_args()
    
    # ============================================================
    # Compute theoretical predictions
    # ============================================================
    phi = np.linspace(0.01, 0.99, 500)  # avoid boundary singularities
    
    print("Computing theoretical ρ*(ϕ) from corrected E_diag...")
    print(f"  Parameters: b={args.base}, L={args.L}, γ={args.gamma}")
    
    # 1. Diagonal inverse law (Theorem 2, corrected)
    rho_diag, E_diag = compute_rho_star_diagonal(phi, args.base, args.L, use_exact=True)
    
    # 2. Full functional cosh solution (Appendix E.1)
    rho_cosh = compute_rho_cosh(phi)
    
    # 3. Practical allocations
    rho_geo = geometric_allocation(phi, args.d)
    rho_sig = sigmoid_allocation(phi, args.base, args.d)
    rho_anc = anchored_sigmoid_allocation(phi)

    rho_sig = maybe_override_with_real_density("sigmoid", rho_sig, phi, args)
    rho_anc = maybe_override_with_real_density("anchored_sigmoid", rho_anc, phi, args)
    
    # ============================================================
    # Verify E_diag direction (sanity check)
    # ============================================================
    print(f"\n  E_diag(0.01) = {E_diag[0]:.4f}")
    print(f"  E_diag(0.99) = {E_diag[-1]:.4f}")
    print(f"  E_diag increasing? {E_diag[-1] > E_diag[0]}  ✓" if E_diag[-1] > E_diag[0] 
          else f"  E_diag increasing? {E_diag[-1] > E_diag[0]}  ✗ CHECK YOUR MATH")
    
    print(f"\n  ρ*_diag(0) / ρ*_diag(1) = {rho_diag[0]/rho_diag[-1]:.3f}")
    print(f"  ρ*_cosh(0) / ρ*_cosh(1) = {rho_cosh[0]/rho_cosh[-1]:.3f} = cosh(1) ≈ 1.543")
    
    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1.2, 1]})
    
    # --- Left panel: ρ(ϕ) comparison ---
    ax = axes[0]
    
    # Theoretical predictions (shaded region between diagonal and cosh)
    ax.fill_between(phi, rho_diag, rho_cosh, alpha=0.15, color='royalblue',
                    label='Theory band (diagonal → cosh)')
    ax.plot(phi, rho_diag, '--', color='royalblue', linewidth=1.5, alpha=0.8,
            label=r'$\rho^\star_{\mathrm{diag}} \propto 1/E_{\mathrm{diag}}$')
    ax.plot(phi, rho_cosh, '-', color='navy', linewidth=2,
            label=r'$\rho^\star_{\mathrm{full}} \propto \cosh(1-\phi)$')
    
    # Practical allocations
    ax.plot(phi, rho_geo, ':', color='gray', linewidth=1.5, alpha=0.7,
            label='Geometric (uniform)')
    ax.plot(phi, rho_sig, '-', color='orangered', linewidth=1.5, alpha=0.8,
            label='Sigmoid')
    ax.plot(phi, rho_anc, '-', color='darkgreen', linewidth=2,
            label='Anchored-Sigmoid')
    
    ax.set_xlabel(r'$\phi$ (log-frequency coordinate)', fontsize=12)
    ax.set_ylabel(r'$\rho(\phi)$ (spectral density)', fontsize=12)
    ax.set_title('(a) Theoretical vs. practical frequency allocations', fontsize=11)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    ax.set_xlim(0, 1)
    
    # Annotate high/low frequency
    ax.annotate('High freq\n(local)', xy=(0.05, 0.02), fontsize=8, color='gray',
                xycoords='axes fraction')
    ax.annotate('Low freq\n(global)', xy=(0.85, 0.02), fontsize=8, color='gray',
                xycoords='axes fraction')
    
    ax.grid(True, alpha=0.3)
    
    # --- Right panel: E_diag(ϕ) ---
    ax2 = axes[1]
    
    ax2.plot(phi, E_diag, '-', color='royalblue', linewidth=2,
             label=r'$E_{\mathrm{diag}}(\phi)$ (exact Ci)')
    
    # Show the corrected affine approximation
    E_affine = 0.5 * (1 + (phi * np.log(args.base) - GAMMA_EM - np.log(2)) / np.log(args.L))
    ax2.plot(phi, E_affine, '--', color='coral', linewidth=1.5, alpha=0.8,
             label=r'Affine approx. (corrected Eq.~14)')
    
    # Mark the 1/2 anchor
    ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.annotate(r'$\frac{1}{2}$ anchor', xy=(0.02, 0.5), fontsize=8, color='gray',
                va='bottom')
    
    ax2.set_xlabel(r'$\phi$ (log-frequency coordinate)', fontsize=12)
    ax2.set_ylabel(r'$E_{\mathrm{diag}}(\phi)$', fontsize=12)
    ax2.set_title(r'(b) Diagonal energy $E_{\mathrm{diag}}$ (increasing $\Rightarrow$ high-freq bias)',
                  fontsize=10)
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output)
    plt.savefig(args.output_png)
    print(f"\n  Figure saved to: {args.output}")
    print(f"  Figure saved to: {args.output_png}")
    
    # ============================================================
    # Print key numbers for paper
    # ============================================================
    print("\n" + "="*60)
    print("KEY NUMBERS FOR PAPER")
    print("="*60)
    print(f"  E_diag slope:  {(E_diag[-1]-E_diag[0])/(phi[-1]-phi[0]):.4f} per unit ϕ")
    print(f"  ρ*_diag ratio: {rho_diag[0]/rho_diag[-1]:.3f}  (high/low freq)")
    print(f"  ρ*_cosh ratio: {rho_cosh[0]/rho_cosh[-1]:.3f}  (= cosh(1))")
    print(f"  Theory band width at ϕ=0: {abs(rho_cosh[0]-rho_diag[0]):.3f}")
    print(f"  Theory band width at ϕ=1: {abs(rho_cosh[-1]-rho_diag[-1]):.3f}")
    
    # Check if anchored-sigmoid falls within the theory band
    in_band = np.all((rho_anc >= np.minimum(rho_diag, rho_cosh) * 0.8) & 
                     (rho_anc <= np.maximum(rho_diag, rho_cosh) * 1.2))
    print(f"  Anc-Sigmoid within 20% of theory band? {in_band}")
    
    print("\n  Suggested caption:")
    if in_band:
        print("  'Left: Theoretical optimal density (blue band between diagonal")
        print("   inverse law and full-functional cosh solution) compared with")
        print("   practical allocations. Anchored-sigmoid (green) closely tracks")
        print("   the theory band. Right: Corrected diagonal energy E_diag(ϕ)")
        print("   is monotonically increasing, confirming high-frequency bias.'")
    else:
        print("  'Left: Theoretical optimal density (blue band) compared with")
        print("   practical allocations. Anchored-sigmoid (green) follows the")
        print("   same monotonic trend but only partially overlaps the theory")
        print("   band under current settings. Right: Corrected diagonal energy")
        print("   E_diag(ϕ) is monotonically increasing.'")


if __name__ == "__main__":
    main()
