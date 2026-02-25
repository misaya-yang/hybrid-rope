import numpy as np
from scipy.integrate import quad
from scipy.special import sici
import matplotlib.pyplot as plt
import os

# Parameters
d = 128
L_val = 16384
gamma = 1.0

phi_vals = np.linspace(0.001, 0.999, 100) # Avoid exact 0/1 for stability

out_dir = r"E:\rope\hybrid-rope\results\theory_validation"
os.makedirs(out_dir, exist_ok=True)

print("Starting Numerical Validation...")

def D_prior(Delta, gamma_val=1.0, L=L_val):
    # Normalized power-law prior
    if gamma_val == 1.0:
        Z = np.log(L)
    else:
        Z, _ = quad(lambda x: x**(-gamma_val), 1, L)
    return (Delta**(-gamma_val)) / Z

def run_e_diag_validation(b):
    print(f"\n--- Validating E_diag for b={b} ---")
    
    def e_diag_exact(phi):
        freq = 2 * (b**(-phi))
        def integrand(Delta):
            return D_prior(Delta, gamma_val=1.0, L=L_val) * np.cos(freq * Delta)
        integral_val, _ = quad(integrand, 1, L_val, limit=200)
        return 0.5 * (1.0 + integral_val)

    def e_diag_ci(phi):
        freq = 2 * (b**(-phi))
        ci_L, _ = sici(freq * L_val)
        ci_1, _ = sici(freq)
        return 0.5 + (1.0 / (2 * np.log(L_val))) * (ci_L - ci_1)

    def e_diag_approx_bulk(phi):
        A = 0.5
        B = np.log(b) / (2 * np.log(L_val))
        return A + B * phi

    exact_vals = [e_diag_exact(p) for p in phi_vals]
    approx_ci_vals = [e_diag_ci(p) for p in phi_vals]
    approx_bulk_vals = [e_diag_approx_bulk(p) for p in phi_vals]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(phi_vals, exact_vals, label='Exact $E_{\\text{diag}}(\\phi)$ (Numerical Integral)', color='black', linewidth=3)
    plt.plot(phi_vals, approx_ci_vals, label='Theorem 2 Exact Formula (Ci functions)', linestyle='--', color='blue', linewidth=2)
    plt.plot(phi_vals, approx_bulk_vals, label=f'Bulk Approximation (Linear $A+B\\phi$)', linestyle=':', color='red', linewidth=2)

    plt.title(f'Validation of Diagonal Form $E_{{\\text{{diag}}}}$ & Convexity (d={d}, b={b}, L={L_val})', fontsize=14)
    plt.xlabel('Normalized Frequency Coordinate $\\phi$', fontsize=12)
    plt.ylabel('Energy Potential $E(\\phi)$', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    filename = os.path.join(out_dir, f'e_diag_validation_b{b}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filename}")

    # Error checking
    max_diff_ci = np.max(np.abs(np.array(exact_vals) - np.array(approx_ci_vals)))
    print(f"Max absolute difference between Numerical Exact and Theorem 2 Ci Formula: {max_diff_ci:.6f}")
    
    # Check max difference at boundaries vs bulk for Taylor approx
    max_diff_bulk = np.max(np.abs(np.array(exact_vals) - np.array(approx_bulk_vals)))
    print(f"Max absolute difference between Numerical Exact and Linear Bulk Approx: {max_diff_bulk:.6f}")
    
    return exact_vals


exact_vals_10k = run_e_diag_validation(10000)

print("\n--- Validating O(1/ln b) Residuals ---")

# Compute full C[rho, D] for uniform rho=1 vs \int (rho^2 * E_diag)
def compute_residual(b_val):
    # Full Integral: \int_1^L D(Delta) * [ \int_0^1 \cos(b^{-phi} \Delta) d\phi ]^2 d\Delta
    def inner_int(Delta):
        def cos_phi(phi):
            return np.cos((b_val**(-phi)) * Delta)
        inner_val, _ = quad(cos_phi, 0, 1, limit=100)
        return inner_val
    
    def full_integrand(Delta):
        return D_prior(Delta, 1.0, L_val) * (inner_int(Delta)**2)
        
    full_c, _ = quad(full_integrand, 1, L_val, limit=100)
    
    # E_diag Integral
    def e_diag_exact_b(phi):
        freq = 2 * (b_val**(-phi))
        def integrand(Delta):
            return D_prior(Delta, 1.0, L_val) * np.cos(freq * Delta)
        integral_val, _ = quad(integrand, 1, L_val, limit=200)
        return 0.5 * (1.0 + integral_val)

    test_phi = np.linspace(0.001, 0.999, 100)
    test_ediag = [e_diag_exact_b(p) for p in test_phi]
    e_diag_int = np.trapezoid(test_ediag, test_phi)
    
    residual = np.abs(full_c - e_diag_int)
    pred_bound = 1.0 / np.log(b_val)
    
    print(f"b={b_val:<7} | Full C: {full_c:.4f} | E_diag_int: {e_diag_int:.4f} | ")
    print(f"Residual R = |C - E_diag| = {residual:.6f}")
    print(f"Theoretical Bound 1/ln(b) = {pred_bound:.6f}")
    print(f"R / Bound = {residual/pred_bound:.4f}  <-- Ratio should be strictly controlled & shrinking.")
    
    return residual, pred_bound

b_test_vals = [1000, 10000, 100000, 500000]
residuals = []
bounds = []

for base in b_test_vals:
    r, bnd = compute_residual(base)
    residuals.append(r)
    bounds.append(bnd)

# Plot residuals vs bounds
plt.figure(figsize=(8, 5))
plt.plot(np.log10(b_test_vals), residuals, 'o-', label='Empirical Cross-Term Residual $R$', color='red', linewidth=2)
plt.plot(np.log10(b_test_vals), bounds, 's--', label='Theoretical Bound $\\mathcal{O}(1/\\ln b)$', color='black', linewidth=2)
plt.xlabel('$\\log_{10}(b)$ (Base Scale)')
plt.ylabel('Error Magnitude')
plt.title('Broadband Diagonal Approximation Residual Decay', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

filename_res = os.path.join(out_dir, 'residual_decay.png')
plt.savefig(filename_res, dpi=300, bbox_inches='tight')
print(f"Residual plot saved to {filename_res}")
print("\nValidation Script Complete!")
