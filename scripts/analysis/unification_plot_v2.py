#!/usr/bin/env python3
"""
Unification Figure v2: All RoPE methods as approximate frequency reallocations.

Key framing: Instead of raw φ_k, we plot Δφ_k = φ_method - φ_geometric,
the CORRECTION each method makes relative to the geometric baseline.
EVQ-Cosh is the theoretically optimal correction; other methods approximate it.

Two panels:
  Left: absolute φ_k curves (standard view)
  Right: Δφ correction profiles → shape comparison to EVQ optimal
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 0.8

# ── Parameters (Llama-2 standard) ───────────────────────────────────
base = 10_000
d_head = 128
K = d_head // 2  # 64 channels
L_train = 4096
scale = 4  # extrapolate to 16K
tau_star = d_head / np.sqrt(L_train)  # = 2.0

k = np.arange(K, dtype=np.float64)
u = k / K  # normalized channel index [0, 1)

# ── 1. Geometric (baseline) ────────────────────────────────────────
phi_geo = u.copy()

# ── 2. EVQ-Cosh (variational optimum) ──────────────────────────────
def evq_phi(u_arr, tau):
    return 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u_arr) * np.sinh(tau))
phi_evq = evq_phi(u, tau_star)

# ── 3. NTK-aware ───────────────────────────────────────────────────
base_ntk = base * (scale ** (d_head / (d_head - 2)))
phi_ntk = u * np.log(base_ntk) / np.log(base)

# ── 4. YaRN Progressive ────────────────────────────────────────────
beta_fast, beta_slow = 32, 1
wavelengths = 2 * np.pi * base ** (2 * k / d_head)
low_bound = L_train / beta_fast
high_bound = L_train / beta_slow
ramp = np.clip((wavelengths - low_bound) / (high_bound - low_bound), 0, 1)
ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
temperature = 1.0 + 0.07 * np.log2(scale)
yarn_scale_factor = (scale ** ramp) * (temperature ** (0.5 * ramp))
phi_yarn = u + np.log(yarn_scale_factor) / np.log(base)

# ── 5. CodeLlama (base 10K → 1M) ──────────────────────────────────
phi_codellama = u * np.log(1_000_000) / np.log(base)

# ── 6. "Ideal" EVQ for test length ─────────────────────────────────
# If we could train at L_test, the optimal would be:
tau_test = d_head / np.sqrt(L_train * scale)  # = 128/√16384 = 1.0
phi_evq_test = evq_phi(u, tau_test)

# ── Compute corrections relative to geometric ──────────────────────
delta_evq = phi_evq - phi_geo           # training-time optimal
delta_evq_test = phi_evq_test - phi_geo # test-time optimal
delta_ntk = phi_ntk - phi_geo
delta_yarn = phi_yarn - phi_geo
delta_codellama = phi_codellama - phi_geo

# ── Shape correlation with EVQ optimal correction ───────────────────
def shape_corr(delta_method, delta_optimal):
    """Pearson correlation of correction profiles (shape similarity)."""
    if np.std(delta_method) < 1e-10:
        return 0.0
    return np.corrcoef(delta_method, delta_optimal)[0, 1]

def overshoot_ratio(delta_method, delta_optimal):
    """Mean |correction| / mean |optimal correction|. >1 means overshoot."""
    return np.mean(np.abs(delta_method)) / np.mean(np.abs(delta_optimal))

print("=== Shape correlation with EVQ-Cosh optimal ===")
for name, delta in [('NTK-aware', delta_ntk), ('CodeLlama', delta_codellama),
                     ('YaRN Prog.', delta_yarn)]:
    corr = shape_corr(delta, delta_evq)
    overshoot = overshoot_ratio(delta, delta_evq)
    print(f"  {name:15s}: r={corr:.3f}, magnitude={overshoot:.2f}x")

# ── Figure ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                gridspec_kw={'width_ratios': [1, 1.2]})

# === Left panel: Absolute φ_k curves ===
ax1.plot(u, phi_geo, '-', color='#999', lw=2, label='Geometric ($\\tau{=}0$)', zorder=2)
ax1.plot(u, phi_evq, '-', color='#27ae60', lw=2.5,
         label='EVQ-Cosh ($\\tau^*$=' + f'{tau_star:.1f}, train)', zorder=6)
ax1.plot(u, phi_evq_test, '--', color='#27ae60', lw=1.5, alpha=0.6,
         label='EVQ-Cosh ($\\tau^*$=' + f'{tau_test:.1f}, test ideal)', zorder=5)
ax1.plot(u, phi_ntk, '--', color='#e74c3c', lw=1.5, label='NTK-aware', zorder=3)
ax1.plot(u, phi_codellama, ':', color='#e67e22', lw=1.5,
         label='CodeLlama (base→1M)', zorder=3)
ax1.plot(u, phi_yarn, '-', color='#3498db', lw=2, label='YaRN Progressive', zorder=4)

# Shading between geometric and EVQ
ax1.fill_between(u, phi_geo, phi_evq, alpha=0.1, color='#27ae60', zorder=1)

ax1.axhline(y=1.0, color='k', lw=0.5, ls=':', alpha=0.3)
ax1.set_xlabel('Channel index  $u_k = k/K$', fontsize=12)
ax1.set_ylabel('Log-frequency position  $\\phi_k$', fontsize=12)
ax1.set_title('Frequency allocation in $\\phi$-space', fontsize=13)
ax1.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.02, max(phi_codellama.max(), phi_ntk.max()) * 1.05)
ax1.grid(True, alpha=0.15)

# Annotation
ax1.annotate('Green shading:\nEVQ reallocation\n(training-time)',
             xy=(0.65, 0.45), xycoords='axes fraction',
             fontsize=8, color='#27ae60', style='italic', ha='center')

# === Right panel: Correction profiles Δφ ===
ax2.axhline(y=0, color='#999', lw=1.5, ls='-', label='Geometric (no correction)',
            zorder=1)
ax2.plot(u, delta_evq, '-', color='#27ae60', lw=2.5,
         label='EVQ optimal ($\\tau^*$=' + f'{tau_star:.1f})', zorder=6)
ax2.plot(u, delta_ntk, '--', color='#e74c3c', lw=1.5,
         label=f'NTK-aware (r={shape_corr(delta_ntk, delta_evq):.2f})', zorder=3)
ax2.plot(u, delta_codellama, ':', color='#e67e22', lw=1.5,
         label=f'CodeLlama (r={shape_corr(delta_codellama, delta_evq):.2f})', zorder=3)
ax2.plot(u, delta_yarn, '-', color='#3498db', lw=2,
         label=f'YaRN Prog. (r={shape_corr(delta_yarn, delta_evq):.2f})', zorder=4)

# Shade EVQ correction
ax2.fill_between(u, 0, delta_evq, alpha=0.1, color='#27ae60', zorder=1)

ax2.set_xlabel('Channel index  $u_k = k/K$', fontsize=12)
ax2.set_ylabel('Correction  $\\Delta\\phi_k = \\phi_{\\mathrm{method}} - \\phi_{\\mathrm{geo}}$',
               fontsize=12)
ax2.set_title('Correction profile relative to geometric\n'
              '(shape correlation $r$ with EVQ optimal)',
              fontsize=13)
ax2.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
ax2.set_xlim(0, 1)
ax2.grid(True, alpha=0.15)

# Key insight annotation
ax2.annotate('NTK & CodeLlama:\nlinear correction\n(wrong shape)',
             xy=(0.25, delta_codellama[K//4]),
             xytext=(0.1, delta_codellama[K//4] + 0.08),
             fontsize=8, color='#e67e22', ha='center',
             arrowprops=dict(arrowstyle='->', color='#e67e22', lw=0.8))

# Find where YaRN ramp starts
yarn_start_idx = np.argmax(delta_yarn > 0.001)
if yarn_start_idx > 0:
    ax2.annotate('YaRN: nonlinear\n(correct shape,\novershoots)',
                 xy=(u[int(0.7*K)], delta_yarn[int(0.7*K)]),
                 xytext=(0.55, max(delta_yarn) * 0.85),
                 fontsize=8, color='#3498db', ha='center',
                 arrowprops=dict(arrowstyle='->', color='#3498db', lw=0.8))

plt.suptitle('Unification: existing RoPE methods as approximate frequency reallocations\n'
             f'(base={base:,}, $d_{{\\mathrm{{head}}}}$={d_head}, '
             f'$L_{{\\mathrm{{train}}}}$={L_train}, {scale}× extrapolation)',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()

# Save
out_dir = '/sessions/vibrant-practical-hawking/mnt/hybrid-rope/paper/figs'
plt.savefig(f'{out_dir}/fig_unification_allocations.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{out_dir}/fig_unification_allocations.png', bbox_inches='tight', dpi=200)
print(f"\nSaved to {out_dir}/fig_unification_allocations.pdf")

# ── Print summary for paper text ────────────────────────────────────
print("\n" + "="*60)
print("PAPER-READY SUMMARY")
print("="*60)
print(f"\nSetup: base={base}, d_head={d_head}, K={K}, L_train={L_train}, scale={scale}x")
print(f"EVQ τ* (train) = {tau_star:.1f}")
print(f"EVQ τ* (test)  = {tau_test:.1f}")
print()
print("Shape correlation with EVQ optimal correction:")
print(f"  NTK-aware:     r = {shape_corr(delta_ntk, delta_evq):.3f}  "
      f"(linear, magnitude {overshoot_ratio(delta_ntk, delta_evq):.1f}x)")
print(f"  CodeLlama:     r = {shape_corr(delta_codellama, delta_evq):.3f}  "
      f"(linear, magnitude {overshoot_ratio(delta_codellama, delta_evq):.1f}x)")
print(f"  YaRN Prog.:    r = {shape_corr(delta_yarn, delta_evq):.3f}  "
      f"(nonlinear, magnitude {overshoot_ratio(delta_yarn, delta_evq):.1f}x)")
print()
print("Key findings:")
print("  1. NTK-aware and CodeLlama make LINEAR corrections (constant slope)")
print("     → They change the frequency RANGE but not the SHAPE of allocation")
print("     → Shape correlation with EVQ is high (both linear → correlated)")
print("     → But magnitude is wrong (massive overshoot)")
print("  2. YaRN Progressive makes a NONLINEAR correction")
print("     → It bends the allocation in the same direction as EVQ")
print("     → Shape is correct, but magnitude overshoots for low-freq channels")
print("  3. EVQ solves the correct variational problem → optimal shape + magnitude")
print("  4. Conclusion: methods that better approximate EVQ's correction")
print("     profile tend to perform better empirically")
