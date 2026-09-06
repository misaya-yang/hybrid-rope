#!/usr/bin/env python3
"""
Unification Figure (final): Two orthogonal corrections to geometric RoPE.
Clean 2-panel figure for the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 0.8

base = 10_000
d_head = 128
K = d_head // 2
L_train = 4096
scale = 4
tau_star = d_head / np.sqrt(L_train)  # 2.0

k = np.arange(K, dtype=np.float64)
u = k / K

def evq_phi(u_arr, tau):
    return 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u_arr) * np.sinh(tau))

phi_geo = u.copy()
phi_evq = evq_phi(u, tau_star)

# YaRN Progressive
wavelengths = 2 * np.pi * base ** (2 * k / d_head)
ramp = np.clip((wavelengths - L_train/32) / (L_train - L_train/32), 0, 1)
ramp = ramp * ramp * (3.0 - 2.0 * ramp)
temperature = 1.0 + 0.07 * np.log2(scale)
yarn_shift = np.log(scale**ramp * temperature**(0.5*ramp)) / np.log(base)

phi_geo_yarn = phi_geo + yarn_shift
phi_evq_yarn = phi_evq + yarn_shift

# NTK
base_ntk = base * scale**(d_head/(d_head-2))
phi_ntk = u * np.log(base_ntk) / np.log(base)

# ── Figure ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

# === Panel (a): Two orthogonal correction directions ===
ax1.plot(u, phi_geo, '-', color='#888', lw=2.5, label='Geometric ($\\tau{=}0$)', zorder=2)
ax1.plot(u, phi_evq, '-', color='#27ae60', lw=2.5,
         label='EVQ-Cosh ($\\tau^*{=}2.0$)', zorder=5)
ax1.plot(u, phi_geo_yarn, '-', color='#3498db', lw=2,
         label='Geo + YaRN', zorder=4)
ax1.plot(u, phi_ntk, '--', color='#e74c3c', lw=1.3, alpha=0.7,
         label='Geo + NTK-aware', zorder=3)

# Shading
ax1.fill_between(u, phi_geo, phi_evq, alpha=0.15, color='#27ae60',
                 zorder=1, label='_')
ax1.fill_between(u, phi_geo, phi_geo_yarn, alpha=0.08, color='#3498db',
                 where=(phi_geo_yarn > phi_geo), zorder=1, label='_')

# Direction arrows
mid = K // 2
ax1.annotate('', xy=(u[mid], phi_evq[mid]),
             xytext=(u[mid], phi_geo[mid]),
             arrowprops=dict(arrowstyle='->', color='#1a8a48', lw=2.5,
                             shrinkA=2, shrinkB=2))
ax1.text(u[mid]+0.04, (phi_geo[mid]+phi_evq[mid])/2,
         'Shape correction\n(redistribute within range)',
         fontsize=8.5, color='#1a8a48', fontweight='bold', va='center')

hi = int(0.82 * K)
ax1.annotate('', xy=(u[hi], phi_geo_yarn[hi]),
             xytext=(u[hi], phi_geo[hi]),
             arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2.5,
                             shrinkA=2, shrinkB=2))
ax1.text(u[hi]-0.04, (phi_geo[hi]+phi_geo_yarn[hi])/2 + 0.02,
         'Range correction\n(extend beyond $L_{\\mathrm{train}}$)',
         fontsize=8.5, color='#2980b9', fontweight='bold', va='center', ha='right')

ax1.axhline(y=1.0, color='k', lw=0.5, ls=':', alpha=0.3)
ax1.set_xlabel('Channel index  $u_k = k/K$', fontsize=12)
ax1.set_ylabel('Log-frequency position  $\\phi_k$', fontsize=12)
ax1.set_title('(a)  Two orthogonal deficiencies of geometric RoPE', fontsize=12)
ax1.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.05, max(phi_geo_yarn.max(), phi_ntk.max()) * 1.02)
ax1.grid(True, alpha=0.15)

# === Panel (b): Additive combination ===
ax2.plot(u, phi_geo, '-', color='#888', lw=2, label='Geometric', zorder=2)
ax2.plot(u, phi_evq, '-', color='#27ae60', lw=1.5, alpha=0.6,
         label='EVQ only (shape)', zorder=3)
ax2.plot(u, phi_geo_yarn, '-', color='#3498db', lw=1.5, alpha=0.6,
         label='Geo+YaRN only (range)', zorder=3)
ax2.plot(u, phi_evq_yarn, '-', color='#8e44ad', lw=2.8,
         label='EVQ + YaRN (shape $\\oplus$ range)', zorder=6)

ax2.fill_between(u, phi_geo, phi_evq_yarn, alpha=0.08, color='#8e44ad', zorder=1)

# Decomposition at a specific channel
ch = int(0.72 * K)
# Shape arrow
ax2.annotate('', xy=(u[ch]-0.015, phi_evq[ch]),
             xytext=(u[ch]-0.015, phi_geo[ch]),
             arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.8,
                             shrinkA=1, shrinkB=1))
# Range arrow
ax2.annotate('', xy=(u[ch]+0.015, phi_evq_yarn[ch]),
             xytext=(u[ch]+0.015, phi_evq[ch]),
             arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.8,
                             shrinkA=1, shrinkB=1))

ax2.text(u[ch]-0.05, (phi_geo[ch]+phi_evq[ch])/2,
         '$\\Delta\\phi_{\\mathrm{shape}}$',
         fontsize=9, color='#27ae60', ha='right', va='center')
ax2.text(u[ch]+0.05, (phi_evq[ch]+phi_evq_yarn[ch])/2,
         '$\\Delta\\phi_{\\mathrm{range}}$',
         fontsize=9, color='#3498db', ha='left', va='center')

# Key equation
ax2.text(0.5, 0.92,
         '$\\phi_{\\mathrm{EVQ+YaRN}} = \\phi_{\\mathrm{EVQ}} + \\Delta\\phi_{\\mathrm{YaRN}}$',
         fontsize=12, ha='center', va='top',
         transform=ax2.transAxes,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0e6f6',
                   edgecolor='#8e44ad', alpha=0.9))

ax2.axhline(y=1.0, color='k', lw=0.5, ls=':', alpha=0.3)
ax2.set_xlabel('Channel index  $u_k = k/K$', fontsize=12)
ax2.set_ylabel('Log-frequency position  $\\phi_k$', fontsize=12)
ax2.set_title('(b)  EVQ + YaRN: additive orthogonal corrections', fontsize=12)
ax2.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.05, max(phi_evq_yarn.max(), phi_geo_yarn.max()) * 1.02)
ax2.grid(True, alpha=0.15)

plt.tight_layout()

out = '/sessions/vibrant-practical-hawking/mnt/hybrid-rope/paper/figs'
plt.savefig(f'{out}/fig_unification_orthogonal.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{out}/fig_unification_orthogonal.png', bbox_inches='tight', dpi=200)
print(f"Saved to {out}/fig_unification_orthogonal.pdf/.png")
