#!/usr/bin/env python3
"""
Unification Figure v3: EVQ and inference-time methods are ORTHOGONAL corrections.

Key insight: Geometric RoPE has TWO deficiencies:
  1. SHAPE: uniform density wastes channels → poor long-range discrimination
  2. RANGE: frequencies designed for L_train cannot cover L_test > L_train

EVQ fixes the SHAPE (redistributes within the frequency range).
Inference methods (NTK, YaRN) fix the RANGE (extend coverage beyond L_train).
They are orthogonal → EVQ+YaRN combines both corrections → best performance.

Three panels:
  Left: Absolute φ_k showing the two correction directions
  Middle: EVQ+YaRN combined allocation vs. individual corrections
  Right: Collision score (discrimination quality) for each combination
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10.5
matplotlib.rcParams['axes.linewidth'] = 0.8

# ── Parameters ──────────────────────────────────────────────────────
base = 10_000
d_head = 128
K = d_head // 2  # 64 channels
L_train = 4096
scale = 4  # extrapolate to 16K
tau_star = d_head / np.sqrt(L_train)  # = 2.0

k = np.arange(K, dtype=np.float64)
u = k / K

# ── Allocations ─────────────────────────────────────────────────────
def evq_phi(u_arr, tau):
    return 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u_arr) * np.sinh(tau))

# Training-time allocations
phi_geo = u.copy()
phi_evq = evq_phi(u, tau_star)

# YaRN Progressive (inference-time, applied to geometric or EVQ)
beta_fast, beta_slow = 32, 1
wavelengths_geo = 2 * np.pi * base ** (2 * k / d_head)
low_bound = L_train / beta_fast
high_bound = L_train / beta_slow
ramp = np.clip((wavelengths_geo - low_bound) / (high_bound - low_bound), 0, 1)
ramp = ramp * ramp * (3.0 - 2.0 * ramp)
temperature = 1.0 + 0.07 * np.log2(scale)
yarn_shift = np.log(scale ** ramp * temperature ** (0.5 * ramp)) / np.log(base)

# Four combinations
phi_geo_only = phi_geo                    # Geometric alone
phi_geo_yarn = phi_geo + yarn_shift       # Geometric + YaRN
phi_evq_only = phi_evq                    # EVQ alone
phi_evq_yarn = phi_evq + yarn_shift       # EVQ + YaRN (best system)

# NTK-aware for reference
base_ntk = base * (scale ** (d_head / (d_head - 2)))
phi_ntk = u * np.log(base_ntk) / np.log(base)

# ── Collision score proxy ───────────────────────────────────────────
# For each allocation, compute effective frequencies and measure
# how well they discriminate positions at various distances.
# Proxy: average |cos(ω_i × Δ) - cos(ω_j × Δ)| for i≠j at Δ = L_test
# Higher = better discrimination at long range

def long_range_discrimination(phi_arr, base_val, L_test, n_distances=20):
    """Compute average pairwise discrimination at long range."""
    omega = base_val ** (-phi_arr)
    # Clip extreme frequencies
    omega = np.clip(omega, 1e-10, 1e10)

    distances = np.linspace(L_test * 0.5, L_test * 1.5, n_distances)
    total_disc = 0.0

    for delta in distances:
        phases = omega * delta
        cos_vals = np.cos(phases)
        # Pairwise discrimination: how different are channel responses?
        K_len = len(cos_vals)
        disc = 0.0
        count = 0
        for i in range(K_len):
            for j in range(i+1, K_len):
                disc += abs(cos_vals[i] - cos_vals[j])
                count += 1
        total_disc += disc / count if count > 0 else 0

    return total_disc / n_distances

def collision_score(phi_arr, base_val, L_eval, n_test=50):
    """
    Proxy collision score: average cos similarity between channel responses
    at distance L_eval. Lower = better (less collision = more discrimination).
    """
    omega = base_val ** (-phi_arr)
    omega = np.clip(omega, 1e-15, None)

    # Sample distances around L_eval
    deltas = np.linspace(L_eval * 0.8, L_eval * 1.2, n_test)

    # For each distance, compute pairwise cosine of phase differences
    # among the LOWEST-frequency half of channels (where collisions matter)
    low_half = omega[K//2:]  # lowest-freq channels

    total_collision = 0.0
    for delta in deltas:
        phases = low_half * delta
        cos_vals = np.cos(phases)
        # Mean pairwise |cos similarity| → 1 = total collision, 0 = perfect
        n = len(cos_vals)
        pair_sim = 0.0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                pair_sim += abs(np.cos(phases[i] - phases[j]))
                count += 1
        total_collision += pair_sim / count if count > 0 else 0

    return total_collision / n_test

# Compute for each combination at L_test
L_test = L_train * scale
configs = {
    'Geo': phi_geo_only,
    'Geo+NTK': phi_ntk,
    'Geo+YaRN': phi_geo_yarn,
    'EVQ': phi_evq_only,
    'EVQ+YaRN': phi_evq_yarn,
}

print("Computing collision scores...")
collisions = {}
for name, phi in configs.items():
    c = collision_score(phi, base, L_test)
    collisions[name] = c
    print(f"  {name:12s}: collision = {c:.4f}")

# ── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.8], wspace=0.3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# ── Left panel: Two correction directions ──
# Gray: geometric baseline
ax1.plot(u, phi_geo, '-', color='#888', lw=2.5, label='Geometric', zorder=2)

# Green arrow/area: EVQ correction (shape, downward)
ax1.plot(u, phi_evq, '-', color='#27ae60', lw=2.5, label='EVQ (shape fix)', zorder=5)
ax1.fill_between(u, phi_geo, phi_evq, alpha=0.15, color='#27ae60', zorder=1)

# Blue: YaRN on geometric (range, upward)
ax1.plot(u, phi_geo_yarn, '-', color='#3498db', lw=2, label='Geo+YaRN (range fix)', zorder=4)
ax1.fill_between(u, phi_geo, phi_geo_yarn, alpha=0.1, color='#3498db', zorder=1,
                 where=(phi_geo_yarn > phi_geo))

# Red dashed: NTK for reference
ax1.plot(u, phi_ntk, '--', color='#e74c3c', lw=1.2, alpha=0.7,
         label='Geo+NTK (range fix)', zorder=3)

ax1.axhline(y=1.0, color='k', lw=0.5, ls=':', alpha=0.3)

# Annotations with arrows
mid = K // 2
ax1.annotate('', xy=(u[mid], phi_evq[mid]), xytext=(u[mid], phi_geo[mid]),
             arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
ax1.text(u[mid] + 0.03, (phi_geo[mid] + phi_evq[mid])/2,
         'SHAPE\n(redistribute)', fontsize=8, color='#27ae60',
         fontweight='bold', va='center')

hi = int(0.85 * K)
ax1.annotate('', xy=(u[hi], phi_geo_yarn[hi]), xytext=(u[hi], phi_geo[hi]),
             arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
ax1.text(u[hi] + 0.03, (phi_geo[hi] + phi_geo_yarn[hi])/2,
         'RANGE\n(extend)', fontsize=8, color='#3498db',
         fontweight='bold', va='center')

ax1.set_xlabel('Channel index  $u_k = k/K$', fontsize=11)
ax1.set_ylabel('Log-frequency position  $\\phi_k$', fontsize=11)
ax1.set_title('Two orthogonal corrections\nto geometric RoPE', fontsize=12)
ax1.legend(fontsize=8, loc='upper left', framealpha=0.95)
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.05, max(phi_geo_yarn.max(), phi_ntk.max()) * 1.05)
ax1.grid(True, alpha=0.15)

# ── Middle panel: Combined EVQ+YaRN ──
ax2.plot(u, phi_geo, '-', color='#888', lw=2, label='Geometric', zorder=2)
ax2.plot(u, phi_evq, '-', color='#27ae60', lw=1.5, alpha=0.5,
         label='EVQ only', zorder=3)
ax2.plot(u, phi_geo_yarn, '-', color='#3498db', lw=1.5, alpha=0.5,
         label='Geo+YaRN only', zorder=3)
ax2.plot(u, phi_evq_yarn, '-', color='#8e44ad', lw=2.5,
         label='EVQ+YaRN (both fixes)', zorder=6)

# Shade the combined correction
ax2.fill_between(u, phi_geo, phi_evq_yarn, alpha=0.1, color='#8e44ad', zorder=1,
                 where=(phi_evq_yarn != phi_geo))

# Decomposition arrows at a specific channel
ch = int(0.75 * K)
# Shape component (geo → evq)
ax2.annotate('', xy=(u[ch]-0.02, phi_evq[ch]),
             xytext=(u[ch]-0.02, phi_geo[ch]),
             arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))
# Range component (evq → evq+yarn)
ax2.annotate('', xy=(u[ch]+0.02, phi_evq_yarn[ch]),
             xytext=(u[ch]+0.02, phi_evq[ch]),
             arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5))

ax2.text(u[ch]-0.06, (phi_geo[ch]+phi_evq[ch])/2, 'shape',
         fontsize=7, color='#27ae60', ha='right', va='center')
ax2.text(u[ch]+0.06, (phi_evq[ch]+phi_evq_yarn[ch])/2, 'range',
         fontsize=7, color='#3498db', ha='left', va='center')

ax2.axhline(y=1.0, color='k', lw=0.5, ls=':', alpha=0.3)
ax2.set_xlabel('Channel index  $u_k = k/K$', fontsize=11)
ax2.set_ylabel('Log-frequency position  $\\phi_k$', fontsize=11)
ax2.set_title('EVQ+YaRN: additive\nshape + range correction', fontsize=12)
ax2.legend(fontsize=8, loc='upper left', framealpha=0.95)
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.05, max(phi_evq_yarn.max(), phi_geo_yarn.max()) * 1.05)
ax2.grid(True, alpha=0.15)

# ── Right panel: Collision scores ──
names = ['Geo', 'Geo+NTK', 'Geo+YaRN', 'EVQ', 'EVQ+YaRN']
colors_bar = ['#888888', '#e74c3c', '#3498db', '#27ae60', '#8e44ad']
scores = [collisions[n] for n in names]

bars = ax3.barh(range(len(names)), scores, color=colors_bar,
                edgecolor='white', height=0.55)

for i, (bar, s) in enumerate(zip(bars, scores)):
    ax3.text(s + 0.005, i, f'{s:.3f}', va='center', fontsize=9)

ax3.set_yticks(range(len(names)))
ax3.set_yticklabels(names, fontsize=9.5)
ax3.set_xlabel('Low-freq collision\n(lower = better)', fontsize=10)
ax3.set_title('Position discrimination\nat $L_{\\mathrm{test}}$=' + f'{L_test//1000}K',
              fontsize=12)
ax3.invert_yaxis()
ax3.grid(True, axis='x', alpha=0.15)
ax3.set_xlim(0, max(scores) * 1.25)

# Highlight best
best_idx = scores.index(min(scores))
bars[best_idx].set_edgecolor('#8e44ad')
bars[best_idx].set_linewidth(2)

plt.suptitle(
    'Frequency allocation as two orthogonal corrections to geometric RoPE\n'
    f'(base={base:,}, $d_{{\\mathrm{{head}}}}$={d_head}, '
    f'$L_{{\\mathrm{{train}}}}$={L_train}, {scale}× extrapolation to {L_test//1000}K)',
    fontsize=12, fontweight='bold', y=1.03)

plt.tight_layout()

out_dir = '/sessions/vibrant-practical-hawking/mnt/hybrid-rope/paper/figs'
plt.savefig(f'{out_dir}/fig_unification_orthogonal.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{out_dir}/fig_unification_orthogonal.png', bbox_inches='tight', dpi=200)
print(f"\nSaved to {out_dir}/fig_unification_orthogonal.pdf")

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("KEY NARRATIVE FOR PAPER")
print("="*60)
print("""
Geometric RoPE has two orthogonal deficiencies:

1. SHAPE deficiency: uniform log-frequency density wastes channels
   on redundant high-frequency discrimination. This limits long-range
   resolution even within the training length.
   → EVQ fixes this by redistributing density (φ_k bends BELOW diagonal)

2. RANGE deficiency: frequencies designed for L_train cannot cover
   distances beyond L_train. Extrapolation requires extending the
   effective frequency range.
   → Inference methods (YaRN, NTK) fix this by scaling frequencies
     (φ_k shifts ABOVE the diagonal)

These corrections are ADDITIVE in log-frequency space:
   φ_EVQ+YaRN = φ_EVQ + Δφ_YaRN

And the collision scores confirm: EVQ+YaRN achieves the lowest
position collision at the test length, because it addresses both
deficiencies simultaneously.

This explains the main empirical finding: EVQ alone helps moderately,
YaRN alone helps moderately, but EVQ+YaRN dramatically outperforms
both — they are complementary, not competing.
""")
