"""
Figure: EVQ vs GEO Advantage as a Function of Training Amount
=============================================================
Two key lines:
  - Blue: Raw extrapolation advantage (EVQ vs GEO, no YaRN)
  - Red: +YaRN composition advantage (EVQ+YaRN+FT vs GEO+YaRN+FT)

X-axis: Training regime (tokens / model_params ratio, or just labeled points)
Message: Raw EVQ advantage diminishes with more training, but composition advantage stays negative.

Data sources:
  - 8K model, 500M tokens (Phase 18a, 3-seed)
  - 4K model, 50% ckpt = 500M tokens (Phase 18b)
  - 4K model, 100% = 1B tokens (Phase 18b)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# =============================================================================
# DATA FROM PHASE 18 REPORT
# =============================================================================

# --- Panel A: 4K Model Training Progression (432M MLA, seed=42) ---
# X = training percentage (of 1B total)
# Y = % difference EVQ vs GEO at 2x extrapolation (8K for 4K model)

panel_a_labels = ['50%\n(500M)', '100%\n(1B)']
panel_a_x = [0.5, 1.0]

# Raw extrapolation: EVQ vs GEO @ 8K (4K model)
# 50%: EVQ 96.61 vs GEO 91.43 = +5.7%
# 100%: EVQ 85.85 vs GEO 77.26 = +11.1%
raw_extrap_4k = [+5.7, +11.1]

# +YaRN inference only @ 8K
# 50%: EVQ 40.74 vs GEO 42.06 = -3.1%
# 100%: EVQ 31.4 vs GEO 32.2 = -2.5% (approx from report)
yarn_inference_4k = [-3.1, -2.5]

# +YaRN+FT @ 8K (target length)
# 50%: PENDING
# 100%: EVQ 31.043 vs GEO 31.834 = -2.5%
yarn_ft_4k = [None, -2.5]

# --- Panel B: 8K Model (432M MLA, 3-seed, 500M tokens) ---
# Standalone comparison at 2x (16K)
# EVQ -31.1% raw, EVQ+YaRN(s=4) -39.7%
mla_8k_raw = -31.1
mla_8k_yarn_s4 = -39.7  # inference-only YaRN, no FT

# --- Combined view: all data points we have ---
# Format: (label, tokens_per_param, raw_delta, yarn_delta)
# tokens_per_param = total_tokens / model_params
# 432M model
# 8K model 500M: 500M/432M ≈ 1.16, but trained at 8K (undertrained)
# 4K model 50%: 500M/432M ≈ 1.16, trained at 4K
# 4K model 100%: 1B/432M ≈ 2.31

# =============================================================================
# FIGURE 1: Two-panel figure
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel A: 4K Model, training amount progression ---
x_pos = [0, 1]

# Raw extrapolation line
ax1.plot(x_pos, raw_extrap_4k, 'o-', color='#2166ac', linewidth=2.5,
         markersize=10, label='Raw extrapolation (EVQ vs GEO)', zorder=5)

# YaRN inference-only line
ax1.plot(x_pos, yarn_inference_4k, 's--', color='#b2182b', linewidth=2.5,
         markersize=10, label='+YaRN inference (no FT)', zorder=5)

# YaRN+FT point (only 100% available)
ax1.plot([1], [-2.5], 'D-', color='#d6604d', markersize=12,
         label='+YaRN+FT', zorder=5)

# Zero line
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Fill regions
ax1.fill_between(x_pos, 0, raw_extrap_4k, alpha=0.1, color='#2166ac')
ax1.fill_between(x_pos, 0, yarn_inference_4k, alpha=0.1, color='#b2182b')

ax1.set_xticks(x_pos)
ax1.set_xticklabels(panel_a_labels, fontsize=12)
ax1.set_ylabel('EVQ vs GEO (%, negative = EVQ wins)', fontsize=12)
ax1.set_xlabel('Training amount (of 1B total)', fontsize=12)
ax1.set_title('Panel A: 4K Model → 8K Extrapolation\n(432M MLA, d_rope=32, seed=42)',
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.set_ylim(-10, 15)
ax1.grid(True, alpha=0.3)

# Annotations
ax1.annotate('GEO wins raw\nextrapolation', xy=(1, 11.1), xytext=(0.6, 13),
            fontsize=9, ha='center', color='#2166ac',
            arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1.5))
ax1.annotate('But EVQ+YaRN\nalways wins', xy=(1, -2.5), xytext=(0.4, -7),
            fontsize=9, ha='center', color='#b2182b',
            arrowprops=dict(arrowstyle='->', color='#b2182b', lw=1.5))
ax1.annotate('13.6pp\nswing', xy=(1, 4), fontsize=11, ha='center',
            fontweight='bold', color='purple',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# --- Panel B: Cross-regime comparison ---
# Three regimes: undertrained 8K, undertrained 4K, fully-trained 4K
regimes = ['8K model\n500M tok\n(undertrained)', '4K model\n500M tok\n(50%)', '4K model\n1B tok\n(100%)']
x_pos_b = [0, 1, 2]

raw_vals = [-31.1, +5.7, +11.1]
yarn_vals = [-39.7, -3.1, -2.5]  # 8K: +YaRN(s=4) inf, 4K: +YaRN(s=2) inf/FT

bars_raw = ax2.bar([x - 0.18 for x in x_pos_b], raw_vals, 0.35,
                   color='#2166ac', alpha=0.8, label='Raw extrapolation @ 2×')
bars_yarn = ax2.bar([x + 0.18 for x in x_pos_b], yarn_vals, 0.35,
                    color='#b2182b', alpha=0.8, label='+YaRN composition @ 2×')

# Zero line
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1.5)

# Value labels
for bar, val in zip(bars_raw, raw_vals):
    y_offset = -3 if val > 0 else 2
    ax2.text(bar.get_x() + bar.get_width()/2, val + y_offset,
             f'{val:+.1f}%', ha='center', va='bottom' if val < 0 else 'top',
             fontsize=10, fontweight='bold', color='#2166ac')
for bar, val in zip(bars_yarn, yarn_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val - 2,
             f'{val:+.1f}%', ha='center', va='top',
             fontsize=10, fontweight='bold', color='#b2182b')

ax2.set_xticks(x_pos_b)
ax2.set_xticklabels(regimes, fontsize=10)
ax2.set_ylabel('EVQ vs GEO (%, negative = EVQ wins)', fontsize=12)
ax2.set_title('Panel B: Raw vs +YaRN Across Training Regimes\n(432M MLA, d_rope=32)',
              fontsize=13, fontweight='bold')
ax2.legend(loc='lower left', fontsize=10)
ax2.set_ylim(-50, 20)
ax2.grid(True, alpha=0.3, axis='y')

# Key message annotation
ax2.annotate('Red bars always negative:\nEVQ+YaRN wins regardless\nof training amount',
            xy=(2.18, -2.5), xytext=(1.8, -35),
            fontsize=10, ha='center', color='#b2182b', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fde0dd', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='#b2182b', lw=1.5))

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_composition_curve.png', dpi=150, bbox_inches='tight')
plt.savefig(OUT_DIR / 'fig_composition_curve.pdf', bbox_inches='tight')
print("Saved fig_composition_curve.png and .pdf")


# =============================================================================
# FIGURE 2: Comprehensive training progression (for appendix or main)
# Shows the 4 tables from Phase 18 as a single visual
# =============================================================================

fig2, ax3 = plt.subplots(1, 1, figsize=(10, 6))

# 4K model, all methods, PPL@8K progression with training
# Baseline, +YaRN inference, +YaRN+FT
methods = ['GEO\nbaseline', 'EVQ\nbaseline', 'GEO\n+YaRN', 'EVQ\n+YaRN', 'GEO\n+YaRN+FT', 'EVQ\n+YaRN+FT']
ppl_8k = [77.26, 85.85, 32.2, 31.4, 31.834, 31.043]
colors = ['#4393c3', '#d6604d', '#4393c3', '#d6604d', '#4393c3', '#d6604d']
hatches = ['', '', '//', '//', 'xx', 'xx']

bars = ax3.bar(range(len(methods)), ppl_8k, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Value labels
for i, (bar, val) in enumerate(zip(bars, ppl_8k)):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.set_xticks(range(len(methods)))
ax3.set_xticklabels(methods, fontsize=10)
ax3.set_ylabel('PPL @ 8K', fontsize=12)
ax3.set_title('4K Model (1B tokens) → 8K: Method Comparison\n432M MLA, d_rope=32, seed=42',
              fontsize=13, fontweight='bold')

# Add arrows showing the reversal
ax3.annotate('', xy=(1, 86), xytext=(0, 77),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax3.text(0.5, 83, 'EVQ +11%\n(loses)', ha='center', fontsize=9, color='red')

ax3.annotate('', xy=(5, 31), xytext=(4, 32),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax3.text(4.5, 33.5, 'EVQ -2.5%\n(wins!)', ha='center', fontsize=9, color='green', fontweight='bold')

# Add "13.6pp reversal" bracket
ax3.annotate('13.6pp structural reversal',
            xy=(3, 25), fontsize=12, ha='center', fontweight='bold', color='purple',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3, axis='y')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4393c3', label='GEO'),
                   Patch(facecolor='#d6604d', label='EVQ'),
                   Patch(facecolor='white', edgecolor='black', hatch='//', label='+YaRN inference'),
                   Patch(facecolor='white', edgecolor='black', hatch='xx', label='+YaRN+FT (500 steps)')]
ax3.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_ppl_reversal.png', dpi=150, bbox_inches='tight')
plt.savefig(OUT_DIR / 'fig_ppl_reversal.pdf', bbox_inches='tight')
print("Saved fig_ppl_reversal.png and .pdf")
