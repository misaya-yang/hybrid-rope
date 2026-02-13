#!/usr/bin/env python3
"""Generate visualization for LLaMA Shape vs Theta experiment"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Load results
results_path = Path(__file__).parent.parent / "results" / "llama_shape_theta_min" / "results.json"
with open(results_path) as f:
    data = json.load(f)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: PPL Comparison (log scale)
configs = ['geo_10k', 'sigmoid_t100k']
ppl_2k = [data[c]['ppl_2k'] for c in configs]
ppl_16k = [data[c]['ppl_16k'] for c in configs]

x = np.arange(len(configs))
width = 0.35

ax1 = axes[0]
bars1 = ax1.bar(x - width/2, ppl_2k, width, label='PPL@2048', color='#2ecc71')
bars2 = ax1.bar(x + width/2, ppl_16k, width, label='PPL@16384', color='#e74c3c')

ax1.set_ylabel('Perplexity (log scale)')
ax1.set_title('LLaMA-3-8B: RoPE Config vs PPL')
ax1.set_xticks(x)
ax1.set_xticklabels(['Geometric\n(θ=10k)', 'Sigmoid\n(T=100k)'])
ax1.set_yscale('log')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars1, ppl_2k):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, ppl_16k):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Collapse Ratio
ax2 = axes[1]
collapse_ratios = [data[c]['collapse_ratio'] for c in configs]
colors = ['#e74c3c', '#2ecc71']
bars = ax2.bar(configs, collapse_ratios, color=colors)

ax2.set_ylabel('Collapse Ratio (PPL@16k / PPL@2k)')
ax2.set_title('Collapse Ratio Comparison (Lower is Better)')
ax2.set_xticklabels(['Geometric\n(θ=10k)', 'Sigmoid\n(T=100k)'])
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, collapse_ratios):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
             f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotation
ax2.annotate('20.5x\nimprovement!', xy=(1, 1.5), xytext=(0.5, 10),
            fontsize=12, ha='center', color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()

# Save figure
output_path = Path(__file__).parent.parent / "results" / "llama_shape_theta_min" / "figures" / "ppl_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved figure to {output_path}")

# Also save as SVG
svg_path = output_path.with_suffix('.svg')
plt.savefig(svg_path, bbox_inches='tight')
print(f"Saved SVG to {svg_path}")

plt.show()