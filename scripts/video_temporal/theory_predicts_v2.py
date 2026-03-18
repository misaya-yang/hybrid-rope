"""Theory Predicts Experiment v2: Worst-case discrimination metric.

Key insight: dead channel COUNT is too coarse. The right metric is
D_min = min_{delta=1..T} sum_k 2(1-cos(w_k * delta))
= worst-case pair discrimination across all temporal positions.

If D_min is high, ALL position pairs are distinguishable -> good extrapolation.
If D_min is low, some positions "collide" -> model confuses frames.

EVQ keeps D_min high by spreading frequencies uniformly in log-space,
avoiding the "phase collision" where multiple channels hit 2*pi simultaneously.
"""
import numpy as np
import json
import os


def geo_inv_freq(K, base):
    return base ** (-np.arange(K, dtype=np.float64) / K)

def evq_inv_freq(K, tau, base):
    idx = np.arange(K, dtype=np.float64)
    u = (idx + 0.5) / float(K)
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u) * np.sinh(tau))
    return base ** (-phi)


def compute_discrimination_curve(inv_freq, T_max=128):
    """Compute D(delta) for delta = 1..T_max.
    D(delta) = sum_k 2*(1 - cos(w_k * delta))
    """
    deltas = np.arange(1, T_max + 1, dtype=np.float64)
    # Shape: [T_max, K]
    phases = np.outer(deltas, inv_freq)
    D = np.sum(2 * (1 - np.cos(phases)), axis=1)
    return deltas, D


def compute_yarn_discrimination(inv_freq, T_train, T_target, T_max=None):
    """Compute D(delta) under YaRN scaling.

    YaRN NTK-aware interpolation: scale factor s = T_target / T_train.
    For each frequency w_k:
        - If w_k * T_train > pi (high-freq): keep w_k unchanged
        - If w_k * T_train < pi (low-freq): scale w_k by 1/s (interpolation)
        - Smooth transition via ramp function

    Simplified: linear ramp between the two regimes.
    """
    if T_max is None:
        T_max = T_target

    s = T_target / T_train
    K = len(inv_freq)

    # YaRN ramp: compute per-channel scaling
    # Based on the original YaRN paper:
    # alpha = (w * T_train / pi - low) / (high - low), clamped to [0,1]
    # scaled_w = w * (1 - alpha) / s + w * alpha
    low, high = 1.0, 32.0  # YaRN defaults

    freqs_train = inv_freq * T_train  # phase at training length
    alpha = (freqs_train / np.pi - low) / (high - low)
    alpha = np.clip(alpha, 0, 1)

    yarn_freq = inv_freq * (1 - alpha) / s + inv_freq * alpha

    deltas = np.arange(1, T_max + 1, dtype=np.float64)
    phases = np.outer(deltas, yarn_freq)
    D = np.sum(2 * (1 - np.cos(phases)), axis=1)
    return deltas, D


# ============================================================
# Compute for all base sweep configs
# ============================================================
K_t = 16
T_train = 32
T_extrap = 128

bases = [100, 500, 1000, 5000, 10000, 50000]
tau_evq = 1.5

exp_yarn_far = {
    100:   {'geo': 0.0161, 'evq': 0.0189},
    500:   {'geo': 0.0210, 'evq': 0.0207},
    1000:  {'geo': 0.0517, 'evq': 0.0178},
    5000:  {'geo': 0.0787, 'evq': 0.0210},
    10000: {'geo': 0.0401, 'evq': 0.0137},
    50000: {'geo': 0.1895, 'evq': 0.0262},
}

exp_no_yarn = {
    100:   {'geo': 0.296, 'evq': 0.356},
    500:   {'geo': 0.224, 'evq': 0.146},
    1000:  {'geo': 0.286, 'evq': 0.132},
    5000:  {'geo': 0.172, 'evq': 0.111},
    10000: {'geo': 0.107, 'evq': 0.048},
    50000: {'geo': 0.247, 'evq': 0.108},
}

print("=" * 80)
print("  D_min: Worst-Case Discrimination Analysis")
print("=" * 80)
print()

results = {}
for base in bases:
    geo = geo_inv_freq(K_t, base)
    evq = evq_inv_freq(K_t, tau_evq, base)

    # Raw (no YaRN)
    _, D_geo_raw = compute_discrimination_curve(geo, T_extrap)
    _, D_evq_raw = compute_discrimination_curve(evq, T_extrap)

    # YaRN-scaled
    _, D_geo_yarn = compute_yarn_discrimination(geo, T_train, T_extrap, T_extrap)
    _, D_evq_yarn = compute_yarn_discrimination(evq, T_train, T_extrap, T_extrap)

    # D_min over extrapolation region (delta > T_train)
    extrap_slice = slice(T_train - 1, T_extrap)  # delta = T_train..T_extrap

    geo_Dmin_yarn = np.min(D_geo_yarn[extrap_slice])
    evq_Dmin_yarn = np.min(D_evq_yarn[extrap_slice])
    geo_Dmin_raw = np.min(D_geo_raw[extrap_slice])
    evq_Dmin_raw = np.min(D_evq_raw[extrap_slice])

    # Also compute D_min over full range
    geo_Dmin_full = np.min(D_geo_yarn)
    evq_Dmin_full = np.min(D_evq_yarn)

    # D at delta=1 (adjacent frame)
    geo_D1 = D_geo_yarn[0]
    evq_D1 = D_evq_yarn[0]

    results[base] = {
        'geo_Dmin_yarn_extrap': float(geo_Dmin_yarn),
        'evq_Dmin_yarn_extrap': float(evq_Dmin_yarn),
        'geo_Dmin_raw_extrap': float(geo_Dmin_raw),
        'evq_Dmin_raw_extrap': float(evq_Dmin_raw),
        'geo_Dmin_yarn_full': float(geo_Dmin_full),
        'evq_Dmin_yarn_full': float(evq_Dmin_full),
        'geo_D1_yarn': float(geo_D1),
        'evq_D1_yarn': float(evq_D1),
        'D_geo_yarn': D_geo_yarn.tolist(),
        'D_evq_yarn': D_evq_yarn.tolist(),
        'D_geo_raw': D_geo_raw.tolist(),
        'D_evq_raw': D_evq_raw.tolist(),
    }

    print("  base=%d:" % base)
    print("    YaRN D_min(extrap): GEO=%.3f  EVQ=%.3f  ratio=%.2fx" % (
        geo_Dmin_yarn, evq_Dmin_yarn, evq_Dmin_yarn / (geo_Dmin_yarn + 1e-12)))
    print("    Raw  D_min(extrap): GEO=%.3f  EVQ=%.3f  ratio=%.2fx" % (
        geo_Dmin_raw, evq_Dmin_raw, evq_Dmin_raw / (geo_Dmin_raw + 1e-12)))
    print("    Actual YaRN MSE:    GEO=%.4f  EVQ=%.4f  delta=%.0f%%" % (
        exp_yarn_far[base]['geo'], exp_yarn_far[base]['evq'],
        (exp_yarn_far[base]['evq'] - exp_yarn_far[base]['geo']) / exp_yarn_far[base]['geo'] * 100))
    print()


# ============================================================
# Correlation: D_min vs MSE
# ============================================================
def pearson(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1])

# Collect all 12 points (6 bases x 2 methods)
all_Dmin_yarn = ([results[b]['geo_Dmin_yarn_extrap'] for b in bases] +
                  [results[b]['evq_Dmin_yarn_extrap'] for b in bases])
all_Dmin_raw = ([results[b]['geo_Dmin_raw_extrap'] for b in bases] +
                 [results[b]['evq_Dmin_raw_extrap'] for b in bases])
all_mse_yarn = ([exp_yarn_far[b]['geo'] for b in bases] +
                 [exp_yarn_far[b]['evq'] for b in bases])
all_mse_raw = ([exp_no_yarn[b]['geo'] for b in bases] +
                [exp_no_yarn[b]['evq'] for b in bases])

r_Dmin_yarn = pearson(all_Dmin_yarn, all_mse_yarn)
r_Dmin_raw = pearson(all_Dmin_raw, all_mse_raw)

# GEO-only correlation (6 points)
geo_Dmin_yarn_vec = [results[b]['geo_Dmin_yarn_extrap'] for b in bases]
geo_mse_yarn_vec = [exp_yarn_far[b]['geo'] for b in bases]
r_geo_only = pearson(geo_Dmin_yarn_vec, geo_mse_yarn_vec)

print("=" * 80)
print("  CORRELATION")
print("=" * 80)
print("  D_min(YaRN extrap) vs YaRN MSE:     r = %.3f (all 12 points)" % r_Dmin_yarn)
print("  D_min(raw extrap)  vs Raw MSE:       r = %.3f (all 12 points)" % r_Dmin_raw)
print("  D_min(YaRN) vs MSE (GEO only):       r = %.3f (6 points)" % r_geo_only)
print()


# ============================================================
# Generate figure
# ============================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    c_geo = '#d62728'
    c_evq = '#1f77b4'

    # Panel A: D(delta) curves for base=10000 (our main config)
    ax1 = fig.add_subplot(gs[0, 0])
    deltas = np.arange(1, T_extrap + 1)
    ax1.plot(deltas, results[10000]['D_geo_yarn'], color=c_geo, linewidth=2, label='GEO base=10000')
    ax1.plot(deltas, results[10000]['D_evq_yarn'], color=c_evq, linewidth=2, label='EVQ base=10000')
    ax1.axvline(x=T_train, color='gray', linestyle='--', alpha=0.5, label='Train boundary (32f)')
    # Mark D_min
    geo_dmin_idx = T_train - 1 + np.argmin(results[10000]['D_geo_yarn'][T_train-1:])
    evq_dmin_idx = T_train - 1 + np.argmin(results[10000]['D_evq_yarn'][T_train-1:])
    ax1.scatter([geo_dmin_idx + 1], [results[10000]['D_geo_yarn'][geo_dmin_idx]],
                color=c_geo, s=100, zorder=5, marker='v', edgecolors='black')
    ax1.scatter([evq_dmin_idx + 1], [results[10000]['D_evq_yarn'][evq_dmin_idx]],
                color=c_evq, s=100, zorder=5, marker='v', edgecolors='black')
    ax1.set_xlabel('Position gap $\\Delta$', fontsize=11)
    ax1.set_ylabel('Discrimination $D(\\Delta)$', fontsize=11)
    ax1.set_title('(a) Discrimination Curve (base=10000, YaRN)', fontsize=12)
    ax1.legend(fontsize=9)

    # Panel B: D_min vs MSE scatter (THE key plot)
    ax2 = fig.add_subplot(gs[0, 1])

    geo_Dmin_list = [results[b]['geo_Dmin_yarn_extrap'] for b in bases]
    evq_Dmin_list = [results[b]['evq_Dmin_yarn_extrap'] for b in bases]
    geo_mse_list = [exp_yarn_far[b]['geo'] for b in bases]
    evq_mse_list = [exp_yarn_far[b]['evq'] for b in bases]

    ax2.scatter(geo_Dmin_list, geo_mse_list, c=c_geo, s=100, marker='o',
                label='GEO', zorder=5, edgecolors='black', linewidth=0.5)
    ax2.scatter(evq_Dmin_list, evq_mse_list, c=c_evq, s=100, marker='s',
                label='EVQ', zorder=5, edgecolors='black', linewidth=0.5)
    for i, b in enumerate(bases):
        ax2.annotate(str(b), (geo_Dmin_list[i], geo_mse_list[i]),
                     fontsize=8, ha='left', va='bottom', color=c_geo)
        ax2.annotate(str(b), (evq_Dmin_list[i], evq_mse_list[i]),
                     fontsize=8, ha='left', va='bottom', color=c_evq)

    ax2.set_xlabel('$D_{min}$ (worst-case discrimination, YaRN extrap)', fontsize=11)
    ax2.set_ylabel('Far extrapolation MSE', fontsize=11)
    ax2.set_title('(b) $D_{min}$ Predicts MSE (r=%.2f)' % r_Dmin_yarn, fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_yscale('log')

    # Panel C: D(delta) curves for multiple bases (GEO only, shows problem)
    ax3 = fig.add_subplot(gs[1, 0])
    colors_base = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#d62728', '#e377c2']
    for i, b in enumerate(bases):
        ax3.plot(deltas, results[b]['D_geo_yarn'], color=colors_base[i],
                 linewidth=1.5, label='b=%d' % b, alpha=0.8)
    ax3.axvline(x=T_train, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Position gap $\\Delta$', fontsize=11)
    ax3.set_ylabel('Discrimination $D(\\Delta)$', fontsize=11)
    ax3.set_title('(c) GEO Discrimination Degrades with Base', fontsize=12)
    ax3.legend(fontsize=8, ncol=2)
    ax3.set_xlim(0, T_extrap)

    # Panel D: EVQ D(delta) curves (stable across bases)
    ax4 = fig.add_subplot(gs[1, 1])
    for i, b in enumerate(bases):
        ax4.plot(deltas, results[b]['D_evq_yarn'], color=colors_base[i],
                 linewidth=1.5, label='b=%d' % b, alpha=0.8)
    ax4.axvline(x=T_train, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Position gap $\\Delta$', fontsize=11)
    ax4.set_ylabel('Discrimination $D(\\Delta)$', fontsize=11)
    ax4.set_title('(d) EVQ Discrimination Stable Across Base', fontsize=12)
    ax4.legend(fontsize=8, ncol=2)
    ax4.set_xlim(0, T_extrap)

    # Match y-axis limits for panels C and D
    ymax = max(ax3.get_ylim()[1], ax4.get_ylim()[1])
    ymin = min(ax3.get_ylim()[0], ax4.get_ylim()[0])
    ax3.set_ylim(ymin, ymax)
    ax4.set_ylim(ymin, ymax)

    fig.suptitle('Worst-Case Discrimination Predicts Extrapolation Performance',
                 fontsize=14, fontweight='bold', y=0.98)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                           'results', 'video_dit', 'paper_figures')
    os.makedirs(out_dir, exist_ok=True)

    for ext in ['pdf', 'png']:
        path = os.path.join(out_dir, 'fig_discrimination_predicts.%s' % ext)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print("  Saved: %s" % path)

    plt.close()

except ImportError:
    print("  [matplotlib not available]")

# Save data
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', 'video_dit')
with open(os.path.join(out_dir, 'discrimination_analysis_v2.json'), 'w') as f:
    # Don't save the full D curves (too large), just metrics
    save_data = {}
    for b in bases:
        save_data[str(b)] = {k: v for k, v in results[b].items()
                              if not k.startswith('D_')}
    json.dump({
        'description': 'Worst-case discrimination D_min predicts extrapolation MSE',
        'correlations': {
            'D_min_yarn_vs_yarn_mse': r_Dmin_yarn,
            'D_min_raw_vs_raw_mse': r_Dmin_raw,
            'D_min_yarn_geo_only': r_geo_only,
        },
        'metrics': save_data,
    }, f, indent=2)
print("\n  Saved: discrimination_analysis_v2.json")
