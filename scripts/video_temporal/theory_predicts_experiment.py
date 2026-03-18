"""Theory Predicts Experiment: dead-channel effective rank → MSE.

Zero-cost local script. No GPU needed.
Computes theoretical metrics for every (base, tau) config,
then overlays actual DiT MSE data to show theory predicts experiment.

Generates paper-ready figure: fig_theory_predicts.pdf
"""
import numpy as np
import json
import os

# ============================================================
# 1. EVQ-Cosh frequency computation (exact copy from video_dit.py)
# ============================================================

def geo_inv_freq(K, base):
    """Standard geometric RoPE inv_freq."""
    return base ** (-np.arange(K, dtype=np.float64) / K)

def evq_inv_freq(K, tau, base):
    """EVQ-Cosh inv_freq."""
    idx = np.arange(K, dtype=np.float64)
    u = (idx + 0.5) / float(K)
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0 / tau) * np.arcsinh((1.0 - u) * np.sinh(tau))
    return base ** (-phi)


# ============================================================
# 2. Theoretical metrics
# ============================================================

def compute_metrics(inv_freq, T_train, T_extrap=128):
    """Compute theoretical metrics for a set of inv_freq values.

    Returns dict with:
        alive: number of channels with phase > 0.1 rad at T_train
        dead: number of channels with phase < 0.1 rad at T_train
        eff_rank: effective rank (channels with phase > 0.1 rad at T_extrap/4)
        D_adj: adjacent-frame discrimination = sum 2(1-cos(w_k))
        D_far: far-extrap discrimination at t=T_extrap-1 vs t=T_extrap
        min_discrim: minimum discrimination across all delta_t in [1, T_extrap]
        total_phase: sum of all phase accumulations at T_train
    """
    K = len(inv_freq)

    # Phase at training length
    phases_train = inv_freq * T_train  # phase at T_train
    alive = np.sum(phases_train > 0.1)
    dead = K - alive

    # Phase at extrapolation length
    phases_extrap = inv_freq * T_extrap
    eff_rank = np.sum(phases_extrap > 0.1 * (T_extrap / T_train))

    # Adjacent frame discrimination: D(1) = sum_k 2(1-cos(w_k))
    D_adj = np.sum(2 * (1 - np.cos(inv_freq)))

    # Far extrapolation discrimination: at frame T_extrap-1 vs T_extrap
    D_far = np.sum(2 * (1 - np.cos(inv_freq)))  # same formula, position-independent for RoPE

    # Minimum pair discrimination across all delta_t
    # D(delta) = sum_k 2(1-cos(w_k * delta))
    # Minimum occurs at the largest delta that maps to near-zero total phase change
    # For practical purposes: compute D at delta=1 (smallest gap)
    # and D at delta=T_train (largest gap in training)
    D_at_T = np.sum(2 * (1 - np.cos(inv_freq * T_train)))

    # Total phase budget: sum of all phases at T_train
    total_phase = np.sum(phases_train)

    # Worst-case collision metric: minimum cosine similarity between ANY two positions
    # Practical: compute position embeddings at t and t+1 for t in [0, T_extrap]
    # and find minimum || RoPE(t) - RoPE(t+1) ||^2
    # This equals D_adj (constant for RoPE since it only depends on delta, not absolute position)

    # More informative: effective frequency spread
    freq_spread = np.max(inv_freq) / (np.min(inv_freq) + 1e-12)

    # Log-uniform coverage: how uniformly frequencies cover log-space
    log_freqs = np.log(inv_freq + 1e-12)
    log_gaps = np.diff(np.sort(log_freqs))
    coverage_uniformity = np.std(log_gaps) / (np.mean(log_gaps) + 1e-12)  # lower = more uniform

    return {
        'alive': int(alive),
        'dead': int(dead),
        'eff_rank': int(eff_rank),
        'D_adj': float(D_adj),
        'D_at_T': float(D_at_T),
        'total_phase': float(total_phase),
        'freq_spread': float(freq_spread),
        'coverage_uniformity': float(coverage_uniformity),
        'min_phase_rad': float(np.min(phases_train)),
        'max_phase_rad': float(np.max(phases_train)),
    }


# ============================================================
# 3. Compute for all base sweep configs (DiT K_t=16, T_train=32)
# ============================================================

K_t = 16
T_train = 32
T_extrap = 128

bases = [100, 500, 1000, 5000, 10000, 50000]
tau_evq = 1.5

# Actual experimental data (from base_sweep_h2h.json)
exp_yarn_far = {
    100:   {'geo': 0.0161, 'evq': 0.0189},
    500:   {'geo': 0.0210, 'evq': 0.0207},
    1000:  {'geo': 0.0517, 'evq': 0.0178},
    5000:  {'geo': 0.0787, 'evq': 0.0210},
    10000: {'geo': 0.0401, 'evq': 0.0137},
    50000: {'geo': 0.1895, 'evq': 0.0262},
}

exp_no_yarn_far = {
    100:   {'geo': 0.296, 'evq': 0.356},
    500:   {'geo': 0.224, 'evq': 0.146},
    1000:  {'geo': 0.286, 'evq': 0.132},
    5000:  {'geo': 0.172, 'evq': 0.111},
    10000: {'geo': 0.107, 'evq': 0.048},
    50000: {'geo': 0.247, 'evq': 0.108},
}

exp_32f = {
    100:   {'geo': 0.00973, 'evq': 0.00961},
    500:   {'geo': 0.00937, 'evq': 0.00955},
    1000:  {'geo': 0.00940, 'evq': 0.00932},
    5000:  {'geo': 0.00969, 'evq': 0.00962},
    10000: {'geo': 0.00987, 'evq': 0.00978},
    50000: {'geo': 0.01020, 'evq': 0.00960},
}

print("=" * 80)
print("  THEORY PREDICTS EXPERIMENT: Dead Channel Analysis")
print("=" * 80)
print()
print("  Model: 129.6M Video DiT, K_t=%d, T_train=%d, T_extrap=%d" % (K_t, T_train, T_extrap))
print()

# Compute theoretical predictions
results = {}
for base in bases:
    geo = geo_inv_freq(K_t, base)
    evq = evq_inv_freq(K_t, tau_evq, base)

    m_geo = compute_metrics(geo, T_train, T_extrap)
    m_evq = compute_metrics(evq, T_train, T_extrap)

    results[base] = {'geo': m_geo, 'evq': m_evq}

    print("  base=%d:" % base)
    print("    GEO: alive=%d/%d, D_adj=%.4f, total_phase=%.1f, min_phase=%.4f rad" % (
        m_geo['alive'], K_t, m_geo['D_adj'], m_geo['total_phase'], m_geo['min_phase_rad']))
    print("    EVQ: alive=%d/%d, D_adj=%.4f, total_phase=%.1f, min_phase=%.4f rad" % (
        m_evq['alive'], K_t, m_evq['D_adj'], m_evq['total_phase'], m_evq['min_phase_rad']))
    print("    Actual YaRN far MSE:  GEO=%.4f  EVQ=%.4f  delta=%.0f%%" % (
        exp_yarn_far[base]['geo'], exp_yarn_far[base]['evq'],
        (exp_yarn_far[base]['evq'] - exp_yarn_far[base]['geo']) / exp_yarn_far[base]['geo'] * 100))
    print()

# ============================================================
# 4. Correlation analysis
# ============================================================
print("=" * 80)
print("  CORRELATION: Theory vs Experiment")
print("=" * 80)

# Collect vectors
geo_alive = [results[b]['geo']['alive'] for b in bases]
evq_alive = [results[b]['evq']['alive'] for b in bases]
geo_Dadj = [results[b]['geo']['D_adj'] for b in bases]
evq_Dadj = [results[b]['evq']['D_adj'] for b in bases]
geo_mse_yarn = [exp_yarn_far[b]['geo'] for b in bases]
evq_mse_yarn = [exp_yarn_far[b]['evq'] for b in bases]
geo_mse_raw = [exp_no_yarn_far[b]['geo'] for b in bases]
evq_mse_raw = [exp_no_yarn_far[b]['evq'] for b in bases]

# Combined: all 12 data points (6 bases x 2 methods)
all_alive = geo_alive + evq_alive
all_Dadj = geo_Dadj + evq_Dadj
all_mse_yarn = geo_mse_yarn + evq_mse_yarn
all_mse_raw = geo_mse_raw + evq_mse_raw
all_labels = ['GEO b%d' % b for b in bases] + ['EVQ b%d' % b for b in bases]

# Pearson correlation
def pearson(x, y):
    x, y = np.array(x), np.array(y)
    return np.corrcoef(x, y)[0, 1]

# For D_adj vs MSE, expect NEGATIVE correlation (more discrimination -> lower MSE)
r_Dadj_yarn = pearson(all_Dadj, all_mse_yarn)
r_Dadj_raw = pearson(all_Dadj, all_mse_raw)
r_alive_yarn = pearson(all_alive, all_mse_yarn)
r_alive_raw = pearson(all_alive, all_mse_raw)

print()
print("  D_adj vs YaRN far MSE: r = %.3f" % r_Dadj_yarn)
print("  D_adj vs Raw far MSE:  r = %.3f" % r_Dadj_raw)
print("  Alive vs YaRN far MSE: r = %.3f" % r_alive_yarn)
print("  Alive vs Raw far MSE:  r = %.3f" % r_alive_raw)
print()

# ============================================================
# 5. Industry model predictions
# ============================================================
print("=" * 80)
print("  PREDICTIONS FOR INDUSTRY MODELS")
print("=" * 80)
print()

industry_models = {
    'Wan2.1 (1.3B/14B)': {'K_t': 22, 'T_train': 21, 'base': 10000},
    'CogVideoX (2B/5B)': {'K_t': 8, 'T_train': 13, 'base': 10000},
    'Open-Sora-v2':       {'K_t': 22, 'T_train': 51, 'base': 10000},
    'HunyuanVideo':       {'K_t': 8, 'T_train': 33, 'base': 10000},
    'Latte-XL':           {'K_t': 12, 'T_train': 16, 'base': 10000},
}

# Find optimal tau for each model
print("  %-25s | GEO dead%% | EVQ* dead%% | tau* | D_adj ratio" % "Model")
print("  " + "-" * 75)

for name, cfg in industry_models.items():
    K = cfg['K_t']
    T = cfg['T_train']
    B = cfg['base']

    geo = geo_inv_freq(K, B)
    m_geo = compute_metrics(geo, T)

    # Sweep tau to find optimal
    best_tau = 0
    best_Dadj = m_geo['D_adj']
    best_dead = m_geo['dead']
    for tau_try in np.arange(0.1, 3.01, 0.1):
        evq_try = evq_inv_freq(K, tau_try, B)
        m_try = compute_metrics(evq_try, T)
        if m_try['D_adj'] > best_Dadj:
            best_Dadj = m_try['D_adj']
            best_tau = tau_try
            best_dead = m_try['dead']

    evq_opt = evq_inv_freq(K, best_tau, B)
    m_evq = compute_metrics(evq_opt, T)

    ratio = m_evq['D_adj'] / m_geo['D_adj'] if m_geo['D_adj'] > 0 else float('inf')

    print("  %-25s | %5.1f%%    | %5.1f%%     | %.1f  | %.2fx" % (
        name, m_geo['dead'] / K * 100, best_dead / K * 100, best_tau, ratio))

print()

# ============================================================
# 6. The key "prediction → verification" table
# ============================================================
print("=" * 80)
print("  PREDICTION → VERIFICATION (DiT base sweep)")
print("=" * 80)
print()
print("  Theory predicts: EVQ advantage = f(dead channel gap)")
print()
print("  %-8s | GEO dead | EVQ dead | Gap | Predicted | Actual YaRN delta" % "base")
print("  " + "-" * 72)

for base in bases:
    g = results[base]['geo']
    e = results[base]['evq']
    gap = g['dead'] - e['dead']
    actual_delta = (exp_yarn_far[base]['evq'] - exp_yarn_far[base]['geo']) / exp_yarn_far[base]['geo'] * 100

    # Prediction: if gap > 0, EVQ should win. Larger gap → larger win.
    if gap > 0:
        predicted = "EVQ wins (gap=%d)" % gap
    elif gap == 0:
        predicted = "Tie (gap=0)"
    else:
        predicted = "GEO wins (gap=%d)" % gap

    print("  %-8d | %d/%d     | %d/%d     | %+d  | %-20s | %+.0f%%" % (
        base, g['dead'], K_t, e['dead'], K_t, gap, predicted, actual_delta))

print()

# ============================================================
# 7. Generate figure (if matplotlib available)
# ============================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Color scheme
    c_geo = '#d62728'
    c_evq = '#1f77b4'

    # Panel A: D_adj vs YaRN MSE (scatter, all 12 points)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(geo_Dadj, geo_mse_yarn, c=c_geo, s=80, marker='o', label='GEO', zorder=5, edgecolors='black', linewidth=0.5)
    ax1.scatter(evq_Dadj, evq_mse_yarn, c=c_evq, s=80, marker='s', label='EVQ', zorder=5, edgecolors='black', linewidth=0.5)
    # Label each point with base value
    for i, b in enumerate(bases):
        ax1.annotate(str(b), (geo_Dadj[i], geo_mse_yarn[i]), fontsize=7, ha='left', va='bottom', color=c_geo)
        ax1.annotate(str(b), (evq_Dadj[i], evq_mse_yarn[i]), fontsize=7, ha='left', va='bottom', color=c_evq)
    ax1.set_xlabel('Discrimination D(1) = $\\sum_k 2(1-\\cos\\omega_k)$', fontsize=10)
    ax1.set_ylabel('YaRN Far Extrapolation MSE', fontsize=10)
    ax1.set_title('(a) Discrimination Predicts MSE (r=%.2f)' % r_Dadj_yarn, fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_yscale('log')

    # Panel B: Dead channels per base (grouped bar)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(bases))
    w = 0.35
    bars_geo = [results[b]['geo']['dead'] for b in bases]
    bars_evq = [results[b]['evq']['dead'] for b in bases]
    ax2.bar(x - w/2, bars_geo, w, color=c_geo, label='GEO', edgecolor='black', linewidth=0.5)
    ax2.bar(x + w/2, bars_evq, w, color=c_evq, label='EVQ', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(b) for b in bases], fontsize=9)
    ax2.set_xlabel('Base frequency', fontsize=10)
    ax2.set_ylabel('Dead channels (phase < 0.1 rad)', fontsize=10)
    ax2.set_title('(b) Dead Channels: GEO vs EVQ', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, K_t)
    ax2.axhline(y=K_t/2, color='gray', linestyle='--', alpha=0.3)

    # Panel C: MSE improvement vs dead channel gap
    ax3 = fig.add_subplot(gs[1, 0])
    gaps = [results[b]['geo']['dead'] - results[b]['evq']['dead'] for b in bases]
    deltas_yarn = [(exp_yarn_far[b]['evq'] - exp_yarn_far[b]['geo']) / exp_yarn_far[b]['geo'] * 100 for b in bases]
    deltas_raw = [(exp_no_yarn_far[b]['evq'] - exp_no_yarn_far[b]['geo']) / exp_no_yarn_far[b]['geo'] * 100 for b in bases]

    ax3.scatter(gaps, deltas_yarn, c=c_evq, s=100, marker='s', label='YaRN', zorder=5, edgecolors='black', linewidth=0.5)
    ax3.scatter(gaps, deltas_raw, c=c_evq, s=100, marker='^', label='No YaRN', zorder=5, edgecolors='black', linewidth=0.5, alpha=0.6)
    for i, b in enumerate(bases):
        ax3.annotate('b=%d' % b, (gaps[i], deltas_yarn[i]), fontsize=7, ha='left', va='bottom')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Dead channel gap (GEO dead - EVQ dead)', fontsize=10)
    ax3.set_ylabel('EVQ MSE improvement (%)', fontsize=10)
    ax3.set_title('(c) Dead Channel Gap Predicts Improvement', fontsize=11)
    ax3.legend(fontsize=9)

    # Panel D: Industry model predictions
    ax4 = fig.add_subplot(gs[1, 1])
    model_names = list(industry_models.keys())
    geo_dead_pcts = []
    evq_dead_pcts = []
    for name, cfg in industry_models.items():
        K = cfg['K_t']
        T = cfg['T_train']
        B = cfg['base']
        geo = geo_inv_freq(K, B)
        m_geo = compute_metrics(geo, T)
        # Use tau=1.5 for consistency
        evq = evq_inv_freq(K, 1.5, B)
        m_evq = compute_metrics(evq, T)
        geo_dead_pcts.append(m_geo['dead'] / K * 100)
        evq_dead_pcts.append(m_evq['dead'] / K * 100)

    y_pos = np.arange(len(model_names))
    ax4.barh(y_pos - 0.17, geo_dead_pcts, 0.34, color=c_geo, label='GEO', edgecolor='black', linewidth=0.5)
    ax4.barh(y_pos + 0.17, evq_dead_pcts, 0.34, color=c_evq, label='EVQ', edgecolor='black', linewidth=0.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([n.replace(' (1.3B/14B)', '\n(1.3B/14B)').replace(' (2B/5B)', '\n(2B/5B)') for n in model_names], fontsize=8)
    ax4.set_xlabel('Dead temporal channels (%)', fontsize=10)
    ax4.set_title('(d) Industry Model Predictions', fontsize=11)
    ax4.legend(fontsize=9, loc='lower right')
    ax4.set_xlim(0, 60)
    ax4.axvline(x=25, color='gray', linestyle='--', alpha=0.3, label='')

    fig.suptitle('Theory Predicts Experiment: Frequency Allocation and Dead Channels', fontsize=13, fontweight='bold', y=0.98)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', 'video_dit', 'paper_figures')
    os.makedirs(out_dir, exist_ok=True)

    for ext in ['pdf', 'png']:
        path = os.path.join(out_dir, 'fig_theory_predicts.%s' % ext)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print("  Saved: %s" % path)

    plt.close()

except ImportError:
    print("  [matplotlib not available, skipping figure generation]")

# ============================================================
# 8. Save all computed data
# ============================================================
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', 'video_dit')
output = {
    'description': 'Theory predicts experiment: dead channel analysis for base sweep',
    'model': 'Video DiT 129.6M, K_t=16, T_train=32, T_extrap=128',
    'theoretical_metrics': {str(b): results[b] for b in bases},
    'experimental_mse': {
        'yarn_far': {str(b): exp_yarn_far[b] for b in bases},
        'no_yarn_far': {str(b): exp_no_yarn_far[b] for b in bases},
        '32f_indist': {str(b): exp_32f[b] for b in bases},
    },
    'correlations': {
        'D_adj_vs_yarn_mse': r_Dadj_yarn,
        'D_adj_vs_raw_mse': r_Dadj_raw,
        'alive_vs_yarn_mse': r_alive_yarn,
        'alive_vs_raw_mse': r_alive_raw,
    },
}

with open(os.path.join(out_dir, 'theory_predicts_experiment.json'), 'w') as f:
    json.dump(output, f, indent=2)
print("\n  Saved: theory_predicts_experiment.json")
