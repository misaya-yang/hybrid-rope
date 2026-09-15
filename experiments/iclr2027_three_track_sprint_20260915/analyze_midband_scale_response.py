"""CPU-only comparison of fixed public RoPE constructors and phase response scales.

No model loading, table fitting, GPU calls, or task evaluation.  All constructors
are compared on one shared band; turn-ramp YaRN is explicitly a different form.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.fixed_rope_three_interfaces_20260913.tables import analytic_exponents


def band(base, pairs, length):
    omega = base**(-np.arange(pairs)/pairs)
    turns = omega*length/(2*np.pi)
    return omega, int(np.flatnonzero(turns > 32)[-1]), int(np.flatnonzero(turns < 1)[0])


def build(base, pairs, length, scale):
    omega, lo, hi = band(base, pairs, length)
    t = np.clip((np.arange(pairs)-lo)/(hi-lo), 0., 1.)
    blend = 1-t+t/scale
    y = -np.log(blend)/np.log(scale)
    out = {'Y_index': y}
    for name, method in (('P', 'mrpro'), ('T', 'tailspline'),
                         ('C', 'tailspline_dose_control'), ('U', 'uni'), ('B', 'bm')):
        out[name] = analytic_exponents(method, pairs, low=lo, high=hi)
    # Paper turn-ramp: keep this separate from the official index-ramp code.
    r = omega*length/(2*np.pi)
    gamma = np.clip((r-1)/31, 0., 1.)
    out['Y_turns'] = -np.log(gamma+(1-gamma)/scale)/np.log(scale)
    return omega, lo, hi, out


def phase_distance(delta_frequency, phase_budget):
    result = np.full_like(delta_frequency, np.inf)
    np.divide(phase_budget, abs(delta_frequency), out=result,
              where=abs(delta_frequency) > 1e-17)
    return result


def ratio_summary(base, pairs, length, scale):
    omega, lo, hi, profiles = build(base, pairs, length, scale)
    active = np.arange(lo+1, hi)
    rows = []
    for k in active:
        rows.append({'slot': int(k), 'q': int(k-lo), 't': float((k-lo)/(hi-lo)),
                     'native_turns': float(omega[k]*length/(2*np.pi)),
                     'm': {name: float(m[k]) for name, m in profiles.items()},
                     'wavelength_multiplier': {name: float(scale**m[k]) for name, m in profiles.items()},
                     'T_over_P_wavelength': float(scale**(profiles['T'][k]-profiles['P'][k]))})
    compare = profiles['P'][active]-profiles['Y_index'][active]
    distance = {}
    for name in ('Y_index', 'P', 'T', 'C', 'B', 'U'):
        nu = omega*scale**(-profiles[name])
        # Full intervals on which pointwise phase deviation <= pi/2.
        native_h = phase_distance(omega[active]-nu[active], np.pi/2)
        dilation_h = phase_distance(nu[active]-omega[active]/scale, np.pi/2)
        distance[name] = {
            'native_reference_horizon': native_h.tolist(),
            'dilation_reference_horizon': dilation_h.tolist(),
            'dilation_covered_slots_at_rL': {
                str(r): active[dilation_h >= r*length].tolist() for r in (1, 2, 4, 8, 16)},
            'native_covered_slots_at_local_distance': {
                str(d): active[native_h >= d].tolist() for d in (128, 512, 1024, 2048)},
        }
    return dict(base=base, pairs=pairs, native_length=length, scale=scale, band=[lo, hi],
                changed_pairs=hi-lo-1, max_T_P_wavelength_ratio=max(r['T_over_P_wavelength'] for r in rows),
                P_slower_than_Y_index_slots=active[compare > 1e-12].tolist(),
                P_faster_than_Y_index_slots=active[compare < -1e-12].tolist(),
                slots=rows, phase_budget_radians=math.pi/2, reference_horizons=distance)


def find_crossovers(n=17):
    result = []
    # Solve the nontrivial S>1 crossing in log S without scipy.
    for q in range(1, n):
        t, p = q/n, q*(q+1)/(n*(n+1))
        def f(log_s):
            return -math.log1p(t*math.expm1(-log_s))/log_s-p
        left, right = 1e-4, 1.
        while f(right) > 0 and right < 1024:
            right *= 2
        for _ in range(90):
            mid = (left+right)/2
            if f(mid) > 0: left = mid
            else: right = mid
        result.append(dict(q=q, t=t, scale=math.exp((left+right)/2)))
    return result


def checks():
    rng = np.random.default_rng(20260915)
    max_elasticity_error = max_anchor_error = max_tail_identity_error = 0.
    for _ in range(300):
        t = rng.uniform(.001, .999)
        scale = rng.uniform(1.01, 32)
        h = 1e-5
        wavelength = lambda z: 1/(1-t+t/z)
        numerical = (math.log(wavelength(scale*math.exp(h)))-math.log(wavelength(scale*math.exp(-h))))/(2*h)
        analytic = t/(scale*(1-t)+t)
        max_elasticity_error = max(max_elasticity_error, abs(numerical-analytic))
        assert wavelength(scale) < 1/(1-t)
        w, d, r, beta = rng.uniform(.001, 2), rng.uniform(.01, 100), rng.uniform(.1, 1), rng.uniform(0, 1)
        a, b = math.cos(w*d*r), math.sin(w*d*r)
        ref_a, ref_b = math.cos(w*d*beta), math.sin(w*d*beta)
        # Exact rotation norm from any base phase versus frequency difference.
        measured = math.hypot(a-ref_a, b-ref_b)
        predicted = 2*abs(math.sin(w*d*(r-beta)/2))
        max_anchor_error = max(max_anchor_error, abs(measured-predicted))
    for n in range(1, 65):
        p = analytic_exponents('mrpro', n+1, low=0, high=n)
        t = analytic_exponents('tailspline', n+1, low=0, high=n)
        q = np.arange(n+1)
        claimed = q*(n-q)*(3*n+q+1)/(n*(n+1)*(2*n+1))
        max_tail_identity_error = max(max_tail_identity_error, float(max(abs(t-p-claimed))))
        assert np.all(t >= p-1e-12)
    # Under an explicitly declared stretched-reference content model, the same
    # channel coefficients favor smaller residual phase before the first crossing.
    errors = []
    for _ in range(300):
        slow, fast = sorted(rng.uniform(.01, 1, 2))
        beta = rng.uniform(0, slow)
        d = rng.uniform(.01, math.pi/(fast-beta))
        amp = rng.uniform(.01, 3)
        score_slow = amp*math.cos((slow-beta)*d)
        score_fast = amp*math.cos((fast-beta)*d)
        assert score_slow >= score_fast-1e-12
        errors.append(score_slow-score_fast)
    # Explicit sign counterexample outside the branch: reducing residual phase
    # from 2*pi to pi worsens a positively aligned cosine response.
    counterexample = {'larger_residual_phase': 2*math.pi, 'smaller_residual_phase': math.pi,
                      'larger_score': 1., 'smaller_score': -1.}
    # Averaging can mask deterioration: one easier row gains probability while
    # a barely correct second row crosses zero despite positive mean improvement.
    margins_p = np.array([-3., .1]); margins_t = np.array([-2., -.1])
    assert margins_t.mean() > margins_p.mean()
    assert np.mean(margins_t > 0) < np.mean(margins_p > 0)
    assert max_elasticity_error < 1e-9 and max_anchor_error < 1e-12 and max_tail_identity_error < 1e-12
    return dict(scale_elasticity_cases=300, max_elasticity_error=max_elasticity_error,
                rotation_identity_cases=300, max_rotation_identity_error=max_anchor_error,
                finite_profile_cases=64, max_profile_identity_error=max_tail_identity_error,
                conditional_cosine_cases=300, min_conditional_score_difference=float(min(errors)),
                periodic_counterexample=counterexample,
                mean_margin_vs_accuracy_counterexample=dict(P=margins_p.tolist(),T=margins_t.tolist()))


COLORS = {'Y_index': '#596b82', 'P': '#d9792b', 'T': '#277a6e', 'C': '#8358a5'}
LABELS = {'Y_index': 'YaRN (index blend)', 'P': 'MrPro', 'T': 'TailSpline', 'C': 'Equal-displacement C'}


def draw(out):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4), layout='constrained')
    for ax, scale in zip(axes[0], (4, 16)):
        omega, lo, hi, p = build(500000., 64, 8192, scale)
        slots = np.arange(lo, hi+1)
        for name in ('Y_index', 'P', 'T'):
            ax.plot(slots, scale**p[name][slots], color=COLORS[name], lw=2, label=LABELS[name])
        ax.set(title=f'Wavelength growth at S={scale}', xlabel='Rotary pair index', ylabel='Wavelength / native wavelength')
        ax.grid(alpha=.18)
    axes[0, 0].legend(frameon=False)
    ax = axes[1, 0]
    scales = np.geomspace(1.001, 64, 150)
    for name in ('Y_index', 'P', 'T'):
        values = []
        for scale in scales:
            _, _, _, p = build(500000., 64, 8192, scale)
            values.append(scale**p[name][26])
        ax.plot(scales, values, color=COLORS[name], lw=2, label=LABELS[name])
    t = 8/17
    ax.axhline(1/(1-t), color=COLORS['Y_index'], ls=':', lw=1.5, label='YaRN limit at pair 26')
    ax.set(xscale='log', title='Same pair, different response to scale (pair 26)',
           xlabel='Extension factor S', ylabel='Wavelength / native wavelength')
    ax.grid(alpha=.18); ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    omega, lo, hi, p = build(500000., 64, 8192, 4)
    slots = np.arange(lo+1, hi)
    for name in ('P', 'T', 'C'):
        nu = omega*4**(-p[name])
        hn = phase_distance(omega[slots]-nu[slots], np.pi/2)/8192
        he = phase_distance(nu[slots]-omega[slots]/4, np.pi/2)/8192
        ax.plot(slots, he, color=COLORS[name], lw=2, label=LABELS[name]+' vs full interpolation')
        ax.plot(slots, hn, color=COLORS[name], lw=1.3, ls='--')
    ax.axhline(4, color='black', lw=.7, ls=':')
    ax.set(yscale='log', title='Distance interval with phase deviation ≤ π/2 (S=4)',
           xlabel='Rotary pair index', ylabel='Maximum distance / native window')
    ax.text(.02,.02,'Dashed: relative to native; solid: relative to full interpolation',
            transform=ax.transAxes, fontsize=7.5)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left'); ax.grid(alpha=.18)
    fig.suptitle('Public-parameter geometry: scale response and distance-reference trade-offs', fontsize=13)
    fig.savefig(out/'midband_scale_response.png', dpi=200)
    fig.savefig(out/'midband_scale_response.svg')
    plt.close(fig)


def existing_task_summaries():
    """Descriptive regrouping of complete reports, without new significance claims."""
    reports = Path(__file__).parent/'reports'
    output = {}
    for name in ('clean16k_tailspline_vs_mrpro.json', 'clean_ruler200_tailspline_vs_mrpro.json'):
        data = json.loads((reports/name).read_text())
        length = str(data['lengths'][0])
        tasks = {arm:data['summaries'][arm]['by_length'][length]['tasks']
                 for arm in ('mrpro', 'tailspline')}
        per_task = {}
        for task in tasks['mrpro']:
            p, t = tasks['mrpro'][task], tasks['tailspline'][task]
            per_task[task] = {'P_percent':100*p['official'], 'T_percent':100*t['official'],
                              'delta_pp':100*(t['official']-p['official']),
                              'P_cap_rate':p['cap_rate'], 'T_cap_rate':t['cap_rate']}
        groups = {}
        for prefix in ('niah_single_', 'niah_multikey_'):
            names = [task for task in per_task if task.startswith(prefix)]
            if len(names) != 3:
                raise ValueError(f'Expected three tasks for {prefix}')
            groups[prefix] = {key:float(np.mean([per_task[task][key] for task in names]))
                              for key in ('P_percent', 'T_percent', 'delta_pp')}
        output[length] = {'source':f'experiments/iclr2027_three_track_sprint_20260915/reports/{name}',
                          'scope':'Post-hoc descriptive grouping; no new grouped confidence intervals.',
                          'tasks':per_task, 'groups':groups}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'CPU public-grid identities plus descriptive reuse of existing task reports; no new model evaluation.',
              'shared_band_models': {}, 'checks': checks(), 'Y_index_P_crossovers_n17': find_crossovers(),
              'existing_task_summaries':existing_task_summaries()}
    for model, length in (('Llama_public_grid', 8192), ('OLMo_public_grid', 4096)):
        result['shared_band_models'][model] = {
            str(scale): ratio_summary(500000., 64, length, scale) for scale in (2, 4, 8, 16)}
    result['status'] = 'PASS'
    (args.out/'midband_scale_response_checks.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    draw(args.out)
    print(json.dumps({'status':'PASS','checks':result['checks'],
                      'llama_max_T_P_wavelength_ratio':{s:x['max_T_P_wavelength_ratio'] for s,x in result['shared_band_models']['Llama_public_grid'].items()}}, indent=2))


if __name__ == '__main__': main()
