#!/usr/bin/env python3
"""CPU checks for exact RoPE design identities; no model or task evidence."""
import argparse
import cmath
import itertools
import json
import math
from pathlib import Path


def differences(values):
    return [b - a for a, b in zip(values, values[1:])]


def verify_scale_transport():
    """Analytic band coordinates; not a benchmark-selected deployment table."""
    cases = []
    for label, base, pairs, low, high, native_length in (
        ('llama', 500000.0, 64, 16, 34, 8192),
        ('olmo', 500000.0, 64, 14, 31, 4096),
        ('qwen', 1000000.0, 64, 22, 39, 32768),
    ):
        n = high - low
        band_span = n * math.log(base) / pairs
        t = [q / n for q in range(n + 1)]
        pro = [q * (q + 1) / (n * (n + 1)) for q in range(n + 1)]
        bm = [q * (q + 1) * (3*n + 2 - 2*q) /
              (n * (n + 1) * (n + 2)) for q in range(n + 1)]
        reverse = [2*x - p for x, p in zip(t, pro)]
        mix = [0.25*b + 0.75*r for b, r in zip(bm, reverse)]
        for q in range(1, n):
            assert mix[q] > bm[q] > pro[q]
            exact_offset = q*(n-q)/(n*(n+1)) * (0.75 + 0.25*(2*q-n)/(n+2))
            assert abs(mix[q] - t[q] - exact_offset) < 1e-14
            assert mix[q] > t[q]
        old_s, new_s = 4.0, 8.0
        old_b, new_b = math.log(old_s), math.log(new_s)
        alpha = old_b*(band_span + new_b)/(new_b*(band_span + old_b))
        transported = [alpha*m + (1-alpha)*x for m, x in zip(mix, t)]
        old_u = [(band_span*x + old_b*m)/(band_span+old_b) for x,m in zip(t,mix)]
        new_u = [(band_span*x + new_b*m)/(band_span+new_b) for x,m in zip(t,transported)]
        naive_u = [(band_span*x + new_b*m)/(band_span+new_b) for x,m in zip(t,mix)]
        error = max(abs(a-b) for a,b in zip(old_u,new_u))
        assert error < 1e-14 and 0 < alpha < 1
        assert all(x > 0 for x in differences(transported))
        for q in range(1,n):
            assert t[q] < transported[q] < mix[q]
            # Native continuous-lag operator worst error is monotone in |nu-omega|.
            omega = base ** (-(low+q)/pairs)
            nu_old = omega * new_s ** (-mix[q])
            nu_new = omega * new_s ** (-transported[q])
            for length in (8.0, 4096.0, 8192.0):
                worst_old = 2*math.sin(min(length*(omega-nu_old),math.pi)/2)
                worst_new = 2*math.sin(min(length*(omega-nu_new),math.pi)/2)
                assert worst_new <= worst_old + 1e-14
        # Scale transport is transitive (4 -> 8 -> 16 equals 4 -> 16).
        b16 = math.log(16.0)
        a816 = new_b*(band_span+b16)/(b16*(band_span+new_b))
        a416 = old_b*(band_span+b16)/(b16*(band_span+old_b))
        assert abs(a816*alpha-a416) < 1e-14
        q = n // 2
        native_phase_error = [native_length*base**(-(low+j)/pairs)*(1-new_s**(-transported[j]))
                              for j in range(1,n)]
        assert min(native_phase_error) > math.pi
        cases.append({
            'model_geometry': label, 'band': [low, high], 'native_band_log_span': band_span,
            'source_scale': old_s, 'target_scale': new_s, 'alpha': alpha,
            'band_u_preservation_max_error': error,
            'naive_band_u_drift_max': max(abs(a-b) for a,b in zip(old_u,naive_u)),
            'middle_slot': low+q, 'middle_pro_m': pro[q], 'middle_mix_m': mix[q],
            'middle_transported_m': transported[q],
            'middle_mix_over_pro_frequency_s4': old_s**(pro[q]-mix[q]),
            'middle_mix_over_pro_frequency_s8': new_s**(pro[q]-mix[q]),
            'max_transport_over_naive_frequency_s8': max(new_s**(a-b) for a,b in zip(mix,transported)),
            'min_transported_native_unwrapped_phase_error': min(native_phase_error),
            'unsaturated_native_operator_bound_slots': sum(x < math.pi for x in native_phase_error),
            'operator_bound_decision': 'all interior slots saturate at 2; no native-window ranking certificate',
            'source_m': mix, 'transported_m': transported,
            'scope': 'band-relative u fixed; full-spectrum z is not held fixed',
        })
    return cases


def verify_functional_interval():
    """Exact sign interval examples and their limitation, with no model claim."""
    # For -sin(nu*d), d in [0,H], nonpositivity iff 0 <= nu*H <= pi.
    length, scale = 4096.0, 8.0
    examples = []
    for native_phase in (0.2, 0.75*math.pi, math.pi):
        omega = native_phase / length
        cap = min(omega, math.pi/(scale*length))
        assert cap*scale*length <= math.pi + 1e-14
        assert min(-math.sin(cap*d*scale*length/1000) for d in range(1001)) >= -1-1e-14
        assert max(-math.sin(cap*d*scale*length/1000) for d in range(1001)) <= 1e-14
        examples.append({'native_phase': native_phase, 'least_change_scale_ratio': cap/omega,
                         'pi_scale_ratio': 1/scale})
    # Positive semantic cos requires a different cap: pi/(2H), not pi/H.
    omega = 0.75*math.pi/length
    sin_cap = math.pi/(scale*length)
    cosine_at_sin_cap_endpoint = math.cos(sin_cap*scale*length)
    assert cosine_at_sin_cap_endpoint < 0
    # Freezing Q/K is essential. A changed content phase can reverse any sign claim.
    return {'negative_sine_examples': examples,
            'positive_cosine_at_negative_sine_cap': cosine_at_sin_cap_endpoint,
            'scope': 'fixed signed component over a continuous lag interval, not full-model utility'}


def minimal_ordered_profile(lower, upper):
    """Coordinatewise minimum satisfying interval bounds and ordered exponents."""
    if len(lower) != len(upper) or not lower:
        raise ValueError('nonempty equal-length bounds required')
    result = []
    running = 0.0
    for lo, hi in zip(lower, upper):
        if not math.isfinite(lo) or math.isnan(hi) or lo < 0 or hi < lo:
            raise ValueError('invalid exponent interval')
        running = max(running, lo)
        if running > hi:
            return None
        result.append(running)
    return result


def verify_ordered_intervals():
    # Enumerate independently: all integer bound pairs and all ordered profiles.
    # Discrete examples include zero plateaux, compression >1, and infeasibility.
    grid = range(4)
    intervals = [(a,b) for a in grid for b in grid if a <= b]
    profiles = list(itertools.combinations_with_replacement(grid, 3))
    feasible_count = 0
    for bounds in itertools.product(intervals, repeat=3):
        lower, upper = zip(*bounds)
        actual = minimal_ordered_profile(lower, upper)
        feasible = [v for v in profiles if all(a <= x <= b for x,(a,b) in zip(v,bounds))]
        assert (actual is None) == (not feasible)
        if feasible:
            feasible_count += 1
            assert tuple(actual) in feasible
            assert all(all(a <= b for a,b in zip(actual,v)) for v in feasible)
    return {'bound_systems_checked': len(intervals)**3, 'feasible_systems': feasible_count,
            'scope': 'exhaustive small-system check of the proved interval theorem'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    n, scale = 17, 4.0
    profiles = {
        'index_ramp_yarn': [-math.log(1 - (1 - 1 / scale) * q / n, scale)
                            for q in range(n + 1)],
        'mrpro': [q * (q + 1) / (n * (n + 1)) for q in range(n + 1)],
        'uniform': [q / n for q in range(n + 1)],
        'bm': [q * (q + 1) * (3 * n + 2 - 2 * q) /
               (n * (n + 1) * (n + 2)) for q in range(n + 1)],
    }
    summaries = {}
    for name, m in profiles.items():
        eps = differences(m)
        assert abs(sum(eps) - 1) < 1e-12
        assert abs(sum(m[1:-1]) - sum((n - q) * eps[q - 1]
                                      for q in range(1, n + 1))) < 1e-12
        summaries[name] = {'sum_m_interior': sum(m[1:-1]),
                           'first_increment': eps[0], 'last_increment': eps[-1],
                           'increments_increasing': all(x > 0 for x in differences(eps))}
    assert summaries['index_ramp_yarn']['increments_increasing']
    paper_ramps = {}
    for s in (4.0, 16.0, 32.0, 64.0):
        r = [32 ** (1 - q / n) for q in range(n + 1)]
        m = [-math.log((32 - s + (s - 1) * v) / (31 * s), s) for v in r]
        second = differences(differences(m))
        paper_ramps[str(s)] = {'min_second_difference': min(second),
                                'max_second_difference': max(second)}
        if s < 32:
            assert max(second) < 0
        elif s == 32:
            assert max(map(abs, second)) < 1e-12
        else:
            assert min(second) > 0
    # Fixed endpoint allocation identity in the paper's normalized z coordinate.
    k_count, base = 64, 500000.0
    g0 = math.log(base) / k_count
    m = [0.0] * 14 + profiles['bm'] + [1.0] * (64 - 14 - 18)
    nu = [math.exp(-k * g0) * scale ** (-v) for k, v in enumerate(m)]
    span = (k_count - 1) * g0 + math.log(scale)
    z_error = max(abs(math.log(nu[0] / nu[k]) / span -
                      (k * g0 + m[k] * math.log(scale)) / span) for k in range(k_count))
    assert z_error < 1e-12
    # Three-gap perturbation preserves mass and first moment, and is nontrivial.
    eps = differences(profiles['bm'])
    stencil = [0.0] * n
    stencil[6:9] = [1.0, -2.0, 1.0]
    eta = 0.25 * min(eps[6], eps[7], eps[8])
    moved = [a + eta * b for a, b in zip(eps, stencil)]
    assert min(moved) > 0 and abs(sum(moved) - sum(eps)) < 1e-12
    assert abs(sum(q * (a - b) for q, (a, b) in enumerate(zip(moved, eps)))) < 1e-12
    # Reversal of a strictly increasing multiset strictly changes its first moment.
    pro = differences(profiles['mrpro'])
    assert sum(q * v for q, v in enumerate(pro)) > sum(q * v for q, v in enumerate(reversed(pro)))
    # Any nonzero phase difference admits both signs of score response via C.
    delta = cmath.exp(0.7j) - cmath.exp(0.6j)
    positive = (delta.conjugate() * delta).real
    negative = (-delta.conjugate() * delta).real
    assert positive > 0 and negative < 0
    # Exact finite response, derivative, and sine bound for one frozen Q/K term.
    c, distance, freq, perturb = complex(0.4, -0.8), 100.0, 0.013, 1e-6
    score = lambda x: (c * cmath.exp(1j * distance * math.exp(x))).real
    x = math.log(freq)
    derivative = -distance * freq * (c * cmath.exp(1j * distance * freq)).imag
    derivative_error = abs((score(x + perturb) - score(x - perturb)) / (2 * perturb) - derivative)
    assert derivative_error < 1e-7
    actual = abs(score(x + 0.1) - score(x))
    bound = 2 * abs(c) * abs(math.sin(distance * freq * math.expm1(0.1) / 2))
    assert actual <= bound + 1e-12
    result = {'status': 'CPU_IDENTITIES_VERIFIED', 'model_execution': False,
              'scope': 'standard-library algebra and constructed examples, not task prediction',
              'profiles_n17_s4': summaries, 'rotation_ramp_yarn': paper_ramps,
              'z_identity_max_error': z_error, 'finite_response_derivative_error': derivative_error,
              'signed_response_counterexample': [positive, negative],
              'scale_transport': verify_scale_transport(),
              'functional_interval_examples': verify_functional_interval(),
              'ordered_interval_solver': verify_ordered_intervals(),
              'three_gap_stencil': {'indices_zero_based': [6, 7, 8], 'values': [1, -2, 1],
                                   'feasible_example_eta': eta}}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
