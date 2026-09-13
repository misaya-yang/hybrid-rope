#!/usr/bin/env python3
"""CPU checks for exact RoPE design identities; no model or task evidence."""
import argparse
import cmath
import json
import math
from pathlib import Path


def differences(values):
    return [b - a for a, b in zip(values, values[1:])]


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
              'three_gap_stencil': {'indices_zero_based': [6, 7, 8], 'values': [1, -2, 1],
                                   'feasible_example_eta': eta}}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
