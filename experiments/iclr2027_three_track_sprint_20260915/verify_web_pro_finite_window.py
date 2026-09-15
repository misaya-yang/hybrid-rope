"""Independent CPU checks of the September 15 Web Pro theory proposal.

No checkpoint, CUDA, task generation, fitting, or existing experiment mutation.
Requires numpy, scipy, and mpmath. Run with BLAS threads limited to one.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logsumexp, softmax

SEED = 20260915
RNG = np.random.default_rng(SEED)


def profiles(n):
    q = np.arange(n + 1, dtype=float)
    t = q * (3*n*n + 3*n + 1 - q*q) / (n*(n+1)*(2*n+1))
    p = q*(q+1)/(n*(n+1))
    w = 3*n/(2*(2*n+1))
    c = (1-w)*q/n + w*(2*q/n-p)
    return t, p, c


def tables(fp32=False):
    native = np.exp(-np.log(500000.)*np.arange(64)/64)
    if fp32:
        native = native.astype(np.float32).astype(float)
    turns = native * 8192/(2*np.pi)
    l = np.flatnonzero(turns > 32)[-1]
    h = np.flatnonzero(turns < 1)[0]
    assert (l, h) == (18, 35)
    output = {}
    for name, m in zip(("T", "P", "C"), profiles(h-l)):
        whole = np.r_[np.zeros(l), m, np.ones(63-h)]
        omega = native*4**(-whole)
        output[name] = omega.astype(np.float32).astype(float) if fp32 else omega
    return output


def rho(delta, length):
    # Principal differences avoid removable singularities at multiples of 2*pi.
    z = (np.asarray(delta) + np.pi) % (2*np.pi) - np.pi
    ratio = np.sinc(length*z/(2*np.pi))/np.sinc(z/(2*np.pi))
    value = 2 - 2*ratio*np.cos((length-1)*z/2)
    # Taylor evaluation prevents cancellation when checking tiny perturbations.
    a = (length-1)*(2*length-1)/6
    b = (length-1)*(2*length-1)*(3*length*length-3*length-1)/30
    return np.where(np.abs(length*z) < 1e-3, a*z*z-b*z**4/12, value)


def rotation(omega, d):
    return block_diag(*[np.array([[np.cos(w*d), -np.sin(w*d)],
                                 [np.sin(w*d), np.cos(w*d)]]) for w in omega])


def costs(omega, target, length):
    same = rho(target[:, None] - omega[None, :], length)
    flipped = rho(target[:, None] + omega[None, :], length)
    return np.minimum(same, flipped), same <= flipped


def best_orthogonal(omega, target, length):
    cost, same = costs(omega, target, length)
    rows, cols = linear_sum_assignment(cost)
    out = np.zeros((2*len(omega), 2*len(omega)))
    for i, j in zip(rows, cols):
        out[2*i:2*i+2, 2*j:2*j+2] = np.diag([1, 1 if same[i, j] else -1])
    return float(cost[rows, cols].mean()), out


def kernel_error(omega, target, length, s):
    si = np.linalg.inv(s)
    return float(sum(np.linalg.norm(s @ rotation(omega, d) @ si
                                   - rotation(target, d), "fro")**2
                     for d in range(length))/(2*len(omega)*length))


def block_capacity_slack(s):
    # The submatrix-capacity condition that guarantees a dominated bistochastic.
    k = len(s)//2
    w = np.array([[np.linalg.norm(s[2*i:2*i+2, 2*j:2*j+2], 'fro')**2/2
                   for j in range(k)] for i in range(k)])
    sigma = np.linalg.svd(s, compute_uv=False)[-1]**2
    slack = float('inf')
    for a in range(1, 1 << k):
        rows = [i for i in range(k) if a >> i & 1]
        for b in range(1, 1 << k):
            cols = [j for j in range(k) if b >> j & 1]
            if len(rows)+len(cols) > k:
                slack = min(slack, float(w[np.ix_(rows, cols)].sum()
                                        - sigma*(len(rows)+len(cols)-k)))
    return slack


def matching_checks():
    max_opt_error = 0.
    min_orth_slack = min_general_slack = min_capacity_slack = float('inf')
    cases = 240
    for i in range(cases):
        k, length = i % 4 + 1, i % 29 + 2
        a, b = RNG.uniform(.001, np.pi-.001, (2, k))
        if i % 8 == 0:
            b = a[RNG.permutation(k)]
        value, opt = best_orthogonal(a, b, length)
        max_opt_error = max(max_opt_error, abs(kernel_error(a, b, length, opt)-value))
        u = np.linalg.qr(RNG.normal(size=(2*k, 2*k)))[0]
        v = np.linalg.qr(RNG.normal(size=(2*k, 2*k)))[0]
        min_orth_slack = min(min_orth_slack, kernel_error(a, b, length, u)-value)
        singular = np.geomspace(1, 10**(i % 5), 2*k)
        s = (u*singular)@v.T
        bound = value/np.linalg.cond(s)**2
        min_general_slack = min(min_general_slack, kernel_error(a, b, length, s)-bound)
        min_capacity_slack = min(min_capacity_slack, block_capacity_slack(s))
    assert max_opt_error < 1e-11
    assert min_orth_slack > -1e-11 and min_general_slack > -1e-10
    assert min_capacity_slack > -1e-6
    # Reflections are necessary if opposite signed frequencies represent same pair.
    value, opt = best_orthogonal(np.array([.4]), np.array([-.4]), 16)
    assert value == 0 and kernel_error([.4], [-.4], 16, opt) < 1e-20
    return dict(cases=cases, max_orthogonal_attainment_error=max_opt_error,
                min_orthogonal_slack=min_orth_slack, min_general_slack=min_general_slack,
                min_capacity_slack=min_capacity_slack, condition_numbers_up_to=10000)


def cross_integral(x, y, length):
    a, b = (x-y)*length, (x+y)*length
    sinc = lambda z: np.sinc(z/np.pi)
    odd = lambda z: 0. if z == 0 else 2*np.sin(z/2)**2/z
    return .5*np.array([[sinc(a)+sinc(b), odd(b)-odd(a)],
                        [odd(b)+odd(a), sinc(a)-sinc(b)]])


def rank_integral(omega, length):
    # Independent origin-window implementation, using 2x2 linear solves.
    self_gram = [cross_integral(w, w, length) for w in omega]
    frame = 2*len(omega)
    for i in range(len(omega)):
        for j in range(i):
            h = cross_integral(omega[i], omega[j], length)
            frame += 2*np.trace(np.linalg.solve(self_gram[i], h)
                                @ np.linalg.solve(self_gram[j], h.T))
    return (2*len(omega))**2/frame


def rank_centered_domain(omega, length, discrete=False):
    # Shift the domain to its midpoint; each pair rotates orthogonally.
    d, s = omega[:, None]-omega[None, :], omega[:, None]+omega[None, :]
    if discrete:
        mean_cos = lambda z: np.sinc(length*z/(2*np.pi))/np.sinc(z/(2*np.pi))
    else:
        mean_cos = lambda z: np.sinc(length*z/(2*np.pi))
    cc, ss = (mean_cos(d)+mean_cos(s))/2, (mean_cos(d)-mean_cos(s))/2
    cc = cc/np.sqrt(np.diag(cc)[:, None]*np.diag(cc)[None, :])
    ss = ss/np.sqrt(np.diag(ss)[:, None]*np.diag(ss)[None, :])
    return (2*len(omega))**2/(np.square(cc).sum()+np.square(ss).sum())


def rank_direct(omega, length, triangular=False):
    d = np.arange(length)
    weights = 2*(length-d)/(length*(length+1)) if triangular else np.full(length, 1/length)
    phase = d[:, None]*omega[None, :]
    features = np.stack((np.cos(phase), np.sin(phase)), axis=-1).reshape(length, -1)
    for i in range(len(omega)):
        sl = slice(2*i, 2*i+2)
        f = features[:, sl]
        chol = np.linalg.cholesky(f.T@(weights[:, None]*f))
        features[:, sl] = np.linalg.solve(chol, f.T).T
    gram = features.T@(weights[:, None]*features)
    return np.trace(gram)**2/np.square(gram).sum()


def rank_checks():
    output, error = {}, 0.
    expected = {8192: [7.0682, 7.7161, 7.1044],
                16384: [8.2834, 8.7399, 8.3291],
                32768: [10.0773, 10.2149, 10.0926]}
    for length in expected:
        output[str(length)] = {}
        for name, omega in tables().items():
            origin = rank_integral(omega, length)
            symmetric = rank_centered_domain(omega, length)
            error = max(error, abs(origin-symmetric))
            integer = rank_centered_domain(omega, length, True)
            direct = rank_direct(omega, length)
            assert abs(integer-direct) < 1e-8
            output[str(length)][name] = dict(continuous_uniform=origin, integer_uniform=integer,
                integer_direct=direct, integer_causal_pair=rank_direct(omega, length, True),
                fp32_continuous=rank_centered_domain(tables(True)[name], length))
        got = [output[str(length)][n]['continuous_uniform'] for n in ('T', 'P', 'C')]
        assert np.max(np.abs(np.array(got)-expected[length])) < .00005
    assert error < 1e-8
    return dict(max_independent_continuous_error=error, windows=output)


def phase_checks():
    data = {}
    tab = tables()
    for length in (8192, 16384, 32768):
        data[str(length)] = {}
        for ref in ('P', 'C'):
            dist, _ = best_orthogonal(tab['T'], tab[ref], length)
            slot = np.mean(rho(tab['T']-tab[ref], length))
            data[str(length)][f'T/{ref}'] = dict(slot_rms=float(np.sqrt(slot)),
                best_orthogonal_rms=float(np.sqrt(dist)),
                max_unwrapped_phase=float((length-1)*max(abs(tab['T']-tab[ref]))))
    max_sum_error = 0.
    for length in (1, 2, 3, 16, 129):
        deltas = np.r_[0., 2*np.pi, -2*np.pi, RNG.uniform(-np.pi, np.pi, 100)]
        direct = np.mean(4*np.sin(np.arange(length)[:, None]*deltas/2)**2, axis=0)
        max_sum_error = max(max_sum_error, float(max(abs(rho(deltas, length)-direct))))
        if length > 1:
            near = RNG.uniform(0, np.pi/(length-1), 200)
            quadratic = (length-1)*(2*length-1)/6*near**2
            assert np.all(rho(near, length) <= quadratic+1e-10)
            assert np.all(rho(near, length) >= 4/np.pi**2*quadratic-1e-10)
            far = np.linspace(2*np.pi/length, np.pi, 200)
            assert min(rho(far, length)) >= 1-1e-10
    assert max_sum_error < 1e-11
    return dict(max_direct_sum_error=max_sum_error, comparisons=data)


def boundary_checks():
    t, p, c = profiles(17)
    native = np.exp(-np.log(500000.)*np.arange(64)/64)
    out = {}
    for name, m in zip(('T', 'P', 'C'), (t, p, c)):
        ef, et = m[1], 1-m[-2]
        front_slope = native[19]*(1-4**(-ef))
        tail_slope = native[34]/4*(4**et-1)
        out[name] = dict(entry_increment=float(ef), exit_increment=float(et),
            entry_monotone_distance_limit=float(np.pi/front_slope),
            exit_monotone_distance_limit=float(np.pi/tail_slope))
    for n in range(1, 65):
        t, p, c = profiles(n)
        q = np.arange(n+1)
        formula = q*(n-q)*(3*n+q+1)/(n*(n+1)*(2*n+1))
        assert max(abs(t-p-formula)) < 1e-14
        assert np.all(t-p >= -1e-14)
    # Centered feature eigenvalues: high precision avoids catastrophic cancellation.
    mp.mp.dps = 75
    asymptotic = []
    for text_x in ('0.1', '0.01', '0.001', '0.0001'):
        x = mp.mpf(text_x)
        cc = (1+mp.sin(x)/x)/2-(mp.sin(x/2)/(x/2))**2
        ss = (1-mp.sin(x)/x)/2
        asymptotic.append(dict(x=float(x), largest_ratio=float(ss/(x*x/12)),
                               smallest_ratio=float(cc/(x**4/720))))
    assert abs(asymptotic[-1]['smallest_ratio']-1) < 1e-8
    assert abs(asymptotic[-1]['largest_ratio']-1) < 1e-8
    # The proposed 3-slot perturbation is feasible only below a monotonicity bound.
    perturb = np.zeros(18); perturb[1:4] = [1, -2, 1]
    assert perturb.sum() == 0 and perturb[0] == perturb[-1] == 0
    increment_change = np.diff(perturb)
    active = increment_change != 0
    safe_radius = min(np.diff(profiles(17)[0])[active]/abs(increment_change[active]))
    return dict(boundary_phase=out, centered_eigenvalue_asymptotics=asymptotic,
                entry_three_slot_strict_amplitude_bound=float(safe_radius))


def content_checks():
    max_odds_error = max_path_error = max_readout_error = 0.
    for i in range(100):
        score, eta = RNG.normal(size=(2, 20))
        e = np.arange(20) < 7
        p, pt = softmax(score), softmax(score+eta)
        logodds = lambda z: logsumexp(z[e])-logsumexp(z[~e])
        rhs = logsumexp(score[e]+eta[e])-logsumexp(score[e])
        rhs -= logsumexp(score[~e]+eta[~e])-logsumexp(score[~e])
        max_odds_error = max(max_odds_error, abs(logodds(score+eta)-logodds(score)-rhs))
        favorable = np.where(e, .5+RNG.random(20), -.5-RNG.random(20))
        assert softmax(score+favorable)[e].sum() >= expit(logodds(score)+1)-1e-13
        value = np.where(e, 2., -1.)
        max_readout_error = max(max_readout_error,
                               abs(value@(pt-p)-3*(pt[e].sum()-p[e].sum())))
        t, _, c = profiles(17)
        delta = t-c
        omega = np.exp(-np.arange(18)/4)
        distance = RNG.uniform(0, 100, 20)
        amplitude = RNG.uniform(.1, 2, (20, 18))
        phase = RNG.uniform(-np.pi, np.pi, (20, 18))
        def at(time):
            theta = distance[:, None]*omega*4**(-(c+time*delta))
            ell = (amplitude*np.cos(phase+theta)).sum(axis=1)
            u = amplitude*theta*np.sin(phase+theta)
            h = softmax(ell[e])@u[e]-softmax(ell[~e])@u[~e]
            paired = sum(-delta[q]*(h[17-q]-h[q]) for q in range(1, 9))*np.log(4)
            return logodds(ell), paired
        time, step = .4, 1e-6
        finite = (at(time+step)[0]-at(time-step)[0])/(2*step)
        max_path_error = max(max_path_error, abs(finite-at(time)[1]))
    assert max_odds_error < 1e-12 and max_path_error < 1e-7 and max_readout_error < 1e-12
    # Missing zero-mean assumption in the proposal: covariance alone is insufficient.
    feature_difference = np.array([1., 0.])
    mean = np.array([2., 0.]); covariance = np.eye(2)
    stated = float(feature_difference@covariance@feature_difference)
    correct = stated+float(feature_difference@mean)**2
    return dict(cases=100, max_finite_odds_error=max_odds_error,
                max_phase_path_derivative_error=max_path_error, max_readout_error=max_readout_error,
                nonzero_mean_counterexample=dict(covariance_only=stated, actual_second_moment=correct))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = dict(seed=SEED, scope='Public-parameter algebra and CPU numerical checks; no model evaluation.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name, check in (('matching', matching_checks), ('rank', rank_checks),
                        ('phase', phase_checks), ('boundary', boundary_checks), ('content', content_checks)):
        result[name] = check()
        print(f'{name}: PASS', flush=True)
    result['status'] = 'PASS'
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
