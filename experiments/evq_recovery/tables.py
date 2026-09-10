"""Fixed-support shape controls and a separately labelled YaRN reference."""
from __future__ import annotations

import hashlib
import math
import numpy as np


def digest_array(values):
    return hashlib.sha256(np.asarray(values, dtype='<f4').tobytes()).hexdigest()


def anchored_cosh(k, tau):
    u = (np.arange(k, dtype=np.float64) + .5) / k
    q = u if abs(tau) < 1e-7 else 1 - np.arcsinh((1-u)*np.sinh(tau))/tau
    return (q-q[0])/(q[-1]-q[0])


def exponential(k, strength):
    u = np.linspace(0., 1., k)
    if strength < 1e-7:
        return u
    return -np.log1p(-u*(-np.expm1(-strength)))/strength


def hybrid(k, tau):
    # Preserve 16 high-frequency pairs on this K=64 model, following the old r=16 control.
    u = np.linspace(0., 1., k)
    split = k//4
    out = u.copy()
    out[split:] = u[split] + (1-u[split])*anchored_cosh(k-split, tau)
    return out


def match_deformation(builder, k, rms):
    native = np.linspace(0., 1., k)
    lo, hi = 0., 24.
    if np.sqrt(np.mean((builder(k, hi)-native)**2)) < rms:
        raise ValueError('requested deformation unavailable in control family')
    for _ in range(80):
        mid = (lo+hi)/2
        if np.sqrt(np.mean((builder(k, mid)-native)**2)) < rms:
            lo = mid
        else:
            hi = mid
    value = (lo+hi)/2
    return builder(k, value), value


def construct(config, native_values, tau=2., scale=4.):
    native = np.asarray(native_values, dtype=np.float32)
    k = len(native)
    if k < 8 or not np.all(native[:-1] > native[1:]) or np.any(native <= 0):
        raise ValueError('positive ordered native frequencies required')
    logfreq = -np.log(native.astype(np.float64))
    base_z = np.linspace(0., 1., k)
    z = anchored_cosh(k, tau)
    rms = float(np.sqrt(np.mean((z-base_z)**2)))
    exp_z, lam = match_deformation(exponential, k, rms)
    hybrid_z, hybrid_tau = match_deformation(hybrid, k, rms)
    result = {}
    for name, nodes, parameters in [
        ('Native', base_z, {}), ('Cosh', z, {'tau': tau}),
        ('Exponential', exp_z, {'lambda': lam}),
        ('Hybrid', hybrid_z, {'tau': hybrid_tau, 'native_high_pairs': k//4}),
    ]:
        values = np.exp(-(logfreq[0]+(logfreq[-1]-logfreq[0])*nodes)).astype(np.float32)
        values[[0, -1]] = native[[0, -1]]
        if name == 'Native':
            values = native.copy()
        if not np.all(values[:-1] > values[1:]):
            raise ValueError(f'{name}: invalid frequency order')
        result[name] = dict(values=values.tolist(), sha256=digest_array(values), gain=1.,
                            scope='same native endpoints; only interior shape changes',
                            rms_normalized_deformation=float(np.sqrt(np.mean((nodes-base_z)**2))),
                            parameters=parameters)
    # Use the repository's previously parity-tested official operator.
    import torch
    from scripts.lib.rope.official_yarn import official_yarn_on_inv_freq
    base = config.get('rope_theta', config.get('rope_parameters', {}).get('rope_theta'))
    if base is None:
        raise ValueError('missing native RoPE base')
    values, gain, meta = official_yarn_on_inv_freq(torch.from_numpy(native.copy()),
        head_dim=k*2, base=base, scale=scale,
        original_max_position_embeddings=config['max_position_embeddings'])
    values = values.float().numpy()
    result['YaRN'] = dict(values=values.tolist(), sha256=digest_array(values), gain=float(gain),
        scope='practical extension reference; differs in range and gain; not a pure shape control',
        parameters=meta)
    return result
