"""NumPy algebra check for shared K/V with inverse query rotation on output.

Fixed selected keys, fixed pre-rotation features, no quantization or model run.
This checks an operator identity; it is not a frequency optimizer or benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rotate(x, angles):
    """Identity on leading NoPE coordinates; interleaved pairs on trailing RoPE."""
    pairs = angles.shape[-1]
    out = x.copy()
    source = x[..., -2*pairs:].reshape(*x.shape[:-1], pairs, 2)
    target = out[..., -2*pairs:].reshape(*x.shape[:-1], pairs, 2)
    c, s = np.cos(angles), np.sin(angles)
    target[..., 0] = source[..., 0]*c-source[..., 1]*s
    target[..., 1] = source[..., 0]*s+source[..., 1]*c
    return out


def weights(q, k, scale, sink):
    logits = scale*np.einsum('hd,nd->hn', q, k, optimize=False)
    joined = np.concatenate([logits, np.asarray(sink)[:, None]], axis=1)
    maximum = joined.max(axis=1, keepdims=True)
    exp = np.exp(joined-maximum)
    return (exp/exp.sum(axis=1, keepdims=True))[:, :-1]


def absolute_forward(q, k, pq, pk, frequencies, scale, sink):
    q_rot = rotate(q, pq*frequencies)
    kv_rot = rotate(k, pk[:, None]*frequencies)
    a = weights(q_rot, kv_rot, scale, sink)
    rotated_output = np.einsum('hn,nd->hd', a, kv_rot, optimize=False)
    return rotate(rotated_output, -pq*frequencies)


def relative_terms(q, k, pq, pk, frequencies, direction, scale, sink):
    distance = pq-pk
    u = rotate(k, -distance[:, None]*frequencies)
    du = np.zeros_like(u)
    pairs = len(frequencies)
    selected = u[:, -2*pairs:].reshape(len(k), pairs, 2)
    target = du[:, -2*pairs:].reshape(len(k), pairs, 2)
    rate = -distance[:, None]*direction
    target[..., 0] = -rate*selected[..., 1]
    target[..., 1] = rate*selected[..., 0]
    a = weights(q, u, scale, sink)
    dz = scale*np.einsum('hd,nd->hn', q, du, optimize=False)
    mean_dz = (a*dz).sum(axis=1, keepdims=True)
    output = np.einsum('hn,nd->hd', a, u, optimize=False)
    routing = np.einsum('hn,nd->hd', a*(dz-mean_dz), u, optimize=False)
    transport = np.einsum('hn,nd->hd', a, du, optimize=False)
    return output, routing, transport, u, a


def verify():
    rng = np.random.default_rng(20260907)
    q, k = rng.normal(size=(3, 8)), rng.normal(size=(6, 8))
    pq, pk = 1000., np.array([0., 2., 20., 300., 600., 950.])
    freq, direction = np.array([.04, .009, .002]), rng.normal(size=3)*.001
    scale, sink = 8**-.5, np.array([-.4, .2, 1.])
    projected = rng.normal(size=(24, 5))
    epsilon = 1e-5
    output, routing, transport, u, a = relative_terms(q, k, pq, pk, freq, direction, scale, sink)
    absolute = absolute_forward(q, k, pq, pk, freq, scale, sink)
    shifted = absolute_forward(q, k, pq+12345, pk+12345, freq, scale, sink)
    numerical = (absolute_forward(q, k, pq, pk, freq+epsilon*direction, scale, sink)
                 -absolute_forward(q, k, pq, pk, freq-epsilon*direction, scale, sink))/(2*epsilon)
    full = routing+transport
    projected_difference = np.einsum('d,df->f', (full-numerical).ravel(), projected, optimize=False)
    # Query-coordinate Hessian-vector: alpha * covariance of the transported keys,
    # including a zero-valued sink. This is before q normalization/output projection.
    v = rng.normal(size=q.shape)
    uv = np.einsum('hd,nd->hn', v, u, optimize=False)
    hv = scale*(np.einsum('hn,nd->hd', a*uv, u, optimize=False)
                -output*(output*v).sum(axis=1, keepdims=True))
    numerical_hv = (absolute_forward(q+epsilon*v, k, pq, pk, freq, scale, sink)
                    -absolute_forward(q-epsilon*v, k, pq, pk, freq, scale, sink))/(2*epsilon)
    # With one selected key and no sink, attention is identically 1: the routing
    # derivative vanishes, while value transport still responds to frequency.
    _, one_routing, one_transport, _, _ = relative_terms(q[:1], k[:1], pq, pk[:1],
        freq, direction, scale, np.array([-np.inf]))
    result = dict(status='NUMPY_FIXED_OPERATOR_IDENTITY_CHECKED_NOT_MODEL_CAPABILITY',
        absolute_relative_max_abs=float(np.max(np.abs(output-absolute))),
        position_shift_max_abs=float(np.max(np.abs(absolute-shifted))),
        frequency_directional_derivative_max_abs=float(np.max(np.abs(full-numerical))),
        projected_derivative_max_abs=float(np.max(np.abs(projected_difference))),
        query_hessian_vector_max_abs=float(np.max(np.abs(hv-numerical_hv))),
        minimum_query_covariance_quadratic=float(np.min((v*hv).sum(axis=1))),
        one_key_routing_norm=float(np.sqrt(np.sum(one_routing**2))),
        one_key_value_transport_norm=float(np.sqrt(np.sum(one_transport**2))),
        limits='Fixed selection and unquantized pre-RoPE q/kv. No model state drift, selection derivative, weight training, gain change or capability optimization.')
    for key in ('absolute_relative_max_abs', 'position_shift_max_abs',
                'frequency_directional_derivative_max_abs', 'projected_derivative_max_abs',
                'query_hessian_vector_max_abs'):
        assert result[key] < 1e-7, (key, result[key])
    assert result['minimum_query_covariance_quadratic'] >= -1e-12
    assert result['one_key_routing_norm'] == 0 and result['one_key_value_transport_norm'] > 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = verify()
    result['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.out:
        with args.out.open('x') as f:
            f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
