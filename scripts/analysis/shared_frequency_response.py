"""Exact frozen-QKV response to a shared RoPE table; NumPy, no model forward.

The Jacobian retains cancellation across keys and query heads. It describes a
local attention block, not a downstream loss gradient or a method selector.
"""
from __future__ import annotations

import argparse
import json
import numpy as np


def response(q, k, v, projection, frequencies, positions, *, gain=1.0,
             parameter_scale=None, compute_jacobian=True):
    """Return output, shared-frequency Jacobian, and independent-noise energy.

    Layouts: q[H,Q,D], k/v[G,L,D], projection[O,H*D]; split-half rotary pairs.
    Coordinates satisfy nu = frequencies + parameter_scale * delta. Raw
    frequency derivatives use scale=1; Native frequencies provide relative-to-
    Native coordinates that remain usable when a deployment frequency is zero.
    Set compute_jacobian=False for finite table-change replay; it returns only
    output and avoids constructing unused derivatives and diagnostic energies.
    """
    q, k, v, projection, frequencies = [np.asarray(x, dtype=np.float64)
                                       for x in (q, k, v, projection, frequencies)]
    positions = np.asarray(positions)
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("QKV layout")
    heads, queries, dim = q.shape
    groups, length, key_dim = k.shape
    pairs = dim // 2
    if heads == 0 or queries == 0 or dim == 0 or dim % 2 or key_dim != dim or groups == 0 or heads % groups:
        raise ValueError("rotary/GQA layout")
    if projection.ndim != 2 or projection.shape[1] != heads * dim:
        raise ValueError("output projection layout")
    if frequencies.shape != (pairs,) or positions.shape != (queries,):
        raise ValueError("frequency/query positions")
    if np.any(positions != positions.astype(int)) or np.any(positions < 0) or np.any(positions >= length):
        raise ValueError("positions must index the cached causal sequence")
    scale = np.ones(pairs) if parameter_scale is None else np.asarray(parameter_scale, dtype=float)
    if scale.shape != (pairs,) or not np.isfinite(scale).all():
        raise ValueError("parameter scale")
    if not all(np.isfinite(x).all() for x in (q, k, v, projection, frequencies)) or not np.isfinite(gain):
        raise ValueError("nonfinite inputs")
    output = np.zeros((queries, projection.shape[0]))
    if compute_jacobian:
        jacobian = np.zeros((*output.shape, pairs))
        independent_energy = np.zeros((queries, pairs))
        head_coherent_energy = np.zeros_like(independent_energy)
    factor = gain ** 2 / np.sqrt(dim)
    # A shared table has the same phases for every query head. Reuse them;
    # recomputing these trigonometric arrays per head dominates CPU replay.
    phase_cache = []
    for pos in positions.astype(int):
        distance = pos - np.arange(pos+1)
        phase = distance[:, None] * frequencies
        phase_cache.append((distance, np.cos(phase), np.sin(phase)))
    for h in range(heads):
        kh, vh = k[h // (heads // groups)], v[h // (heads // groups)]
        wh = projection[:, h*dim:(h+1)*dim]
        gram = wh.T @ wh if compute_jacobian else None
        if compute_jacobian:
            value_energy = ((vh @ gram) * vh).sum(axis=1)
        for t, pos in enumerate(positions.astype(int)):
            # Causal slicing avoids masked 0 * NaN and excludes unseen keys.
            keys, values = kh[:pos+1], vh[:pos+1]
            distance, cosine, sine = phase_cache[t]
            query = q[h, t]
            c = query[:pairs] * keys[:, :pairs] + query[pairs:] * keys[:, pairs:]
            # With distance = query_position - key_position and rotate_half
            # (-x_second, x_first), q_rot @ k_rot uses this sign for sine.
            s = query[:pairs] * keys[:, pairs:] - query[pairs:] * keys[:, :pairs]
            logits = factor * (c*cosine + s*sine).sum(axis=1)
            probability = np.exp(logits - logits.max())
            probability /= probability.sum()
            head_output = probability @ values
            output[t] += wh @ head_output
            if not compute_jacobian:
                continue
            dz = factor * (-c*sine + s*cosine) * distance[:, None] * scale
            weighted = probability[:, None] * dz
            # Sum signed key responses BEFORE taking a norm or Gram product.
            local_jacobian = values.T @ weighted - np.outer(head_output, weighted.sum(axis=0))
            projected_jacobian = wh @ local_jacobian
            jacobian[t] += projected_jacobian
            head_coherent_energy[t] += np.square(projected_jacobian).sum(axis=0)
            projected_mean = gram @ head_output
            energy = value_energy[:pos+1] - 2 * (values @ projected_mean) + head_output @ projected_mean
            independent_energy[t] += (np.square(weighted) * np.maximum(energy, 0)[:, None]).sum(axis=0)
    if not compute_jacobian:
        return dict(output=output)
    return dict(output=output, jacobian=jacobian,
                shared_gram=np.einsum('qoj,qok->jk', jacobian, jacobian) / queries,
                independent_noise_diagonal=independent_energy.mean(axis=0),
                within_head_diagonal=head_coherent_energy.mean(axis=0))


def self_check():
    rng = np.random.default_rng(20260907)
    q = rng.normal(size=(4, 2, 6))
    k, v = rng.normal(size=(2, 7, 6)), rng.normal(size=(2, 7, 6))
    projection = rng.normal(size=(5, 24))
    frequencies, scale, positions = np.array([.7, .17, 0.]), np.array([1., .3, .1]), np.array([3, 6])
    args = (q, k, v, projection)
    actual = response(*args, frequencies, positions, gain=1.13, parameter_scale=scale)

    def direct_rotation_output(freq):
        # Independent oracle: rotate Q and K at absolute positions, then use
        # ordinary causal GQA. Differencing response() itself cannot detect a
        # consistent sign error in both its output and analytic derivative.
        def rotate(x, pos):
            phase = pos[:, None] * freq
            cosine = np.concatenate((np.cos(phase), np.cos(phase)), axis=-1)
            sine = np.concatenate((np.sin(phase), np.sin(phase)), axis=-1)
            half = x.shape[-1] // 2
            rotated_half = np.concatenate((-x[..., half:], x[..., :half]), axis=-1)
            return x * cosine + rotated_half * sine

        rotated_q = rotate(q, positions)
        rotated_k = np.repeat(rotate(k, np.arange(k.shape[1])), 2, axis=0)
        repeated_v = np.repeat(v, 2, axis=0)
        logits = rotated_q @ rotated_k.transpose(0, 2, 1) * (1.13**2 / np.sqrt(q.shape[-1]))
        logits = np.where(positions[:, None] >= np.arange(k.shape[1]), logits, -np.inf)
        probability = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probability /= probability.sum(axis=-1, keepdims=True)
        outputs = (probability @ repeated_v).transpose(1, 0, 2).reshape(positions.size, -1)
        return outputs @ projection.T

    rotation_error = float(np.max(np.abs(actual['output'] - direct_rotation_output(frequencies))))
    if rotation_error > 1e-12:
        raise AssertionError(f"split-half rotation convention mismatch: {rotation_error}")
    finite_output = response(*args, frequencies, positions, gain=1.13, compute_jacobian=False)['output']
    if not np.array_equal(finite_output, actual['output']):
        raise AssertionError("output-only replay differs")
    differences = []
    for j in range(3):
        step = np.zeros(3); step[j] = 1e-6 * scale[j]
        plus = direct_rotation_output(frequencies+step)
        minus = direct_rotation_output(frequencies-step)
        fd = (plus-minus) / 2e-6
        differences.append(float(np.max(np.abs(fd-actual['jacobian'][:, :, j]))))
    if max(differences) > 1e-7:
        raise AssertionError(differences)
    if np.linalg.eigvalsh(actual['shared_gram']).min() < -1e-10:
        raise AssertionError("Jacobian Gram must be PSD")
    # Two identical heads with opposite output projections cancel exactly.
    qc = np.repeat(q[:1], 2, axis=0)
    wh = projection[:, :6]
    cancellation = response(qc, k[:1], v[:1], np.concatenate((wh, -wh), axis=1),
                            frequencies, positions, parameter_scale=scale)
    if not np.allclose(cancellation['shared_gram'], 0, atol=1e-25):
        raise AssertionError("shared head cancellation lost")
    if not cancellation['within_head_diagonal'].sum() > 0:
        raise AssertionError("example must have nonzero individual-head response")
    return dict(status='NUMPY_ALGEBRA_CHECKED_NO_MODEL_RUN',
                direct_rotation_max_abs=rotation_error,
                finite_difference_max_abs=max(differences),
                zero_frequency_derivative_norm=float(np.linalg.norm(actual['jacobian'][:, :, 2])),
                canceling_heads_shared_energy=float(np.trace(cancellation['shared_gram'])),
                canceling_heads_individual_energy=float(cancellation['within_head_diagonal'].sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_check(), indent=2))
