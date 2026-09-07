"""Explicit static transforms, with native/derived identities kept separate."""
from __future__ import annotations

import hashlib
import math
import numpy as np
import torch
from scripts.lib.rope.official_yarn import official_yarn_on_inv_freq


def tensor_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<f4').tobytes()).hexdigest()


def native_table(dim, base):
    return (1 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))).numpy()


def check_table(values, dim):
    a = np.asarray(values)
    if a.dtype != np.float32 or a.shape != (dim // 2,) or dim % 2:
        raise ValueError('table dtype/shape mismatch')
    if not np.isfinite(a).all() or not np.all(a > 0) or not np.all(a[:-1] > a[1:]):
        raise ValueError('table must be finite, positive and strictly descending')


def transform(source, *, dim, base, reference_length, scale, method):
    check_table(source, dim)
    if not math.isfinite(scale) or scale < 1 or base <= 1 or reference_length <= 0:
        raise ValueError('invalid transform geometry')
    if method == 'identity':
        return source.copy(), 1.0, {'method': 'identity'}
    if method == 'yarn':
        a, gain, meta = official_yarn_on_inv_freq(
            torch.from_numpy(source.copy()), head_dim=dim, base=base, scale=scale,
            original_max_position_embeddings=reference_length)
        return a.float().numpy(), gain, meta
    if method not in ('mrpro', 'mruni'):
        raise ValueError('unknown transform')
    reference = native_table(dim, base).astype(np.float64)
    turns = reference * reference_length / (2 * math.pi)
    fast, slow = np.flatnonzero(turns > 32), np.flatnonzero(turns < 1)
    if not len(fast) or not len(slow):
        raise ValueError('MrRoPE reference lacks 32/1-turn boundaries')
    lo, hi = int(fast[-1]), int(slow[0])
    if hi <= lo:
        raise ValueError('empty MrRoPE band')
    n = hi - lo
    t = np.clip(np.arange(dim // 2) - lo, 0, n)
    exponent = t / n if method == 'mruni' else t * (t + 1) / (n * (n + 1))
    result = (source.astype(np.float64) / scale ** exponent).astype(np.float32)
    native = np.array_equal(source, native_table(dim, base))
    return result, 1 + .1 * math.log(scale), {
        'method': method, 'reference': 'native_endpoint_grid', 'low': lo, 'high': hi,
        'label': 'MrRoPE paper-equation reproduction' if native else 'shared-reference MrRoPE-component control',
        'cumulative_exponents': exponent.tolist(), 'source': 'arxiv:2601.22181v1 eq13-16,26',
    }


def install_static(model, values, gain):
    """Install after model dtype/device conversion; preserve FP32 phase accuracy."""
    rotary = model.model.rotary_emb
    if getattr(rotary, 'rope_type', 'default') != 'default':
        raise ValueError('dynamic/scaled checkpoint unsupported; cannot silently stack transforms')
    dim = model.config.hidden_size // model.config.num_attention_heads
    check_table(values, dim)
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError('invalid rotary amplitude')
    rotary.inv_freq = torch.from_numpy(values.copy()).to(device=next(model.parameters()).device)
    rotary.original_inv_freq = rotary.inv_freq.clone()
    rotary.attention_scaling = float(gain)
    return rotary


def verify_static(model, values, gain):
    r = model.model.rotary_emb
    if r.inv_freq.dtype != torch.float32 or tensor_sha(r.inv_freq.detach().cpu().numpy()) != tensor_sha(values):
        raise RuntimeError('runtime table drift')
    if float(r.attention_scaling) != float(gain):
        raise RuntimeError('runtime amplitude drift')
