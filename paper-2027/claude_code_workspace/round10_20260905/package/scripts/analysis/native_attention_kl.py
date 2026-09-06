"""Finite FP64 Native→candidate attention KL on the same pre-RoPE Q/K.

This is a mathematical diagnostic on Native latent vectors, not bitwise BF16
kernel equivalence, a V/downstream-feedback model, a task-loss predictor, or a
profile selector. No weights, fitting parameters, or intervention are derived.

Keys occupy positions 0..T-1. Each selected query sees *all* keys through its
own position, including itself. GQA repeats each KV head over a contiguous
group of query heads, and RoPE pairs dimension i with dimension i+D/2.

``gain`` is the candidate cos/sin amplitude on BOTH Q and K, hence gain**2 in
logits. Native amplitude is one; ``attention_scale`` multiplies both logits.
For slot i, the signed difference is candidate-minus-Native, INCLUDING gain.
All reported moments use the full Native attention distribution p. Thus
off_diagonal_cancellation = Var_p(sum_i delta_i) - sum_i Var_p(delta_i);
negative means cancellation, positive means reinforcement.

The separately reported gain difference is (gain**2-1) times the ungained
active-frequency logit. It is already included in the slot/total differences;
do not add it again or interpret it as an additive share of KL.
"""

from __future__ import annotations

import math

import numpy as np


def _real_array(value, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _slot_logits(query, keys, relative_positions, inv, attention_scale):
    pairs = len(inv)
    a = query[:pairs] * keys[:, :pairs] + query[pairs:] * keys[:, pairs:]
    b = query[:pairs] * keys[:, pairs:] - query[pairs:] * keys[:, :pairs]
    phase = relative_positions[:, None] * inv[None, :]
    return attention_scale * (a * np.cos(phase) + b * np.sin(phase))


def _log_softmax(logits):
    shifted = logits - np.max(logits)
    return shifted - np.log(np.exp(shifted).sum())


def _moments(values, probabilities):
    mean = probabilities @ values
    variance = probabilities @ np.square(values - mean)
    return mean, variance


def native_attention_kl(q, k, query_positions, native_inv, active_inv, *,
                        gain: float = 1.0, attention_scale: float | None = None) -> dict:
    """Return per-head/query KL and Native-p-weighted mechanical moments.

    q: [Hq,Q,D]; k: [Hkv,T,D]; query_positions: integer [Q] in [0,T).
    native_inv/active_inv: positive [D/2]. All analysis is float64; inputs are
    never mutated. Default attention_scale is 1/sqrt(D). Mean KL gives every
    requested head/query equal weight. Only specified queries are evaluated:
    working score storage is O(T*D), never an implicit all-token T-by-T matrix.

    Returns ``kl[Hq,Q]``, ``mean_kl``, slot mean/variance ``[Hq,Q,D/2]``, and
    total variance, signed off-diagonal cancellation, and gain mean/variance
    ``[Hq,Q]``. The gain attribution path is fixed in the module docstring.
    """
    q, k = _real_array(q, "q"), _real_array(k, "k")
    native_inv = _real_array(native_inv, "native_inv")
    active_inv = _real_array(active_inv, "active_inv")
    positions = np.asarray(query_positions)
    if q.ndim != 3 or k.ndim != 3 or any(size <= 0 for size in (*q.shape, *k.shape)):
        raise ValueError("q and k must be nonempty rank-three arrays")
    heads, queries, dimension = q.shape
    kv_heads, tokens, key_dimension = k.shape
    if dimension != key_dimension or dimension % 2 or heads % kv_heads:
        raise ValueError("matching even head dimensions and divisible GQA head counts are required")
    if (positions.shape != (queries,) or not np.issubdtype(positions.dtype, np.integer)
            or np.any(positions < 0) or np.any(positions >= tokens)):
        raise ValueError("query_positions must be integer [Q] within the complete key sequence")
    pairs = dimension // 2
    if (native_inv.shape != (pairs,) or active_inv.shape != (pairs,)
            or np.any(native_inv <= 0) or np.any(active_inv <= 0)):
        raise ValueError("inverse-frequency arrays must be positive [D/2]")
    gain = float(gain)
    attention_scale = 1 / math.sqrt(dimension) if attention_scale is None else float(attention_scale)
    if not math.isfinite(gain) or gain < 0 or not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("gain must be finite/nonnegative and attention_scale finite/positive")
    shape = heads, queries
    output = {name: np.zeros(shape, dtype=np.float64) for name in (
        "kl", "total_logit_delta_variance", "off_diagonal_cancellation",
        "gain_logit_delta_mean", "gain_logit_delta_variance")}
    output.update({name: np.zeros((*shape, pairs), dtype=np.float64) for name in (
        "slot_logit_delta_mean", "slot_logit_delta_variance")})
    output.update(mean_kl=0.0, analysis_precision="float64", gain=gain,
                  attention_scale=attention_scale, gain_logit_multiplier=gain * gain)
    if not math.isfinite(output["gain_logit_multiplier"]):
        raise ValueError("squared gain is outside finite FP64 range")
    if gain == 1.0 and np.array_equal(native_inv, active_inv):
        return output
    repeats, gain_squared = heads // kv_heads, gain * gain
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for qi, position_value in enumerate(positions):
                position = int(position_value)
                relative = position - np.arange(position + 1, dtype=np.float64)
                for head in range(heads):
                    keys = k[head // repeats, :position + 1]
                    native_slots = _slot_logits(q[head, qi], keys, relative, native_inv, attention_scale)
                    active_slots = (native_slots if np.array_equal(native_inv, active_inv) else
                                    _slot_logits(q[head, qi], keys, relative, active_inv, attention_scale))
                    log_p = _log_softmax(native_slots.sum(axis=1))
                    probabilities = np.exp(log_p)
                    log_candidate = _log_softmax(gain_squared * active_slots.sum(axis=1))
                    kl = float(probabilities @ (log_p - log_candidate))
                    if not math.isfinite(kl) or kl < -1e-10:
                        raise FloatingPointError("KL numerical validity failed")
                    output["kl"][head, qi] = max(0.0, kl)
                    delta = gain_squared * active_slots - native_slots
                    mean, variance = _moments(delta, probabilities)
                    output["slot_logit_delta_mean"][head, qi] = mean
                    output["slot_logit_delta_variance"][head, qi] = variance
                    _, total_variance = _moments(delta.sum(axis=1), probabilities)
                    output["total_logit_delta_variance"][head, qi] = total_variance
                    output["off_diagonal_cancellation"][head, qi] = total_variance - variance.sum()
                    gain_delta = (gain_squared - 1) * active_slots.sum(axis=1)
                    gain_mean, gain_variance = _moments(gain_delta, probabilities)
                    output["gain_logit_delta_mean"][head, qi] = gain_mean
                    output["gain_logit_delta_variance"][head, qi] = gain_variance
    except FloatingPointError as error:
        raise ValueError("diagnostic exceeds reliable finite FP64 arithmetic") from error
    output["mean_kl"] = float(output["kl"].mean())
    return output
