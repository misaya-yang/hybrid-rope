#!/usr/bin/env python3
"""CPU algebra checks for a prefix-preserving RoPE coordinate handoff.

This is not a model-quality evaluator.  It verifies only three identities:

1. the prefix branch can call Native RoPE unchanged;
2. suffix queries paired with long-rotated cached prefix keys recover the
   stationary long-table relative phase; and
3. the historical boundary-slope phase map has a non-zero cross-boundary
   mismatch whenever the key is before the boundary and the frequency moves.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np


def _vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or array.size % 2:
        raise ValueError(f"{name} must be one non-empty split-half rotary vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _frequencies(value: np.ndarray, *, pairs: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (pairs,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError("frequencies must be one finite positive value per rotary pair")
    return array


def rotate_split_half(
    value: np.ndarray,
    *,
    position: int,
    frequencies: np.ndarray,
    scaling: float = 1.0,
) -> np.ndarray:
    """Apply OLMo-style split-half pair rotations at one integer position."""

    vector = _vector(value, name="value")
    pairs = vector.size // 2
    omega = _frequencies(frequencies, pairs=pairs)
    if not math.isfinite(float(scaling)) or float(scaling) <= 0:
        raise ValueError("rotary scaling must be finite and positive")
    angle = omega * int(position)
    return rotate_split_half_angles(vector, angles=angle) * float(scaling)


def rotate_split_half_angles(value: np.ndarray, *, angles: np.ndarray) -> np.ndarray:
    """Rotate split-half pairs by arbitrary finite signed angles."""

    vector = _vector(value, name="value")
    pairs = vector.size // 2
    angle = np.asarray(angles, dtype=np.float64)
    if angle.shape != (pairs,) or not np.isfinite(angle).all():
        raise ValueError("angles must be one finite value per rotary pair")
    cosine = np.cos(angle)
    sine = np.sin(angle)
    first, second = vector[:pairs], vector[pairs:]
    return np.concatenate(
        (first * cosine - second * sine, second * cosine + first * sine)
    )


def rephase_cached_key(
    native_rotated_key: np.ndarray,
    *,
    position: int,
    native_frequencies: np.ndarray,
    long_frequencies: np.ndarray,
    native_scaling: float = 1.0,
    long_scaling: float = 1.0,
) -> np.ndarray:
    """Convert an already-Native-rotated key into the frozen long frame."""

    key = _vector(native_rotated_key, name="native_rotated_key")
    pairs = key.size // 2
    native = _frequencies(native_frequencies, pairs=pairs)
    long = _frequencies(long_frequencies, pairs=pairs)
    for name, value in (("native_scaling", native_scaling), ("long_scaling", long_scaling)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    delta_angles = (long - native) * int(position)
    return (
        rotate_split_half_angles(key, angles=delta_angles)
        * (float(long_scaling) / float(native_scaling))
    )


def rotary_score(
    query: np.ndarray,
    key: np.ndarray,
    *,
    query_position: int,
    key_position: int,
    frequencies: np.ndarray,
    scaling: float = 1.0,
) -> float:
    rotated_query = rotate_split_half(
        query, position=query_position, frequencies=frequencies, scaling=scaling
    )
    rotated_key = rotate_split_half(
        key, position=key_position, frequencies=frequencies, scaling=scaling
    )
    return float(rotated_query @ rotated_key)


def prefix_handoff_score(
    query: np.ndarray,
    key: np.ndarray,
    *,
    query_position: int,
    key_position: int,
    native_frequencies: np.ndarray,
    long_frequencies: np.ndarray,
    boundary: int,
    native_scaling: float = 1.0,
    long_scaling: float = 1.0,
) -> float:
    """Use Native for prefix queries and one stationary long frame afterward.

    A suffix query uses the long frame for *both* query and cached key, even
    when the key's hidden state was produced during the Native prefix pass.
    """

    if int(boundary) <= 0:
        raise ValueError("boundary must be positive")
    frequencies = (
        native_frequencies if int(query_position) < int(boundary) else long_frequencies
    )
    scaling = native_scaling if int(query_position) < int(boundary) else long_scaling
    return rotary_score(
        query,
        key,
        query_position=query_position,
        key_position=key_position,
        frequencies=frequencies,
        scaling=scaling,
    )


def boundary_slope_phase(
    frequency_native: float,
    frequency_long: float,
    *,
    position: int,
    boundary: int,
) -> float:
    """Historical continuous absolute-phase map used by the failed operator."""

    if position < boundary:
        return float(frequency_native) * int(position)
    return (
        float(frequency_native) * int(boundary)
        + float(frequency_long) * (int(position) - int(boundary))
    )


def boundary_slope_cross_error(
    frequency_native: float,
    frequency_long: float,
    *,
    query_position: int,
    key_position: int,
    boundary: int,
) -> float:
    """Phase error versus a stationary long table for one cross-boundary pair."""

    if not key_position < boundary <= query_position:
        raise ValueError("expected key < boundary <= query")
    observed = boundary_slope_phase(
        frequency_native,
        frequency_long,
        position=query_position,
        boundary=boundary,
    ) - boundary_slope_phase(
        frequency_native,
        frequency_long,
        position=key_position,
        boundary=boundary,
    )
    stationary = float(frequency_long) * (int(query_position) - int(key_position))
    return float(observed - stationary)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def causal_attention_reference(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    native_frequencies: np.ndarray,
    long_frequencies: np.ndarray,
    boundary: int,
    native_scaling: float = 1.0,
    long_scaling: float = 1.0,
) -> np.ndarray:
    """Direct definition of prefix-Native/suffix-long causal attention."""

    q = np.asarray(queries, dtype=np.float64)
    k = np.asarray(keys, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if q.ndim != 2 or k.shape != q.shape or v.ndim != 2 or v.shape[0] != q.shape[0]:
        raise ValueError("queries/keys must match [T,D] and values must be [T,Dv]")
    if not 0 < int(boundary) < q.shape[0]:
        raise ValueError("boundary must lie strictly inside the sequence")
    scale = math.sqrt(q.shape[1])
    output = np.empty_like(v)
    for query_position in range(q.shape[0]):
        frequencies = (
            native_frequencies
            if query_position < int(boundary)
            else long_frequencies
        )
        scaling = native_scaling if query_position < int(boundary) else long_scaling
        scores = np.asarray(
            [
                rotary_score(
                    q[query_position],
                    k[key_position],
                    query_position=query_position,
                    key_position=key_position,
                    frequencies=frequencies,
                    scaling=scaling,
                )
                / scale
                for key_position in range(query_position + 1)
            ],
            dtype=np.float64,
        )
        output[query_position] = _softmax(scores) @ v[: query_position + 1]
    return output


def causal_attention_cached(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    native_frequencies: np.ndarray,
    long_frequencies: np.ndarray,
    boundary: int,
    native_scaling: float = 1.0,
    long_scaling: float = 1.0,
) -> np.ndarray:
    """Tokenwise cache implementation of the same handoff definition.

    Prefix steps cache Native-rotated keys.  At the boundary they are converted
    in place to the long frame; suffix steps append only long-rotated keys.
    """

    q = np.asarray(queries, dtype=np.float64)
    k = np.asarray(keys, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if q.ndim != 2 or k.shape != q.shape or v.ndim != 2 or v.shape[0] != q.shape[0]:
        raise ValueError("queries/keys must match [T,D] and values must be [T,Dv]")
    if not 0 < int(boundary) < q.shape[0]:
        raise ValueError("boundary must lie strictly inside the sequence")

    active_keys: list[np.ndarray] = []
    cached_values: list[np.ndarray] = []
    output = np.empty_like(v)
    scale = math.sqrt(q.shape[1])
    for position in range(q.shape[0]):
        if position == int(boundary):
            active_keys = [
                rephase_cached_key(
                    cached_key,
                    position=key_position,
                    native_frequencies=native_frequencies,
                    long_frequencies=long_frequencies,
                    native_scaling=native_scaling,
                    long_scaling=long_scaling,
                )
                for key_position, cached_key in enumerate(active_keys)
            ]
        current_frequencies = (
            native_frequencies if position < int(boundary) else long_frequencies
        )
        current_scaling = native_scaling if position < int(boundary) else long_scaling
        active_keys.append(
            rotate_split_half(
                k[position],
                position=position,
                frequencies=current_frequencies,
                scaling=current_scaling,
            )
        )
        cached_values.append(v[position])
        if position < int(boundary):
            rotated_query = rotate_split_half(
                q[position],
                position=position,
                frequencies=native_frequencies,
                scaling=native_scaling,
            )
        else:
            rotated_query = rotate_split_half(
                q[position],
                position=position,
                frequencies=long_frequencies,
                scaling=long_scaling,
            )
        scores = np.asarray(
            [float(rotated_query @ cached_key) / scale for cached_key in active_keys],
            dtype=np.float64,
        )
        output[position] = _softmax(scores) @ np.stack(cached_values)
    return output


def toy_causal_stack(
    hidden: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    native_frequencies: np.ndarray,
    long_frequencies: np.ndarray,
    boundary: int,
    cached: bool,
    native_scaling: float = 1.0,
    long_scaling: float = 1.0,
) -> np.ndarray:
    """Model-free multi-layer causal stack used only for cache semantics.

    Each layer has dense Q/K/V/O maps, causal rotary attention, a residual,
    and tanh.  It is not intended to approximate a trained transformer.
    """

    state = np.asarray(hidden, dtype=np.float64)
    if state.ndim != 2:
        raise ValueError("hidden must be [T,D]")
    attention = causal_attention_cached if cached else causal_attention_reference
    for query_map, key_map, value_map, output_map in layers:
        queries = state @ np.asarray(query_map, dtype=np.float64)
        keys = state @ np.asarray(key_map, dtype=np.float64)
        values = state @ np.asarray(value_map, dtype=np.float64)
        attended = attention(
            queries,
            keys,
            values,
            native_frequencies=native_frequencies,
            long_frequencies=long_frequencies,
            boundary=boundary,
            native_scaling=native_scaling,
            long_scaling=long_scaling,
        )
        state = np.tanh(state + attended @ np.asarray(output_map, dtype=np.float64))
    return state


def _validate_gqa_inputs(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    q = np.asarray(queries, dtype=np.float64)
    k = np.asarray(keys, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("GQA queries/keys/values must be rank-three")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("GQA sequence lengths must match")
    if k.shape[:2] != v.shape[:2] or q.shape[2] != k.shape[2]:
        raise ValueError("GQA key/value heads and rotary dimensions must match")
    if k.shape[1] <= 0 or q.shape[1] % k.shape[1]:
        raise ValueError("query-head count must be divisible by KV-head count")
    return q, k, v, q.shape[1] // k.shape[1]


def split_chunks_at_boundary(
    chunk_sizes: tuple[int, ...],
    *,
    boundary: int,
) -> tuple[int, ...]:
    """Split an input schedule so no attention call straddles the handoff."""

    chunks = tuple(int(size) for size in chunk_sizes)
    if not chunks or any(size <= 0 for size in chunks):
        raise ValueError("chunk sizes must be positive")
    total = sum(chunks)
    if not 0 < int(boundary) < total:
        raise ValueError("boundary must lie strictly inside the chunk schedule")
    result: list[int] = []
    start = 0
    for size in chunks:
        end = start + size
        if start < int(boundary) < end:
            result.extend((int(boundary) - start, end - int(boundary)))
        else:
            result.append(size)
        start = end
    return tuple(result)


def gqa_attention_reference(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    native_frequencies: np.ndarray,
    long_frequencies: np.ndarray,
    boundary: int,
    native_scaling: float = 1.0,
    long_scaling: float = 1.0,
) -> np.ndarray:
    """Direct causal handoff for MHA, GQA, or MQA head layouts."""

    q, k, v, group_size = _validate_gqa_inputs(queries, keys, values)
    if not 0 < int(boundary) < q.shape[0]:
        raise ValueError("boundary must lie strictly inside the sequence")
    output = np.empty((q.shape[0], q.shape[1], v.shape[2]), dtype=np.float64)
    scale = math.sqrt(q.shape[2])
    for position in range(q.shape[0]):
        frequencies = native_frequencies if position < int(boundary) else long_frequencies
        scaling = native_scaling if position < int(boundary) else long_scaling
        for query_head in range(q.shape[1]):
            key_head = query_head // group_size
            scores = np.asarray(
                [
                    rotary_score(
                        q[position, query_head],
                        k[key_position, key_head],
                        query_position=position,
                        key_position=key_position,
                        frequencies=frequencies,
                        scaling=scaling,
                    )
                    / scale
                    for key_position in range(position + 1)
                ],
                dtype=np.float64,
            )
            output[position, query_head] = (
                _softmax(scores) @ v[: position + 1, key_head]
            )
    return output


def gqa_attention_chunked_cache(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    native_frequencies: np.ndarray,
    long_frequencies: np.ndarray,
    boundary: int,
    chunk_sizes: tuple[int, ...],
    native_scaling: float = 1.0,
    long_scaling: float = 1.0,
) -> np.ndarray:
    """Chunk-order-invariant cache simulation for MHA/GQA/MQA handoff."""

    q, k, v, group_size = _validate_gqa_inputs(queries, keys, values)
    requested_chunks = tuple(int(size) for size in chunk_sizes)
    if (
        not requested_chunks
        or any(size <= 0 for size in requested_chunks)
        or sum(requested_chunks) != q.shape[0]
    ):
        raise ValueError("positive chunk sizes must sum to the sequence length")
    if not 0 < int(boundary) < q.shape[0]:
        raise ValueError("boundary must lie strictly inside the sequence")
    chunks = split_chunks_at_boundary(requested_chunks, boundary=boundary)

    active_keys: list[np.ndarray] = []
    cached_values: list[np.ndarray] = []
    output = np.empty((q.shape[0], q.shape[1], v.shape[2]), dtype=np.float64)
    scale = math.sqrt(q.shape[2])
    start = 0
    for chunk_size in chunks:
        for position in range(start, start + chunk_size):
            if position == int(boundary):
                active_keys = [
                    np.stack(
                        [
                            rephase_cached_key(
                                cached_key[head],
                                position=key_position,
                                native_frequencies=native_frequencies,
                                long_frequencies=long_frequencies,
                                native_scaling=native_scaling,
                                long_scaling=long_scaling,
                            )
                            for head in range(k.shape[1])
                        ]
                    )
                    for key_position, cached_key in enumerate(active_keys)
                ]
            frequencies = (
                native_frequencies if position < int(boundary) else long_frequencies
            )
            scaling = native_scaling if position < int(boundary) else long_scaling
            active_keys.append(
                np.stack(
                    [
                        rotate_split_half(
                            k[position, head],
                            position=position,
                            frequencies=frequencies,
                            scaling=scaling,
                        )
                        for head in range(k.shape[1])
                    ]
                )
            )
            cached_values.append(v[position])
            for query_head in range(q.shape[1]):
                key_head = query_head // group_size
                rotated_query = rotate_split_half(
                    q[position, query_head],
                    position=position,
                    frequencies=frequencies,
                    scaling=scaling,
                )
                scores = np.asarray(
                    [
                        float(rotated_query @ cached_key[key_head]) / scale
                        for cached_key in active_keys
                    ],
                    dtype=np.float64,
                )
                head_values = np.stack(cached_values)[:, key_head]
                output[position, query_head] = _softmax(scores) @ head_values
        start += chunk_size
    return output


def deterministic_check() -> dict[str, float | int | str]:
    rng = np.random.default_rng(20260903)
    native = np.asarray([1.0, 0.37, 0.11, 0.03], dtype=np.float64)
    long = np.asarray([1.0, 0.19, 0.055, 0.0075], dtype=np.float64)
    query = rng.standard_normal(8)
    key = rng.standard_normal(8)
    boundary = 16
    native_scaling = 0.9
    long_scaling = 1.1

    raw_rephase_key = rng.standard_normal(8)
    native_rotated_key = rotate_split_half(
        raw_rephase_key,
        position=13,
        frequencies=native,
        scaling=native_scaling,
    )
    rephased_key = rephase_cached_key(
        native_rotated_key,
        position=13,
        native_frequencies=native,
        long_frequencies=long,
        native_scaling=native_scaling,
        long_scaling=long_scaling,
    )
    direct_long_key = rotate_split_half(
        raw_rephase_key,
        position=13,
        frequencies=long,
        scaling=long_scaling,
    )

    prefix_reference = rotary_score(
        query,
        key,
        query_position=12,
        key_position=3,
        frequencies=native,
    )
    prefix_observed = prefix_handoff_score(
        query,
        key,
        query_position=12,
        key_position=3,
        native_frequencies=native,
        long_frequencies=long,
        boundary=boundary,
    )
    suffix_reference = rotary_score(
        query,
        key,
        query_position=23,
        key_position=3,
        frequencies=long,
    )
    suffix_observed = prefix_handoff_score(
        query,
        key,
        query_position=23,
        key_position=3,
        native_frequencies=native,
        long_frequencies=long,
        boundary=boundary,
    )
    phase_error = boundary_slope_cross_error(
        native[1],
        long[1],
        query_position=23,
        key_position=3,
        boundary=boundary,
    )
    expected_phase_error = (native[1] - long[1]) * (boundary - 3)

    sequence_length = 24
    queries = rng.standard_normal((sequence_length, 8))
    keys = rng.standard_normal((sequence_length, 8))
    values = rng.standard_normal((sequence_length, 6))
    reference_attention = causal_attention_reference(
        queries,
        keys,
        values,
        native_frequencies=native,
        long_frequencies=long,
        boundary=boundary,
        native_scaling=native_scaling,
        long_scaling=long_scaling,
    )
    cached_attention = causal_attention_cached(
        queries,
        keys,
        values,
        native_frequencies=native,
        long_frequencies=long,
        boundary=boundary,
        native_scaling=native_scaling,
        long_scaling=long_scaling,
    )
    pure_native = causal_attention_reference(
        queries,
        keys,
        values,
        native_frequencies=native,
        long_frequencies=native,
        boundary=boundary,
        native_scaling=native_scaling,
        long_scaling=native_scaling,
    )
    hidden = rng.standard_normal((sequence_length, 8))
    layers = []
    for _ in range(3):
        layers.append(
            tuple(rng.standard_normal((8, 8)) / math.sqrt(8) for _ in range(4))
        )
    stack_reference = toy_causal_stack(
        hidden,
        layers,
        native_frequencies=native,
        long_frequencies=long,
        boundary=boundary,
        cached=False,
    )
    stack_cached = toy_causal_stack(
        hidden,
        layers,
        native_frequencies=native,
        long_frequencies=long,
        boundary=boundary,
        cached=True,
    )
    stack_native = toy_causal_stack(
        hidden,
        layers,
        native_frequencies=native,
        long_frequencies=native,
        boundary=boundary,
        cached=False,
    )
    gqa_queries = rng.standard_normal((18, 4, 8))
    gqa_keys = rng.standard_normal((18, 2, 8))
    gqa_values = rng.standard_normal((18, 2, 5))
    gqa_reference = gqa_attention_reference(
        gqa_queries,
        gqa_keys,
        gqa_values,
        native_frequencies=native,
        long_frequencies=long,
        boundary=11,
        native_scaling=native_scaling,
        long_scaling=long_scaling,
    )
    gqa_chunked = gqa_attention_chunked_cache(
        gqa_queries,
        gqa_keys,
        gqa_values,
        native_frequencies=native,
        long_frequencies=long,
        boundary=11,
        chunk_sizes=(4, 9, 5),
        native_scaling=native_scaling,
        long_scaling=long_scaling,
    )
    gqa_native = gqa_attention_reference(
        gqa_queries,
        gqa_keys,
        gqa_values,
        native_frequencies=native,
        long_frequencies=native,
        boundary=11,
        native_scaling=native_scaling,
        long_scaling=native_scaling,
    )
    mqa_reference = gqa_attention_reference(
        gqa_queries,
        gqa_keys[:, :1],
        gqa_values[:, :1],
        native_frequencies=native,
        long_frequencies=long,
        boundary=11,
        native_scaling=native_scaling,
        long_scaling=long_scaling,
    )
    mqa_chunked = gqa_attention_chunked_cache(
        gqa_queries,
        gqa_keys[:, :1],
        gqa_values[:, :1],
        native_frequencies=native,
        long_frequencies=long,
        boundary=11,
        chunk_sizes=(11, 1, 6),
        native_scaling=native_scaling,
        long_scaling=long_scaling,
    )

    prefix_error = abs(prefix_observed - prefix_reference)
    suffix_error = abs(suffix_observed - suffix_reference)
    identity_error = abs(phase_error - expected_phase_error)
    cache_error = float(np.max(np.abs(cached_attention - reference_attention)))
    prefix_attention_error = float(
        np.max(np.abs(cached_attention[:boundary] - pure_native[:boundary]))
    )
    stack_cache_error = float(np.max(np.abs(stack_cached - stack_reference)))
    stack_prefix_error = float(
        np.max(np.abs(stack_cached[:boundary] - stack_native[:boundary]))
    )
    gqa_cache_error = float(np.max(np.abs(gqa_chunked - gqa_reference)))
    gqa_prefix_error = float(np.max(np.abs(gqa_chunked[:11] - gqa_native[:11])))
    mqa_cache_error = float(np.max(np.abs(mqa_chunked - mqa_reference)))
    rephase_error = float(np.max(np.abs(rephased_key - direct_long_key)))
    if prefix_error != 0.0 or suffix_error != 0.0:
        raise RuntimeError("handoff branch identity failed")
    if identity_error > 1e-12 or math.isclose(phase_error, 0.0, abs_tol=1e-12):
        raise RuntimeError("boundary-slope mismatch identity failed")
    if cache_error > 1e-12 or prefix_attention_error != 0.0:
        raise RuntimeError("causal cache handoff identity failed")
    if stack_cache_error > 1e-12 or stack_prefix_error != 0.0:
        raise RuntimeError("multi-layer causal handoff identity failed")
    if gqa_cache_error > 1e-12 or gqa_prefix_error != 0.0 or mqa_cache_error > 1e-12:
        raise RuntimeError("chunked GQA/MQA handoff identity failed")
    if rephase_error > 1e-12:
        raise RuntimeError("in-place cached-key rephasing identity failed")

    return {
        "status": "CPU_ALGEBRA_PASS",
        "seed": 20260903,
        "pairs": int(native.size),
        "boundary": boundary,
        "native_scaling": native_scaling,
        "long_scaling": long_scaling,
        "cached_key_rephase_vs_direct_long_max_abs_error": rephase_error,
        "prefix_native_score_error": prefix_error,
        "suffix_stationary_long_score_error": suffix_error,
        "boundary_slope_phase_error": phase_error,
        "boundary_slope_identity_error": identity_error,
        "cached_vs_direct_handoff_max_abs_error": cache_error,
        "prefix_attention_vs_native_max_abs_error": prefix_attention_error,
        "toy_stack_layers": len(layers),
        "toy_stack_cached_vs_direct_max_abs_error": stack_cache_error,
        "toy_stack_prefix_vs_native_max_abs_error": stack_prefix_error,
        "gqa_query_heads": 4,
        "gqa_kv_heads": 2,
        "gqa_chunk_sizes": "4,9,5",
        "gqa_effective_chunk_sizes": ",".join(
            str(value) for value in split_chunks_at_boundary((4, 9, 5), boundary=11)
        ),
        "gqa_chunked_vs_direct_max_abs_error": gqa_cache_error,
        "gqa_prefix_vs_native_max_abs_error": gqa_prefix_error,
        "mqa_kv_heads": 1,
        "mqa_chunk_sizes": "11,1,6",
        "mqa_effective_chunk_sizes": ",".join(
            str(value) for value in split_chunks_at_boundary((11, 1, 6), boundary=11)
        ),
        "mqa_chunked_vs_direct_max_abs_error": mqa_cache_error,
        "scope": "model-free rotary/causal-attention algebra only; no trained-transformer quality claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(deterministic_check(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
