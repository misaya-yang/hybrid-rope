"""Scheme B Native-distance probability projection for frozen OLMo.

The production path evaluates three exact partitions on the *same current*
post-q_norm/k_norm Q/K trajectory:

* Native local logits and values;
* Native far logits (for their total probability mass);
* candidate far logits and values (for the conditional distribution in F).

``mass_projection`` combines Native local probabilities with the Native total
far mass and candidate conditional probabilities inside the far partition.
``mass_raw`` is the required control: Native local and candidate far logits are
placed under one ordinary softmax without the far-mass correction.

The custom cache deliberately stores post-normalization, pre-RoPE keys.  Every
attention layer is patched together, so cached keys are consumed only by this
operator until the returned restore callback is invoked.  This module never
loads a model or launches an experiment by itself.
"""
from __future__ import annotations

import argparse
import json
import math
import types
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.nn.attention.flex_attention import flex_attention

from experiments.nongeometric_screen.distance_operator import masks


METHOD_IDS = {
    "mass_projection": "NATIVE_FOLLOWUP_B_DISTANCE_MASS_V1",
    "mass_raw": "NATIVE_FOLLOWUP_B_DISTANCE_RAW_CONTROL_V1",
}

_compiled_flex = None


def _as_numpy_table(table, name: str) -> np.ndarray:
    """Return one gain-one, positive, strictly decreasing FP32 table."""
    if isinstance(table, Mapping):
        values = table.get("values_float32")
        gain = float(table.get("gain", 1.0))
    else:
        values, gain = table, 1.0
    if not math.isfinite(gain) or gain != 1.0:
        raise ValueError(f"{name} must use gain=1")
    result = np.asarray(values, dtype=np.float32)
    if (
        result.ndim != 1
        or result.size == 0
        or not np.isfinite(result).all()
        or np.any(result <= 0)
        or not np.all(result[:-1] > result[1:])
    ):
        raise ValueError(f"{name} must be a finite, positive, strictly decreasing vector")
    return result


def _validate_dense_inputs(native_logits, candidate_logits, distances, local_radius: int):
    native = np.asarray(native_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    distance = np.asarray(distances, dtype=np.int64)
    if native.ndim < 1 or native.shape != candidate.shape or native.shape != distance.shape:
        raise ValueError("native, candidate, and distances must have the same nonempty shape")
    if native.shape[-1] == 0 or not np.isfinite(native).all() or not np.isfinite(candidate).all():
        raise ValueError("logits must be finite and nonempty")
    if np.any(distance < 0):
        raise ValueError("future keys must be removed before mass projection")
    if not isinstance(local_radius, (int, np.integer)) or local_radius < 0:
        raise ValueError("local_radius must be a nonnegative integer")
    return native, candidate, distance


def mass_projection_logits(native_logits, candidate_logits, distances, local_radius: int):
    """Dense reference logits for Scheme B.

    The last axis is the complete visible-key axis.  Local logits are exactly
    Native.  Candidate far logits receive one row-wise constant so their LSE is
    exactly the Native far LSE.  Rows without far keys return Native unchanged.
    """
    native, candidate, distance = _validate_dense_inputs(
        native_logits, candidate_logits, distances, local_radius
    )
    far = distance > local_radius
    result = native.copy()
    flat_native = native.reshape(-1, native.shape[-1])
    flat_candidate = candidate.reshape(-1, candidate.shape[-1])
    flat_far = far.reshape(-1, far.shape[-1])
    flat_result = result.reshape(-1, result.shape[-1])
    for row, row_far in enumerate(flat_far):
        if not row_far.any():
            continue
        n = flat_native[row, row_far]
        c = flat_candidate[row, row_far]
        n_lse = np.logaddexp.reduce(n)
        c_lse = np.logaddexp.reduce(c)
        flat_result[row, row_far] = c + n_lse - c_lse
    return result


def mass_raw_logits(native_logits, candidate_logits, distances, local_radius: int):
    """Dense B_raw control: Native local and candidate far, one softmax."""
    native, candidate, distance = _validate_dense_inputs(
        native_logits, candidate_logits, distances, local_radius
    )
    return np.where(distance > local_radius, candidate, native)


def _rotate(raw: torch.Tensor, positions: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    if raw.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    pairs = raw.shape[-1] // 2
    if frequencies.shape != (pairs,):
        raise ValueError("frequency table does not match the attention head dimension")
    phase = positions.float()[:, None] * frequencies[None, :]
    cos, sin = phase.cos().to(raw.dtype), phase.sin().to(raw.dtype)
    x, y = raw[..., :pairs], raw[..., pairs:]
    return torch.cat((x * cos - y * sin, y * cos + x * sin), dim=-1)


def _flex():
    global _compiled_flex
    if _compiled_flex is None:
        _compiled_flex = torch.compile(flex_attention, dynamic=True)
    return _compiled_flex


def _partition_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    scale: float,
):
    options = {"FORCE_USE_FLEX_ATTENTION": query.shape[-2] > 1}
    output, lse = _flex()(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=True,
        return_lse=True,
        kernel_options=options,
    )
    finite = torch.isfinite(lse)
    output = torch.where(finite[..., None], output, 0)
    return output, lse


def _combine_partition_outputs(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    far_output: torch.Tensor,
    far_lse_for_mass: torch.Tensor,
) -> torch.Tensor:
    """Combine two partition-conditional expectations under one denominator."""
    denominator = torch.logaddexp(local_lse, far_lse_for_mass)
    local_weight = torch.exp(local_lse - denominator)
    far_weight = torch.where(
        torch.isfinite(far_lse_for_mass),
        torch.exp(far_lse_for_mass - denominator),
        torch.zeros_like(far_lse_for_mass),
    )
    return (
        local_output.float() * local_weight[..., None]
        + far_output.float() * far_weight[..., None]
    ).to(value_dtype := local_output.dtype)


def _mass_attention(
    raw_query: torch.Tensor,
    raw_key: torch.Tensor,
    value: torch.Tensor,
    native: torch.Tensor,
    candidate: torch.Tensor,
    local_radius: int,
    scale: float,
    method: str,
) -> torch.Tensor:
    if method not in METHOD_IDS:
        raise ValueError(f"unsupported mass method: {method}")
    if raw_query.shape[0] != 1 or raw_key.shape[0] != 1:
        raise ValueError("mass projection currently requires unpadded batch one")
    q_len, k_len = raw_query.shape[-2], raw_key.shape[-2]
    if q_len <= 0 or k_len < q_len:
        raise ValueError("invalid query/key lengths")
    device = raw_query.device
    query_positions = torch.arange(k_len - q_len, k_len, device=device)
    key_positions = torch.arange(k_len, device=device)
    local_mask, far_mask = masks(q_len, k_len, local_radius, device)

    native_query = _rotate(raw_query, query_positions, native)
    native_key = _rotate(raw_key, key_positions, native)
    candidate_query = _rotate(raw_query, query_positions, candidate)
    candidate_key = _rotate(raw_key, key_positions, candidate)

    local_output, local_lse = _partition_attention(
        native_query, native_key, value, local_mask, scale
    )
    candidate_far_output, candidate_far_lse = _partition_attention(
        candidate_query, candidate_key, value, far_mask, scale
    )
    if method == "mass_projection":
        # Only the Native far LSE is needed.  FlexAttention also returns its
        # conditional value expectation; discard it after making empty rows safe.
        _, native_far_lse = _partition_attention(
            native_query, native_key, value, far_mask, scale
        )
        far_lse_for_mass = native_far_lse
    else:
        far_lse_for_mass = candidate_far_lse
    return _combine_partition_outputs(
        local_output, local_lse, candidate_far_output, far_lse_for_mass
    )


def _project_qkv(module, hidden_states: torch.Tensor):
    """OLMo2 projections in the exact post-q_norm/k_norm, pre-RoPE order."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_norm(module.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return query, key, value


def install(model, native_table, candidate_table, method: str):
    """Install B or B_raw on every OLMo2 attention layer.

    Returns an idempotent restore callback.  Do not restore while a cache made
    by this operator is still in use: cached keys are intentionally raw,
    post-normalization and pre-RoPE.
    """
    if method not in METHOD_IDS:
        raise ValueError(f"method must be one of {tuple(METHOD_IDS)}")
    config = model.config
    if getattr(config, "model_type", None) != "olmo2":
        raise ValueError("the first frozen mass-projection runtime is OLMo2-only")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or not len(layers):
        raise ValueError("model.model.layers is required")
    native_values = _as_numpy_table(native_table, "native_table")
    candidate_values = _as_numpy_table(candidate_table, "candidate_table")
    if native_values.shape != candidate_values.shape:
        raise ValueError("Native and candidate tables must share one geometry")
    head_dim = int(
        getattr(config, "head_dim", 0)
        or getattr(config, "hidden_size", 0) // getattr(config, "num_attention_heads", 1)
    )
    if head_dim <= 0 or native_values.size * 2 != head_dim:
        raise ValueError("table geometry differs from config.head_dim")
    native_length = int(getattr(config, "max_position_embeddings", 0) or 0)
    if native_length <= 1:
        raise ValueError("config.max_position_embeddings must define the Native length")
    local_radius = max(1, native_length // 16)

    table_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    originals = []
    restored = False

    def tables_for(device):
        key = str(device)
        if key not in table_cache:
            table_cache[key] = (
                torch.as_tensor(native_values, dtype=torch.float32, device=device),
                torch.as_tensor(candidate_values, dtype=torch.float32, device=device),
            )
        return table_cache[key]

    def make_forward(original):
        def forward(
            module,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            del position_embeddings  # both clocks are reconstructed from raw Q/K
            if module.training:
                raise RuntimeError("mass projection is frozen-inference only")
            if kwargs.get("output_attentions"):
                raise ValueError("mass projection does not materialize attention weights")
            if hidden_states.shape[0] != 1:
                raise ValueError("mass projection requires unpadded batch one")
            if attention_mask is not None:
                raise ValueError("mass projection requires the unpadded causal fast path")

            raw_query, raw_key, value = _project_qkv(module, hidden_states)
            if past_key_values is not None:
                raw_key, value = past_key_values.update(
                    raw_key,
                    value,
                    module.layer_idx,
                    {"cache_position": cache_position},
                )
            q_len, k_len = raw_query.shape[-2], raw_key.shape[-2]
            if cache_position is not None:
                expected = torch.arange(k_len - q_len, k_len, device=cache_position.device)
                if cache_position.ndim != 1 or not torch.equal(cache_position, expected):
                    raise ValueError("mass projection requires contiguous absolute positions")
            native, candidate = tables_for(raw_query.device)
            output = _mass_attention(
                raw_query,
                raw_key,
                value,
                native,
                candidate,
                local_radius,
                float(module.scaling),
                method,
            )
            input_shape = hidden_states.shape[:-1]
            output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            return module.o_proj(output), None

        forward.__wrapped__ = original
        return forward

    for layer in layers:
        attention = layer.self_attn
        if getattr(attention, "_native_mass_projection_method", None) is not None:
            raise RuntimeError("mass projection is already installed")
        for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"):
            if not hasattr(attention, name):
                raise ValueError(f"OLMo attention is missing {name}")
        original = attention.forward
        originals.append((attention, original))
        attention.forward = types.MethodType(make_forward(original), attention)
        attention._native_mass_projection_method = method

    def restore():
        nonlocal restored
        if restored:
            return
        for attention, original in originals:
            attention.forward = original
            delattr(attention, "_native_mass_projection_method")
        table_cache.clear()
        restored = True

    restore.method_id = METHOD_IDS[method]
    restore.local_radius = local_radius
    return restore


def _plan(method: str) -> dict:
    if method not in METHOD_IDS:
        raise ValueError(method)
    return {
        "status": "PLAN_ONLY",
        "method": method,
        "method_id": METHOD_IDS[method],
        "model_loaded": False,
        "gpu_execution": False,
        "note": "Import install() from the frozen runner; this module never launches a model.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(METHOD_IDS), default="mass_projection")
    args = parser.parse_args()
    print(json.dumps(_plan(args.method), sort_keys=True))
