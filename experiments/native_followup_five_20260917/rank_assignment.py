"""Scheme C: Native confidence multiset with candidate full-key ranking.

For every frozen OLMo-2 attention row, Native RoPE supplies the complete
attention-probability multiset.  Candidate RoPE supplies only a stable ordering
of the complete visible-key axis.  The sorted Native probabilities are assigned
to candidate-ranked key indices, then one PV product is evaluated.

The cache stores post-q_norm/k_norm, pre-RoPE keys.  This module does not load a
checkpoint or execute a GPU job.  Its CLI is PLAN_ONLY.
"""
from __future__ import annotations

import argparse
import json
import math
import types
from collections.abc import Mapping

import numpy as np
import torch


METHOD_ID = "NATIVE_FOLLOWUP_C_NATIVE_CONFIDENCE_RANK_V1"
METHOD_IDS = {"rank_assignment": METHOD_ID}
QUERY_TILE_SIZE = 16


def _table(table, name: str) -> np.ndarray:
    if isinstance(table, Mapping):
        payload = table.get("table", table)
        values = payload.get("values_float32")
        gain = float(payload.get("gain", 1.0))
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
        raise ValueError(f"{name} must be finite, positive, and strictly decreasing")
    return result


def stable_rank_probabilities(native_logits, candidate_logits, visible=None):
    """Dense reference assignment on the complete last-axis key domain.

    Candidate ties are resolved by Native score descending and then original
    key index, exactly as frozen in the design contract.  Masked keys receive
    exactly zero probability and sort after every finite visible candidate.
    """
    native = torch.as_tensor(native_logits)
    candidate = torch.as_tensor(candidate_logits, device=native.device)
    if native.ndim < 1 or native.shape != candidate.shape or native.shape[-1] == 0:
        raise ValueError("Native and candidate logits must share one nonempty shape")
    if not torch.isfinite(native).all() or not torch.isfinite(candidate).all():
        raise ValueError("unmasked logits must be finite")
    if visible is None:
        visible_mask = torch.ones_like(native, dtype=torch.bool)
    else:
        visible_mask = torch.as_tensor(visible, device=native.device, dtype=torch.bool)
        if visible_mask.shape != native.shape:
            try:
                visible_mask = torch.broadcast_to(visible_mask, native.shape)
            except RuntimeError as error:
                raise ValueError("visibility mask does not broadcast to logits") from error
    if not visible_mask.any(dim=-1).all():
        raise ValueError("every attention row needs at least one visible key")

    negative_infinity = torch.tensor(float("-inf"), device=native.device, dtype=native.dtype)
    native_masked = native.masked_fill(~visible_mask, negative_infinity)
    candidate_masked = candidate.masked_fill(~visible_mask, negative_infinity)
    native_probabilities = torch.softmax(native_masked.float(), dim=-1)
    confidence = torch.sort(native_probabilities, dim=-1, descending=True, stable=True).values
    # Stable lexicographic ordering: original key order is the implicit final
    # key; sorting Native first makes it the secondary key, then sorting the
    # gathered candidate values makes candidate score the primary key.
    native_rank = torch.argsort(native_masked.float(), dim=-1, descending=True, stable=True)
    candidate_in_native_order = torch.gather(candidate_masked.float(), -1, native_rank)
    primary_rank = torch.argsort(
        candidate_in_native_order, dim=-1, descending=True, stable=True,
    )
    candidate_rank = torch.gather(native_rank, -1, primary_rank)
    assigned = torch.zeros_like(confidence)
    assigned.scatter_(-1, candidate_rank, confidence)
    assigned.masked_fill_(~visible_mask, 0.0)
    return assigned


def _rotate(raw: torch.Tensor, positions: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    if raw.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    pairs = raw.shape[-1] // 2
    if frequencies.shape != (pairs,):
        raise ValueError("frequency table differs from attention head dimension")
    with torch.autocast(device_type=raw.device.type, enabled=False):
        phase = positions.float()[:, None] * frequencies[None, :]
        cosine, sine = phase.cos(), phase.sin()
    cosine, sine = cosine.to(raw.dtype), sine.to(raw.dtype)
    first, second = raw[..., :pairs], raw[..., pairs:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)


def _project_qkv(module, hidden_states: torch.Tensor):
    """Match OLMo-2's post-q_norm/k_norm, pre-RoPE projection order."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_norm(module.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return query, key, value


def _causal_visibility(query_length: int, key_length: int, device) -> torch.Tensor:
    if query_length <= 0 or key_length < query_length:
        raise ValueError("invalid causal query/key lengths")
    offset = key_length - query_length
    query_positions = torch.arange(offset, key_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    return key_positions[None, :] <= query_positions[:, None]


def _check_unpadded_mask(attention_mask, causal: torch.Tensor, *, batch: int) -> None:
    if attention_mask is None:
        return
    if attention_mask.ndim == 2:
        if attention_mask.shape != (batch, causal.shape[-1]) or not attention_mask.bool().all():
            raise ValueError("rank assignment requires unpadded batch one")
        return
    if attention_mask.ndim == 4:
        if attention_mask.shape[0] != batch or attention_mask.shape[-2:] != causal.shape:
            raise ValueError("four-dimensional attention mask shape differs")
        # Standard additive masks are zero on visible cells and a large negative
        # value (or -inf) elsewhere.  No other masking is accepted in this pilot.
        observed = attention_mask[:, :1].float() > -1.0
        expected = causal[None, None]
        if not torch.equal(observed, expected.expand_as(observed)):
            raise ValueError("rank assignment requires the unpadded causal mask")
        return
    raise ValueError("attention mask must be None, [B,K], or [B,1,Q,K]")


def rank_assignment_attention(
    raw_query: torch.Tensor,
    raw_key: torch.Tensor,
    value: torch.Tensor,
    native: torch.Tensor,
    candidate: torch.Tensor,
    scale: float,
    *,
    attention_mask=None,
    query_tile_size: int = QUERY_TILE_SIZE,
) -> torch.Tensor:
    """Execute full-key stable assignment and one PV, tiling queries only."""
    if raw_query.shape[0] != 1 or raw_key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("rank assignment requires batch=1")
    if query_tile_size <= 0:
        raise ValueError("query_tile_size must be positive")
    query_heads, key_heads = raw_query.shape[1], raw_key.shape[1]
    if query_heads % key_heads or key_heads != value.shape[1]:
        raise ValueError("invalid Q/K/V head mapping")
    query_length, key_length = raw_query.shape[-2], raw_key.shape[-2]
    causal = _causal_visibility(query_length, key_length, raw_query.device)
    _check_unpadded_mask(attention_mask, causal, batch=1)

    query_positions = torch.arange(key_length - query_length, key_length, device=raw_query.device)
    key_positions = torch.arange(key_length, device=raw_query.device)
    native_query = _rotate(raw_query, query_positions, native)
    native_key = _rotate(raw_key, key_positions, native)
    candidate_query = _rotate(raw_query, query_positions, candidate)
    candidate_key = _rotate(raw_key, key_positions, candidate)
    if query_heads != key_heads:
        groups = query_heads // key_heads
        native_key = native_key.repeat_interleave(groups, dim=1)
        candidate_key = candidate_key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    native_key_t = native_key.float().transpose(-1, -2)
    candidate_key_t = candidate_key.float().transpose(-1, -2)
    value_float = value.float()
    outputs = []
    for start in range(0, query_length, query_tile_size):
        stop = min(start + query_tile_size, query_length)
        native_logits = torch.matmul(native_query[:, :, start:stop].float(), native_key_t)
        candidate_logits = torch.matmul(candidate_query[:, :, start:stop].float(), candidate_key_t)
        native_logits.mul_(float(scale))
        candidate_logits.mul_(float(scale))
        visible = causal[start:stop][None, None]
        probabilities = stable_rank_probabilities(native_logits, candidate_logits, visible)
        outputs.append(torch.matmul(probabilities, value_float).to(value.dtype))
    return torch.cat(outputs, dim=-2)


def install(model, native_table, candidate_table, method: str = "rank_assignment"):
    """Install Scheme C on all OLMo-2 layers; return idempotent restore."""
    if method != "rank_assignment":
        raise ValueError("method must be rank_assignment")
    config = model.config
    if getattr(config, "model_type", None) != "olmo2":
        raise ValueError("rank assignment is OLMo2-only")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or not len(layers):
        raise ValueError("model.model.layers is required")
    native_values = _table(native_table, "native_table")
    candidate_values = _table(candidate_table, "candidate_table")
    if native_values.shape != candidate_values.shape:
        raise ValueError("Native and candidate tables must share one geometry")
    head_dim = int(
        getattr(config, "head_dim", 0)
        or getattr(config, "hidden_size", 0) // getattr(config, "num_attention_heads", 1)
    )
    if head_dim <= 0 or native_values.size * 2 != head_dim:
        raise ValueError("table geometry differs from model head_dim")

    # Exact identity uses the stock attention/cache path, including its kernels.
    if np.array_equal(native_values, candidate_values):
        def restore_identity():
            return None
        restore_identity.method_id = METHOD_ID
        restore_identity.identity_fast_path = True
        return restore_identity

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
            del position_embeddings
            if module.training:
                raise RuntimeError("rank assignment is frozen-inference only")
            if kwargs.get("output_attentions"):
                raise ValueError("rank assignment does not materialize full attention weights")
            if hidden_states.shape[0] != 1:
                raise ValueError("rank assignment requires batch=1")
            raw_query, raw_key, value = _project_qkv(module, hidden_states)
            if past_key_values is not None:
                raw_key, value = past_key_values.update(
                    raw_key,
                    value,
                    module.layer_idx,
                    {"cache_position": cache_position},
                )
            query_length, key_length = raw_query.shape[-2], raw_key.shape[-2]
            if cache_position is not None:
                expected = torch.arange(
                    key_length - query_length, key_length, device=cache_position.device,
                )
                if cache_position.ndim != 1 or not torch.equal(cache_position, expected):
                    raise ValueError("rank assignment requires contiguous absolute positions")
            native, candidate = tables_for(raw_query.device)
            output = rank_assignment_attention(
                raw_query,
                raw_key,
                value,
                native,
                candidate,
                float(module.scaling),
                attention_mask=attention_mask,
            )
            input_shape = hidden_states.shape[:-1]
            output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            return module.o_proj(output), None

        forward.__wrapped__ = original
        return forward

    for layer in layers:
        attention = layer.self_attn
        if getattr(attention, "_native_rank_assignment_method", None) is not None:
            raise RuntimeError("rank assignment is already installed")
        for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"):
            if not hasattr(attention, name):
                raise ValueError(f"OLMo attention is missing {name}")
        original = attention.forward
        originals.append((attention, original))
        attention.forward = types.MethodType(make_forward(original), attention)
        attention._native_rank_assignment_method = method

    def restore():
        nonlocal restored
        if restored:
            return
        for attention, original in originals:
            attention.forward = original
            delattr(attention, "_native_rank_assignment_method")
        table_cache.clear()
        restored = True

    restore.method_id = METHOD_ID
    restore.identity_fast_path = False
    return restore


def _plan() -> dict:
    return {
        "status": "PLAN_ONLY",
        "method": "rank_assignment",
        "method_id": METHOD_ID,
        "model_loaded": False,
        "gpu_execution": False,
        "note": "Import install() from an explicitly authorized frozen runner.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(_plan(), sort_keys=True))
