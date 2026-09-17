"""Scheme A even/odd parity attention for OLMo-2.

The operator forms four signed rotary Q/K branches and sends their summed
bilinear score through exactly one attention normalization and one PV product.
It does not average four independently normalized attention outputs.

Installing the operator only patches an already-loaded model.  This module has
no command-line execution path and never loads a checkpoint or starts a GPU run.
"""
from __future__ import annotations

import types
import numpy as np


METHOD_IDS = {
    "even_only": "native_followup_A_even_only_v1",
    "odd_only": "native_followup_A_odd_only_control_v1",
}
QUERY_TILE_SIZE = 64


def _table_values(table: dict, *, name: str) -> tuple[np.ndarray, float]:
    payload = table.get("table", table)
    values = np.asarray(payload.get("values_float32"), dtype=np.float32)
    gain = float(payload.get("gain"))
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must contain a nonempty values_float32 vector")
    if not np.isfinite(values).all() or np.any(values <= 0) or np.any(np.diff(values) >= 0):
        raise ValueError(f"{name} frequencies must be finite, positive, and strictly decreasing")
    if not np.isfinite(gain) or gain != 1.0:
        raise ValueError(f"{name} gain must remain exactly 1")
    return values, gain


def _frequency_embeddings(positions, values, *, dtype):
    """Return split-half cos/sin embeddings using FP32 phase arithmetic."""
    import torch

    frequencies = torch.as_tensor(values, device=positions.device, dtype=torch.float32)
    if positions.ndim != 2:
        raise ValueError("position_ids must have shape [batch, tokens]")
    with torch.autocast(device_type=positions.device.type, enabled=False):
        phase = positions.float()[..., None] * frequencies[None, None, :]
        phase = torch.cat((phase, phase), dim=-1)
        cos, sin = phase.cos(), phase.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _rotate_split_half(value, cos, sin):
    half = value.shape[-1] // 2
    if half == 0 or value.shape[-1] % 2:
        raise ValueError("rotary head dimension must be positive and even")
    rotated = value.new_empty(value.shape)
    rotated[..., :half] = -value[..., half:]
    rotated[..., half:] = value[..., :half]
    return value * cos[:, None] + rotated * sin[:, None]


def build_parity_states(
    query,
    key,
    positions,
    native_values,
    candidate_values,
    *,
    method: str,
):
    """Build Q/K whose one dot product is the requested parity score.

    ``even_only`` computes ``A*cos(d*candidate)+B*sin(d*native)``.
    ``odd_only`` computes ``A*cos(d*native)+B*sin(d*candidate)``.
    The four branches are concatenated along head dimension; signed one-half
    factors are placed on Q only, preserving the model's original scale.
    """
    import torch

    if method not in METHOD_IDS:
        raise ValueError(f"unknown parity method: {method}")
    if query.ndim != 4 or key.ndim != 4 or query.shape[0] != key.shape[0]:
        raise ValueError("query and key must have shape [batch, heads, tokens, head_dim]")
    if query.shape[-2:] != key.shape[-2:] or query.shape[-1] % 2:
        raise ValueError("query/key token and head dimensions must match and be even")
    pairs = query.shape[-1] // 2
    native = np.asarray(native_values, dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    if native.shape != (pairs,) or candidate.shape != (pairs,):
        raise ValueError("frequency count must equal half the attention head dimension")
    if not np.isfinite(native).all() or not np.isfinite(candidate).all():
        raise ValueError("frequency vectors contain nonfinite values")

    ncos, nsin = _frequency_embeddings(positions, native, dtype=query.dtype)
    ccos, csin = _frequency_embeddings(positions, candidate, dtype=query.dtype)
    qn_pos = _rotate_split_half(query, ncos, nsin)
    kn_pos = _rotate_split_half(key, ncos, nsin)
    qn_neg = _rotate_split_half(query, ncos, -nsin)
    kn_neg = _rotate_split_half(key, ncos, -nsin)
    qc_pos = _rotate_split_half(query, ccos, csin)
    kc_pos = _rotate_split_half(key, ccos, csin)
    qc_neg = _rotate_split_half(query, ccos, -csin)
    kc_neg = _rotate_split_half(key, ccos, -csin)

    if method == "even_only":
        q_parts = (qc_pos, qc_neg, qn_pos, -qn_neg)
        k_parts = (kc_pos, kc_neg, kn_pos, kn_neg)
    else:
        q_parts = (qn_pos, qn_neg, qc_pos, -qc_neg)
        k_parts = (kn_pos, kn_neg, kc_pos, kc_neg)
    query_augmented = torch.cat(tuple(part * 0.5 for part in q_parts), dim=-1)
    key_augmented = torch.cat(k_parts, dim=-1)
    return query_augmented, key_augmented


def project_olmo_qkv(module, hidden_states):
    """Apply OLMo projections and its mandatory post-projection Q/K norms."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_norm(module.q_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return query, key, value


def single_normalization_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    *,
    query_tile_size: int = QUERY_TILE_SIZE,
    **kwargs,
):
    """Compute one global softmax/PV per query row using bounded query tiles.

    The score of each row is formed from the complete visible key axis before
    normalization.  Tiling only bounds memory; it does not partition the key
    domain or normalize the four parity branches separately.
    """
    import torch

    if module.training:
        raise ValueError("frozen parity attention requires model.eval()")
    if kwargs.get("output_attentions"):
        raise ValueError("parity attention does not materialize full attention weights")
    if query_tile_size <= 0:
        raise ValueError("query_tile_size must be positive")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("augmented Q/K dimensions differ")
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("Q/K/V batch dimensions differ")
    query_heads, key_heads = query.shape[1], key.shape[1]
    if query_heads % key_heads or key_heads != value.shape[1]:
        raise ValueError("unsupported Q/K/V head mapping")
    if query_heads != key_heads:
        groups = query_heads // key_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    batch, _, query_length, _ = query.shape
    key_length = key.shape[-2]
    outputs = []
    key_transposed = key.float().transpose(-1, -2)
    value_float = value.float()
    minimum = torch.finfo(torch.float32).min
    for start in range(0, query_length, query_tile_size):
        stop = min(start + query_tile_size, query_length)
        scores = torch.matmul(query[:, :, start:stop].float(), key_transposed)
        scores.mul_(float(module.scaling))
        if attention_mask is None:
            offset = key_length - query_length
            q_positions = torch.arange(start + offset, stop + offset, device=query.device)
            k_positions = torch.arange(key_length, device=query.device)
            scores.masked_fill_(k_positions[None, :] > q_positions[:, None], minimum)
        elif attention_mask.ndim == 4:
            scores.add_(attention_mask[:, :, start:stop, :key_length].float())
        elif attention_mask.ndim == 2:
            if attention_mask.shape != (batch, key_length):
                raise ValueError("two-dimensional attention mask shape differs")
            scores.masked_fill_(~attention_mask[:, None, None, :].bool(), minimum)
        else:
            raise ValueError("attention mask must be None, [B,K], or [B,1,Q,K]")
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probabilities, value_float).to(dtype=value.dtype))
    return torch.cat(outputs, dim=-2), None


def install(model, native_table: dict, candidate_table: dict, method: str):
    """Patch every OLMo attention layer and return an idempotent restore callback."""
    if method not in METHOD_IDS:
        raise ValueError(f"method must be one of {sorted(METHOD_IDS)}")
    if getattr(model.config, "model_type", None) != "olmo2":
        raise ValueError("parity attention currently supports OLMo-2 only")
    native, _ = _table_values(native_table, name="native_table")
    candidate, _ = _table_values(candidate_table, name="candidate_table")
    if native.shape != candidate.shape:
        raise ValueError("native and candidate frequency counts differ")

    # An exact identity request must retain the stock kernel and cache layout.
    if np.array_equal(native, candidate):
        def restore_identity():
            return None

        restore_identity.method_id = METHOD_IDS[method]
        return restore_identity

    position_context: dict[str, object] = {}

    def position_hook(_module, args, kwargs):
        positions = args[1] if len(args) > 1 else kwargs.get("position_ids")
        if positions is None:
            raise RuntimeError("OLMo rotary embedding did not receive position_ids")
        position_context["position_ids"] = positions

    position_handle = model.model.rotary_emb.register_forward_pre_hook(position_hook, with_kwargs=True)
    originals = []

    def patched_forward(original):
        def forward(module, hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, **kwargs):
            del position_embeddings  # Both geometries are rebuilt from the same absolute positions.
            positions = position_context.get("position_ids")
            if positions is None:
                raise RuntimeError("missing absolute positions for parity attention")
            input_shape = hidden_states.shape[:-1]
            query, key, value = project_olmo_qkv(module, hidden_states)
            query, key = build_parity_states(
                query, key, positions, native, candidate, method=method,
            )
            if past_key_values is not None:
                key, value = past_key_values.update(key, value, module.layer_idx)
            output, weights = single_normalization_attention(
                module, query, key, value, attention_mask, **kwargs,
            )
            output = output.reshape(*input_shape, -1).contiguous()
            return module.o_proj(output), weights

        forward.__name__ = getattr(original, "__name__", "forward")
        return forward

    for block in model.model.layers:
        attention = block.self_attn
        if not hasattr(attention, "q_norm") or not hasattr(attention, "k_norm"):
            position_handle.remove()
            raise RuntimeError("OLMo parity attention requires q_norm and k_norm")
        if attention.head_dim != 2 * native.size:
            position_handle.remove()
            raise ValueError("frequency count differs from the model rotary dimension")
        originals.append((attention, attention.forward))
        attention.forward = types.MethodType(patched_forward(attention.forward), attention)

    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        for attention, original in originals:
            attention.forward = original
        position_handle.remove()
        restored = True

    restore.method_id = METHOD_IDS[method]
    return restore
