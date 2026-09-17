"""Per-layer static RoPE override for bounded native mechanism interventions."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def parse_layer_range(value: str, *, num_hidden_layers: int) -> tuple[int, ...]:
    first, separator, last = value.partition(":")
    if not separator:
        raise ValueError("layer range must be START:STOP with an exclusive stop")
    start, stop = int(first), int(last)
    if not 0 <= start < stop <= num_hidden_layers:
        raise ValueError("layer override range lies outside the model")
    return tuple(range(start, stop))


def _position_embeddings(hidden_states, *, inv_freq, gain: float, position_start: int):
    import torch

    batch, tokens = hidden_states.shape[:2]
    positions = torch.arange(
        position_start, position_start + tokens,
        device=hidden_states.device, dtype=torch.float32,
    ).expand(batch, -1)
    expanded = inv_freq[None, :, None].float().expand(batch, -1, 1)
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        frequencies = (expanded @ positions[:, None, :]).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos() * gain, embedding.sin() * gain


def install_layer_phase_override(
    model,
    *,
    layers: Iterable[int],
    values_float32: Iterable[float],
    gain: float,
):
    """Replace ``position_embeddings`` only inside selected attention layers.

    The hook is valid for unpadded batch-1 causal inference.  Cache length is
    read before the layer appends its current K/V, so decode positions match the
    stock global rotary module exactly.
    """
    import torch

    selected = tuple(sorted(set(int(value) for value in layers)))
    blocks = model.model.layers
    if not selected or any(value < 0 or value >= len(blocks) for value in selected):
        raise ValueError("layer override selection is empty or outside the model")
    values = np.asarray(list(values_float32), dtype=np.float32)
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    if (
        values.shape != (head_dim // 2,) or not np.isfinite(values).all()
        or np.any(values <= 0) or np.any(values[:-1] <= values[1:])
    ):
        raise ValueError("layer override table does not match full-rotary head geometry")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("layer override gain must be finite and positive")
    device = next(model.parameters()).device
    inv_freq = torch.from_numpy(values.copy()).to(device=device)
    handles = []

    def pre_hook(layer_index: int):
        def hook(_module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError("layer override requires unpadded batch-1 hidden states")
            past = kwargs.get("past_key_values")
            if past is None and len(args) > 3:
                past = args[3]
            position_start = int(past.get_seq_length(layer_index)) if past is not None else 0
            replacement = _position_embeddings(
                hidden, inv_freq=inv_freq, gain=float(gain), position_start=position_start,
            )
            if len(args) > 1:
                mutable = list(args)
                mutable[1] = replacement
                args = tuple(mutable)
            else:
                kwargs = dict(kwargs)
                kwargs["position_embeddings"] = replacement
            return args, kwargs
        return hook

    for layer in selected:
        handles.append(blocks[layer].self_attn.register_forward_pre_hook(
            pre_hook(layer), with_kwargs=True,
        ))
    return handles


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()
