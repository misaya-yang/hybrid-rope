"""Bifocal attention with explicit call-wide, row-wise, or fixed horizon semantics.

Inference-only single-sequence reference for Qwen2/3. No padding, arbitrary
position jumps, dropout, sliding layers, or compressed caches are supported.
The reference backend is bounded to small inputs; real long runs require FA2.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def rotate(x, positions, inv_freq):
    # HF split-half rotary convention, [B, H, N, D].
    phases = positions.float()[:, None] * inv_freq.float()[None, :]
    phases = torch.cat((phases, phases), dim=-1)[None, None]
    half = x.shape[-1] // 2
    opposite = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * phases.cos().to(x.dtype) + opposite * phases.sin().to(x.dtype)


def groups(start, count, window, mode, fixed_horizon=None):
    end = start + count
    if mode == "native":
        return [(0, count, 1)]
    if mode in ("call", "fixed"):
        horizon = end if mode == "call" else fixed_horizon
        if horizon is None or horizon < end:
            raise ValueError("fixed horizon must cover every input and generated token")
        return [(0, count, max(1, math.ceil(horizon / window)))]
    if mode != "row":
        raise ValueError(mode)
    out = []
    a = start
    while a < end:
        g = a // window + 1
        b = min(end, g * window)
        out.append((a - start, b - start, g))
        a = b
    return out


def reference(q, k, v, start, inv_freq, group, local_window, scale):
    nq, nk = q.shape[-2], k.shape[-2]
    if nk > 2048:
        raise RuntimeError("dense diagnostic backend is limited to 2048 keys")
    qp = torch.arange(start, start + nq, device=q.device)
    kp = torch.arange(nk, device=q.device)
    repeat = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    near = rotate(q, qp, inv_freq).float() @ rotate(k, kp, inv_freq).float().transpose(-1, -2)
    if group == 1:
        logits = near
    else:
        far = rotate(q, qp // group, inv_freq).float() @ rotate(k, kp // group, inv_freq).float().transpose(-1, -2)
        logits = torch.where((qp[:, None] - kp[None, :]) <= local_window, near, far)
    logits = (logits * scale).masked_fill(kp[None, :] > qp[:, None], -torch.inf)
    return (logits.softmax(-1) @ v.float()).to(q.dtype)


def flash(q, k, v, start, inv_freq, group, local_window, scale):
    """Same A+B-C decomposition as bifocal literature; FP32 LSE merge.

    q must be a suffix of k for FA2's bottom-right causal alignment. This routine
    is a functional baseline, not a new optimized kernel. Cancellation error is
    explicitly checked against the dense reference before long model runs.
    """
    nq, nk = q.shape[-2], k.shape[-2]
    assert start + nq == nk
    qp = torch.arange(start, start + nq, device=q.device)
    kp = torch.arange(nk, device=q.device)
    qb, kb = rotate(q, qp, inv_freq), rotate(k, kp, inv_freq)

    def fa(qi, ki, window=(-1, -1)):
        # PyTorch 2.8 exposes its bundled FA2 with LSE and local windows.
        result = torch.ops.aten._flash_attention_forward(
            qi.transpose(1, 2), ki.transpose(1, 2), v.transpose(1, 2),
            None, None, nq, nk, 0.0, True, False, scale=scale,
            window_size_left=window[0], window_size_right=window[1])
        output, lse = result[:2]
        return output.transpose(1, 2).float(), lse.float()[..., None]

    if group == 1:
        return fa(qb, kb)[0].to(q.dtype)
    qg, kg = rotate(q, qp // group, inv_freq), rotate(k, kp // group, inv_freq)
    # A is local native, B full grouped, C local grouped.
    a, la = fa(qb, kb, (local_window, 0))
    b, lb = fa(qg, kg)
    c, lc = fa(qg, kg, (local_window, 0))
    m = torch.maximum(torch.maximum(la, lb), lc)
    wa, wb, wc = (la - m).exp(), (lb - m).exp(), (lc - m).exp()
    den = wa + (wb - wc)
    if not bool(torch.all(den > 0)):
        raise FloatingPointError("nonpositive bifocal merge denominator")
    return ((wa * a + wb * b - wc * c) / den).to(q.dtype)


class ScheduledAttention(nn.Module):
    def __init__(self, original, inv_freq, window, local_window, mode,
                 backend="reference", fixed_horizon=None):
        super().__init__()
        if getattr(original, "sliding_window", None) is not None:
            raise ValueError("sliding attention is outside this first comparison")
        self.original = original
        self.layer_idx = original.layer_idx
        self.config = original.config
        self.head_dim = original.head_dim
        self.window, self.local_window = window, local_window
        self.mode, self.backend, self.fixed_horizon = mode, backend, fixed_horizon
        self.register_buffer("inv_freq", inv_freq.detach().float().clone())

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, past_key_value=None, cache_position=None,
                position_ids=None, **kwargs):
        if self.training:
            raise RuntimeError("this comparison is inference-only")
        if hidden_states.shape[0] != 1:
            raise ValueError("single unpadded sequence required")
        cache = past_key_values if past_key_values is not None else past_key_value
        n = hidden_states.shape[1]
        start = cache.get_seq_length(self.layer_idx) if cache is not None else 0
        positions = torch.arange(start, start + n, device=hidden_states.device)
        for supplied in (cache_position, position_ids):
            if supplied is not None and not torch.equal(supplied.flatten(), positions):
                raise ValueError("logical positions must be contiguous from zero")
        if attention_mask is not None:
            if attention_mask.ndim == 2 and not bool(torch.all(attention_mask == 1)):
                raise ValueError("padding is unsupported")
            if attention_mask.ndim == 4:
                expected = torch.arange(start + n, device=positions.device)[None, :] <= positions[:, None]
                actual = attention_mask[0, 0, -n:, :start + n]
                actual = actual if actual.dtype == torch.bool else actual >= 0
                if not torch.equal(actual, expected):
                    raise ValueError("only the standard full causal mask is supported")
        o = self.original
        shape = (1, n, -1, self.head_dim)
        q, k = o.q_proj(hidden_states).view(shape), o.k_proj(hidden_states).view(shape)
        if hasattr(o, "q_norm"):
            q, k = o.q_norm(q), o.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = o.v_proj(hidden_states).view(shape).transpose(1, 2)
        if cache is not None:
            # Raw K is intentional; this cache cannot be reused by unpatched attention.
            k, v = cache.update(k, v, self.layer_idx, {"cache_position": positions})
        kernel = {"reference": reference, "flash": flash}[self.backend]
        pieces = []
        for a, b, g in groups(start, n, self.window, self.mode, self.fixed_horizon):
            key_end = start + b
            pieces.append(kernel(q[..., a:b, :], k[..., :key_end, :], v[..., :key_end, :],
                                 start + a, self.inv_freq, g, self.local_window, o.scaling))
        output = torch.cat(pieces, dim=-2).transpose(1, 2).reshape(1, n, -1)
        return o.o_proj(output), None


def install(model, window, local_window, mode, backend="reference", fixed_horizon=None):
    if model.config.model_type not in ("qwen2", "qwen3"):
        raise ValueError("only Qwen2/3 are audited for this experiment")
    if not 0 <= local_window < window:
        raise ValueError("local window must be smaller than the native window")
    inv_freq = model.model.rotary_emb.inv_freq
    for layer in model.model.layers:
        original = layer.self_attn.original if isinstance(layer.self_attn, ScheduledAttention) else layer.self_attn
        layer.self_attn = ScheduledAttention(original, inv_freq, window, local_window,
                                             mode, backend, fixed_horizon)
    model.eval()
    return model
