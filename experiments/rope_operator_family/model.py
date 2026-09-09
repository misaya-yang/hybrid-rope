"""Fuse one fitted operator into a real HF model with a physically compact cache."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .operator import OperatorFactors, Shape, rotate


def _mask_slice(mask, q_start, q_end, k_start, k_end):
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask[:, None, None, k_start:k_end].bool()
    if mask.ndim != 4:
        raise ValueError("attention mask must be a 2D padding mask or 4D causal mask")
    return mask[..., q_start:q_end, k_start:k_end]


def chunked_attention(q: Tensor, k: Tensor, v: Tensor, mask, scale: float,
                      query_chunk: int = 128, key_chunk: int = 1024) -> Tensor:
    """Exact online softmax without an N x N allocation; differentiable reference."""
    nq, nk = q.shape[-2], k.shape[-2]
    result = []
    for q_start in range(0, nq, query_chunk):
        q_end = min(q_start + query_chunk, nq)
        qb = q[..., q_start:q_end, :].float()
        m = torch.full((*qb.shape[:-1], 1), -torch.inf, device=q.device)
        denominator = torch.zeros_like(m)
        accumulator = torch.zeros((*qb.shape[:-1], v.shape[-1]), device=q.device)
        qp = nk - nq + torch.arange(q_start, q_end, device=q.device)
        for k_start in range(0, nk, key_chunk):
            k_end = min(k_start + key_chunk, nk)
            logits = (qb @ k[..., k_start:k_end, :].float().transpose(-1, -2)) * scale
            valid = torch.arange(k_start, k_end, device=q.device)[None, :] <= qp[:, None]
            logits = logits.masked_fill(~valid, -torch.inf)
            block_mask = _mask_slice(mask, q_start, q_end, k_start, k_end)
            if block_mask is not None:
                if block_mask.dtype == torch.bool:
                    logits = logits.masked_fill(~block_mask, -torch.inf)
                else:
                    logits = logits + block_mask.float()
                    logits = logits.masked_fill(block_mask <= -torch.finfo(block_mask.dtype).max / 2, -torch.inf)
            new_m = torch.maximum(m, logits.amax(-1, keepdim=True))
            safe_m = torch.where(torch.isfinite(new_m), new_m, torch.zeros_like(new_m))
            old_weight = torch.where(torch.isfinite(m), (m - safe_m).exp(), torch.zeros_like(m))
            weights = (logits - safe_m).exp()
            accumulator = old_weight * accumulator + weights @ v[..., k_start:k_end, :].float()
            denominator = old_weight * denominator + weights.sum(-1, keepdim=True)
            m = new_m
        result.append((accumulator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)).to(q.dtype))
    return torch.cat(result, dim=-2)


def _project(module: nn.Linear, weight: Tensor, bias: Tensor | None):
    with torch.no_grad():
        module.weight.copy_(weight)
        if module.bias is not None:
            module.bias.copy_(torch.zeros_like(module.bias) if bias is None else bias)


class CompactAttention(nn.Module):
    """Persistent cache contains only rotated k_R and content c, once per token.

    HF DynamicCache supports different key/value widths. We use those two slots
    for rotary keys and content latents; expanded K/V never enter the cache.
    """
    def __init__(self, original, factors: OperatorFactors, layer_idx: int, backend="auto"):
        super().__init__()
        self.shape = factors.shape
        self.layer_idx, self.backend = layer_idx, backend
        self.config = original.config
        self.head_dim = self.shape.head_dim
        self.scaling = float(getattr(original, "scaling", self.head_dim ** -0.5))
        self.attention_dropout = 0.0
        self.is_causal = True
        self.sliding_window = None
        self.phase_scale = factors.phase_scale
        self.phase = nn.Parameter(factors.phase.detach().float().clone())
        hidden = original.q_proj.in_features
        h, d, r, p = self.shape.heads, self.head_dim, self.shape.content_rank, self.shape.rotary_dim
        dtype, device = original.q_proj.weight.dtype, original.q_proj.weight.device
        factory = dict(dtype=dtype, device=device)
        bias = any(m.bias is not None for m in (original.q_proj, original.k_proj, original.v_proj))
        self.q_content = nn.Linear(hidden, h * r, bias=bias, **factory)
        self.q_rotary = nn.Linear(hidden, h * p, bias=bias, **factory)
        self.content = nn.Linear(hidden, r, bias=bias, **factory)
        self.k_rotary = nn.Linear(hidden, p, bias=bias, **factory)
        self.value_up = nn.Parameter(factors.U.detach().to(**factory).clone())
        self.o_proj = copy.deepcopy(original.o_proj)
        with torch.no_grad():
            f = factors.to(device=device, dtype=torch.float32)
            qw = original.q_proj.weight.float().reshape(h, d, hidden)
            qb = original.q_proj.bias.float().reshape(h, d) if original.q_proj.bias is not None else None
            kw, vw = original.k_proj.weight.float(), original.v_proj.weight.float()
            kb = original.k_proj.bias.float() if original.k_proj.bias is not None else torch.zeros(kw.shape[0], device=device)
            vb = original.v_proj.bias.float() if original.v_proj.bias is not None else torch.zeros(vw.shape[0], device=device)
            _project(self.q_content, torch.einsum("hdx,hdr->hrx", qw, f.P).flatten(0, 1),
                     torch.einsum("hd,hdr->hr", qb, f.P).flatten() if qb is not None else None)
            _project(self.q_rotary, torch.einsum("hdx,hdp->hpx", qw, f.A).flatten(0, 1),
                     torch.einsum("hd,hdp->hp", qb, f.A).flatten() if qb is not None else None)
            _project(self.content, f.C.T @ torch.cat((kw, vw)), f.C.T @ torch.cat((kb, vb)))
            _project(self.k_rotary, f.B.T @ kw, f.B.T @ kb)
        self.phase.data = self.phase.data.to(device=device, dtype=torch.float32)
        self.last_backend = None

    def _rot(self, tensor, positions):
        # rotate() takes [tokens, heads, dims]; batch loop avoids position copies.
        return torch.stack([rotate(row, pos, self.phase / self.phase_scale)
                            for row, pos in zip(tensor, positions)])

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, past_key_value=None, cache_position=None,
                position_ids=None, **kwargs):
        cache = past_key_values if past_key_values is not None else past_key_value
        b, n, _ = hidden_states.shape
        h, r, p = self.shape.heads, self.shape.content_rank, self.shape.rotary_dim
        if position_ids is None:
            if cache_position is None:
                seen = cache.get_seq_length(self.layer_idx) if cache is not None else 0
                cache_position = torch.arange(seen, seen + n, device=hidden_states.device)
            position_ids = cache_position.reshape(1, n)
        positions = position_ids.expand(b, -1)
        qc = self.q_content(hidden_states).reshape(b, n, h, r).transpose(1, 2)
        qr = self._rot(self.q_rotary(hidden_states).reshape(b, n, h, p), positions).transpose(1, 2)
        kr = self._rot(self.k_rotary(hidden_states).reshape(b, n, 1, p), positions).transpose(1, 2)
        content = self.content(hidden_states).unsqueeze(1)
        if cache is not None:
            kr, content = cache.update(kr, content, self.layer_idx, {"cache_position": cache_position})
        query = torch.cat((qc, qr), dim=-1)
        keys = torch.cat((content, kr), dim=-1)
        backend = self.backend
        if backend == "auto":
            backend = "sdpa" if query.shape[-1] <= 256 else "chunked"
        self.last_backend = backend
        if backend == "sdpa":
            nq, nk = query.shape[-2], keys.shape[-2]
            mask = attention_mask
            if mask is not None and mask.ndim == 2:
                mask = mask[:, None, None, :].bool()
                causal = torch.arange(nk, device=query.device)[None, :] <= (nk - nq + torch.arange(nq, device=query.device))[:, None]
                mask = mask & causal
            if mask is None and 1 < nq < nk:
                mask = torch.arange(nk, device=query.device)[None, :] <= (nk - nq + torch.arange(nq, device=query.device))[:, None]
            values = F.pad(content, (0, p))
            aggregate = F.scaled_dot_product_attention(
                query.contiguous(), keys.contiguous(), values.contiguous(), attn_mask=mask,
                dropout_p=0.0, is_causal=mask is None and nq == nk and nq > 1,
                scale=self.scaling, enable_gqa=True)[..., :r]
        elif backend == "chunked":
            aggregate = chunked_attention(query, keys, content, attention_mask, self.scaling)
        else:
            raise ValueError(f"unknown attention backend: {backend}")
        output = torch.einsum("bhtr,hrd->bhtd", aggregate, self.value_up)
        output = output.transpose(1, 2).reshape(b, n, h * self.head_dim)
        return self.o_proj(output), None


def cache_bytes(cache) -> int:
    """Count backing storage once, including capacity beyond a tensor view."""
    storages = {}
    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if isinstance(value, Tensor) and value.numel():
                storage = value.untyped_storage()
                storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    if not getattr(cache, "layers", None):
        for name in ("key_cache", "value_cache"):
            for value in getattr(cache, name, []):
                if isinstance(value, Tensor) and value.numel():
                    storage = value.untyped_storage()
                    storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def install_factors(model, directory: str | Path, backend="auto"):
    directory = Path(directory)
    for index, layer in enumerate(model.model.layers):
        saved = torch.load(directory / f"layer_{index:03d}.pt", map_location="cpu", weights_only=True)
        factors = OperatorFactors(Shape(**saved["shape"]), saved["phase_scale"])
        factors.load_state_dict(saved["state"])
        layer.self_attn = CompactAttention(layer.self_attn, factors, index, backend)
    return model
