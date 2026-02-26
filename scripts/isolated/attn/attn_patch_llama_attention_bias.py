#!/usr/bin/env python3
"""
Runtime monkey-patch for LlamaAttention attention-logit bias.

Patch equation:
    logits = sem_logits - gamma * gate * S2(delta)

Safety:
- Default mode is OFF (no behavior change).
- Patch is runtime-only and can be restored.
- No model/config files are modified.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class AttentionBiasConfig:
    mode: str = "off"  # off | bias | bias+gate
    gamma_mode: str = "constant"  # constant | per-layer | head-group
    gamma: float = 0.0
    gamma_by_layer: str = ""  # CSV, e.g. "0,1e-4,3e-4,..."
    gamma_head_low: float = 0.0
    gamma_head_high: float = 0.0
    tau: float = 0.0
    tg: float = 1.0
    s2_power: float = 2.0
    s2_table_path: str = ""  # optional 1D tensor indexed by delta
    require_s2_table: bool = False
    enabled: bool = False
    # Safety guard: pre-RoPE semantic gating is mathematically mismatched unless
    # internal RoPE-applied Q/K are used. Keep disabled by default.
    allow_prerope_gate: bool = False
    # Guard against quadratic memory blow-up in bias+gate mode.
    gate_max_qk: int = 4_194_304  # 2048 * 2048


class PatchHandle:
    def __init__(self) -> None:
        self._entries: List[Tuple[torch.nn.Module, object]] = []

    def register(self, module: torch.nn.Module, original_forward: object) -> None:
        self._entries.append((module, original_forward))

    def restore(self) -> None:
        for module, original_forward in self._entries:
            module.forward = original_forward  # type: ignore[attr-defined]
        self._entries.clear()


def _parse_gamma_by_layer(text: str) -> List[float]:
    vals: List[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    return vals


def _distance_s2_table(
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
    cfg: AttentionBiasConfig,
    power: float,
    cache: Dict[Tuple[str, int, int, str, float, str], torch.Tensor],
    s2_1d_cache: Dict[Tuple[str, str, str, int], torch.Tensor],
) -> torch.Tensor:
    s2_path = str(cfg.s2_table_path or "").strip()
    key = (str(device), int(q_len), int(k_len), str(dtype), float(power), s2_path)
    if key in cache:
        return cache[key]

    q_idx = torch.arange(q_len, device=device)
    k_idx = torch.arange(k_len, device=device)
    delta = (q_idx[:, None] - k_idx[None, :]).abs().to(torch.float32)

    s2 = None
    if s2_path:
        path_obj = Path(s2_path)
        if path_obj.exists():
            max_delta = int(max(q_len, k_len))
            cache_key = (path_obj.as_posix(), str(device), str(dtype), max_delta)
            if cache_key not in s2_1d_cache:
                raw = torch.load(path_obj, map_location="cpu")
                if isinstance(raw, dict):
                    raw = raw.get("s2_by_delta", raw.get("s2", raw.get("values", raw)))
                vec = torch.as_tensor(raw, dtype=torch.float32).view(-1)
                if vec.numel() <= 1:
                    raise RuntimeError(f"Invalid s2_table_path content: {s2_path}")
                if vec.numel() < max_delta:
                    pad = vec[-1].expand(max_delta - vec.numel())
                    vec = torch.cat([vec, pad], dim=0)
                elif vec.numel() > max_delta:
                    vec = vec[:max_delta]
                s2_1d_cache[cache_key] = vec.to(device=device, dtype=dtype)
            vec = s2_1d_cache[cache_key]
            delta_idx = delta.to(torch.long).clamp(min=0, max=vec.numel() - 1)
            s2 = vec[delta_idx]
        else:
            if bool(cfg.require_s2_table):
                raise FileNotFoundError(f"require_s2_table=True but s2_table_path not found: {s2_path}")
            warnings.warn(
                f"s2_table_path not found: {s2_path}; fallback to power-law s2.",
                RuntimeWarning,
                stacklevel=2,
            )
    elif bool(cfg.require_s2_table):
        raise RuntimeError("require_s2_table=True but s2_table_path is empty.")
    if s2 is None:
        s2 = torch.pow(delta + 1.0, -float(power)).to(dtype=dtype)
    cache[key] = s2
    return s2


def _gamma_tensor(
    cfg: AttentionBiasConfig,
    layer_idx: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mode = str(cfg.gamma_mode).strip().lower()
    if mode == "per-layer":
        table = _parse_gamma_by_layer(cfg.gamma_by_layer)
        g = table[layer_idx] if layer_idx < len(table) else (table[-1] if table else cfg.gamma)
        return torch.full((1, 1, 1, 1), float(g), device=device, dtype=dtype)
    if mode == "head-group":
        lo = float(cfg.gamma_head_low)
        hi = float(cfg.gamma_head_high if cfg.gamma_head_high > 0 else cfg.gamma)
        half = max(1, num_heads // 2)
        gh = torch.empty((1, num_heads, 1, 1), device=device, dtype=dtype)
        gh[:, :half] = lo
        gh[:, half:] = hi
        return gh
    return torch.full((1, 1, 1, 1), float(cfg.gamma), device=device, dtype=dtype)


def _is_zero_gamma(gamma_t: torch.Tensor) -> bool:
    return bool(torch.count_nonzero(gamma_t).item() == 0)


def _ensure_additive_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        z = torch.zeros((), device=mask.device, dtype=torch.float32)
        ninf = torch.full((), -1e9, device=mask.device, dtype=torch.float32)
        return torch.where(mask, z, ninf)
    return mask


def _compute_gate_from_sem_logits(sem_logits: torch.Tensor, tau: float, tg: float) -> torch.Tensor:
    temp = max(float(tg), 1e-6)
    max_sem = sem_logits.max(dim=-1, keepdim=True).values
    # stop-gradient on semantic logit branch
    z = (max_sem - sem_logits.detach() - float(tau)) / temp
    return torch.sigmoid(z).to(dtype=sem_logits.dtype)


def _infer_attention_dims(self, hidden_states: torch.Tensor) -> Tuple[int, int, int]:
    """
    Infer (num_heads, num_key_value_heads, head_dim) robustly across
    transformer versions where attention modules expose different attributes.
    """
    hidden_size = int(hidden_states.shape[-1])
    q_proj = getattr(self, "q_proj", None)
    k_proj = getattr(self, "k_proj", None)
    q_out = int(getattr(q_proj, "out_features", hidden_size))
    k_out = int(getattr(k_proj, "out_features", hidden_size))
    n_heads_hint = int(getattr(self, "num_heads", 0) or 0)
    n_kv_hint = int(getattr(self, "num_key_value_heads", 0) or 0)
    hdim_hint = int(getattr(self, "head_dim", 0) or 0)

    hdim = 0
    if hdim_hint > 0 and q_out % hdim_hint == 0 and k_out % hdim_hint == 0:
        hdim = hdim_hint
    elif n_heads_hint > 0 and q_out % n_heads_hint == 0:
        hdim = q_out // n_heads_hint
    elif n_kv_hint > 0 and k_out % n_kv_hint == 0:
        hdim = k_out // n_kv_hint
    else:
        for cand in (128, 64, 256, 32, 16, 8):
            if q_out % cand == 0 and k_out % cand == 0:
                hdim = cand
                break
        if hdim <= 0:
            hdim = max(1, math.gcd(q_out, k_out))

    n_heads = max(1, q_out // hdim)
    n_kv = max(1, k_out // hdim)
    return n_heads, n_kv, hdim


def _find_llama_attention_modules(model: torch.nn.Module) -> List[Tuple[int, torch.nn.Module]]:
    """
    Prefer true transformer layer index (0..L-1) to keep per-layer gamma mapping correct.
    Fall back to module traversal only when layer container is unavailable.
    """
    modules: List[Tuple[int, torch.nn.Module]] = []
    layer_container = getattr(getattr(model, "model", None), "layers", None)
    if layer_container is not None:
        for layer_idx, layer in enumerate(layer_container):
            for child in layer.modules():
                if "llamaattention" in child.__class__.__name__.lower():
                    modules.append((layer_idx, child))
        if modules:
            return modules

    for idx, module in enumerate(model.modules()):
        if "llamaattention" in module.__class__.__name__.lower():
            modules.append((idx, module))
    return modules


def apply_llama_attention_bias_patch(
    model: torch.nn.Module,
    cfg: AttentionBiasConfig,
) -> PatchHandle:
    handle = PatchHandle()
    if not cfg.enabled or str(cfg.mode).lower() == "off":
        return handle

    s2_cache: Dict[Tuple[str, int, int, str, float, str], torch.Tensor] = {}
    s2_1d_cache: Dict[Tuple[str, str, str, int], torch.Tensor] = {}
    mode = str(cfg.mode).lower()
    gate_enabled = bool(mode == "bias+gate" and cfg.allow_prerope_gate)
    if mode == "bias+gate" and not gate_enabled:
        warnings.warn(
            "attn_bias_mode=bias+gate requested but allow_prerope_gate=False; "
            "falling back to bias mode to avoid pre-RoPE gating mismatch.",
            RuntimeWarning,
            stacklevel=2,
        )

    modules = _find_llama_attention_modules(model)

    for layer_idx, module in modules:
        original_forward = module.forward
        handle.register(module, original_forward)

        def wrapped_forward(
            self,  # noqa: ANN001
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.Tensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            _original_forward=original_forward,
            _layer_idx=layer_idx,
            **kwargs,
        ):
            if str(cfg.mode).lower() == "off":
                return _original_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            bsz, q_len, _ = hidden_states.shape
            k_len = q_len
            if attention_mask is not None and attention_mask.ndim >= 4:
                k_len = int(attention_mask.shape[-1])

            n_heads, n_kv, hdim = _infer_attention_dims(self, hidden_states)
            dtype = hidden_states.dtype
            device = hidden_states.device

            s2 = _distance_s2_table(
                q_len=q_len,
                k_len=k_len,
                device=device,
                dtype=dtype,
                cfg=cfg,
                power=float(cfg.s2_power),
                cache=s2_cache,
                s2_1d_cache=s2_1d_cache,
            )
            gamma_t = _gamma_tensor(cfg, _layer_idx, n_heads, device=device, dtype=dtype)
            if _is_zero_gamma(gamma_t) and not gate_enabled:
                return _original_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            bias = -gamma_t * s2.view(1, 1, q_len, k_len)

            if gate_enabled:
                use_gate = int(q_len) * int(k_len) <= int(cfg.gate_max_qk)
                if not use_gate:
                    warnings.warn(
                        "bias+gate disabled for this batch because q_len*k_len exceeds gate_max_qk; "
                        "falling back to bias-only to avoid OOM.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if use_gate:
                    # Approximate semantic logits from current hidden_states.
                    q = self.q_proj(hidden_states)
                    k = self.k_proj(hidden_states)

                    q = q.view(bsz, q_len, n_heads, hdim).transpose(1, 2)
                    k = k.view(bsz, q_len, n_kv, hdim).transpose(1, 2)
                    if n_kv != n_heads:
                        rep = max(1, math.ceil(float(n_heads) / float(n_kv)))
                        k = k.repeat_interleave(rep, dim=1)[:, :n_heads, :, :]
                    sem_logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(float(hdim))
                    gate = _compute_gate_from_sem_logits(sem_logits, tau=float(cfg.tau), tg=float(cfg.tg))
                    if sem_logits.shape[-1] != k_len:
                        gate = gate[..., :k_len]
                    bias = bias.to(gate.dtype) * gate.to(bias.dtype)

            if attention_mask is None:
                patched_mask = bias
            else:
                m = _ensure_additive_mask(attention_mask)
                if m.ndim == 2:
                    m = m[:, None, None, :]
                elif m.ndim == 3:
                    m = m[:, None, :, :]
                patched_mask = m + bias.to(dtype=m.dtype)

            return _original_forward(
                hidden_states=hidden_states,
                attention_mask=patched_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        module.forward = wrapped_forward.__get__(module, module.__class__)  # type: ignore[attr-defined]

    return handle
