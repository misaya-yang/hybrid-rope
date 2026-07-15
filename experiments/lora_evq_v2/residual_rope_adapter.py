#!/usr/bin/env python3
"""Add a zero-gated residual RoPE score branch to Llama attention."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Mapping

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.rope.schedules import evq_cosh_inv_freq, geometric_inv_freq


class ResidualRoPEBranch(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        query_heads: int,
        key_value_heads: int,
        branch_dim: int,
        base: float,
        method: str,
        tau: float,
        projection_mode: str = "learned",
        query_gate_start: int | None = None,
    ) -> None:
        super().__init__()
        if branch_dim <= 0 or branch_dim % 2:
            raise ValueError("branch_dim must be a positive even integer")
        if projection_mode not in {"learned", "native_qk"}:
            raise ValueError("projection_mode must be learned or native_qk")
        if query_gate_start is not None and query_gate_start <= 0:
            raise ValueError("query_gate_start must be positive")
        self.query_heads = int(query_heads)
        self.key_value_heads = int(key_value_heads)
        self.branch_dim = int(branch_dim)
        self.base = float(base)
        self.tau = float(tau)
        self.projection_mode = projection_mode
        self.query_gate_start = query_gate_start
        self.q_proj = (
            nn.Linear(hidden_size, query_heads * branch_dim, bias=False)
            if projection_mode == "learned"
            else None
        )
        self.k_proj = (
            nn.Linear(hidden_size, key_value_heads * branch_dim, bias=False)
            if projection_mode == "learned"
            else None
        )
        self.alpha = nn.Parameter(torch.zeros(query_heads))
        self.register_buffer("inv_freq", torch.empty(branch_dim // 2), persistent=True)
        self.enabled = True
        self.set_method(method)

    def set_method(self, method: str) -> None:
        if method == "native_geo":
            inv_freq = geometric_inv_freq(self.branch_dim, self.base, dtype=torch.float32)
        elif method == "evq_cosh":
            inv_freq = evq_cosh_inv_freq(
                self.branch_dim,
                self.tau,
                self.base,
                midpoint=True,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"unsupported residual RoPE method: {method}")
        self.method = method
        self.inv_freq.copy_(inv_freq.to(device=self.inv_freq.device))

    def project(
        self,
        hidden_states: torch.Tensor | None,
        cache_position: torch.Tensor | None,
        *,
        native_query: torch.Tensor | None = None,
        native_key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.projection_mode == "learned":
            if hidden_states is None or self.q_proj is None or self.k_proj is None:
                raise ValueError("learned residual projection requires hidden_states")
            batch, length, _ = hidden_states.shape
            query = self.q_proj(hidden_states).view(
                batch, length, self.query_heads, self.branch_dim
            ).transpose(1, 2)
            key = self.k_proj(hidden_states).view(
                batch, length, self.key_value_heads, self.branch_dim
            ).transpose(1, 2)
        else:
            if native_query is None or native_key is None:
                raise ValueError("native_qk residual projection requires native Q/K")
            query, key = native_query, native_key
            batch, _, length, width = query.shape
            if (
                width != self.branch_dim
                or key.shape != (batch, self.key_value_heads, length, self.branch_dim)
                or query.shape[1] != self.query_heads
            ):
                raise ValueError("native Q/K shape differs from the residual branch contract")

        if cache_position is None:
            positions = torch.arange(length, device=hidden_states.device).view(1, length)
        elif cache_position.ndim == 1:
            positions = cache_position.view(1, -1)
        elif cache_position.ndim == 2:
            positions = cache_position
        else:
            raise ValueError("cache_position must be rank 1 or 2")
        if positions.shape[-1] != length:
            raise ValueError("cache_position length differs from hidden_states")
        if positions.shape[0] == 1 and batch != 1:
            positions = positions.expand(batch, -1)
        if positions.shape[0] != batch:
            raise ValueError("cache_position batch differs from hidden_states")

        phases = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        phases = torch.cat((phases, phases), dim=-1)
        cos = phases.cos().to(dtype=query.dtype).unsqueeze(1)
        sin = phases.sin().to(dtype=query.dtype).unsqueeze(1)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        query = query * self.alpha.to(query.dtype).view(1, -1, 1, 1)
        if self.query_gate_start is not None:
            query = query * (positions >= self.query_gate_start).to(query.dtype).view(
                batch, 1, length, 1
            )
        return query, key

    def bypasses(self, cache_position: torch.Tensor | None, length: int) -> bool:
        if self.query_gate_start is None:
            return False
        last_position = length - 1 if cache_position is None else int(cache_position.max())
        return last_position < self.query_gate_start


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _attention_interface(module: nn.Module):
    from transformers.models.llama.modeling_llama import eager_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    implementation = module.config._attn_implementation
    if implementation == "eager":
        return eager_attention_forward
    getter = getattr(ALL_ATTENTION_FUNCTIONS, "get_interface", None)
    return getter(implementation, eager_attention_forward) if getter else ALL_ATTENTION_FUNCTIONS[implementation]


def _residual_attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    branch: ResidualRoPEBranch = self.residual_rope_branch
    zero_gate = not bool(torch.count_nonzero(branch.alpha.detach()))
    if not branch.enabled or (
        not self.training
        and (zero_gate or branch.bypasses(cache_position, hidden_states.shape[1]))
    ):
        return self.__dict__["_residual_original_forward"](
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    residual_query, residual_key = branch.project(
        hidden_states,
        cache_position,
        native_query=query_states if branch.projection_mode == "native_qk" else None,
        native_key=key_states if branch.projection_mode == "native_qk" else None,
    )
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = torch.cat((query_states, residual_query), dim=-1)
    key_states = torch.cat((key_states, residual_key), dim=-1)
    value_states = torch.cat(
        (value_states, value_states.new_zeros(*value_states.shape[:-1], branch.branch_dim)),
        dim=-1,
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface = _attention_interface(self)
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output[..., : self.head_dim]
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def attention_modules(model: nn.Module) -> list[nn.Module]:
    modules = [
        module
        for module in model.modules()
        if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj", "layer_idx"))
    ]
    expected = int(model.config.num_hidden_layers)
    if len(modules) != expected:
        raise RuntimeError(f"found {len(modules)} attention modules; expected {expected}")
    return modules


def attach_residual_rope(
    model: nn.Module,
    *,
    branch_dim: int = 8,
    method: str = "native_geo",
    tau: float = 1.414,
    seed: int = 42,
    projection_mode: str = "learned",
    query_gate_start: int | None = None,
) -> None:
    torch.manual_seed(seed)
    for module in attention_modules(model):
        if hasattr(module, "residual_rope_branch"):
            raise RuntimeError("residual RoPE branch is already attached")
        if projection_mode == "native_qk" and branch_dim != int(module.head_dim):
            raise ValueError("native_qk branch_dim must equal the model attention head_dim")
        branch = ResidualRoPEBranch(
            hidden_size=int(model.config.hidden_size),
            query_heads=int(model.config.num_attention_heads),
            key_value_heads=int(model.config.num_key_value_heads),
            branch_dim=branch_dim,
            base=float(getattr(model.config, "rope_theta", 500000.0)),
            method=method,
            tau=tau,
            projection_mode=projection_mode,
            query_gate_start=query_gate_start,
        ).to(device=module.q_proj.weight.device)
        if branch.q_proj is not None and branch.k_proj is not None:
            branch.q_proj.to(dtype=module.q_proj.weight.dtype)
            branch.k_proj.to(dtype=module.q_proj.weight.dtype)
        module.add_module("residual_rope_branch", branch)
        module.__dict__["_residual_original_forward"] = module.forward
        module.forward = MethodType(_residual_attention_forward, module)


def set_residual_method(model: nn.Module, method: str) -> None:
    for module in attention_modules(model):
        module.residual_rope_branch.set_method(method)


def set_residual_enabled(model: nn.Module, enabled: bool) -> None:
    for module in attention_modules(model):
        module.residual_rope_branch.enabled = bool(enabled)


def set_residual_alpha(model: nn.Module, value: float) -> None:
    with torch.no_grad():
        for module in attention_modules(model):
            module.residual_rope_branch.alpha.fill_(float(value))


def set_residual_active_layers(model: nn.Module, layer_indices: set[int]) -> None:
    expected = set(range(int(model.config.num_hidden_layers)))
    if not layer_indices or not layer_indices <= expected:
        raise ValueError("residual active layers are empty or out of range")
    for module in attention_modules(model):
        module.residual_rope_branch.enabled = int(module.layer_idx) in layer_indices


def freeze_base(model: nn.Module) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable = []
    for module in attention_modules(model):
        for parameter in module.residual_rope_branch.parameters():
            parameter.requires_grad = bool(module.residual_rope_branch.enabled)
            if parameter.requires_grad:
                trainable.append(parameter)
    return trainable


def residual_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if ".residual_rope_branch." in name
    }


def load_residual_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    current = model.state_dict()
    expected = {name for name in current if ".residual_rope_branch." in name}
    if set(state) != expected:
        raise ValueError("residual state keys differ from the attached model")
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))


def alpha_summary(model: nn.Module) -> dict[str, float]:
    values = torch.cat(
        [
            module.residual_rope_branch.alpha.detach().float().cpu()
            for module in attention_modules(model)
            if module.residual_rope_branch.enabled
        ]
    )
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "l1_mean": float(values.abs().mean()),
    }


def self_test() -> None:
    torch.manual_seed(42)
    q_native = torch.randn(2, 4, 7, 8)
    k_native = torch.randn(2, 4, 7, 8)
    q_extra = torch.randn(2, 4, 7, 4)
    k_extra = torch.randn(2, 4, 7, 4)
    alpha = torch.randn(1, 4, 1, 1)
    combined = torch.cat((q_native, alpha * q_extra), -1) @ torch.cat(
        (k_native, k_extra), -1
    ).transpose(-1, -2)
    expected = q_native @ k_native.transpose(-1, -2) + alpha * (
        q_extra @ k_extra.transpose(-1, -2)
    )
    torch.testing.assert_close(combined, expected)
    probabilities = combined.softmax(dim=-1)
    value = torch.randn(2, 4, 7, 8)
    padded_value = torch.cat((value, torch.zeros(2, 4, 7, 4)), dim=-1)
    padded_output = probabilities @ padded_value
    torch.testing.assert_close(padded_output[..., :8], probabilities @ value)
    assert not bool(torch.count_nonzero(padded_output[..., 8:]))
    branch = ResidualRoPEBranch(
        hidden_size=16,
        query_heads=4,
        key_value_heads=2,
        branch_dim=8,
        base=500000.0,
        method="native_geo",
        tau=1.414,
    )
    native = branch.inv_freq.clone()
    branch.set_method("evq_cosh")
    assert branch.inv_freq.shape == native.shape and not torch.allclose(branch.inv_freq, native)
    reused = ResidualRoPEBranch(
        hidden_size=16,
        query_heads=4,
        key_value_heads=2,
        branch_dim=8,
        base=500000.0,
        method="evq_cosh",
        tau=1.414,
        projection_mode="native_qk",
        query_gate_start=3,
    )
    reused.alpha.data.fill_(0.1)
    raw_query = torch.randn(1, 4, 5, 8)
    raw_key = torch.randn(1, 2, 5, 8)
    query, key = reused.project(
        None,
        torch.arange(5),
        native_query=raw_query,
        native_key=raw_key,
    )
    assert query.shape == raw_query.shape and key.shape == raw_key.shape
    assert not bool(torch.count_nonzero(query[:, :, :3]))
    assert bool(torch.count_nonzero(query[:, :, 3:]))
    assert reused.bypasses(torch.arange(3), 3)
    assert not reused.bypasses(torch.arange(5), 5)


if __name__ == "__main__":
    self_test()
    print("residual_rope_adapter self-test ok")
