#!/usr/bin/env python3
"""Self-contained 151.9M decoder architecture for the paired experiment."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + 1e-6
        ).to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq: int, inv_freq: torch.Tensor) -> None:
        super().__init__()
        if int(dim) % 2:
            raise ValueError(f"rotary dimension must be even, got {dim}")
        if tuple(inv_freq.shape) != (int(dim) // 2,):
            raise ValueError(
                f"inv_freq must have shape {(int(dim) // 2,)}, got {inv_freq.shape}"
            )
        self.register_buffer("inv_freq", inv_freq)
        self.attention_scaling = 1.0
        self._build(int(max_seq))

    def _build(self, seq_len: int) -> None:
        positions = torch.arange(
            int(seq_len), dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )
        frequencies = torch.outer(positions, self.inv_freq)
        embedding = torch.cat([frequencies, frequencies], dim=-1)
        self.register_buffer("cos_c", embedding.cos(), persistent=False)
        self.register_buffer("sin_c", embedding.sin(), persistent=False)
        self._max = int(seq_len)

    def forward(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if int(length) > self._max:
            self._build(int(length))
        cos = self.cos_c[: int(length)]
        sin = self.sin_c[: int(length)]
        if float(self.attention_scaling) != 1.0:
            cos = cos * float(self.attention_scaling)
            sin = sin * float(self.attention_scaling)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # Keeping the cached table in float32 protects phase construction; casting
    # at application time prevents BF16 attention activations from upcasting.
    cos_value = cos.to(dtype=x.dtype)
    sin_value = sin.to(dtype=x.dtype)
    return x * cos_value + rotate_half(x) * sin_value


class Attention(nn.Module):
    def __init__(self, config: dict[str, Any], rope: RotaryEmbedding) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        self.num_heads = int(config["num_heads"])
        self.head_dim = int(config["head_dim"])
        if hidden != self.num_heads * self.head_dim:
            raise ValueError(
                f"hidden_size={hidden} must equal num_heads*head_dim="
                f"{self.num_heads * self.head_dim}"
            )
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = (
            self.qkv(x)
            .view(batch, length, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope(length)
        query = apply_rope(query, cos[None, None], sin[None, None])
        key = apply_rope(key, cos[None, None], sin[None, None])
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        return self.output(
            attended.transpose(1, 2).reshape(batch, length, -1)
        )


class MLP(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        intermediate = int(config["intermediate_size"])
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: dict[str, Any], rope: RotaryEmbedding) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        self.norm1 = RMSNorm(hidden)
        self.attention = Attention(config, rope)
        self.norm2 = RMSNorm(hidden)
        self.mlp = MLP(config)

    @property
    def attn(self) -> Attention:
        """Compatibility alias used by the evaluator's rotary injector."""
        return self.attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config: dict[str, Any], inv_freq: torch.Tensor) -> None:
        super().__init__()
        self.config = dict(config)
        self.num_layers = int(config["num_layers"])
        hidden = int(config["hidden_size"])
        self.embedding = nn.Embedding(int(config["vocab_size"]), hidden)
        rope = RotaryEmbedding(
            int(config["head_dim"]),
            int(config["max_position_embeddings"]),
            inv_freq,
        )
        self.blocks = nn.ModuleList(
            [Block(config, rope) for _ in range(self.num_layers)]
        )
        self.final_norm = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, int(config["vocab_size"]), bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize)
        residual_scale = 1.0 / math.sqrt(2 * self.num_layers)
        for block in self.blocks:
            nn.init.normal_(
                block.attention.output.weight, std=0.02 * residual_scale
            )
            nn.init.normal_(block.mlp.down.weight, std=0.02 * residual_scale)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        if hidden.is_cuda and torch.is_autocast_enabled():
            hidden = hidden.to(dtype=torch.get_autocast_dtype("cuda"))
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))

    def extend_rope(self, length: int) -> None:
        self.blocks[0].attention.rope._build(int(length))
