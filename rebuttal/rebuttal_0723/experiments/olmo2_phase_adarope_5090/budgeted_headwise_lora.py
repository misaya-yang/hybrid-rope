"""Budgeted-table headwise Q/K LoRA for general long-context adaptation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


HEADS = 16
HEAD_DIM = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().float().numpy().astype("<f4", copy=False)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _split_order_delta(delta: np.ndarray) -> np.ndarray:
    order = np.asarray(
        [index for pair in range(64) for index in (pair, pair + 64)],
        dtype=np.int64,
    )
    split = np.zeros_like(delta)
    split[np.ix_(order, order)] = delta
    return split


def _direction(delta: np.ndarray, rank: int) -> torch.Tensor:
    split = _split_order_delta(np.asarray(delta, dtype=np.float32))
    _u, _s, vh = np.linalg.svd(split, full_matrices=False)
    return torch.from_numpy(np.ascontiguousarray(vh[:rank]))


class HeadwiseLoRANorm(nn.Module):
    """Frozen Q/K norm followed by an independent low-rank map per head."""

    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float,
        initial_down: torch.Tensor,
    ) -> None:
        super().__init__()
        if initial_down.shape != (rank, HEAD_DIM):
            raise ValueError("headwise direction shape drift")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.down = nn.Parameter(initial_down.repeat(HEADS, 1, 1))
        self.up = nn.Parameter(torch.zeros(HEADS, HEAD_DIM, rank))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        native = self.base(value)
        if native.shape[-1] != HEADS * HEAD_DIM:
            raise RuntimeError("OLMo-2 Q/K norm width drift")
        shaped = native.reshape(*native.shape[:-1], HEADS, HEAD_DIM)
        work = shaped.to(dtype=self.down.dtype)
        low = torch.einsum("...hd,hrd->...hr", work, self.down)
        update = torch.einsum("...hr,hdr->...hd", low, self.up)
        return (
            shaped + (update * self.scale).to(dtype=shaped.dtype)
        ).reshape_as(native)


def install_budgeted_headwise_qk(
    model: Any,
    *,
    target_table: Path,
    transport_map: Path,
    rank: int = 16,
    alpha: float = 16.0,
) -> dict[str, Any]:
    if (
        getattr(model.config, "model_type", None) != "olmo2"
        or int(model.config.hidden_size) != 2048
        or int(model.config.num_hidden_layers) != 16
        or int(model.config.num_attention_heads) != HEADS
        or int(model.config.num_key_value_heads) != HEADS
        or int(rank) <= 0
        or int(rank) > HEAD_DIM
    ):
        raise RuntimeError("registered headwise adapter requires OLMo-2 1B MHA")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    table_path = target_table.resolve()
    values = np.load(table_path, allow_pickle=False)
    if values.dtype != np.float32 or values.shape != (64,):
        raise RuntimeError("budgeted table must be float32 [64]")
    target = torch.from_numpy(np.ascontiguousarray(values)).float()
    if not torch.isfinite(target).all() or not torch.all(target[:-1] > target[1:]):
        raise RuntimeError("budgeted table is invalid")
    rotary = model.model.rotary_emb
    native = rotary.inv_freq.detach().cpu().float().clone()
    with torch.no_grad():
        rotary.inv_freq.copy_(target.to(rotary.inv_freq))
    if hasattr(rotary, "original_inv_freq"):
        rotary.original_inv_freq = rotary.inv_freq.detach().clone()
    rotary.attention_scaling = 1.0

    map_path = transport_map.resolve()
    map_meta_path = map_path.with_suffix(".json")
    map_meta = json.loads(map_meta_path.read_text(encoding="utf-8"))
    if (
        map_meta.get("status") != "ROPE_TRANSPORT_MAP_EXPORTED"
        or map_meta.get("map_file_sha256") != sha256_file(map_path)
        or map_meta.get("target_table_float32_sha256") != tensor_sha256(target)
        or int(map_meta.get("rank_bound_per_head", -1)) != int(rank)
    ):
        raise RuntimeError("transport initialization identity drift")
    with np.load(map_path, allow_pickle=False) as payload:
        query_down = _direction(payload["query_delta"], int(rank))
        key_down = _direction(payload["key_delta"], int(rank))

    modules = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        attention.q_norm = HeadwiseLoRANorm(
            attention.q_norm,
            rank=int(rank),
            alpha=float(alpha),
            initial_down=query_down,
        )
        attention.k_norm = HeadwiseLoRANorm(
            attention.k_norm,
            rank=int(rank),
            alpha=float(alpha),
            initial_down=key_down,
        )
        modules += 2
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if (
        modules != 32
        or not trainable
        or any(".q_norm." not in name and ".k_norm." not in name for name, _ in trainable)
        or any(not (name.endswith(".down") or name.endswith(".up")) for name, _ in trainable)
    ):
        raise RuntimeError("trainable scope escaped headwise Q/K LoRA")
    return {
        "method": "budgeted_headwise_qk_lora",
        "target_table_file_sha256": sha256_file(table_path),
        "target_table_float32_sha256": tensor_sha256(target),
        "native_table_float32_sha256": tensor_sha256(native),
        "transport_map_file_sha256": sha256_file(map_path),
        "transport_map_metadata_sha256": sha256_file(map_meta_path),
        "rank_per_head": int(rank),
        "alpha": float(alpha),
        "modules": modules,
        "trainable_parameters": int(sum(value.numel() for _, value in trainable)),
        "trainable_parameter_tensors": len(trainable),
        "initial_function": "exact zero LoRA update; transport singular directions initialize down factors",
        "short_mode": "Native rotary and base Q/K norms; adapter bypassed",
        "long_mode": "budgeted rotary and headwise Q/K LoRA",
    }


def adapter_state(model: Any) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if name.endswith(".down") or name.endswith(".up")
    }
    if len(state) != 64:
        raise RuntimeError(f"expected 64 headwise tensors, got {len(state)}")
    return state


def load_adapter_state(model: Any, state: dict[str, torch.Tensor]) -> None:
    expected = adapter_state(model)
    if set(state) != set(expected):
        raise RuntimeError("headwise adapter state keys drift")
    own = model.state_dict()
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != own[name].shape or not torch.isfinite(value).all():
                raise RuntimeError(f"headwise adapter tensor drift: {name}")
            own[name].copy_(value.to(own[name]))
