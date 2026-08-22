"""Training-free length-conditioned RoPE for mature checkpoints.

The short branch calls the checkpoint's Native rotary module directly.  The
long branch uses one frozen non-geometric table plus a deterministic attention
amplitude.  The branch is chosen once from the request budget and must remain
fixed for every cached decode step.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn


Mode = Literal["short", "long", "mixed"]


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().float().numpy().astype("<f4", copy=False)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def matched_attention_scaling(factor: float) -> float:
    """Length-only amplitude used for a matched zero-training comparison."""
    if not math.isfinite(float(factor)) or float(factor) < 1.0:
        raise ValueError("context factor must be finite and at least one")
    return 1.0 + 0.1 * math.log(float(factor))


@dataclass
class RequestBudgetState:
    reference_length: int = 4096
    long_context_budget: int | None = None
    forced_mode: Literal["short", "long"] | None = None
    mode: Mode | None = None
    row_is_long: torch.Tensor | None = None

    def force_for_budget(self, total_context_budget: int) -> Mode:
        budget = int(total_context_budget)
        if budget <= 0:
            raise ValueError("total context budget must be positive")
        if (
            self.long_context_budget is not None
            and budget > int(self.long_context_budget)
        ):
            raise ValueError(
                "request budget exceeds the configured long-context profile"
            )
        self.forced_mode = "short" if budget <= self.reference_length else "long"
        self.mode = self.forced_mode
        self.row_is_long = None
        return self.mode

    def clear(self) -> None:
        self.forced_mode = None
        self.mode = None
        self.row_is_long = None

    def update(self, position_ids: torch.Tensor) -> Mode:
        if position_ids.ndim != 2 or position_ids.numel() == 0:
            raise RuntimeError("length-conditioned RoPE requires nonempty [B,S] positions")
        if self.forced_mode is not None:
            self.mode = self.forced_mode
            self.row_is_long = None
            return self.mode
        row_is_long = position_ids.amax(dim=-1) >= int(self.reference_length)
        if (
            position_ids.shape[1] == 1
            and self.mode == "short"
            and bool(torch.any(row_is_long).item())
        ):
            raise RuntimeError(
                "cached decoding crossed the Native/long boundary; bind the "
                "branch from the request budget before prefill"
            )
        self.row_is_long = row_is_long
        if bool(torch.all(~row_is_long).item()):
            self.mode = "short"
        elif bool(torch.all(row_is_long).item()):
            self.mode = "long"
        else:
            self.mode = "mixed"
        return self.mode

    def gate(self, value: torch.Tensor) -> torch.Tensor:
        if self.mode != "mixed" or self.row_is_long is None:
            raise RuntimeError("row gate exists only in mixed mode")
        shape = (self.row_is_long.numel(),) + (1,) * (value.ndim - 1)
        return self.row_is_long.reshape(shape).to(device=value.device)


class LengthConditionedRotaryEmbedding(nn.Module):
    """Dispatch to exact Native or frozen non-geometric long RoPE."""

    def __init__(
        self,
        native_rotary: nn.Module,
        *,
        long_inv_freq: torch.Tensor,
        long_attention_scaling: float,
        long_name: str,
        state: RequestBudgetState,
    ) -> None:
        super().__init__()
        native = getattr(native_rotary, "inv_freq", None)
        if not isinstance(native, torch.Tensor):
            raise ValueError("Native rotary module has no inverse-frequency tensor")
        target = torch.as_tensor(long_inv_freq, dtype=torch.float32).reshape(-1)
        if (
            target.shape != native.shape
            or not torch.isfinite(target).all()
            or not bool(torch.all(target > 0))
            or not bool(torch.all(target[:-1] > target[1:]))
        ):
            raise ValueError("long inverse-frequency table is invalid")
        if (
            not math.isfinite(float(long_attention_scaling))
            or float(long_attention_scaling) <= 0.0
        ):
            raise ValueError("long attention scaling must be finite and positive")
        self.native = native_rotary
        self.long = copy.deepcopy(native_rotary)
        replacement = target.to(
            device=self.long.inv_freq.device,
            dtype=self.long.inv_freq.dtype,
        )
        with torch.no_grad():
            self.long.inv_freq.copy_(replacement)
        if hasattr(self.long, "original_inv_freq"):
            self.long.original_inv_freq = self.long.inv_freq.detach().clone()
        self.long.attention_scaling = float(long_attention_scaling)
        self.long_name = str(long_name)
        self.state = state
        self.native_sha256 = tensor_sha256(native.detach().clone())
        self.long_sha256 = tensor_sha256(self.long.inv_freq)

    def forward(
        self,
        value: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.state.update(position_ids)
        if mode == "short":
            return self.native(value, position_ids)
        if mode == "long":
            return self.long(value, position_ids)
        native_cos, native_sin = self.native(value, position_ids)
        long_cos, long_sin = self.long(value, position_ids)
        gate = self.state.gate(native_cos)
        return (
            torch.where(gate, long_cos, native_cos),
            torch.where(gate, long_sin, native_sin),
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "method": "length_conditioned_non_geometric_rope",
            "reference_length": int(self.state.reference_length),
            "configured_long_context_budget": (
                None
                if self.state.long_context_budget is None
                else int(self.state.long_context_budget)
            ),
            "short_branch": {
                "operator": "Native rotary module direct call",
                "inv_freq_sha256_float32": self.native_sha256,
                "attention_scaling": float(
                    getattr(self.native, "attention_scaling", 1.0)
                ),
            },
            "long_branch": {
                "operator": self.long_name,
                "inv_freq_sha256_float32": self.long_sha256,
                "attention_scaling": float(self.long.attention_scaling),
            },
            "cache_policy": "branch fixed from total request budget before prefill",
            "learned_parameters": 0,
        }


def install_length_conditioned_rope(
    model: Any,
    *,
    long_inv_freq: torch.Tensor,
    long_attention_scaling: float,
    long_name: str,
    reference_length: int = 4096,
    long_context_budget: int,
) -> tuple[RequestBudgetState, dict[str, Any]]:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is None:
        raise ValueError("model has no shared rotary module")
    if int(long_context_budget) <= int(reference_length):
        raise ValueError("long context budget must exceed the reference length")
    state = RequestBudgetState(
        reference_length=int(reference_length),
        long_context_budget=int(long_context_budget),
    )
    wrapped = LengthConditionedRotaryEmbedding(
        rotary,
        long_inv_freq=long_inv_freq,
        long_attention_scaling=float(long_attention_scaling),
        long_name=long_name,
        state=state,
    )
    model.model.rotary_emb = wrapped
    return state, wrapped.receipt()
