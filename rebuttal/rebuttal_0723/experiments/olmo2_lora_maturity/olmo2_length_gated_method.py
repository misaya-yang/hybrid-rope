"""OLMo-2 length-gated EVQ+LoRA with exact short-mode preservation.

For every row whose maximum position ID is below 4096, this method calls the
original Native rotary module and the LoRA wrappers return their frozen base
linears directly. For a row reaching position 4096 or above, the full EVQ
rotary module and Q/K/V/O LoRA branch are active for the entire row.

The gate is global per sequence, not per token. This preserves a consistent
frequency grid within every sequence and makes the <=4K branch exactly the
untouched checkpoint computation rather than an approximate regularization
target.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    endpoint_evq_inv_freq,
    endpoint_geo_inv_freq,
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    trainable_named_parameters,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    LoRALinear,
)


SHORT_CONTEXT_LIMIT = 4_096
LENGTH_GATED_FREQUENCY_NAME = (
    "length_gated_native_le4k_evq_gt4k"
)
NATIVE_FREQUENCY_SHA256 = (
    "dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34"
)
EVQ_FREQUENCY_SHA256 = (
    "917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607"
)
GateMode = Literal["short", "long", "mixed"]
ForcedMode = Optional[Literal["short", "long"]]


class LengthModeState:
    """Forward-local sequence-mode state shared by rotary and LoRA modules."""

    def __init__(self, *, short_context_limit: int) -> None:
        if int(short_context_limit) <= 0:
            raise ValueError("short context limit must be positive")
        self.short_context_limit = int(short_context_limit)
        self.forced_mode: ForcedMode = None
        self.mode: GateMode | None = None
        self.row_is_long: torch.Tensor | None = None

    def force(self, mode: ForcedMode) -> None:
        if mode not in {None, "short", "long"}:
            raise ValueError(f"invalid forced length mode: {mode!r}")
        self.forced_mode = mode
        self.mode = mode
        self.row_is_long = None

    def update(self, position_ids: torch.Tensor) -> GateMode:
        if position_ids.ndim != 2 or position_ids.numel() == 0:
            raise RuntimeError("length gate requires nonempty [B, S] positions")
        if self.forced_mode is not None:
            self.mode = self.forced_mode
            self.row_is_long = None
            return self.mode
        row_is_long = (
            position_ids.amax(dim=-1) >= self.short_context_limit
        )
        if (
            position_ids.shape[1] == 1
            and self.mode == "short"
            and bool(torch.any(row_is_long).item())
        ):
            raise RuntimeError(
                "cached generation crossed the 4K length gate; choose and "
                "force the branch from the total context budget before "
                "creating the KV cache"
            )
        self.row_is_long = row_is_long
        if bool(torch.all(~row_is_long).item()):
            self.mode = "short"
        elif bool(torch.all(row_is_long).item()):
            self.mode = "long"
        else:
            self.mode = "mixed"
        return self.mode

    def broadcast_gate(self, value: torch.Tensor) -> torch.Tensor:
        if self.mode != "mixed" or self.row_is_long is None:
            raise RuntimeError("row gate is only defined in mixed mode")
        rows = self.row_is_long
        if rows.numel() not in {1, value.shape[0]}:
            raise RuntimeError("length-gate batch dimension drift")
        shape = (rows.numel(),) + (1,) * (value.ndim - 1)
        return rows.reshape(shape).to(
            device=value.device,
            dtype=value.dtype,
        )


class LengthGatedRotaryEmbedding(nn.Module):
    """Dispatch each sequence to the untouched Native or full-EVQ module."""

    def __init__(
        self,
        native_rotary: nn.Module,
        state: LengthModeState,
    ) -> None:
        super().__init__()
        native = (
            native_rotary.inv_freq.detach().cpu().to(torch.float32)
        )
        if (
            not torch.equal(native, endpoint_geo_inv_freq())
            or tensor_sha256(native) != NATIVE_FREQUENCY_SHA256
        ):
            raise RuntimeError("length gate did not receive Native OLMo RoPE")
        self.native = native_rotary
        self.evq = copy.deepcopy(native_rotary)
        replacement = endpoint_evq_inv_freq().to(
            device=self.evq.inv_freq.device,
            dtype=self.evq.inv_freq.dtype,
        )
        with torch.no_grad():
            self.evq.inv_freq.copy_(replacement)
        self.evq.original_inv_freq = self.evq.inv_freq
        if tensor_sha256(self.evq.inv_freq) != EVQ_FREQUENCY_SHA256:
            raise RuntimeError("length-gated EVQ frequency identity drift")
        self.state = state

    def forward(
        self,
        value: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.state.update(position_ids)
        if mode == "short":
            return self.native(value, position_ids)
        if mode == "long":
            return self.evq(value, position_ids)
        native_cos, native_sin = self.native(value, position_ids)
        evq_cos, evq_sin = self.evq(value, position_ids)
        gate = self.state.broadcast_gate(native_cos)
        return (
            torch.where(gate.bool(), evq_cos, native_cos),
            torch.where(gate.bool(), evq_sin, native_sin),
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "active_frequency": LENGTH_GATED_FREQUENCY_NAME,
            "active_sha256_float32": EVQ_FREQUENCY_SHA256,
            "short_branch": {
                "maximum_position_id": SHORT_CONTEXT_LIMIT - 1,
                "frequency": "native_endpoint_rope",
                "frequency_sha256_float32": NATIVE_FREQUENCY_SHA256,
                "rotary_dispatch": "original_module_direct_call",
            },
            "long_branch": {
                "minimum_maximum_position_id": SHORT_CONTEXT_LIMIT,
                "frequency": "evq_endpoint_cosh",
                "frequency_sha256_float32": EVQ_FREQUENCY_SHA256,
                "scope": "entire_sequence",
            },
            "gate_axis": "per_sequence_max_position_id",
            "cached_generation_policy": (
                "force short or long from the total context budget before "
                "the prompt forward; short-to-long KV-cache transitions fail"
            ),
            "claim_boundary": (
                "short-branch structural identity only; long capability "
                "still requires strict autoregressive exact+EOS evaluation"
            ),
        }


class LengthGatedLoRALinear(LoRALinear):
    """LoRA that returns its base linear directly in short mode."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        *,
        state: LengthModeState,
    ) -> None:
        super().__init__(
            base,
            rank=rank,
            alpha=alpha,
            output_mask=None,
        )
        self.length_mode_state = state

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mode = self.length_mode_state.mode
        if mode is None:
            raise RuntimeError(
                "rotary length gate must execute before attention projections"
            )
        if mode == "short":
            return self.base(value)
        update = F.linear(F.linear(value, self.a), self.b)
        if mode == "long":
            return self.base(value) + update * self.scale
        gate = self.length_mode_state.broadcast_gate(update)
        return self.base(value) + update * self.scale * gate


class LengthGatedEOSVocabRowHead(nn.Module):
    """Change only the EOS logit using the collapsed rank-one row update.

    A one-output-row LoRA product has no more expressivity than one hidden-size
    vector. Parameterizing that product directly avoids the zero-B first-step
    gradient dead zone while preserving the same rank-one function class.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        *,
        eos_token_id: int,
        state: LengthModeState,
    ) -> None:
        super().__init__()
        if int(rank) != 1 or not math.isfinite(float(alpha)):
            raise ValueError("invalid EOS-head LoRA rank or alpha")
        if not 0 <= int(eos_token_id) < int(base.out_features):
            raise ValueError("EOS token ID is outside the lm_head vocabulary")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.eos_token_id = int(eos_token_id)
        self.length_mode_state = state
        self.delta_weight = nn.Parameter(torch.zeros(base.in_features))
        self.eos_bias = nn.Parameter(torch.zeros(()))

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        logits = self.base(value)
        mode = self.length_mode_state.mode
        if mode is None:
            raise RuntimeError(
                "rotary length gate must execute before the EOS lm_head"
            )
        if mode == "short":
            return logits
        delta = F.linear(
            value, self.delta_weight.unsqueeze(0)
        ).squeeze(-1)
        delta = delta * self.scale + self.eos_bias.to(
            device=delta.device,
            dtype=delta.dtype,
        )
        if mode == "mixed":
            delta = delta * self.length_mode_state.broadcast_gate(delta)
        eos = (
            logits[..., self.eos_token_id : self.eos_token_id + 1]
            + delta.unsqueeze(-1).to(dtype=logits.dtype)
        )
        return torch.cat(
            (
                logits[..., : self.eos_token_id],
                eos,
                logits[..., self.eos_token_id + 1 :],
            ),
            dim=-1,
        )


def freeze_length_gated_qkvo_adapter(model: Any) -> dict[str, Any]:
    """Freeze every installed length-gated QKVO LoRA tensor."""

    parameter_names: list[str] = []
    parameter_count = 0
    module_count = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, LengthGatedLoRALinear):
            continue
        module_count += 1
        for name in ("a", "b"):
            parameter = getattr(module, name)
            parameter.requires_grad_(False)
            parameter_names.append(f"model.{module_name}.{name}")
            parameter_count += int(parameter.numel())
    if module_count == 0:
        raise RuntimeError("no length-gated QKVO LoRA modules were installed")
    return {
        "qkvo_lora_modules": module_count,
        "parameter_tensors": len(parameter_names),
        "parameters": parameter_count,
        "parameter_names": parameter_names,
        "trainable_after_freeze": False,
    }


def install_length_gated_eos_vocab_row_head(
    model: Any,
    state: LengthModeState,
    *,
    rank: int,
    alpha: float,
    eos_token_id: int,
) -> tuple[LengthGatedEOSVocabRowHead, dict[str, Any]]:
    """Install a long-only EOS-row adapter over the frozen base lm_head."""

    qkvo_modules = [
        module
        for module in model.modules()
        if isinstance(module, LengthGatedLoRALinear)
    ]
    if not qkvo_modules or any(
        module.a.requires_grad or module.b.requires_grad
        for module in qkvo_modules
    ):
        raise RuntimeError(
            "load and freeze the length-gated QKVO parent before "
            "installing the EOS child adapter"
        )
    base = getattr(model, "lm_head", None)
    if not isinstance(base, nn.Linear):
        raise RuntimeError("length-gated EOS adapter requires a linear lm_head")
    head = LengthGatedEOSVocabRowHead(
        base,
        rank=int(rank),
        alpha=float(alpha),
        eos_token_id=int(eos_token_id),
        state=state,
    )
    model.lm_head = head
    return head, {
        "adaptation": "length_gated_eos_vocab_row",
        "scope": "long_mode_only",
        "modified_vocab_rows": [int(eos_token_id)],
        "short_dispatch": "base_lm_head_direct_call",
        "rank": int(rank),
        "alpha": float(alpha),
        "scalar_bias": True,
        "trainable_parameters": int(
            head.delta_weight.numel() + head.eos_bias.numel()
        ),
        "trainable_parameter_tensors": 2,
        "parameterization": "direct_single_eos_row_delta",
        "rank1_equivalent": True,
    }


def eos_head_trainable_named_parameters(
    model: Any,
) -> list[tuple[str, nn.Parameter]]:
    """Enumerate only the EOS child state after verifying a frozen parent."""

    values: list[tuple[str, nn.Parameter]] = []
    qkvo_modules = 0
    eos_heads = 0
    for module_name, module in model.named_modules():
        if isinstance(module, LengthGatedLoRALinear):
            qkvo_modules += 1
            if module.a.requires_grad or module.b.requires_grad:
                raise RuntimeError("QKVO parent is not frozen")
        elif isinstance(module, LengthGatedEOSVocabRowHead):
            eos_heads += 1
            values.extend(
                (
                    (
                        f"model.{module_name}.delta_weight",
                        module.delta_weight,
                    ),
                    (f"model.{module_name}.eos_bias", module.eos_bias),
                )
            )
    if qkvo_modules == 0 or eos_heads != 1:
        raise RuntimeError(
            "EOS child requires a frozen QKVO parent and exactly one EOS head"
        )
    names = [name for name, _ in values]
    if len(names) != len(set(names)):
        raise RuntimeError("EOS child parameter names are not unique")
    if any(not parameter.requires_grad for _, parameter in values):
        raise RuntimeError("EOS child parameter unexpectedly frozen")
    all_trainable = {
        name
        for name, _ in trainable_named_parameters(model, None)
    }
    if all_trainable != set(names):
        raise RuntimeError("trainable scope is not EOS-head-only")
    return values


def install_length_gated_qkvo(
    model: Any,
    *,
    rank: int,
    alpha: float,
) -> tuple[LengthModeState, dict[str, Any]]:
    """Install full-EVQ long mode and QKVO LoRA on OLMo-2 only."""

    if (
        getattr(model.config, "model_type", None) != "olmo2"
        or int(model.config.hidden_size) != 2_048
        or int(model.config.num_hidden_layers) != 16
        or int(model.config.num_attention_heads) != 16
        or int(model.config.num_key_value_heads) != 16
    ):
        raise RuntimeError("length-gated method requires OLMo-2 1.485B")
    if int(rank) <= 0 or not math.isfinite(float(alpha)):
        raise ValueError("invalid length-gated LoRA rank or alpha")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = LengthModeState(
        short_context_limit=SHORT_CONTEXT_LIMIT,
    )
    rotary = LengthGatedRotaryEmbedding(
        model.model.rotary_emb,
        state,
    )
    model.model.rotary_emb = rotary
    for layer in model.model.layers:
        attention = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            base = getattr(attention, name)
            if not isinstance(base, nn.Linear):
                raise RuntimeError(
                    f"length gate expected a linear {name}"
                )
            setattr(
                attention,
                name,
                LengthGatedLoRALinear(
                    base,
                    rank=int(rank),
                    alpha=float(alpha),
                    state=state,
                ),
            )
    named = trainable_named_parameters(model, None)
    if not named:
        raise RuntimeError("length-gated adapter has no trainable parameters")
    if any(
        not any(
            f".{projection}." in name
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        for name, _ in named
    ):
        raise RuntimeError("length-gated trainable scope escaped QKVO LoRA")
    return state, {
        **rotary.receipt(),
        "adaptation": "length_gated_qkvo_answer",
        "rank": int(rank),
        "alpha": float(alpha),
        "short_lora_dispatch": "base_linear_direct_call",
        "long_lora_dispatch": "qkvo_lora",
        "trainable_parameters": int(
            sum(parameter.numel() for _, parameter in named)
        ),
        "trainable_parameter_tensors": len(named),
    }
