#!/usr/bin/env python3
"""Native-preserving far-query chord high-pass attention residual.

Short requests delegate to the untouched OLMo-2 attention module.  Long
requests use one augmented Q/K attention score and one softmax:

    score = <q_native, k_native> / sqrt(d)
          + gate(position_q) * gain * <u, (I-R_omega(delta)) v> / sqrt(d)

The residual frequencies are eight fixed log-spaced wavelengths from 2x to
10x the 16K target length.  They have no phase wrap through 16K.  The scalar
bilinear response contains both sine and ``1-cos`` terms; its isotropic squared
energy is proportional to the phase-chord kernel ``1-cos(omega*delta)``.  It
is exactly zero at delta zero.  Values are zero-padded in the residual
coordinates, so the augmented SDPA returns the ordinary Native value
aggregation in its first ``d`` coordinates.

This is a new post-hoc attention operator inspired by the phase-chord result.
It is not EVQ-Cosh validation and is not evidence until a registered run
completes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.olmo2.modeling_olmo2 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    MODEL_CONTRACT,
    endpoint_geo_inv_freq,
    tensor_sha256,
)


METHOD_ID = "native_preserving_far_pass_chord_residual_v1"
ADAPTATION_NAME = "far_pass_chord_residual"
PREPARED_STATUS = "OLMO2_FAR_PASS_CHORD_RESIDUAL_PREPARED_NO_GPU"
READY_STATUS = "OLMO2_FAR_PASS_CHORD_RESIDUAL_GPU_READY"
RESULT_STATUS = "OLMO2_FAR_PASS_CHORD_RESIDUAL_COMPLETE"
ADAPTER_FORMAT_VERSION = 1
TARGET_LONG_LENGTH = 16_384
RESIDUAL_PAIRS = 8
WAVELENGTH_MIN = 2.0 * TARGET_LONG_LENGTH
WAVELENGTH_MAX = 10.0 * TARGET_LONG_LENGTH


def _rope_theta(config: Any) -> float:
    """Read the Native base across Transformers 4.x and 5.x configs."""

    value = getattr(config, "rope_theta", None)
    if value is None:
        parameters = getattr(config, "rope_parameters", None)
        if not isinstance(parameters, dict):
            raise RuntimeError("OLMo-2 config has no RoPE base")
        value = parameters.get("rope_theta")
    if value is None:
        raise RuntimeError("OLMo-2 config has no rope_theta")
    return float(value)


def far_pass_inv_freq(
    *,
    pairs: int = RESIDUAL_PAIRS,
    wavelength_min: float = WAVELENGTH_MIN,
    wavelength_max: float = WAVELENGTH_MAX,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the registered no-wrap logarithmic far-pass frequencies."""

    count = int(pairs)
    lower = float(wavelength_min)
    upper = float(wavelength_max)
    if count < 2 or not 0.0 < lower < upper:
        raise ValueError("invalid far-pass wavelength grid")
    ratio = (upper / lower) ** (1.0 / float(count - 1))
    wavelength = torch.tensor(
        [lower * ratio**index for index in range(count)],
        dtype=torch.float64,
    )
    frequency = (2.0 * math.pi / wavelength).to(dtype=dtype)
    if not bool(torch.all(frequency[:-1] > frequency[1:])):
        raise RuntimeError("far-pass frequencies are not strictly decreasing")
    return frequency


def _uniform_chord_mean(
    frequency: torch.Tensor,
    *,
    start: float,
    end: float,
) -> torch.Tensor:
    values = frequency.to(dtype=torch.float64)
    width = float(end) - float(start)
    if width <= 0.0:
        raise ValueError("chord interval must have positive width")
    return 1.0 - (
        torch.sin(values * float(end))
        - torch.sin(values * float(start))
    ) / (values * width)


def far_pass_frequency_receipt() -> dict[str, Any]:
    frequency = far_pass_inv_freq()
    wavelength = 2.0 * math.pi / frequency.to(dtype=torch.float64)
    near = _uniform_chord_mean(frequency, start=0.0, end=4_096.0)
    far = _uniform_chord_mean(
        frequency, start=4_096.0, end=float(TARGET_LONG_LENGTH)
    )
    return {
        "pairs": RESIDUAL_PAIRS,
        "residual_dimension": 2 * RESIDUAL_PAIRS,
        "augmented_head_dimension": 128 + 4 * RESIDUAL_PAIRS,
        "wavelength_tokens": wavelength.tolist(),
        "inv_freq": frequency.to(dtype=torch.float32).tolist(),
        "inv_freq_sha256_float32": tensor_sha256(frequency.float()),
        "maximum_phase_at_16k": float(
            frequency.max().double() * TARGET_LONG_LENGTH
        ),
        "mean_near_chord_0_4k": float(near.mean()),
        "mean_far_chord_4k_16k": float(far.mean()),
        "far_to_near_ratio": float(far.mean() / near.mean()),
        "selection": (
            "8 log-spaced wavelengths over [2*L_target,10*L_target]; "
            "no phase wrap through L_target"
        ),
    }


@dataclass(frozen=True)
class FarPassChordConfig:
    """Immutable method configuration saved with every adapter."""

    threshold_position: int = 4_096
    projection_rank: int = 64
    residual_pairs: int = RESIDUAL_PAIRS
    initial_logit_gain: float = 0.1
    rms_norm_eps: float = 1e-6
    wavelength_min: float = WAVELENGTH_MIN
    wavelength_max: float = WAVELENGTH_MAX
    initialization_seed: int = 20_260_821
    content_value_dim: int = 0
    content_projection_rank: int = 64
    initial_content_gain: float = 0.1

    @property
    def residual_dim(self) -> int:
        return 2 * int(self.residual_pairs)

    @property
    def augmented_head_dim(self) -> int:
        return 128 + 2 * self.residual_dim

    def validate(self, model_config: Any | None = None) -> None:
        if int(self.threshold_position) != 4_096:
            raise ValueError("registered method requires threshold position 4096")
        if int(self.projection_rank) <= 0:
            raise ValueError("projection rank must be positive")
        if int(self.residual_pairs) != RESIDUAL_PAIRS:
            raise ValueError(
                f"registered method requires {RESIDUAL_PAIRS} residual pairs"
            )
        if not 0.0 < float(self.initial_logit_gain) <= 1.0:
            raise ValueError("initial logit gain must lie in (0, 1]")
        if float(self.rms_norm_eps) <= 0.0:
            raise ValueError("RMSNorm epsilon must be positive")
        if (
            float(self.wavelength_min) != WAVELENGTH_MIN
            or float(self.wavelength_max) != WAVELENGTH_MAX
        ):
            raise ValueError("registered far-pass wavelength range drift")
        if self.augmented_head_dim != 160:
            raise ValueError("registered augmented head dimension must be 160")
        if self.augmented_head_dim % 8 or self.augmented_head_dim > 256:
            raise ValueError("augmented head dimension is not Flash eligible")
        if int(self.content_value_dim) not in {
            0,
            2 * int(self.residual_dim),
        }:
            raise ValueError(
                "content value width must be zero or fill all added dims"
            )
        if int(self.content_projection_rank) <= 0:
            raise ValueError("content projection rank must be positive")
        if not 0.0 < float(self.initial_content_gain) <= 1.0:
            raise ValueError("initial content gain must lie in (0, 1]")
        # The strict OLMo-2 head-dimension check belongs to installation.
        # Tiny CPU contract models intentionally use a smaller Native head.


class LowRankResidualProjection(nn.Module):
    """Independent low-rank projection; it never modifies a Native weight."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.a = nn.Parameter(torch.empty(self.rank, self.input_dim))
        self.b = nn.Parameter(torch.empty(self.output_dim, self.rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.b, a=math.sqrt(5))
        with torch.no_grad():
            self.b.mul_(0.01)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(hidden_states, self.a), self.b)


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("softplus target must be positive")
    return math.log(math.expm1(value))


def _cache_sequence_length(
    past_key_values: Cache | None,
    layer_index: int,
) -> int:
    if past_key_values is None:
        return 0
    try:
        return int(past_key_values.get_seq_length(layer_index))
    except TypeError:
        return int(past_key_values.get_seq_length())


def _cache_head_width(
    past_key_values: Cache | None,
    layer_index: int,
) -> int | None:
    if _cache_sequence_length(past_key_values, layer_index) == 0:
        return None
    if not isinstance(past_key_values, DynamicCache):
        raise RuntimeError(
            "far-pass chord residual currently admits only DynamicCache"
        )
    layer = past_key_values.layers[layer_index]
    keys = getattr(layer, "keys", None)
    if keys is None:
        raise RuntimeError("DynamicCache layer has no cached keys")
    return int(keys.shape[-1])


class FarPassChordAttention(nn.Module):
    """Wrap one OLMo-2 attention layer without changing its Native path."""

    def __init__(
        self,
        native_attention: nn.Module,
        *,
        method_config: FarPassChordConfig,
    ) -> None:
        super().__init__()
        method_config.validate(native_attention.config)
        self.native_attention = native_attention
        self.method_config = method_config
        self.config = native_attention.config
        self.layer_idx = int(native_attention.layer_idx)
        self.head_dim = int(native_attention.head_dim)
        self.num_heads = int(self.config.num_attention_heads)
        self.num_key_value_heads = int(self.config.num_key_value_heads)
        if self.num_heads != self.num_key_value_heads:
            raise RuntimeError(
                "the registered OLMo-2 residual requires equal Q and KV heads"
            )
        self.num_key_value_groups = int(
            native_attention.num_key_value_groups
        )
        if self.num_key_value_groups != 1:
            raise RuntimeError("the registered residual does not admit GQA")
        self.scaling = float(native_attention.scaling)
        self.attention_dropout = float(native_attention.attention_dropout)
        self.is_causal = bool(native_attention.is_causal)
        residual_width = self.num_heads * int(method_config.residual_dim)
        self.residual_q = LowRankResidualProjection(
            input_dim=int(self.config.hidden_size),
            output_dim=residual_width,
            rank=int(method_config.projection_rank),
        )
        self.residual_k = LowRankResidualProjection(
            input_dim=int(self.config.hidden_size),
            output_dim=residual_width,
            rank=int(method_config.projection_rank),
        )
        self.raw_logit_gain = nn.Parameter(
            torch.tensor(
                _inverse_softplus(method_config.initial_logit_gain),
                dtype=torch.float32,
            )
        )
        self.residual_v: LowRankResidualProjection | None = None
        self.residual_o: LowRankResidualProjection | None = None
        self.raw_content_gain: nn.Parameter | None = None
        content_dim = int(method_config.content_value_dim)
        if content_dim:
            content_width = self.num_heads * content_dim
            self.residual_v = LowRankResidualProjection(
                input_dim=int(self.config.hidden_size),
                output_dim=content_width,
                rank=int(method_config.content_projection_rank),
            )
            self.residual_o = LowRankResidualProjection(
                input_dim=content_width,
                output_dim=int(self.config.hidden_size),
                rank=int(method_config.content_projection_rank),
            )
            self.raw_content_gain = nn.Parameter(
                torch.tensor(
                    _inverse_softplus(method_config.initial_content_gain),
                    dtype=torch.float32,
                )
            )
        inv_freq = far_pass_inv_freq(
            pairs=int(method_config.residual_pairs),
            wavelength_min=float(method_config.wavelength_min),
            wavelength_max=float(method_config.wavelength_max),
            dtype=torch.float32,
        )
        self.register_buffer("chord_inv_freq", inv_freq, persistent=True)
        self.route_enabled = False

    @property
    def logit_gain(self) -> torch.Tensor:
        return F.softplus(self.raw_logit_gain)

    @property
    def content_gain(self) -> torch.Tensor:
        if self.raw_content_gain is None:
            raise RuntimeError("content transport is disabled")
        return F.softplus(self.raw_content_gain)

    def set_route(self, enabled: bool) -> None:
        self.route_enabled = bool(enabled)

    def _residual_cos_sin(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise RuntimeError("position_ids must have shape [batch, sequence]")
        inv = self.chord_inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        inv = inv.to(hidden_states.device)
        positions = position_ids[:, None, :].float()
        device_type = (
            hidden_states.device.type
            if hidden_states.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            frequencies = (inv @ positions).transpose(1, 2)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            return embedding.cos(), embedding.sin()

    def _active_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        cache_position: torch.LongTensor | None,
        position_ids: torch.LongTensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cached_width = _cache_head_width(
            past_key_values, self.layer_idx
        )
        expected_cache_width = self.head_dim + 2 * int(
            self.method_config.residual_dim
        )
        if (
            cached_width is not None
            and cached_width != expected_cache_width
        ):
            raise RuntimeError(
                "cannot enable chord residual on a Native-width cache"
            )
        input_shape = hidden_states.shape[:-1]
        native_shape = (*input_shape, -1, self.head_dim)

        query_native = self.native_attention.q_norm(
            self.native_attention.q_proj(hidden_states)
        )
        key_native = self.native_attention.k_norm(
            self.native_attention.k_proj(hidden_states)
        )
        value_native = self.native_attention.v_proj(hidden_states)
        query_native = query_native.view(native_shape).transpose(1, 2)
        key_native = key_native.view(native_shape).transpose(1, 2)
        value_native = value_native.view(native_shape).transpose(1, 2)
        native_cos, native_sin = position_embeddings
        query_native, key_native = apply_rotary_pos_emb(
            query_native,
            key_native,
            native_cos,
            native_sin,
        )

        residual_dim = int(self.method_config.residual_dim)
        residual_shape = (*input_shape, -1, residual_dim)
        query_residual = self.residual_q(hidden_states).view(
            residual_shape
        ).transpose(1, 2)
        key_residual = self.residual_k(hidden_states).view(
            residual_shape
        ).transpose(1, 2)
        query_residual = F.rms_norm(
            query_residual.float(),
            (residual_dim,),
            eps=float(self.method_config.rms_norm_eps),
        ).to(query_native.dtype)
        key_residual = F.rms_norm(
            key_residual.float(),
            (residual_dim,),
            eps=float(self.method_config.rms_norm_eps),
        ).to(key_native.dtype)
        residual_cos, residual_sin = self._residual_cos_sin(
            hidden_states, position_ids
        )
        query_rotated, key_rotated = apply_rotary_pos_emb(
            query_residual,
            key_residual,
            residual_cos,
            residual_sin,
        )
        query_gate = (
            position_ids >= int(self.method_config.threshold_position)
        ).to(dtype=query_residual.dtype)
        query_gate = query_gate[:, None, :, None]
        gain_root = self.logit_gain.sqrt().to(
            device=query_residual.device,
            dtype=query_residual.dtype,
        )
        query_unrotated = query_residual * query_gate * gain_root
        query_rotated = -query_rotated * query_gate * gain_root
        key_unrotated = key_residual * gain_root
        key_rotated = key_rotated * gain_root

        # The two added dot products equal
        #   u_q^T v_t - (R(p_q)u_q)^T(R(p_t)v_t)
        # = u_q^T[I-R(omega*(p_t-p_q))]v_t.
        query_states = torch.cat(
            (query_native, query_unrotated, query_rotated), dim=-1
        )
        key_states = torch.cat(
            (key_native, key_unrotated, key_rotated), dim=-1
        )
        content_dim = int(self.method_config.content_value_dim)
        if content_dim:
            if self.residual_v is None:
                raise RuntimeError("content value projection is missing")
            content_shape = (*input_shape, -1, content_dim)
            value_residual = self.residual_v(hidden_states).view(
                content_shape
            ).transpose(1, 2)
            value_states = torch.cat(
                (value_native, value_residual.to(value_native.dtype)),
                dim=-1,
            )
        else:
            value_states = F.pad(value_native, (0, 2 * residual_dim))

        if past_key_values is not None:
            cache_kwargs = {
                "sin": native_sin,
                "cos": native_cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attention_interface = eager_attention_forward
        implementation = str(self.config._attn_implementation)
        if implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[implementation]
        attention_output, attention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=(
                0.0 if not self.training else self.attention_dropout
            ),
            scaling=self.scaling,
            **kwargs,
        )
        if int(attention_output.shape[-1]) != expected_cache_width:
            raise RuntimeError("augmented attention output width drift")
        native_output = attention_output[..., : self.head_dim]
        native_output = native_output.reshape(
            *input_shape, -1
        ).contiguous()
        native_output = self.native_attention.o_proj(native_output)
        if content_dim:
            if self.residual_o is None:
                raise RuntimeError("content output projection is missing")
            residual_output = attention_output[..., self.head_dim :]
            residual_output = residual_output.reshape(
                *input_shape, -1
            ).contiguous()
            residual_output = self.residual_o(residual_output)
            output_gate = query_gate.squeeze(1)
            content_gain = self.content_gain.to(
                device=residual_output.device,
                dtype=residual_output.dtype,
            )
            native_output = native_output + (
                residual_output * output_gate * content_gain
            )
        return native_output, attention_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.route_enabled:
            cached_width = _cache_head_width(
                past_key_values, self.layer_idx
            )
            if cached_width is not None and cached_width != self.head_dim:
                raise RuntimeError(
                    "cannot disable chord residual on an augmented cache"
                )
            return self.native_attention(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        if position_ids is None:
            raise RuntimeError("active chord residual requires position_ids")
        return self._active_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
            **kwargs,
        )


def _target_layers(model: nn.Module) -> list[tuple[int, nn.Module]]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("model does not expose model.layers")
    return [(index, layer) for index, layer in enumerate(layers)]


def install_far_pass_chord_residual(
    model: nn.Module,
    method_config: FarPassChordConfig,
    *,
    strict_model_contract: bool = True,
) -> dict[str, Any]:
    """Freeze the model and replace every self-attention with the wrapper."""
    method_config.validate(model.config)
    if strict_model_contract:
        actual = {
            "hidden_size": int(model.config.hidden_size),
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "num_attention_heads": int(model.config.num_attention_heads),
            "num_key_value_heads": int(model.config.num_key_value_heads),
            "head_dim": int(
                getattr(
                    model.config,
                    "head_dim",
                    int(model.config.hidden_size)
                    // int(model.config.num_attention_heads),
                )
            ),
            "rope_theta": _rope_theta(model.config),
        }
        expected = {
            name: MODEL_CONTRACT[name]
            for name in actual
        }
        if actual != expected:
            raise RuntimeError(
                f"OLMo-2 residual model drift: {actual} != {expected}"
            )
        native = model.model.rotary_emb.inv_freq.detach().cpu().float()
        if not torch.equal(native, endpoint_geo_inv_freq()):
            raise RuntimeError("global Native RoPE buffer was modified")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(method_config.initialization_seed))
        for index, layer in _target_layers(model):
            if isinstance(layer.self_attn, FarPassChordAttention):
                raise RuntimeError("far-pass chord residual is already installed")
            if int(layer.self_attn.layer_idx) != index:
                raise RuntimeError("OLMo-2 attention layer index drift")
            layer.self_attn = FarPassChordAttention(
                layer.self_attn,
                method_config=method_config,
            )
    validation = validate_far_only_installation(
        model,
        strict_model_contract=strict_model_contract,
    )
    return {
        "method_id": METHOD_ID,
        "adaptation": ADAPTATION_NAME,
        "config": asdict(method_config),
        **validation,
    }


def far_only_trainable_named_parameters(
    model: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    allowed_suffixes = (
        ".residual_q.a",
        ".residual_q.b",
        ".residual_k.a",
        ".residual_k.b",
        ".raw_logit_gain",
        ".residual_v.a",
        ".residual_v.b",
        ".residual_o.a",
        ".residual_o.b",
        ".raw_content_gain",
    )
    values = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if any(
        not name.endswith(allowed_suffixes)
        for name, _ in values
    ):
        raise RuntimeError("trainable scope escaped far-pass chord residual")
    return values


def validate_far_only_installation(
    model: nn.Module,
    *,
    strict_model_contract: bool = True,
) -> dict[str, Any]:
    wrappers = [
        module
        for module in model.modules()
        if isinstance(module, FarPassChordAttention)
    ]
    expected_layers = int(model.config.num_hidden_layers)
    if len(wrappers) != expected_layers:
        raise RuntimeError(
            f"expected {expected_layers} chord residual layers, got "
            f"{len(wrappers)}"
        )
    configs = {wrapper.method_config for wrapper in wrappers}
    if len(configs) != 1:
        raise RuntimeError("far-pass chord configuration differs across layers")
    method_config = configs.pop()
    expected_chord = far_pass_inv_freq(
        pairs=int(method_config.residual_pairs),
        wavelength_min=float(method_config.wavelength_min),
        wavelength_max=float(method_config.wavelength_max),
        dtype=torch.float32,
    )
    for wrapper in wrappers:
        observed = wrapper.chord_inv_freq.detach().cpu().float()
        if not torch.equal(observed, expected_chord):
            raise RuntimeError(
                f"chord residual frequency drift at layer {wrapper.layer_idx}"
            )
    named = far_only_trainable_named_parameters(model)
    content_enabled = int(method_config.content_value_dim) > 0
    expected_tensors = (10 if content_enabled else 5) * expected_layers
    if len(named) != expected_tensors:
        raise RuntimeError(
            f"expected {expected_tensors} residual tensors, got {len(named)}"
        )
    global_native_hash = None
    trainable_parameters = int(
        sum(parameter.numel() for _, parameter in named)
    )
    if strict_model_contract:
        native = model.model.rotary_emb.inv_freq.detach().cpu().float()
        expected_native = endpoint_geo_inv_freq()
        if not torch.equal(native, expected_native):
            raise RuntimeError("global Native RoPE buffer was modified")
        global_native_hash = tensor_sha256(native)
        registered = far_pass_frequency_receipt()
        if tensor_sha256(expected_chord) != registered[
            "inv_freq_sha256_float32"
        ]:
            raise RuntimeError("residual chord tensor is not registered")
        expected_parameters = 9_961_504 if content_enabled else 4_718_608
        if trainable_parameters != expected_parameters:
            raise RuntimeError(
                "registered far-pass trainable-parameter count drift: "
                f"{trainable_parameters}"
            )
    return {
        "trainable_parameter_tensors": len(named),
        "trainable_parameters": trainable_parameters,
        "parameter_names": [name for name, _ in named],
        "global_rope": "native_untouched",
        "global_native_inv_freq_sha256_float32": global_native_hash,
        "residual_operator": "far_pass_I_minus_R_phase_chord",
        "content_transport": (
            "learned_augmented_value_and_long_query_output"
            if content_enabled
            else "zero_padded_values"
        ),
        "residual_frequency_receipt": far_pass_frequency_receipt(),
        "residual_inv_freq_sha256_float32": tensor_sha256(expected_chord),
    }


def set_far_pass_chord_route(model: nn.Module, enabled: bool) -> None:
    wrappers = [
        module
        for module in model.modules()
        if isinstance(module, FarPassChordAttention)
    ]
    expected = int(model.config.num_hidden_layers)
    if len(wrappers) != expected:
        raise RuntimeError(
            f"expected {expected} chord residual layers, got {len(wrappers)}"
        )
    for wrapper in wrappers:
        wrapper.set_route(bool(enabled))


def route_for_budget(model: nn.Module, total_budget: int) -> bool:
    wrappers = [
        module
        for module in model.modules()
        if isinstance(module, FarPassChordAttention)
    ]
    if not wrappers:
        raise RuntimeError("far-pass chord residual is not installed")
    thresholds = {
        int(wrapper.method_config.threshold_position)
        for wrapper in wrappers
    }
    if len(thresholds) != 1:
        raise RuntimeError("residual threshold differs across layers")
    enabled = int(total_budget) > thresholds.pop()
    set_far_pass_chord_route(model, enabled)
    return enabled


def _tensor_bundle_sha256(
    values: list[tuple[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(values):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\0")
        digest.update(
            value.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def load_frozen_parent_adapter(
    path: Path,
    model: nn.Module,
) -> dict[str, Any]:
    """Load a standard repository LoRA before installing the residual."""

    from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
        trainable_named_parameters,
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state")
    if not isinstance(state, dict):
        raise RuntimeError("parent adapter has no tensor state")
    expected = dict(trainable_named_parameters(model, None))
    if set(state) != set(expected):
        raise RuntimeError(
            "parent adapter parameter mismatch: "
            f"missing={sorted(set(expected)-set(state))}, "
            f"extra={sorted(set(state)-set(expected))}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            source = state[name]
            if tuple(source.shape) != tuple(parameter.shape):
                raise RuntimeError(f"parent adapter shape drift for {name}")
            parameter.copy_(
                source.to(device=parameter.device, dtype=parameter.dtype)
            )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError("parent adapter metadata is malformed")
    return dict(metadata)


def save_far_only_adapter(
    path: Path,
    model: nn.Module,
    *,
    metadata: dict[str, Any],
) -> str:
    named = far_only_trainable_named_parameters(model)
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in named
    }
    payload = {
        "format_version": ADAPTER_FORMAT_VERSION,
        "method_id": METHOD_ID,
        "state": state,
        "state_sha256": _tensor_bundle_sha256(list(state.items())),
        "metadata": metadata,
    }
    temporary = path.with_name(path.name + ".incomplete")
    torch.save(payload, temporary)
    temporary.replace(path)
    return _sha256_file(path)


def peek_far_only_adapter(path: Path) -> tuple[FarPassChordConfig, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        int(payload.get("format_version", -1)) != ADAPTER_FORMAT_VERSION
        or payload.get("method_id") != METHOD_ID
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise RuntimeError("far-pass chord adapter identity drift")
    metadata = payload["metadata"]
    config = FarPassChordConfig(**metadata["method_config"])
    config.validate()
    return config, metadata


def load_far_only_adapter(
    path: Path,
    model: nn.Module,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config, metadata = peek_far_only_adapter(path)
    wrappers = [
        module
        for module in model.modules()
        if isinstance(module, FarPassChordAttention)
    ]
    if len(wrappers) != int(model.config.num_hidden_layers):
        raise RuntimeError("install residual wrappers before loading adapter")
    if any(wrapper.method_config != config for wrapper in wrappers):
        raise RuntimeError("installed residual configuration drift")
    if (
        expected_checkpoint_sha256 is not None
        and metadata.get("base_checkpoint_sha256")
        != expected_checkpoint_sha256
    ):
        raise RuntimeError("far-pass chord base checkpoint drift")
    state = payload["state"]
    if payload.get("state_sha256") != _tensor_bundle_sha256(
        list(state.items())
    ):
        raise RuntimeError("far-pass chord adapter tensor digest drift")
    expected = dict(far_only_trainable_named_parameters(model))
    if set(state) != set(expected):
        raise RuntimeError(
            "far-pass chord adapter parameter names do not match installation"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            source = state[name]
            if tuple(source.shape) != tuple(parameter.shape):
                raise RuntimeError(f"adapter shape drift for {name}")
            parameter.copy_(
                source.to(device=parameter.device, dtype=parameter.dtype)
            )
    validate_far_only_installation(
        model,
        strict_model_contract=(
            int(model.config.hidden_size)
            == int(MODEL_CONTRACT["hidden_size"])
            and int(model.config.num_hidden_layers)
            == int(MODEL_CONTRACT["num_hidden_layers"])
        ),
    )
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
