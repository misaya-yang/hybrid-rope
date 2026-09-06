"""Target-free, profile-driven RoPE phase extension.

The operator in this module has no requested target length.  It evaluates the
phase at the positions it is given and therefore does not need a request
budget or mutable length gate.  A model-specific :class:`ModelRoPEProfile`
owns every quantity that is allowed to differ between models.

For a pair ``k`` and position ``p`` the phase is

``theta_k(p) = native_theta_k(p)`` for ``p <= L_native`` and
``theta_k(L_native) + (1 - m_k) * slope_k * (p - L_native)`` otherwise.

The optional query gain is applied by :func:`apply_target_free_qk` to queries
only.  Keys, including keys already stored in a KV cache, are never scaled by
that gain.  This module contains no learned parameters and never changes an
already-computed key when a later position is evaluated.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


PairLayout = Literal["interleaved", "half_split"]
NativePhaseMode = Literal["linear", "external"]


def float32_tensor_sha256(value: torch.Tensor | np.ndarray | Sequence[float]) -> str:
    """Hash a tensor using the repository's canonical little-endian FP32 form."""

    tensor = torch.as_tensor(value).detach().cpu().to(torch.float32).contiguous()
    array = np.asarray(tensor.numpy(), dtype="<f4")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported config value: {type(value).__name__}")


def _finite_vector(value: Any, *, name: str) -> tuple[float, ...]:
    tensor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return tuple(float(item) for item in tensor.tolist())


def _validate_nonnegative_positions(position_ids: torch.Tensor) -> None:
    if not isinstance(position_ids, torch.Tensor):
        raise TypeError("position_ids must be a torch.Tensor")
    if position_ids.ndim == 0 or position_ids.numel() == 0:
        raise ValueError("position_ids must be nonempty")
    if not bool(torch.isfinite(position_ids).all().item()):
        raise ValueError("position_ids must be finite")
    if bool(torch.any(position_ids < 0).item()):
        raise ValueError("position_ids must be nonnegative")


@dataclass(frozen=True)
class ModelRoPEProfile:
    """Immutable model-specific evidence and coefficients for target-free RoPE.

    ``native_phase_sha256`` hashes the canonical native phase probe at integer
    positions ``0..native_context_length`` when no explicit phase probe is
    supplied.  The hash is separate from ``native_inv_freq_sha256`` so a
    caller can bind both the frequency buffer and the phase construction.
    ``native_boundary_slope`` defaults to ``native_inv_freq`` for standard
    linear RoPE, but is recorded explicitly so a model with another verified
    boundary slope does not silently inherit that assumption.
    """

    native_context_length: int
    native_context_length_source: str
    head_dim: int
    rotary_dim: int
    pair_count: int
    native_inv_freq: tuple[float, ...]
    native_boundary_slope: tuple[float, ...]
    movement_coefficients: tuple[float, ...]
    native_inv_freq_sha256: str
    native_phase_sha256: str
    native_phase_hash_scope: str
    native_rope_config: Mapping[str, Any]
    native_scaling_config: Mapping[str, Any]
    gain_coefficient: float
    gain_coefficient_source: str
    pair_layout: PairLayout = "half_split"
    native_phase_mode: NativePhaseMode = "linear"

    def __post_init__(self) -> None:
        if int(self.native_context_length) <= 0:
            raise ValueError("native_context_length must be positive")
        if not str(self.native_context_length_source).strip():
            raise ValueError("native_context_length_source is required")
        if int(self.head_dim) <= 0 or int(self.head_dim) % 2:
            raise ValueError("head_dim must be a positive even integer")
        if int(self.rotary_dim) <= 0 or int(self.rotary_dim) % 2:
            raise ValueError("rotary_dim must be a positive even integer")
        if int(self.rotary_dim) > int(self.head_dim):
            raise ValueError("rotary_dim cannot exceed head_dim")
        expected_pairs = int(self.rotary_dim) // 2
        if int(self.pair_count) != expected_pairs:
            raise ValueError("pair_count must equal rotary_dim // 2")
        vectors = (
            self.native_inv_freq,
            self.native_boundary_slope,
            self.movement_coefficients,
        )
        if any(len(vector) != expected_pairs for vector in vectors):
            raise ValueError("profile vectors must all have pair_count entries")
        if any(not math.isfinite(float(value)) for vector in vectors for value in vector):
            raise ValueError("profile vectors must be finite")
        if any(float(value) <= 0.0 for value in self.native_inv_freq):
            raise ValueError("native inverse frequencies must be positive")
        if any(float(value) <= 0.0 for value in self.native_boundary_slope):
            raise ValueError("native boundary slopes must be positive")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.movement_coefficients):
            raise ValueError("movement coefficients must be in [0, 1]")
        if not math.isfinite(float(self.gain_coefficient)):
            raise ValueError("gain_coefficient must be finite")
        if not str(self.gain_coefficient_source).strip():
            raise ValueError("gain_coefficient_source is required")
        if self.pair_layout not in {"interleaved", "half_split"}:
            raise ValueError(f"unsupported pair layout: {self.pair_layout!r}")
        if self.native_phase_mode not in {"linear", "external"}:
            raise ValueError(f"unsupported native phase mode: {self.native_phase_mode!r}")
        if len(str(self.native_inv_freq_sha256)) != 64:
            raise ValueError("native_inv_freq_sha256 must be a SHA-256 digest")
        if len(str(self.native_phase_sha256)) != 64:
            raise ValueError("native_phase_sha256 must be a SHA-256 digest")
        object.__setattr__(
            self,
            "native_rope_config",
            MappingProxyType(_json_safe(dict(self.native_rope_config))),
        )
        object.__setattr__(
            self,
            "native_scaling_config",
            MappingProxyType(_json_safe(dict(self.native_scaling_config))),
        )

    @classmethod
    def from_native(
        cls,
        native_inv_freq: torch.Tensor | np.ndarray | Sequence[float],
        *,
        native_context_length: int,
        native_context_length_source: str,
        head_dim: int,
        rotary_dim: int | None = None,
        movement_coefficients: torch.Tensor | np.ndarray | Sequence[float],
        native_boundary_slope: (
            torch.Tensor | np.ndarray | Sequence[float] | None
        ) = None,
        native_rope_config: Mapping[str, Any] | None = None,
        native_scaling_config: Mapping[str, Any] | None = None,
        gain_coefficient: float,
        gain_coefficient_source: str,
        pair_layout: PairLayout = "half_split",
        native_phase: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        native_phase_hash_scope: str | None = None,
    ) -> "ModelRoPEProfile":
        """Create a profile from an already-frozen native frequency buffer.

        No model or checkpoint is loaded.  If ``native_phase`` is omitted,
        the standard linear phase probe over every integer position from zero
        through the native boundary is hashed.  A caller with a verified
        nonstandard phase implementation may provide the exact phase probe
        and its scope instead.
        """

        inv = _finite_vector(native_inv_freq, name="native_inv_freq")
        dim = int(head_dim) if rotary_dim is None else int(rotary_dim)
        if dim % 2:
            raise ValueError("rotary_dim must be even")
        pair_count = dim // 2
        if len(inv) != pair_count:
            raise ValueError("native_inv_freq length must equal rotary_dim // 2")
        movement = _finite_vector(
            movement_coefficients,
            name="movement_coefficients",
        )
        if len(movement) != pair_count:
            raise ValueError("movement_coefficients length must equal pair_count")
        slope = inv if native_boundary_slope is None else _finite_vector(
            native_boundary_slope,
            name="native_boundary_slope",
        )
        if len(slope) != pair_count:
            raise ValueError("native_boundary_slope length must equal pair_count")
        context = int(native_context_length)
        if context <= 0:
            raise ValueError("native_context_length must be positive")
        if native_phase is None:
            positions = torch.arange(context + 1, dtype=torch.float64)
            phase = positions[:, None] * torch.as_tensor(inv, dtype=torch.float64)[None, :]
            scope = native_phase_hash_scope or "integer positions 0..L_native inclusive"
        else:
            phase = torch.as_tensor(native_phase)
            scope = native_phase_hash_scope or "caller-supplied native phase probe"
        return cls(
            native_context_length=context,
            native_context_length_source=str(native_context_length_source),
            head_dim=int(head_dim),
            rotary_dim=dim,
            pair_count=pair_count,
            native_inv_freq=inv,
            native_boundary_slope=slope,
            movement_coefficients=movement,
            native_inv_freq_sha256=float32_tensor_sha256(inv),
            native_phase_sha256=float32_tensor_sha256(phase),
            native_phase_hash_scope=scope,
            native_rope_config=dict(native_rope_config or {}),
            native_scaling_config=dict(native_scaling_config or {}),
            gain_coefficient=float(gain_coefficient),
            gain_coefficient_source=str(gain_coefficient_source),
            pair_layout=pair_layout,
            native_phase_mode="linear" if native_phase is None else "external",
        )

    @property
    def effective_boundary_slope(self) -> tuple[float, ...]:
        return tuple(
            (1.0 - movement) * slope
            for movement, slope in zip(
                self.movement_coefficients,
                self.native_boundary_slope,
            )
        )

    @property
    def effective_boundary_slope_sha256(self) -> str:
        return float32_tensor_sha256(self.effective_boundary_slope)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence receipt without model weights."""

        return {
            "native_context_length": int(self.native_context_length),
            "native_context_length_source": self.native_context_length_source,
            "head_dim": int(self.head_dim),
            "rotary_dim": int(self.rotary_dim),
            "pair_count": int(self.pair_count),
            "pair_layout": self.pair_layout,
            "native_phase_mode": self.native_phase_mode,
            "native_inv_freq_sha256": self.native_inv_freq_sha256,
            "native_phase_sha256": self.native_phase_sha256,
            "native_phase_hash_scope": self.native_phase_hash_scope,
            "native_rope_config": dict(self.native_rope_config),
            "native_scaling_config": dict(self.native_scaling_config),
            "gain_coefficient": float(self.gain_coefficient),
            "gain_coefficient_source": self.gain_coefficient_source,
            "movement_coefficients_sha256": float32_tensor_sha256(
                self.movement_coefficients
            ),
            "effective_boundary_slope_sha256": self.effective_boundary_slope_sha256,
            "learned_parameters": 0,
        }


class TargetFreeRoPE(nn.Module):
    """Evaluate target-free phases and apply query-only gain to Q/K tensors."""

    def __init__(
        self,
        profile: ModelRoPEProfile,
        *,
        native_phase_provider: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.profile = profile
        if profile.native_phase_mode == "external" and native_phase_provider is None:
            raise ValueError("external Native phase profile requires a runtime provider")
        self.native_phase_provider = native_phase_provider
        self._current_query_gain: torch.Tensor | None = None
        self.register_buffer(
            "native_inv_freq",
            torch.as_tensor(profile.native_inv_freq, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "native_boundary_slope",
            torch.as_tensor(profile.native_boundary_slope, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "movement_coefficients",
            torch.as_tensor(profile.movement_coefficients, dtype=torch.float32),
            persistent=False,
        )

    def phase(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Return ``[..., pair_count]`` phases; no future/request state is read."""

        _validate_nonnegative_positions(position_ids)
        positions = position_ids.to(dtype=torch.float64)
        inv = self.native_inv_freq.to(
            device=positions.device,
            dtype=torch.float64,
        )
        slope = self.native_boundary_slope.to(
            device=positions.device,
            dtype=torch.float64,
        )
        movement = self.movement_coefficients.to(
            device=positions.device,
            dtype=torch.float64,
        )
        boundary = float(self.profile.native_context_length)
        if self.profile.native_phase_mode == "linear":
            native = positions[..., None] * inv
            boundary_phase = boundary * inv
        else:
            assert self.native_phase_provider is not None
            provider_positions = torch.minimum(
                positions,
                torch.tensor(boundary, device=positions.device, dtype=positions.dtype),
            )
            native = self.native_phase_provider(provider_positions).to(
                device=positions.device,
                dtype=torch.float64,
            )
            expected_shape = (*positions.shape, int(self.profile.pair_count))
            if tuple(native.shape) != expected_shape:
                raise RuntimeError(
                    f"Native phase provider returned {tuple(native.shape)}, "
                    f"expected {expected_shape}"
                )
            boundary_positions = torch.full_like(positions, boundary)
            boundary_values = self.native_phase_provider(boundary_positions).to(
                device=positions.device,
                dtype=torch.float64,
            )
            if tuple(boundary_values.shape) != expected_shape:
                raise RuntimeError("Native boundary phase provider shape drift")
            boundary_phase = boundary_values
        extension = boundary_phase + (1.0 - movement) * slope * (
            positions[..., None] - boundary
        )
        return torch.where(
            (positions <= boundary)[..., None],
            native,
            extension,
        )

    def query_gain(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Return the scalar query gain per position."""

        _validate_nonnegative_positions(position_ids)
        positions = position_ids.to(dtype=torch.float64)
        ratio = (positions + 1.0) / float(self.profile.native_context_length)
        log_ratio = torch.log(torch.clamp(ratio, min=1.0))
        return (1.0 + float(self.profile.gain_coefficient) * log_ratio) ** 2

    def cos_sin(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase = self.phase(position_ids)
        cos_pair = phase.cos()
        sin_pair = phase.sin()
        if self.profile.pair_layout == "interleaved":
            cos = torch.stack((cos_pair, cos_pair), dim=-1).reshape(
                *cos_pair.shape[:-1], self.profile.rotary_dim
            )
            sin = torch.stack((sin_pair, sin_pair), dim=-1).reshape(
                *sin_pair.shape[:-1], self.profile.rotary_dim
            )
        else:
            cos = torch.cat((cos_pair, cos_pair), dim=-1)
            sin = torch.cat((sin_pair, sin_pair), dim=-1)
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin

    def forward(
        self,
        value: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return rotary cos/sin in the common ``(batch, seq, rotary_dim)`` form."""

        self._current_query_gain = self.query_gain(position_ids).detach()
        cos, sin = self.cos_sin(position_ids, dtype=value.dtype)
        return cos.to(device=value.device), sin.to(device=value.device)

    def rotate(
        self,
        value: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        query: bool,
    ) -> torch.Tensor:
        """Rotate ``[..., sequence, head_dim]`` and optionally apply query gain."""

        _validate_nonnegative_positions(position_ids)
        if value.ndim < 2:
            raise ValueError("value must have sequence and head dimensions")
        if int(value.shape[-2]) != int(position_ids.shape[-1]):
            raise ValueError("sequence length and position_ids length differ")
        if value.shape[0] != position_ids.shape[0] and position_ids.ndim > 1:
            raise ValueError("batch dimension and position_ids batch differ")
        if int(value.shape[-1]) < int(self.profile.rotary_dim):
            raise ValueError("head dimension is smaller than rotary_dim")
        cos, sin = self.cos_sin(position_ids, dtype=value.dtype)
        cos = cos.to(device=value.device)
        sin = sin.to(device=value.device)
        extra_axes = value.ndim - position_ids.ndim - 1
        if extra_axes < 0:
            raise ValueError("position_ids must describe batch and sequence axes")
        feature_shape = (
            *position_ids.shape[:1],
            *(1 for _ in range(extra_axes)),
            *position_ids.shape[1:],
            self.profile.rotary_dim,
        )
        cos = cos.reshape(feature_shape)
        sin = sin.reshape(feature_shape)
        rotary = value[..., : self.profile.rotary_dim]
        tail = value[..., self.profile.rotary_dim :]
        if self.profile.pair_layout == "interleaved":
            paired = rotary.reshape(*rotary.shape[:-1], self.profile.pair_count, 2)
            even = paired[..., 0]
            odd = paired[..., 1]
            pair_cos = cos[..., 0::2]
            pair_sin = sin[..., 0::2]
            rotated = torch.stack(
                (even * pair_cos - odd * pair_sin, odd * pair_cos + even * pair_sin),
                dim=-1,
            ).reshape_as(rotary)
        else:
            first = rotary[..., : self.profile.pair_count]
            second = rotary[..., self.profile.pair_count :]
            pair_cos = cos[..., : self.profile.pair_count]
            pair_sin = sin[..., : self.profile.pair_count]
            rotated = torch.cat(
                (
                    first * pair_cos - second * pair_sin,
                    second * pair_cos + first * pair_sin,
                ),
                dim=-1,
            )
        output = torch.cat((rotated, tail), dim=-1)
        if query:
            gain = self.query_gain(position_ids).to(
                device=value.device,
                dtype=value.dtype,
            )
            gain_shape = (
                *position_ids.shape[:1],
                *(1 for _ in range(extra_axes)),
                *position_ids.shape[1:],
                1,
            )
            output = output * gain.reshape(gain_shape)
        return output

    @property
    def current_query_gain(self) -> torch.Tensor:
        if self._current_query_gain is None:
            raise RuntimeError("rotary forward must run before query projection")
        return self._current_query_gain

    def apply_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.rotate(query, position_ids, query=True),
            self.rotate(key, position_ids, query=False),
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "method": "target_free_continuous_boundary_slope_rope",
            "phase_formula": (
                "native_theta(p) for p <= L_native; "
                "native_theta(L_native) + (1-m_k)*native_boundary_slope_k*"
                "(p-L_native) otherwise"
            ),
            "query_gain_formula": "[1+c*log(max(1,(p+1)/L_native))]^2",
            "key_scaling": "none",
            "kv_cache_policy": "position-local; previously computed keys are immutable",
            "requires_target_length": False,
            "requires_request_budget": False,
            **self.profile.as_dict(),
        }


def apply_target_free_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids: torch.Tensor,
    profile: ModelRoPEProfile,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stateless convenience wrapper with query-only gain and unscaled keys."""

    return TargetFreeRoPE(profile).apply_qk(query, key, position_ids)


def install_target_free_olmo2(
    model: Any,
    profile: ModelRoPEProfile,
) -> dict[str, Any]:
    """Install the target-free operator on the pinned full-RoPE OLMo-2 path.

    OLMo computes the shared rotary embedding before entering its decoder
    layers.  The replacement stores the position-local query gain for that
    forward; hooks on every Q norm apply it to the complete query vector before
    the standard rotary call.  Keys and the Flash attention implementation are
    unchanged.
    """

    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "olmo2":
        raise RuntimeError("target-free OLMo installer requires an OLMo-2 model")
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    if (
        int(profile.head_dim) != head_dim
        or int(profile.rotary_dim) != head_dim
        or int(config.num_attention_heads) != int(config.num_key_value_heads)
    ):
        raise RuntimeError("current OLMo installer requires full-RoPE MHA identity")
    original = getattr(getattr(model, "model", None), "rotary_emb", None)
    native_inv = getattr(original, "inv_freq", None)
    if not isinstance(native_inv, torch.Tensor):
        raise RuntimeError("model has no hash-bindable Native rotary frequency buffer")
    if float32_tensor_sha256(native_inv) != profile.native_inv_freq_sha256:
        raise RuntimeError("Native rotary frequency hash drift")
    if float(getattr(original, "attention_scaling", 1.0)) != 1.0:
        raise RuntimeError("current OLMo target-free profile requires Native scaling one")

    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())
    target_free = TargetFreeRoPE(profile)
    model.model.rotary_emb = target_free
    handles = []

    def query_gain_hook(
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        gain = target_free.current_query_gain.to(
            device=output.device,
            dtype=output.dtype,
        )
        if output.shape[:2] != gain.shape:
            raise RuntimeError(
                f"query/gain shape drift: output={tuple(output.shape)} "
                f"gain={tuple(gain.shape)}"
            )
        return output * gain[..., None]

    layers = list(model.model.layers)
    if len(layers) != int(config.num_hidden_layers):
        raise RuntimeError("OLMo decoder layer count drift")
    for layer in layers:
        handles.append(layer.self_attn.q_norm.register_forward_hook(query_gain_hook))
    target_free._query_gain_hook_handles = handles
    parameter_count_after = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count_before != parameter_count_after:
        raise RuntimeError("target-free installation changed model parameter count")
    return {
        **target_free.receipt(),
        "integration": "OLMo-2 shared rotary plus complete-query q_norm hooks",
        "query_gain_hooks": len(handles),
        "flash_attention_unchanged": True,
        "parameter_count_before": parameter_count_before,
        "parameter_count_after": parameter_count_after,
    }


def dumps_profile(profile: ModelRoPEProfile) -> str:
    """Serialize a profile receipt deterministically for freezing."""

    return json.dumps(profile.as_dict(), indent=2, sort_keys=True) + "\n"
