"""Minimal AdaRoPE-style OLMo-2 retrofit primitives.

The implementation keeps the ordinary OLMo-2 D128 attention path: Q/K are
rotated before one standard attention-interface call, and DynamicCache stores
the usual D128 keys and values.  Frequency interpolation is endpoint-preserving
and monotone by construction; AdaScale multiplies only the post-RoPE query.
There is deliberately no trainer, data loader, second KV path, or routing.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.olmo2.modeling_olmo2 import eager_attention_forward


METHOD_ID = "olmo2_phase_adarope_headwise_v1"
STATE_VERSION = 2
LAYERS = 16
HEADS = 16
HEAD_DIM = 128
PAIRS = 64
MAX_CONTEXT_BUDGET = 16_384
TARGET_NAMES = {"phase_chord", "matched_exponential", "moment_matched_control"}
MODES = {"control", "scale_only", "freq_only", "joint"}
BETA_RAW_BOUND = 8.0
GAMMA_RAW_BOUND = 8.0


def _validate_table(value: torch.Tensor, *, pairs: int, name: str) -> torch.Tensor:
    table = torch.as_tensor(value, dtype=torch.float32)
    if table.ndim != 1 or table.shape[0] != pairs:
        raise ValueError(f"{name} must have shape [{pairs}]")
    if not torch.isfinite(table).all() or not (table > 0).all():
        raise ValueError(f"{name} must be finite and positive")
    if not torch.all(table[:-1] > table[1:]):
        raise ValueError(f"{name} must be strictly decreasing")
    return table.contiguous()


def _hash_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _positions(
    hidden_states: torch.Tensor,
    *,
    position_ids: torch.Tensor | None,
    cache_position: torch.Tensor | None,
    past_key_values: Cache | None,
    layer_idx: int,
) -> torch.Tensor:
    batch, length = hidden_states.shape[:2]
    if position_ids is not None:
        value = position_ids.to(hidden_states.device)
    elif cache_position is not None:
        value = cache_position.to(hidden_states.device)
    else:
        try:
            start = int(past_key_values.get_seq_length(layer_idx)) if past_key_values is not None else 0
        except TypeError:
            start = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        return torch.arange(start, start + length, device=hidden_states.device)
    if value.ndim == 1 and value.numel() == length:
        return value
    if value.ndim == 2 and value.shape[1] == length and value.shape[0] in {1, batch}:
        return value.expand(batch, length)
    raise ValueError("positions must have shape [T] or [B,T]")


def _validate_phase_context_budget(value: Any, *, total_length: int) -> int:
    if value is None:
        raise ValueError("phase_context_budget is required for PhaseAdaRoPEAttention")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("phase_context_budget must be a scalar integer")
        value = value.detach().item()
    if isinstance(value, bool):
        raise ValueError("phase_context_budget must be a scalar integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("phase_context_budget must be a scalar integer") from error
    if not math.isfinite(numeric) or numeric != math.trunc(numeric):
        raise ValueError("phase_context_budget must be a scalar integer")
    budget = int(numeric)
    if budget < total_length:
        raise ValueError("phase_context_budget must be >= past_length + current_length")
    if budget > MAX_CONTEXT_BUDGET:
        raise ValueError(f"phase_context_budget exceeds registered maximum {MAX_CONTEXT_BUDGET}")
    return budget


def _bind_phase_context_budget(past_key_values: Cache | None, budget: int) -> None:
    if past_key_values is None:
        return
    existing = getattr(past_key_values, "_phase_context_budget", None)
    if existing is None:
        setattr(past_key_values, "_phase_context_budget", int(budget))
        return
    if isinstance(existing, bool) or not isinstance(existing, int) or existing != budget:
        raise ValueError("phase_context_budget cannot change within a cache")


class PhaseAdaRoPE(nn.Module):
    """16x16x64 frequency/temperature parameter bank."""

    def __init__(
        self,
        native_inv_freq: torch.Tensor | Sequence[float],
        target_inv_freq: torch.Tensor | Sequence[float],
        *,
        target_name: str,
        mode: str = "joint",
        num_layers: int = LAYERS,
        num_heads: int = HEADS,
        head_dim: int = HEAD_DIM,
    ) -> None:
        super().__init__()
        if target_name not in TARGET_NAMES:
            raise ValueError(f"target_name must be one of {sorted(TARGET_NAMES)}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        if num_layers != LAYERS or num_heads != HEADS or head_dim != HEAD_DIM:
            raise ValueError("registered AdaRoPE bank requires 16 layers, 16 heads, D128")
        pairs = head_dim // 2
        native = _validate_table(torch.as_tensor(native_inv_freq), pairs=pairs, name="native_inv_freq")
        target = _validate_table(torch.as_tensor(target_inv_freq), pairs=pairs, name="target_inv_freq")
        if not torch.equal(native[[0, -1]], target[[0, -1]]):
            raise ValueError("target table must preserve Native endpoints")
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.pair_count = pairs
        self.target_name = target_name
        self.mode = mode
        self.register_buffer("native_inv_freq", native)
        self.register_buffer("target_inv_freq", target)
        # One scalar per layer/head selects the whole registered target
        # direction. Pairwise freedom would no longer be this intervention.
        self.alpha = nn.Parameter(torch.zeros(num_layers, num_heads), requires_grad=mode in {"freq_only", "joint"})
        self.raw_beta = nn.Parameter(torch.zeros(num_layers, num_heads), requires_grad=mode in {"scale_only", "joint"})
        self.raw_gamma = nn.Parameter(torch.zeros(num_layers, num_heads), requires_grad=mode in {"scale_only", "joint"})

    def project_parameters(self) -> None:
        """Project frequency interpolation and bounded temperature parameters in-place."""
        with torch.no_grad():
            self.alpha.clamp_(0.0, 1.0)
            self.raw_beta.clamp_(-BETA_RAW_BOUND, BETA_RAW_BOUND)
            self.raw_gamma.clamp_(-GAMMA_RAW_BOUND, GAMMA_RAW_BOUND)

    def alpha_projected(self) -> torch.Tensor:
        return self.alpha.clamp(0.0, 1.0)

    def frequency_table(self, layer_idx: int) -> torch.Tensor:
        if not 0 <= int(layer_idx) < self.num_layers:
            raise IndexError("layer_idx is out of range")
        alpha = self.alpha_projected()[int(layer_idx)].view(-1, 1)
        native_log = self.native_inv_freq.log().view(1, -1)
        target_log = self.target_inv_freq.log().view(1, -1)
        interpolated = torch.exp((1.0 - alpha) * native_log + alpha * target_log)
        exact_native = self.native_inv_freq.view(1, -1).expand_as(interpolated)
        interpolated = torch.cat((exact_native[:, :1], interpolated[:, 1:-1], exact_native[:, -1:]), dim=-1)
        # Exact Native forward value at zero, with a straight-through target
        # direction so the first optimizer step has a nonzero frequency grad.
        native_forward_with_grad = exact_native + (interpolated - interpolated.detach())
        target_exact = self.target_inv_freq.view(1, -1).expand_as(interpolated)
        target_forward_with_grad = target_exact + (interpolated - interpolated.detach())
        return torch.where(
            alpha == 0.0,
            native_forward_with_grad,
            torch.where(alpha == 1.0, target_forward_with_grad, interpolated),
        )

    def temperature(self, layer_idx: int, length: int, *, l_ref: int) -> torch.Tensor:
        if length <= 0 or l_ref <= 0:
            raise ValueError("length and l_ref must be positive")
        beta = self.raw_beta[int(layer_idx)]
        if length <= l_ref:
            # Keep the whole in-window regime exactly Native, including after
            # training.  ``ones_like`` preserves dtype/device and is bitwise
            # equal to the identity scale.
            return torch.ones_like(beta)
        gamma = self.raw_gamma[int(layer_idx)]
        x = torch.log(torch.tensor(float(length) / float(l_ref), device=beta.device, dtype=beta.dtype))
        exponent = 1.0 + torch.nn.functional.softplus(gamma)
        log_scale = beta * x.pow(exponent)
        log_scale = log_scale.clamp(min=-torch.log(torch.tensor(4.0, device=beta.device, dtype=beta.dtype)), max=torch.log(torch.tensor(4.0, device=beta.device, dtype=beta.dtype)))
        return log_scale.exp()

    def cos_sin(self, layer_idx: int, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        table = self.frequency_table(layer_idx)
        if positions.ndim == 1:
            angles = torch.einsum("t,hp->htp", positions.float(), table)
        elif positions.ndim == 2:
            angles = torch.einsum("bt,hp->bhtp", positions.float(), table)
        else:
            raise ValueError("positions must have shape [T] or [B,T]")
        return angles.cos(), angles.sin()

    def parameter_groups(self, *, frequency_lr: float, temperature_lr: float) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        if self.alpha.requires_grad:
            groups.append({"params": [self.alpha], "lr": float(frequency_lr), "weight_decay": 0.0, "name": "adarope_frequency"})
        temperature = [parameter for parameter in (self.raw_beta, self.raw_gamma) if parameter.requires_grad]
        if temperature:
            groups.append({"params": temperature, "lr": float(temperature_lr), "weight_decay": 0.0, "name": "adarope_temperature"})
        return groups


class PhaseAdaRoPEAttention(nn.Module):
    """One OLMo-2 attention layer using a shared 16x16x64 bank."""

    def __init__(self, native_attention: nn.Module, bank: PhaseAdaRoPE, *, l_ref: int = 4096, strict: bool = True) -> None:
        super().__init__()
        # Re-register the original projection modules at the wrapper's own
        # names. Keeping ``native_attention`` as a child would silently change
        # PEFT/state-dict paths to ``self_attn.native_attention.q_proj``.
        self.q_proj = native_attention.q_proj
        self.k_proj = native_attention.k_proj
        self.v_proj = native_attention.v_proj
        self.o_proj = native_attention.o_proj
        self.q_norm = native_attention.q_norm
        self.k_norm = native_attention.k_norm
        self.bank = bank
        self.config = native_attention.config
        self.layer_idx = int(native_attention.layer_idx)
        self.head_dim = int(native_attention.head_dim)
        self.num_heads = int(self.config.num_attention_heads)
        self.num_key_value_heads = int(self.config.num_key_value_heads)
        if strict and (self.head_dim, self.num_heads, self.num_key_value_heads) != (128, 16, 16):
            raise ValueError("strict AdaRoPE attention requires OLMo-2 MHA D128")
        if self.num_heads != self.num_key_value_heads:
            raise ValueError("PhaseAdaRoPEAttention currently requires MHA")
        self.num_key_value_groups = int(native_attention.num_key_value_groups)
        self.scaling = float(native_attention.scaling)
        self.attention_dropout = float(native_attention.attention_dropout)
        self.is_causal = bool(getattr(native_attention, "is_causal", True))
        self.l_ref = int(l_ref)

    def _attention_interface(self):
        implementation = str(getattr(self.config, "_attn_implementation", "eager"))
        if implementation == "eager":
            return eager_attention_forward
        getter = getattr(ALL_ATTENTION_FUNCTIONS, "get_interface", None)
        if getter is not None:
            try:
                return getter(implementation, eager_attention_forward)
            except TypeError:
                return getter(implementation)
        return ALL_ATTENTION_FUNCTIONS[implementation]

    @staticmethod
    def _rotate(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        original_dtype = value.dtype
        pairs = value.shape[-1] // 2
        left, right = value[..., :pairs], value[..., pairs:]
        if cos.ndim == 3:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        rotated = torch.cat((left * cos - right * sin, left * sin + right * cos), dim=-1)
        return rotated.to(dtype=original_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        phase_context_budget: int | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B,T,H]")
        input_shape = hidden_states.shape[:-1]
        shape = (*input_shape, -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states)).view(shape).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states)).view(shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        try:
            past_length = int(past_key_values.get_seq_length(self.layer_idx)) if past_key_values is not None else 0
        except TypeError:
            past_length = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        total_length = past_length + int(hidden_states.shape[1])
        context_budget = _validate_phase_context_budget(phase_context_budget, total_length=total_length)
        _bind_phase_context_budget(past_key_values, context_budget)
        positions = _positions(hidden_states, position_ids=position_ids, cache_position=cache_position, past_key_values=past_key_values, layer_idx=self.layer_idx)
        cos, sin = self.bank.cos_sin(self.layer_idx, positions)
        query = self._rotate(query, cos, sin)
        key = self._rotate(key, cos, sin)
        query = query * self.bank.temperature(self.layer_idx, context_budget, l_ref=self.l_ref).to(query.dtype).view(1, self.num_heads, 1, 1)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
        attention_output, weights = self._attention_interface()(self, query, key, value, attention_mask, dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs)
        output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(output), weights


def install_phase_adarope(model: nn.Module, *, target_inv_freq: torch.Tensor | Sequence[float], target_name: str, mode: str = "joint", l_ref: int = 4096, strict: bool = True) -> tuple[PhaseAdaRoPE, dict[str, Any]]:
    layers = list(getattr(getattr(model, "model", None), "layers", []))
    if len(layers) != LAYERS:
        raise ValueError("registered AdaRoPE installation requires 16 layers")
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    native = getattr(rotary, "inv_freq", None)
    if native is None:
        raise ValueError("model has no Native rotary frequency table")
    bank = PhaseAdaRoPE(native.detach().float(), target_inv_freq, target_name=target_name, mode=mode)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for index, layer in enumerate(layers):
        if int(layer.self_attn.layer_idx) != index:
            raise ValueError("attention layer index drift")
        layer.self_attn = PhaseAdaRoPEAttention(layer.self_attn, bank, l_ref=l_ref, strict=strict)
    receipt = phase_adarope_receipt(bank)
    return bank, receipt


def phase_adarope_receipt(bank: PhaseAdaRoPE) -> dict[str, Any]:
    return {
        "method_id": METHOD_ID,
        "state_version": STATE_VERSION,
        "target_name": bank.target_name,
        "mode": bank.mode,
        "layers": bank.num_layers,
        "heads": bank.num_heads,
        "head_dim": bank.head_dim,
        "pairs": bank.pair_count,
        "native_inv_freq_sha256": _hash_tensor(bank.native_inv_freq),
        "target_inv_freq_sha256": _hash_tensor(bank.target_inv_freq),
        "alpha_shape": list(bank.alpha.shape),
        "temperature_shape": list(bank.raw_beta.shape),
        "max_context_budget": MAX_CONTEXT_BUDGET,
        "cache_head_dim": bank.head_dim,
        "extra_kv_width": 0,
    }


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(_hash_tensor(state[name]).encode("ascii"))
    return digest.hexdigest()


def validate_phase_adarope_state(bank: PhaseAdaRoPE) -> None:
    """Reject malformed sidecars before they can alter a model."""
    if not torch.isfinite(bank.alpha).all() or not bool(((bank.alpha >= 0.0) & (bank.alpha <= 1.0)).all()):
        raise ValueError("AdaRoPE alpha is non-finite or outside [0,1]")
    if not torch.isfinite(bank.raw_beta).all() or not torch.isfinite(bank.raw_gamma).all():
        raise ValueError("AdaRoPE temperature state is non-finite")
    if not bool((bank.raw_beta.abs() <= BETA_RAW_BOUND).all()):
        raise ValueError("AdaRoPE beta is outside its finite bound")
    if not bool((bank.raw_gamma.abs() <= GAMMA_RAW_BOUND).all()):
        raise ValueError("AdaRoPE gamma is outside its finite bound")
    for layer_idx in range(bank.num_layers):
        table = bank.frequency_table(layer_idx)
        if not torch.isfinite(table).all() or not bool(torch.all(table[..., :-1] > table[..., 1:])):
            raise ValueError(f"AdaRoPE realized table is not strictly decreasing at layer {layer_idx}")
        if not torch.equal(table[..., 0], bank.native_inv_freq[0].expand(bank.num_heads)):
            raise ValueError(f"AdaRoPE fast endpoint drift at layer {layer_idx}")
        if not torch.equal(table[..., -1], bank.native_inv_freq[-1].expand(bank.num_heads)):
            raise ValueError(f"AdaRoPE slow endpoint drift at layer {layer_idx}")


def save_phase_adarope_state(path: str | Path, bank: PhaseAdaRoPE, *, metadata: dict[str, Any] | None = None) -> None:
    validate_phase_adarope_state(bank)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in bank.state_dict().items() if name in {"alpha", "raw_beta", "raw_gamma"}}
    payload = {"method_id": METHOD_ID, "state_version": STATE_VERSION, "metadata": {**phase_adarope_receipt(bank), **(metadata or {})}, "state": state, "state_sha256": _state_hash(state)}
    temporary = destination.with_name(destination.name + ".incomplete")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _read_phase_adarope_payload(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("method_id") != METHOD_ID or int(payload.get("state_version", -1)) != STATE_VERSION:
        raise ValueError("AdaRoPE state identity drift")
    state = payload.get("state")
    if not isinstance(state, dict) or payload.get("state_sha256") != _state_hash(state):
        raise ValueError("AdaRoPE state hash mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("AdaRoPE metadata is malformed")
    required = {"alpha", "raw_beta", "raw_gamma"}
    if set(state) != required:
        raise ValueError("AdaRoPE state keys drift")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"AdaRoPE state is non-finite for {name}")
    if not bool(((state["alpha"] >= 0.0) & (state["alpha"] <= 1.0)).all()):
        raise ValueError("AdaRoPE sidecar alpha is outside [0,1]")
    if not bool((state["raw_beta"].abs() <= BETA_RAW_BOUND).all()):
        raise ValueError("AdaRoPE sidecar beta is outside its finite bound")
    if not bool((state["raw_gamma"].abs() <= GAMMA_RAW_BOUND).all()):
        raise ValueError("AdaRoPE sidecar gamma is outside its finite bound")
    return metadata, state


def load_phase_adarope_components(
    path: str | Path,
    bank: PhaseAdaRoPE,
    components: str | Sequence[str],
) -> dict[str, Any]:
    """Load only ``alpha`` or ``temperature`` from a validated Stage1 sidecar."""
    if isinstance(components, str):
        requested = (components,)
    else:
        requested = tuple(components)
    allowed = {"alpha", "temperature"}
    if not requested or any(component not in allowed for component in requested) or len(set(requested)) != len(requested):
        raise ValueError("components must be unique values from alpha/temperature")
    metadata, state = _read_phase_adarope_payload(path)
    expected = {name: value for name, value in bank.state_dict().items() if name in {"alpha", "raw_beta", "raw_gamma"}}
    for name, value in expected.items():
        if tuple(state[name].shape) != tuple(value.shape):
            raise ValueError(f"AdaRoPE state shape drift for {name}")
    current = phase_adarope_receipt(bank)
    if "alpha" in requested:
        for key in ("target_name", "native_inv_freq_sha256", "target_inv_freq_sha256"):
            if metadata.get(key) != current.get(key):
                raise ValueError(f"AdaRoPE alpha target identity drift for {key}")
    if "temperature" in requested:
        if metadata.get("mode") != "scale_only":
            raise ValueError("temperature components require a scale_only sidecar")
        if not torch.equal(state["alpha"], torch.zeros_like(state["alpha"])):
            raise ValueError("temperature sidecar must have alpha=0")
        # Temperature is target-independent, but it is still architecture and
        # Native-coordinate specific; never import it across model identities.
        for key in ("method_id", "layers", "heads", "head_dim", "pairs", "native_inv_freq_sha256"):
            if metadata.get(key) != current.get(key):
                raise ValueError(f"AdaRoPE temperature model identity drift for {key}")
    with torch.no_grad():
        if "alpha" in requested:
            expected["alpha"].copy_(state["alpha"].to(dtype=expected["alpha"].dtype, device=expected["alpha"].device))
        if "temperature" in requested:
            for name in ("raw_beta", "raw_gamma"):
                expected[name].copy_(state[name].to(dtype=expected[name].dtype, device=expected[name].device))
    validate_phase_adarope_state(bank)
    return {**metadata, "loaded_components": list(requested)}


def load_phase_adarope_state(path: str | Path, bank: PhaseAdaRoPE, *, strict: bool = True) -> dict[str, Any]:
    metadata, state = _read_phase_adarope_payload(path)
    expected = {name: value for name, value in bank.state_dict().items() if name in {"alpha", "raw_beta", "raw_gamma"}}
    for name, value in expected.items():
        if tuple(state[name].shape) != tuple(value.shape):
            raise ValueError(f"AdaRoPE state shape drift for {name}")
    with torch.no_grad():
        for name in expected:
            expected[name].copy_(state[name].to(dtype=expected[name].dtype, device=expected[name].device))
    validate_phase_adarope_state(bank)
    if strict:
        current = phase_adarope_receipt(bank)
        for key in ("method_id", "target_name", "layers", "heads", "head_dim", "pairs", "native_inv_freq_sha256", "target_inv_freq_sha256"):
            if metadata.get(key) != current.get(key):
                raise ValueError(f"AdaRoPE metadata drift for {key}")
    return dict(metadata)
