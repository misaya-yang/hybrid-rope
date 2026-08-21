"""R3-prime heterogeneous RoPE protocol and CPU-only model preflight.

The existing GPT/GQA/MLA implementation in
``scripts.core_text_phases.run_gqa_evq_experiment`` shares one
``RotaryEmbedding`` object across all blocks.  A per-layer table therefore
requires cloning that *RoPE module* once per block; the attention implementation
and its SDPA kernel remain untouched.  Per-head tables are deliberately gated
because they require an explicit head axis in the cos/sin contract and have
different semantics for GQA and MLA.

This module is design/preflight code only.  It never calls a training loop and
never moves a model to CUDA.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


SCHEMA_VERSION = 1
SUPPORTED_ATTENTION_TYPES = ("mha", "gqa", "mla")
DEFAULT_BASE = 500_000.0
DEFAULT_TRAIN_LENGTH = 8_192


def _as_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _as_positive_float(value: Any, *, name: str) -> float:
    result = _as_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {result}")
    return result


def _as_nonnegative_float(value: Any, *, name: str) -> float:
    result = _as_float(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {result}")
    return result


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _unwrap_r0(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept common R0 wrappers without prescribing one upstream schema."""

    for key in ("r0", "R0", "allocation", "heterogeneous_rope"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


def _parse_inv_freq(values: Any, *, expected: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a JSON array")
    if len(values) != expected:
        raise ValueError(
            f"{name} must have {expected} values, got {len(values)}"
        )
    parsed = tuple(_as_positive_float(v, name=f"{name}[{i}]") for i, v in enumerate(values))
    # The table is an inverse-frequency table in the existing RoPE contract:
    # highest angular frequency first, slowest last.  Rejecting malformed
    # ordering here is safer than silently training a different operator.
    for i, (left, right) in enumerate(zip(parsed, parsed[1:])):
        if not left > right:
            raise ValueError(
                f"{name} must be strictly decreasing; index {i}: {left} <= {right}"
            )
    return parsed


@dataclass(frozen=True)
class LayerFrequencySpec:
    """One layer's realized frequency specification."""

    layer: int
    tau: float | None
    multiplier_m: float | None
    inv_freq: tuple[float, ...] | None
    source: str


@dataclass(frozen=True)
class HeterogeneousRopePlan:
    """Validated R0-derived per-layer plan.

    ``tau`` is the direct EVQ-Cosh operating value.  ``multiplier_m`` uses the
    repository's operating-rule parameterization
    ``tau = m * effective_dim / sqrt(train_length)``.  A layer may instead
    provide an explicit inverse-frequency array.
    """

    num_layers: int
    rope_dim: int
    base: float
    train_length: int
    effective_dim: float
    attention_type: str
    num_heads: int
    head_dim: int
    d_rope: int | None
    d_nope: int | None
    n_kv_heads: int | None
    layers: tuple[LayerFrequencySpec, ...]
    source_schema_version: int | None = None
    source_label: str = "R0"

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if len(self.layers) != self.num_layers:
            raise ValueError(
                f"layers length {len(self.layers)} != num_layers {self.num_layers}"
            )
        if tuple(item.layer for item in self.layers) != tuple(range(self.num_layers)):
            raise ValueError("layer entries must be contiguous and ordered from 0")
        if self.attention_type not in SUPPORTED_ATTENTION_TYPES:
            raise ValueError(
                f"unsupported attention_type={self.attention_type!r}; "
                f"expected {SUPPORTED_ATTENTION_TYPES}"
            )
        if self.rope_dim <= 0 or self.rope_dim % 2:
            raise ValueError(f"rope_dim must be positive and even, got {self.rope_dim}")
        if self.base <= 1.0:
            raise ValueError(f"base must be > 1, got {self.base}")
        if self.train_length <= 0:
            raise ValueError(f"train_length must be positive, got {self.train_length}")

    @property
    def n_freqs(self) -> int:
        return self.rope_dim // 2

    @property
    def taus(self) -> tuple[float, ...]:
        result: list[float] = []
        for item in self.layers:
            if item.tau is None:
                raise ValueError("taus are unavailable when a layer uses explicit inv_freq")
            result.append(float(item.tau))
        return tuple(result)

    @property
    def all_layers_share_tau(self) -> bool:
        try:
            values = self.taus
        except ValueError:
            return False
        return all(math.isclose(values[0], value, rel_tol=0.0, abs_tol=1e-12) for value in values[1:])

    def as_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_schema_version": self.source_schema_version,
            "source_label": self.source_label,
            "num_layers": self.num_layers,
            "rope_dim": self.rope_dim,
            "n_freqs": self.n_freqs,
            "base": self.base,
            "train_length": self.train_length,
            "effective_dim": self.effective_dim,
            "attention_type": self.attention_type,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "d_rope": self.d_rope,
            "d_nope": self.d_nope,
            "n_kv_heads": self.n_kv_heads,
            "layers": [
                {
                    "layer": item.layer,
                    "tau": item.tau,
                    "m": item.multiplier_m,
                    "inv_freq_supplied": item.inv_freq is not None,
                    "source": item.source,
                }
                for item in self.layers
            ],
        }


def _layer_entry_map(container: Any, *, num_layers: int, name: str) -> dict[int, Any]:
    if isinstance(container, Mapping):
        result: dict[int, Any] = {}
        for key, value in container.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} layer key must be an integer: {key!r}") from exc
            if index in result:
                raise ValueError(f"duplicate {name} layer {index}")
            result[index] = value
    elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        result = {index: value for index, value in enumerate(container)}
    else:
        raise ValueError(f"{name} must be an array or object keyed by layer")
    expected = set(range(num_layers))
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ValueError(f"{name} layers mismatch; missing={missing}, extra={extra}")
    return result


def _infer_num_layers(root: Mapping[str, Any], model: Mapping[str, Any]) -> int:
    value = _first(root, "num_layers", "n_layers", default=None)
    if value is None:
        value = _first(model, "num_layers", "n_layers", default=None)
    if value is not None:
        return int(value)
    for key in (
        "per_layer_tau",
        "layer_tau",
        "tau_by_layer",
        "per_layer_m",
        "layer_m",
        "m_by_layer",
        "per_layer_inv_freq",
        "layer_inv_freq",
        "layers",
    ):
        if key not in root:
            continue
        value = root[key]
        if isinstance(value, Mapping):
            return len(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return len(value)
    raise ValueError("R0 JSON must provide num_layers or a per-layer array/object")


def _extract_layer_payloads(root: Mapping[str, Any], *, num_layers: int) -> tuple[list[Any], str]:
    # Explicit keys take precedence over the permissive ``layers`` shorthand.
    for key, kind in (
        ("per_layer_inv_freq", "inv_freq"),
        ("layer_inv_freq", "inv_freq"),
        ("per_layer_inverse_frequency", "inv_freq"),
        ("per_layer_tau", "tau"),
        ("layer_tau", "tau"),
        ("tau_by_layer", "tau"),
        ("per_layer_m", "m"),
        ("layer_m", "m"),
        ("m_by_layer", "m"),
    ):
        if key in root:
            values = _layer_entry_map(root[key], num_layers=num_layers, name=key)
            return [values[index] for index in range(num_layers)], kind

    if "per_layer" in root:
        container = root["per_layer"]
        if isinstance(container, Mapping):
            for key, kind in (
                ("inv_freq", "inv_freq"),
                ("inverse_frequency", "inv_freq"),
                ("tau", "tau"),
                ("m", "m"),
            ):
                if key in container:
                    values = _layer_entry_map(
                        container[key], num_layers=num_layers, name=f"per_layer.{key}"
                    )
                    return [values[index] for index in range(num_layers)], kind
        values = _layer_entry_map(container, num_layers=num_layers, name="per_layer")
        return [values[index] for index in range(num_layers)], "entry"

    if "layers" in root:
        values = _layer_entry_map(root["layers"], num_layers=num_layers, name="layers")
        return [values[index] for index in range(num_layers)], "entry"

    for key, kind in (("inv_freq", "inv_freq"), ("inverse_frequency", "inv_freq"), ("tau", "tau"), ("m", "m")):
        if key in root and not isinstance(root[key], Sequence):
            return [root[key] for _ in range(num_layers)], kind
        if key in root and isinstance(root[key], Sequence) and not isinstance(root[key], (str, bytes)):
            values = _layer_entry_map(root[key], num_layers=num_layers, name=key)
            return [values[index] for index in range(num_layers)], kind
    raise ValueError(
        "R0 JSON must provide per-layer tau/m/inv_freq or a scalar tau/m/inv_freq"
    )


def load_r0_json(path: str | Path) -> HeterogeneousRopePlan:
    """Load an R0 JSON with direct ``tau``, operating-rule ``m``, or inv_freq.

    Accepted shorthand examples are documented in README.md.  The parser is
    intentionally tolerant about wrappers (``r0``, ``allocation``) but strict
    about layer count, frequency shape, positivity, and ordering.
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("R0 JSON root must be an object")
    root = _unwrap_r0(payload)
    model = root.get("model") if isinstance(root.get("model"), Mapping) else payload.get("model", {})
    if not isinstance(model, Mapping):
        model = {}

    attention_type = str(
        _first(root, "attention_type", "attn_type", default=_first(model, "attention_type", "attn_type", default="mha"))
    ).lower()
    num_layers = _infer_num_layers(root, model)
    head_dim = int(_first(root, "head_dim", default=_first(model, "head_dim", default=64)))
    d_rope_value = _first(
        root,
        "d_rope",
        "rotary_dim",
        "rope_dim",
        default=_first(model, "d_rope", "rotary_dim", "rope_dim", default=None),
    )
    if attention_type == "mla":
        if d_rope_value is None:
            d_rope_value = head_dim
        rope_dim = int(d_rope_value)
    else:
        rope_dim = int(d_rope_value if d_rope_value is not None else head_dim)
    base = _as_positive_float(
        _first(root, "base", "rope_theta", default=_first(model, "base", "rope_theta", default=DEFAULT_BASE)),
        name="base",
    )
    train_length = int(
        _first(root, "train_length", "seq_len", "L_train", default=_first(model, "train_length", "seq_len", "L_train", default=DEFAULT_TRAIN_LENGTH))
    )
    effective_dim = _as_positive_float(
        _first(root, "effective_dim", "d_eff", default=_first(model, "effective_dim", "d_eff", default=rope_dim)),
        name="effective_dim",
    )
    num_heads = int(_first(root, "num_heads", default=_first(model, "num_heads", default=1)))
    d_nope_value = _first(root, "d_nope", default=_first(model, "d_nope", default=None))
    n_kv_value = _first(root, "n_kv_heads", default=_first(model, "n_kv_heads", default=None))

    layer_specs: list[LayerFrequencySpec] = []
    n_freqs = rope_dim // 2
    # A single direct table is a valid shared input; replicate it before the
    # per-layer parser.  Nested rows remain available through
    # ``per_layer_inv_freq`` or ``layers[].inv_freq``.
    shared_direct = _first(root, "shared_inv_freq", "shared_inverse_frequency", default=None)
    if shared_direct is None:
        for key in ("inv_freq", "inverse_frequency"):
            value = root.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) == n_freqs and all(not isinstance(item, Sequence) for item in value):
                    shared_direct = value
                    break
    if shared_direct is not None:
        payloads, payload_kind = [shared_direct] * num_layers, "inv_freq"
    else:
        payloads, payload_kind = _extract_layer_payloads(root, num_layers=num_layers)
    for index, payload_value in enumerate(payloads):
        tau: float | None = None
        multiplier: float | None = None
        inv_freq: tuple[float, ...] | None = None
        source_label = payload_kind
        if payload_kind == "entry":
            if not isinstance(payload_value, Mapping):
                # A bare list under ``layers`` is interpreted as tau values.
                tau = _as_nonnegative_float(payload_value, name=f"layers[{index}].tau")
                source_label = "layers.tau"
            else:
                entry = payload_value
                entry_index = _first(entry, "layer", "layer_index", "index", default=index)
                if int(entry_index) != index:
                    raise ValueError(f"layers[{index}] declares layer {entry_index}")
                if any(key in entry for key in ("inv_freq", "inverse_frequency")):
                    inv_freq = _parse_inv_freq(
                        _first(entry, "inv_freq", "inverse_frequency"),
                        expected=n_freqs,
                        name=f"layers[{index}].inv_freq",
                    )
                    source_label = "layers.inv_freq"
                elif "tau" in entry:
                    tau = _as_nonnegative_float(entry["tau"], name=f"layers[{index}].tau")
                    source_label = "layers.tau"
                elif "m" in entry or "multiplier_m" in entry:
                    multiplier = _as_nonnegative_float(
                        _first(entry, "m", "multiplier_m"),
                        name=f"layers[{index}].m",
                    )
                    source_label = "layers.m"
                else:
                    raise ValueError(f"layers[{index}] needs tau, m, or inv_freq")
        elif payload_kind == "inv_freq":
            inv_freq = _parse_inv_freq(payload_value, expected=n_freqs, name=f"layer_inv_freq[{index}]")
        elif payload_kind == "tau":
            tau = _as_nonnegative_float(payload_value, name=f"tau[{index}]")
        elif payload_kind == "m":
            multiplier = _as_nonnegative_float(payload_value, name=f"m[{index}]")
        if multiplier is not None:
            tau = multiplier * effective_dim / math.sqrt(float(train_length))
        layer_specs.append(
            LayerFrequencySpec(
                layer=index,
                tau=tau,
                multiplier_m=multiplier,
                inv_freq=inv_freq,
                source=source_label,
            )
        )

    return HeterogeneousRopePlan(
        num_layers=num_layers,
        rope_dim=rope_dim,
        base=base,
        train_length=train_length,
        effective_dim=effective_dim,
        attention_type=attention_type,
        num_heads=num_heads,
        head_dim=head_dim,
        d_rope=None if d_rope_value is None else int(d_rope_value),
        d_nope=None if d_nope_value is None else int(d_nope_value),
        n_kv_heads=None if n_kv_value is None else int(n_kv_value),
        layers=tuple(layer_specs),
        source_schema_version=int(payload.get("schema_version")) if payload.get("schema_version") is not None else None,
    )


def plan_from_values(
    *,
    num_layers: int,
    rope_dim: int,
    base: float = DEFAULT_BASE,
    train_length: int = DEFAULT_TRAIN_LENGTH,
    effective_dim: float | None = None,
    tau: float | Sequence[float] | None = None,
    m: float | Sequence[float] | None = None,
    inv_freq: Sequence[Sequence[float]] | None = None,
    attention_type: str = "mha",
    num_heads: int = 1,
    head_dim: int | None = None,
    d_rope: int | None = None,
    d_nope: int | None = None,
    n_kv_heads: int | None = None,
) -> HeterogeneousRopePlan:
    """Build a plan directly for tests or an interactive preflight."""

    if effective_dim is None:
        effective_dim = float(rope_dim)
    if sum(value is not None for value in (tau, m, inv_freq)) != 1:
        raise ValueError("provide exactly one of tau, m, or inv_freq")
    if head_dim is None:
        head_dim = rope_dim

    def values(value: Any, name: str) -> list[Any]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != num_layers:
                raise ValueError(f"{name} length {len(value)} != num_layers {num_layers}")
            return list(value)
        return [value for _ in range(num_layers)]

    layers: list[LayerFrequencySpec] = []
    if inv_freq is not None:
        rows = values(inv_freq, "inv_freq")
        for index, row in enumerate(rows):
            layers.append(
                LayerFrequencySpec(
                    layer=index,
                    tau=None,
                    multiplier_m=None,
                    inv_freq=_parse_inv_freq(row, expected=rope_dim // 2, name=f"inv_freq[{index}]"),
                    source="direct",
                )
            )
    elif tau is not None:
        for index, value in enumerate(values(tau, "tau")):
            layers.append(
                LayerFrequencySpec(index, _as_nonnegative_float(value, name=f"tau[{index}]"), None, None, "tau")
            )
    else:
        for index, value in enumerate(values(m, "m")):
            multiplier = _as_nonnegative_float(value, name=f"m[{index}]")
            layers.append(
                LayerFrequencySpec(
                    index,
                    multiplier * float(effective_dim) / math.sqrt(float(train_length)),
                    multiplier,
                    None,
                    "m",
                )
            )
    return HeterogeneousRopePlan(
        num_layers=num_layers,
        rope_dim=rope_dim,
        base=_as_positive_float(base, name="base"),
        train_length=int(train_length),
        effective_dim=float(effective_dim),
        attention_type=attention_type,
        num_heads=int(num_heads),
        head_dim=int(head_dim),
        d_rope=d_rope,
        d_nope=d_nope,
        n_kv_heads=n_kv_heads,
        layers=tuple(layers),
    )


def build_model_config(
    *,
    tier: str = "50m",
    attention_type: str = "mha",
    seq_len: int | None = None,
    d_rope: int | None = None,
    d_nope: int | None = None,
    n_kv_heads: int | None = None,
) -> dict[str, Any]:
    """Return an existing GPT/GQA/MLA config without touching training code."""

    from scripts.core_text_phases.run_evq_sweep import TIER_CONFIGS

    if tier not in TIER_CONFIGS:
        raise ValueError(f"unknown tier {tier!r}; expected {sorted(TIER_CONFIGS)}")
    config = dict(TIER_CONFIGS[tier])
    config["attn_type"] = attention_type
    if seq_len is not None:
        original_seq_len = int(config["seq_len"])
        config["seq_len"] = int(seq_len)
        config["max_position_embeddings"] = int(seq_len)
        config["batch_size"] = max(1, int(config["batch_size"] * original_seq_len / int(seq_len)))
    if attention_type == "mla":
        config["d_rope"] = int(d_rope if d_rope is not None else min(32, config["head_dim"]))
        config["d_nope"] = int(d_nope if d_nope is not None else config["head_dim"] - config["d_rope"])
        config.setdefault("v_head_dim", config["head_dim"])
        config.setdefault("kv_lora_rank", max(64, config["hidden_size"] // 4))
    elif d_rope is not None:
        raise ValueError("d_rope is only valid for attention_type='mla'")
    if attention_type == "gqa":
        if n_kv_heads is None:
            raise ValueError("gqa preflight requires n_kv_heads")
        config["n_kv_heads"] = int(n_kv_heads)
    elif n_kv_heads is not None:
        raise ValueError("n_kv_heads is only valid for attention_type='gqa'")
    return config


def build_layer_inv_freqs(
    plan: HeterogeneousRopePlan,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """Materialize one inverse-frequency tensor per layer."""

    from scripts.lib.rope.schedules import evq_cosh_inv_freq

    result: list[torch.Tensor] = []
    for item in plan.layers:
        if item.inv_freq is not None:
            value = torch.tensor(item.inv_freq, dtype=dtype)
        else:
            if item.tau is None:
                raise ValueError(f"layer {item.layer} has neither tau nor inv_freq")
            value = evq_cosh_inv_freq(
                head_dim=plan.rope_dim,
                tau=float(item.tau),
                base=float(plan.base),
                midpoint=True,
            ).to(dtype=dtype)
        if tuple(value.shape) != (plan.n_freqs,):
            raise ValueError(
                f"layer {item.layer} realized shape {tuple(value.shape)} != {(plan.n_freqs,)}"
            )
        if not torch.isfinite(value).all() or not (value > 0).all():
            raise ValueError(f"layer {item.layer} realized inv_freq is not finite positive")
        if not torch.all(value[:-1] > value[1:]):
            raise ValueError(f"layer {item.layer} realized inv_freq is not strictly decreasing")
        result.append(value.contiguous())
    return tuple(result)


def hash_tensor_raw(value: torch.Tensor) -> str:
    """Hash raw contiguous bytes while preserving the model buffer dtype."""

    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _clone_rope_with_inv_freq(rope: torch.nn.Module, inv_freq: torch.Tensor) -> torch.nn.Module:
    clone = copy.deepcopy(rope)
    old = getattr(clone, "inv_freq", None)
    if not torch.is_tensor(old) or old.ndim != 1:
        raise ValueError("existing rotary module has no 1-D inv_freq buffer")
    if tuple(old.shape) != tuple(inv_freq.shape):
        raise ValueError(
            f"inv_freq shape mismatch: existing={tuple(old.shape)} new={tuple(inv_freq.shape)}"
        )
    with torch.no_grad():
        old.copy_(inv_freq.to(device=old.device, dtype=old.dtype))
    max_seq = int(getattr(clone, "_max", getattr(clone, "max_seq_len_cached", 0)))
    if max_seq <= 0:
        raise ValueError("existing rotary module has no positive cache length")
    if not hasattr(clone, "_build"):
        raise ValueError("existing rotary module has no _build cache constructor")
    clone._build(max_seq)
    return clone


def _attention_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("existing GPT model has no blocks list")
    result: list[torch.nn.Module] = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if attention is None:
            attention = getattr(block, "attention", None)
        if attention is None or not hasattr(attention, "rope"):
            raise ValueError(f"block {index} has no attention.rope module")
        result.append(attention)
    return result


def install_layerwise_rope(
    model: torch.nn.Module,
    layer_inv_freqs: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """Clone the existing RoPE module once per layer and replace its buffer."""

    attentions = _attention_blocks(model)
    if len(attentions) != len(layer_inv_freqs):
        raise ValueError(
            f"layer count mismatch: model={len(attentions)} plan={len(layer_inv_freqs)}"
        )
    original_ids = [id(attention.rope) for attention in attentions]
    attention_classes_before = [type(attention).__name__ for attention in attentions]
    template = attentions[0].rope
    for attention, inv_freq in zip(attentions, layer_inv_freqs):
        attention.rope = _clone_rope_with_inv_freq(template, inv_freq)
    realized_ids = [id(attention.rope) for attention in attentions]
    if len(set(realized_ids)) != len(realized_ids):
        raise RuntimeError("layerwise RoPE installation did not produce independent modules")
    return {
        "layers": len(attentions),
        "shared_object_before": len(set(original_ids)) == 1,
        "unique_objects_after": len(set(realized_ids)),
        "attention_classes_before": attention_classes_before,
        "attention_classes_after": [type(attention).__name__ for attention in attentions],
        "attention_kernel_unchanged": attention_classes_before
        == [type(attention).__name__ for attention in attentions],
    }


def extend_layerwise_rope(model: torch.nn.Module, length: int) -> None:
    """Extend every cloned cache; this is not called by training preflight."""

    for attention in _attention_blocks(model):
        attention.rope._build(int(length))


def model_parameter_contract(model: torch.nn.Module) -> dict[str, Any]:
    entries = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "parameter_tensor_count": len(entries),
        "parameter_schema_sha256": hashlib.sha256(encoded).hexdigest(),
        "parameters": entries,
    }


def _rope_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [attention.rope for attention in _attention_blocks(model)]


def realized_frequency_receipt(model: torch.nn.Module) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, rope in enumerate(_rope_modules(model)):
        inv = getattr(rope, "inv_freq", None)
        if not torch.is_tensor(inv):
            raise ValueError(f"layer {index} has no tensor inv_freq")
        rows.append(
            {
                "layer": index,
                "shape": list(inv.shape),
                "dtype": str(inv.dtype),
                "sha256_raw": hash_tensor_raw(inv),
                "first": float(inv[0]),
                "last": float(inv[-1]),
                "minimum": float(inv.min()),
                "maximum": float(inv.max()),
            }
        )
    return {
        "layer_count": len(rows),
        "unique_hashes": sorted({row["sha256_raw"] for row in rows}),
        "layers": rows,
    }


def per_head_feasibility_gate(model: torch.nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    """Report why per-head training is gated for the current shared-table model."""

    attentions = _attention_blocks(model)
    rope_ids = [id(attention.rope) for attention in attentions]
    inv_shapes = [list(attention.rope.inv_freq.shape) for attention in attentions]
    attention_type = str(config.get("attn_type", "mha")).lower()
    return {
        "requested": "per_head",
        "status": "GATED_NOT_IMPLEMENTED",
        "feasible_in_principle": True,
        "training_allowed": False,
        "attention_kernel_change_required": False,
        "current_contract": {
            "attention_type": attention_type,
            "shared_rope_objects": len(set(rope_ids)) == 1,
            "inv_freq_shapes": inv_shapes,
            "cos_sin_contract": "[sequence, rotary_dim], broadcast across heads",
        },
        "blocking_reasons": [
            "The existing GPT/GQA/MLA path broadcasts one 1-D table across heads.",
            "Per-head tables require an explicit head axis in cos/sin construction and application.",
            "GQA needs separate query-head versus KV-head frequency semantics; MLA needs per-head d_rope routing.",
        ],
        "next_gate": "Implement and verify a head-axis RoPE module before any per-head training.",
    }
