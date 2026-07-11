#!/usr/bin/env python3
"""
EVQ-Cosh LoRA Training for LLaMA-3-8B-Instruct
================================================
Supporting experiment: validates LoRA r=64 with EVQ-cosh τ=1.414 against
the phase-transition collapse observed at r=16 (PPL 77.1).

Usage:
    # Full training (requires GPU)
    python train_evq_lora.py --output_dir ./checkpoints/evq_r64

    # Dry-run (no GPU, validates config only)
    python train_evq_lora.py --dry_run --output_dir ./checkpoints/evq_r64_dry

    # Custom tau / rank
    python train_evq_lora.py --tau 1.0 --lora_r 32 --output_dir ./checkpoints/evq_r32
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.rope.schedules import (
    evq_cosh_inv_freq as _canonical_evq_cosh_inv_freq,
    geometric_inv_freq as _canonical_geometric_inv_freq,
)

try:
    from .legacy_lora_protocol import (
        canonical_json_sha256,
        canonical_training_protocol,
        sha256_file,
        validate_legacy_protocol,
        validate_legacy_runtime_packages,
        validate_training_source_receipt,
    )
except ImportError:  # direct script execution
    from legacy_lora_protocol import (
        canonical_json_sha256,
        canonical_training_protocol,
        sha256_file,
        validate_legacy_protocol,
        validate_legacy_runtime_packages,
        validate_training_source_receipt,
    )


@dataclass(frozen=True)
class LoraRopeGeometry:
    head_dim: int
    rope_base: float


LEGACY_CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
)


def validate_strict_legacy_args(args: argparse.Namespace) -> None:
    """Reject any drift from the historical scientific/runtime contract."""
    expected = {
        "max_seq_len": 8192,
        "max_samples": 8000,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_targets": "q_proj,k_proj,v_proj,o_proj",
        "max_steps": 300,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "warmup_steps": 60,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "save_steps": 100,
        "bf16": True,
        "load_in_4bit": False,
        "compile": True,
        "compile_mode": "default",
    }
    if args.rope_method not in ("native_geo", "evq_cosh"):
        raise ValueError("strict legacy protocol supports only native_geo and evq_cosh")
    if args.seed not in (42, 43, 44):
        raise ValueError("strict legacy protocol requires seed 42, 43, or 44")
    if args.rope_method == "evq_cosh" and not math.isclose(
        float(args.tau), 1.414, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("strict legacy EVQ protocol requires tau=1.414")
    for key, expected_value in expected.items():
        actual = getattr(args, key)
        if isinstance(expected_value, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(
                f"strict legacy protocol mismatch for {key}: "
                f"{actual!r} != {expected_value!r}"
            )


def resolve_legacy_resume_checkpoint(
    output_dir: Union[os.PathLike, str],
    requested: Optional[str],
    *,
    expected_protocol: Optional[Dict[str, Any]] = None,
    expected_runtime_packages: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    """Resolve an explicit checkpoint or the newest internally consistent one."""
    if requested in (None, "", "none"):
        return None
    output_dir = Path(output_dir)
    if requested != "auto":
        path = Path(requested)
        if not path.is_dir():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        candidates = [path]
    else:
        candidates = []
        for path in output_dir.glob("checkpoint-*"):
            suffix = path.name.removeprefix("checkpoint-")
            if path.is_dir() and suffix.isdigit():
                candidates.append(path)
        candidates.sort(key=lambda path: int(path.name.removeprefix("checkpoint-")), reverse=True)
        if not candidates:
            return None
    for path in candidates:
        missing = [name for name in LEGACY_CHECKPOINT_FILES if not (path / name).is_file()]
        if missing:
            if requested == "auto":
                continue
            raise FileNotFoundError(
                f"resume checkpoint is incomplete ({', '.join(missing)}): {path}"
            )
        state_path = path / "trainer_state.json"
        if not state_path.is_file():
            if requested == "auto":
                continue
            raise FileNotFoundError(f"resume checkpoint lacks trainer_state.json: {path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        directory_step = int(path.name.removeprefix("checkpoint-"))
        if int(state.get("global_step", -1)) != directory_step:
            if requested == "auto":
                continue
            raise ValueError(f"resume checkpoint step mismatch: {path}")
        if expected_protocol is not None:
            try:
                validate_legacy_checkpoint_receipt(
                    path,
                    expected_protocol,
                    expected_runtime_packages,
                )
            except (FileNotFoundError, TypeError, ValueError):
                if requested == "auto":
                    continue
                raise
        return path
    raise RuntimeError("no internally consistent recovery checkpoint was found")


def current_legacy_runtime_packages() -> Dict[str, str]:
    """Return and validate the exact package lock used by the legacy control."""
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "accelerate", "datasets", "triton")
    }
    return validate_legacy_runtime_packages(packages)


def write_legacy_checkpoint_receipt(
    checkpoint_dir: Union[os.PathLike, str],
    protocol: Dict[str, Any],
    runtime_packages: Dict[str, str],
) -> Path:
    """Atomically bind a recovery checkpoint to protocol, code, data, and runtime."""
    checkpoint_dir = Path(checkpoint_dir)
    protocol = validate_legacy_protocol(protocol)
    runtime_packages = validate_legacy_runtime_packages(runtime_packages)
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"checkpoint lacks trainer_state.json: {checkpoint_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    suffix = checkpoint_dir.name.removeprefix("checkpoint-")
    if not suffix.isdigit() or int(state.get("global_step", -1)) != int(suffix):
        raise ValueError(f"checkpoint step mismatch: {checkpoint_dir}")
    receipt = {
        "format_version": 1,
        "global_step": int(suffix),
        "protocol_sha256": canonical_json_sha256(protocol),
        "method": protocol["method"],
        "seed": protocol["seed"],
        "data_manifest_sha256": protocol["data_manifest_sha256"],
        "model_manifest_sha256": protocol["model_manifest_sha256"],
        "code_sha256": protocol["code_sha256"],
        "runtime_packages": runtime_packages,
        "files": {
            name: {
                "size": (checkpoint_dir / name).stat().st_size,
                "sha256": sha256_file(checkpoint_dir / name),
            }
            for name in LEGACY_CHECKPOINT_FILES
        },
    }
    path = checkpoint_dir / "checkpoint_receipt.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def validate_legacy_checkpoint_receipt(
    checkpoint_dir: Union[os.PathLike, str],
    expected_protocol: Dict[str, Any],
    expected_runtime_packages: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Reject a resume candidate created under any different experiment state."""
    checkpoint_dir = Path(checkpoint_dir)
    expected_protocol = validate_legacy_protocol(expected_protocol)
    if expected_runtime_packages is None:
        raise ValueError("strict resume requires expected runtime packages")
    expected_runtime_packages = validate_legacy_runtime_packages(expected_runtime_packages)
    path = checkpoint_dir / "checkpoint_receipt.json"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint lacks protocol receipt: {checkpoint_dir}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    suffix = checkpoint_dir.name.removeprefix("checkpoint-")
    expected = {
        "format_version": 1,
        "global_step": int(suffix),
        "protocol_sha256": canonical_json_sha256(expected_protocol),
        "method": expected_protocol["method"],
        "seed": expected_protocol["seed"],
        "data_manifest_sha256": expected_protocol["data_manifest_sha256"],
        "model_manifest_sha256": expected_protocol["model_manifest_sha256"],
        "code_sha256": expected_protocol["code_sha256"],
        "runtime_packages": expected_runtime_packages,
        "files": {
            name: {
                "size": (checkpoint_dir / name).stat().st_size,
                "sha256": sha256_file(checkpoint_dir / name),
            }
            for name in LEGACY_CHECKPOINT_FILES
        },
    }
    if receipt != expected:
        raise ValueError(f"checkpoint protocol receipt mismatch: {checkpoint_dir}")
    return receipt


def _load_strict_legacy_data(manifest_path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("objective") != "legacy_longalign_full_token_causal_lm_v2":
        raise ValueError("prepared data objective is not the legacy LongAlign protocol")
    validate_training_source_receipt(manifest.get("source", {}))
    expected_preparation = {
        "max_samples": 8000,
        "max_seq_len": 8192,
        "minimum_tokens": 64,
        "validation_ratio": 0.02,
        "split_seed": 42,
        "selection_order": "first_supported_rows_before_tokenization",
        "labels": "all_non_padding_input_tokens",
        "variable_length": True,
    }
    if manifest.get("preparation") != expected_preparation:
        raise ValueError("prepared legacy LongAlign settings mismatch")
    loaded: Dict[str, Any] = {}
    for key in ("tokens", "offsets", "train_indices", "validation_indices"):
        record = manifest.get("files", {}).get(key, {})
        name = str(record.get("name", ""))
        if not name or Path(name).name != name:
            raise ValueError(f"prepared legacy {key} path is not a safe basename")
        path = manifest_path.parent / name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"prepared legacy {key} tensor is missing or hash-mismatched")
        loaded[key] = torch.load(path, map_location="cpu", weights_only=True)
    validate_prepared_training_data(loaded, manifest, vocab_size=128256)
    return loaded, manifest


def validate_prepared_training_data(
    loaded: Dict[str, Any],
    manifest: Dict[str, Any],
    *,
    vocab_size: int,
) -> None:
    """Validate frozen tensors fully before loading an 8B model onto the GPU."""
    tokens = loaded.get("tokens")
    offsets = loaded.get("offsets")
    train_indices = loaded.get("train_indices")
    validation_indices = loaded.get("validation_indices")
    tensors = {
        "tokens": tokens,
        "offsets": offsets,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    for name, value in tensors.items():
        if not torch.is_tensor(value) or value.ndim != 1:
            raise ValueError(f"prepared legacy {name} must be a 1-D tensor")
    if offsets.numel() < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != tokens.numel():
        raise ValueError("prepared legacy offsets do not cover the token tensor")
    lengths = offsets[1:].to(torch.int64) - offsets[:-1].to(torch.int64)
    if torch.any(lengths <= 0) or int(lengths.min()) < 64 or int(lengths.max()) > 8192:
        raise ValueError("prepared legacy row lengths must lie in [64, 8192]")
    if tokens.numel() == 0 or int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size:
        raise ValueError("prepared legacy token IDs fall outside the model vocabulary")

    row_count = offsets.numel() - 1
    normalized_indices: Dict[str, torch.Tensor] = {}
    for name, value in (
        ("train_indices", train_indices),
        ("validation_indices", validation_indices),
    ):
        indices = value.to(torch.int64)
        if indices.numel() == 0:
            raise ValueError(f"prepared legacy {name} must not be empty")
        if int(indices.min()) < 0 or int(indices.max()) >= row_count:
            raise ValueError(f"prepared legacy {name} contains an out-of-range row")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError(f"prepared legacy {name} contains duplicate rows")
        normalized_indices[name] = indices
    all_indices = torch.cat(
        [normalized_indices["train_indices"], normalized_indices["validation_indices"]]
    )
    if torch.unique(all_indices).numel() != all_indices.numel():
        raise ValueError("prepared legacy train and validation splits overlap")
    if all_indices.numel() != row_count or not torch.equal(
        torch.sort(all_indices).values,
        torch.arange(row_count, dtype=torch.int64),
    ):
        raise ValueError("prepared legacy splits do not cover every tokenized row exactly once")

    statistics = manifest.get("statistics", {})
    expected_statistics = {
        "tokenized_rows": row_count,
        "train_rows": train_indices.numel(),
        "validation_rows": validation_indices.numel(),
        "minimum_length": int(lengths.min()),
        "maximum_length": int(lengths.max()),
    }
    for name, expected in expected_statistics.items():
        if int(statistics.get(name, -1)) != expected:
            raise ValueError(f"prepared legacy statistics mismatch for {name}")


def _ensure_immutable_protocol(output_dir: Path, protocol: Dict[str, Any]) -> Path:
    validate_legacy_protocol(protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_protocol.json"
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != protocol:
            raise RuntimeError("legacy run protocol mismatch; use a fresh output directory")
        return path
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def legacy_training_code_sha256() -> str:
    """Bind resumes and all six arms to the exact training implementation."""
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("legacy_lora_protocol.py").resolve(),
        (PROJECT_ROOT / "scripts/lib/rope/schedules.py").resolve(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def legacy_runtime_identity(
    validated_packages: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    packages: Dict[str, Any] = dict(validated_packages or {})
    if not packages:
        for name in ("torch", "transformers", "peft", "accelerate", "datasets", "triton"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
    identity = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        identity["cuda_device"] = torch.cuda.get_device_name(0)
        identity["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return identity


def resolve_model_rope_geometry(
    config: Any,
    head_dim_override: Optional[int] = None,
    rope_base_override: Optional[float] = None,
) -> LoraRopeGeometry:
    """Resolve rotary geometry from config, with explicit overrides when requested."""
    head_dim = head_dim_override if head_dim_override is not None else getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        num_heads = getattr(config, "num_attention_heads", None)
        if not hidden_size or not num_heads or int(hidden_size) % int(num_heads) != 0:
            raise ValueError(
                "Cannot infer head_dim: config needs head_dim or divisible "
                "hidden_size/num_attention_heads"
            )
        head_dim = int(hidden_size) // int(num_heads)
    head_dim = int(head_dim)
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")

    rope_base = rope_base_override
    if rope_base is None:
        rope_base = getattr(config, "rope_theta", None)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_base is None and isinstance(rope_scaling, dict):
            rope_base = rope_scaling.get("rope_theta")
    rope_base = float(rope_base if rope_base is not None else 10_000.0)
    if rope_base <= 0:
        raise ValueError(f"rope_base must be positive, got {rope_base}")
    return LoraRopeGeometry(head_dim=head_dim, rope_base=rope_base)


def validate_legacy_model_geometry(config: Any, geometry: LoraRopeGeometry) -> None:
    expected = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
    }
    for field, value in expected.items():
        if getattr(config, field, None) != value:
            raise ValueError(
                f"strict legacy protocol model geometry mismatch for {field}: "
                f"{getattr(config, field, None)!r} != {value!r}"
            )
    if geometry.head_dim != 128 or not math.isclose(
        geometry.rope_base, 500_000.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("strict legacy protocol requires head_dim=128 and rope_base=500000")
    if getattr(config, "rope_scaling", None) not in (None, {}):
        raise ValueError("strict legacy protocol requires unscaled native LLaMA RoPE config")


def evaluation_strategy_kwargs(training_arguments_cls, value: str = "no") -> Dict[str, str]:
    """Return the evaluation-strategy keyword supported by this Transformers version."""
    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    if "eval_strategy" in parameters:
        return {"eval_strategy": value}
    if "evaluation_strategy" in parameters:
        return {"evaluation_strategy": value}
    raise RuntimeError("TrainingArguments supports neither eval_strategy nor evaluation_strategy")


def public_model_identifier(value: Optional[str]) -> Optional[str]:
    """Keep public Hub IDs while removing local-machine path prefixes."""
    if value is None:
        return None
    raw = str(value)
    expanded = Path(raw).expanduser()
    if expanded.is_absolute() or raw.startswith((".", "~")):
        return expanded.name
    return raw


def public_artifact_identifier(value: Optional[str]) -> Optional[str]:
    """Return only the basename of a local artifact path."""
    if value is None:
        return None
    return Path(str(value)).name


def training_dataset_identifier(
    args: argparse.Namespace,
    strict_manifest: Optional[Dict[str, Any]],
) -> str:
    if strict_manifest is not None:
        source_id = strict_manifest.get("source", {}).get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("strict prepared-data manifest lacks source_id")
        return source_id
    if args.local_data_path is not None:
        return public_artifact_identifier(args.local_data_path) or "local_data"
    return str(args.dataset_name)


def load_frequency_artifact(
    path: Union[os.PathLike, str],
    expected_method: Optional[str] = None,
) -> tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    """Load and validate a frequency artifact, returning path-safe provenance."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Required frequency artifact not found: {artifact_path.name}")

    data = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not torch.is_tensor(data.get("inv_freq")):
        raise RuntimeError(
            f"Invalid frequency artifact {artifact_path.name}: expected a metadata dict with inv_freq"
        )
    method = data.get("method")
    if not isinstance(method, str) or not method:
        raise RuntimeError(f"Invalid frequency artifact {artifact_path.name}: missing method")
    if expected_method is not None and method != expected_method:
        raise RuntimeError(
            f"Frequency artifact method mismatch: expected {expected_method}, found {method}"
        )

    inv_freq = data["inv_freq"].detach().cpu()
    if inv_freq.ndim != 1 or inv_freq.numel() == 0:
        raise RuntimeError(
            f"Invalid frequency artifact {artifact_path.name}: inv_freq must be a non-empty 1-D tensor"
        )
    recorded_head_dim = data.get("head_dim")
    if recorded_head_dim is not None and int(recorded_head_dim) != 2 * inv_freq.numel():
        raise RuntimeError(
            f"Invalid frequency artifact {artifact_path.name}: head_dim={recorded_head_dim} "
            f"but inv_freq has {inv_freq.numel()} channels"
        )

    digest = hashlib.sha256()
    with artifact_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    provenance = {
        "artifact": artifact_path.name,
        "sha256": digest.hexdigest(),
        "method": method,
        "head_dim": int(recorded_head_dim) if recorded_head_dim is not None else 2 * inv_freq.numel(),
        "base": float(data["base"]) if data.get("base") is not None else None,
        "tau": float(data["tau"]) if data.get("tau") is not None else None,
        "midpoint": bool(data["midpoint"]) if data.get("midpoint") is not None else None,
    }
    return inv_freq, data, provenance


# ---------------------------------------------------------------------------
# EVQ-Cosh frequency computation (from theory)
# ---------------------------------------------------------------------------

def compute_evq_cosh_inv_freq(
    head_dim: int,
    base: float,
    tau: float,
    midpoint: bool = True,
) -> torch.Tensor:
    """Compute EVQ-cosh inverse frequencies.

    φ_k(τ) = 1 - (1/τ) * arcsinh((1 - u_k) * sinh(τ))
    inv_freq_k = base^{-φ_k}

    Args:
        head_dim: dimension per attention head (e.g. 128)
        base: RoPE theta base (e.g. 500000)
        tau: temperature parameter (theory: d_head/√L)
        midpoint: if True, use u_k = (2k-1)/(2K) (midpoint quantization)
                  if False, use u_k = k/K (boundary quantization)
    """
    return _canonical_evq_cosh_inv_freq(
        head_dim=head_dim,
        tau=tau,
        base=base,
        midpoint=midpoint,
        dtype=torch.float64,
    )


def compute_geometric_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    """Standard geometric RoPE inverse frequencies."""
    return _canonical_geometric_inv_freq(head_dim=head_dim, base=base, dtype=torch.float64)


def build_training_inv_freq(
    rope_method: str,
    head_dim: int,
    base: float,
    tau: float,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Build the exact schedule requested by a LoRA training protocol.

    ``tau=0`` on the midpoint EVQ grid is not the native LLaMA geometric
    schedule: every channel is shifted by half a quantization cell.  Native
    geometric controls therefore use an explicit method instead of overloading
    the EVQ temperature.
    """
    if rope_method == "native_geo":
        return compute_geometric_inv_freq(head_dim, base), {
            "method": "native_geo",
            "tau": None,
            "midpoint": False,
        }
    if rope_method == "evq_cosh":
        return compute_evq_cosh_inv_freq(
            head_dim=head_dim,
            base=base,
            tau=tau,
            midpoint=True,
        ), {
            "method": "evq_cosh",
            "tau": float(tau),
            "midpoint": True,
        }
    raise ValueError(
        f"Unsupported rope_method={rope_method!r}; expected 'evq_cosh' or 'native_geo'"
    )


# ---------------------------------------------------------------------------
# RoPE injection
# ---------------------------------------------------------------------------

def find_rotary_modules(model: torch.nn.Module):
    """Find all modules with inv_freq buffer."""
    out = []
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq") and torch.is_tensor(module.inv_freq):
            out.append((name, module))
    return out


def verify_model_inv_freq(
    model: torch.nn.Module,
    expected_inv_freq: torch.Tensor,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Fail unless every rotary module contains the requested frequency table."""
    modules = find_rotary_modules(model)
    if not modules:
        raise RuntimeError("No rotary modules with inv_freq found during verification")

    expected = expected_inv_freq.detach().cpu().reshape(-1).to(torch.float64)
    max_error = 0.0
    for name, module in modules:
        actual = module.inv_freq.detach().cpu().reshape(-1).to(torch.float64)
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"Frequency shape mismatch at {name}: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        error = (actual - expected).abs().max().item()
        if error >= tolerance:
            raise RuntimeError(
                f"Frequency mismatch at {name}: max_error={error:.2e}, "
                f"tolerance={tolerance:.2e}"
            )
        max_error = max(max_error, error)
    return {"verified_count": len(modules), "max_error": max_error}


def inject_inv_freq(model: torch.nn.Module, inv_freq: torch.Tensor) -> Dict[str, Any]:
    """Inject custom inv_freq into all rotary modules."""
    modules = find_rotary_modules(model)
    if not modules:
        raise RuntimeError("No rotary modules with inv_freq found in model.")

    expected = inv_freq.detach().cpu().view(-1)
    changed = []

    for name, module in modules:
        old = module.inv_freq
        if old.numel() != expected.numel():
            raise RuntimeError(
                f"Shape mismatch at {name}: old={tuple(old.shape)} new={tuple(expected.shape)}"
            )
        with torch.no_grad():
            old.copy_(expected.to(device=old.device, dtype=old.dtype))
        # Clear cached cos/sin
        for attr in ("_cos_cached", "_sin_cached", "cos_cached", "sin_cached",
                      "_cos_cache", "_sin_cache", "max_seq_len_cached"):
            if hasattr(module, attr):
                val = getattr(module, attr)
                if isinstance(val, (int, float)):
                    setattr(module, attr, 0)
                else:
                    setattr(module, attr, None)
        changed.append(name)

    return {"patched_count": len(modules), "changed_modules": changed}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(
    tokenizer,
    dataset_name: str = "THUDM/LongAlign-10k",
    max_seq_len: int = 8192,
    max_samples: int = 8000,
    val_ratio: float = 0.02,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and tokenize long-context training data. Caches tokenized result to disk."""
    import hashlib, random
    from datasets import load_dataset

    # Check cache
    if cache_dir:
        cache_key = hashlib.md5(f"{dataset_name}_{max_seq_len}_{max_samples}".encode()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, f"tokenized_{cache_key}.pt")
        if os.path.exists(cache_path):
            print(f"[DATA] Loading cached tokenized data from {cache_path}")
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"[DATA] Train: {len(cached['train'])}, Val: {len(cached['val'])}")
            return cached

    print(f"[DATA] Loading dataset: {dataset_name}")
    if dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
        ds = load_dataset("json", data_files=dataset_name, split="train")
    else:
        ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    print(f"[DATA] Raw samples: {len(ds)}")

    # Normalize to messages format
    processed = []
    for item in ds:
        messages = None

        # Format 1: messages array (LongAlign style)
        if "messages" in item and item["messages"]:
            messages = item["messages"]
        # Format 2: instruction/input/output
        elif "instruction" in item:
            user_text = item.get("instruction", "")
            if item.get("input"):
                user_text = f"{user_text}\n\n{item['input']}"
            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": item.get("output", "")},
            ]
        # Format 3: question/context/answer
        elif "question" in item:
            ctx = item.get("context", "")
            q = item.get("question", "")
            user_text = f"{ctx}\n\n{q}" if ctx else q
            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": item.get("answer", item.get("answers", ""))},
            ]

        if messages:
            processed.append(messages)

        if len(processed) >= max_samples:
            break

    print(f"[DATA] Processed samples: {len(processed)}")

    # Tokenize
    tokenized = []
    skipped = 0
    for i, msgs in enumerate(processed):
        if i % 1000 == 0 and i > 0:
            print(f"[DATA] Tokenizing {i}/{len(processed)}...")
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            parts = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
            text = "<|begin_of_text|>" + "".join(parts)

        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_tensors=None,
        )
        if len(enc["input_ids"]) < 64:
            skipped += 1
            continue
        tokenized.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

    print(f"[DATA] Tokenized: {len(tokenized)}, skipped (too short): {skipped}")

    # Token length distribution
    lengths = [len(t["input_ids"]) for t in tokenized]
    print(f"[DATA] Token lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")
    print(f"[DATA] Samples at max_seq_len: {sum(1 for l in lengths if l >= max_seq_len - 10)}")

    # Split train/val
    n_val = max(1, int(len(tokenized) * val_ratio))
    random.seed(42)
    indices = list(range(len(tokenized)))
    random.shuffle(indices)
    val_indices = set(indices[:n_val])
    train_data = [tokenized[i] for i in indices if i not in val_indices]
    val_data = [tokenized[i] for i in val_indices]

    print(f"[DATA] Train: {len(train_data)}, Val: {len(val_data)}")

    result = {"train": train_data, "val": val_data}

    # Save cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(result, cache_path)
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"[DATA] Cached to {cache_path} ({size_mb:.0f}MB)")

    return result


class TokenizedDataset(torch.utils.data.Dataset):
    """Simple dataset from pre-tokenized data."""
    def __init__(self, data: List[Dict], max_seq_len: int):
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"][:self.max_seq_len]
        attention_mask = item["attention_mask"][:self.max_seq_len]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": list(input_ids),
        }


class CompactTokenizedDataset(torch.utils.data.Dataset):
    """Variable-length rows backed by one compact int32 token tensor."""

    def __init__(
        self,
        tokens: torch.Tensor,
        offsets: torch.Tensor,
        indices: torch.Tensor,
        max_seq_len: int,
    ):
        if tokens.ndim != 1 or offsets.ndim != 1 or indices.ndim != 1:
            raise ValueError("compact legacy tensors must all be one-dimensional")
        if offsets.numel() < 2 or int(offsets[0]) != 0 or int(offsets[-1]) != tokens.numel():
            raise ValueError("compact legacy offsets do not cover the token tensor")
        if not torch.all(offsets[1:] >= offsets[:-1]):
            raise ValueError("compact legacy offsets must be monotonic")
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= offsets.numel() - 1):
            raise ValueError("compact legacy split indices are out of bounds")
        self.tokens = tokens
        self.offsets = offsets
        self.indices = indices
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, idx):
        row = int(self.indices[idx])
        start = int(self.offsets[row])
        end = min(int(self.offsets[row + 1]), start + self.max_seq_len)
        input_ids = self.tokens[start:end].to(torch.long).tolist()
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": list(input_ids),
        }


class PaddingCollator:
    """Pad variable-length samples to the longest in the batch."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EVQ-Cosh LoRA Training")

    # Model
    p.add_argument("--model_name", type=str,
                   default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--output_dir", type=str, required=True)

    # EVQ-Cosh parameters
    p.add_argument(
        "--rope_method",
        choices=["evq_cosh", "native_geo"],
        default="evq_cosh",
        help="Training frequency schedule. Use native_geo for an exact LLaMA geometric control.",
    )
    p.add_argument("--tau", type=float, default=1.414,
                   help="EVQ-cosh temperature (ignored for --rope_method native_geo)")
    p.add_argument("--rope_base", type=float, default=None,
                   help="RoPE theta base override (default: infer from model config)")
    p.add_argument("--head_dim", type=int, default=None,
                   help="Attention head dimension override (default: infer from model config)")

    # LoRA
    p.add_argument("--lora_r", type=int, default=64,
                   help="LoRA rank (theory requires r >= K = d_head/2)")
    p.add_argument("--lora_alpha", type=int, default=128,
                   help="LoRA alpha (default: 2 * lora_r)")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str,
                   default="q_proj,k_proj,v_proj,o_proj")

    # Training (bf16 full-precision LoRA — lr lower than QLoRA's 2e-4)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=60)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_seq_len", type=int, default=8192)

    # Data
    p.add_argument("--dataset_name", type=str,
                   default="THUDM/LongAlign-10k",
                   help="HuggingFace dataset name")
    p.add_argument("--max_samples", type=int, default=8000)
    p.add_argument("--local_data_path", type=str, default=None,
                   help="Path to local JSONL data (overrides --dataset_name)")
    p.add_argument(
        "--prepared_data_manifest",
        type=Path,
        default=None,
        help="Frozen legacy LongAlign manifest; required by strict protocol mode",
    )
    p.add_argument(
        "--model_manifest",
        type=Path,
        default=None,
        help="Full model-byte manifest; required by strict protocol mode",
    )

    # Quantization (96GB GPU: default bf16 full precision, no quantization needed)
    p.add_argument("--load_in_4bit", action="store_true", default=False,
                   help="Use 4-bit QLoRA (only if GPU < 40GB)")
    p.add_argument("--no_4bit", action="store_true", default=False)

    # Control
    p.add_argument("--dry_run", action="store_true",
                   help="Validate config without training (no GPU required)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument(
        "--strict_legacy_protocol",
        action="store_true",
        help="Lock the fresh Geo/EVQ 3-seed LongAlign rebuttal protocol",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        default="none",
        help="none, auto, or an explicit checkpoint directory",
    )
    p.add_argument("--compile", action="store_true", help="Enable torch.compile through Trainer")
    p.add_argument("--compile_mode", default="default", choices=("default", "reduce-overhead", "max-autotune"))

    return p.parse_args()


def validate_theory(args) -> Dict[str, Any]:
    """Validate experiment parameters against theoretical predictions."""
    K = args.head_dim // 2
    tau_theory = args.head_dim / math.sqrt(args.max_seq_len)
    r_ratio = args.lora_r / K

    rope_method = getattr(args, "rope_method", "evq_cosh")
    tau_match = abs(args.tau - tau_theory) < 0.1 if rope_method == "evq_cosh" else None
    checks = {
        "rope_method": rope_method,
        "head_dim": args.head_dim,
        "K_channels": K,
        "lora_r": args.lora_r,
        "r_over_K": r_ratio,
        "tau_set": args.tau,
        "tau_theory": round(tau_theory, 4),
        "tau_match": tau_match,
        "phase_transition_safe": args.lora_r >= K,
    }

    print("\n" + "=" * 60)
    print("THEORETICAL VALIDATION")
    print("=" * 60)
    print(f"  head_dim       = {args.head_dim}")
    print(f"  K (channels)   = {K}")
    print(f"  LoRA rank r    = {args.lora_r}")
    print(f"  r / K          = {r_ratio:.2f} {'✅' if r_ratio >= 1.0 else '⚠️' if r_ratio >= 0.5 else '❌'}")
    print(f"  RoPE method    = {rope_method}")
    print(f"  τ (set)        = {args.tau if rope_method == 'evq_cosh' else 'n/a'}")
    print(f"  τ* (theory)    = {tau_theory:.4f}")
    print(f"  τ match        = {'n/a (native geometric control)' if tau_match is None else '✅' if tau_match else '⚠️'}")
    print(f"  Phase-safe     = {'✅' if checks['phase_transition_safe'] else '❌ DANGER'}")

    if r_ratio < 0.5:
        print(f"\n  ⚠️  WARNING: r/K = {r_ratio:.2f} < 0.5")
        print(f"  Theory predicts EVQ will FAIL (phase transition at r_c ≈ K = {K})")
        print(f"  Recommend: r >= {K} (r/K >= 1.0)")

    print("=" * 60 + "\n")
    return checks


def compute_and_save_inv_freq(args, preserve_existing: bool = False) -> torch.Tensor:
    """Compute the requested frequencies and save them for reproducibility."""
    inv_freq, schedule_meta = build_training_inv_freq(
        rope_method=args.rope_method,
        head_dim=args.head_dim,
        base=args.rope_base,
        tau=args.tau,
    )
    inv_freq_geo = compute_geometric_inv_freq(args.head_dim, args.rope_base)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    freq_path = os.path.join(args.output_dir, "custom_inv_freq.pt")
    if preserve_existing and os.path.exists(freq_path):
        recorded, _, _ = load_frequency_artifact(
            freq_path,
            expected_method=args.rope_method,
        )
        if not torch.allclose(recorded.to(torch.float64), inv_freq.to(torch.float64), rtol=0.0, atol=1e-12):
            raise RuntimeError("existing strict frequency artifact does not match the protocol")
        inv_freq = recorded.to(torch.float64)
        print(f"[FREQ] Reusing verified artifact {freq_path}")
    else:
        torch.save({
            "inv_freq": inv_freq,
            "tau": schedule_meta["tau"],
            "head_dim": args.head_dim,
            "base": args.rope_base,
            "method": schedule_meta["method"],
            "midpoint": schedule_meta["midpoint"],
        }, freq_path)
        print(f"[FREQ] Saved to {freq_path}")

    # Diagnostic comparison
    K = args.head_dim // 2
    print(f"\n[FREQ] {schedule_meta['method']} vs native geometric comparison:")
    print(f"  {'Chan':>4s}  {'Selected':>12s}  {'Geo':>12s}  {'Ratio':>8s}")
    for k in [0, K//4, K//2, 3*K//4, K-1]:
        e = inv_freq[k].item()
        g = inv_freq_geo[k].item()
        print(f"  {k:4d}  {e:12.6f}  {g:12.6f}  {e/g:8.4f}")

    return inv_freq


def main():
    args = parse_args()
    if args.no_4bit:
        args.load_in_4bit = False

    # Resolve model-specific rotary geometry before computing or saving a schedule.
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    geometry = resolve_model_rope_geometry(
        model_config,
        head_dim_override=args.head_dim,
        rope_base_override=args.rope_base,
    )
    args.head_dim = geometry.head_dim
    args.rope_base = geometry.rope_base
    print(
        f"[ROPE] Resolved model geometry: head_dim={args.head_dim}, "
        f"rope_base={args.rope_base:g}"
    )

    strict_protocol = None
    strict_runtime_packages = None
    strict_data = None
    strict_manifest = None
    if args.strict_legacy_protocol:
        validate_strict_legacy_args(args)
        validate_legacy_model_geometry(model_config, geometry)
        if args.prepared_data_manifest is None or args.model_manifest is None:
            raise ValueError(
                "strict legacy mode requires --prepared_data_manifest and --model_manifest"
            )
        if args.local_data_path is not None:
            raise ValueError("strict legacy mode reads only the frozen prepared-data manifest")
        if not args.model_manifest.is_file():
            raise FileNotFoundError(args.model_manifest)
        model_manifest_data = json.loads(args.model_manifest.read_text(encoding="utf-8"))
        try:
            from .prepare_legacy_model_manifest import validate_model_manifest
        except ImportError:
            from prepare_legacy_model_manifest import validate_model_manifest
        validate_model_manifest(Path(args.model_name), model_manifest_data, verify_hashes=False)
        strict_data, strict_manifest = _load_strict_legacy_data(args.prepared_data_manifest)
        strict_protocol = canonical_training_protocol(
            method=args.rope_method,
            seed=args.seed,
            data_manifest_sha256=sha256_file(args.prepared_data_manifest),
            model_manifest_sha256=sha256_file(args.model_manifest),
            code_sha256=legacy_training_code_sha256(),
        )
        _ensure_immutable_protocol(Path(args.output_dir), strict_protocol)
        strict_runtime_packages = current_legacy_runtime_packages()
    dataset_identifier = training_dataset_identifier(args, strict_manifest)
    resume_checkpoint = resolve_legacy_resume_checkpoint(
        args.output_dir,
        args.resume_from_checkpoint,
        expected_protocol=strict_protocol,
        expected_runtime_packages=strict_runtime_packages,
    )
    if resume_checkpoint is not None:
        print(f"[RESUME] CPU preflight selected {resume_checkpoint}")

    # 1. Theoretical validation
    theory_checks = validate_theory(args)

    # 2. Compute EVQ-cosh frequencies
    inv_freq = compute_and_save_inv_freq(
        args,
        preserve_existing=args.strict_legacy_protocol,
    )

    if args.dry_run:
        print("\n[DRY RUN] Config validated. Exiting without training.")
        config = {
            "model": public_model_identifier(args.model_name),
            "rope_method": args.rope_method,
            "tau": args.tau,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_steps": args.max_steps,
            "max_seq_len": args.max_seq_len,
            "dataset": dataset_identifier,
            "theory": theory_checks,
        }
        config_path = os.path.join(args.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"[DRY RUN] Config saved to {config_path}")
        return

    # ---- Below requires GPU ----
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        TrainerCallback,
        BitsAndBytesConfig,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(args.seed)

    # 3. Load tokenizer
    print(f"\n[MODEL] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.model_name).is_dir(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.strict_legacy_protocol:
        try:
            from .prepare_positional_distill_data import tokenizer_source_fingerprint
        except ImportError:
            from prepare_positional_distill_data import tokenizer_source_fingerprint
        if strict_manifest["tokenizer"] != tokenizer_source_fingerprint(args.model_name):
            raise RuntimeError("runtime tokenizer does not match frozen legacy data")

    # 4. Load model
    precision = "4-bit QLoRA" if args.load_in_4bit else "bf16 full precision"
    print(f"[MODEL] Loading model ({precision})")
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "attn_implementation": "sdpa",
        "device_map": {"": 0} if args.strict_legacy_protocol else "auto",
        "local_files_only": Path(args.model_name).is_dir(),
    }
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    if args.strict_legacy_protocol:
        device_map = getattr(model, "hf_device_map", None)
        if not isinstance(device_map, dict) or not device_map:
            raise RuntimeError("strict legacy model lacks an explicit single-GPU device map")
        if any(str(device).lower() in {"cpu", "disk"} for device in device_map.values()):
            raise RuntimeError("strict legacy training forbids CPU or disk offload")
        if any(str(device).lower() not in {"0", "cuda", "cuda:0"} for device in device_map.values()):
            raise RuntimeError(f"strict legacy model escaped cuda:0: {device_map}")
    model.config.use_cache = False

    # 5. Inject the exact schedule recorded by this run.
    print(f"[ROPE] Injecting {args.rope_method} frequencies...")
    inject_result = inject_inv_freq(model, inv_freq)
    print(f"[ROPE] Patched {inject_result['patched_count']} modules: "
          f"{inject_result['changed_modules'][:3]}...")

    verification = verify_model_inv_freq(model, inv_freq)
    print(
        f"[ROPE] Injection verification: modules={verification['verified_count']}, "
        f"max_error={verification['max_error']:.2e}"
    )

    # 6. Prepare for LoRA
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [m.strip() for m in args.lora_targets.split(",")]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"[LORA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 7. Re-verify inv_freq not overwritten by PEFT
    post_peft = verify_model_inv_freq(model, inv_freq)
    print(
        f"[ROPE] Post-PEFT verification: modules={post_peft['verified_count']}, "
        f"max_error={post_peft['max_error']:.2e}"
    )

    # 8. Load data
    if args.strict_legacy_protocol:
        data = strict_data
    else:
        data = load_training_data(
            tokenizer=tokenizer,
            dataset_name=args.local_data_path or args.dataset_name,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            cache_dir=args.output_dir,
        )

    if args.strict_legacy_protocol:
        train_dataset = CompactTokenizedDataset(
            data["tokens"], data["offsets"], data["train_indices"], args.max_seq_len
        )
        val_dataset = CompactTokenizedDataset(
            data["tokens"], data["offsets"], data["validation_indices"], args.max_seq_len
        )
    else:
        train_dataset = TokenizedDataset(data["train"], args.max_seq_len)
        val_dataset = TokenizedDataset(data["val"], args.max_seq_len)

    # 9. Training
    training_kwargs = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "bf16": args.bf16,
        "fp16": not args.bf16,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps" if args.strict_legacy_protocol else "no",
        "save_steps": args.save_steps,
        "save_total_limit": 2 if args.strict_legacy_protocol else None,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "seed": args.seed,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
    }
    if args.strict_legacy_protocol:
        training_kwargs.update({
            "torch_compile": args.compile,
            "torch_compile_backend": "inductor",
            "torch_compile_mode": args.compile_mode,
        })
    training_kwargs.update(evaluation_strategy_kwargs(TrainingArguments))
    training_args = TrainingArguments(**training_kwargs)

    data_collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    callbacks = []
    if args.strict_legacy_protocol:
        class LegacyCheckpointReceiptCallback(TrainerCallback):
            def on_save(self, training_args, state, control, **kwargs):
                checkpoint_dir = Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
                write_legacy_checkpoint_receipt(
                    checkpoint_dir,
                    strict_protocol,
                    strict_runtime_packages,
                )
                return control

        callbacks.append(LegacyCheckpointReceiptCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print(f"\n{'=' * 60}")
    print(f"TRAINING START")
    print(f"  Model:     {args.model_name}")
    rope_label = (
        f"EVQ-cosh τ={args.tau}" if args.rope_method == "evq_cosh" else "native geometric"
    )
    print(f"  RoPE:      {rope_label}")
    print(f"  LoRA:      r={args.lora_r}, α={args.lora_alpha}")
    print(f"  Steps:     {args.max_steps}")
    print(f"  Seq len:   {args.max_seq_len}")
    print(f"  Data:      {dataset_identifier} ({len(train_dataset)} samples)")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    if resume_checkpoint is not None:
        print(f"[RESUME] Continuing from {resume_checkpoint}")
    trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    train_time = time.time() - t0

    print(f"\n[DONE] Training completed in {train_time/3600:.2f} hours")

    # 10. Save
    print("[SAVE] Saving adapter + custom inv_freq...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    # Save experiment metadata
    meta = {
        "model": public_model_identifier(args.model_name),
        "rope_method": args.rope_method,
        "tau": args.tau if args.rope_method == "evq_cosh" else None,
        "rope_base": args.rope_base,
        "head_dim": args.head_dim,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_targets": target_modules,
        "max_steps": args.max_steps,
        "max_seq_len": args.max_seq_len,
        "dataset": dataset_identifier,
        "train_samples": len(train_dataset),
        "train_time_hours": round(train_time / 3600, 3),
        "theory_checks": theory_checks,
        "train_loss_final": trainer.state.log_history[-1].get("train_loss")
                           if trainer.state.log_history else None,
    }
    if args.strict_legacy_protocol:
        adapter_path = Path(args.output_dir) / "adapter_model.safetensors"
        frequency_path = Path(args.output_dir) / "custom_inv_freq.pt"
        if trainer.state.global_step != 300:
            raise RuntimeError(
                f"strict legacy run ended at step {trainer.state.global_step}, expected 300"
            )
        meta.update({
            "objective": "legacy_longalign_full_token_causal_lm_v2",
            "status": "complete",
            "global_step": trainer.state.global_step,
            "protocol": strict_protocol,
            "protocol_sha256": canonical_json_sha256(strict_protocol),
            "adapter_sha256": sha256_file(adapter_path),
            "frequency_sha256": sha256_file(frequency_path),
            "data_manifest_sha256": sha256_file(args.prepared_data_manifest),
            "model_manifest_sha256": sha256_file(args.model_manifest),
            "code_sha256": legacy_training_code_sha256(),
            "runtime": legacy_runtime_identity(strict_runtime_packages),
        })
    meta_path = os.path.join(args.output_dir, "experiment_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[SAVE] Metadata saved to {meta_path}")
    print(f"[SAVE] Adapter saved to {args.output_dir}")
    print(f"\n✅ Training complete. Next: run eval_evq_lora.py --adapter_dir {args.output_dir}")


if __name__ == "__main__":
    main()
