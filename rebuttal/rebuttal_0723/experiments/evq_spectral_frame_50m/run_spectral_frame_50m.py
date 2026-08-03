#!/usr/bin/env python3
"""Run the matched 50M EVQ spectral-frame experiment.

This runner preserves the architecture, data order, optimizer semantics, and
evaluation offsets of ``run_50m_native_vs_evq_lerope.py`` while adding three
candidate frequency constructions and two attribution controls:

* bounded observed-band EVQ-LeRoPE;
* fixed integer-period coherence EVQ;
* fixed head-factorized Native/EVQ/integer-period RoPE;
* Native-LeRoPE;
* a within-head mixed-grid control with the same global frequency multiset as
  the head-factorized arm.

Preparation and CPU preflight are not experiment results. GPU modes require a
matching CPU-ready receipt; the full suite additionally requires a matching
runtime-ready receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[4]
LEGACY_RUNNER = (
    REPO_ROOT
    / "rebuttal/rebuttal_0723/experiments"
    / "run_50m_native_vs_evq_lerope.py"
)
SCHEDULE_PATH = REPO_ROOT / "scripts/lib/rope/schedules.py"

METHOD_ID = "evq_spectral_frame_50m_v2"
PREPARED_STATUS = "EVQ_SPECTRAL_FRAME_50M_PREPARED_NO_GPU_V2"
CPU_READY_STATUS = "EVQ_SPECTRAL_FRAME_50M_CPU_READY_V2"
RUNTIME_READY_STATUS = "EVQ_SPECTRAL_FRAME_50M_RUNTIME_READY_V2"
RESULT_STATUS = "EVQ_SPECTRAL_FRAME_50M_COMPLETE_V2"

ARMS = (
    "native",
    "evq_fixed",
    "native_lerope",
    "evq_lerope",
    "phase_observed_evq",
    "integer_period_coherence_evq",
    "head_factorized",
    "within_head_mixed_control",
)
SEEDS = (42, 43, 44)
REGISTERED_RESONANT_PERIODS = (
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    20,
    25,
    30,
    32,
    36,
    42,
    54,
    60,
    74,
    89,
    119,
)


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


legacy = load_module("evq_lerope_legacy_reference", LEGACY_RUNNER)
core = load_module("evq_lerope_spectral_core", LEGACY_RUNNER)
Protocol = core.Protocol


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def ordinary_parameter_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if name.startswith("rope."):
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def kernel_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    train_length: int,
) -> dict[str, float]:
    candidate = candidate.detach().double().cpu()
    reference = reference.detach().double().cpu()
    local = torch.arange(1, train_length, dtype=torch.float64)
    external = torch.arange(
        train_length, 4 * train_length + 1, dtype=torch.float64
    )

    def kernel(frequencies: torch.Tensor, distance: torch.Tensor):
        return torch.cos(
            distance[:, None] * frequencies[None, :]
        ).mean(dim=1)

    local_candidate = kernel(candidate, local)
    local_reference = kernel(reference, local)
    external_candidate = kernel(candidate, external)
    return {
        "local_mse_to_evq": float(
            torch.mean((local_candidate - local_reference) ** 2)
        ),
        "external_mean_squared_coherence": float(
            torch.mean(external_candidate**2)
        ),
        "external_max_abs_coherence": float(
            torch.max(torch.abs(external_candidate))
        ),
    }


def rounded_unique_periods(
    evq_wavelengths: torch.Tensor,
    observed_count: int,
) -> list[int]:
    periods: list[int] = []
    used: set[int] = set()
    for value in evq_wavelengths[:observed_count].tolist():
        center = max(2, int(round(float(value))))
        if center in used:
            for radius in range(1, 256):
                options = [
                    item
                    for item in (center - radius, center + radius)
                    if item >= 2 and item not in used
                ]
                if options:
                    center = min(
                        options, key=lambda item: abs(item - value)
                    )
                    break
        used.add(center)
        periods.append(center)
    if any(left >= right for left, right in zip(periods, periods[1:])):
        raise RuntimeError("initial resonant periods are not increasing")
    return periods


def resonant_objective(
    periods: list[int],
    evq_frequencies: torch.Tensor,
    observed_count: int,
    train_length: int,
) -> tuple[float, dict[str, float], torch.Tensor]:
    evq = evq_frequencies.detach().double().cpu()
    evq_wavelengths = 2.0 * math.pi / evq
    wavelengths = torch.cat(
        (
            torch.tensor(periods, dtype=torch.float64),
            evq_wavelengths[observed_count:],
        )
    )
    frequencies = 2.0 * math.pi / wavelengths
    metrics = kernel_metrics(frequencies, evq, train_length)
    score = (
        metrics["external_max_abs_coherence"]
        + 0.5 * metrics["external_mean_squared_coherence"]
        + 2.0 * metrics["local_mse_to_evq"]
    )
    return float(score), metrics, frequencies


def build_resonant_design(
    protocol: Protocol, *, verify_search: bool
) -> tuple[torch.Tensor, dict[str, Any]]:
    evq = core.evq_inv_freq(protocol).double()
    wavelengths = 2.0 * math.pi / evq
    observed_count = int(torch.sum(wavelengths <= protocol.train_length))
    if observed_count != len(REGISTERED_RESONANT_PERIODS):
        raise RuntimeError("resonant observed-band count drift")
    initial_periods = rounded_unique_periods(wavelengths, observed_count)
    initial_score, initial_metrics, _ = resonant_objective(
        initial_periods, evq, observed_count, protocol.train_length
    )
    periods = initial_periods[:]
    score = initial_score
    metrics = initial_metrics
    passes = 0
    if verify_search:
        for pass_index in range(20):
            changed = False
            for index in range(observed_count):
                lower = periods[index - 1] + 1 if index else 5
                upper = (
                    periods[index + 1] - 1
                    if index < observed_count - 1
                    else protocol.train_length
                )
                target = float(wavelengths[index])
                best_score = score
                best_period = periods[index]
                best_metrics = metrics
                for candidate in range(lower, upper + 1):
                    if abs(candidate - round(target)) > 16:
                        continue
                    proposed = periods[:]
                    proposed[index] = candidate
                    value, value_metrics, _ = resonant_objective(
                        proposed,
                        evq,
                        observed_count,
                        protocol.train_length,
                    )
                    if value < best_score - 1e-12:
                        best_score = value
                        best_period = candidate
                        best_metrics = value_metrics
                if best_period != periods[index]:
                    periods[index] = best_period
                    score = best_score
                    metrics = best_metrics
                    changed = True
            passes = pass_index + 1
            if not changed:
                break
        if tuple(periods) != REGISTERED_RESONANT_PERIODS:
            raise RuntimeError(
                "deterministic resonant coordinate-descent result drift"
            )
    else:
        periods = list(REGISTERED_RESONANT_PERIODS)
    final_score, final_metrics, final_frequency = resonant_objective(
        periods, evq, observed_count, protocol.train_length
    )
    evq_metrics = kernel_metrics(evq, evq, protocol.train_length)
    if not bool(torch.all(final_frequency[:-1] > final_frequency[1:])):
        raise RuntimeError("resonant grid is not strictly decreasing")
    if (
        final_metrics["local_mse_to_evq"] >= 0.02
        or final_metrics["external_mean_squared_coherence"]
        >= evq_metrics["external_mean_squared_coherence"]
        or final_metrics["external_max_abs_coherence"]
        >= evq_metrics["external_max_abs_coherence"]
    ):
        raise RuntimeError("registered resonant design gate failed")
    design = {
        "algorithm": "deterministic_integer_period_coordinate_descent_v1",
        "design_range": [
            protocol.train_length,
            4 * protocol.train_length,
        ],
        "range_boundary": (
            "registered once for the 4x study; not target-range agnostic"
        ),
        "objective": (
            "external_max_abs + 0.5*external_mean_squared "
            "+ 2.0*local_mse_to_evq"
        ),
        "observed_count": observed_count,
        "initial_periods": initial_periods,
        "registered_periods": periods,
        "search_passes": passes,
        "initial_objective": initial_score,
        "final_objective": final_score,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "fixed_evq_metrics": evq_metrics,
        "frequency_sha256_float32": tensor_sha256(
            final_frequency.float()
        ),
    }
    return final_frequency.float(), design


class SpectralRotaryEmbedding(nn.Module):
    """Shared or head-specific fixed/learned spectral allocation."""

    def __init__(self, protocol: Protocol, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        self.protocol = protocol
        self.arm = arm
        native = core.native_inv_freq(protocol)
        evq = core.evq_inv_freq(protocol)
        resonant, _ = build_resonant_design(
            protocol, verify_search=False
        )
        heads = protocol.num_heads
        pairs = protocol.head_dim // 2
        if arm in ("native", "native_lerope"):
            head_base = native[None, :].expand(heads, pairs).clone()
        elif arm in (
            "evq_fixed",
            "evq_lerope",
            "phase_observed_evq",
        ):
            head_base = evq[None, :].expand(heads, pairs).clone()
        elif arm == "integer_period_coherence_evq":
            head_base = resonant[None, :].expand(heads, pairs).clone()
        elif arm == "head_factorized":
            head_base = torch.stack(
                (
                    native,
                    native,
                    native,
                    native,
                    evq,
                    evq,
                    resonant,
                    resonant,
                )
            )
        else:
            source = torch.stack(
                (
                    native,
                    native,
                    native,
                    native,
                    evq,
                    evq,
                    resonant,
                    resonant,
                )
            )
            head_base = torch.stack(
                [
                    torch.sort(
                        torch.stack(
                            [
                                source[(head + band) % heads, band]
                                for band in range(pairs)
                            ]
                        ),
                        descending=True,
                    ).values
                    for head in range(heads)
                ]
            )
            factorized_multiset = torch.sort(source.flatten()).values
            mixed_multiset = torch.sort(head_base.flatten()).values
            if not torch.equal(factorized_multiset, mixed_multiset):
                raise RuntimeError(
                    "within-head control changed global frequency multiset"
                )
        if head_base.shape != (heads, pairs):
            raise RuntimeError("head-factorized allocation shape drift")
        self.register_buffer(
            "base_head_inv_freq", head_base.float(), persistent=True
        )
        rotations = (
            protocol.train_length * evq / (2.0 * math.pi)
        )
        if arm in ("native_lerope", "evq_lerope"):
            band_mask = torch.ones_like(evq)
        elif arm == "phase_observed_evq":
            band_mask = (rotations >= 1.0).to(evq.dtype)
        else:
            band_mask = torch.zeros_like(evq)
        self.register_buffer(
            "learnable_mask", band_mask.float(), persistent=True
        )
        self.log_frequency_scale = nn.Parameter(
            torch.zeros_like(evq, dtype=torch.float32),
            requires_grad=bool(band_mask.any()),
        )
        maximum_log_shift = torch.full_like(evq, float("inf"))
        evq_log = torch.log(evq)
        gaps = evq_log[:-1] - evq_log[1:]
        maximum_log_shift[0] = gaps[0]
        maximum_log_shift[-1] = gaps[-1]
        maximum_log_shift[1:-1] = torch.minimum(gaps[:-1], gaps[1:])
        maximum_log_shift = 0.45 * maximum_log_shift
        if arm != "phase_observed_evq":
            maximum_log_shift.zero_()
        self.register_buffer(
            "maximum_log_shift",
            maximum_log_shift.float(),
            persistent=True,
        )
        target_wavelength = (
            protocol.dominant_wavelength_ratio * protocol.train_length
        )
        wavelengths = 2.0 * math.pi / evq.double()
        self.selected_band = int(
            torch.argmin(
                torch.abs(torch.log(wavelengths / target_wavelength))
            )
        )

    def current_head_inv_freq(self) -> torch.Tensor:
        if self.arm == "phase_observed_evq":
            log_shift = (
                self.learnable_mask
                * self.maximum_log_shift
                * torch.tanh(self.log_frequency_scale)
            )
        else:
            log_shift = (
                self.learnable_mask * self.log_frequency_scale
            )
        scale = torch.exp(log_shift)
        return self.base_head_inv_freq * scale[None, :]

    def current_inv_freq(self) -> torch.Tensor:
        return self.current_head_inv_freq()[0]

    def frequency_parameters(self) -> list[nn.Parameter]:
        if self.log_frequency_scale.requires_grad:
            return [self.log_frequency_scale]
        return []

    def validate_current(self) -> None:
        values = self.current_head_inv_freq().detach()
        if not torch.isfinite(values).all():
            raise RuntimeError(f"{self.arm} contains non-finite frequency")
        if not bool(torch.all(values[:, :-1] > values[:, 1:])):
            raise RuntimeError(
                f"{self.arm} frequency ordering failed within a head"
            )

    def forward(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            sequence_length, device=device, dtype=torch.float32
        )
        frequencies = (
            positions[:, None, None]
            * self.current_head_inv_freq()
            .to(device=device, dtype=torch.float32)[None, :, :]
        ).permute(1, 0, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


class HeadAwareAttention(nn.Module):
    """Legacy attention with explicit per-head RoPE broadcasting."""

    def __init__(self, protocol: Protocol) -> None:
        super().__init__()
        self.protocol = protocol
        self.qkv = nn.Linear(
            protocol.hidden_size,
            3 * protocol.hidden_size,
            bias=False,
        )
        self.output = nn.Linear(
            protocol.hidden_size,
            protocol.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        packed = self.qkv(hidden).view(
            batch,
            length,
            3,
            self.protocol.num_heads,
            self.protocol.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = packed.unbind(0)
        if cosine.ndim != 3 or sine.ndim != 3:
            raise RuntimeError("spectral RoPE must be [heads,length,dim]")
        cosine = cosine[None, :, :, :]
        sine = sine[None, :, :, :]
        query = core.apply_rope(query, cosine, sine)
        key = core.apply_rope(key, cosine, sine)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        return self.output(
            attended.transpose(1, 2).reshape(batch, length, -1)
        )


core.SelectiveRotaryEmbedding = SpectralRotaryEmbedding
core.Attention = HeadAwareAttention


def build_model(protocol: Protocol, arm: str) -> nn.Module:
    core.set_seed(protocol.seed)
    model = core.GPT(protocol, arm)
    model.rope.validate_current()
    return model


def arm_frequency_receipt(
    model: nn.Module, protocol: Protocol
) -> dict[str, Any]:
    values = model.rope.current_head_inv_freq().detach().cpu()
    unique_head_hashes = [
        tensor_sha256(row.float()) for row in values
    ]
    return {
        "shape": list(values.shape),
        "values_float32": values.float().tolist(),
        "sha256_float32": tensor_sha256(values.float()),
        "per_head_sha256_float32": unique_head_hashes,
        "strictly_decreasing_per_head": bool(
            torch.all(values[:, :-1] > values[:, 1:])
        ),
        "learnable_bands": [
            int(index)
            for index in torch.nonzero(
                model.rope.learnable_mask, as_tuple=False
            ).flatten()
        ],
        "frozen_bands": [
            int(index)
            for index in torch.nonzero(
                model.rope.learnable_mask == 0, as_tuple=False
            ).flatten()
        ],
        "selected_band": int(model.rope.selected_band),
        "selected_initial_wavelength": float(
            2.0
            * math.pi
            / model.rope.base_head_inv_freq[
                0, model.rope.selected_band
            ]
        ),
        "parameterization": (
            "bounded_tanh_log_residual_with_0.45_adjacent_gap_radius"
            if model.rope.arm == "phase_observed_evq"
            else (
                "unbounded_shared_log_residual"
                if model.rope.log_frequency_scale.requires_grad
                else "fixed"
            )
        ),
        "maximum_log_shift": (
            model.rope.maximum_log_shift.detach().cpu().tolist()
        ),
        "head_assignment": (
            [
                "native",
                "native",
                "native",
                "native",
                "evq",
                "evq",
                "resonant",
                "resonant",
            ]
            if model.rope.arm == "head_factorized"
            else (
                ["cyclic_within_head_mixture"] * protocol.num_heads
                if model.rope.arm == "within_head_mixed_control"
                else [model.rope.arm] * protocol.num_heads
            )
        ),
    }


def configure_cuda() -> dict[str, Any]:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "compiled_architectures": torch.cuda.get_arch_list(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp_enabled": (
            torch.backends.cuda.cudnn_sdp_enabled()
            if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
            else None
        ),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    }


def contract(
    protocol: Protocol,
    train_data: Path,
    validation_data: Path,
) -> dict[str, Any]:
    protocol.validate()
    resonant, resonant_design = build_resonant_design(
        protocol, verify_search=False
    )
    native = core.native_inv_freq(protocol)
    evq = core.evq_inv_freq(protocol)
    protocol_values = asdict(protocol)
    protocol_values["eval_lengths"] = list(protocol.eval_lengths)
    return {
        "method_id": METHOD_ID,
        "classification": "REGISTERED_DESIGN_NOT_RESULT",
        "protocol": protocol_values,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "data": {
            "train_path": str(train_data.resolve()),
            "train_sha256": file_sha256(train_data.resolve()),
            "validation_path": str(validation_data.resolve()),
            "validation_sha256": file_sha256(
                validation_data.resolve()
            ),
        },
        "frequency_identity": {
            "native_sha256_float32": tensor_sha256(native.float()),
            "evq_sha256_float32": tensor_sha256(evq.float()),
            "integer_period_sha256_float32": tensor_sha256(
                resonant.float()
            ),
            "integer_period_design": resonant_design,
        },
        "code": {
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "legacy_runner_path": str(LEGACY_RUNNER.resolve()),
            "legacy_runner_sha256": file_sha256(
                LEGACY_RUNNER.resolve()
            ),
            "schedule_path": str(SCHEDULE_PATH.resolve()),
            "schedule_sha256": file_sha256(SCHEDULE_PATH.resolve()),
        },
        "derived": {
            "train_rows": protocol.train_rows,
            "optimizer_steps": protocol.optimizer_steps,
            "processed_storage_tokens": (
                protocol.train_rows * protocol.train_length
            ),
            "supervised_next_tokens": (
                protocol.train_rows * (protocol.train_length - 1)
            ),
        },
    }


def load_and_validate_data(
    train_data: Path,
    validation_data: Path,
    protocol: Protocol,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, list[int]]]:
    rows = core.load_training_rows(train_data, protocol)
    validation = core.load_validation_tokens(validation_data)
    offsets = core.evaluation_offsets(validation.numel(), protocol)
    return rows, validation, offsets


def verify_receipt(
    path: Path, status: str, expected_contract: dict[str, Any]
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != status
        or payload.get("contract") != expected_contract
    ):
        raise RuntimeError(f"{path} contract drift")
    return payload


def parity_preflight(
    protocol: Protocol,
    rows: torch.Tensor,
    validation: torch.Tensor,
    offsets: dict[int, list[int]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    _, registered_design_verification = build_resonant_design(
        protocol, verify_search=True
    )
    legacy_parity_arms = {
        "native",
        "evq_fixed",
        "native_lerope",
        "evq_lerope",
    }
    sample = rows[:2, :16]
    ordinary_hashes: dict[str, str] = {}
    for arm in ARMS:
        model = build_model(protocol, arm)
        ordinary_hashes[arm] = ordinary_parameter_sha256(model)
        logits = model(sample[:, :-1])
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{arm} CPU forward is non-finite")
        loss = F.cross_entropy(
            logits.reshape(-1, protocol.vocab_size),
            sample[:, 1:].reshape(-1),
        )
        loss.backward()
        loss_value = float(loss.detach())
        frequency_gradient = None
        if model.rope.log_frequency_scale.requires_grad:
            gradient = model.rope.log_frequency_scale.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"{arm} frequency gradient invalid")
            learned = model.rope.learnable_mask.bool()
            if float(torch.linalg.vector_norm(gradient[learned])) == 0.0:
                raise RuntimeError(f"{arm} learned gradient is zero")
            frozen_max = float(
                gradient[~learned].abs().max()
                if bool((~learned).any())
                else torch.tensor(0.0)
            )
            if frozen_max != 0.0:
                raise RuntimeError(f"{arm} frozen gradient is non-zero")
            frequency_gradient = {
                "learned_l2": float(
                    torch.linalg.vector_norm(gradient[learned])
                ),
                "frozen_max_abs": frozen_max,
            }
        del logits, loss
        model.zero_grad(set_to_none=True)
        parity = None
        if arm in legacy_parity_arms:
            core.set_seed(protocol.seed)
            reference = legacy.GPT(protocol, arm)
            reference_hash = legacy.trainable_state_sha256(reference)
            if ordinary_hashes[arm] != reference_hash:
                raise RuntimeError(
                    f"{arm} ordinary initialization differs from legacy"
                )
            with torch.no_grad():
                reference_logits = reference(sample[:, :-1])
                current_logits = model(sample[:, :-1])
            difference = (current_logits - reference_logits).abs()
            parity = {
                "reference_ordinary_sha256": reference_hash,
                "max_abs_logit_difference": float(difference.max()),
                "bitwise_equal": bool(
                    torch.equal(current_logits, reference_logits)
                ),
            }
            if not parity["bitwise_equal"]:
                raise RuntimeError(f"{arm} legacy logit parity failed")
            parity_offsets = {
                length: starts[:2] for length, starts in offsets.items()
            }
            current_evaluation = evaluate(
                model=model,
                validation=validation,
                offsets=parity_offsets,
                device=torch.device("cpu"),
            )
            reference_evaluation = legacy.evaluate_arm(
                model=reference,
                validation_tokens=validation,
                offsets=parity_offsets,
                device=torch.device("cpu"),
            )
            evaluation_differences = {
                str(length): abs(
                    current_evaluation[str(length)]["mean_nll"]
                    - reference_evaluation[str(length)]["mean_nll"]
                )
                for length in parity_offsets
            }
            parity["evaluation_mean_nll_abs_difference"] = (
                evaluation_differences
            )
            if max(evaluation_differences.values()) > 3e-6:
                raise RuntimeError(
                    f"{arm} batched evaluation parity failed"
                )
            del reference
        checks[arm] = {
            "ordinary_initialization_sha256": ordinary_hashes[arm],
            "loss": loss_value,
            "frequency": arm_frequency_receipt(model, protocol),
            "frequency_gradient": frequency_gradient,
            "legacy_parity": parity,
        }
        del model
    if len(set(ordinary_hashes.values())) != 1:
        raise RuntimeError("ordinary model initialization differs across arms")
    observed = checks["phase_observed_evq"]["frequency"]
    if (
        observed["learnable_bands"] != list(range(22))
        or observed["frozen_bands"] != list(range(22, 32))
    ):
        raise RuntimeError("phase-observed band partition drift")
    factorized = torch.tensor(
        checks["head_factorized"]["frequency"]["values_float32"]
    )
    mixed = torch.tensor(
        checks["within_head_mixed_control"]["frequency"][
            "values_float32"
        ]
    )
    if not torch.equal(
        torch.sort(factorized.flatten()).values,
        torch.sort(mixed.flatten()).values,
    ):
        raise RuntimeError("head-control global multiset parity failed")
    return {
        "ordinary_initialization_sha256": next(
            iter(ordinary_hashes.values())
        ),
        "registered_integer_period_design_verification": (
            registered_design_verification
        ),
        "checks": checks,
    }


def make_optimizer(
    model: nn.Module,
    protocol: Protocol,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, list[nn.Parameter], list[nn.Parameter]]:
    frequency = model.rope.frequency_parameters()
    frequency_ids = {id(parameter) for parameter in frequency}
    ordinary = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in frequency_ids
    ]
    groups: list[dict[str, Any]] = [
        {
            "params": ordinary,
            "weight_decay": protocol.weight_decay,
            "lr_scale": 1.0,
        }
    ]
    if frequency:
        groups.append(
            {
                "params": frequency,
                "weight_decay": 0.0,
                "lr_scale": protocol.frequency_lr_multiplier,
            }
        )
    optimizer = torch.optim.AdamW(
        groups,
        lr=protocol.learning_rate,
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )
    return optimizer, ordinary, frequency


def compiled_forward_model(
    model: nn.Module, compile_mode: str
) -> nn.Module:
    if compile_mode == "none":
        return model
    return torch.compile(
        model,
        mode=compile_mode,
        dynamic=False,
        fullgraph=False,
    )


def train_one(
    *,
    arm: str,
    protocol: Protocol,
    rows: torch.Tensor,
    device: torch.device,
    micro_batch: int,
    compile_mode: str,
    output: Path,
    max_steps: int | None = None,
    save_checkpoint: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    if protocol.global_batch_sequences % micro_batch:
        raise RuntimeError("micro-batch must divide global batch")
    model = build_model(protocol, arm).to(device)
    initial_weight_hash = ordinary_parameter_sha256(model)
    initial_frequency = arm_frequency_receipt(model, protocol)
    optimizer, ordinary, frequency = make_optimizer(
        model, protocol, device
    )
    forward_model = compiled_forward_model(model, compile_mode)
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(protocol.seed + 1_000_003)
    permutation = torch.randperm(
        protocol.train_rows, generator=permutation_generator
    )
    accumulation = protocol.global_batch_sequences // micro_batch
    steps = protocol.optimizer_steps
    if max_steps is not None:
        steps = min(steps, max_steps)
    output.mkdir(parents=True, exist_ok=False)
    losses: list[float] = []
    step_seconds: list[float] = []
    measure_individual_steps = max_steps is not None
    first_gradient = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        if device.type == "cuda" and measure_individual_steps:
            torch.cuda.synchronize()
        step_started = time.perf_counter()
        learning_rate = core.learning_rate_at(step, protocol)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * float(group["lr_scale"])
        optimizer.zero_grad(set_to_none=True)
        indices = permutation[
            (step - 1) * protocol.global_batch_sequences :
            step * protocol.global_batch_sequences
        ]
        step_losses: list[float] = []
        for slot in range(accumulation):
            selection = indices[
                slot * micro_batch : (slot + 1) * micro_batch
            ]
            batch = rows[selection].to(device, non_blocking=True)
            with core.autocast_context(device):
                logits = forward_model(batch[:, :-1])
                raw_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch[:, 1:].reshape(-1),
                )
                loss = raw_loss / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"{arm} non-finite loss at {step}")
            loss.backward()
            step_losses.append(float(raw_loss.detach().cpu()))
        torch.nn.utils.clip_grad_norm_(
            ordinary, protocol.gradient_clip
        )
        if frequency:
            gradient = frequency[0].grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"{arm} frequency gradient invalid")
            if step == 1:
                learned = model.rope.learnable_mask.bool()
                learned_l2 = float(
                    torch.linalg.vector_norm(gradient[learned])
                )
                frozen_max = float(
                    gradient[~learned].abs().max()
                    if bool((~learned).any())
                    else torch.tensor(0.0, device=gradient.device)
                )
                if learned_l2 == 0.0 or frozen_max != 0.0:
                    raise RuntimeError(
                        f"{arm} learned/frozen gradient gate failed"
                    )
                first_gradient = {
                    "l2": float(torch.linalg.vector_norm(gradient)),
                    "max_abs": float(gradient.abs().max()),
                    "learned_l2": learned_l2,
                    "frozen_max_abs": frozen_max,
                    "per_band": gradient.detach().cpu().tolist(),
                }
            torch.nn.utils.clip_grad_norm_(
                frequency, protocol.frequency_gradient_clip
            )
        optimizer.step()
        model.rope.validate_current()
        if device.type == "cuda" and measure_individual_steps:
            torch.cuda.synchronize()
        if measure_individual_steps:
            step_seconds.append(time.perf_counter() - step_started)
        mean_loss = float(np.mean(step_losses))
        losses.append(mean_loss)
        if step == 1 or step % 50 == 0 or step == steps:
            elapsed = time.perf_counter() - started
            record = {
                "arm": arm,
                "seed": protocol.seed,
                "step": step,
                "steps": steps,
                "loss": mean_loss,
                "mean_loss_last_20": float(np.mean(losses[-20:])),
                "learning_rate": learning_rate,
                "elapsed_seconds": elapsed,
                "processed_tokens": (
                    step
                    * protocol.global_batch_sequences
                    * protocol.train_length
                ),
                "tokens_per_second": (
                    step
                    * protocol.global_batch_sequences
                    * protocol.train_length
                    / elapsed
                ),
            }
            with (output / "train_log.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    checkpoint_path = None
    checkpoint_hash = None
    if save_checkpoint:
        checkpoint = output / "model.pt"
        torch.save(model.state_dict(), checkpoint)
        checkpoint_path = checkpoint.name
        checkpoint_hash = file_sha256(checkpoint)
    steady_steps = (
        step_seconds[1:] if len(step_seconds) > 1 else step_seconds
    )
    receipt = {
        "arm": arm,
        "seed": protocol.seed,
        "initial_ordinary_weight_sha256": initial_weight_hash,
        "initial_frequency": initial_frequency,
        "final_frequency": arm_frequency_receipt(model, protocol),
        "frequency_log_scale": (
            model.rope.log_frequency_scale.detach().cpu().tolist()
        ),
        "first_frequency_gradient": first_gradient,
        "optimizer": {
            "name": "AdamW",
            "fused": device.type == "cuda",
            "betas": [0.9, 0.95],
        },
        "micro_batch": micro_batch,
        "accumulation": accumulation,
        "global_batch": protocol.global_batch_sequences,
        "compile_mode": compile_mode,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "first_step_seconds": (
            step_seconds[0] if step_seconds else None
        ),
        "steady_step_seconds_mean": (
            float(np.mean(steady_steps)) if steady_steps else None
        ),
        "step_seconds": step_seconds,
        "estimated_registered_run_seconds": (
            step_seconds[0]
            + max(0, protocol.optimizer_steps - 1)
            * float(np.mean(steady_steps))
            if step_seconds
            else None
        ),
        "processed_tokens": (
            steps
            * protocol.global_batch_sequences
            * protocol.train_length
        ),
        "tokens_per_second": (
            steps
            * protocol.global_batch_sequences
            * protocol.train_length
            / elapsed
        ),
        "final_mean_loss_last_20": float(np.mean(losses[-20:])),
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.type == "cuda"
            else None
        ),
        "peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved())
            if device.type == "cuda"
            else None
        ),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_hash,
    }
    return model, receipt


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    validation: torch.Tensor,
    offsets: dict[int, list[int]],
    device: torch.device,
    token_batch_budget: int = 4096,
) -> dict[str, Any]:
    model.eval()
    result: dict[str, Any] = {}
    for length, starts in offsets.items():
        batch_size = max(
            1, min(len(starts), token_batch_budget // length)
        )
        losses: list[float] = []
        suffix_losses: list[float] = []
        suffix_target_tokens = min(
            model.protocol.train_length, length - 1
        )
        for index in range(0, len(starts), batch_size):
            chunk_starts = starts[index : index + batch_size]
            batch = torch.stack(
                [
                    validation[start : start + length]
                    for start in chunk_starts
                ]
            ).to(device, non_blocking=True)
            with core.autocast_context(device):
                logits = model(batch[:, :-1])
                token_loss_matrix = F.cross_entropy(
                    logits.transpose(1, 2),
                    batch[:, 1:],
                    reduction="none",
                )
                token_losses = token_loss_matrix.mean(dim=1)
                suffix_token_losses = token_loss_matrix[
                    :, -suffix_target_tokens:
                ].mean(dim=1)
            losses.extend(
                float(value) for value in token_losses.detach().cpu()
            )
            suffix_losses.extend(
                float(value)
                for value in suffix_token_losses.detach().cpu()
            )
        mean_nll = float(np.mean(losses))
        suffix_mean_nll = float(np.mean(suffix_losses))
        result[str(length)] = {
            "chunks": len(losses),
            "offsets": starts,
            "per_chunk_nll": losses,
            "mean_nll": mean_nll,
            "ppl": float(math.exp(mean_nll)),
            "suffix_target_tokens": suffix_target_tokens,
            "per_chunk_suffix_nll": suffix_losses,
            "suffix_mean_nll": suffix_mean_nll,
            "suffix_ppl": float(math.exp(suffix_mean_nll)),
            "evaluation_batch_size": batch_size,
        }
    return result


def paired_bootstrap_summary(
    candidate: list[float],
    native: list[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, Any]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        native, dtype=np.float64
    )
    if difference.ndim != 1 or difference.size == 0:
        raise RuntimeError("paired bootstrap requires non-empty vectors")
    generator = np.random.default_rng(seed)
    selections = generator.integers(
        0, difference.size, size=(samples, difference.size)
    )
    bootstrap_means = difference[selections].mean(axis=1)
    return {
        "unit": "paired_fixed_validation_chunk_within_seed",
        "samples": samples,
        "observed_mean": float(difference.mean()),
        "ci95_percentile": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "bootstrap_probability_mean_below_zero": float(
            np.mean(bootstrap_means < 0.0)
        ),
    }


def validate_frequency_receipt(
    receipt: dict[str, Any],
    *,
    arm: str,
    initial: dict[str, Any] | None,
) -> torch.Tensor:
    values = torch.tensor(receipt["values_float32"], dtype=torch.float32)
    if list(values.shape) != [8, 32]:
        raise RuntimeError(f"{arm} frequency shape drift")
    if tensor_sha256(values) != receipt["sha256_float32"]:
        raise RuntimeError(f"{arm} frequency hash drift")
    if not torch.isfinite(values).all():
        raise RuntimeError(f"{arm} frequency is non-finite")
    decreasing = bool(torch.all(values[:, :-1] > values[:, 1:]))
    if not decreasing or not receipt["strictly_decreasing_per_head"]:
        raise RuntimeError(f"{arm} frequency ordering drift")
    shared_arms = {
        "native",
        "evq_fixed",
        "native_lerope",
        "evq_lerope",
        "phase_observed_evq",
        "integer_period_coherence_evq",
    }
    if arm in shared_arms and not all(
        torch.equal(values[0], values[head])
        for head in range(1, values.shape[0])
    ):
        raise RuntimeError(f"{arm} unexpectedly differs across heads")
    if initial is not None:
        initial_values = torch.tensor(
            initial["values_float32"], dtype=torch.float32
        )
        fixed_arms = {
            "native",
            "evq_fixed",
            "integer_period_coherence_evq",
            "head_factorized",
            "within_head_mixed_control",
        }
        if arm in fixed_arms and not torch.equal(values, initial_values):
            raise RuntimeError(f"{arm} fixed frequency changed")
        if arm == "phase_observed_evq":
            if receipt["learnable_bands"] != list(range(22)):
                raise RuntimeError("phase-observed learned bands drift")
            if receipt["frozen_bands"] != list(range(22, 32)):
                raise RuntimeError("phase-observed frozen bands drift")
            if not torch.equal(values[:, 22:], initial_values[:, 22:]):
                raise RuntimeError("phase-observed frozen values changed")
    return values


def validate_result_payload(
    *,
    result_path: Path,
    payload: dict[str, Any],
    expected_contract: dict[str, Any],
    offsets: dict[int, list[int]],
    expected_runtime_hash: str,
    arm: str,
    seed: int,
) -> None:
    contract_hash = canonical_json_sha256(expected_contract)
    if (
        payload.get("status") != RESULT_STATUS
        or payload.get("contract_sha256") != contract_hash
        or payload.get("runtime_ready_receipt_sha256")
        != expected_runtime_hash
        or payload.get("arm") != arm
        or payload.get("seed") != seed
    ):
        raise RuntimeError(f"{result_path} result identity drift")
    protocol = expected_contract["protocol"]
    training = payload["training"]
    expected_tokens = (
        int(protocol["optimizer_steps"])
        * int(protocol["global_batch_sequences"])
        * int(protocol["train_length"])
    )
    if (
        training["arm"] != arm
        or int(training["seed"]) != seed
        or int(training["steps"]) != int(protocol["optimizer_steps"])
        or int(training["processed_tokens"]) != expected_tokens
        or int(training["global_batch"])
        != int(protocol["global_batch_sequences"])
        or int(training["micro_batch"])
        * int(training["accumulation"])
        != int(protocol["global_batch_sequences"])
    ):
        raise RuntimeError(f"{result_path} training contract drift")
    checkpoint_name = training.get("checkpoint")
    if checkpoint_name != "model.pt":
        raise RuntimeError(f"{result_path} checkpoint identity drift")
    checkpoint_path = result_path.parent / checkpoint_name
    if (
        not checkpoint_path.is_file()
        or file_sha256(checkpoint_path)
        != training.get("checkpoint_sha256")
    ):
        raise RuntimeError(f"{result_path} checkpoint hash drift")
    validate_frequency_receipt(
        training["initial_frequency"], arm=arm, initial=None
    )
    validate_frequency_receipt(
        training["final_frequency"],
        arm=arm,
        initial=training["initial_frequency"],
    )
    evaluation = payload["evaluation"]
    if set(evaluation) != {"128", "256", "512"}:
        raise RuntimeError(f"{result_path} evaluation keys drift")
    for length, expected_offsets in offsets.items():
        cell = evaluation[str(length)]
        actual_offsets = [int(value) for value in cell["offsets"]]
        if (
            actual_offsets != expected_offsets
            or len(set(actual_offsets)) != len(expected_offsets)
            or int(cell["chunks"]) != len(expected_offsets)
        ):
            raise RuntimeError(f"{result_path} offsets drift at {length}")
        losses = np.asarray(cell["per_chunk_nll"], dtype=np.float64)
        suffix = np.asarray(
            cell["per_chunk_suffix_nll"], dtype=np.float64
        )
        if (
            losses.size != len(expected_offsets)
            or suffix.size != len(expected_offsets)
            or not np.isfinite(losses).all()
            or not np.isfinite(suffix).all()
        ):
            raise RuntimeError(f"{result_path} raw losses invalid")
        mean_nll = float(losses.mean())
        suffix_mean = float(suffix.mean())
        expected_suffix = min(int(protocol["train_length"]), length - 1)
        checks = (
            abs(float(cell["mean_nll"]) - mean_nll),
            abs(float(cell["ppl"]) - math.exp(mean_nll)),
            abs(float(cell["suffix_mean_nll"]) - suffix_mean),
            abs(float(cell["suffix_ppl"]) - math.exp(suffix_mean)),
        )
        if max(checks) > 1e-10 or int(
            cell["suffix_target_tokens"]
        ) != expected_suffix:
            raise RuntimeError(
                f"{result_path} aggregate loss drift at {length}"
            )


def aggregate_results(
    output: Path,
    expected_contract: dict[str, Any],
    offsets: dict[int, list[int]],
    expected_runtime_hash: str,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for seed in SEEDS:
        seed_key = str(seed)
        rows[seed_key] = {}
        for arm in ARMS:
            result_path = output / f"seed{seed}" / arm / "results.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            validate_result_payload(
                result_path=result_path,
                payload=payload,
                expected_contract=expected_contract,
                offsets=offsets,
                expected_runtime_hash=expected_runtime_hash,
                arm=arm,
                seed=seed,
            )
            rows[seed_key][arm] = payload
        initial_hashes = {
            rows[seed_key][arm]["training"][
                "initial_ordinary_weight_sha256"
            ]
            for arm in ARMS
        }
        if len(initial_hashes) != 1:
            raise RuntimeError(
                f"ordinary initialization differs within seed {seed}"
            )
        native = rows[seed_key]["native"]["evaluation"]
        decisions[seed_key] = {}
        for arm in ARMS:
            if arm == "native":
                continue
            candidate = rows[seed_key][arm]["evaluation"]
            deltas = {
                key: (
                    candidate[key]["mean_nll"]
                    - native[key]["mean_nll"]
                )
                for key in ("128", "256", "512")
            }
            suffix_deltas = {
                key: (
                    candidate[key]["suffix_mean_nll"]
                    - native[key]["suffix_mean_nll"]
                )
                for key in ("128", "256", "512")
            }
            bootstrap = {}
            suffix_bootstrap = {}
            for length_index, key in enumerate(("128", "256", "512")):
                bootstrap[key] = paired_bootstrap_summary(
                    candidate[key]["per_chunk_nll"],
                    native[key]["per_chunk_nll"],
                    seed=seed * 10_000 + length_index,
                )
                suffix_bootstrap[key] = paired_bootstrap_summary(
                    candidate[key]["per_chunk_suffix_nll"],
                    native[key]["per_chunk_suffix_nll"],
                    seed=seed * 10_000 + 100 + length_index,
                )
            decisions[seed_key][arm] = {
                "nll_delta_candidate_minus_native": deltas,
                "suffix_nll_delta_candidate_minus_native": suffix_deltas,
                "paired_bootstrap": bootstrap,
                "paired_suffix_bootstrap": suffix_bootstrap,
                "per_seed_gate": bool(
                    deltas["128"] <= 0.02
                    and deltas["256"] < 0.0
                    and deltas["512"] < 0.0
                ),
            }
    aggregate: dict[str, Any] = {}
    for arm in ARMS:
        if arm == "native":
            continue
        deltas_by_length: dict[str, list[float]] = {}
        for length in ("128", "256", "512"):
            deltas_by_length[length] = [
                decisions[str(seed)][arm][
                    "nll_delta_candidate_minus_native"
                ][length]
                for seed in SEEDS
            ]
        all_seed_gate = bool(
            all(
                decisions[str(seed)][arm]["per_seed_gate"]
                for seed in SEEDS
            )
        )
        all_seed_512_win = bool(
            all(value < 0.0 for value in deltas_by_length["512"])
        )
        aggregate[arm] = {
            "mean_nll_delta_candidate_minus_native": {
                length: float(np.mean(values))
                for length, values in deltas_by_length.items()
            },
            "seed_delta_range_candidate_minus_native": {
                length: [float(min(values)), float(max(values))]
                for length, values in deltas_by_length.items()
            },
            "all_seed_512_win": all_seed_512_win,
            "all_seed_gate": all_seed_gate,
            "ppl_screen_eligible_for_capability_followup": bool(
                all_seed_gate
                and float(np.mean(deltas_by_length["128"])) <= 0.01
                and all_seed_512_win
            ),
        }
    return {
        "status": RESULT_STATUS,
        "classification": "PPL_SCREEN_ONLY_NOT_CAPABILITY_EVIDENCE",
        "selection_boundary": (
            "all three seeds are used for candidate selection; "
            "there is no independent confirmation seed"
        ),
        "contract": expected_contract,
        "seeds": rows,
        "per_seed_decisions": decisions,
        "aggregate": aggregate,
    }


def cuda_evaluation_batch_parity(
    *,
    model: nn.Module,
    validation: torch.Tensor,
    offsets: dict[int, list[int]],
    device: torch.device,
    tolerance: float = 5e-4,
) -> dict[str, Any]:
    probe_offsets = {
        length: starts[:8] for length, starts in offsets.items()
    }
    individual = evaluate(
        model=model,
        validation=validation,
        offsets=probe_offsets,
        device=device,
        token_batch_budget=1,
    )
    registered = evaluate(
        model=model,
        validation=validation,
        offsets=probe_offsets,
        device=device,
        token_batch_budget=4096,
    )
    maximum = 0.0
    cells: dict[str, Any] = {}
    for key in ("128", "256", "512"):
        full = np.max(
            np.abs(
                np.asarray(individual[key]["per_chunk_nll"])
                - np.asarray(registered[key]["per_chunk_nll"])
            )
        )
        suffix = np.max(
            np.abs(
                np.asarray(individual[key]["per_chunk_suffix_nll"])
                - np.asarray(
                    registered[key]["per_chunk_suffix_nll"]
                )
            )
        )
        maximum = max(maximum, float(full), float(suffix))
        cells[key] = {
            "full_nll_max_abs_difference": float(full),
            "suffix_nll_max_abs_difference": float(suffix),
            "individual_batch_size": individual[key][
                "evaluation_batch_size"
            ],
            "registered_batch_size": registered[key][
                "evaluation_batch_size"
            ],
        }
    if not math.isfinite(maximum) or maximum > tolerance:
        raise RuntimeError("CUDA evaluation batching parity failed")
    return {
        "tolerance": tolerance,
        "maximum_abs_difference": maximum,
        "cells": cells,
    }


def archive_partial_directory(path: Path) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    candidate = path.with_name(f"{path.name}.aborted-{timestamp}")
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = path.with_name(
            f"{path.name}.aborted-{timestamp}-{suffix}"
        )
    path.rename(candidate)
    return candidate


def checkpoint_size_bytes(protocol: Protocol) -> int:
    model = build_model(protocol, "native")
    size = sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )
    del model
    return int(size)


def nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise RuntimeError(f"no existing parent for {path}")
        current = current.parent
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "prepare",
            "cpu-preflight",
            "runtime-probe",
            "train-suite",
        ),
        required=True,
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--prepared-receipt", type=Path, required=True)
    parser.add_argument("--cpu-ready-receipt", type=Path)
    parser.add_argument("--runtime-ready-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu"
    )
    parser.add_argument("--micro-batch", type=int, default=128)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "none",
            "default",
            "max-autotune-no-cudagraphs",
        ),
        default="none",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_data = args.train_data.resolve()
    validation_data = args.validation_data.resolve()
    base_protocol = Protocol(seed=42)
    expected_contract = contract(
        base_protocol, train_data, validation_data
    )
    output = args.output.resolve()

    if args.mode == "prepare":
        load_and_validate_data(
            train_data, validation_data, base_protocol
        )
        if output.exists():
            raise FileExistsError(output)
        payload = {
            "status": PREPARED_STATUS,
            "classification": "OFFLINE_PREPARATION_NOT_RESULT",
            "contract": expected_contract,
        }
        atomic_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    prepared = verify_receipt(
        args.prepared_receipt.resolve(),
        PREPARED_STATUS,
        expected_contract,
    )
    rows, validation, offsets = load_and_validate_data(
        train_data, validation_data, base_protocol
    )

    if args.mode == "cpu-preflight":
        if output.exists():
            raise FileExistsError(output)
        parity = parity_preflight(
            base_protocol, rows, validation, offsets
        )
        payload = {
            "status": CPU_READY_STATUS,
            "classification": "CPU_READY_NOT_RESULT",
            "contract": expected_contract,
            "prepared_receipt_sha256": file_sha256(
                args.prepared_receipt.resolve()
            ),
            "parity": parity,
        }
        atomic_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.cpu_ready_receipt is None:
        raise RuntimeError("GPU modes require --cpu-ready-receipt")
    cpu_ready = verify_receipt(
        args.cpu_ready_receipt.resolve(),
        CPU_READY_STATUS,
        expected_contract,
    )
    if (
        cpu_ready.get("prepared_receipt_sha256")
        != file_sha256(args.prepared_receipt.resolve())
    ):
        raise RuntimeError("CPU-ready receipt is not bound to prepared")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU mode requires available CUDA")
    runtime = configure_cuda()
    if (
        not runtime["bf16_supported"]
        or not runtime["flash_sdp_enabled"]
        or runtime["mem_efficient_sdp_enabled"]
        or runtime["math_sdp_enabled"]
        or runtime["cudnn_sdp_enabled"]
    ):
        raise RuntimeError("Blackwell backend contract failed")
    capability = runtime["compute_capability"]
    architecture = f"sm_{capability[0]}{capability[1]}"
    if architecture not in runtime["compiled_architectures"]:
        raise RuntimeError(
            f"active architecture {architecture} is not compiled"
        )
    device = torch.device("cuda")

    if args.mode == "runtime-probe":
        if output.exists():
            raise FileExistsError(output)
        probe_protocol = replace(
            base_protocol,
            train_tokens_requested=(
                5 * base_protocol.global_batch_sequences
                * base_protocol.train_length
            ),
        )
        compile_candidates = ["none"]
        if args.compile_mode != "none":
            compile_candidates.append(args.compile_mode)
        probe_models: dict[str, nn.Module] = {}
        probe_trainings: dict[str, Any] = {}
        for compile_mode in compile_candidates:
            directory_name = compile_mode.replace("-", "_")
            model, training = train_one(
                arm="phase_observed_evq",
                protocol=probe_protocol,
                rows=rows[: probe_protocol.train_rows],
                device=device,
                micro_batch=args.micro_batch,
                compile_mode=compile_mode,
                output=output / f"probe_{directory_name}",
                max_steps=5,
                save_checkpoint=False,
            )
            training["estimated_registered_run_seconds"] = (
                float(training["first_step_seconds"])
                + (base_protocol.optimizer_steps - 1)
                * float(training["steady_step_seconds_mean"])
            )
            estimate = training["estimated_registered_run_seconds"]
            if estimate is None or not math.isfinite(estimate):
                raise RuntimeError("runtime timing estimate is invalid")
            probe_models[compile_mode] = model
            probe_trainings[compile_mode] = training
        selected_mode = min(
            compile_candidates,
            key=lambda name: probe_trainings[name][
                "estimated_registered_run_seconds"
            ],
        )
        selected_model = probe_models[selected_mode]
        evaluation_parity = cuda_evaluation_batch_parity(
            model=selected_model,
            validation=validation,
            offsets=offsets,
            device=device,
        )
        for name in list(probe_models):
            if name != selected_mode:
                del probe_models[name]
        payload = {
            "status": RUNTIME_READY_STATUS,
            "classification": "RUNTIME_PROBE_NOT_RESULT",
            "contract": expected_contract,
            "prepared_receipt_sha256": file_sha256(
                args.prepared_receipt.resolve()
            ),
            "cpu_ready_receipt_sha256": file_sha256(
                args.cpu_ready_receipt.resolve()
            ),
            "runtime": runtime,
            "selected_execution": {
                "micro_batch": args.micro_batch,
                "accumulation": (
                    base_protocol.global_batch_sequences
                    // args.micro_batch
                ),
                "global_batch": (
                    base_protocol.global_batch_sequences
                ),
                "compile_mode": selected_mode,
                "selection_metric": (
                    "first_step_plus_914_times_mean_steps_2_to_5"
                ),
            },
            "execution_candidates": probe_trainings,
            "cuda_evaluation_batch_parity": evaluation_parity,
        }
        atomic_json(output / "runtime_ready.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.runtime_ready_receipt is None:
        raise RuntimeError("train-suite requires runtime-ready receipt")
    runtime_ready = verify_receipt(
        args.runtime_ready_receipt.resolve(),
        RUNTIME_READY_STATUS,
        expected_contract,
    )
    runtime_ready_path = args.runtime_ready_receipt.resolve()
    runtime_hash = file_sha256(runtime_ready_path)
    if (
        runtime_ready.get("prepared_receipt_sha256")
        != file_sha256(args.prepared_receipt.resolve())
        or runtime_ready.get("cpu_ready_receipt_sha256")
        != file_sha256(args.cpu_ready_receipt.resolve())
    ):
        raise RuntimeError(
            "runtime-ready receipt is not bound to current CPU-ready receipt"
        )
    selected = runtime_ready["selected_execution"]
    micro_batch = int(selected["micro_batch"])
    compile_mode = str(selected["compile_mode"])
    output.mkdir(parents=True, exist_ok=True)
    contract_hash = canonical_json_sha256(expected_contract)
    complete_pairs: set[tuple[int, str]] = set()
    for seed in SEEDS:
        for arm in ARMS:
            arm_output = output / f"seed{seed}" / arm
            partial_output = output / f"seed{seed}" / f"{arm}.incomplete"
            if partial_output.exists():
                archive_partial_directory(partial_output)
            result_path = arm_output / "results.json"
            if result_path.exists():
                existing = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
                validate_result_payload(
                    result_path=result_path,
                    payload=existing,
                    expected_contract=expected_contract,
                    offsets=offsets,
                    expected_runtime_hash=runtime_hash,
                    arm=arm,
                    seed=seed,
                )
                complete_pairs.add((seed, arm))
            elif arm_output.exists():
                archive_partial_directory(arm_output)
    remaining = len(SEEDS) * len(ARMS) - len(complete_pairs)
    checkpoint_bytes = checkpoint_size_bytes(base_protocol)
    required_bytes = int(
        remaining * checkpoint_bytes * 1.25 + 1024**3
    )
    disk = shutil.disk_usage(nearest_existing_parent(output))
    if disk.free < required_bytes:
        raise RuntimeError(
            "insufficient free space for remaining checkpoints: "
            f"free={disk.free}, required={required_bytes}"
        )
    for seed in SEEDS:
        protocol = replace(base_protocol, seed=seed)
        for arm in ARMS:
            if (seed, arm) in complete_pairs:
                continue
            arm_output = output / f"seed{seed}" / arm
            partial_output = output / f"seed{seed}" / f"{arm}.incomplete"
            model, training = train_one(
                arm=arm,
                protocol=protocol,
                rows=rows,
                device=device,
                micro_batch=micro_batch,
                compile_mode=compile_mode,
                output=partial_output,
                max_steps=None,
                save_checkpoint=True,
            )
            evaluation = evaluate(
                model=model,
                validation=validation,
                offsets=offsets,
                device=device,
            )
            payload = {
                "status": RESULT_STATUS,
                "classification": (
                    "PPL_SCREEN_ONLY_NOT_CAPABILITY_EVIDENCE"
                ),
                "contract_sha256": contract_hash,
                "runtime_ready_receipt_sha256": runtime_hash,
                "arm": arm,
                "seed": seed,
                "training": training,
                "evaluation": evaluation,
            }
            atomic_json(partial_output / "results.json", payload)
            del model
            partial_output.rename(arm_output)
    aggregate = aggregate_results(
        output, expected_contract, offsets, runtime_hash
    )
    aggregate_path = output / "results.json"
    atomic_json(aggregate_path, aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
