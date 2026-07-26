#!/usr/bin/env python3
"""CPU-only phase and LoRA audit for the mature-model EVQ transplant."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

import torch
from safetensors import safe_open

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    endpoint_evq_inv_freq,
    endpoint_geo_inv_freq,
    tensor_sha256,
)
from .prepare_data import atomic_json, sha256_file


ADAPTER_NAMES = (
    "native_stage_a",
    "native_final",
    "evq_stage_a",
    "evq_final",
    "evq_seed2_stage_a",
    "evq_seed2_final",
)
MODULE_PATTERN = re.compile(
    r"^model\.model\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<projection>[qkvo]_proj)\.(?P<factor>[ab])$"
)
DISTANCES = (128, 512, 1_024, 2_048, 4_095, 8_191, 16_383)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    for name in ADAPTER_NAMES:
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=Path,
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_adapter(
    path: Path,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, float]], dict[str, Any]]:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    state = payload["state"]
    metadata = payload["metadata"]
    rank = int(metadata["rank"])
    scale = float(metadata["alpha"]) / rank
    factors: dict[str, dict[str, torch.Tensor]] = {}
    for name, value in state.items():
        match = MODULE_PATTERN.match(name)
        if match is None:
            raise RuntimeError(f"unexpected adapter tensor {name!r}")
        module = name.rsplit(".", 1)[0]
        factors.setdefault(module, {})[match.group("factor")] = (
            value.detach().cpu().float()
        )
    modules = {}
    for name, values in factors.items():
        if set(values) != {"a", "b"}:
            raise RuntimeError(f"incomplete LoRA factors for {name}")
        a, b = values["a"], values["b"]
        if a.shape[0] != rank or b.shape[1] != rank:
            raise RuntimeError(f"LoRA rank drift for {name}")
        modules[name] = (a, b, scale)
    if len(modules) != 64:
        raise RuntimeError(f"expected 64 LoRA modules, got {len(modules)}")
    return modules, metadata


def _matrix_inner(
    left: tuple[torch.Tensor, torch.Tensor, float],
    right: tuple[torch.Tensor, torch.Tensor, float],
) -> float:
    left_a, left_b, left_scale = left
    right_a, right_b, right_scale = right
    value = torch.sum(
        (left_b.T @ right_b) * (left_a @ right_a.T)
    )
    return float(value) * left_scale * right_scale


def _matrix_norm(
    value: tuple[torch.Tensor, torch.Tensor, float],
) -> float:
    return math.sqrt(max(0.0, _matrix_inner(value, value)))


def _base_weight_key(adapter_module: str) -> str:
    prefix = "model.model."
    if not adapter_module.startswith(prefix):
        raise RuntimeError(f"unexpected adapter module {adapter_module!r}")
    return "model." + adapter_module[len(prefix) :] + ".weight"


def _projection(adapter_module: str) -> str:
    match = MODULE_PATTERN.match(adapter_module + ".a")
    if match is None:
        raise RuntimeError(f"cannot parse module {adapter_module!r}")
    return str(match.group("projection"))


def _layer(adapter_module: str) -> int:
    match = MODULE_PATTERN.match(adapter_module + ".a")
    if match is None:
        raise RuntimeError(f"cannot parse module {adapter_module!r}")
    return int(match.group("layer"))


def _summarize_adapter(
    modules: dict[str, tuple[torch.Tensor, torch.Tensor, float]],
    base_norms: dict[str, float],
    metadata: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    rows = []
    for name, value in sorted(modules.items()):
        norm = _matrix_norm(value)
        base_norm = base_norms[name]
        rows.append(
            {
                "module": name,
                "layer": _layer(name),
                "projection": _projection(name),
                "effective_delta_fro": norm,
                "base_weight_fro": base_norm,
                "relative_delta_fro": norm / base_norm,
            }
        )
    by_projection: dict[str, Any] = {}
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        selected = [
            row for row in rows if row["projection"] == projection
        ]
        relatives = [row["relative_delta_fro"] for row in selected]
        by_projection[projection] = {
            "modules": len(selected),
            "mean_relative_delta_fro": sum(relatives) / len(relatives),
            "median_relative_delta_fro": median(relatives),
            "maximum_relative_delta_fro": max(relatives),
            "effective_delta_energy": sum(
                row["effective_delta_fro"] ** 2 for row in selected
            ),
        }
    by_layer = {}
    for layer in range(16):
        selected = [row for row in rows if row["layer"] == layer]
        delta = math.sqrt(
            sum(row["effective_delta_fro"] ** 2 for row in selected)
        )
        base = math.sqrt(
            sum(row["base_weight_fro"] ** 2 for row in selected)
        )
        by_layer[str(layer)] = {
            "relative_joint_delta_fro": delta / base,
            "joint_delta_fro": delta,
        }
    qk_energy = sum(
        by_projection[name]["effective_delta_energy"]
        for name in ("q_proj", "k_proj")
    )
    total_energy = sum(
        value["effective_delta_energy"]
        for value in by_projection.values()
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "metadata": metadata,
        "by_projection": by_projection,
        "by_layer": by_layer,
        "qk_fraction_of_effective_delta_energy": qk_energy / total_energy,
        "module_rows": rows,
    }


def _transition(
    before: dict[str, tuple[torch.Tensor, torch.Tensor, float]],
    after: dict[str, tuple[torch.Tensor, torch.Tensor, float]],
    base_norms: dict[str, float],
) -> dict[str, Any]:
    if set(before) != set(after):
        raise RuntimeError("adapter module set drift")
    rows = []
    for name in sorted(before):
        before_norm = _matrix_norm(before[name])
        after_norm = _matrix_norm(after[name])
        inner = _matrix_inner(before[name], after[name])
        difference = math.sqrt(
            max(0.0, before_norm**2 + after_norm**2 - 2.0 * inner)
        )
        cosine = inner / max(before_norm * after_norm, 1e-30)
        rows.append(
            {
                "module": name,
                "layer": _layer(name),
                "projection": _projection(name),
                "effective_delta_difference_fro": difference,
                "difference_relative_to_base": (
                    difference / base_norms[name]
                ),
                "before_after_cosine": cosine,
            }
        )
    by_projection = {}
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        selected = [
            row for row in rows if row["projection"] == projection
        ]
        by_projection[projection] = {
            "joint_difference_fro": math.sqrt(
                sum(
                    row["effective_delta_difference_fro"] ** 2
                    for row in selected
                )
            ),
            "mean_difference_relative_to_base": sum(
                row["difference_relative_to_base"] for row in selected
            )
            / len(selected),
            "median_before_after_cosine": median(
                row["before_after_cosine"] for row in selected
            ),
        }
    return {"by_projection": by_projection, "module_rows": rows}


def _frequency_audit() -> dict[str, Any]:
    native = endpoint_geo_inv_freq().double()
    evq = endpoint_evq_inv_freq().double()
    difference = evq - native
    selected = []
    for index in (0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 60, 63):
        selected.append(
            {
                "index": index,
                "native": float(native[index]),
                "evq": float(evq[index]),
                "ratio": float(evq[index] / native[index]),
                "maximum_unwrapped_phase_error_at_4k": float(
                    abs(difference[index]) * 4_095
                ),
            }
        )
    distance_rows = {}
    for distance in DISTANCES:
        phase = (
            (difference * distance + math.pi) % (2.0 * math.pi)
        ) - math.pi
        absolute = phase.abs()
        distance_rows[str(distance)] = {
            "mean_absolute_wrapped_phase_error": float(absolute.mean()),
            "median_absolute_wrapped_phase_error": float(
                absolute.median()
            ),
            "p90_absolute_wrapped_phase_error": float(
                torch.quantile(absolute, 0.9)
            ),
            "pairs_below_0p1_rad": int((absolute < 0.1).sum()),
            "mean_phase_alignment_cosine": float(
                torch.cos(difference * distance).mean()
            ),
        }
    return {
        "native_sha256_float32": tensor_sha256(native.float()),
        "evq_sha256_float32": tensor_sha256(evq.float()),
        "pairs": int(native.numel()),
        "selected_pairs": selected,
        "pairs_with_maximum_unwrapped_4k_error_below": {
            str(threshold): int(
                (difference.abs() * 4_095 < threshold).sum()
            )
            for threshold in (0.1, 0.5, 1.0, math.pi, 2.0 * math.pi)
        },
        "distance_rows": distance_rows,
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    adapters = {
        name: Path(getattr(args, name)).resolve()
        for name in ADAPTER_NAMES
    }
    loaded = {
        name: _load_adapter(path)
        for name, path in adapters.items()
    }
    module_names = set(next(iter(loaded.values()))[0])
    if any(set(value[0]) != module_names for value in loaded.values()):
        raise RuntimeError("adapter module identity drift")

    base_norms: dict[str, float] = {}
    with safe_open(
        model_path,
        framework="pt",
        device="cpu",
    ) as handle:
        for module in sorted(module_names):
            tensor = handle.get_tensor(_base_weight_key(module)).float()
            base_norms[module] = float(torch.linalg.vector_norm(tensor))

    summaries = {
        name: _summarize_adapter(
            modules,
            base_norms,
            metadata,
            adapters[name],
        )
        for name, (modules, metadata) in loaded.items()
    }
    transitions = {
        "native_stage_a_to_final": _transition(
            loaded["native_stage_a"][0],
            loaded["native_final"][0],
            base_norms,
        ),
        "evq_stage_a_to_final": _transition(
            loaded["evq_stage_a"][0],
            loaded["evq_final"][0],
            base_norms,
        ),
        "evq_seed2_stage_a_to_final": _transition(
            loaded["evq_seed2_stage_a"][0],
            loaded["evq_seed2_final"][0],
            base_norms,
        ),
    }
    receipt = {
        "status": "OLMO2_POSTHOC_TRANSPLANT_CPU_AUDIT_COMPLETE",
        "claim_boundary": (
            "Descriptive phase and adapter-weight diagnostics only. "
            "Norms do not identify causal circuits or prove that one "
            "projection family causes the task regression."
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(model_path),
        "frequency": _frequency_audit(),
        "adapters": summaries,
        "stage_transitions": transitions,
    }
    atomic_json(args.output.resolve(), receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(args.output.resolve()),
                "frequency": receipt["frequency"],
                "adapter_projection_summaries": {
                    name: value["by_projection"]
                    for name, value in summaries.items()
                },
                "stage_transition_projection_summaries": {
                    name: value["by_projection"]
                    for name, value in transitions.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
