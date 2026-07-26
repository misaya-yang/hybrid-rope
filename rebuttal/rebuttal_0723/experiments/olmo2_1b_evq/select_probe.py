#!/usr/bin/env python3
"""Admit the fastest numerically matched RTX Pro 6000 runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LOSS_RELATIVE_TOLERANCE = 5.0e-4
GRAD_NORM_RELATIVE_TOLERANCE = 5.0e-3
MINIMUM_TOKENS_PER_SECOND = 20_000.0
THROUGHPUT_NEAR_TIE_TOLERANCE = 0.02


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-12)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probes = {
        backend: json.loads(
            (args.probe_dir / f"probe_{backend}.json").read_text(
                encoding="utf-8"
            )
        )
        for backend in ("native", "liger")
    }
    for backend, probe in probes.items():
        if probe.get("status") != "GPU_PROBE_PASS":
            raise RuntimeError(f"{backend} probe did not pass")
        if "RTX PRO 6000" not in probe["gpu"].upper():
            raise RuntimeError(
                f"{backend} probe ran on unexpected GPU {probe['gpu']!r}"
            )
        if probe["compute_capability"] != [12, 0]:
            raise RuntimeError(
                f"{backend} probe compute capability is not SM120"
            )
    identity_fields = (
        "gpu",
        "compute_capability",
        "precision",
        "attention",
        "compile",
        "schedule",
        "sequence_length",
        "global_batch_sequences",
        "microbatch_sequences",
        "gradient_accumulation",
        "warmup_steps",
        "measured_steps",
    )
    for field in identity_fields:
        if probes["native"][field] != probes["liger"][field]:
            raise RuntimeError(f"probe identity mismatch at {field}")

    parity = {
        field: relative_difference(
            float(probes["native"][field]),
            float(probes["liger"][field]),
        )
        for field in (
            "first_measured_loss",
            "last_measured_loss",
            "first_measured_ce_loss",
            "last_measured_ce_loss",
            "first_measured_z_loss",
            "last_measured_z_loss",
            "first_measured_grad_norm",
            "last_measured_grad_norm",
        )
    }
    loss_ok = all(
        difference <= LOSS_RELATIVE_TOLERANCE
        for field, difference in parity.items()
        if "grad_norm" not in field
    )
    grad_ok = all(
        difference <= GRAD_NORM_RELATIVE_TOLERANCE
        for field, difference in parity.items()
        if "grad_norm" in field
    )
    candidates = ["native"]
    if loss_ok and grad_ok:
        candidates.append("liger")
    fastest_throughput = max(
        float(probes[name]["tokens_per_second"]) for name in candidates
    )
    near_tied = [
        name
        for name in candidates
        if float(probes[name]["tokens_per_second"])
        >= fastest_throughput * (1.0 - THROUGHPUT_NEAR_TIE_TOLERANCE)
    ]
    selected = min(
        near_tied,
        key=lambda name: max(
            int(probes[name]["peak_memory_bytes"]),
            int(probes[name]["peak_memory_reserved_bytes"]),
        ),
    )
    throughput = float(probes[selected]["tokens_per_second"])
    if throughput < MINIMUM_TOKENS_PER_SECOND:
        raise RuntimeError(
            f"selected probe is only {throughput:.1f} tok/s; "
            f"minimum is {MINIMUM_TOKENS_PER_SECOND:.1f}"
        )
    receipt = {
        "status": "GPU_RUNTIME_SELECTED",
        "selected_loss_backend": selected,
        "tokens_per_second": throughput,
        "estimated_1000_step_hours": 2_097_152_000 / throughput / 3600,
        "liger_parity": {
            "loss_pass": loss_ok,
            "gradient_norm_pass": grad_ok,
            "relative_differences": parity,
            "loss_tolerance": LOSS_RELATIVE_TOLERANCE,
            "grad_norm_tolerance": GRAD_NORM_RELATIVE_TOLERANCE,
        },
        "selection_rule": (
            "Liger is eligible only after native parity; among backends within "
            "2% of the fastest quick probe, select the lowest-memory backend "
            "to admit the larger-microbatch sweep"
        ),
        "runtime": probes[selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
