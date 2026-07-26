#!/usr/bin/env python3
"""Choose microbatch, then admit only a sustained Blackwell benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MINIMUM_TOKENS_PER_SECOND = 20_000.0
MAXIMUM_MEMORY_FRACTION = 0.94
EXPECTED_GPU_NAME_FRAGMENT = "RTX PRO 6000 Blackwell"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_blackwell_probe(
    receipt: dict[str, Any],
    *,
    minimum_warmup: int,
    minimum_measured: int,
) -> None:
    if receipt.get("status") != "GPU_PROBE_PASS":
        raise RuntimeError("GPU probe did not pass")
    if EXPECTED_GPU_NAME_FRAGMENT not in str(receipt.get("gpu")):
        raise RuntimeError(f"unexpected GPU: {receipt.get('gpu')!r}")
    expected = {
        "compute_capability": [12, 0],
        "torch_version": "2.8.0+cu128",
        "cuda_version": "12.8",
        "precision": "amp_bf16",
        "schedule": "evq",
        "sequence_length": 4_096,
        "global_batch_sequences": 512,
        "optimizer_state_initialized": True,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise RuntimeError(f"GPU probe contract drift: {field}")
    microbatch = int(receipt["microbatch_sequences"])
    if (
        microbatch not in (4, 8)
        or int(receipt["gradient_accumulation"]) != 512 // microbatch
    ):
        raise RuntimeError("GPU probe accumulation contract drift")
    attention = receipt.get("attention", {})
    if (
        attention.get("implementation") != "evq_flash_only_sdpa"
        or attention.get("flash_enabled") is not True
        or attention.get("math_enabled") is not False
        or attention.get("memory_efficient_enabled") is not False
        or attention.get("cudnn_enabled") is not False
    ):
        raise RuntimeError("GPU probe is not Flash-only")
    compile_receipt = receipt.get("compile", {})
    if (
        compile_receipt.get("enabled") is not True
        or compile_receipt.get("mode") != "max-autotune-no-cudagraphs"
        or not compile_receipt.get("cache")
    ):
        raise RuntimeError("GPU probe compile contract drift")
    if (
        int(receipt.get("warmup_steps", -1)) < minimum_warmup
        or int(receipt.get("measured_steps", -1)) < minimum_measured
    ):
        raise RuntimeError("GPU probe duration is below the required gate")
    finite_fields = (
        "tokens_per_second",
        "first_measured_loss",
        "last_measured_loss",
        "first_measured_ce_loss",
        "last_measured_ce_loss",
        "first_measured_z_loss",
        "last_measured_z_loss",
        "first_measured_grad_norm",
        "last_measured_grad_norm",
        "peak_memory_bytes",
        "peak_memory_reserved_bytes",
        "total_memory_bytes",
    )
    for field in finite_fields:
        value = float(receipt.get(field, float("nan")))
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"GPU probe has invalid numeric field: {field}")


def choose(args: argparse.Namespace) -> None:
    backend = load(args.backend_receipt)
    if backend.get("status") != "GPU_RUNTIME_SELECTED":
        raise RuntimeError("backend receipt is not selected")
    candidates = [backend["runtime"]]
    validate_blackwell_probe(
        candidates[0], minimum_warmup=5, minimum_measured=20
    )
    if args.microbatch8_probe.is_file():
        extra = load(args.microbatch8_probe)
        if (
            extra.get("status") == "GPU_PROBE_PASS"
            and extra["loss_backend"] == backend["selected_loss_backend"]
            and extra["microbatch_sequences"] == 8
            and max(
                extra["peak_memory_bytes"],
                extra["peak_memory_reserved_bytes"],
            )
            <= MAXIMUM_MEMORY_FRACTION * extra["total_memory_bytes"]
        ):
            validate_blackwell_probe(
                extra, minimum_warmup=5, minimum_measured=20
            )
            candidates.append(extra)
    selected = max(candidates, key=lambda row: row["tokens_per_second"])
    receipt = {
        "status": "GPU_RUNTIME_CANDIDATE_SELECTED",
        "selected_loss_backend": selected["loss_backend"],
        "selected_microbatch_sequences": selected["microbatch_sequences"],
        "selected_gradient_accumulation": selected["gradient_accumulation"],
        "quick_probe_tokens_per_second": selected["tokens_per_second"],
        "quick_probe": selected,
        "candidates": [
            {
                "loss_backend": row["loss_backend"],
                "microbatch_sequences": row["microbatch_sequences"],
                "tokens_per_second": row["tokens_per_second"],
                "peak_memory_bytes": row["peak_memory_bytes"],
                "peak_memory_reserved_bytes": row[
                    "peak_memory_reserved_bytes"
                ],
            }
            for row in candidates
        ],
        "selection_rule": (
            "fastest eligible quick probe; memory must stay below 94% "
            "of physical VRAM"
        ),
    }
    write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    candidate = load(args.candidate_receipt)
    sustained = load(args.sustained_probe)
    if candidate.get("status") != "GPU_RUNTIME_CANDIDATE_SELECTED":
        raise RuntimeError("runtime candidate receipt is invalid")
    validate_blackwell_probe(
        sustained, minimum_warmup=100, minimum_measured=300
    )
    expected = (
        candidate["selected_loss_backend"],
        candidate["selected_microbatch_sequences"],
    )
    actual = (
        sustained["loss_backend"],
        sustained["microbatch_sequences"],
    )
    if actual != expected:
        raise RuntimeError(
            f"sustained runtime {actual} does not match candidate {expected}"
        )
    quick = candidate["quick_probe"]
    stable_fields = (
        "gpu",
        "compute_capability",
        "torch_version",
        "cuda_version",
        "precision",
        "attention",
        "compile",
        "schedule",
        "sequence_length",
        "global_batch_sequences",
        "loss_backend",
        "microbatch_sequences",
        "gradient_accumulation",
    )
    for field in stable_fields:
        if sustained.get(field) != quick.get(field):
            raise RuntimeError(f"sustained/quick runtime drift: {field}")
    throughput = float(sustained["tokens_per_second"])
    if throughput < MINIMUM_TOKENS_PER_SECOND:
        raise RuntimeError(
            f"sustained throughput {throughput:.1f} tok/s is below gate"
        )
    if (
        max(
            sustained["peak_memory_bytes"],
            sustained["peak_memory_reserved_bytes"],
        )
        > MAXIMUM_MEMORY_FRACTION * sustained["total_memory_bytes"]
    ):
        raise RuntimeError("sustained probe leaves insufficient VRAM margin")
    receipt = {
        "status": "GPU_RUNTIME_SELECTED",
        "selected_loss_backend": actual[0],
        "selected_microbatch_sequences": actual[1],
        "selected_gradient_accumulation": sustained[
            "gradient_accumulation"
        ],
        "tokens_per_second": throughput,
        "estimated_1000_step_compute_hours_lower_bound": (
            2_097_152_000 / throughput / 3600
        ),
        "sustained_probe": sustained,
        "candidate_receipt": candidate,
        "selection_rule": (
            "100 discarded warmup plus 300 measured microsteps; "
            ">=20K tok/s and <=94% peak allocated VRAM"
        ),
    }
    write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    choose_parser = subparsers.add_parser("choose")
    choose_parser.add_argument("--backend-receipt", type=Path, required=True)
    choose_parser.add_argument("--microbatch8-probe", type=Path, required=True)
    choose_parser.add_argument("--output", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument(
        "--candidate-receipt", type=Path, required=True
    )
    finalize_parser.add_argument("--sustained-probe", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "choose":
        choose(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
