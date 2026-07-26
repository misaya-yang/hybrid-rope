#!/usr/bin/env python3
"""Create a no-GPU READY receipt for Native-to-EVQ routing morph."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import torch
import transformers

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .preflight_4k_natural_multiquery import (
    file_entry,
    validate_pair_collection,
)
from .train_4k_natural_multiquery_morph import (
    LENGTH,
    READY_STATUS,
    protocol_from_args,
)
from .train_4k_stage_a import ready_checkpoint_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage-ready-receipt", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--routing-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--counterfactual-margin", type=float, default=1.0)
    parser.add_argument(
        "--counterfactual-margin-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--natural-eval-rows", type=int, default=16)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.run_output.resolve()
    receipt_output = args.receipt_output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if receipt_output.exists():
        raise FileExistsError(receipt_output)

    checkpoint = args.checkpoint.resolve()
    stage_ready = args.stage_ready_receipt.resolve()
    parent = args.parent_adapter.resolve()
    prepared = args.prepared_data.resolve()
    routing = args.routing_data.resolve()
    background = args.background_dir.resolve()
    trainer = args.trainer.resolve()
    checkpoint_digest = ready_checkpoint_digest(checkpoint, stage_ready)
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    metadata = dict(payload.get("metadata", {}))
    expected_parent = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": "native",
        "adaptation": "qkvo_answer",
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
    }
    for name, expected in expected_parent.items():
        if metadata.get(name) != expected:
            raise RuntimeError(f"parent adapter metadata drift for {name}")

    protocol = protocol_from_args(args)
    routing_receipt = validate_pair_collection(routing)
    receipt: dict[str, Any] = {
        "status": READY_STATUS,
        "reviewer_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "existing_evidence": (
            "Independent natural multi-query training makes the Native "
            "adapter route correctly, whereas direct full-EVQ training "
            "preserves long-context NLL but fails the capability gate."
        ),
        "smallest_missing_evidence": (
            "Whether learning source-dependent routing before gradually "
            "changing the frequency substrate preserves the routing "
            "capability at the full EVQ endpoint."
        ),
        "smallest_executable_plan": (
            "Continue the passed Native adapter for 300 matched 4K-only "
            "steps while linearly morphing Native inverse frequencies to "
            "full EVQ; compare against an equal-budget Native continuation."
        ),
        "stop_condition": (
            "Require at least 80% held-out source-token exact at full EVQ. "
            "A failing morph receives no RULER evaluation."
        ),
        "protocol": protocol,
        "inputs": {
            "checkpoint": {
                "digest": checkpoint_digest,
                "config": file_entry(checkpoint / "config.json"),
                "weights": file_entry(
                    checkpoint / "model.safetensors"
                ),
            },
            "stage_ready_receipt": file_entry(stage_ready),
            "parent_adapter": {
                **file_entry(parent),
                "metadata": metadata,
            },
            "prepared_natural_replay": file_entry(
                prepared
                / "longalign_paired_L4096"
                / "manifest.json"
            ),
            "routing_data": {
                "manifest_sha256": routing_receipt["manifest"][
                    "sha256"
                ],
                **routing_receipt,
            },
            "background": file_entry(background / "manifest.json"),
        },
        "trainer": {
            "path": str(trainer),
            **file_entry(trainer),
        },
        "run_output": str(output),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "liger_kernel": importlib.metadata.version("liger-kernel"),
        },
        "evidence_boundary": (
            "The training generator is independent of RULER code and rows. "
            "The morph tests a post-hoc adaptation procedure, not a claim "
            "that pretrained Native weights are natively EVQ-compatible."
        ),
    }
    if routing_receipt["source_row_overlap"] != 0:
        raise RuntimeError("train/calibration source rows overlap")
    atomic_json(receipt_output, receipt)
    print(
        json.dumps(
            {
                "status": READY_STATUS,
                "receipt": str(receipt_output),
                "receipt_sha256": sha256_file(receipt_output),
                "protocol": protocol,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
