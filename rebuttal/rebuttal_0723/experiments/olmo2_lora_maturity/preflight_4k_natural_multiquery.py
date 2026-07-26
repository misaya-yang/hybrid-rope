#!/usr/bin/env python3
"""Create a no-GPU READY receipt for one natural multi-query LoRA arm."""

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

from .train_4k_counterfactual_routing import (
    FAMILY_PATTERN,
    LENGTH,
    PAIR_COLLECTION_STATUSES,
    PAIR_SET_STATUSES,
)
from .train_4k_stage_a import ready_checkpoint_digest


READY_STATUS = "OLMO2_4K_NATURAL_MULTIQUERY_READY"
PAIR_SET_STATUS = "OLMO2_4K_COUNTERFACTUAL_NATURAL_MULTIQUERY_SET_PREPARED"
PAIR_COLLECTION_STATUS = (
    "OLMO2_4K_COUNTERFACTUAL_NATURAL_MULTIQUERY_DATA_PREPARED"
)


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
    parser.add_argument(
        "--frequency", choices=("native", "evq"), required=True
    )
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


def file_entry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def validate_pair_collection(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != PAIR_COLLECTION_STATUS
        or manifest.get("status") not in PAIR_COLLECTION_STATUSES
    ):
        raise RuntimeError("natural multi-query collection status drift")
    if manifest.get("ruler_generator_used") is not False:
        raise RuntimeError("natural multi-query data used a RULER generator")
    if int(manifest.get("ruler_or_niah_rows", -1)) != 0:
        raise RuntimeError("natural multi-query RULER/NIAH row count drift")
    sets = {}
    for name in ("train", "calibration"):
        set_root = root / name
        set_manifest_path = set_root / "manifest.json"
        set_manifest = json.loads(
            set_manifest_path.read_text(encoding="utf-8")
        )
        if (
            set_manifest.get("status") != PAIR_SET_STATUS
            or set_manifest.get("status") not in PAIR_SET_STATUSES
        ):
            raise RuntimeError(f"{name} pair-set status drift")
        if int(set_manifest["maximum_training_length"]) != LENGTH:
            raise RuntimeError(f"{name} pair-set length drift")
        if int(set_manifest["maximum_training_position_id"]) != LENGTH - 1:
            raise RuntimeError(f"{name} pair-set position drift")
        if int(set_manifest["queries_per_sequence"]) != 16:
            raise RuntimeError(f"{name} query-count drift")
        if int(set_manifest["answer_tokens_per_variant"]) != 128:
            raise RuntimeError(f"{name} source-answer density drift")
        if set_manifest.get("ruler_generator_used") is not False:
            raise RuntimeError(f"{name} used a RULER generator")
        files = {
            filename: file_entry(set_root / filename)
            for filename in (
                "input_ids.npy",
                "labels.npy",
                "lengths.npy",
                "rows.jsonl",
            )
        }
        for filename, entry in files.items():
            expected = set_manifest["files"][filename]
            if entry["sha256"] != expected:
                raise RuntimeError(f"{name}/{filename} hash drift")
        sets[name] = {
            "manifest": file_entry(set_manifest_path),
            "files": files,
            "rows": int(set_manifest["shape"][0]),
            "queries_per_sequence": int(
                set_manifest["queries_per_sequence"]
            ),
            "answer_tokens_per_variant": int(
                set_manifest["answer_tokens_per_variant"]
            ),
        }
    return {
        "manifest": file_entry(manifest_path),
        "sets": sets,
        "source_row_overlap": int(
            manifest.get("train_calibration_source_row_overlap", -1)
        ),
        "ruler_generator_used": False,
        "ruler_or_niah_rows": 0,
    }


def main() -> None:
    args = parse_args()
    if int(args.steps) <= 0:
        raise ValueError("steps must be positive")
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
        "frequency": str(args.frequency),
        "adaptation": "qkvo_answer",
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
    }
    for name, expected in expected_parent.items():
        if metadata.get(name) != expected:
            raise RuntimeError(f"parent adapter metadata drift for {name}")

    natural_manifest = (
        prepared / "longalign_paired_L4096" / "manifest.json"
    )
    background_manifest = background / "manifest.json"
    protocol = {
        "frequency": str(args.frequency),
        "steps": int(args.steps),
        "hard_maximum_training_length": LENGTH,
        "hard_maximum_training_position_id": LENGTH - 1,
        "family_pattern": list(FAMILY_PATTERN),
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "global_batch_size": int(
            args.micro_batch_size
            * args.gradient_accumulation_steps
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "counterfactual_margin": float(args.counterfactual_margin),
        "counterfactual_margin_weight": float(
            args.counterfactual_margin_weight
        ),
        "compile_mode": str(args.compile_mode),
        "natural_eval_rows": int(args.natural_eval_rows),
        "seed": int(args.seed),
    }
    receipt = {
        "status": READY_STATUS,
        "reviewer_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "existing_evidence": (
            "Official-RULER-generator routing transfers within NIAH, while "
            "seven non-RULER objectives fail broad RULER screening."
        ),
        "smallest_missing_evidence": (
            "Whether dense source-dependent supervision from an independent "
            "natural-text generator transfers beyond its training task."
        ),
        "smallest_executable_plan": (
            "Run one EVQ and one matched Native 4K-only LoRA arm from their "
            "frozen Stage-A adapters, then screen only after the held-out "
            "multi-query calibration passes."
        ),
        "stop_condition": (
            "Require at least 80% held-out source-token exact after 300 "
            "steps. A failing arm receives no RULER evaluation."
        ),
        "protocol": protocol,
        "inputs": {
            "checkpoint": {
                "digest": checkpoint_digest,
                "config": file_entry(checkpoint / "config.json"),
                "weights": file_entry(checkpoint / "model.safetensors"),
            },
            "stage_ready_receipt": file_entry(stage_ready),
            "parent_adapter": {
                **file_entry(parent),
                "metadata": metadata,
            },
            "prepared_natural_replay": file_entry(natural_manifest),
            "routing_data": validate_pair_collection(routing),
            "background": file_entry(background_manifest),
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
            "The training generator imports no RULER code and uses no RULER "
            "rows, templates, identities, or value lists. The task still "
            "trains generic long-range retrieval and must be evaluated on "
            "held-out task families."
        ),
    }
    if receipt["inputs"]["routing_data"]["source_row_overlap"] != 0:
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
