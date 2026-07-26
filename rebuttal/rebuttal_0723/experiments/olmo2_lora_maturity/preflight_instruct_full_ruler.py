#!/usr/bin/env python3
"""Create a no-GPU READY receipt for the 13-task 4K RULER diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from .prepare_data import atomic_json, sha256_file
from .prepare_instruct_ruler_transfer import (
    DATA_STATUS,
    DEFAULT_TASKS,
    RULER_COMMIT,
    TASK_CONFIGS,
)


MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
TOKENIZER_SHA256 = (
    "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca"
)
PARENT_READY_SHA256 = (
    "3988096b444212439ce288787a26c5a35104c73345aed61688ccbb5f544ab22d"
)
ADAPTERS = {
    "native_stage_a": {
        "sha256": (
            "f4bc6c471cd395eb0df25f6513d150d3"
            "d66a7133c2ac3a57406765708843574e"
        ),
        "frequency": "native",
        "seed": 20_260_725,
        "stage": "stage_a_4k",
    },
    "native_final": {
        "sha256": (
            "6570ab94aec68431dd4e261eb3ef342e"
            "f72253357aa65a0d37b0a018df2f3f8d"
        ),
        "frequency": "native",
        "seed": 20_260_725,
        "stage": "counterfactual_routing_4k",
    },
    "evq_stage_a": {
        "sha256": (
            "47e72c5e58be443a3f088787b58df415"
            "538f7cdb12e9f76d799dbb2b85155ee0"
        ),
        "frequency": "evq",
        "seed": 20_260_725,
        "stage": "stage_a_4k",
    },
    "evq_final_seed20260725": {
        "sha256": (
            "95ceeb70117c73233915760a9756b9b2"
            "a98416ec188b125054da8ced75cad16a"
        ),
        "frequency": "evq",
        "seed": 20_260_725,
        "stage": "counterfactual_routing_4k",
    },
    "evq_final_seed20260726": {
        "sha256": (
            "fdf6dfc249cb216c3effe22a2ee96fe4"
            "39a5aff9a11a99007f022c5fe7c3b085"
        ),
        "frequency": "evq",
        "seed": 20_260_726,
        "stage": "counterfactual_routing_4k",
    },
}
ARMS = (
    "native_unadapted",
    "native_stage_a",
    "native_final",
    "evq_injected",
    "evq_stage_a",
    "evq_final_seed20260725",
    "evq_final_seed20260726",
)
CODE_RELATIVES = (
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
    "prepare_instruct_ruler_transfer.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
    "evaluate_instruct_ruler_transfer.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
    "preflight_instruct_full_ruler.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_conversion.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_lora_ood_factorial.py",
    "rebuttal/rebuttal_0723/experiments/olmo2_1b_evq/"
    "evaluate_ruler.py",
    "scripts/lib/rope/schedules.py",
    "tests/test_olmo2_full_ruler.py",
)


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_adapter(name: str, path: Path) -> dict[str, Any]:
    expected = ADAPTERS[name]
    record = file_record(path)
    if record["sha256"] != expected["sha256"]:
        raise RuntimeError(f"adapter SHA drift: {name}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = dict(payload.get("metadata", {}))
    required = {
        "base_checkpoint_sha256": MODEL_SHA256,
        "frequency": expected["frequency"],
        "adaptation": "qkvo_answer",
        "rank": 64,
        "alpha": 128.0,
        "training_sequence_length": 4_096,
        "seed": expected["seed"],
        "stage": expected["stage"],
    }
    for field, value in required.items():
        if metadata.get(field) != value:
            raise RuntimeError(
                f"adapter metadata drift: {name}:{field}"
            )
    record["metadata"] = metadata
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-ready", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--native-stage-a", type=Path, required=True)
    parser.add_argument("--native-final", type=Path, required=True)
    parser.add_argument("--evq-stage-a", type=Path, required=True)
    parser.add_argument(
        "--evq-final-seed20260725",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--evq-final-seed20260726",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--limit-per-cell", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("preflight must hide CUDA")
    if torch.cuda.is_available():
        raise RuntimeError("preflight unexpectedly sees CUDA")
    if not 1 <= int(args.limit_per_cell) <= 100:
        raise RuntimeError("limit-per-cell must be in [1, 100]")

    checkpoint = args.checkpoint.resolve()
    parent_ready = args.parent_ready.resolve()
    data_root = args.data_root.resolve()
    code_root = args.code_root.resolve()
    output_root = args.output_root.resolve()
    receipt_path = args.receipt.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if receipt_path.exists():
        raise FileExistsError(receipt_path)

    model = file_record(checkpoint / "model.safetensors")
    tokenizer = file_record(checkpoint / "tokenizer.json")
    if model["sha256"] != MODEL_SHA256:
        raise RuntimeError("model SHA drift")
    if tokenizer["sha256"] != TOKENIZER_SHA256:
        raise RuntimeError("tokenizer SHA drift")
    if sha256_file(parent_ready) != PARENT_READY_SHA256:
        raise RuntimeError("parent READY receipt drift")

    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "status": DATA_STATUS,
        "ruler_commit": RULER_COMMIT,
        "tasks": list(DEFAULT_TASKS),
        "lengths": [4_096],
        "samples_per_cell": 100,
        "seed": 20_260_728,
    }
    for field, value in expected_manifest.items():
        if manifest.get(field) != value:
            raise RuntimeError(f"data manifest drift: {field}")
    if Path(manifest["checkpoint"]).resolve() != checkpoint:
        raise RuntimeError("data checkpoint drift")
    if manifest["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise RuntimeError("data tokenizer drift")
    data_files: dict[str, Any] = {}
    for task in DEFAULT_TASKS:
        if manifest["task_configs"][task] != TASK_CONFIGS[task]:
            raise RuntimeError(f"task config drift: {task}")
        entry = manifest["cells"][task]["4096"]
        path = data_root / entry["relative_path"]
        record = file_record(path)
        if record["sha256"] != entry["sha256"]:
            raise RuntimeError(f"data SHA drift: {task}")
        if int(entry["rows"]) != 100:
            raise RuntimeError(f"data row-count drift: {task}")
        if entry["official_metric"] != TASK_CONFIGS[task][
            "official_metric"
        ]:
            raise RuntimeError(f"metric drift: {task}")
        data_files[task] = record

    adapter_paths = {
        "native_stage_a": args.native_stage_a.resolve(),
        "native_final": args.native_final.resolve(),
        "evq_stage_a": args.evq_stage_a.resolve(),
        "evq_final_seed20260725": (
            args.evq_final_seed20260725.resolve()
        ),
        "evq_final_seed20260726": (
            args.evq_final_seed20260726.resolve()
        ),
    }
    adapters = {
        name: verify_adapter(name, path)
        for name, path in adapter_paths.items()
    }
    code = {
        relative: file_record(code_root / relative)
        for relative in CODE_RELATIVES
    }

    output_root.mkdir(parents=True)
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < 2_000_000_000:
        raise RuntimeError("less than 2 GB output space remains")
    module = (
        "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
        "evaluate_instruct_ruler_transfer"
    )
    commands: dict[str, list[str]] = {}
    for arm in ARMS:
        frequency = "evq" if arm.startswith("evq") else "native"
        command = [
            sys.executable,
            "-m",
            module,
            "--checkpoint",
            str(checkpoint),
            "--ready-receipt",
            str(parent_ready),
            "--data-root",
            str(data_root),
            "--output",
            str(output_root / arm),
            "--frequency",
            frequency,
            "--lengths",
            "4096",
            "--limit-per-cell",
            str(int(args.limit_per_cell)),
        ]
        if arm in adapter_paths:
            command.extend(("--adapter", str(adapter_paths[arm])))
        commands[arm] = command

    receipt = {
        "format_version": 1,
        "status": "OLMO2_INSTRUCT_FULL_RULER_4K_READY",
        "review_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "scientific_contract": {
            "role": (
                "internal complete-task diagnostic for training design; "
                "reviewer-facing reporting may select only directly "
                "responsive results with explicit scope"
            ),
            "existing_evidence": (
                "fresh all-long-gap 8K n=100: Native 0/100, EVQ "
                "49/100 and 48/100 on niah_single_1"
            ),
            "smallest_missing_evidence": (
                "identify which official RULER capabilities are present, "
                "lost at EVQ injection, recovered by natural Stage A, or "
                "learned by routing adaptation"
            ),
            "training": "none",
            "tasks": list(DEFAULT_TASKS),
            "lengths": [4_096],
            "examples_per_cell": int(args.limit_per_cell),
            "arms": list(ARMS),
            "stop_condition": (
                "complete all seven frozen stages; do not launch a new "
                "training recipe until the task-wise failure profile is "
                "summarized"
            ),
        },
        "checkpoint": {"model": model, "tokenizer": tokenizer},
        "parent_ready": file_record(parent_ready),
        "data": {
            "manifest": file_record(manifest_path),
            "files": data_files,
            "external_assets": manifest["external_assets"],
        },
        "adapters": adapters,
        "code": code,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
        },
        "storage": {
            "output_root": str(output_root),
            "free_bytes": int(free_bytes),
        },
        "commands": commands,
    }
    atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(receipt_path),
                "sha256": sha256_file(receipt_path),
                "tasks": len(DEFAULT_TASKS),
                "arms": len(ARMS),
                "examples": (
                    len(DEFAULT_TASKS)
                    * len(ARMS)
                    * int(args.limit_per_cell)
                ),
                "free_bytes": int(free_bytes),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
