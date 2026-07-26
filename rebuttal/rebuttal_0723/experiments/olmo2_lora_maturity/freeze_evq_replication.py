#!/usr/bin/env python3
"""Freeze one EVQ LoRA seed replication and its matched 8K n=100 eval."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from .freeze_conversion_evidence import (
    read_json,
    read_jsonl,
    reconstruct_training_order,
    sha256_file,
    write_json,
)
from .freeze_n100_extension import wilson_interval


def copy_one(
    *,
    base: Path,
    output: Path,
    relative_path: str,
) -> dict[str, Any]:
    source = base / relative_path
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output / "artifacts" / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha256_file(source)
    if sha256_file(destination) != digest:
        raise RuntimeError(f"copy digest drift: {relative_path}")
    return {
        "relative_path": relative_path,
        "source_path": str(source),
        "frozen_path": str(destination),
        "bytes": source.stat().st_size,
        "sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--parent-freeze", type=Path, required=True)
    parser.add_argument("--n100-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference-seed", type=int, default=20_260_725)
    args = parser.parse_args()

    base = args.base.resolve()
    parent = args.parent_freeze.resolve()
    n100_parent = args.n100_freeze.resolve()
    output = args.output.resolve()
    seed = int(args.seed)
    reference_seed = int(args.reference_seed)
    if output.exists():
        raise FileExistsError(output)

    parent_receipt = read_json(parent / "FREEZE_RECEIPT.json")
    n100_receipt = read_json(n100_parent / "FREEZE_RECEIPT.json")
    if (
        parent_receipt.get("status")
        != "OLMO2_MATCHED_CONVERSION_EVIDENCE_FROZEN"
    ):
        raise RuntimeError("matched parent evidence freeze is incomplete")
    if (
        n100_receipt.get("status")
        != "OLMO2_8K_N100_MATCHED_EXTENSION_FROZEN"
    ):
        raise RuntimeError("n=100 parent evidence freeze is incomplete")

    output.mkdir(parents=True)
    stage_a = f"instruct_evq_stage_a_4k_20m_s{seed}"
    routing = f"instruct_evq_counterfactual_routing_4k_300_s{seed}"
    evaluation = (
        f"instruct_evq_counterfactual_routing_ruler_niah_8k100_s{seed}"
    )
    relatives = []
    for stage in (stage_a, routing):
        for filename in ("adapter.pt", "results.json", "train_log.jsonl"):
            relatives.append(f"runs/{stage}/{filename}")
    for filename in ("results.json", "examples.jsonl"):
        relatives.append(f"runs/{evaluation}/{filename}")
    inventory = [
        copy_one(base=base, output=output, relative_path=relative)
        for relative in relatives
    ]

    stage_result = read_json(base / f"runs/{stage_a}/results.json")
    routing_result = read_json(base / f"runs/{routing}/results.json")
    eval_result = read_json(base / f"runs/{evaluation}/results.json")
    examples = read_jsonl(base / f"runs/{evaluation}/examples.jsonl")
    if len(examples) != 100:
        raise RuntimeError("replication evaluation is not n=100")
    if (
        eval_result["adapter"]["sha256"]
        != routing_result["adapter_sha256"]
    ):
        raise RuntimeError("replication eval adapter hash drift")
    if (
        stage_result["frequency"]["active_sha256_float32"]
        != routing_result["frequency"]["active_sha256_float32"]
    ):
        raise RuntimeError("replication frequency hash drift")

    reference_path = (
        base
        / (
            "runs/instruct_evq_counterfactual_routing_ruler_niah_"
            f"8k100_s{reference_seed}/examples.jsonl"
        )
    )
    reference_examples = read_jsonl(reference_path)
    if [row["row_sha256"] for row in examples] != [
        row["row_sha256"] for row in reference_examples
    ]:
        raise RuntimeError("replication is not evaluated on the same rows")

    strict = sum(
        int(bool(row["first_number_exact"])) for row in examples
    )
    substring = sum(
        int(bool(row["official_string_match"])) for row in examples
    )
    both = sum(
        bool(left["first_number_exact"])
        and bool(right["first_number_exact"])
        for left, right in zip(reference_examples, examples)
    )
    reference_only = sum(
        bool(left["first_number_exact"])
        and not bool(right["first_number_exact"])
        for left, right in zip(reference_examples, examples)
    )
    replication_only = sum(
        bool(right["first_number_exact"])
        and not bool(left["first_number_exact"])
        for left, right in zip(reference_examples, examples)
    )
    neither = 100 - both - reference_only - replication_only

    order = reconstruct_training_order(base, seed)
    write_json(output / "training_order.json", order)
    summary = {
        "status": "OLMO2_EVQ_SEED_REPLICATION_FROZEN",
        "metric_boundary": (
            "second EVQ LoRA training seed; same official RULER "
            "niah_single_1 rows; autoregressive strict first-number "
            "exact is primary; this is seed stability, not a new task"
        ),
        "seed": seed,
        "reference_seed": reference_seed,
        "parent_freezes": {
            "matched_native_evq": {
                "path": str(parent),
                "inventory_sha256": parent_receipt["inventory_sha256"],
            },
            "matched_8k_n100": {
                "path": str(n100_parent),
                "inventory_sha256": n100_receipt["inventory_sha256"],
            },
        },
        "checkpoint_sha256": routing_result["checkpoint_sha256"],
        "frequency_sha256_float32": routing_result["frequency"][
            "active_sha256_float32"
        ],
        "stage_a": {
            "adapter_sha256": stage_result["adapter_sha256"],
            "natural_nll": stage_result["natural_nll"],
            "training": stage_result["training"],
        },
        "counterfactual": {
            "adapter_sha256": routing_result["adapter_sha256"],
            "initial_routing_calibration": routing_result[
                "initial_routing_calibration"
            ],
            "final_routing_calibration": routing_result[
                "final_routing_calibration"
            ],
            "natural_nll": routing_result["natural_nll"],
            "training": routing_result["training"],
        },
        "optimizer_protocol": {
            "optimizer": "torch.optim.AdamW",
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "fused": True,
            "stage_a_learning_rate": 1e-4,
            "stage_a_warmup_steps": 31,
            "counterfactual_learning_rate": 5e-5,
            "counterfactual_warmup_steps": 20,
            "schedule": "cosine",
            "optimizer_state_saved": False,
        },
        "evaluation_8k_n100": {
            "rows": 100,
            "strict_correct": strict,
            "strict_first_number_exact": strict / 100.0,
            "strict_wilson_95": wilson_interval(strict, 100),
            "official_substring_correct": substring,
            "official_substring_exact": substring / 100.0,
            "examples_sha256": eval_result["results"][
                "examples_sha256"
            ],
            "same_row_order_as_reference": True,
        },
        "paired_seed_stability": {
            "both_correct": both,
            "reference_seed_only_correct": reference_only,
            "replication_seed_only_correct": replication_only,
            "neither_correct": neither,
        },
    }
    write_json(output / "replication_summary.json", summary)

    for path in (
        output / "training_order.json",
        output / "replication_summary.json",
    ):
        inventory.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "source_path": None,
                "frozen_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(
        output / "inventory.json",
        {
            "status": "IMMUTABLE_EVQ_REPLICATION_INVENTORY",
            "files": sorted(
                inventory, key=lambda item: item["relative_path"]
            ),
        },
    )
    inventory_digest = sha256_file(output / "inventory.json")
    write_json(
        output / "FREEZE_RECEIPT.json",
        {
            "status": "OLMO2_EVQ_SEED_REPLICATION_FROZEN",
            "output": str(output),
            "inventory_sha256": inventory_digest,
            "file_count": len(inventory),
        },
    )

    for root, directories, files in os.walk(output):
        for name in files:
            os.chmod(
                Path(root) / name,
                stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH,
            )
        for name in directories:
            os.chmod(
                Path(root) / name,
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH,
            )
    os.chmod(
        output,
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
