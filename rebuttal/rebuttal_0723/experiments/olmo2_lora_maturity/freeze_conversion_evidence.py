#!/usr/bin/env python3
"""Freeze completed Native/EVQ conversion evidence without altering sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any

import numpy as np
import torch


SOURCE_PAIR = re.compile(
    r"(?:One of the )?special magic numbers? for ([^\n:]+?) is: ([0-9]+)"
)
QUERY_PATTERNS = (
    re.compile(r"Which access code belongs to ([^?]+)\?"),
    re.compile(r"Return the identifier assigned to ([^.]+)\."),
    re.compile(
        r"What is the special magic number for (.+?) "
        r"mentioned in the provided text\?"
    ),
)
FAMILY_PATTERN = ("routing", "routing", "natural")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def copy_one(
    *,
    base: Path,
    output: Path,
    relative_path: str,
    inventory: list[dict[str, Any]],
    required: bool = True,
) -> None:
    source = base / relative_path
    if not source.is_file():
        if required:
            raise FileNotFoundError(source)
        return
    destination = output / "artifacts" / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_digest = sha256_file(source)
    destination_digest = sha256_file(destination)
    if source_digest != destination_digest:
        raise RuntimeError(f"copy digest drift: {relative_path}")
    inventory.append(
        {
            "relative_path": relative_path,
            "source_path": str(source),
            "frozen_path": str(destination),
            "bytes": source.stat().st_size,
            "sha256": source_digest,
        }
    )


def extract_key_values(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    source_keys: list[str] = []
    source_values: list[str] = []
    queries: list[str] = []
    answers: list[str] = []
    for row in rows:
        pairs = SOURCE_PAIR.findall(str(row["input"]))
        source_keys.extend(key.strip() for key, _ in pairs)
        source_values.extend(value for _, value in pairs)
        query = None
        for pattern in QUERY_PATTERNS:
            match = pattern.search(str(row["input"]))
            if match is not None:
                query = match.group(1).strip()
                break
        if query is None:
            raise RuntimeError(f"could not parse query in {path}")
        queries.append(query)
        answers.extend(str(value) for value in row["outputs"])
    return {
        "rows": len(rows),
        "source_keys": sorted(set(source_keys)),
        "source_values": sorted(set(source_values)),
        "queries": sorted(set(queries)),
        "answers": sorted(set(answers)),
    }


def overlap(left: list[str], right: list[str]) -> dict[str, Any]:
    values = sorted(set(left) & set(right))
    return {"count": len(values), "values": values}


def reconstruct_training_order(base: Path, seed: int) -> dict[str, Any]:
    view = base / "data/prepared_maturity_v1/longalign_paired_L4096"
    split = np.load(view / "split.npy", allow_pickle=False)
    natural_rows = np.flatnonzero(split == 0)

    stage_generator = torch.Generator(device="cpu")
    stage_generator.manual_seed(seed + 40_001)
    stage_steps: list[list[list[int]]] = []
    for _ in range(611):
        micro_steps: list[list[int]] = []
        for _ in range(2):
            selection = torch.randint(
                len(natural_rows),
                (4,),
                generator=stage_generator,
            ).numpy()
            micro_steps.append(
                [int(value) for value in natural_rows[selection]]
            )
        stage_steps.append(micro_steps)

    routing_rows_path = (
        base / "data/routing_pairs_4k_v1/train/rows.jsonl"
    )
    routing_count = sum(
        1
        for line in routing_rows_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )
    route_generator = torch.Generator(device="cpu")
    route_generator.manual_seed(seed + 71_001)
    route_steps: list[dict[str, Any]] = []
    for step in range(1, 301):
        family = FAMILY_PATTERN[(step - 1) % len(FAMILY_PATTERN)]
        micro_steps = []
        for _ in range(2):
            if family == "routing":
                indices = torch.randint(
                    routing_count,
                    (2,),
                    generator=route_generator,
                ).numpy()
            else:
                selection = torch.randint(
                    len(natural_rows),
                    (4,),
                    generator=route_generator,
                ).numpy()
                indices = natural_rows[selection]
            micro_steps.append([int(value) for value in indices])
        route_steps.append(
            {
                "step": step,
                "family": family,
                "micro_batches": micro_steps,
            }
        )

    return {
        "status": "DETERMINISTIC_TRAINING_ORDER_RECONSTRUCTED",
        "scope": (
            "identical for the matched Native and EVQ arms because both "
            "used the same seed, row pools, and sampling code"
        ),
        "seed": seed,
        "stage_a": {
            "generator_seed": seed + 40_001,
            "steps": 611,
            "gradient_accumulation_steps": 2,
            "micro_batch_size": 4,
            "natural_training_row_count": len(natural_rows),
            "sampled_row_indices": stage_steps,
        },
        "counterfactual": {
            "generator_seed": seed + 71_001,
            "steps": 300,
            "family_pattern": list(FAMILY_PATTERN),
            "gradient_accumulation_steps": 2,
            "micro_batch_size": 4,
            "routing_pair_batch_size": 2,
            "routing_row_count": routing_count,
            "natural_training_row_count": len(natural_rows),
            "sampled_row_indices": route_steps,
        },
    }


def formal_summary(base: Path) -> dict[str, Any]:
    arms = {}
    for name in ("native", "evq"):
        stage = read_json(
            base
            / f"runs/instruct_{name}_stage_a_4k_20m_s20260725/results.json"
        )
        final = read_json(
            base
            / (
                f"runs/instruct_{name}_counterfactual_routing_4k_300_"
                "s20260725/results.json"
            )
        )
        ruler = read_json(
            base
            / (
                f"runs/instruct_{name}_counterfactual_routing_ruler_"
                "niah_20_s20260725/results.json"
            )
        )
        arms[name] = {
            "stage_a_adapter_sha256": stage["adapter_sha256"],
            "final_adapter_sha256": final["adapter_sha256"],
            "frequency_sha256_float32": final["frequency"][
                "active_sha256_float32"
            ],
            "initial_routing_calibration": final[
                "initial_routing_calibration"
            ],
            "final_routing_calibration": final[
                "final_routing_calibration"
            ],
            "natural_nll": final["natural_nll"],
            "ruler_protocol": ruler["protocol"],
            "ruler_results": ruler["results"],
            "ruler_examples_sha256": ruler["results"][
                "examples_sha256"
            ],
        }
    return {
        "status": "OLMO2_MATCHED_CONVERSION_EVIDENCE_FROZEN",
        "metric_boundary": (
            "single seed; RULER niah_single_1 only; strict first-number "
            "exact is primary, official substring match is secondary"
        ),
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_260_725)
    args = parser.parse_args()

    base = args.base.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    inventory: list[dict[str, Any]] = []

    run_files = []
    for name in ("native", "evq"):
        for stage in (
            f"instruct_{name}_stage_a_4k_20m_s20260725",
            f"instruct_{name}_counterfactual_routing_4k_300_s20260725",
        ):
            for filename in ("adapter.pt", "results.json", "train_log.jsonl"):
                run_files.append(f"runs/{stage}/{filename}")
        ruler_stage = (
            f"instruct_{name}_counterfactual_routing_ruler_niah_20_"
            "s20260725"
        )
        for filename in ("results.json", "examples.jsonl"):
            run_files.append(f"runs/{ruler_stage}/{filename}")

    diagnostic_runs = (
        "instruct_native_ruler_niah_20_s20260725",
        "instruct_evq_unadapted_ruler_niah_20_s20260725",
        "instruct_evq_stage_a_ruler_niah_20_s20260725",
    )
    for stage in diagnostic_runs:
        for filename in ("results.json", "examples.jsonl"):
            run_files.append(f"runs/{stage}/{filename}")

    data_files = (
        "data/routing_pairs_4k_v1/manifest.json",
        "data/routing_pairs_4k_v1/raw_train/routing_train/test.jsonl",
        (
            "data/routing_pairs_4k_v1/raw_calibration/"
            "routing_calibration/test.jsonl"
        ),
        "data/routing_pairs_4k_v1/train/manifest.json",
        "data/routing_pairs_4k_v1/train/rows.jsonl",
        "data/routing_pairs_4k_v1/train/input_ids.npy",
        "data/routing_pairs_4k_v1/train/labels.npy",
        "data/routing_pairs_4k_v1/calibration/manifest.json",
        "data/routing_pairs_4k_v1/calibration/rows.jsonl",
        "data/routing_pairs_4k_v1/calibration/input_ids.npy",
        "data/routing_pairs_4k_v1/calibration/labels.npy",
        "data/ruler_olmo2_screen_v2/manifest.json",
        (
            "data/ruler_olmo2_screen_v2/L4096/niah_single_1/"
            "test.jsonl"
        ),
        (
            "data/ruler_olmo2_screen_v2/L8192/niah_single_1/"
            "test.jsonl"
        ),
        (
            "data/ruler_olmo2_screen_v2/L16384/niah_single_1/"
            "test.jsonl"
        ),
        (
            "data/prepared_maturity_v1/longalign_paired_L4096/"
            "manifest.json"
        ),
        (
            "data/prepared_maturity_v1/longalign_paired_L4096/"
            "split.npy"
        ),
        (
            "data/prepared_maturity_v1/longalign_paired_L4096/"
            "rows.jsonl"
        ),
        "receipts/instruct_4k_conversion_ready.json",
    )
    for relative_path in (*run_files, *data_files):
        copy_one(
            base=base,
            output=output,
            relative_path=relative_path,
            inventory=inventory,
        )

    code_root = base / "code"
    code_files = (
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "train_4k_stage_a.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "train_4k_counterfactual_routing.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "prepare_4k_routing_pairs.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "evaluate_instruct_ruler_screen.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/"
        "train_screen.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_conversion.py",
        "rebuttal/rebuttal_0723/experiments/olmo2_lora_ood_factorial.py",
    )
    for relative_path in code_files:
        source = code_root / relative_path
        destination = output / "code" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        inventory.append(
            {
                "relative_path": f"code/{relative_path}",
                "source_path": str(source),
                "frozen_path": str(destination),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )

    order = reconstruct_training_order(base, int(args.seed))
    write_json(output / "training_order.json", order)

    routing_root = base / "data/routing_pairs_4k_v1"
    ruler_root = base / "data/ruler_olmo2_screen_v2"
    named_sets = {
        "train": extract_key_values(
            routing_root / "raw_train/routing_train/test.jsonl"
        ),
        "calibration": extract_key_values(
            routing_root
            / "raw_calibration/routing_calibration/test.jsonl"
        ),
        "eval_4k": extract_key_values(
            ruler_root / "L4096/niah_single_1/test.jsonl"
        ),
        "eval_8k": extract_key_values(
            ruler_root / "L8192/niah_single_1/test.jsonl"
        ),
        "eval_16k": extract_key_values(
            ruler_root / "L16384/niah_single_1/test.jsonl"
        ),
    }
    alternate_sets = {}
    for split_name in ("train", "calibration"):
        rows = read_jsonl(
            routing_root / split_name / "rows.jsonl"
        )
        alternate_sets[split_name] = sorted(
            {str(row["alternate_value"]) for row in rows}
        )
        named_sets[split_name]["alternate_values"] = alternate_sets[
            split_name
        ]

    pairwise = {}
    names = list(named_sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            pairwise[f"{left_name}__{right_name}"] = {
                field: overlap(
                    named_sets[left_name][field],
                    named_sets[right_name][field],
                )
                for field in (
                    "source_keys",
                    "source_values",
                    "queries",
                    "answers",
                )
            }
    overlap_receipt = {
        "status": "VALUE_QUERY_OVERLAP_AUDITED",
        "sets": named_sets,
        "pairwise": pairwise,
        "interpretation": (
            "training/calibration versus official evaluation must have "
            "zero query and answer overlap; cross-length evaluation may "
            "share rows and is a paired length comparison"
        ),
    }
    write_json(output / "value_query_overlap.json", overlap_receipt)

    summary = formal_summary(base)
    summary["optimizer_protocol"] = {
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
        "note": (
            "the completed runs did not persist Adam moments; exact "
            "optimizer configuration, seed, code, data, and sample order "
            "are frozen for deterministic rerun"
        ),
    }
    summary["seed"] = int(args.seed)
    write_json(output / "formal_summary.json", summary)

    generated = (
        output / "training_order.json",
        output / "value_query_overlap.json",
        output / "formal_summary.json",
    )
    for path in generated:
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
            "status": "IMMUTABLE_EVIDENCE_INVENTORY",
            "base": str(base),
            "files": sorted(
                inventory, key=lambda item: item["relative_path"]
            ),
        },
    )
    inventory_digest = sha256_file(output / "inventory.json")
    write_json(
        output / "FREEZE_RECEIPT.json",
        {
            "status": "OLMO2_MATCHED_CONVERSION_EVIDENCE_FROZEN",
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
    print(
        json.dumps(
            {
                "status": "OLMO2_MATCHED_CONVERSION_EVIDENCE_FROZEN",
                "output": str(output),
                "inventory_sha256": inventory_digest,
                "file_count": len(inventory),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
