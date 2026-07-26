#!/usr/bin/env python3
"""Freeze the matched 8K n=100 RULER extension and paired statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from .freeze_conversion_evidence import (
    extract_key_values,
    overlap,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)


def canonical(row: dict[str, Any]) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def rows_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(canonical(row) for row in rows).encode("utf-8")
    ).hexdigest()


def wilson_interval(correct: int, total: int) -> list[float]:
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (
        proportion + z * z / (2.0 * total)
    ) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half, center + half]


def mcnemar_exact_two_sided(
    evq_only: int, native_only: int
) -> float:
    discordant = evq_only + native_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(evq_only, native_only) + 1)
    )
    return min(1.0, 2.0 * (0.5**discordant) * tail)


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
    source_digest = sha256_file(source)
    if sha256_file(destination) != source_digest:
        raise RuntimeError(f"copy digest drift: {relative_path}")
    return {
        "relative_path": relative_path,
        "source_path": str(source),
        "frozen_path": str(destination),
        "bytes": source.stat().st_size,
        "sha256": source_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--parent-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = args.base.resolve()
    parent = args.parent_freeze.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    parent_receipt = read_json(parent / "FREEZE_RECEIPT.json")
    if (
        parent_receipt.get("status")
        != "OLMO2_MATCHED_CONVERSION_EVIDENCE_FROZEN"
    ):
        raise RuntimeError("parent evidence freeze is not complete")
    parent_summary = read_json(parent / "formal_summary.json")

    data_relative = (
        "data/ruler_olmo2_screen_v3_n100/L8192/"
        "niah_single_1/test.jsonl"
    )
    relatives = [
        "data/ruler_olmo2_screen_v3_n100/manifest.json",
        data_relative,
    ]
    for arm in ("native", "evq"):
        stage = (
            f"instruct_{arm}_counterfactual_routing_ruler_niah_"
            "8k100_s20260725"
        )
        relatives.extend(
            [
                f"runs/{stage}/results.json",
                f"runs/{stage}/examples.jsonl",
            ]
        )
    inventory = [
        copy_one(base=base, output=output, relative_path=relative)
        for relative in relatives
    ]

    old_data = read_jsonl(
        base
        / (
            "data/ruler_olmo2_screen_v2/L8192/"
            "niah_single_1/test.jsonl"
        )
    )
    new_data = read_jsonl(base / data_relative)
    if len(old_data) != 20 or len(new_data) != 100:
        raise RuntimeError("RULER extension row-count drift")
    if any(
        canonical(old) != canonical(new)
        for old, new in zip(old_data, new_data[:20])
    ):
        raise RuntimeError("n=100 first 20 rows differ from n=20")

    arm_examples = {}
    arm_results = {}
    prefix_fields = (
        "nominal_length",
        "local_index",
        "source_row_index",
        "row_sha256",
        "references",
        "prediction",
        "official_string_match",
        "first_number_exact",
    )
    for arm in ("native", "evq"):
        old_stage = (
            f"instruct_{arm}_counterfactual_routing_ruler_niah_"
            "20_s20260725"
        )
        new_stage = (
            f"instruct_{arm}_counterfactual_routing_ruler_niah_"
            "8k100_s20260725"
        )
        old_examples = [
            row
            for row in read_jsonl(
                base / f"runs/{old_stage}/examples.jsonl"
            )
            if int(row["nominal_length"]) == 8_192
        ]
        new_examples = read_jsonl(
            base / f"runs/{new_stage}/examples.jsonl"
        )
        if len(old_examples) != 20 or len(new_examples) != 100:
            raise RuntimeError(f"{arm} example-count drift")
        if any(
            any(old.get(field) != new.get(field) for field in prefix_fields)
            for old, new in zip(old_examples, new_examples[:20])
        ):
            raise RuntimeError(f"{arm} first-20 prediction drift")
        result = read_json(base / f"runs/{new_stage}/results.json")
        expected_adapter = parent_summary["arms"][arm][
            "final_adapter_sha256"
        ]
        if result["adapter"]["sha256"] != expected_adapter:
            raise RuntimeError(f"{arm} adapter hash drift")
        if (
            result["data"]["files"]["8192"]["sha256"]
            != sha256_file(base / data_relative)
        ):
            raise RuntimeError(f"{arm} data hash drift")
        arm_examples[arm] = new_examples
        arm_results[arm] = result

    if [
        row["row_sha256"] for row in arm_examples["native"]
    ] != [row["row_sha256"] for row in arm_examples["evq"]]:
        raise RuntimeError("Native and EVQ examples are not paired")

    strict = {
        arm: sum(
            int(bool(row["first_number_exact"]))
            for row in arm_examples[arm]
        )
        for arm in ("native", "evq")
    }
    substring = {
        arm: sum(
            int(bool(row["official_string_match"]))
            for row in arm_examples[arm]
        )
        for arm in ("native", "evq")
    }
    new80_strict = {
        arm: sum(
            int(bool(row["first_number_exact"]))
            for row in arm_examples[arm][20:]
        )
        for arm in ("native", "evq")
    }
    evq_only = sum(
        bool(evq["first_number_exact"])
        and not bool(native["first_number_exact"])
        for native, evq in zip(
            arm_examples["native"], arm_examples["evq"]
        )
    )
    native_only = sum(
        bool(native["first_number_exact"])
        and not bool(evq["first_number_exact"])
        for native, evq in zip(
            arm_examples["native"], arm_examples["evq"]
        )
    )

    routing_root = base / "data/routing_pairs_4k_v1"
    n100_values = extract_key_values(base / data_relative)
    train_values = extract_key_values(
        routing_root / "raw_train/routing_train/test.jsonl"
    )
    calibration_values = extract_key_values(
        routing_root
        / "raw_calibration/routing_calibration/test.jsonl"
    )
    overlap_receipt = {
        split: {
            field: overlap(
                source[field],
                n100_values[field],
            )
            for field in (
                "source_keys",
                "source_values",
                "queries",
                "answers",
            )
        }
        for split, source in (
            ("train", train_values),
            ("calibration", calibration_values),
        )
    }

    summary = {
        "status": "OLMO2_8K_N100_MATCHED_EXTENSION_FROZEN",
        "metric_boundary": (
            "single seed; official RULER niah_single_1 only; "
            "autoregressive strict first-number exact is primary"
        ),
        "parent_freeze": {
            "path": str(parent),
            "inventory_sha256": parent_receipt["inventory_sha256"],
        },
        "data": {
            "path": str(base / data_relative),
            "sha256": sha256_file(base / data_relative),
            "manifest_sha256": sha256_file(
                base / "data/ruler_olmo2_screen_v3_n100/manifest.json"
            ),
            "rows": 100,
            "first20_exactly_match_prior_data": True,
            "first20_canonical_sha256": rows_sha256(new_data[:20]),
            "new80_canonical_sha256": rows_sha256(new_data[20:]),
            "train_calibration_overlap": overlap_receipt,
        },
        "arms": {
            arm: {
                "adapter_sha256": arm_results[arm]["adapter"]["sha256"],
                "strict_first_number_exact": strict[arm] / 100.0,
                "strict_correct": strict[arm],
                "strict_wilson_95": wilson_interval(
                    strict[arm], 100
                ),
                "official_substring_exact": substring[arm] / 100.0,
                "new80_strict_correct": new80_strict[arm],
                "new80_strict_exact": new80_strict[arm] / 80.0,
                "examples_sha256": arm_results[arm]["results"][
                    "examples_sha256"
                ],
                "first20_predictions_match_prior_run": True,
            }
            for arm in ("native", "evq")
        },
        "paired_comparison": {
            "evq_only_correct": evq_only,
            "native_only_correct": native_only,
            "strict_difference_percentage_points": (
                strict["evq"] - strict["native"]
            ),
            "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(
                evq_only, native_only
            ),
        },
    }
    write_json(output / "n100_summary.json", summary)
    inventory.append(
        {
            "relative_path": "n100_summary.json",
            "source_path": None,
            "frozen_path": str(output / "n100_summary.json"),
            "bytes": (output / "n100_summary.json").stat().st_size,
            "sha256": sha256_file(output / "n100_summary.json"),
        }
    )
    write_json(
        output / "inventory.json",
        {
            "status": "IMMUTABLE_N100_EVIDENCE_INVENTORY",
            "files": sorted(
                inventory, key=lambda item: item["relative_path"]
            ),
        },
    )
    inventory_digest = sha256_file(output / "inventory.json")
    write_json(
        output / "FREEZE_RECEIPT.json",
        {
            "status": "OLMO2_8K_N100_MATCHED_EXTENSION_FROZEN",
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
