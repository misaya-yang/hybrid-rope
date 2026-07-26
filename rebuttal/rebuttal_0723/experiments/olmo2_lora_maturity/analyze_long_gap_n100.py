#!/usr/bin/env python3
"""Validate and summarize the fresh all-long-gap 8K RULER n=100 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"\d+")
ARM_NAMES = ("native", "evq_seed_a", "evq_seed_b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive total")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - radius, center + radius]


def exact_mcnemar(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    smaller = min(a_only, b_only)
    tail = sum(
        math.comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def strict_correct(row: dict[str, Any]) -> bool:
    return float(row["first_number_exact"]) == 1.0


def classify_prediction(row: dict[str, Any]) -> str:
    gold = str(row["references"][0])
    match = NUMBER_RE.search(str(row["prediction"]))
    if match is None:
        return "no_number"
    if match.group(0) == gold:
        return "strict_exact"
    if float(row["official_string_match"]) == 1.0:
        return "official_only_wrong_first_number"
    return "wrong_first_number"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = sum(strict_correct(row) for row in rows)
    official = sum(
        float(row["official_string_match"]) == 1.0 for row in rows
    )
    classes: dict[str, int] = {}
    for row in rows:
        label = classify_prediction(row)
        classes[label] = classes.get(label, 0) + 1
    return {
        "examples": len(rows),
        "strict_exact": exact,
        "strict_exact_rate": exact / len(rows),
        "strict_exact_wilson_95": wilson_interval(exact, len(rows)),
        "official_string_match": official,
        "official_string_match_rate": official / len(rows),
        "prediction_classes": dict(sorted(classes.items())),
    }


def paired_summary(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(rows_a) != len(rows_b):
        raise ValueError("paired arms have different row counts")
    both_correct = a_only = b_only = both_wrong = 0
    for row_a, row_b in zip(rows_a, rows_b):
        a_correct = strict_correct(row_a)
        b_correct = strict_correct(row_b)
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct:
            a_only += 1
        elif b_correct:
            b_only += 1
        else:
            both_wrong += 1
    total = len(rows_a)
    return {
        "examples": total,
        "both_correct": both_correct,
        "arm_a_only": a_only,
        "arm_b_only": b_only,
        "both_wrong": both_wrong,
        "agreement_rate": (both_correct + both_wrong) / total,
        "exact_mcnemar_two_sided_p": exact_mcnemar(a_only, b_only),
    }


def gap_band(gap: int) -> str:
    if gap <= 4095:
        return "3934-4095"
    if gap <= 5119:
        return "4096-5119"
    if gap <= 6143:
        return "5120-6143"
    if gap <= 7167:
        return "6144-7167"
    return "7168+"


def validate_and_analyze(
    *,
    manifest_path: Path,
    data_path: Path,
    result_paths: dict[str, Path],
    example_paths: dict[str, Path],
) -> dict[str, Any]:
    if set(result_paths) != set(ARM_NAMES):
        raise ValueError("result arm names do not match the registered arms")
    if set(example_paths) != set(ARM_NAMES):
        raise ValueError("example arm names do not match the registered arms")

    manifest = read_json(manifest_path)
    if (
        manifest.get("status")
        != "OLMO2_INSTRUCT_RULER_LONG_GAP_SCREEN_PREPARED"
    ):
        raise RuntimeError("long-gap data manifest has the wrong status")
    if manifest.get("task") != "niah_single_1":
        raise RuntimeError("long-gap task drift")
    if manifest.get("lengths") != [8192]:
        raise RuntimeError("long-gap length drift")
    if int(manifest.get("samples_per_length", 0)) != 100:
        raise RuntimeError("long-gap sample-count drift")

    data_rows = read_jsonl(data_path)
    if len(data_rows) != 100:
        raise RuntimeError("long-gap data file must contain 100 rows")
    data_sha = sha256_file(data_path)
    registered_data = manifest["files"]["8192"]
    if data_sha != registered_data["sha256"]:
        raise RuntimeError("long-gap data hash drift")
    if int(registered_data["rows"]) != 100:
        raise RuntimeError("manifest row-count drift")

    maximum_training_gap = int(
        manifest["training_gap_support"]["maximum_tokens"]
    )
    if maximum_training_gap != 3933:
        raise RuntimeError("training-gap support drift")
    overlap = manifest["selected_overlap_with_forbidden"]
    for field in ("queries", "source_keys", "source_values", "answers"):
        if int(overlap[field]["count"]) != 0:
            raise RuntimeError(f"forbidden {field} overlap is nonzero")

    results = {name: read_json(path) for name, path in result_paths.items()}
    arms = {name: read_jsonl(path) for name, path in example_paths.items()}
    reference_rows = arms["native"]
    identity_fields = (
        "local_index",
        "row_sha256",
        "references",
        "source_row_index",
        "source_token_position_answer",
        "input_tokens",
        "nominal_length",
        "task",
    )
    for name in ARM_NAMES:
        result = results[name]
        rows = arms[name]
        if result.get("status") != "OLMO2_INSTRUCT_RULER_SCREEN_COMPLETE":
            raise RuntimeError(f"{name} result is incomplete")
        if len(rows) != 100:
            raise RuntimeError(f"{name} does not have 100 predictions")
        if result["data"]["files"]["8192"]["sha256"] != data_sha:
            raise RuntimeError(f"{name} data hash drift")
        if result["data"]["manifest_sha256"] != sha256_file(manifest_path):
            raise RuntimeError(f"{name} manifest hash drift")
        if result["results"]["examples_sha256"] != sha256_file(
            example_paths[name]
        ):
            raise RuntimeError(f"{name} example hash drift")
        reported = float(
            result["results"]["cells"]["8192"]["first_number_exact"]
        )
        observed = sum(strict_correct(row) for row in rows) / len(rows)
        if reported != observed:
            raise RuntimeError(f"{name} aggregate drift")
        for index, (reference, candidate, data_row) in enumerate(
            zip(reference_rows, rows, data_rows)
        ):
            for field in identity_fields:
                if candidate[field] != reference[field]:
                    raise RuntimeError(
                        f"pairing drift arm={name} row={index} field={field}"
                    )
            if int(candidate["local_index"]) != index:
                raise RuntimeError(f"local index drift at row {index}")
            if int(candidate["source_row_index"]) != int(data_row["index"]):
                raise RuntimeError(f"source row drift at row {index}")
            if candidate["references"] != data_row["outputs"]:
                raise RuntimeError(f"reference drift at row {index}")
            if int(candidate["source_token_position_answer"]) != int(
                data_row["token_position_answer"]
            ):
                raise RuntimeError(f"source position drift at row {index}")
            gap = int(candidate["input_tokens"]) - int(
                candidate["source_token_position_answer"]
            )
            if gap <= maximum_training_gap:
                raise RuntimeError(
                    f"row {index} gap {gap} is not beyond training support"
                )

    expected_frequency = {
        "native": (
            "native_endpoint_rope",
            "dde15c31724177356ae954d6e11fb337"
            "e6fccef56e4520a905cac3f0d9885b34",
        ),
        "evq_seed_a": (
            "evq_endpoint_cosh",
            "917a52426b4ac986545c8ec73b115daa"
            "e3c6515d6b9047f09d30c972ea1a4607",
        ),
        "evq_seed_b": (
            "evq_endpoint_cosh",
            "917a52426b4ac986545c8ec73b115daa"
            "e3c6515d6b9047f09d30c972ea1a4607",
        ),
    }
    for name, (frequency_name, frequency_sha) in expected_frequency.items():
        if results[name]["frequency"]["active_frequency"] != frequency_name:
            raise RuntimeError(f"{name} frequency identity drift")
        if (
            results[name]["frequency"]["active_sha256_float32"]
            != frequency_sha
        ):
            raise RuntimeError(f"{name} frequency hash drift")

    gaps = [
        int(row["input_tokens"])
        - int(row["source_token_position_answer"])
        for row in reference_rows
    ]
    band_labels = (
        "3934-4095",
        "4096-5119",
        "5120-6143",
        "6144-7167",
        "7168+",
    )
    band_summaries: dict[str, dict[str, Any]] = {}
    for name, rows in arms.items():
        band_summaries[name] = {}
        for label in band_labels:
            selected = [
                row
                for row, gap in zip(rows, gaps)
                if gap_band(gap) == label
            ]
            band_summaries[name][label] = summarize_rows(selected)

    per_example = []
    for native, seed_a, seed_b, gap in zip(
        arms["native"],
        arms["evq_seed_a"],
        arms["evq_seed_b"],
        gaps,
    ):
        per_example.append(
            {
                "local_index": int(native["local_index"]),
                "row_sha256": native["row_sha256"],
                "reference": native["references"][0],
                "generation_boundary_gap_tokens": gap,
                "native_prediction": native["prediction"],
                "native_strict_exact": int(strict_correct(native)),
                "evq_seed_a_prediction": seed_a["prediction"],
                "evq_seed_a_strict_exact": int(strict_correct(seed_a)),
                "evq_seed_b_prediction": seed_b["prediction"],
                "evq_seed_b_strict_exact": int(strict_correct(seed_b)),
            }
        )

    return {
        "status": "OLMO2_FRESH_ALL_LONG_GAP_N100_ANALYSIS_COMPLETE",
        "claim_boundary": (
            "Fresh official RULER niah_single_1 8K n=100 evaluation. "
            "Every source-to-generation gap exceeds all frozen routing-"
            "training gaps. This is same-task autoregressive retrieval, "
            "not full RULER or unseen-task transfer."
        ),
        "metric_boundary": (
            "Greedy autoregressive strict first-number exact is primary. "
            "This screen does not compute teacher-forced NLL or rank."
        ),
        "data": {
            "manifest_sha256": sha256_file(manifest_path),
            "data_sha256": data_sha,
            "rows": 100,
            "minimum_gap_tokens": min(gaps),
            "maximum_gap_tokens": max(gaps),
            "maximum_training_gap_tokens": maximum_training_gap,
            "all_rows_beyond_training_gap": True,
            "selected_overlap_with_forbidden": overlap,
            "ruler_commit": manifest["ruler_commit"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
        },
        "arms": {
            name: {
                "summary": summarize_rows(arms[name]),
                "adapter_sha256": results[name]["adapter"]["sha256"],
                "adapter_seed": results[name]["adapter"]["metadata"]["seed"],
                "frequency_sha256_float32": results[name]["frequency"][
                    "active_sha256_float32"
                ],
                "results_sha256": sha256_file(result_paths[name]),
                "examples_sha256": sha256_file(example_paths[name]),
                "fixed_gap_bands": band_summaries[name],
            }
            for name in ARM_NAMES
        },
        "paired_native_vs_evq_seed_a": paired_summary(
            arms["native"], arms["evq_seed_a"]
        ),
        "paired_native_vs_evq_seed_b": paired_summary(
            arms["native"], arms["evq_seed_b"]
        ),
        "paired_evq_seeds": paired_summary(
            arms["evq_seed_a"], arms["evq_seed_b"]
        ),
        "per_example": per_example,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--native-examples", type=Path, required=True)
    parser.add_argument("--evq-seed-a-results", type=Path, required=True)
    parser.add_argument("--evq-seed-a-examples", type=Path, required=True)
    parser.add_argument("--evq-seed-b-results", type=Path, required=True)
    parser.add_argument("--evq-seed-b-examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_and_analyze(
        manifest_path=args.manifest.resolve(),
        data_path=args.data.resolve(),
        result_paths={
            "native": args.native_results.resolve(),
            "evq_seed_a": args.evq_seed_a_results.resolve(),
            "evq_seed_b": args.evq_seed_b_results.resolve(),
        },
        example_paths={
            "native": args.native_examples.resolve(),
            "evq_seed_a": args.evq_seed_a_examples.resolve(),
            "evq_seed_b": args.evq_seed_b_examples.resolve(),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "strict_exact": {
                    name: result["arms"][name]["summary"]["strict_exact"]
                    for name in ARM_NAMES
                },
                "evq_seed_agreement": result["paired_evq_seeds"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
