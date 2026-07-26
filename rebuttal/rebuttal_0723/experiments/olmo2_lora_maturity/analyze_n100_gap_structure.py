#!/usr/bin/env python3
"""Audit the distance structure of the frozen 8K n=100 RULER results.

This is a post-hoc, CPU-only diagnostic.  Its primary split is determined only
from the maximum source-to-answer gap observed in the frozen 4K routing
training rows; no evaluation outcome is used to choose that threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"\d+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
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


def hypergeom_probability(
    top_left: int,
    row_one: int,
    row_two: int,
    column_one: int,
) -> float:
    total = row_one + row_two
    return (
        math.comb(row_one, top_left)
        * math.comb(row_two, column_one - top_left)
        / math.comb(total, column_one)
    )


def fisher_exact_two_sided(
    a: int,
    b: int,
    c: int,
    d: int,
) -> dict[str, float | None]:
    row_one = a + b
    row_two = c + d
    column_one = a + c
    lower = max(0, column_one - row_two)
    upper = min(row_one, column_one)
    observed = hypergeom_probability(a, row_one, row_two, column_one)
    p_value = 0.0
    for candidate in range(lower, upper + 1):
        probability = hypergeom_probability(
            candidate,
            row_one,
            row_two,
            column_one,
        )
        if probability <= observed + 1e-15:
            p_value += probability
    odds_ratio = None
    if b * c != 0:
        odds_ratio = (a * d) / (b * c)
    return {
        "odds_ratio": odds_ratio,
        "two_sided_p": min(1.0, p_value),
    }


def exact_mcnemar(seed_a_only: int, seed_b_only: int) -> float:
    discordant = seed_a_only + seed_b_only
    if discordant == 0:
        return 1.0
    smaller = min(seed_a_only, seed_b_only)
    tail = sum(
        math.comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def classify_prediction(row: dict[str, Any]) -> str:
    gold = str(row["references"][0])
    match = NUMBER_RE.search(str(row["prediction"]))
    if match is None:
        return "no_number"
    first_number = match.group(0)
    if first_number == gold:
        return "strict_exact"
    if float(row["official_string_match"]) == 1.0:
        return "official_only_wrong_first_number"
    return "wrong_first_number"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(int(float(row["first_number_exact"]) == 1.0) for row in rows)
    official = sum(
        int(float(row["official_string_match"]) == 1.0) for row in rows
    )
    error_types: dict[str, int] = {}
    for row in rows:
        label = classify_prediction(row)
        error_types[label] = error_types.get(label, 0) + 1
    return {
        "examples": total,
        "strict_exact": exact,
        "strict_exact_rate": exact / total if total else None,
        "strict_exact_wilson_95": wilson_interval(exact, total),
        "official_string_match": official,
        "official_string_match_rate": (
            official / total if total else None
        ),
        "prediction_classes": dict(sorted(error_types.items())),
    }


def validate_alignment(
    arms: dict[str, list[dict[str, Any]]],
) -> None:
    lengths = {name: len(rows) for name, rows in arms.items()}
    if set(lengths.values()) != {100}:
        raise ValueError(f"expected exactly 100 rows per arm, got {lengths}")
    reference_name = next(iter(arms))
    reference_rows = arms[reference_name]
    identity_fields = (
        "local_index",
        "row_sha256",
        "references",
        "source_row_index",
        "source_token_position_answer",
        "input_tokens",
    )
    for name, rows in arms.items():
        for index, (reference, candidate) in enumerate(
            zip(reference_rows, rows)
        ):
            for field in identity_fields:
                if candidate[field] != reference[field]:
                    raise ValueError(
                        f"row mismatch arm={name} index={index} field={field}"
                    )


def add_gap(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["_generation_boundary_gap_tokens"] = (
            int(row["input_tokens"])
            - int(row["source_token_position_answer"])
        )


def gap_band(gap: int) -> str:
    lower = (gap // 1024) * 1024
    upper = lower + 1023
    if lower >= 7168:
        return "7168+"
    return f"{lower}-{upper}"


def group_summary(
    rows: list[dict[str, Any]],
    max_training_gap: int,
) -> dict[str, Any]:
    within = [
        row
        for row in rows
        if row["_generation_boundary_gap_tokens"] <= max_training_gap
    ]
    beyond = [
        row
        for row in rows
        if row["_generation_boundary_gap_tokens"] > max_training_gap
    ]
    within_summary = summarize_rows(within)
    beyond_summary = summarize_rows(beyond)
    a = int(within_summary["strict_exact"])
    b = int(within_summary["examples"]) - a
    c = int(beyond_summary["strict_exact"])
    d = int(beyond_summary["examples"]) - c
    bands: dict[str, dict[str, Any]] = {}
    for label in (
        "0-1023",
        "1024-2047",
        "2048-3071",
        "3072-4095",
        "4096-5119",
        "5120-6143",
        "6144-7167",
        "7168+",
    ):
        selected = [
            row
            for row in rows
            if gap_band(row["_generation_boundary_gap_tokens"]) == label
        ]
        bands[label] = summarize_rows(selected) if selected else {
            "examples": 0,
            "strict_exact": 0,
            "strict_exact_rate": None,
        }
    return {
        "within_observed_training_gap": within_summary,
        "beyond_observed_training_gap": beyond_summary,
        "risk_difference_within_minus_beyond": (
            float(within_summary["strict_exact_rate"])
            - float(beyond_summary["strict_exact_rate"])
        ),
        "fisher_exact_within_vs_beyond": fisher_exact_two_sided(
            a,
            b,
            c,
            d,
        ),
        "fixed_1024_token_gap_bands": bands,
    }


def paired_seed_summary(
    seed_a_rows: list[dict[str, Any]],
    seed_b_rows: list[dict[str, Any]],
    max_training_gap: int,
    subset: str,
) -> dict[str, Any]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if len(seed_a_rows) != len(seed_b_rows):
        raise ValueError("paired seed rows have different lengths")
    for seed_a, seed_b in zip(seed_a_rows, seed_b_rows):
        gap = int(seed_a["_generation_boundary_gap_tokens"])
        if subset == "all":
            selected.append((seed_a, seed_b))
        elif subset == "within" and gap <= max_training_gap:
            selected.append((seed_a, seed_b))
        elif subset == "beyond" and gap > max_training_gap:
            selected.append((seed_a, seed_b))
    both_correct = 0
    seed_a_only = 0
    seed_b_only = 0
    both_wrong = 0
    for seed_a, seed_b in selected:
        a_correct = float(seed_a["first_number_exact"]) == 1.0
        b_correct = float(seed_b["first_number_exact"]) == 1.0
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct:
            seed_a_only += 1
        elif b_correct:
            seed_b_only += 1
        else:
            both_wrong += 1
    return {
        "examples": len(selected),
        "both_correct": both_correct,
        "seed_a_only": seed_a_only,
        "seed_b_only": seed_b_only,
        "both_wrong": both_wrong,
        "agreement_rate": (
            (both_correct + both_wrong) / len(selected)
            if selected
            else None
        ),
        "exact_mcnemar_two_sided_p": exact_mcnemar(
            seed_a_only,
            seed_b_only,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--native-examples", type=Path, required=True)
    parser.add_argument("--evq-seed-a-examples", type=Path, required=True)
    parser.add_argument("--evq-seed-b-examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "training_rows": args.training_rows.resolve(),
        "native": args.native_examples.resolve(),
        "evq_seed_a": args.evq_seed_a_examples.resolve(),
        "evq_seed_b": args.evq_seed_b_examples.resolve(),
    }
    training_rows = load_jsonl(paths["training_rows"])
    training_gaps = [
        int(row["answer_start"])
        - int(row["source_token_position_answer"])
        for row in training_rows
    ]
    if not training_gaps:
        raise ValueError("routing training rows are empty")
    max_training_gap = max(training_gaps)
    arms = {
        "native": load_jsonl(paths["native"]),
        "evq_seed_a": load_jsonl(paths["evq_seed_a"]),
        "evq_seed_b": load_jsonl(paths["evq_seed_b"]),
    }
    validate_alignment(arms)
    for rows in arms.values():
        add_gap(rows)

    per_example: list[dict[str, Any]] = []
    if not (
        len(arms["native"])
        == len(arms["evq_seed_a"])
        == len(arms["evq_seed_b"])
    ):
        raise ValueError("arm lengths differ before per-example pairing")
    for native, seed_a, seed_b in zip(
        arms["native"],
        arms["evq_seed_a"],
        arms["evq_seed_b"],
    ):
        gap = int(native["_generation_boundary_gap_tokens"])
        per_example.append(
            {
                "local_index": int(native["local_index"]),
                "row_sha256": native["row_sha256"],
                "reference": native["references"][0],
                "source_token_position_answer": int(
                    native["source_token_position_answer"]
                ),
                "input_tokens": int(native["input_tokens"]),
                "generation_boundary_gap_tokens": gap,
                "gap_group": (
                    "within_observed_training_gap"
                    if gap <= max_training_gap
                    else "beyond_observed_training_gap"
                ),
                "native_strict_exact": int(
                    float(native["first_number_exact"]) == 1.0
                ),
                "evq_seed_a_strict_exact": int(
                    float(seed_a["first_number_exact"]) == 1.0
                ),
                "evq_seed_b_strict_exact": int(
                    float(seed_b["first_number_exact"]) == 1.0
                ),
                "evq_seed_a_prediction_class": classify_prediction(seed_a),
                "evq_seed_b_prediction_class": classify_prediction(seed_b),
            }
        )

    result = {
        "status": "POST_HOC_CPU_ONLY_GAP_STRUCTURE_AUDIT",
        "claim_boundary": (
            "Diagnostic decomposition of one official RULER niah_single_1 "
            "8K n=100 set. It is not a preregistered endpoint, full RULER, "
            "or unseen-task evidence."
        ),
        "primary_split_rule": (
            "Evaluation generation-boundary gap is input_tokens minus the "
            "source answer-token position. The threshold is the maximum "
            "answer_start minus source answer-token position in frozen 4K "
            "routing training rows and is chosen without evaluation labels."
        ),
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "training_gap_support": {
            "rows": len(training_gaps),
            "minimum_tokens": min(training_gaps),
            "maximum_tokens": max_training_gap,
        },
        "row_alignment": {
            "arms": list(arms),
            "rows_per_arm": 100,
            "identity_fields_checked": [
                "local_index",
                "row_sha256",
                "references",
                "source_row_index",
                "source_token_position_answer",
                "input_tokens",
            ],
            "all_equal": True,
        },
        "arms": {
            name: {
                "overall": summarize_rows(rows),
                "by_gap_support": group_summary(
                    rows,
                    max_training_gap,
                ),
            }
            for name, rows in arms.items()
        },
        "evq_seed_pairing": {
            subset: paired_seed_summary(
                arms["evq_seed_a"],
                arms["evq_seed_b"],
                max_training_gap,
                subset,
            )
            for subset in ("all", "within", "beyond")
        },
        "per_example": per_example,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
