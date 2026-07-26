#!/usr/bin/env python3
"""Paired summary for the four-arm OLMo-2 LoRA generalization matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)


ARM_NAMES = ("geo_base", "evq_base", "geo_lora", "evq_lora")
COMPARISONS = (
    ("geo_lora", "geo_base"),
    ("evq_lora", "evq_base"),
    ("evq_lora", "geo_lora"),
    ("evq_base", "geo_base"),
)


def load_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "OLMO2_MATCHED_LORA_GENERALIZATION_COMPLETE":
        raise RuntimeError(f"incomplete result: {path}")
    return value


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["length"]),
        int(row["row"]),
        str(row["key"]),
        str(row["value_word"]),
        str(row["document_source"]),
        int(row["document_row"]),
        int(row["document_crop_offset"]),
        str(row["template_id"]),
        float(row["source_fraction"]),
        int(row["distractor_count"]),
        int(row["gold_token_id"]),
        int(row["alternate_token_id"]),
    )


def bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int,
    samples: int = 20_000,
) -> list[float]:
    if values.ndim != 1 or len(values) < 2:
        raise RuntimeError("bootstrap values must be a nontrivial vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        local = values[
            rng.integers(0, len(values), size=len(values))
        ]
        estimates[index] = statistic(local)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    probability = sum(
        math.comb(trials, value)
        for value in range(0, min(successes, trials - successes) + 1)
    ) / (2**trials)
    return min(1.0, 2.0 * probability)


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = np.asarray([row["exact"] for row in rows], dtype=np.float64)
    nll = np.asarray(
        [row["answer_nll"] for row in rows], dtype=np.float64
    )
    rank = np.asarray(
        [row["answer_rank"] for row in rows], dtype=np.float64
    )
    delta_rank = np.asarray(
        [row["source_delta_rank"] for row in rows],
        dtype=np.float64,
    )
    deletion = np.asarray(
        [row["source_deletion_nll_gap"] for row in rows],
        dtype=np.float64,
    )
    swap = np.asarray(
        [row["swap_follow_score"] for row in rows],
        dtype=np.float64,
    )
    return {
        "n": len(rows),
        "exact_match": float(exact.mean()),
        "mean_answer_nll": float(nll.mean()),
        "median_answer_rank": float(np.median(rank)),
        "median_source_delta_rank": float(np.median(delta_rank)),
        "mean_source_deletion_nll_gap": float(deletion.mean()),
        "swap_follow_positive_fraction": float((swap > 0).mean()),
    }


def compare(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    candidate_map = {row_key(row): row for row in candidate}
    reference_map = {row_key(row): row for row in reference}
    if len(candidate_map) != len(candidate) or len(reference_map) != len(
        reference
    ):
        raise RuntimeError("duplicate paired row key")
    if set(candidate_map) != set(reference_map):
        raise RuntimeError("paired result rows do not match")
    keys = sorted(candidate_map)
    candidate_exact = np.asarray(
        [candidate_map[key]["exact"] for key in keys],
        dtype=np.float64,
    )
    reference_exact = np.asarray(
        [reference_map[key]["exact"] for key in keys],
        dtype=np.float64,
    )
    exact_delta = candidate_exact - reference_exact
    candidate_nll = np.asarray(
        [candidate_map[key]["answer_nll"] for key in keys],
        dtype=np.float64,
    )
    reference_nll = np.asarray(
        [reference_map[key]["answer_nll"] for key in keys],
        dtype=np.float64,
    )
    nll_delta = candidate_nll - reference_nll
    candidate_rank = np.asarray(
        [candidate_map[key]["answer_rank"] for key in keys],
        dtype=np.float64,
    )
    reference_rank = np.asarray(
        [reference_map[key]["answer_rank"] for key in keys],
        dtype=np.float64,
    )
    log_rank_ratio = np.log(candidate_rank) - np.log(reference_rank)
    wins = int(
        np.sum((candidate_exact == 1) & (reference_exact == 0))
    )
    losses = int(
        np.sum((candidate_exact == 0) & (reference_exact == 1))
    )
    return {
        "n": len(keys),
        "exact_match_delta": float(exact_delta.mean()),
        "exact_match_delta_bootstrap_95": bootstrap_interval(
            exact_delta,
            np.mean,
            seed=seed,
        ),
        "exact_discordant_candidate_wins": wins,
        "exact_discordant_candidate_losses": losses,
        "mcnemar_exact_two_sided_p": exact_binomial_two_sided(
            min(wins, losses), wins + losses
        ),
        "mean_answer_nll_delta": float(nll_delta.mean()),
        "mean_answer_nll_delta_bootstrap_95": bootstrap_interval(
            nll_delta,
            np.mean,
            seed=seed + 1,
        ),
        "median_log_answer_rank_ratio": float(
            np.median(log_rank_ratio)
        ),
        "median_log_answer_rank_ratio_bootstrap_95": bootstrap_interval(
            log_rank_ratio,
            np.median,
            seed=seed + 2,
        ),
        "answer_rank_improved_fraction": float(
            (candidate_rank < reference_rank).mean()
        ),
        "answer_rank_worsened_fraction": float(
            (candidate_rank > reference_rank).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ARM_NAMES:
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=Path,
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        name: getattr(args, name).resolve() for name in ARM_NAMES
    }
    results = {
        name: load_result(path) for name, path in paths.items()
    }
    expected = results["geo_base"]["protocol"][
        "eval_dataset_sha256"
    ]
    for name, result in results.items():
        if result["protocol"]["eval_dataset_sha256"] != expected:
            raise RuntimeError(f"eval dataset drift for {name}")
        checks = result["split_contract"]["checks"]
        if not checks or not all(checks.values()):
            raise RuntimeError(f"split contract failed for {name}")
        if result["protocol"]["eval_examples_per_length"] < 100:
            raise RuntimeError(f"formal sample count too small for {name}")
    expected_identity = {
        "geo_base": ("geo", "baseline"),
        "evq_base": ("evq", "baseline"),
        "geo_lora": ("geo", "qkvo_answer"),
        "evq_lora": ("evq", "qkvo_answer"),
    }
    for name, identity in expected_identity.items():
        actual = (
            results[name]["schedule"],
            results[name]["adaptation"],
        )
        if actual != identity:
            raise RuntimeError(
                f"arm identity drift for {name}: {actual} != {identity}"
            )

    lengths = sorted(
        {int(row["length"]) for row in results["geo_base"]["rows"]}
    )
    report: dict[str, Any] = {
        "status": "OLMO2_MATCHED_LORA_PAIRED_SUMMARY_COMPLETE",
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "eval_dataset_sha256": expected,
        "lengths": lengths,
        "arms": {},
        "comparisons": {},
    }
    for length in lengths:
        key = str(length)
        report["arms"][key] = {}
        rows_by_arm = {
            name: [
                row
                for row in result["rows"]
                if int(row["length"]) == length
            ]
            for name, result in results.items()
        }
        for name, rows in rows_by_arm.items():
            report["arms"][key][name] = summarize_arm(rows)
        report["comparisons"][key] = {}
        for index, (candidate, reference) in enumerate(COMPARISONS):
            comparison_key = f"{candidate}_minus_{reference}"
            report["comparisons"][key][comparison_key] = compare(
                rows_by_arm[candidate],
                rows_by_arm[reference],
                seed=20_260_725 + length + 100 * index,
            )
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
