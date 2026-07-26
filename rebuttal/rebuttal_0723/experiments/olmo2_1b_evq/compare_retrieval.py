#!/usr/bin/env python3
"""Paired OLMo Geo/EVQ teacher-forced retrieval comparison and gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    sha256_file,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.compare_eval import (
    bootstrap_summary,
    write_json,
)


def identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["length"],
        row["source_fraction"],
        row["distractor_count"],
        row["gold_token_id"],
        row["key"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo", type=Path, required=True)
    parser.add_argument("--evq", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    geo = json.loads(args.geo.read_text(encoding="utf-8"))
    evq = json.loads(args.evq.read_text(encoding="utf-8"))
    if geo.get("status") != "RETRIEVAL_EVALUATION_COMPLETE":
        raise RuntimeError("Geo retrieval evaluation is incomplete")
    if evq.get("status") != "RETRIEVAL_EVALUATION_COMPLETE":
        raise RuntimeError("EVQ retrieval evaluation is incomplete")
    if geo["schedule"] != "geo" or evq["schedule"] != "evq":
        raise RuntimeError("retrieval schedules are reversed")
    if (
        geo["retrieval_manifest_sha256"]
        != evq["retrieval_manifest_sha256"]
    ):
        raise RuntimeError("retrieval manifests differ")
    geo_rows = {identity(row): row for row in geo["rows"]}
    evq_rows = {identity(row): row for row in evq["rows"]}
    if geo_rows.keys() != evq_rows.keys():
        raise RuntimeError("Geo/EVQ retrieval examples differ")

    comparisons: dict[str, Any] = {}
    for length in (4_096, 8_192, 16_384):
        keys = [key for key in geo_rows if key[0] == length]
        comparisons[str(length)] = {}
        for metric in (
            "answer_token_nll",
            "answer_token_rank",
            "gold_minus_best_distractor_logit",
            "next_token_exact_match",
        ):
            differences = np.asarray(
                [
                    float(evq_rows[key]["sourced"][metric])
                    - float(geo_rows[key]["sourced"][metric])
                    for key in keys
                ],
                dtype=np.float64,
            )
            comparisons[str(length)][metric] = bootstrap_summary(
                differences, samples=args.bootstrap_samples
            )
        deletion_differences = np.asarray(
            [
                float(evq_rows[key]["source_deletion_nll_gap"])
                - float(geo_rows[key]["source_deletion_nll_gap"])
                for key in keys
            ],
            dtype=np.float64,
        )
        comparisons[str(length)][
            "source_deletion_nll_gap"
        ] = bootstrap_summary(
            deletion_differences, samples=args.bootstrap_samples
        )

    geo_4k = [row for row in geo["rows"] if row["length"] == 4_096]
    exact_4k = float(
        np.mean([row["sourced"]["next_token_exact_match"] for row in geo_4k])
    )
    deletion_gap_4k = np.asarray(
        [row["source_deletion_nll_gap"] for row in geo_4k],
        dtype=np.float64,
    )
    rank_4k = float(
        np.median([row["sourced"]["answer_token_rank"] for row in geo_4k])
    )
    capability_gate = bool(
        (exact_4k >= 0.10 or deletion_gap_4k.mean() >= 0.10)
        and rank_4k < 1_000
        and deletion_gap_4k.std(ddof=1) > 0
    )
    receipt = {
        "status": "RETRIEVAL_COMPARISON_COMPLETE",
        "delta_definition": "EVQ - Geo; signs depend on metric",
        "geo_result_sha256": sha256_file(args.geo),
        "evq_result_sha256": sha256_file(args.evq),
        "comparisons": comparisons,
        "capability_gate": {
            "pass": capability_gate,
            "geo_4k_exact_match": exact_4k,
            "geo_4k_mean_source_deletion_nll_gap": float(
                deletion_gap_4k.mean()
            ),
            "geo_4k_median_answer_rank": rank_4k,
            "next_action": (
                "prepare and run bounded RULER/NIAH subset"
                if capability_gate
                else "stop at probability/rank evidence; do not report RULER"
            ),
        },
        "uncertainty_boundary": (
            "paired prompt bootstrap only; not an independent training-seed CI"
        ),
    }
    write_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
