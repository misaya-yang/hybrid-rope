#!/usr/bin/env python3
"""Paired Geo-minus-EVQ NLL comparison with document bootstrap CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    SEED,
    sha256_file,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def bootstrap_summary(
    differences: np.ndarray, *, samples: int
) -> dict[str, Any]:
    if differences.ndim != 1 or len(differences) < 2:
        raise ValueError("paired differences require at least two rows")
    rng = np.random.Generator(np.random.PCG64(SEED + 17))
    indices = rng.integers(
        0, len(differences), size=(samples, len(differences))
    )
    means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "n": len(differences),
        "mean_delta_evq_minus_geo": float(differences.mean()),
        "median_delta_evq_minus_geo": float(np.median(differences)),
        "sample_std": float(differences.std(ddof=1)),
        "fraction_evq_better": float((differences < 0).mean()),
        "bootstrap_samples": samples,
        "bootstrap_95_ci": [float(lower), float(upper)],
    }


def rows_by_id(result: dict[str, Any], dataset: str) -> dict[int, dict]:
    return {
        int(row["row"]): row["metrics"]
        for row in result["results"][dataset]["rows"]
    }


def paired_metric(
    geo_rows: dict[int, dict],
    evq_rows: dict[int, dict],
    *,
    length: str,
    metric: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if geo_rows.keys() != evq_rows.keys():
        raise RuntimeError("Geo/EVQ evaluation row IDs differ")
    values = np.asarray(
        [
            float(evq_rows[index][length][metric])
            - float(geo_rows[index][length][metric])
            for index in sorted(geo_rows)
        ],
        dtype=np.float64,
    )
    return bootstrap_summary(values, samples=bootstrap_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo", type=Path, required=True)
    parser.add_argument("--evq", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    geo = json.loads(args.geo.read_text(encoding="utf-8"))
    evq = json.loads(args.evq.read_text(encoding="utf-8"))
    if geo.get("status") != "EVALUATION_COMPLETE":
        raise RuntimeError("Geo evaluation is incomplete")
    if evq.get("status") != "EVALUATION_COMPLETE":
        raise RuntimeError("EVQ evaluation is incomplete")
    if geo["schedule"] != "geo" or evq["schedule"] != "evq":
        raise RuntimeError("evaluation schedule identities are reversed")
    if geo["eval_manifest_sha256"] != evq["eval_manifest_sha256"]:
        raise RuntimeError("Geo/EVQ evaluation manifests differ")

    comparisons: dict[str, Any] = {}
    for dataset in ("official_validation", "long_documents"):
        if (
            geo["results"][dataset]["anchor_sha256"]
            != evq["results"][dataset]["anchor_sha256"]
        ):
            raise RuntimeError(f"{dataset}: anchor hash mismatch")
        geo_rows = rows_by_id(geo, dataset)
        evq_rows = rows_by_id(evq, dataset)
        lengths = sorted(
            set.intersection(
                *(set(row.keys()) for row in geo_rows.values())
            ),
            key=int,
        )
        comparisons[dataset] = {
            length: {
                metric: paired_metric(
                    geo_rows,
                    evq_rows,
                    length=length,
                    metric=metric,
                    bootstrap_samples=args.bootstrap_samples,
                )
                for metric in ("full_nll", "tail_1024_nll")
            }
            for length in lengths
        }

    primary_8k = comparisons["long_documents"]["8192"][
        "tail_1024_nll"
    ]
    primary_16k = comparisons["long_documents"]["16384"][
        "tail_1024_nll"
    ]
    in_range_4k = comparisons["long_documents"]["4096"]["full_nll"]
    strong_positive = (
        primary_8k["mean_delta_evq_minus_geo"] < 0
        and primary_16k["mean_delta_evq_minus_geo"] < 0
        and (
            abs(primary_8k["mean_delta_evq_minus_geo"]) >= 0.05
            or abs(primary_16k["mean_delta_evq_minus_geo"]) >= 0.05
        )
        and (
            primary_8k["bootstrap_95_ci"][1] < 0
            or primary_16k["bootstrap_95_ci"][1] < 0
        )
        and in_range_4k["mean_delta_evq_minus_geo"] <= 0.05
    )
    receipt = {
        "status": "PAIRED_EVALUATION_COMPLETE",
        "delta_definition": "NLL_EVQ - NLL_Geo; negative favors EVQ",
        "geo_result_sha256": sha256_file(args.geo),
        "evq_result_sha256": sha256_file(args.evq),
        "eval_manifest_sha256": geo["eval_manifest_sha256"],
        "comparisons": comparisons,
        "preregistered_nll_gate": {
            "strong_positive_without_capability_clause": strong_positive,
            "primary_8k_tail": primary_8k,
            "primary_16k_tail": primary_16k,
            "in_range_4k_full": in_range_4k,
            "capability_gate_pending": True,
        },
        "uncertainty_boundary": (
            "document bootstrap quantifies held-out evaluation sampling only; "
            "it is not an independent training-seed interval"
        ),
    }
    write_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
