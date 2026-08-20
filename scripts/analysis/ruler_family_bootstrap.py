#!/usr/bin/env python3
"""Paired family-stratified bootstrap for tracked RULER predictions."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "rebuttal/rebuttal_0723/theory_results/olmo2_qk_phase_adaptation_20260729"


def load(path):
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = row["task"], row["nominal_length"], row["row_sha256"]
        if key in rows:
            raise ValueError(f"duplicate row: {key}")
        rows[key] = float(row["official_task_score"])
    return rows


def quantile(values, probability):
    values = sorted(values)
    position = probability * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, default=OWNER / "ruler13_native_examples.jsonl")
    parser.add_argument("--evq", type=Path, default=OWNER / "ruler13_evq_examples.jsonl")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_820)
    args = parser.parse_args()

    native, evq = load(args.native), load(args.evq)
    if native.keys() != evq.keys():
        raise ValueError("arms do not contain the same task/length/row identities")

    cells = defaultdict(list)
    for task, length, row_sha256 in sorted(native):
        cells[length, task].append(evq[task, length, row_sha256] - native[task, length, row_sha256])
    cell_sizes = {len(values) for values in cells.values()}
    if cell_sizes != {20}:
        raise ValueError(f"expected 20 rows in every family/length cell, got {sorted(cell_sizes)}")

    results = {}
    for length in sorted({length for length, _ in cells}):
        rng = random.Random(args.seed + length)
        families = [cells[length, task] for cell_length, task in sorted(cells) if cell_length == length]
        point = sum(sum(values) / len(values) for values in families) / len(families)
        draws = [
            sum(
                sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                for values in families
            )
            / len(families)
            for _ in range(args.samples)
        ]
        results[str(length)] = {
            "evq_minus_native_percentage_points": 100 * point,
            "paired_family_stratified_ci95_percentage_points": [
                100 * quantile(draws, 0.025),
                100 * quantile(draws, 0.975),
            ],
        }

    print(json.dumps({
        "schema": "evq_cosh.ruler_family_bootstrap.v1",
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "families_per_length": len(cells) // len(results),
        "examples_per_family_length": 20,
        "native_sha256": sha256(args.native),
        "evq_sha256": sha256(args.evq),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
