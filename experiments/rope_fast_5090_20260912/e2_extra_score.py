#!/usr/bin/env python3
"""Score one added E2 arm against the completed Native/BM/MrPro baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.rope_fast_5090_20260912.e2_run import validate_saved_rows
from experiments.rope_fast_5090_20260912.e2_score import (
    TASKS,
    bootstrap_test,
    macro,
    read_rows,
    stratified_cluster_draws,
)


BASELINES = ("native_g1", "bm_g4", "mrpro_g4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--candidate", default="C42V24")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.prepared / "inputs.jsonl")
    inputs = {row["row_id"]: row for row in rows}
    if len(rows) != 778 or len(inputs) != 778:
        raise ValueError("E2 input panel is not 778 unique rows")
    generation = json.loads((args.prepared / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos_ids = {eos} if isinstance(eos, int) else set(eos)

    arms = {}
    locations = {name: args.baseline_run / f"{name}.jsonl" for name in BASELINES}
    locations[args.candidate] = args.candidate_run / f"{args.candidate}.jsonl"
    for name, path in locations.items():
        values = read_rows(path)
        validate_saved_rows(values, rows, name, eos_ids)
        if len(values) != len(rows):
            raise ValueError(f"incomplete arm: {name}")
        arms[name] = {row["row_id"]: row for row in values}

    long_ids = [row_id for row_id, row in inputs.items() if row["input_tokens"] > 4096]
    short_ids = [row_id for row_id, row in inputs.items() if row["input_tokens"] <= 4096]
    if len(long_ids) != 631 or len(short_ids) != 147:
        raise ValueError("E2 long/short split differs")

    summaries = {}
    short = {}
    for name, values in arms.items():
        overall, by_task = macro(values, long_ids)
        summaries[name] = {"rows": len(long_ids), "five_task_equal_macro_f1": overall, "by_task": by_task}
        short[name] = {}
        for task in TASKS:
            ids = [row_id for row_id in short_ids if inputs[row_id]["task"] == task]
            if ids:
                short[name][task] = {
                    "rows": len(ids),
                    "mean_f1": float(np.mean([values[row_id]["whole_response_f1"] for row_id in ids])),
                }

    contrasts = {}
    for index, baseline in enumerate(BASELINES):
        coefficients = {args.candidate: 1.0, baseline: -1.0}
        draws = stratified_cluster_draws(
            inputs, arms, coefficients, long_ids, draws=20_000, seed=2_026_091_300 + index
        )
        contrasts[f"{args.candidate}_minus_{baseline}"] = {
            "estimate": (
                summaries[args.candidate]["five_task_equal_macro_f1"]
                - summaries[baseline]["five_task_equal_macro_f1"]
            ),
            **bootstrap_test(draws),
        }

    result = {
        "status": "COMPLETE",
        "scope": "post-E3 practical development comparison on the fixed prior E2 pool; not an independent C42V24 confirmation",
        "metric": "five-task equal macro whole-response token F1",
        "long_pool_rows": len(long_ids),
        "arms": summaries,
        "contrasts": contrasts,
        "short_window_by_task": short,
        "inference": "20,000 paired document-cluster bootstrap draws within task, then equal-task macro",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arms": summaries, "contrasts": contrasts}, sort_keys=True))


if __name__ == "__main__":
    main()
