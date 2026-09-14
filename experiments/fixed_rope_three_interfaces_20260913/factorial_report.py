#!/usr/bin/env python3
"""Analyze the paired 2x2 table-by-gain intervention from completed rows."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .pipeline import atomic_json


FORMAT = "PAIRED_TABLE_GAIN_FACTORIAL_REPORT_V1"
ARMS = ("Y00", "Y01", "Y10", "Y11")
EFFECTS = ("F_g0", "F_g1", "G_T0", "G_T1", "interaction", "delta_F", "delta_G", "combined")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def row_score(row: dict) -> float:
    for name in ("ruler_official_score", "official_score"):
        if name in row:
            value = float(row[name])
            if np.isfinite(value):
                return value
    raise ValueError("generation row lacks a finite official score")


def normalize_rows(paths: list[Path]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in paths:
        for row in read_jsonl(path):
            prompt = str(row.get("prompt_sha256", ""))
            if not prompt:
                raise ValueError(f"row in {path} lacks prompt_sha256")
            normalized = {
                **row,
                "prompt_sha256": prompt,
                "task": str(row["task"]),
                "length_cap": int(row["length_cap"]),
                "official_score": row_score(row),
            }
            previous = result.get(prompt)
            if previous is not None:
                comparable = ("task", "length_cap", "official_score")
                if any(previous[name] != normalized[name] for name in comparable):
                    raise ValueError(f"conflicting duplicate prompt {prompt}")
            else:
                result[prompt] = normalized
    return result


def filter_rows(
    rows: dict[str, dict], *, tasks: set[str] | None, lengths: set[int] | None,
) -> dict[str, dict]:
    """Select the declared mechanism panel from broader reusable run files."""
    return {
        prompt: row for prompt, row in rows.items()
        if (tasks is None or row["task"] in tasks)
        and (lengths is None or row["length_cap"] in lengths)
    }


def row_effects(y00: float, y01: float, y10: float, y11: float) -> dict[str, float]:
    values = {
        "F_g0": y10 - y00,
        "F_g1": y11 - y01,
        "G_T0": y01 - y00,
        "G_T1": y11 - y10,
        "interaction": y11 - y10 - y01 + y00,
        "delta_F": 0.5 * ((y10 - y00) + (y11 - y01)),
        "delta_G": 0.5 * ((y01 - y00) + (y11 - y10)),
        "combined": y11 - y00,
    }
    if abs(values["delta_F"] + values["delta_G"] - values["combined"]) > 1e-12:
        raise AssertionError("symmetric factorial decomposition identity failed")
    return values


def _mean_effects(records: list[dict]) -> dict[str, float]:
    if not records:
        raise ValueError("factorial summary cell is empty")
    return {name: float(np.mean([record[name] for record in records])) for name in EFFECTS}


def build_report(
    arms: dict[str, dict[str, dict]],
    *,
    tasks: list[str],
    lengths: list[int],
    rows_per_cell: int,
    draws: int,
    seed: int,
) -> dict:
    if set(arms) != set(ARMS):
        raise ValueError(f"factorial input must contain exactly {ARMS}")
    prompts = {name: set(rows) for name, rows in arms.items()}
    reference = prompts["Y00"]
    if any(value != reference for value in prompts.values()):
        raise ValueError("factorial arms are not paired on exactly the same prompts")
    joined = []
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for prompt in sorted(reference):
        source = arms["Y00"][prompt]
        task, length = source["task"], source["length_cap"]
        if any(
            arms[name][prompt]["task"] != task or arms[name][prompt]["length_cap"] != length
            for name in ARMS[1:]
        ):
            raise ValueError(f"prompt metadata differs across arms: {prompt}")
        if task not in tasks or length not in lengths:
            raise ValueError(f"unexpected factorial cell {task}/{length}")
        scores = {name: arms[name][prompt]["official_score"] for name in ARMS}
        record = {
            "prompt_sha256": prompt, "task": task, "length_cap": length,
            "scores": scores, **row_effects(*(scores[name] for name in ARMS)),
        }
        joined.append(record)
        cells[(task, length)].append(record)
    missing = [
        (task, length, len(cells[(task, length)]))
        for task in tasks for length in lengths
        if len(cells[(task, length)]) != rows_per_cell
    ]
    if missing:
        raise ValueError(f"factorial cell coverage differs from the frozen quota: {missing}")

    by_task_length = {
        f"{task}|{length}": {
            "rows": len(cells[(task, length)]),
            "arm_means": {
                name: float(np.mean([record["scores"][name] for record in cells[(task, length)]]))
                for name in ARMS
            },
            "effects": _mean_effects(cells[(task, length)]),
        }
        for task in tasks for length in lengths
    }
    by_task = {
        task: _mean_effects([record for record in joined if record["task"] == task])
        for task in tasks
    }
    by_length = {
        str(length): {
            name: float(np.mean([by_task_length[f"{task}|{length}"]["effects"][name] for task in tasks]))
            for name in EFFECTS
        }
        for length in lengths
    }
    task_equal = {
        name: float(np.mean([by_task[task][name] for task in tasks])) for name in EFFECTS
    }

    rng = np.random.default_rng(seed)
    bootstrap = {name: np.empty(draws, dtype=np.float64) for name in EFFECTS}
    for draw in range(draws):
        task_values = {name: [] for name in EFFECTS}
        for task in tasks:
            length_values = {name: [] for name in EFFECTS}
            for length in lengths:
                cell = cells[(task, length)]
                selected = rng.integers(0, len(cell), size=len(cell))
                for name in EFFECTS:
                    length_values[name].append(float(np.mean([cell[int(index)][name] for index in selected])))
            for name in EFFECTS:
                task_values[name].append(float(np.mean(length_values[name])))
        for name in EFFECTS:
            bootstrap[name][draw] = float(np.mean(task_values[name]))

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "status": FORMAT,
        "arms": {
            "Y00": "T0,g0", "Y01": "T0,g1", "Y10": "T1,g0", "Y11": "T1,g1",
        },
        "tasks": tasks,
        "lengths": lengths,
        "rows_per_task_length": rows_per_cell,
        "paired_prompts": len(reference),
        "by_task_length": by_task_length,
        "by_task": by_task,
        "by_length": by_length,
        "task_equal_over_declared_cells": task_equal,
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "resampling": "paired rows within each frozen task-length cell; tasks and lengths fixed",
            "effects": {
                name: {
                    "mean": float(values.mean()),
                    "interval95": interval(values),
                    "positive_resample_fraction": float(np.mean(values > 0.0)),
                }
                for name, values in bootstrap.items()
            },
        },
        "interpretation_contract": {
            "mechanism_panel_not_core6_auc": True,
            "interaction_retained": True,
            "crossing_zero_is_inconclusive_not_no_effect": True,
        },
    }


def parse_sources(values: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        if "=" not in value:
            raise ValueError("--source must be ARM=PATH")
        arm, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if arm not in ARMS or not path.is_file():
            raise ValueError(f"invalid factorial source {value}")
        result[arm].append(path)
    if set(result) != set(ARMS):
        raise ValueError(f"--source must cover exactly {ARMS}")
    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], required=True, help="ARM=PATH")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--length", type=int, action="append", default=[])
    parser.add_argument("--rows-per-cell", type=int, default=12)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.rows_per_cell < 1 or args.bootstrap_draws < 1:
        raise ValueError("invalid output or analysis budget")
    paths = parse_sources(args.source)
    task_filter = set(args.task) if args.task else None
    length_filter = set(args.length) if args.length else None
    arms = {
        name: filter_rows(normalize_rows(value), tasks=task_filter, lengths=length_filter)
        for name, value in paths.items()
    }
    tasks = sorted(task_filter or {row["task"] for row in arms["Y00"].values()})
    lengths = sorted(length_filter or {row["length_cap"] for row in arms["Y00"].values()})
    report = build_report(
        arms, tasks=tasks, lengths=lengths, rows_per_cell=args.rows_per_cell,
        draws=args.bootstrap_draws, seed=args.bootstrap_seed,
    )
    report["source_files"] = {name: [str(path) for path in value] for name, value in paths.items()}
    atomic_json(args.out, report)
    print(json.dumps({
        "status": report["status"], "paired_prompts": report["paired_prompts"],
        "effects": report["task_equal_over_declared_cells"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
