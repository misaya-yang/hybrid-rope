#!/usr/bin/env python3
"""Summarize frozen-table range generation without mixing official and strict metrics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


BASELINES = ("Native", "BM_g4", "MrPro_g4", "C42V24_g4")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("empty mean")
    return sum(values) / len(values)


def log_auc(curve: dict[int, float]) -> float:
    caps = sorted(curve)
    area = sum(
        0.5 * (curve[left] + curve[right]) * (math.log(right) - math.log(left))
        for left, right in zip(caps, caps[1:])
    )
    return area / (math.log(caps[-1]) - math.log(caps[0]))


def normalized(text: str) -> str:
    return text.strip().lower().strip(" .,!;:\"'`\n\t")


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must be LABEL=RUN_DIRECTORY")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("arm must be LABEL=RUN_DIRECTORY")
    return label, Path(path)


def summarize_arm(rows: list[dict], main: dict[str, dict], source_cf: dict[str, dict]) -> dict:
    by_id = {row["row_id"]: row for row in rows}
    expected = set(main) | set(source_cf)
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise ValueError("generation rows are duplicated or differ from the prepared split")
    cells = defaultdict(list)
    for row_id, expected_row in main.items():
        row = by_id[row_id]
        for field in ("task", "length_cap", "references", "prompt_sha256"):
            if row[field] != expected_row[field]:
                raise ValueError(f"main row differs at {field}: {row_id}")
        cells[(row["length_cap"], row["task"])].append(row)
    caps = sorted({cap for cap, _ in cells})
    tasks = sorted({task for _, task in cells})
    by_length, task_cells = {}, {}
    for cap in caps:
        per_task = {}
        for task in tasks:
            items = cells[(cap, task)]
            per_task[task] = {
                "rows": len(items),
                "official": mean(row["ruler_official_score"] for row in items),
                "exact_plus_eos": mean(row["exact_plus_eos"] for row in items),
                "eos_rate": mean(row["ended_eos"] for row in items),
                "cap_rate": mean(row["hit_cap"] for row in items),
            }
            task_cells[f"{cap}/{task}"] = per_task[task]
        by_length[str(cap)] = {
            "task_macro_official": mean(value["official"] for value in per_task.values()),
            "task_macro_exact_plus_eos": mean(value["exact_plus_eos"] for value in per_task.values()),
            "tasks": per_task,
        }
    curve = {cap: by_length[str(cap)]["task_macro_official"] for cap in caps}

    groups = defaultdict(list)
    for row_id, expected_row in source_cf.items():
        row = by_id[row_id]
        if row["group_id"] != expected_row["group_id"] or row["world"] != expected_row["world"]:
            raise ValueError(f"source-counterfactual row differs: {row_id}")
        groups[row["group_id"]].append(row)
    pair_cells = defaultdict(list)
    for group_id, items in groups.items():
        if len(items) != 2 or {row["world"] for row in items} != {0, 1}:
            raise ValueError(f"incomplete counterfactual pair: {group_id}")
        followed = all(row["exact_plus_eos"] for row in items) and (
            normalized(items[0]["output_text"]) != normalized(items[1]["output_text"])
        )
        meta = source_cf[items[0]["row_id"]]
        pair_cells[(meta["length_cap"], meta["family"])].append(float(followed))
    pair_summary = {
        f"{cap}/{family}": {"pairs": len(values), "pair_follow": mean(values)}
        for (cap, family), values in sorted(pair_cells.items())
    }
    return {
        "by_length": by_length,
        "task_cells": task_cells,
        "log_length_auc_official": log_auc(curve),
        "interval_min_task_macro_official": min(curve.values()),
        "source_pair_cells": pair_summary,
        "source_pair_follow": mean(value for values in pair_cells.values() for value in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-cf", type=Path, required=True)
    parser.add_argument("--split", choices=("select", "internal_confirm"), required=True)
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main_rows = {
        row["row_id"]: row for row in read_rows(args.data / "rows.jsonl")
        if row["split"] == args.split
    }
    cf_rows = {
        row["row_id"]: row for row in read_rows(args.source_cf / "rows.jsonl")
        if row["split"] == args.split
    }
    if len(main_rows) != 84 or len(cf_rows) != 64:
        raise ValueError("prepared generation split is incomplete")
    summaries = {}
    for label, path in args.arm:
        if label in summaries:
            raise ValueError(f"duplicate arm label: {label}")
        status = json.loads((path / "status.json").read_text())
        if status.get("status") != "COMPLETE":
            raise ValueError(f"arm is incomplete: {label}")
        summaries[label] = summarize_arm(read_rows(path / "generations.jsonl"), main_rows, cf_rows)

    contrasts = {}
    if set(BASELINES) <= set(summaries):
        extended = BASELINES[1:]
        caps = sorted({row["length_cap"] for row in main_rows.values()})
        tasks = sorted({row["task"] for row in main_rows.values()})
        for label, candidate in summaries.items():
            native_regrets = [
                summaries["Native"]["task_cells"][f"{caps[0]}/{task}"]["official"]
                - candidate["task_cells"][f"{caps[0]}/{task}"]["official"]
                for task in tasks
            ]
            endpoint_regrets = [
                max(summaries[arm]["task_cells"][f"{caps[-1]}/{task}"]["official"] for arm in extended)
                - candidate["task_cells"][f"{caps[-1]}/{task}"]["official"]
                for task in tasks
            ]
            all_regrets = [
                max(summaries[arm]["task_cells"][f"{cap}/{task}"]["official"] for arm in extended)
                - candidate["task_cells"][f"{cap}/{task}"]["official"]
                for cap in caps for task in tasks
            ]
            macro_regrets = [
                max(
                    summaries[arm]["by_length"][str(cap)]["task_macro_official"]
                    for arm in extended
                ) - candidate["by_length"][str(cap)]["task_macro_official"]
                for cap in caps
            ]
            source_regrets = [
                max(summaries[arm]["source_pair_cells"][cell]["pair_follow"] for arm in extended)
                - candidate["source_pair_cells"][cell]["pair_follow"]
                for cell in candidate["source_pair_cells"]
            ]
            contrasts[label] = {
                "max_native_task_regret": max(native_regrets),
                "native_task_macro_regret": (
                    summaries["Native"]["by_length"][str(caps[0])]["task_macro_official"]
                    - candidate["by_length"][str(caps[0])]["task_macro_official"]
                ),
                "max_endpoint_task_regret": max(endpoint_regrets),
                "endpoint_task_macro_regret": (
                    max(
                        summaries[arm]["by_length"][str(caps[-1])]["task_macro_official"]
                        for arm in extended
                    ) - candidate["by_length"][str(caps[-1])]["task_macro_official"]
                ),
                "max_task_length_regret": max(all_regrets),
                "worst_length_task_macro_regret": max(macro_regrets),
                "max_source_pair_cell_regret": max(source_regrets),
                "log_length_auc_official": candidate["log_length_auc_official"],
            }
    result = {
        "status": "COMPLETE",
        "split": args.split,
        "metrics_are_separate": "official contains-style task score, exact-plus-EOS, and source pair-follow are not interchangeable",
        "arms": summaries,
        "contrasts": contrasts,
        "scope": "prepared frozen-table development split; no LoRA and no claim beyond this task distribution",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "split": args.split, "arms": list(summaries)}))


if __name__ == "__main__":
    main()
