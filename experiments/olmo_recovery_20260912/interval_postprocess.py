#!/usr/bin/env python3
"""Validate and summarize matched frozen-table RULER curves over 512--16K."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


ARMS = (
    "BetaSym_gamma1p5_g4",
    "BetaSym_gamma3_g4",
    "BM_g4",
    "Native",
)
THREE_TASK_CAPS = (2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384)
THREE_TASKS = ("niah_single_1", "niah_multikey_1", "niah_multivalue")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("empty mean")
    return sum(values) / len(values)


def log_trapezoid(values: dict[int, float], caps: tuple[int, ...]) -> float:
    if set(values) != set(caps):
        raise ValueError("AUC length grid is incomplete")
    area = 0.0
    for left, right in zip(caps, caps[1:]):
        area += 0.5 * (values[left] + values[right]) * (math.log(right) - math.log(left))
    return area / (math.log(caps[-1]) - math.log(caps[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    by_arm = {}
    identity = None
    for arm in ARMS:
        summary = json.loads((args.root / arm / "summary.json").read_text())
        records = rows(args.root / arm / "generations.jsonl")
        if summary.get("status") != "COMPLETE" or len(records) != 416:
            raise ValueError(f"{arm} is incomplete")
        current = {
            row["eval_id"]: (
                row.get("row_id"), row.get("prompt_sha256"), row["task"],
                int(row["length_cap"]), row["references"],
            )
            for row in records
        }
        if len(current) != len(records):
            raise ValueError(f"{arm} contains duplicate rows")
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError(f"{arm} input identities differ")

        cells = defaultdict(list)
        for row in records:
            cells[(int(row["length_cap"]), row["task"])].append(row)
        expected_caps = (512, 1024) + THREE_TASK_CAPS
        by_length = {}
        for cap in expected_caps:
            tasks = ("niah_single_1",) if cap < 2048 else THREE_TASKS
            task_values = {}
            for task in tasks:
                items = cells[(cap, task)]
                if len(items) != 16:
                    raise ValueError(f"{arm}/{cap}/{task} does not contain 16 rows")
                task_values[task] = {
                    "official": mean(row["ruler_official_score"] for row in items),
                    "exact_plus_eos": mean(row["exact_plus_eos"] for row in items),
                    "eos_rate": mean(row["ended_eos"] for row in items),
                    "cap_exhaustion_rate": mean(row["hit_cap"] for row in items),
                }
            by_length[str(cap)] = {
                "rows": 16 * len(tasks),
                "task_macro_official": mean(value["official"] for value in task_values.values()),
                "tasks": task_values,
            }
        macro = {cap: by_length[str(cap)]["task_macro_official"] for cap in THREE_TASK_CAPS}
        single = {
            cap: by_length[str(cap)]["tasks"]["niah_single_1"]["official"]
            for cap in (512, 1024) + THREE_TASK_CAPS
        }
        by_arm[arm] = {
            "rows": len(records),
            "by_length": by_length,
            "aggregates": {
                "three_task_mean_2k_16k": mean(macro.values()),
                "three_task_min_2k_16k": min(macro.values()),
                "three_task_log_length_auc_2k_16k": log_trapezoid(macro, THREE_TASK_CAPS),
                "single_mean_512_16k": mean(single.values()),
                "single_min_512_16k": min(single.values()),
                "single_log_length_auc_512_16k": log_trapezoid(single, (512, 1024) + THREE_TASK_CAPS),
            },
        }

    contrasts = {}
    for arm in ARMS[:2]:
        for baseline in ("BM_g4", "Native"):
            contrasts[f"{arm}_minus_{baseline}"] = {
                "task_macro_official_by_length": {
                    str(cap): (
                        by_arm[arm]["by_length"][str(cap)]["task_macro_official"]
                        - by_arm[baseline]["by_length"][str(cap)]["task_macro_official"]
                    )
                    for cap in (512, 1024) + THREE_TASK_CAPS
                },
                "aggregate_deltas": {
                    key: by_arm[arm]["aggregates"][key] - by_arm[baseline]["aggregates"][key]
                    for key in by_arm[arm]["aggregates"]
                },
            }
    result = {
        "status": "COMPLETE",
        "scope": "matched frozen-table development curves; metrics are descriptive, not automatic gates",
        "rows_per_arm": 416,
        "three_task_caps": list(THREE_TASK_CAPS),
        "short_single_only_caps": [512, 1024],
        "arms": by_arm,
        "contrasts": contrasts,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
