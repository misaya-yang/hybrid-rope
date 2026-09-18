#!/usr/bin/env python3
"""Build the paired three-arm Kanana 64K Full-13 report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS

from .report import atomic_json, load_arm, sha256, summarize


def pair_outcomes(left: dict[str, dict], right: dict[str, dict]) -> dict[str, int]:
    pairs = [
        (float(left[key]["ruler_official_score"]), float(right[key]["ruler_official_score"]))
        for key in sorted(left)
    ]
    wins = sum(a > b for a, b in pairs)
    losses = sum(a < b for a, b in pairs)
    return {"left_wins": wins, "ties": len(pairs) - wins - losses, "left_losses": losses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("tailspline", "official_yarn", "mrpro"):
        parser.add_argument(f"--{arm.replace('_', '-')}", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "tailspline": args.tailspline,
        "official_yarn": args.official_yarn,
        "mrpro": args.mrpro,
    }
    rows = {name: load_arm(arm_paths, tuple(TASKS)) for name, arm_paths in paths.items()}
    prompt_sets = {name: set(arm_rows) for name, arm_rows in rows.items()}
    if len({frozenset(prompts) for prompts in prompt_sets.values()}) != 1:
        raise ValueError("three Full-13 arms are not paired on identical prompts")

    arms = {name: summarize(arm_rows, tuple(TASKS)) for name, arm_rows in rows.items()}
    scores = {name: value["task_equal_official_score"] for name, value in arms.items()}
    report = {
        "status": "KANANA_64K_THREE_ARM_FULL13_COMPLETE_V1",
        "model": "kakaocorp/kanana-1.5-8b-instruct-2505",
        "target_length": 65_536,
        "tasks": list(TASKS),
        "rows_per_task": 10,
        "paired_prompts": 130,
        "metric": "RULER official score, task-equal macro",
        "arms": arms,
        "task_equal_scores": scores,
        "deltas": {
            "tailspline_minus_official_yarn": scores["tailspline"] - scores["official_yarn"],
            "tailspline_minus_mrpro": scores["tailspline"] - scores["mrpro"],
            "mrpro_minus_official_yarn": scores["mrpro"] - scores["official_yarn"],
        },
        "delta_by_task": {
            task: {
                "tailspline_minus_official_yarn": (
                    arms["tailspline"]["by_task"][task]["official_score"]
                    - arms["official_yarn"]["by_task"][task]["official_score"]
                ),
                "tailspline_minus_mrpro": (
                    arms["tailspline"]["by_task"][task]["official_score"]
                    - arms["mrpro"]["by_task"][task]["official_score"]
                ),
                "mrpro_minus_official_yarn": (
                    arms["mrpro"]["by_task"][task]["official_score"]
                    - arms["official_yarn"]["by_task"][task]["official_score"]
                ),
            }
            for task in TASKS
        },
        "paired_row_outcomes": {
            "tailspline_vs_official_yarn": pair_outcomes(rows["tailspline"], rows["official_yarn"]),
            "tailspline_vs_mrpro": pair_outcomes(rows["tailspline"], rows["mrpro"]),
            "mrpro_vs_official_yarn": pair_outcomes(rows["mrpro"], rows["official_yarn"]),
        },
        "source_sha256": {
            name: [sha256(path) for path in arm_paths] for name, arm_paths in paths.items()
        },
    }
    atomic_json(args.out, report)
    print(json.dumps({"scores": scores, "deltas": report["deltas"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
