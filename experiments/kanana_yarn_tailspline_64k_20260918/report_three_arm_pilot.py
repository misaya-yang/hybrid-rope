#!/usr/bin/env python3
"""Build the three-arm Kanana 64K paired pilot report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare import PILOT_TASKS
from .report import atomic_json, load_arm, sha256, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tailspline", type=Path, required=True)
    parser.add_argument("--official-yarn", type=Path, required=True)
    parser.add_argument("--mrpro", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "tailspline": args.tailspline,
        "official_yarn": args.official_yarn,
        "mrpro": args.mrpro,
    }
    rows = {name: load_arm([path], PILOT_TASKS) for name, path in paths.items()}
    prompts = {name: set(value) for name, value in rows.items()}
    if len({frozenset(value) for value in prompts.values()}) != 1:
        raise ValueError("three pilot arms are not paired on identical prompts")
    arms = {name: summarize(value, PILOT_TASKS) for name, value in rows.items()}
    score = {name: value["task_equal_official_score"] for name, value in arms.items()}
    by_task = {
        task: {name: arms[name]["by_task"][task]["official_score"] for name in arms}
        for task in PILOT_TASKS
    }
    report = {
        "status": "KANANA_64K_THREE_ARM_PILOT_COMPLETE_V1",
        "model": "kakaocorp/kanana-1.5-8b-instruct-2505",
        "target_length": 65_536,
        "tasks": list(PILOT_TASKS),
        "rows_per_task": 10,
        "paired_prompts": 20,
        "metric": "RULER official score, task-equal macro",
        "arms": arms,
        "task_equal_scores": score,
        "deltas": {
            "tailspline_minus_official_yarn": score["tailspline"] - score["official_yarn"],
            "tailspline_minus_mrpro": score["tailspline"] - score["mrpro"],
            "mrpro_minus_official_yarn": score["mrpro"] - score["official_yarn"],
        },
        "scores_by_task": by_task,
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    atomic_json(args.out, report)
    print(json.dumps({"scores": score, "deltas": report["deltas"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
