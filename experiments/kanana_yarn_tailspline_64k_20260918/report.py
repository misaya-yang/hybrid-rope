#!/usr/bin/env python3
"""Report the paired Kanana 64K pilot or Full-13 result and decide expansion."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

from .prepare import PILOT_TASKS, ROWS_PER_TASK, TARGET_LENGTH
from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_arm(paths: list[Path], expected_tasks: tuple[str, ...]) -> dict[str, dict]:
    rows = [row for path in paths for row in read_jsonl(path)]
    by_prompt = {}
    counts = Counter()
    for row in rows:
        prompt = str(row.get("prompt_sha256", ""))
        task = str(row.get("task", ""))
        if (
            not prompt or prompt in by_prompt or task not in expected_tasks
            or int(row.get("length_cap", -1)) != TARGET_LENGTH
            or row.get("ruler_official_score") is None
        ):
            raise ValueError("generation output violates the frozen paired contract")
        by_prompt[prompt] = row
        counts[task] += 1
    expected = Counter({task: ROWS_PER_TASK for task in expected_tasks})
    if counts != expected:
        raise ValueError(f"generation coverage drift: {dict(counts)} != {dict(expected)}")
    return by_prompt


def summarize(rows: dict[str, dict], tasks: tuple[str, ...]) -> dict:
    grouped = defaultdict(list)
    for row in rows.values():
        grouped[row["task"]].append(row)
    by_task = {}
    for task in tasks:
        values = grouped[task]
        by_task[task] = {
            "rows": len(values),
            "official_score": sum(float(row["ruler_official_score"]) for row in values) / len(values),
            "eos_rate": sum(bool(row.get("ended_eos")) for row in values) / len(values),
            "cap_rate": sum(bool(row.get("hit_cap")) for row in values) / len(values),
            "empty_rate": sum(bool(row.get("empty")) for row in values) / len(values),
        }
    return {
        "rows": len(rows),
        "task_equal_official_score": sum(
            value["official_score"] for value in by_task.values()
        ) / len(by_task),
        "task_equal_eos_rate": sum(value["eos_rate"] for value in by_task.values()) / len(by_task),
        "task_equal_cap_rate": sum(value["cap_rate"] for value in by_task.values()) / len(by_task),
        "task_equal_empty_rate": sum(value["empty_rate"] for value in by_task.values()) / len(by_task),
        "by_task": by_task,
    }


def build_report(
    *, tailspline_paths: list[Path], yarn_paths: list[Path], mode: str,
    expand_threshold: float,
) -> dict:
    tasks = PILOT_TASKS if mode == "pilot" else tuple(TASKS)
    tailspline = load_arm(tailspline_paths, tasks)
    yarn = load_arm(yarn_paths, tasks)
    if set(tailspline) != set(yarn):
        raise ValueError("TailSpline and YaRN are not paired on identical prompts")
    summaries = {
        "tailspline": summarize(tailspline, tasks),
        "official_yarn": summarize(yarn, tasks),
    }
    task_delta = {
        task: summaries["tailspline"]["by_task"][task]["official_score"]
        - summaries["official_yarn"]["by_task"][task]["official_score"]
        for task in tasks
    }
    delta = (
        summaries["tailspline"]["task_equal_official_score"]
        - summaries["official_yarn"]["task_equal_official_score"]
    )
    pairs = [(float(tailspline[key]["ruler_official_score"]),
              float(yarn[key]["ruler_official_score"])) for key in sorted(tailspline)]
    wins = sum(left > right for left, right in pairs)
    losses = sum(left < right for left, right in pairs)
    ties = len(pairs) - wins - losses
    expand = mode == "pilot" and abs(delta) <= expand_threshold
    return {
        "status": "KANANA_64K_PAIRED_RULER_REPORT_V1",
        "mode": mode,
        "model": "kakaocorp/kanana-1.5-8b-instruct-2505",
        "target_length": TARGET_LENGTH,
        "tasks": list(tasks),
        "rows_per_task": ROWS_PER_TASK,
        "paired_prompts": len(pairs),
        "metric": "RULER official score, task-equal macro",
        "arms": summaries,
        "delta_tailspline_minus_official_yarn": delta,
        "delta_by_task": task_delta,
        "paired_row_outcomes": {"tailspline_wins": wins, "ties": ties, "tailspline_losses": losses},
        "gate": {
            "absolute_delta_threshold": expand_threshold,
            "expand_to_full13": expand,
            "decision": (
                "EXPAND_FULL13" if expand
                else "FULL13_COMPLETE" if mode == "full"
                else "STOP_DIRECTIONALLY_LARGE_PILOT"
            ),
        },
        "source_sha256": {
            "tailspline": [sha256(path) for path in tailspline_paths],
            "official_yarn": [sha256(path) for path in yarn_paths],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--tailspline", type=Path, action="append", required=True)
    parser.add_argument("--official-yarn", type=Path, action="append", required=True)
    parser.add_argument("--expand-threshold", type=float, default=0.10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.expand_threshold <= 1:
        raise ValueError("expand threshold must be in [0,1]")
    report = build_report(
        tailspline_paths=args.tailspline,
        yarn_paths=args.official_yarn,
        mode=args.mode,
        expand_threshold=args.expand_threshold,
    )
    atomic_json(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "mode": report["mode"],
        "scores": {
            arm: value["task_equal_official_score"] for arm, value in report["arms"].items()
        },
        "delta": report["delta_tailspline_minus_official_yarn"],
        "decision": report["gate"]["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
