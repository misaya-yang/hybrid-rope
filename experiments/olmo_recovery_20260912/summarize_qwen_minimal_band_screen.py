#!/usr/bin/env python3
"""Summarize the Qwen1.5B S=2 Native/BM/MrPro/four-band screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LENGTHS = (32768, 65536)
TASKS = ("niah_single_2", "niah_multikey_2", "niah_multiquery")
ARMS = (
    "Native",
    "BM_s2",
    "MrPro_s2",
    "C42Band14_32_s2",
    "C42Band16_34_s2",
    "C42Band22_39_s2",
    "C42Band23_40_s2",
)


def summarize_arm(generation: dict, ppl: dict) -> dict:
    by_length = {}
    for length in LENGTHS:
        task_values = {}
        for task in TASKS:
            matches = [
                value for key, value in generation["generation_metrics"].items()
                if key.endswith(f"/{task}/{length}")
            ]
            if len(matches) != 1 or matches[0]["rows"] != (2 if length == 32768 else 4):
                raise ValueError(f"incomplete generation cell: {task}/{length}")
            task_values[task] = {
                "rows": matches[0]["rows"],
                "official": matches[0]["ruler_official_score"],
                "exact_plus_eos": matches[0]["exact_plus_eos"],
            }
        lm = ppl["by_length"].get(str(length))
        if not lm or lm["documents"] != 2:
            raise ValueError(f"incomplete PPL cell: {length}")
        by_length[str(length)] = {
            "task_macro_official": sum(value["official"] for value in task_values.values()) / len(task_values),
            "task_macro_exact_plus_eos": sum(value["exact_plus_eos"] for value in task_values.values()) / len(task_values),
            "tasks": task_values,
            "tail512_nll": lm["mean_tail_nll"],
            "tail512_ppl": lm["tail_ppl"],
        }
    return {"by_length": by_length}


def dominators(label: str, arms: dict) -> list[str]:
    current = arms[label]["by_length"]
    vector = tuple(current[str(length)]["task_macro_official"] for length in LENGTHS) + tuple(
        -current[str(length)]["tail512_nll"] for length in LENGTHS
    )
    result = []
    for other_label, other in arms.items():
        if other_label == label:
            continue
        candidate = tuple(other["by_length"][str(length)]["task_macro_official"] for length in LENGTHS) + tuple(
            -other["by_length"][str(length)]["tail512_nll"] for length in LENGTHS
        )
        if all(a >= b for a, b in zip(candidate, vector)) and any(a > b for a, b in zip(candidate, vector)):
            result.append(other_label)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for label in ARMS:
        run = args.root / "runs" / label
        arms[label] = summarize_arm(
            json.loads((run / "generation" / "summary.json").read_text()),
            json.loads((run / "ppl" / "summary.json").read_text()),
        )
    dominance = {label: dominators(label, arms) for label in arms}
    result = {
        "status": "COMPLETE",
        "model": "Qwen2.5-1.5B-Instruct",
        "design_scale": 2,
        "lengths": list(LENGTHS),
        "tasks": list(TASKS),
        "arms": arms,
        "descriptive_dominance": dominance,
        "primary_metrics": "task-equal RULER official and paired two-document tail-512 NLL are separate",
        "scope": "development screen; 2 rows/task at 32K and 4 rows/task at 64K",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        label: {
            "official": [arms[label]["by_length"][str(length)]["task_macro_official"] for length in LENGTHS],
            "ppl": [arms[label]["by_length"][str(length)]["tail512_ppl"] for length in LENGTHS],
            "dominated_by": dominance[label],
        }
        for label in ARMS
    }, sort_keys=True))


if __name__ == "__main__":
    main()
