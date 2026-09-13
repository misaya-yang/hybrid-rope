#!/usr/bin/env python3
"""Summarize the minimal Llama S=2 PPL + passkey/NIAH band screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def summarize_arm(summary: dict, lengths: list[int]) -> dict:
    by_length = {}
    for length in lengths:
        cells = [
            value for key, value in summary["generation_metrics"].items()
            if key.endswith(f"/{length}")
        ]
        if len(cells) != 3 or any(cell["rows"] != 4 for cell in cells):
            raise ValueError(f"generation cells incomplete at {length}")
        by_length[str(length)] = {
            "task_macro_official": sum(cell["ruler_official_score"] for cell in cells) / len(cells),
            "task_macro_exact_plus_eos": sum(cell["exact_plus_eos"] for cell in cells) / len(cells),
        }
    lm = {}
    for length in lengths:
        value = summary["lm_metrics"].get(str(length))
        if not value or value["documents"] != 2:
            raise ValueError(f"LM cells incomplete at {length}")
        lm[str(length)] = {
            **value,
            "whole_ppl": math.exp(value["whole_nll"]),
        }
    return {"by_length": by_length, "lm": lm}


def dominated(label: str, arms: dict, lengths: list[int]) -> list[str]:
    current = arms[label]
    values = tuple(current["by_length"][str(length)]["task_macro_official"] for length in lengths) + tuple(
        -current["lm"][str(length)]["whole_nll"] for length in lengths
    )
    result = []
    for other_label, other in arms.items():
        if other_label == label:
            continue
        other_values = tuple(other["by_length"][str(length)]["task_macro_official"] for length in lengths) + tuple(
            -other["lm"][str(length)]["whole_nll"] for length in lengths
        )
        if all(a >= b for a, b in zip(other_values, values)) and any(a > b for a, b in zip(other_values, values)):
            result.append(other_label)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    lengths = [int(length) for length in manifest.get(
        "evaluation_lengths", [manifest["native_length"], manifest["horizon"]]
    )]
    arms = {}
    for label in manifest["arms"]:
        status = json.loads((args.root / "runs" / label / "status.json").read_text())
        if status.get("status") != "COMPLETE":
            raise ValueError(f"arm incomplete: {label}")
        arms[label] = summarize_arm(
            json.loads((args.root / "runs" / label / "summary.json").read_text()),
            lengths,
        )
    dominance = {label: dominated(label, arms, lengths) for label in arms}
    result = {
        "status": "COMPLETE",
        "problem": f"minimal Llama S={manifest['scale']:g} band-position screen",
        "primary_metrics": "per-length task-equal official and two-document PG19 whole PPL are reported separately",
        "evaluation_lengths": lengths,
        "arms": arms,
        "descriptive_dominance": dominance,
        "decision_policy": "dominance is descriptive; no arm is deleted without S=4 confirmation or a protocol failure",
        "scope": "three tasks x four rows at 8K/16K and two fixed PG19 documents; development screen",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "COMPLETE",
        "arms": {
            label: {
                "official": {str(length): value["by_length"][str(length)]["task_macro_official"] for length in lengths},
                "ppl": {str(length): value["lm"][str(length)]["whole_ppl"] for length in lengths},
                "dominated_by": dominance[label],
            }
            for label, value in arms.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
