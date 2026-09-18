#!/usr/bin/env python3
"""Build the paired T/C distance-by-competition interaction report."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re


NUMBER = re.compile(r"(?<!\d)\d{7}(?!\d)")
CELLS = (("near", "weak"), ("far", "weak"), ("near", "strong"), ("far", "strong"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_arm(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    result = {}
    for row in rows:
        if row["row_id"] in result:
            raise ValueError(f"duplicate row {row['row_id']}")
        result[row["row_id"]] = row
    return result


def classify(text: str, source: dict) -> str:
    if not str(text).strip():
        return "empty"
    candidates = set(source["candidate_values"])
    mentioned = {value for value in NUMBER.findall(str(text)) if value in candidates}
    if len(mentioned) > 1:
        return "ambiguous"
    if len(mentioned) == 1:
        value = next(iter(mentioned))
        return "correct" if value == source["target_value"] else "wrong_binding"
    return "other"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def interaction(cell_scores: dict[tuple[str, str], float]) -> float:
    return (
        (cell_scores[("far", "strong")] - cell_scores[("far", "weak")])
        - (cell_scores[("near", "strong")] - cell_scores[("near", "weak")])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--tailspline", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = read_jsonl(args.inputs)
    source = {row["row_id"]: row for row in inputs}
    tailspline, control = load_arm(args.tailspline), load_arm(args.control)
    if set(source) != set(tailspline) or set(source) != set(control):
        raise ValueError("incomplete or mismatched arms")
    rows = []
    for row_id, item in source.items():
        t, c = tailspline[row_id], control[row_id]
        if t["prompt_sha256"] != item["prompt_sha256"] or c["prompt_sha256"] != item["prompt_sha256"]:
            raise ValueError(f"prompt mismatch {row_id}")
        rows.append({
            "row_id": row_id,
            "base_sample_id": item["base_sample_id"],
            "distance": item["distance_condition"],
            "competition": item["competition_condition"],
            "t_class": classify(t.get("output_text", ""), item),
            "c_class": classify(c.get("output_text", ""), item),
            "t_eos": bool(t.get("ended_eos")),
            "c_eos": bool(c.get("ended_eos")),
            "t_cap": bool(t.get("hit_cap")),
            "c_cap": bool(c.get("hit_cap")),
        })

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["distance"], row["competition"])].append(row)
    cell_report, deltas = {}, {}
    for cell in CELLS:
        values = grouped[cell]
        t_score = mean([row["t_class"] == "correct" for row in values])
        c_score = mean([row["c_class"] == "correct" for row in values])
        deltas[cell] = t_score - c_score
        transitions = Counter((row["c_class"], row["t_class"]) for row in values)
        cell_report[f"{cell[0]}_{cell[1]}"] = {
            "rows": len(values),
            "tailspline_accuracy": t_score,
            "control_accuracy": c_score,
            "delta_t_minus_c": t_score - c_score,
            "confirmed_wrong_binding_repairs": transitions[("wrong_binding", "correct")],
            "confirmed_wrong_binding_damages": transitions[("correct", "wrong_binding")],
            "tailspline_classes": dict(Counter(row["t_class"] for row in values)),
            "control_classes": dict(Counter(row["c_class"] for row in values)),
            "tailspline_eos": sum(row["t_eos"] for row in values),
            "control_eos": sum(row["c_eos"] for row in values),
            "tailspline_cap": sum(row["t_cap"] for row in values),
            "control_cap": sum(row["c_cap"] for row in values),
        }
    point = interaction(deltas)

    by_base = defaultdict(list)
    for row in rows:
        by_base[row["base_sample_id"]].append(row)
    base_ids = sorted(by_base)
    rng = random.Random(20260918)
    draws = []
    for _ in range(20000):
        sampled = [rng.choice(base_ids) for _ in base_ids]
        scores = {}
        for cell in CELLS:
            t_values, c_values = [], []
            for base_id in sampled:
                row = next(value for value in by_base[base_id] if (value["distance"], value["competition"]) == cell)
                t_values.append(row["t_class"] == "correct")
                c_values.append(row["c_class"] == "correct")
            scores[cell] = mean(t_values) - mean(c_values)
        draws.append(interaction(scores))
    draws.sort()
    interval = [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]]
    report = {
        "status": "TC_DISTANCE_COMPETITION_COMPLETE_V1",
        "base_samples": len(base_ids),
        "rows_per_arm": len(rows),
        "cells": cell_report,
        "primary_interaction": {
            "definition": "[(T-C)_far,strong-(T-C)_far,weak]-[(T-C)_near,strong-(T-C)_near,weak]",
            "value": point,
            "paired_base_bootstrap_interval95": interval,
            "draws": len(draws),
            "seed": 20260918,
        },
        "interpretation_rule": {
            "positive_interaction": "supports the preregistered distance-by-competition behavior prediction",
            "interval_crosses_zero": "direction observed but interaction unresolved at this sample budget",
            "nonpositive_interaction": "does not support the preregistered prediction",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "interaction": point, "interval95": interval, "cell_deltas": {f"{a}_{b}": deltas[(a,b)] for a,b in CELLS}}, indent=2))


if __name__ == "__main__":
    main()

