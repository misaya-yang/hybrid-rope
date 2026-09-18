#!/usr/bin/env python3
"""Build the fresh paired T/C position-gap confirmation report."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from experiments.tc_distance_competition_20260918.report_distance_competition import classify


CELLS = (
    ("near", "neutral"),
    ("far", "neutral"),
    ("near", "structured_kv"),
    ("far", "structured_kv"),
)
BOOTSTRAP_SEED = 2026091802
BOOTSTRAP_DRAWS = 20000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_arm(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    result = {}
    for row in rows:
        row_id = row["row_id"]
        if row_id in result:
            raise ValueError(f"duplicate row: {row_id}")
        result[row_id] = row
    return result


def mean(values) -> float:
    return sum(values) / len(values) if values else 0.0


def interaction(cell_scores: dict[tuple[str, str], float]) -> float:
    return (
        cell_scores[("far", "structured_kv")]
        - cell_scores[("near", "structured_kv")]
        - cell_scores[("far", "neutral")]
        + cell_scores[("near", "neutral")]
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
    if len(source) != len(inputs):
        raise ValueError("duplicate input row")
    tailspline = load_arm(args.tailspline)
    control = load_arm(args.control)
    if set(source) != set(tailspline) or set(source) != set(control):
        raise ValueError("incomplete or mismatched arms")

    rows = []
    for row_id, item in source.items():
        t = tailspline[row_id]
        c = control[row_id]
        for label, generated in (("tailspline", t), ("control", c)):
            if generated["prompt_sha256"] != item["prompt_sha256"]:
                raise ValueError(f"prompt mismatch: {label}/{row_id}")
            if generated["position_ids_sha256"] != item["position_ids_sha256"]:
                raise ValueError(f"position mismatch: {label}/{row_id}")
        if t["table_file_sha256"] == c["table_file_sha256"]:
            raise ValueError("T/C table hashes unexpectedly match")
        rows.append({
            "row_id": row_id,
            "base_sample_id": item["base_sample_id"],
            "distance": item["distance_condition"],
            "competition": item["competition_condition"],
            "t_class": classify(t.get("output_text", ""), item),
            "c_class": classify(c.get("output_text", ""), item),
            "t_official": float(t["ruler_official_score"]),
            "c_official": float(c["ruler_official_score"]),
            "t_eos": bool(t.get("ended_eos")),
            "c_eos": bool(c.get("ended_eos")),
            "t_cap": bool(t.get("hit_cap")),
            "c_cap": bool(c.get("hit_cap")),
        })

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["distance"], row["competition"])].append(row)
    cell_report = {}
    audit_deltas = {}
    official_deltas = {}
    for cell in CELLS:
        values = grouped[cell]
        if len(values) != 64:
            raise ValueError(f"cell does not contain 64 base samples: {cell}")
        t_accuracy = mean([row["t_class"] == "correct" for row in values])
        c_accuracy = mean([row["c_class"] == "correct" for row in values])
        t_official = mean([row["t_official"] for row in values])
        c_official = mean([row["c_official"] for row in values])
        audit_deltas[cell] = t_accuracy - c_accuracy
        official_deltas[cell] = t_official - c_official
        transitions = Counter((row["c_class"], row["t_class"]) for row in values)
        cell_report[f"{cell[0]}_{cell[1]}"] = {
            "rows": len(values),
            "tailspline_audit_accuracy": t_accuracy,
            "control_audit_accuracy": c_accuracy,
            "audit_delta_t_minus_c": audit_deltas[cell],
            "tailspline_official_accuracy": t_official,
            "control_official_accuracy": c_official,
            "official_delta_t_minus_c": official_deltas[cell],
            "confirmed_wrong_binding_repairs": transitions[("wrong_binding", "correct")],
            "confirmed_wrong_binding_damages": transitions[("correct", "wrong_binding")],
            "tailspline_classes": dict(Counter(row["t_class"] for row in values)),
            "control_classes": dict(Counter(row["c_class"] for row in values)),
            "tailspline_eos": sum(row["t_eos"] for row in values),
            "control_eos": sum(row["c_eos"] for row in values),
            "tailspline_cap": sum(row["t_cap"] for row in values),
            "control_cap": sum(row["c_cap"] for row in values),
        }

    by_base = defaultdict(dict)
    for row in rows:
        cell = (row["distance"], row["competition"])
        if cell in by_base[row["base_sample_id"]]:
            raise ValueError("duplicate base/cell row")
        by_base[row["base_sample_id"]][cell] = row
    base_ids = sorted(by_base)
    if len(base_ids) != 64 or any(set(values) != set(CELLS) for values in by_base.values()):
        raise ValueError("incomplete base-sample factorial")

    rng = random.Random(BOOTSTRAP_SEED)
    audit_draws = []
    official_draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [rng.choice(base_ids) for _ in base_ids]
        audit_scores = {}
        official_scores = {}
        for cell in CELLS:
            sampled_rows = [by_base[base_id][cell] for base_id in sampled]
            audit_scores[cell] = mean([
                (row["t_class"] == "correct") - (row["c_class"] == "correct")
                for row in sampled_rows
            ])
            official_scores[cell] = mean([
                row["t_official"] - row["c_official"]
                for row in sampled_rows
            ])
        audit_draws.append(interaction(audit_scores))
        official_draws.append(interaction(official_scores))

    def interval(values: list[float]) -> list[float]:
        values = sorted(values)
        return [
            values[int(0.025 * len(values))],
            values[int(0.975 * len(values)) - 1],
        ]

    audit_point = interaction(audit_deltas)
    official_point = interaction(official_deltas)
    far_distance_effect = {
        competition: (
            audit_deltas[("far", competition)] - audit_deltas[("near", competition)]
        )
        for competition in ("neutral", "structured_kv")
    }
    report = {
        "status": "TC_DISTANCE_COMPETITION_CONFIRM_COMPLETE_V2",
        "base_samples": len(base_ids),
        "primary_inference_unit": "base_sample",
        "rows_per_arm": len(rows),
        "cells": cell_report,
        "primary_interaction": {
            "metric": "strict behavior-audit correct classification",
            "definition": (
                "[(T-C)_far,structured_kv-(T-C)_near,structured_kv]-"
                "[(T-C)_far,neutral-(T-C)_near,neutral]"
            ),
            "value": audit_point,
            "paired_base_bootstrap_interval95": interval(audit_draws),
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
        },
        "secondary_official_interaction": {
            "metric": "official RULER substring score",
            "value": official_point,
            "paired_base_bootstrap_interval95": interval(official_draws),
        },
        "audit_distance_effect_by_context": far_distance_effect,
        "interpretation_rule": {
            "positive_primary_interaction": (
                "supports the preregistered distance-by-structured-context behavior prediction"
            ),
            "interval_crosses_zero": "interaction remains unresolved at 64 base samples",
            "nonpositive_primary_interaction": "does not support the preregistered interaction",
            "wrong_binding_secondary": (
                "confirmed wrong-binding transitions are secondary and cannot replace the primary interaction"
            ),
        },
        "table_hashes": {
            "tailspline": next(iter(tailspline.values()))["table_file_sha256"],
            "dose_control_c": next(iter(control.values()))["table_file_sha256"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({
        "status": report["status"],
        "primary_interaction": audit_point,
        "primary_interval95": report["primary_interaction"]["paired_base_bootstrap_interval95"],
        "official_interaction": official_point,
        "cell_audit_deltas": {
            f"{distance}_{competition}": audit_deltas[(distance, competition)]
            for distance, competition in CELLS
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
