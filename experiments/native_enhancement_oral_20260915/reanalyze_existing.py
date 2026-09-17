#!/usr/bin/env python3
"""CPU-only reanalysis of the paired Native/H/B/V1/NCP 780-row outputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ARMS = ("native", "halfturn", "reverse", "v1", "ncp")
PAIR_FIELDS = ("task", "prompt_sha256", "references", "input_tokens")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def found_references(row: dict) -> set[str]:
    output = row["output_text"].lower()
    return {str(reference).lower() for reference in row["references"]
            if str(reference).lower() in output}


def typed_candidates(row: dict) -> set[str] | None:
    refs = [str(value) for value in row["references"]]
    text = row["output_text"]
    if refs and all(re.fullmatch(r"\d+", value) for value in refs):
        lengths = sorted({len(value) for value in refs})
        return {value for value in re.findall(r"\b\d+\b", text) if len(value) in lengths}
    uuid = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    if refs and all(re.fullmatch(uuid, value.lower()) for value in refs):
        return set(re.findall(uuid, text.lower()))
    if refs and all(re.fullmatch(r"[A-Z]{5}", value) for value in refs):
        return set(re.findall(r"\b[A-Z]{5}\b", text))
    return None


def row_metrics(row: dict) -> dict:
    references = {str(value).lower() for value in row["references"]}
    found = found_references(row)
    candidates = typed_candidates(row)
    literal = any(row["output_text"].strip() == str(value).strip() for value in row["references"])
    complete = any(normalized(row["output_text"]) == normalized(str(value)) for value in row["references"])
    official = float(row["ruler_official_score"])
    if official == 1.0:
        error = "official_complete"
    elif row.get("empty"):
        error = "empty"
    elif row.get("hit_cap"):
        error = "hit_cap"
    elif found:
        error = "partial_or_extraneous"
    else:
        error = "no_reference_recovered"
    precision = None
    if candidates is not None:
        precision = len(candidates & references) / len(candidates) if candidates else 0.0
    return {
        "official": official,
        "literal_complete_answer": float(literal),
        "normalized_complete_answer": float(complete),
        "reference_set_recall": len(found) / len(references),
        "typed_set_precision": precision,
        "ended_eos": float(bool(row.get("ended_eos"))),
        "hit_cap": float(bool(row.get("hit_cap"))),
        "empty": float(bool(row.get("empty"))),
        "generated_tokens": len(row.get("generated_ids") or []),
        "error_type": error,
    }


def load_run(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    if len(rows) != 780:
        raise ValueError(f"expected 780 rows: {path}")
    mapping = {str(row["row_id"]): row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"duplicate row identity: {path}")
    return mapping


def _means(metrics: list[dict]) -> dict:
    names = (
        "official", "literal_complete_answer", "normalized_complete_answer",
        "reference_set_recall", "ended_eos", "hit_cap", "empty", "generated_tokens",
    )
    result = {name: float(np.mean([row[name] for row in metrics])) for name in names}
    precision = [row["typed_set_precision"] for row in metrics
                 if row["typed_set_precision"] is not None]
    result["typed_set_precision"] = float(np.mean(precision)) if precision else None
    result["typed_set_precision_rows"] = len(precision)
    result["error_types"] = dict(Counter(row["error_type"] for row in metrics))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="ARM=GENERATIONS_JSONL")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {}
    for spec in args.run:
        arm, separator, raw_path = spec.partition("=")
        if not separator or arm not in ARMS or arm in paths:
            raise ValueError("--run requires each unique canonical ARM=PATH")
        paths[arm] = Path(raw_path)
    if set(paths) != set(ARMS):
        raise ValueError(f"five paired arms are required: {ARMS}")
    runs = {arm: load_run(path) for arm, path in paths.items()}
    identities = list(runs["native"])
    if any(set(rows) != set(identities) for rows in runs.values()):
        raise ValueError("arm row identities differ")
    for row_id in identities:
        baseline = runs["native"][row_id]
        for arm, rows in runs.items():
            if any(rows[row_id].get(field) != baseline.get(field) for field in PAIR_FIELDS):
                raise ValueError(f"paired input drift: {arm}/{row_id}")
    detailed = {arm: {row_id: row_metrics(rows[row_id]) for row_id in identities}
                for arm, rows in runs.items()}
    tasks = sorted({runs["native"][row_id]["task"] for row_id in identities})
    arm_summary, task_summary = {}, {}
    for arm in ARMS:
        arm_summary[arm] = _means(list(detailed[arm].values()))
        task_summary[arm] = {
            task: _means([detailed[arm][row_id] for row_id in identities
                          if runs[arm][row_id]["task"] == task])
            for task in tasks
        }
    comparisons = {}
    for arm in ARMS[1:]:
        official_by_task = []
        changed = Counter()
        for task in tasks:
            row_ids = [row_id for row_id in identities if runs[arm][row_id]["task"] == task]
            official_by_task.append(float(np.mean([
                detailed[arm][row_id]["official"] - detailed["native"][row_id]["official"]
                for row_id in row_ids
            ])))
        for row_id in identities:
            delta = detailed[arm][row_id]["official"] - detailed["native"][row_id]["official"]
            changed["candidate_only_better" if delta > 0 else "native_only_better" if delta < 0 else "tie"] += 1
        comparisons[f"{arm}_minus_native"] = {
            "official_task_equal_delta": float(np.mean(official_by_task)),
            "row_outcomes": dict(changed),
        }
    report = {
        "status": "OLMO_NATIVE_EXISTING_FIVE_ARM_REANALYSIS_V1",
        "rows_per_arm": len(identities),
        "tasks": tasks,
        "arm_summary": arm_summary,
        "arm_task_summary": task_summary,
        "comparisons": comparisons,
        "raw_sha256": {arm: sha256(path) for arm, path in paths.items()},
        "metric_contract": {
            "primary": "stored official RULER score; task-equal across 13 tasks",
            "complete_answer": "whole decoded response after outer whitespace or alphanumeric normalization",
            "reference_set_recall": "fraction of distinct declared references appearing literally in output",
            "typed_set_precision": (
                "precision among unambiguous numeric, UUID, or uppercase-five-code candidates; "
                "undefined for free-form lexical and QA outputs"
            ),
            "bootstrap": "not recomputed here; existing official paired reports retain their intervals",
        },
        "scope": "CPU reanalysis of one shared 780-row development panel; not a new model run.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": report["status"], "comparisons": comparisons}, sort_keys=True))


if __name__ == "__main__":
    main()
