#!/usr/bin/env python3
"""Reconcile the Stage 1 behavior classifier with the official RULER scorer."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from transformers import AutoTokenizer

from experiments.tc_distance_competition_20260918.audit_existing_multikey import (
    TASKS,
    classify,
    load_outputs,
    parse_prompt,
    read_jsonl,
)
from scripts.experiments.olmo_fast_screen.ruler_bench import score as official_score


def summarize(rows: list[dict]) -> dict:
    audit_correct = sum(row["audit_class"] == "correct" for row in rows)
    official_correct = sum(row["official_score"] == 1.0 for row in rows)
    mismatch_reasons = Counter(row["reconciliation"] for row in rows)
    return {
        "rows": len(rows),
        "audit_correct": audit_correct,
        "official_correct": official_correct,
        "audit_accuracy": audit_correct / len(rows) if rows else None,
        "official_accuracy": official_correct / len(rows) if rows else None,
        "official_minus_audit_correct": official_correct - audit_correct,
        "classification_counts": dict(Counter(row["audit_class"] for row in rows)),
        "reconciliation_counts": dict(mismatch_reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", action="append", type=Path, required=True)
    parser.add_argument("--tailspline", action="append", type=Path, required=True)
    parser.add_argument("--control", action="append", type=Path, required=True)
    parser.add_argument("--stage1-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stage1 = json.loads(args.stage1_report.read_text())
    if (
        stage1.get("status") != "TC_EXISTING_MULTIKEY_BEHAVIOR_AUDIT_COMPLETE_V2"
        or stage1.get("rows_expected") != 750
        or stage1.get("rows_qualified") != 750
        or stage1.get("mapping_coverage") != 1.0
    ):
        raise ValueError("Stage 1 V2 complete-row report is required")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generations = {
        "tailspline": load_outputs(args.tailspline),
        "dose_control_c": load_outputs(args.control),
    }
    if set(generations["tailspline"]) != set(generations["dose_control_c"]):
        raise ValueError("T/C output row IDs differ")

    sources = {}
    for panel_path in args.panel:
        for source in read_jsonl(panel_path):
            if source.get("task") not in TASKS:
                continue
            row_id = source["row_id"]
            if row_id in sources:
                raise ValueError(f"duplicate source row: {row_id}")
            sources[row_id] = source
    if set(sources) != set(generations["tailspline"]):
        raise ValueError("source and generation row IDs differ")

    rows = []
    mismatches = []
    for row_id, source in sources.items():
        parsed = parse_prompt(source, tokenizer)
        if parsed["source_ambiguous"]:
            raise ValueError(f"V2 source unexpectedly ambiguous: {row_id}")
        for arm, arm_rows in generations.items():
            generated = arm_rows[row_id]
            if generated["prompt_sha256"] != source["prompt_sha256"]:
                raise ValueError(f"prompt identity mismatch: {arm}/{row_id}")
            text = generated.get("output_text", "")
            audit_class = classify(text, parsed)
            recomputed = float(official_score(source, text))
            recorded = float(generated.get("ruler_official_score"))
            if recomputed != recorded:
                raise ValueError(f"recorded official score mismatch: {arm}/{row_id}")
            official_correct = recomputed == 1.0
            audit_correct = audit_class == "correct"
            if official_correct and not audit_correct:
                reconciliation = f"official_correct_audit_{audit_class}"
            elif audit_correct and not official_correct:
                reconciliation = "audit_correct_official_incorrect"
            else:
                reconciliation = "agreement"
            row = {
                "row_id": row_id,
                "length": int(source["length_cap"]),
                "task": source["task"],
                "arm": arm,
                "audit_class": audit_class,
                "official_score": recomputed,
                "reconciliation": reconciliation,
            }
            rows.append(row)
            if reconciliation != "agreement":
                mismatches.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["length"], row["task"], row["arm"])].append(row)
    by_length_task = {}
    for length in sorted({row["length"] for row in rows}):
        by_length_task[str(length)] = {}
        for task in TASKS:
            by_length_task[str(length)][task] = {
                arm: summarize(groups[(length, task, arm)])
                for arm in generations
            }

    contrasts = {}
    for length, tasks in by_length_task.items():
        contrasts[length] = {}
        for task, arms in tasks.items():
            tail = arms["tailspline"]
            control = arms["dose_control_c"]
            contrasts[length][task] = {
                "official_t_minus_c": (
                    tail["official_accuracy"] - control["official_accuracy"]
                ),
                "audit_t_minus_c": tail["audit_accuracy"] - control["audit_accuracy"],
                "difference_in_t_minus_c_official_minus_audit": (
                    tail["official_accuracy"]
                    - control["official_accuracy"]
                    - tail["audit_accuracy"]
                    + control["audit_accuracy"]
                ),
            }

    report = {
        "status": "TC_STAGE1_OFFICIAL_RECONCILIATION_COMPLETE_V1",
        "stage1_report_status": stage1["status"],
        "source_rows": len(sources),
        "arm_rows": {arm: len(values) for arm, values in generations.items()},
        "official_scorer_contract": (
            "NVIDIA RULER substring matching: a one-reference multikey row is correct "
            "whenever the gold value occurs anywhere in the response"
        ),
        "audit_correct_contract": (
            "exactly one prompt candidate value occurs in the response and it is gold"
        ),
        "by_length_task_arm": by_length_task,
        "contrasts": contrasts,
        "mismatch_rows": mismatches,
        "interpretation": (
            "Official-only correct rows contain the gold substring but fail the stricter "
            "behavior classification; scorer disagreement is retained rather than forced away."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({
        "status": report["status"],
        "source_rows": report["source_rows"],
        "mismatch_rows": len(mismatches),
        "contrasts_32768": contrasts["32768"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
