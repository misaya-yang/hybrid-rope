"""Validate the P control unit and decide only whether the S instrument is usable."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import paired_report as P


CONTROLS = ("MR", "OfficialYaRN", "BM", "UNI")
TASKS = (
    "niah_single_2", "niah_multikey_2", "niah_multivalue", "niah_multiquery",
    "vt", "fwe", "qa_1", "qa_2",
)
CAPS = (8192, 16384, 32768)
FLOOR = 0.05
CEILING = 0.95


def rows(path):
    return P.load_jsonl(path)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/autodl-tmp/llama3_planb_20260911")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root = Path(args.root)
    expected = rows(root / "data" / "P" / "rows.jsonl")
    expected_long = expected
    expected_native = [r for r in expected if int(r["length_cap"]) == 8192]
    arms = {arm: rows(root / "results" / "P_l0" / f"{arm}.jsonl")
            for arm in CONTROLS}
    native = rows(root / "results" / "P_native" / "Native.jsonl")
    for path in (root / "results" / "P_l0" / "run_summary.json",
                 root / "results" / "P_native" / "run_summary.json"):
        if json.loads(path.read_text()).get("status") != "COMPLETE":
            raise ValueError(f"P run summary is not COMPLETE: {path}")
    P.require_aligned({name: P.index_arm(value, name) for name, value in arms.items()},
                      expected_long)
    P.require_aligned({"Native": P.index_arm(native, "Native")}, expected_native)

    summaries = {arm: P.macro_accuracy(
        value, required_tasks=TASKS, required_lengths=CAPS) for arm, value in arms.items()}
    native_summary = P.macro_accuracy(
        native, required_tasks=TASKS, required_lengths=[8192])
    cell_values = defaultdict(dict)
    for arm, values in arms.items():
        cells = defaultdict(list)
        for row in values:
            cells[(row["task"], int(row["length_cap"]))].append(
                float(row.get("partial_score", row["correct"])))
        for cell, scores in cells.items():
            cell_values[cell][arm] = sum(scores) / len(scores)
    limited_cells = {}
    for cell, values in sorted(cell_values.items()):
        if all(v <= FLOOR for v in values.values()):
            limited_cells[f"{cell[0]}|{cell[1]}"] = "FLOOR"
        elif all(v >= CEILING for v in values.values()):
            limited_cells[f"{cell[0]}|{cell[1]}"] = "CEILING"
    unusable_lengths = {}
    for cap in (16384, 32768):
        vals = [summaries[arm]["by_length"][cap] for arm in CONTROLS]
        if all(v <= FLOOR for v in vals):
            unusable_lengths[str(cap)] = "ALL_CONTROLS_FLOOR"
        elif all(v >= CEILING for v in vals):
            unusable_lengths[str(cap)] = "ALL_CONTROLS_CEILING"
    report = {
        "status": "READY_FOR_S" if not unusable_lengths else "INSTRUMENT_LIMITED",
        "evidence_tier": "P instrument only; not candidate selection or confirmation",
        "thresholds": {"floor": FLOOR, "ceiling": CEILING},
        "controls": summaries, "native_8k": native_summary,
        "instrument_limited_cells": limited_cells,
        "unusable_long_lengths": unusable_lengths,
        "identity": {"P_rows": len(expected), "native_rows": len(expected_native),
                     "source_identity_revision": expected[0].get("source_identity_revision")},
        "decision_rule": ("S is blocked only if every strong-control macro at an entire long "
                          "length is <=5% or >=95%; individual limited cells remain reported."),
    }
    atomic_json(args.out, report)
    print(json.dumps(report))
    return 0 if report["status"] == "READY_FOR_S" else 3


if __name__ == "__main__":
    raise SystemExit(main())
