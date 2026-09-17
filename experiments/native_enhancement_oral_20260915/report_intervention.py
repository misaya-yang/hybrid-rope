#!/usr/bin/env python3
"""Report the two fixed final-quarter phase interventions on 64 frozen rows."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapping(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    result = {str(row["row_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate row ids: {path}")
    return result


def correct(row: dict) -> float:
    if len(row["references"]) != 1:
        raise ValueError("mechanism row must have one complete-answer reference")
    return float(row["output_text"].strip() == row["references"][0].strip())


def compare(panel: dict[str, dict], baseline: dict[str, dict], intervention: dict[str, dict]) -> dict:
    if set(intervention) != set(panel) or not set(panel).issubset(baseline):
        raise ValueError("intervention rows are not paired to the frozen panel/baseline")
    cells = defaultdict(list)
    deltas = []
    for row_id, source in panel.items():
        for arm, row in (("baseline", baseline[row_id]), ("intervention", intervention[row_id])):
            for field in ("task", "prompt_sha256", "references", "input_tokens"):
                if row.get(field) != source.get(field):
                    raise ValueError(f"{arm} input identity drift: {row_id}/{field}")
        left, right = correct(baseline[row_id]), correct(intervention[row_id])
        cells[source["task"], int(source["length_cap"])].append((left, right))
        deltas.append(right - left)
    by_cell = {}
    for (task, length), values in sorted(cells.items()):
        array = np.asarray(values)
        by_cell[f"{task}:{length}"] = {
            "rows": len(values), "baseline": float(array[:, 0].mean()),
            "intervention": float(array[:, 1].mean()),
            "effect": float((array[:, 1] - array[:, 0]).mean()),
        }
    return {
        "rows": len(panel),
        "baseline": float(np.mean([correct(baseline[row_id]) for row_id in panel])),
        "intervention": float(np.mean([correct(intervention[row_id]) for row_id in panel])),
        "effect": float(np.mean(deltas)),
        "candidate_only_correct": int(np.count_nonzero(np.asarray(deltas) > 0)),
        "baseline_only_correct": int(np.count_nonzero(np.asarray(deltas) < 0)),
        "ties": int(np.count_nonzero(np.asarray(deltas) == 0)),
        "by_task_length": by_cell,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--ncp", type=Path, required=True)
    parser.add_argument("--native-with-ncp-block", type=Path, required=True)
    parser.add_argument("--ncp-with-native-block", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    panel_rows = read_jsonl(args.panel)
    if len(panel_rows) != 64:
        raise ValueError("fixed layer intervention requires exactly 64 preregistered rows")
    panel = {str(row["row_id"]): row for row in panel_rows}
    native, ncp = mapping(args.native), mapping(args.ncp)
    native_patch = mapping(args.native_with_ncp_block)
    ncp_patch = mapping(args.ncp_with_native_block)
    report = {
        "status": "NATIVE_FINAL_QUARTER_PHASE_INTERVENTION_COMPLETE_V1",
        "panel_sha256": sha256(args.panel),
        "native_to_ncp_block": compare(panel, native, native_patch),
        "ncp_to_native_block": compare(panel, ncp, ncp_patch),
        "raw_sha256": {
            "native": sha256(args.native), "ncp": sha256(args.ncp),
            "native_with_ncp_block": sha256(args.native_with_ncp_block),
            "ncp_with_native_block": sha256(args.ncp_with_native_block),
        },
        "scope": (
            "Fixed final-quarter layer block and output-blind 64-row synthetic mechanism subset. "
            "It tests one local mediator; it is not a deployable table or broad task estimate."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": report["status"], "rows": 64}))


if __name__ == "__main__":
    main()
