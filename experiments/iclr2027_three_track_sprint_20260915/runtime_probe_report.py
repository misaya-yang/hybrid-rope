#!/usr/bin/env python3
"""Compare the frozen classic TailSpline batch1 rows with a batch2 replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--batch1", type=Path, required=True)
    parser.add_argument("--batch2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    panel = rows(args.panel)
    wanted = {row["prompt_sha256"] for row in panel}
    if len(panel) != 39 or len(wanted) != 39:
        raise ValueError("runtime probe panel is not 39 unique prompts")
    mappings = {}
    for name, path in (("batch1", args.batch1), ("batch2", args.batch2)):
        values = [row for row in rows(path) if row.get("prompt_sha256") in wanted]
        mapping = {row["prompt_sha256"]: row for row in values}
        if len(values) != 39 or set(mapping) != wanted:
            raise ValueError(f"{name} runtime probe is not complete and paired")
        mappings[name] = mapping
    differences = []
    cells = defaultdict(list)
    for prompt in sorted(wanted):
        left, right = mappings["batch1"][prompt], mappings["batch2"][prompt]
        for key in ("task", "length_cap", "references", "input_tokens"):
            if left.get(key) != right.get(key):
                raise ValueError(f"runtime probe input drift: {prompt}/{key}")
        record = {
            "prompt_sha256": prompt,
            "task": left["task"], "length_cap": int(left["length_cap"]),
            "score_delta_batch2_minus_batch1": float(right["ruler_official_score"]) - float(left["ruler_official_score"]),
            "same_generated_ids": left["generated_ids"] == right["generated_ids"],
            "same_output_text": left["output_text"] == right["output_text"],
            "same_eos": left["ended_eos"] == right["ended_eos"],
            "same_cap": left["hit_cap"] == right["hit_cap"],
        }
        differences.append(record)
        cells[str(record["length_cap"])].append(record)
    deltas = np.asarray([row["score_delta_batch2_minus_batch1"] for row in differences])
    result = {
        "status": "CLASSIC_TAILSPLINE_BATCH_SENSITIVITY_39_COMPLETE_V1",
        "rows": 39,
        "estimate_batch2_minus_batch1": float(deltas.mean()),
        "maximum_absolute_score_delta": float(np.abs(deltas).max()),
        "changed_score_rows": int(np.count_nonzero(deltas)),
        "changed_generation_rows": sum(not row["same_generated_ids"] for row in differences),
        "changed_text_rows": sum(not row["same_output_text"] for row in differences),
        "by_length": {
            length: {
                "rows": len(values),
                "mean_score_delta": float(np.mean([row["score_delta_batch2_minus_batch1"] for row in values])),
                "changed_generation_rows": sum(not row["same_generated_ids"] for row in values),
            }
            for length, values in sorted(cells.items())
        },
        "rows_detail": differences,
        "source_sha256": {"panel": sha256(args.panel), "batch1": sha256(args.batch1), "batch2": sha256(args.batch2)},
        "claim_boundary": "A deterministic 39-cell replay that diagnoses visible batch sensitivity; zero drift does not prove full-panel bit equivalence.",
    }
    atomic_json(args.out, result)
    print(json.dumps({key: result[key] for key in (
        "status", "estimate_batch2_minus_batch1", "maximum_absolute_score_delta",
        "changed_score_rows", "changed_generation_rows",
    )}, indent=2))


if __name__ == "__main__":
    main()
