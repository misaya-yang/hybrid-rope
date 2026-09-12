#!/usr/bin/env python3
"""Audit complete E3 generation records and add strict output diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from experiments.rope_fast_5090_20260912.e3_validate import TASKS, load_rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepared = {row["row_id"]: row for row in load_rows(args.prepared / "screen.jsonl")}
    arms = {name: load_rows(args.run / f"{name}.jsonl") for name in ("C42", "C42V24")}
    if any({row["row_id"] for row in values} != set(prepared) for values in arms.values()):
        raise ValueError("A/B outputs do not exactly match the frozen panel")
    detail = {}
    for arm, values in arms.items():
        cells = defaultdict(list)
        enriched = []
        for record in values:
            source = prepared[record["row_id"]]
            generated = record["generated_ids"]
            strict = normalize(record["output_text"]) in {normalize(ref) for ref in source["references"]}
            hit_cap = len(generated) == source["max_new_tokens"] and not record["ended_eos"]
            item = {"row_id": record["row_id"], "task": source["task"], "length_cap": source["length_cap"],
                    "official_score": record["correct"], "complete_string_exact": strict,
                    "ended_eos": record["ended_eos"], "hit_generation_cap_without_eos": hit_cap,
                    "generated_ids": generated, "output_text": record["output_text"]}
            enriched.append(item); cells[(source["length_cap"], source["task"])].append(item)
        detail[arm] = {
            "rows": len(enriched),
            "by_cap_task": {f"{cap}/{task}": {
                "rows": len(items), "official_mean": sum(x["official_score"] for x in items) / len(items),
                "complete_string_exact_rate": sum(x["complete_string_exact"] for x in items) / len(items),
                "eos_rate": sum(x["ended_eos"] for x in items) / len(items),
                "cap_exhaustion_rate": sum(x["hit_generation_cap_without_eos"] for x in items) / len(items),
            } for (cap, task), items in sorted(cells.items())},
        }
        with args.out.with_name(f"{args.out.stem}_{arm}_rows.jsonl").open("w") as stream:
            for item in enriched:
                stream.write(json.dumps(item, sort_keys=True) + "\n")
    args.out.write_text(json.dumps({"status": "COMPLETE", "arms": detail, "tasks": list(TASKS)}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
