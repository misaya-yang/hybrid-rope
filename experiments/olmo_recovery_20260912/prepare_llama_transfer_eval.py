#!/usr/bin/env python3
"""Build a small held-out Llama generation panel spanning short and long tasks."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


SHORT_CAPS = (512, 1024, 2048, 4096)
LONG_CAPS = (8192, 16384)
LONG_FAMILIES = ("longalign", "longalpaca", "longcite")


def stream(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def panel_row(row: dict, *, task: str, cap: int) -> dict:
    prompt = list(row.get("prompt_ids") or row["input_ids"][:int(row["target_start"])])
    answer_tokens = len(row["input_ids"]) - int(row["target_start"])
    references = row.get("references")
    if not prompt or answer_tokens <= 0 or not references or len(prompt) + answer_tokens > cap:
        raise ValueError("held-out row violates its prompt/answer/cap contract")
    return {
        "row_id": f"{task}:{row['id']}",
        "source_id": row.get("source_id"),
        "task": task,
        "family": "short_instruction" if task == "ultrachat_short" else "natural_long_generation",
        "length_cap": cap,
        "prompt_ids": prompt,
        "input_tokens": len(prompt),
        "references": references,
        "max_new_tokens": answer_tokens,
        "gold_answer_tokens_including_eos": answer_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--longcite", type=Path, required=True)
    parser.add_argument("--short-public", type=Path, required=True)
    parser.add_argument("--rows-per-cell", type=int, default=4)
    parser.add_argument("--long-cap", type=int, action="append", default=[])
    parser.add_argument("--long-family", choices=LONG_FAMILIES, action="append", default=[])
    parser.add_argument("--skip-short", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.rows_per_cell <= 0 or args.out.exists():
        raise ValueError("positive row count and a new output directory are required")
    long_caps = tuple(args.long_cap or LONG_CAPS)
    if len(set(long_caps)) != len(long_caps) or any(cap not in (8192, 16384, 32768, 65536) for cap in long_caps):
        raise ValueError("long caps must be unique members of 8K, 16K, 32K, and 64K")
    long_families = tuple(args.long_family or LONG_FAMILIES)
    if len(set(long_families)) != len(long_families):
        raise ValueError("long families must be unique")

    selected = []
    for cap in long_caps:
        public_rows = list(stream(args.public / f"long_sft_{cap}_dev.jsonl"))
        for family in long_families:
            candidates = (list(stream(args.longcite / f"longcite_{cap}_dev.jsonl"))
                          if family == "longcite"
                          else [row for row in public_rows if row.get("family") == family])
            if len(candidates) < args.rows_per_cell:
                raise ValueError(f"insufficient {family}/{cap} dev rows")
            selected.extend(panel_row(row, task=family, cap=cap) for row in candidates[:args.rows_per_cell])

    short_counts = Counter()
    if not args.skip_short:
        for row in stream(args.short_public / "native_short_sft_dev.jsonl"):
            prompt = row.get("prompt_ids") or row["input_ids"][:int(row["target_start"])]
            answer_tokens = len(row["input_ids"]) - int(row["target_start"])
            cap = next((value for value in SHORT_CAPS if len(prompt) + answer_tokens <= value), None)
            if cap is None or short_counts[cap] >= args.rows_per_cell:
                continue
            selected.append(panel_row(row, task="ultrachat_short", cap=cap))
            short_counts[cap] += 1
            if all(short_counts[value] == args.rows_per_cell for value in SHORT_CAPS):
                break
        if any(short_counts[value] != args.rows_per_cell for value in SHORT_CAPS):
            raise ValueError("insufficient short held-out rows across requested caps")
    ids = [row["row_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate held-out row identity")

    args.out.mkdir(parents=True)
    with (args.out / "screen.jsonl").open("x") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "status": "READY_GPU_NOT_RUN",
        "rows": len(selected),
        "rows_per_cell": args.rows_per_cell,
        "long_caps": list(long_caps),
        "includes_short": not args.skip_short,
        "counts": {f"{cap}/{task}": sum(r["length_cap"] == cap and r["task"] == task for r in selected)
                   for cap in (() if args.skip_short else SHORT_CAPS) + long_caps
                   for task in (("ultrachat_short",) if cap in SHORT_CAPS else long_families)},
        "generation_budget": "gold answer token count including terminal EOS; wrong/nonterminating outputs hit cap",
        "asset_identity_policy": "user_attested_clone/no_sha_validation",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
