#!/usr/bin/env python3
"""Audit the complete 980-row x 2-arm output from ``e3_run`` and score E3."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
import re
from pathlib import Path

from experiments.rope_fast_5090_20260912.e3_validate import COUNTS, TASKS, load_rows
from scripts.experiments.olmo_fast_screen.ruler_bench import score as official_score

ARMS = ("C42", "C42V24")


def normalized_exact(text, reference):
    clean = lambda value: re.sub(r"\s+", " ", value.strip()).lower()
    return clean(text) == clean(reference)


def all_item_recall(row, text):
    cleaned = re.sub(r"[\x00-\x1f]", "\n", text.strip()).strip().lower()
    return sum(reference.lower() in cleaned for reference in row["references"]) / len(row["references"])


def percentile(values, probability):
    ordered = sorted(values); position = probability * (len(ordered) - 1)
    lower = int(position); upper = min(lower + 1, len(ordered) - 1); fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def stratified_bootstrap(differences, draws=20_000):
    if set(differences) != set(TASKS) or any(len(differences[task]) != COUNTS[16_384] for task in TASKS):
        raise ValueError("16K bootstrap cells are incomplete")
    rng = random.Random(2_026_091_203); values = []
    for _ in range(draws):
        task_means = []
        for task in TASKS:
            cell = differences[task]
            task_means.append(sum(cell[rng.randrange(len(cell))] for _ in cell) / len(cell))
        values.append(sum(task_means) / len(task_means))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="completed e3_run output directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepared_rows = load_rows(args.prepared / "screen.jsonl")
    if len(prepared_rows) != 980 or len({row["row_id"] for row in prepared_rows}) != 980:
        raise ValueError("frozen prepared panel is not exactly 980 unique rows")
    prepared = {row["row_id"]: row for row in prepared_rows}
    expected_counts = Counter({(cap, task): count for cap, count in COUNTS.items() for task in TASKS})
    generation = json.loads((args.prepared / "generation_config.json").read_text())
    eos_value = generation["eos_token_id"]
    eos_ids = set(eos_value if isinstance(eos_value, list) else [eos_value])
    arms = {}; detail = {}
    for arm in ARMS:
        values = load_rows(args.run / f"{arm}.jsonl"); ids = [row["row_id"] for row in values]
        if len(values) != 980 or len(set(ids)) != 980 or set(ids) != set(prepared):
            raise ValueError(f"{arm} is not one complete duplicate-free 980-row panel")
        counts = Counter((prepared[row_id]["length_cap"], prepared[row_id]["task"]) for row_id in ids)
        if counts != expected_counts:
            raise ValueError(f"{arm} task/cap grid is incomplete")
        by_id = {}; cells = defaultdict(list); enriched = []
        for record in values:
            source = prepared[record["row_id"]]
            for field in ("task", "length_cap", "references", "prompt_sha256", "max_new_tokens"):
                if field in record and record[field] != source[field]:
                    raise ValueError(f"{arm}/{record['row_id']} input identity drift: {field}")
            generated = record.get("generated_ids")
            if not isinstance(generated, list) or len(generated) > source["max_new_tokens"]:
                raise ValueError(f"{arm}/{record['row_id']} invalid generated token record")
            ended = bool(generated and generated[-1] in eos_ids)
            if record.get("ended_eos") != ended:
                raise ValueError(f"{arm}/{record['row_id']} EOS flag differs from frozen decoder")
            recomputed = official_score(source, record["output_text"])
            if abs(float(record["correct"]) - recomputed) > 1e-12:
                raise ValueError(f"{arm}/{record['row_id']} official score drift")
            refs = source["references"]
            literal = record["output_text"] == refs[0] if len(refs) == 1 else None
            item = {"row_id": record["row_id"], "task": source["task"], "length_cap": source["length_cap"],
                    "official_score": recomputed,
                    "normalized_any_reference_exact": any(normalized_exact(record["output_text"], ref) for ref in refs),
                    "single_reference_literal_exact": literal,
                    "multi_reference_canonical_complete_string_exact": literal if len(refs) == 1 else None,
                    "official_all_item_recall": all_item_recall(source, record["output_text"]),
                    "ended_eos": ended,
                    "hit_generation_cap_without_eos": len(generated) == source["max_new_tokens"] and not ended,
                    "generated_ids": generated, "output_text": record["output_text"]}
            enriched.append(item); by_id[item["row_id"]] = item; cells[(source["length_cap"], source["task"])].append(item)
        arms[arm] = by_id
        detail[arm] = {"rows": len(enriched), "by_cap_task": {f"{cap}/{task}": {
            "rows": len(items), "official_mean": sum(row["official_score"] for row in items) / len(items),
            "all_item_recall_mean": sum(row["official_all_item_recall"] for row in items) / len(items),
            "eos_rate": sum(row["ended_eos"] for row in items) / len(items),
            "cap_exhaustion_rate": sum(row["hit_generation_cap_without_eos"] for row in items) / len(items)}
            for (cap, task), items in sorted(cells.items())}}
        with args.out.with_name(f"{args.out.stem}_{arm}_rows.jsonl").open("x") as stream:
            for item in enriched: stream.write(json.dumps(item, sort_keys=True) + "\n")
    deltas = defaultdict(list); task_delta = {}
    for cap in COUNTS:
        for task in TASKS:
            row_ids = [key for key, row in prepared.items() if row["length_cap"] == cap and row["task"] == task]
            values = [arms["C42V24"][key]["official_score"] - arms["C42"][key]["official_score"] for key in row_ids]
            task_delta[cap, task] = sum(values) / len(values)
            if cap == 16_384: deltas[task] = values
    draws = stratified_bootstrap(deltas)
    primary = {"definition": "C42V24 minus C42 at 16K; seven tasks equally weighted",
        "estimate": sum(task_delta[16_384, task] for task in TASKS) / len(TASKS),
        "paired_task_stratified_bootstrap_ci95": [percentile(draws, .025), percentile(draws, .975)],
        "bootstrap_draws": len(draws), "by_task": {task: task_delta[16_384, task] for task in TASKS}}
    secondary = {"definition": "C42V24 minus C42 at 4K; seven tasks equally weighted",
        "estimate": sum(task_delta[4_096, task] for task in TASKS) / len(TASKS),
        "by_task": {task: task_delta[4_096, task] for task in TASKS}}
    args.out.write_text(json.dumps({"status": "COMPLETE", "rows_per_arm": 980, "arms": detail,
        "primary_16k": primary, "secondary_4k": secondary, "tasks": list(TASKS),
        "string_metric_note": "normalized-any-reference exact is diagnostic only; multi-reference canonical complete-string exact is null because the generator defines no unique legal order"}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
