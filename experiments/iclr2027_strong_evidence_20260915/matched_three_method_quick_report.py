#!/usr/bin/env python3
"""Build a strictly paired NIAH-8x5 and PPL-5 three-method report."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    weight = position - left
    return ordered[left] * (1 - weight) + ordered[right] * weight


def stratified_ci(rows: list[dict], left: str, right: str, seed: int) -> list[float]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    rng = random.Random(seed)
    draws = []
    for _ in range(10000):
        task_means = []
        for task in sorted(by_task):
            group = by_task[task]
            sample = [rng.choice(group) for _ in group]
            task_means.append(sum(item["scores"][left] - item["scores"][right] for item in sample) / len(sample))
        draws.append(sum(task_means) / len(task_means))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def document_ci(rows: list[dict], left: str, right: str, seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = []
    for _ in range(10000):
        sample = [rng.choice(rows) for _ in rows]
        draws.append(sum(item["mean_nll"][left] - item["mean_nll"][right] for item in sample) / len(sample))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict:
    arms = {label: (Path(generations), Path(lm_rows), Path(generation_contract), Path(lm_contract))
            for label, generations, lm_rows, generation_contract, lm_contract in args.arm}
    if set(arms) != {"tailspline", "mrpro", "yarn"}:
        raise ValueError("arms must be exactly tailspline, mrpro, and yarn")
    panel = read_jsonl(args.panel)
    selected = []
    counts: dict[str, int] = defaultdict(int)
    task_set = set(args.task)
    for row in panel:
        task = str(row.get("task"))
        if task not in task_set or int(row.get("length_cap", -1)) != args.target_length:
            continue
        if counts[task] < args.rows_per_task:
            selected.append(row)
            counts[task] += 1
    if counts != {task: args.rows_per_task for task in args.task}:
        raise ValueError(f"panel does not contain the frozen task cells: {dict(counts)}")
    expected_ids = [row["row_id"] for row in selected]
    expected = {row["row_id"]: row for row in selected}

    generation_maps = {}
    owners = {}
    for label, (generation_path, lm_path, generation_contract, lm_contract) in arms.items():
        for path in (generation_path, lm_path, generation_contract, lm_contract):
            if not path.is_file():
                raise FileNotFoundError(path)
        rows = read_jsonl(generation_path)
        mapping = {str(row.get("row_id")): row for row in rows if row.get("row_id") in expected}
        if list(mapping) != expected_ids:
            raise ValueError(f"{label}: generation rows are not the exact frozen ordered subset")
        for row_id, generated in mapping.items():
            source = expected[row_id]
            if generated.get("prompt_sha256") != source.get("prompt_sha256"):
                raise ValueError(f"{label}: prompt identity differs at {row_id}")
            score = generated.get("ruler_official_score")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise ValueError(f"{label}: invalid official score at {row_id}")
        generation_maps[label] = mapping
        owners[label] = {
            "generation_sha256": sha256(generation_path),
            "lm_rows_sha256": sha256(lm_path),
            "generation_contract_sha256": sha256(generation_contract),
            "lm_contract_sha256": sha256(lm_contract),
        }

    paired_rows = []
    for row_id in expected_ids:
        source = expected[row_id]
        paired_rows.append({
            "row_id": row_id,
            "task": source["task"],
            "prompt_sha256": source["prompt_sha256"],
            "scores": {label: float(generation_maps[label][row_id]["ruler_official_score"]) for label in arms},
        })
    by_task = {}
    for task in args.task:
        rows = [row for row in paired_rows if row["task"] == task]
        by_task[task] = {label: sum(row["scores"][label] for row in rows) / len(rows) for label in arms}
    niah_macro = {label: sum(by_task[task][label] for task in args.task) / len(args.task) for label in arms}

    lm_maps = {}
    for label, (_, lm_path, _, _) in arms.items():
        rows = [row for row in read_jsonl(lm_path) if int(row.get("length", -1)) == args.target_length]
        mapping = {(int(row["document"]), int(row["length"])): row for row in rows}
        expected_keys = [(index, args.target_length) for index in range(args.ppl_documents)]
        if any(key not in mapping for key in expected_keys):
            raise ValueError(f"{label}: missing frozen PPL document rows")
        lm_maps[label] = {key: mapping[key] for key in expected_keys}
    ppl_rows = []
    for key in [(index, args.target_length) for index in range(args.ppl_documents)]:
        values = {}
        for label in arms:
            row = lm_maps[label][key]
            count = int(row["whole_target_count"])
            loss = float(row["whole_loss_sum"])
            if count <= 0 or not math.isfinite(loss):
                raise ValueError(f"{label}: invalid LM row {key}")
            values[label] = loss / count
        ppl_rows.append({"document": key[0], "length": key[1], "mean_nll": values})
    ppl = {}
    for label in arms:
        loss = sum(float(lm_maps[label][key]["whole_loss_sum"]) for key in lm_maps[label])
        tokens = sum(int(lm_maps[label][key]["whole_target_count"]) for key in lm_maps[label])
        ppl[label] = {"nll": loss / tokens, "ppl": math.exp(loss / tokens), "target_tokens": tokens}

    comparisons = {}
    pairs = (("tailspline", "mrpro"), ("tailspline", "yarn"), ("mrpro", "yarn"))
    for index, (left, right) in enumerate(pairs):
        key = f"{left}_minus_{right}"
        comparisons[key] = {
            "niah_macro_delta": niah_macro[left] - niah_macro[right],
            "niah_task_stratified_bootstrap_ci95": stratified_ci(paired_rows, left, right, 20261600 + index),
            "mean_document_nll_delta": sum(row["mean_nll"][left] - row["mean_nll"][right] for row in ppl_rows) / len(ppl_rows),
            "document_bootstrap_ci95": document_ci(ppl_rows, left, right, 20261700 + index),
        }
    return {
        "status": "COMPLETE",
        "contract": "official-static-yarn-zero-training-quick-three-method-v1",
        "identity": {
            "condition": args.condition,
            "target_length": args.target_length,
            "tasks": list(args.task),
            "rows_per_task": args.rows_per_task,
            "paired_generation_rows": len(paired_rows),
            "ppl_documents": args.ppl_documents,
            "panel_sha256": sha256(args.panel),
            "ppl_manifest_sha256": sha256(args.ppl_manifest),
            "methods": ["TailSpline", "MrRoPE-Pro", "official static YaRN, zero-training installation"],
        },
        "owners": owners,
        "niah": {"task_equal_macro": niah_macro, "by_task": by_task, "paired_rows": paired_rows},
        "ppl": {"pooled": ppl, "paired_documents": ppl_rows},
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--target-length", type=int, required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--rows-per-task", type=int, default=5)
    parser.add_argument("--ppl-documents", type=int, default=5)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ppl-manifest", type=Path, required=True)
    parser.add_argument("--arm", nargs=5, action="append", metavar=("LABEL", "GENERATIONS", "LM_ROWS", "GENERATION_CONTRACT", "LM_CONTRACT"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build(args)
    atomic_json(args.out, report)
    print(json.dumps({"status": report["status"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
