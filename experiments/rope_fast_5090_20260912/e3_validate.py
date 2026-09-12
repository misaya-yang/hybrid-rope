"""Validate E3 panel identity, completeness, and separation from development data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.rope_fast_5090_20260912.e3_tables import tables, tensor_sha


TASKS = ("niah_single_1", "niah_single_3", "niah_multikey_1", "niah_multikey_3", "niah_multivalue", "cwe", "qa_2")
COUNTS = {4096: 40, 16384: 100}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def iter_rows(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_rows(path))


def block_hashes(ids: list[int], *, width: int = 64) -> set[str]:
    # Ignore the first/last block, which contain task/chat templates shared by design.
    return {digest(ids[start:start + width]) for start in range(width, max(width, len(ids) - width), width)}


def instance_key(row: dict[str, Any]) -> str:
    if row["task"].startswith("qa_") and row.get("qa_source_index") is not None:
        return digest({"task": row["task"], "qa_source_index": row["qa_source_index"]})
    return digest({"task": row["task"], "prompt_sha256": row["prompt_sha256"]})


def validate(prepared: Path, development_screen: Path, prior_registry: Path | None = None) -> dict[str, Any]:
    old_hashes = set()
    inverted: dict[str, set[int]] = defaultdict(set)
    dev_blocks = []
    dev_ids = []
    for index, row in enumerate(iter_rows(development_screen)):
        old_hashes.add(row["prompt_sha256"])
        dev_ids.append(row["row_id"])
        blocks = block_hashes(row["prompt_ids"])
        dev_blocks.append(blocks)
        for value in blocks:
            inverted[value].add(index)
    prior_instances = set()
    if prior_registry is not None:
        for row in iter_rows(prior_registry):
            prior_instances.add(instance_key(row))

    counts = Counter()
    seen = set()
    row_count = 0
    total_tokens = 0
    minimum = None
    maximum = 0
    collection = hashlib.sha256()
    new_instances = set()
    worst = {"containment": 0.0, "new_row": None, "development_row": None}
    for row in iter_rows(prepared / "screen.jsonl"):
        row_count += 1
        if row["row_id"] in seen:
            raise ValueError(f"duplicate row identity: {row['row_id']}")
        seen.add(row["row_id"])
        counts[(row["length_cap"], row["task"])] += 1
        if row["prompt_sha256"] in old_hashes:
            raise ValueError("new panel exactly reuses a development prompt")
        identity = instance_key(row)
        if identity in prior_instances:
            raise ValueError(f"new panel reuses a prior task/reference instance: {row['row_id']}")
        if identity in new_instances:
            raise ValueError(f"new panel repeats one task/reference instance across cells: {row['row_id']}")
        new_instances.add(identity)
        if digest(row["prompt_ids"]) != row["prompt_sha256"]:
            raise ValueError(f"prompt token identity mismatch: {row['row_id']}")
        if row["input_tokens"] != len(row["prompt_ids"]):
            raise ValueError(f"actual token count mismatch: {row['row_id']}")
        if row["input_tokens"] + row["max_new_tokens"] > row["length_cap"]:
            raise ValueError(f"generation reserve exceeds cap: {row['row_id']}")
        if not row["references"]:
            raise ValueError(f"missing references: {row['row_id']}")
        blocks = block_hashes(row["prompt_ids"])
        candidates: Counter[int] = Counter()
        for value in blocks:
            candidates.update(inverted.get(value, ()))
        if candidates:
            index, overlap = candidates.most_common(1)[0]
            containment = overlap / max(1, min(len(blocks), len(dev_blocks[index])))
            if containment > worst["containment"]:
                worst = {"containment": containment, "new_row": row["row_id"], "development_row": dev_ids[index]}
            if containment >= 0.90:
                raise ValueError(f"near-duplicate prompt detected: {worst}")
        tokens = row["input_tokens"]
        total_tokens += tokens
        minimum = tokens if minimum is None else min(minimum, tokens)
        maximum = max(maximum, tokens)
        collection.update(bytes.fromhex(row["prompt_sha256"]))

    expected = Counter({(cap, task): count for cap, count in COUNTS.items() for task in TASKS})
    if row_count != 980 or len(seen) != 980 or counts != expected:
        raise ValueError(f"incomplete 980-row task/cap grid: {counts}")

    table_payload = json.loads((prepared / "tables.json").read_text())
    if table_payload != tables():
        raise ValueError("deployed E3 tables differ from exact reconstruction")
    report = {
        "status": "PASS", "rows": row_count, "development_rows_checked": len(dev_ids),
        "counts_by_cap_task": {f"{cap}/{task}": counts[(cap, task)] for cap in COUNTS for task in TASKS},
        "actual_input_tokens": total_tokens,
        "min_input_tokens": minimum, "max_input_tokens": maximum,
        "prompt_collection_sha256": collection.hexdigest(),
        "development_prompt_overlap": 0, "worst_block_containment": worst,
        "prior_instance_registry_rows": len(prior_instances), "prior_instance_overlap": 0,
        "table_sha256": {name: value["tensor_sha256"] for name, value in table_payload.items()},
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--development-screen", type=Path, required=True)
    parser.add_argument("--prior-registry", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.prepared.resolve(), args.development_screen.resolve(),
        None if args.prior_registry is None else args.prior_registry.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
