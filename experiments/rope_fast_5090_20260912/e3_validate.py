"""Validate E3 panel identity, completeness, and separation from development data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from experiments.rope_fast_5090_20260912.e3_tables import tables


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


def instance_key(row: dict[str, Any]) -> str:
    if row["task"].startswith("qa_") and row.get("qa_source_index") is not None:
        return digest({"task": row["task"], "qa_source_index": row["qa_source_index"]})
    return digest({"task": row["task"], "prompt_sha256": row["prompt_sha256"]})


def validate(prepared: Path, development_screen: Path, prior_registry: Path | None = None) -> dict[str, Any]:
    old_hashes = set()
    dev_ids = []
    prior_instances = set()
    development_manifest = json.loads((development_screen.parent / "manifest.json").read_text())
    for index, row in enumerate(iter_rows(development_screen)):
        if row["task"].startswith("qa_"):
            row["qa_source_index"] = int(row["upstream_index"]) + int(development_manifest["qa_offset"])
        prior_instances.add(instance_key(row))
        old_hashes.add(row["prompt_sha256"])
        dev_ids.append(row["row_id"])
    if prior_registry is not None:
        for row in iter_rows(prior_registry):
            prior_instances.add(instance_key(row))

    counts = Counter()
    seen = set()
    row_count = 0
    total_tokens = 0
    minimum = None
    maximum = 0
    new_instances = set()
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
        if row["input_tokens"] != len(row["prompt_ids"]):
            raise ValueError(f"actual token count mismatch: {row['row_id']}")
        if row["input_tokens"] + row["max_new_tokens"] > row["length_cap"]:
            raise ValueError(f"generation reserve exceeds cap: {row['row_id']}")
        if not row["references"]:
            raise ValueError(f"missing references: {row['row_id']}")
        tokens = row["input_tokens"]
        total_tokens += tokens
        minimum = tokens if minimum is None else min(minimum, tokens)
        maximum = max(maximum, tokens)

    expected = Counter({(cap, task): count for cap, count in COUNTS.items() for task in TASKS})
    if row_count != 980 or len(seen) != 980 or counts != expected:
        raise ValueError(f"incomplete 980-row task/cap grid: {counts}")

    table_payload = json.loads((prepared / "tables.json").read_text())
    expected_tables = tables()
    if set(table_payload) != set(expected_tables):
        raise ValueError("deployed E3 arm set differs")
    for name in table_payload:
        actual = np.asarray(table_payload[name]["values_float32"], dtype=np.float32)
        expected_values = np.asarray(expected_tables[name]["values_float32"], dtype=np.float32)
        if actual.shape != expected_values.shape or not np.array_equal(actual, expected_values):
            raise ValueError(f"deployed E3 table values differ: {name}")
        if not np.isfinite(actual).all() or not np.all(actual > 0) or not np.all(actual[:-1] > actual[1:]):
            raise ValueError(f"invalid E3 table values: {name}")
        if float(table_payload[name]["gain"]) != float(expected_tables[name]["gain"]):
            raise ValueError(f"deployed E3 gain differs: {name}")
    report = {
        "status": "PASS", "rows": row_count, "development_rows_checked": len(dev_ids),
        "counts_by_cap_task": {f"{cap}/{task}": counts[(cap, task)] for cap in COUNTS for task in TASKS},
        "actual_input_tokens": total_tokens,
        "min_input_tokens": minimum, "max_input_tokens": maximum,
        "development_prompt_overlap": 0,
        "overlap_scope": "Exact stored prompt IDs and QA source identities are checked; repeated RULER background blocks are not rescanned.",
        "prior_instance_registry_rows": len(prior_instances), "prior_instance_overlap": 0,
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
