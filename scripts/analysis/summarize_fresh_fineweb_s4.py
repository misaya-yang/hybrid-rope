#!/usr/bin/env python3
"""Summarize paired fresh-FineWeb Native/s4/YaRN/oracle NLL results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


STATUS = "FRESH_FINEWEB_ZERO_PARAMETER_S4_GENERALIZATION_COMPLETE_V1"
METHODS = ("native", "session_s4", "yarn4", "target_aware")
OPTIONAL_CONTROLS = (
    "session_geometric_s4",
    "session_ramp_s4",
    "static_s4",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".incomplete",
        mode="w", encoding="utf-8", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def bootstrap(values: np.ndarray, *, seed: int, samples: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_824)
    args = parser.parse_args()
    root = args.root.resolve(); source_manifest = args.source_manifest.resolve()
    methods = tuple(str(value) for value in args.methods)
    if "native" not in methods or "session_s4" not in methods:
        raise ValueError("summary requires native and session_s4")
    rows: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    receipts: dict[str, Any] = {}
    for method in methods:
        directory = root / method
        method_rows = load_jsonl(directory / "examples.jsonl")
        mapped = {(str(row["row_sha256"]), int(row["multiplier"])): row for row in method_rows}
        if len(mapped) != len(method_rows):
            raise RuntimeError(f"duplicate rows for {method}")
        rows[method] = mapped
        result_path = directory / "results.json"
        receipts[method] = {
            "results_sha256": sha256_file(result_path),
            "examples_sha256": sha256_file(directory / "examples.jsonl"),
            "run_manifest_sha256": sha256_file(directory / "run_manifest.json"),
            "method_receipt": json.loads(result_path.read_text())["method"],
        }
    expected_keys = set(rows["native"])
    if any(set(rows[method]) != expected_keys for method in methods[1:]):
        raise RuntimeError("paired fresh-FineWeb row identity drift")
    cells: dict[str, Any] = {}
    for multiplier in (1, 2, 4):
        keys = sorted(key for key in expected_keys if key[1] == multiplier)
        values = {
            method: np.asarray([float(rows[method][key]["nll"]) for key in keys])
            for method in methods
        }
        contrasts: dict[str, Any] = {}
        for other in ("native", "yarn4", "target_aware"):
            if other not in methods:
                continue
            delta = values["session_s4"] - values[other]
            contrasts[f"session_s4_minus_{other}"] = {
                "mean": float(delta.mean()),
                "median": float(np.median(delta)),
                "minimum": float(delta.min()),
                "maximum": float(delta.max()),
                "fraction_session_s4_better": float(np.mean(delta < 0.0)),
                "paired_row_bootstrap_95": bootstrap(
                    delta,
                    seed=int(args.seed) + multiplier,
                    samples=int(args.bootstrap_samples),
                ),
            }
        cells[str(multiplier)] = {
            "rows": len(keys),
            "target_tokens_per_row": int(rows["native"][keys[0]]["target_tokens"]),
            "mean_tail_nll": {method: float(value.mean()) for method, value in values.items()},
            "contrasts": contrasts,
            "native_session_exact_rows": int(np.sum(values["native"] == values["session_s4"])),
        }
    receipt = {
        "status": STATUS,
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_manifest": json.loads(source_manifest.read_text()),
        "methods": receipts,
        "cells": cells,
        "bootstrap": {
            "unit": "paired source document",
            "samples": int(args.bootstrap_samples),
            "seed": int(args.seed),
            "population_scope": "fixed OLMo checkpoint and deterministic fresh FineWeb-Edu shard002 rows",
        },
        "claim_boundary": (
            "Natural-text teacher-forced tail NLL on one new FineWeb-Edu shard; "
            "not capability, task-population, checkpoint-population, or training evidence."
        ),
    }
    control_rows: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    control_receipts: dict[str, Any] = {}
    for method in OPTIONAL_CONTROLS:
        directory = root / method
        if not (directory / "results.json").is_file():
            continue
        method_rows = load_jsonl(directory / "examples.jsonl")
        mapped = {(str(row["row_sha256"]), int(row["multiplier"])): row for row in method_rows}
        if set(mapped) != expected_keys:
            raise RuntimeError(f"optional control row identity drift: {method}")
        control_rows[method] = mapped
        control_receipts[method] = {
            "results_sha256": sha256_file(directory / "results.json"),
            "examples_sha256": sha256_file(directory / "examples.jsonl"),
            "run_manifest_sha256": sha256_file(directory / "run_manifest.json"),
        }
    if control_rows:
        control_cells: dict[str, Any] = {}
        for multiplier in (1, 2, 4):
            keys = sorted(key for key in expected_keys if key[1] == multiplier)
            session = np.asarray([float(rows["session_s4"][key]["nll"]) for key in keys])
            entries: dict[str, Any] = {}
            for method, mapped in control_rows.items():
                control = np.asarray([float(mapped[key]["nll"]) for key in keys])
                delta = control - session
                entries[f"{method}_minus_session_s4"] = {
                    "control_mean": float(control.mean()),
                    "session_s4_mean": float(session.mean()),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "maximum_delta": float(delta.max()),
                    "fraction_control_better": float(np.mean(delta < 0.0)),
                    "paired_row_bootstrap_95": bootstrap(
                        delta,
                        seed=int(args.seed) + 100 + multiplier,
                        samples=int(args.bootstrap_samples),
                    ),
                }
            control_cells[str(multiplier)] = entries
        receipt["optional_control_methods"] = control_receipts
        receipt["optional_control_cells"] = control_cells
    atomic_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
