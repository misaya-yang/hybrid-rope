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
EXPECTED_METHOD = {
    "native": "native",
    "session_s4": "session_binary_s4",
    "yarn4": "official_yarn",
    "target_aware": "target_aware",
}
CONTROL_PROTOCOL = {
    "session_geometric_s4": ("external_table_session", "same_support_geometric_s4"),
    "session_ramp_s4": ("external_table_session", "nearest_yarn_ramp_s4"),
    "static_s4": ("external_table_static", "derived_s4_static_all_lengths"),
}
NATIVE_TABLE_SHA256 = "dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34"
S2_TABLE_SHA256 = "f94a34381cfb3d05621db41bd3778779b16812246c392bd645013a1a06f80814"
S4_TABLE_SHA256 = "a435d75441444bcea39b73d9cf530005249dc5afdc3cfb5a60fda10ef33312d3"
YARN4_TABLE_SHA256 = "cc9da456982ffce5ca0558e9ea661abc4a880ec002179ce6b9149d45aa4a016c"


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


def validate_run(
    directory: Path, *, expected_method: str, source_manifest_sha256: str,
    common: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = directory / "run_manifest.json"
    result_path = directory / "results.json"
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())
    required = {
        "method": expected_method,
        "tasks": ["pg19"],
        "multipliers": [1, 2, 4],
        "token_manifest_sha256": source_manifest_sha256,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"protocol identity drift for {directory.name}:{key}")
    if result.get("protocol") != manifest:
        raise RuntimeError(f"results/manifest protocol drift for {directory.name}")
    if result.get("run_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError(f"embedded manifest hash drift for {directory.name}")
    if (
        expected_method != "official_yarn"
        and result.get("method", {}).get("method") != expected_method
    ):
        raise RuntimeError(f"method receipt drift for {directory.name}")
    matched = {
        key: manifest.get(key)
        for key in (
            "checkpoint_sha256", "checkpoint_ready_receipt_sha256",
            "script_sha256", "limit_per_cell", "tasks", "multipliers",
            "token_manifest_sha256",
        )
    }
    if common is not None and matched != common:
        raise RuntimeError(f"matched protocol fields drift for {directory.name}")
    return manifest, matched


def validate_method_profiles(
    method: str, method_rows: list[dict[str, Any]],
    method_receipt: dict[str, Any], run_manifest: dict[str, Any],
) -> None:
    if method == "native":
        if any(row.get("active_profile") is not None for row in method_rows):
            raise RuntimeError("Native run unexpectedly carries an active RoPE profile")
        return
    if method == "yarn4":
        expected = {
            "active_frequency": "official_transformers_yarn",
            "active_sha256_float32": YARN4_TABLE_SHA256,
            "independent_expected_sha256_float32": YARN4_TABLE_SHA256,
            "attention_scaling": 1.138629436111989,
        }
        for key, value in expected.items():
            if method_receipt.get(key) != value:
                raise RuntimeError(f"official YaRN receipt drift for {key}")
        if any(row.get("active_profile") is not None for row in method_rows):
            raise RuntimeError("official YaRN should be bound by its model-level receipt")
        return

    for row in method_rows:
        multiplier = int(row["multiplier"])
        profile = row.get("active_profile")
        if not isinstance(profile, dict):
            raise RuntimeError(f"missing active profile for {method} x{multiplier}")
        if method == "session_s4":
            expected = (
                ("native", 1, NATIVE_TABLE_SHA256, 1.0)
                if multiplier == 1
                else ("budgeted_s4_p2", 4, S4_TABLE_SHA256, 1.138629436111989)
            )
        elif method == "target_aware":
            expected = {
                1: ("native", 1, NATIVE_TABLE_SHA256, 1.0),
                2: ("budgeted_s2_p2", 2, S2_TABLE_SHA256, 1.0693147180559945),
                4: ("budgeted_s4_p2", 4, S4_TABLE_SHA256, 1.138629436111989),
            }[multiplier]
        else:
            expected = None
        if expected is not None:
            observed = (
                profile.get("identity"), profile.get("multiplier"),
                profile.get("table_sha256_float32"), profile.get("attention_scaling"),
            )
            if observed != expected:
                raise RuntimeError(f"active profile identity drift for {method} x{multiplier}")
    if method in {"session_s4", "target_aware"}:
        if method_receipt.get("native_inv_freq_sha256") != NATIVE_TABLE_SHA256:
            raise RuntimeError(f"native table receipt drift for {method}")


def validate_control_profiles(
    method: str, method_rows: list[dict[str, Any]],
    method_receipt: dict[str, Any], run_manifest: dict[str, Any],
) -> None:
    expected_table = str(run_manifest["table_sha256_float32"])
    expected_identity = str(run_manifest["table_name"])
    expected_receipt = {
        "active_sha256_float32": expected_table,
        "table_name": expected_identity,
        "table_support": "native_div_factor",
        "long_attention_scaling": 1.138629436111989,
    }
    for key, value in expected_receipt.items():
        if method_receipt.get(key) != value:
            raise RuntimeError(f"control method receipt drift for {method}:{key}")
    for row in method_rows:
        multiplier = int(row["multiplier"])
        profile = row.get("active_profile")
        if not isinstance(profile, dict):
            raise RuntimeError(f"missing control active profile for {method} x{multiplier}")
        use_native = method != "static_s4" and multiplier == 1
        observed = (
            profile.get("identity"), profile.get("table_sha256_float32"),
            profile.get("attention_scaling"),
        )
        expected = (
            ("native", NATIVE_TABLE_SHA256, 1.0)
            if use_native
            else (expected_identity, expected_table, 1.138629436111989)
        )
        if observed != expected:
            raise RuntimeError(f"control active profile drift for {method} x{multiplier}")


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
    source_manifest_sha = sha256_file(source_manifest)
    methods = tuple(str(value) for value in args.methods)
    if "native" not in methods or "session_s4" not in methods:
        raise ValueError("summary requires native and session_s4")
    rows: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    receipts: dict[str, Any] = {}
    common_protocol: dict[str, Any] | None = None
    for method in methods:
        directory = root / method
        run_manifest, matched = validate_run(
            directory, expected_method=EXPECTED_METHOD[method],
            source_manifest_sha256=source_manifest_sha, common=common_protocol,
        )
        if common_protocol is None:
            common_protocol = matched
        method_rows = load_jsonl(directory / "examples.jsonl")
        mapped = {(str(row["row_sha256"]), int(row["multiplier"])): row for row in method_rows}
        if len(mapped) != len(method_rows):
            raise RuntimeError(f"duplicate rows for {method}")
        rows[method] = mapped
        result_path = directory / "results.json"
        result_receipt = json.loads(result_path.read_text())
        validate_method_profiles(
            method, method_rows, result_receipt["method"], run_manifest,
        )
        receipts[method] = {
            "results_sha256": sha256_file(result_path),
            "examples_sha256": sha256_file(directory / "examples.jsonl"),
            "run_manifest_sha256": sha256_file(directory / "run_manifest.json"),
            "method_receipt": result_receipt["method"],
            "validated_protocol": run_manifest,
        }
    expected_keys = set(rows["native"])
    if any(set(rows[method]) != expected_keys for method in methods[1:]):
        raise RuntimeError("paired fresh-FineWeb row identity drift")
    expected_rows = int(common_protocol["limit_per_cell"]) if common_protocol else 0
    for multiplier in (1, 2, 4):
        if sum(key[1] == multiplier for key in expected_keys) != expected_rows:
            raise RuntimeError(f"limit/row-count drift for multiplier {multiplier}")
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
        "source_manifest_sha256": source_manifest_sha,
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
        expected_method, expected_table = CONTROL_PROTOCOL[method]
        run_manifest, _ = validate_run(
            directory, expected_method=expected_method,
            source_manifest_sha256=source_manifest_sha, common=common_protocol,
        )
        expected_control = {
            "table_name": expected_table,
            "table_support": "native_div_factor",
            "factor": 4.0,
            "long_attention_scaling": 1.138629436111989,
        }
        for key, value in expected_control.items():
            if run_manifest.get(key) != value:
                raise RuntimeError(f"control identity drift for {method}:{key}")
        if not run_manifest.get("table_file_sha256") or not run_manifest.get("table_sha256_float32"):
            raise RuntimeError(f"control table receipt missing for {method}")
        mapped = {(str(row["row_sha256"]), int(row["multiplier"])): row for row in method_rows}
        if len(mapped) != len(method_rows):
            raise RuntimeError(f"duplicate optional-control rows: {method}")
        if set(mapped) != expected_keys:
            raise RuntimeError(f"optional control row identity drift: {method}")
        result_receipt = json.loads((directory / "results.json").read_text())
        validate_control_profiles(
            method, method_rows, result_receipt["method"], run_manifest,
        )
        control_rows[method] = mapped
        control_receipts[method] = {
            "results_sha256": sha256_file(directory / "results.json"),
            "examples_sha256": sha256_file(directory / "examples.jsonl"),
            "run_manifest_sha256": sha256_file(directory / "run_manifest.json"),
            "validated_protocol": run_manifest,
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
