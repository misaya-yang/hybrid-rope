#!/usr/bin/env python3
"""Rank frozen tables on a locked finite checkpoint-Q/K replay diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.fixed_rope_three_interfaces_20260913 import TABLE_FORMAT
from experiments.fixed_rope_three_interfaces_20260913.tables import (
    exponents_from_table, find_table, runtime_native_inv_freq, tensor_sha256, validate_table,
)
from .capture_io import file_sha256, load_capture
from .core import evaluate_replay_objective, finite_rho_grid


def labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def table_increments(receipt: dict, pairs: int) -> tuple[np.ndarray, float]:
    exponents = np.asarray(receipt.get("exponents"), dtype=np.float64)
    if exponents.shape != (pairs,):
        construction = find_table(receipt).get("construction", {})
        exponents = np.asarray(
            construction.get("exponents") or construction.get("cumulative_exponents"),
            dtype=np.float64,
        )
    if (
        exponents.shape != (pairs,) or not np.isfinite(exponents).all()
        or np.any(np.diff(exponents) < -1e-9) or abs(exponents[0]) > 1e-9
        or exponents[-1] > 1.0 + 1e-9
    ):
        raise ValueError("table receipt lacks a valid Native-relative exponent vector")
    table = find_table(receipt)
    return np.diff(np.clip(exponents, 0.0, 1.0)), float(table["gain"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--table", type=labeled_path, action="append", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--points-per-doubling", type=int, default=2)
    parser.add_argument("--native-mean-limit", type=float)
    parser.add_argument("--native-group-cvar-limit", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    capture_index = read_json(args.capture_index)
    if capture_index.get("status") != "CHECKPOINT_QK_CAPTURE_COMPLETE_V1":
        raise ValueError("capture index is incomplete")
    capture_contract = Path(capture_index.get("contract_path", ""))
    if (
        not capture_contract.is_file()
        or capture_index.get("contract_sha256") != file_sha256(capture_contract)
    ):
        raise ValueError("capture contract changed after index freeze")
    for record in capture_index["captures"]:
        receipt_path = Path(record["path"]) / "receipt.json"
        if record.get("receipt_sha256") != file_sha256(receipt_path):
            raise ValueError("capture receipt changed after index freeze")
    captures = [load_capture(Path(record["path"]), mmap_mode="r") for record in capture_index["captures"]]
    if not captures:
        raise ValueError("capture index has no captures")
    pairs = len(captures[0].native_inv_freq)
    native = np.asarray(captures[0].native_inv_freq, dtype=np.float32)
    if any(not np.array_equal(np.asarray(capture.native_inv_freq, dtype=np.float32), native) for capture in captures[1:]):
        raise ValueError("capture set mixes different Native frequency tables")
    rhos = finite_rho_grid(args.scale, points_per_doubling=args.points_per_doubling)
    results = {}
    deployments = set()
    for label, path in args.table:
        if label in results:
            raise ValueError(f"duplicate table label {label}")
        receipt = read_json(path)
        if receipt.get("status") != TABLE_FORMAT or float(receipt.get("scale", float("nan"))) != args.scale:
            raise ValueError(f"table {label} has another receipt format or deployment scale")
        if receipt.get("model_id") != capture_index.get("model_id"):
            raise ValueError(f"table {label} belongs to another checkpoint identity")
        geometry = receipt.get("model_geometry") or {}
        if geometry != capture_index.get("model_geometry"):
            raise ValueError(f"table {label} has another checkpoint geometry")
        expected_native = runtime_native_inv_freq(geometry)
        if not np.array_equal(expected_native, native):
            raise ValueError(f"table {label} and capture use different Native frequencies")
        if receipt.get("exponent_reference_sha256_float32") != tensor_sha256(expected_native):
            raise ValueError(f"table {label} exponent reference differs from runtime Native")
        table = find_table(receipt)
        values, nested_gain = validate_table(table, pairs=pairs)
        if tensor_sha256(values) != receipt.get("table_sha256_float32"):
            raise ValueError(f"table {label} frequency hash is inconsistent")
        if float(nested_gain) != float(receipt.get("gain", float("nan"))):
            raise ValueError(f"table {label} gain is inconsistent")
        increments, gain = table_increments(receipt, pairs)
        expected_exponents = exponents_from_table(
            values, expected_native.astype(np.float64), args.scale,
        )
        if not np.array_equal(expected_exponents, np.asarray(receipt["exponents"], dtype=np.float64)):
            raise ValueError(f"table {label} exponent analysis coordinates are inconsistent")
        reconstructed = (
            runtime_native_inv_freq(geometry).astype(np.float64)
            * np.power(args.scale, -np.asarray(receipt["exponents"]))
        ).astype(np.float32)
        residual_slots = np.flatnonzero(reconstructed != values).astype(int).tolist()
        if residual_slots != receipt.get("exponent_reconstruction", {}).get("residual_slots"):
            raise ValueError(f"table {label} exponent reconstruction residual is inconsistent")
        deployment = (tensor_sha256(values), float(gain).hex())
        if deployment in deployments:
            raise ValueError(f"table {label} duplicates another frequency+gain deployment")
        deployments.add(deployment)
        result = evaluate_replay_objective(
            captures, increments, scale=args.scale, gain=gain, rhos=rhos,
            native_mean_limit=args.native_mean_limit,
            native_group_cvar_limit=args.native_group_cvar_limit,
            active_inv_freq=values,
        )
        results[label] = {
            "table_path": str(path.resolve()),
            "table_sha256_float32": receipt.get("table_sha256_float32"),
            "finite_grid_worst_mean_kl": result["finite_grid_worst_mean_kl"],
            "finite_grid_worst_group_mean_kl": result["finite_grid_worst_group_mean_kl"],
            "native_constraints": result["native_constraints"],
            "result": result,
        }
    ranked = sorted(
        results,
        key=lambda label: (
            not results[label]["native_constraints"]["holds"],
            results[label]["finite_grid_worst_group_mean_kl"],
            results[label]["finite_grid_worst_mean_kl"],
        ),
    )
    feasible_rank = [label for label in ranked if results[label]["native_constraints"]["holds"]]
    output = {
        "status": "CHECKPOINT_REPLAY_TABLE_AUDIT_COMPLETE_V1",
        "capture_index": str(args.capture_index.resolve()),
        "capture_index_sha256": file_sha256(args.capture_index),
        "scale": args.scale,
        "rhos": rhos.tolist(),
        "results": results,
        "proxy_rank": ranked,
        "native_feasible_proxy_rank": feasible_rank,
        "interpretation": (
            "retrospective detached-Q/K transport audit only; proxy rank cannot filter "
            "or overwrite official generation results"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "proxy_rank": ranked}, sort_keys=True))


if __name__ == "__main__":
    main()
