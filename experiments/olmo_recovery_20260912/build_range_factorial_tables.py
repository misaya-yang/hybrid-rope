#!/usr/bin/env python3
"""Build the two missing shape x gain cells for a frozen range-solver table."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _validated_table(value: dict, label: str) -> dict:
    table = value.get("table", value)
    frequencies = np.asarray(table.get("values_float32"), dtype=np.float32)
    gain = float(table.get("gain"))
    if (
        frequencies.ndim != 1
        or frequencies.size == 0
        or not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0.0)
        or np.any(frequencies[:-1] <= frequencies[1:])
        or not math.isfinite(gain)
        or gain <= 0.0
    ):
        raise ValueError(f"invalid table: {label}")
    return {
        "values_float32": frequencies.tolist(),
        "gain": gain,
        "construction": dict(table.get("construction", {})),
    }


def build_factorial_tables(base: dict, solver: dict) -> dict[str, dict]:
    """Return only the two missing cells; the base and solver cells are reused."""
    base_table = _validated_table(base, "base")
    solver_table = _validated_table(solver, "solver")
    if len(base_table["values_float32"]) != len(solver_table["values_float32"]):
        raise ValueError("base and solver table dimensions differ")
    shared = {
        "method": "range_solver_shape_gain_factorial_v1",
        "base_gain": base_table["gain"],
        "solver_gain": solver_table["gain"],
        "same_table_all_layers_and_lengths": True,
        "model_weight_updates": 0,
    }
    return {
        "BaseShape_SolverGain": {
            "values_float32": list(base_table["values_float32"]),
            "gain": solver_table["gain"],
            "construction": {
                **shared,
                "shape_source": "base",
                "gain_source": "solver",
            },
        },
        "SolverShape_BaseGain": {
            "values_float32": list(solver_table["values_float32"]),
            "gain": base_table["gain"],
            "construction": {
                **shared,
                "shape_source": "solver",
                "gain_source": "base",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-arm", default="C42V24_g4")
    parser.add_argument("--solver-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoConfig
    from .recovery_v2_runtime import table_for_config

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    base = table_for_config(config, args.base_arm)
    solver = json.loads(args.solver_result.read_text())
    tables = build_factorial_tables(base, solver)
    args.out.mkdir(parents=True, exist_ok=False)
    for label, table in tables.items():
        (args.out / f"{label}.json").write_text(
            json.dumps({"status": "FROZEN", "label": label, "table": table}, indent=2, sort_keys=True) + "\n"
        )
    manifest = {
        "status": "FROZEN",
        "scope": "two missing cells of the C42V24 x SolverC42 shape-gain factorial",
        "base_arm": args.base_arm,
        "solver_result": str(args.solver_result),
        "cells": sorted(tables),
        "reused_cells": ["BaseShape_BaseGain", "SolverShape_SolverGain"],
        "decision_policy": "report task-level redistribution; no single-threshold pruning",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
