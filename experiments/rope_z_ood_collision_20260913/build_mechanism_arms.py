#!/usr/bin/env python3
"""Build audited CPU mechanism arms; default invocation is plan-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .mechanisms import DELTA_GRID, audit_mechanism_deltas


def _read_json_table(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        values = payload
    else:
        table = payload.get("table", payload)
        values = table.get("values_float32", table.get("frequencies"))
    if values is None:
        raise ValueError("JSON does not contain table.values_float32 or frequencies")
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--geo-json", type=Path, help="existing Geo tensor receipt")
    source.add_argument("--base", type=float, help="construct the standard k/K table for a declared pair count")
    parser.add_argument("--pairs", type=int, help="required with --base")
    parser.add_argument("--train-length", type=int, required=True)
    parser.add_argument("--target-factor", type=int, action="append", default=[])
    parser.add_argument("--delta", type=float, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    factors = tuple(args.target_factor or (2, 4, 8))
    deltas = tuple(args.delta or DELTA_GRID)
    plan = {
        "status": "PLAN_ONLY",
        "source": str(args.geo_json) if args.geo_json else {"base": args.base, "pairs": args.pairs},
        "train_length": args.train_length,
        "target_factors": list(factors),
        "delta_grid": list(deltas),
        "out": str(args.out),
        "scope": "CPU operator proxies only; no model task conclusion and no GPU action",
    }
    if not args.execute:
        print(json.dumps(plan, sort_keys=True))
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.geo_json:
        frequencies = _read_json_table(args.geo_json)
    else:
        if args.base is None or args.base <= 1.0 or args.pairs is None or args.pairs < 3:
            raise ValueError("--base requires --pairs >= 3 and base > 1")
        frequencies = np.power(args.base, -np.arange(args.pairs, dtype=np.float64) / args.pairs)
    result = audit_mechanism_deltas(frequencies, args.train_length, factors, deltas)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "receipt_kind": result["receipt_kind"],
        "strict_deltas": result.get("strict_deltas", []),
        "pareto_deltas": result["pareto_deltas"],
        "selected_delta": result["selected_delta"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
