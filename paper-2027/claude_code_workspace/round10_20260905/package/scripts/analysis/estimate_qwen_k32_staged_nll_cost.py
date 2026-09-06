#!/usr/bin/env python3
"""Estimate the staged packed-NLL bill from one sealed timing canary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

STATUS = "QWEN_K32_PACKED_NATURAL_NLL_CANARY_COMPLETE"


def estimate(result: dict, *, rmb_per_hour: float, quantum_seconds: int,
             budget_rmb: float, buffer_fraction: float) -> dict:
    if (
        result.get("status") != STATUS
        or result.get("metrics_exposed") is not False
        or result.get("rows") != 2
        or set(result.get("batch_seconds", {})) != {"32768", "65536"}
    ):
        raise ValueError("invalid or metric-bearing canary result")
    t32 = float(result["batch_seconds"]["32768"])
    t64 = float(result["batch_seconds"]["65536"])
    elapsed = float(result["elapsed_seconds"])
    process_elapsed = float(result["process_elapsed_seconds"])
    if (
        not all(math.isfinite(value) and value > 0 for value in
                (t32, t64, elapsed, process_elapsed, rmb_per_hour))
        or process_elapsed < elapsed
        or quantum_seconds <= 0
        or budget_rmb <= 0
        or not 0 <= buffer_fraction < 1
    ):
        raise ValueError("invalid timing, billing, budget, or buffer")
    process_overhead = process_elapsed - elapsed
    primary_seconds = 64 * (t32 + t64) + process_overhead
    baseline_seconds = 32 * (t32 + t64) + process_overhead
    canary_seconds = process_elapsed
    worst_seconds = canary_seconds + primary_seconds + baseline_seconds
    buffered_seconds = worst_seconds * (1 + buffer_fraction)
    billed_seconds = math.ceil(buffered_seconds / quantum_seconds) * quantum_seconds
    cost = billed_seconds * rmb_per_hour / 3600
    return {
        "status": "STAGED_NLL_COST_ADMITTED" if cost <= budget_rmb else "STAGED_NLL_COST_REJECTED",
        "canary_seconds": canary_seconds,
        "estimated_primary_seconds": primary_seconds,
        "estimated_baseline_seconds": baseline_seconds,
        "estimated_worst_seconds": worst_seconds,
        "buffer_fraction": buffer_fraction,
        "buffered_seconds": buffered_seconds,
        "billing_quantum_seconds": quantum_seconds,
        "billed_seconds": billed_seconds,
        "rmb_per_hour": rmb_per_hour,
        "estimated_cost_rmb": cost,
        "budget_rmb": budget_rmb,
        "decision": "RUN" if cost <= budget_rmb else "DO_NOT_RUN",
        "assumption": (
            "One canary, one two-arm primary process, and conditional one-arm baseline process; "
            "each scientific arm has 32 streams at both lengths."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-result", type=Path, required=True)
    parser.add_argument("--rmb-per-hour", type=float, required=True)
    parser.add_argument("--billing-quantum-seconds", type=int, required=True)
    parser.add_argument("--budget-rmb", type=float, default=8.0)
    parser.add_argument("--buffer-fraction", type=float, default=.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = estimate(json.loads(args.canary_result.read_text()),
                          rmb_per_hour=args.rmb_per_hour,
                          quantum_seconds=args.billing_quantum_seconds,
                          budget_rmb=args.budget_rmb,
                          buffer_fraction=args.buffer_fraction)
    except (ValueError, TypeError, OSError, json.JSONDecodeError):
        report = {"status": "INVALID_CANARY_COST_INPUT", "decision": "DO_NOT_RUN"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["status"])
    return 0 if report["decision"] == "RUN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
