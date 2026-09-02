#!/usr/bin/env python3
"""Admit the far-evidence QA follow-up only when NLL plus QA stays within ten RMB."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

STATUS = "QWEN_K32_FAR_EVIDENCE_QA_CANARY_COMPLETE"


def estimate(canary: dict, nll_cost: dict, *, rmb_per_hour: float,
             quantum_seconds: int, total_budget_rmb: float = 10.0,
             buffer_fraction: float = .2) -> dict:
    if (
        canary.get("status") != STATUS or canary.get("metrics_exposed") is not False
        or canary.get("rows") != 2
        or set(canary.get("batch_seconds", {})) != {"32", "128"}
        or nll_cost.get("status") != "STAGED_NLL_COST_ADMITTED"
        or nll_cost.get("decision") != "RUN"
    ):
        raise ValueError("invalid canary or NLL cost receipt")
    t32, t128 = (float(canary["batch_seconds"][key]) for key in ("32", "128"))
    elapsed, process = float(canary["elapsed_seconds"]), float(canary["process_elapsed_seconds"])
    nll_rmb = float(nll_cost["estimated_cost_rmb"])
    values = (t32, t128, elapsed, process, nll_rmb, rmb_per_hour, total_budget_rmb)
    if (
        not all(math.isfinite(value) and value > 0 for value in values)
        or process < elapsed or quantum_seconds <= 0 or not 0 <= buffer_fraction < 1
    ):
        raise ValueError("invalid timing or budget value")
    overhead = process - elapsed
    # Three arms, 20 rows with a 32-token budget and 10 Qasper rows with 128 tokens.
    full_seconds = 60 * t32 + 30 * t128 + overhead
    qa_seconds = process + full_seconds
    billed_seconds = math.ceil(qa_seconds * (1 + buffer_fraction) / quantum_seconds) * quantum_seconds
    qa_rmb = billed_seconds * rmb_per_hour / 3600
    total = nll_rmb + qa_rmb
    admitted = total <= total_budget_rmb
    return {
        "status": "FAR_QA_COST_ADMITTED" if admitted else "FAR_QA_COST_REJECTED",
        "decision": "RUN" if admitted else "DO_NOT_RUN",
        "estimated_nll_cost_rmb": nll_rmb,
        "estimated_qa_cost_rmb": qa_rmb,
        "estimated_total_cost_rmb": total,
        "total_budget_rmb": total_budget_rmb,
        "canary_seconds": process,
        "estimated_full_qa_seconds": full_seconds,
        "buffer_fraction": buffer_fraction,
        "billing_quantum_seconds": quantum_seconds,
        "billed_qa_seconds": billed_seconds,
        "rmb_per_hour": rmb_per_hour,
        "assumption": "30 fixed rows x 3 arms; 20 rows use 32 decode tokens and 10 use 128.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-result", type=Path, required=True)
    parser.add_argument("--nll-cost-receipt", type=Path, required=True)
    parser.add_argument("--rmb-per-hour", type=float, required=True)
    parser.add_argument("--billing-quantum-seconds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = estimate(
            json.loads(args.canary_result.read_text()),
            json.loads(args.nll_cost_receipt.read_text()),
            rmb_per_hour=args.rmb_per_hour,
            quantum_seconds=args.billing_quantum_seconds)
    except (ValueError, TypeError, OSError, json.JSONDecodeError):
        report = {"status": "INVALID_FAR_QA_COST_INPUT", "decision": "DO_NOT_RUN"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["status"])
    return 0 if report["decision"] == "RUN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
