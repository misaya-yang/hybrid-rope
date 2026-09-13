#!/usr/bin/env python3
"""Freeze an explicit E2 choice without deleting non-selected candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--decision", choices=("strict-feasible", "pareto", "unresolved"), required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    arms = summary.get("arms", {})
    if summary.get("status") != "COMPLETE" or args.candidate not in arms:
        raise ValueError("selection summary is incomplete or does not contain the candidate")
    receipt = {
        "status": "SELECTION_LOCKED",
        "selected_candidate": args.candidate,
        "decision": args.decision,
        "reason": args.reason,
        "source_summary": str(args.summary),
        "selected_metrics": {
            "arm": arms[args.candidate],
            "contrasts": summary.get("contrasts", {}).get(args.candidate),
        },
        "all_candidate_labels_retained": sorted(
            label for label in arms if label not in ("Native", "BM_g4", "MrPro_g4", "C42V24_g4")
        ),
        "claim_boundary": "this is a development choice for internal confirmation, not an independent-confirmation result",
    }
    if not args.execute:
        print(json.dumps({**receipt, "status": "PLAN_ONLY"}, indent=2, sort_keys=True))
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "selected_candidate": args.candidate, "decision": args.decision}, sort_keys=True))


if __name__ == "__main__":
    main()
