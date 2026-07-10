#!/usr/bin/env python3
"""Export the surviving planned Phase 16 runs as a sanitized flat CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "stage",
    "run_id",
    "config_id",
    "tier",
    "seq_len",
    "num_heads",
    "head_dim",
    "tau",
    "theory_tau",
    "seed",
    "train_tokens",
    "eval_lengths",
    "passkey_lengths",
    "passkey_trials",
    "completed_at",
    "train_time_sec",
    "eval_time_sec",
    "inv_freq_hash",
    "ppl_json",
    "passkey_summary_json",
]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(source: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for plan_name in ("pilot_plan.json", "confirm_plan.json"):
        plan = _read_json(source / plan_name)
        if not isinstance(plan, list):
            raise ValueError(f"{plan_name} must contain a list of run specifications")
        for spec in plan:
            run_id = spec["run_id"]
            result_path = source / "runs" / run_id / "result.json"
            if not result_path.exists():
                raise FileNotFoundError(f"planned Phase 16 result is missing: {run_id}")
            result = _read_json(result_path)
            if result.get("run_id") != run_id:
                raise ValueError(f"run-id mismatch in surviving result: {run_id}")
            config_id = (
                f"L{spec['seq_len']}_H{spec['num_heads']}_Dh{spec['head_dim']}"
            )
            rows.append(
                {
                    "stage": spec["stage"],
                    "run_id": run_id,
                    "config_id": config_id,
                    "tier": spec["tier"],
                    "seq_len": spec["seq_len"],
                    "num_heads": spec["num_heads"],
                    "head_dim": spec["head_dim"],
                    "tau": spec["tau"],
                    "theory_tau": spec["theory_tau"],
                    "seed": spec["seed"],
                    "train_tokens": spec["train_tokens"],
                    "eval_lengths": json.dumps(spec["eval_lengths"], separators=(",", ":")),
                    "passkey_lengths": json.dumps(
                        spec["passkey_lengths"], separators=(",", ":")
                    ),
                    "passkey_trials": spec["passkey_trials"],
                    "completed_at": result.get("completed_at", ""),
                    "train_time_sec": result.get("train_time_sec", ""),
                    "eval_time_sec": result.get("eval_time_sec", ""),
                    "inv_freq_hash": result.get("inv_freq_hash", ""),
                    "ppl_json": json.dumps(
                        result.get("ppl", {}), sort_keys=True, separators=(",", ":")
                    ),
                    "passkey_summary_json": json.dumps(
                        result.get("passkey", {}).get("summary", {}),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
    if len(rows) != 99:
        raise ValueError(f"expected 99 planned Phase 16 runs, found {len(rows)}")
    return rows


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.source)
    write_csv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
