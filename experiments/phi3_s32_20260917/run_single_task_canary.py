#!/usr/bin/env python3
"""Run a 10-row Phi-3 TailSpline canary with the model's sliding window intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from experiments.phi3_s32_20260917.run_gate import atomic_json, freeze_table, read_rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_task_panel(source: Path, destination: Path, *, task: str) -> str:
    rows = [row for row in read_rows(source) if row.get("task") == task]
    if len(rows) != 10:
        raise ValueError(f"expected exactly 10 {task} rows, got {len(rows)}")
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text() != payload:
        raise ValueError("existing canary panel drift")
    destination.write_text(payload)
    return sha256(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--task", default="cwe")
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--selection-seed", type=int, default=20260917)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.root / "minimal_eval_manifest.json", {"evaluation_panels": {}, "lengths": []})
    task_panel = args.root / "assets" / f"{args.task}_10.jsonl"
    panel_sha = freeze_task_panel(args.panel, task_panel, task=args.task)
    table = args.root / "tables" / f"tailspline_s{int(args.scale)}.json"
    freeze_table(args.model, table, scale=args.scale)

    out = args.root / "runs" / f"tailspline_s{int(args.scale)}_16384_{args.task}10_sliding_flex_v1"
    command = [
        str(args.python), "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data", str(args.root / "minimal_eval_manifest.json"),
        "--model", str(args.model), "--arm", "Native",
        "--extra-panel", str(task_panel), "--only-extra-panels", "--skip-lm",
        "--length-cap", "16384", "--batch-size", "1",
        "--unmasked-unpadded-generate", "--phi3-sliding-flex",
        "--static-table-json", str(table),
        "--table-label", f"phi3mini4k_tailspline_s{int(args.scale)}_16384_{args.task}10_sliding_flex_v1",
        "--out", str(out), "--execute",
    ]
    status = out / "status.json"
    if not status.exists() or json.loads(status.read_text()).get("status") != "COMPLETE":
        subprocess.run(command, cwd=args.repo, check=True)

    rows = read_rows(out / "generations.jsonl")
    if len(rows) != 10 or {row.get("task") for row in rows} != {args.task}:
        raise ValueError("canary output is not the frozen 10-row task panel")
    report = {
        "status": "PHI3_SLIDING_FLEX_CANARY_COMPLETE_V1",
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "task": args.task,
        "task_selection_seed": args.selection_seed,
        "scale": args.scale,
        "length": 16384,
        "rows": len(rows),
        "panel_sha256": panel_sha,
        "generations_sha256": sha256(out / "generations.jsonl"),
        "attention_backend": "phi3_sliding_window_flex_attention",
        "sliding_window": 2047,
        "official_score": sum(float(row["ruler_official_score"]) for row in rows) / len(rows),
        "empty": sum(bool(row["empty"]) for row in rows),
        "ended_eos": sum(bool(row["ended_eos"]) for row in rows),
        "hit_cap": sum(bool(row["hit_cap"]) for row in rows),
    }
    atomic_json(args.root / "reports" / f"tailspline_s{int(args.scale)}_16384_{args.task}10.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
