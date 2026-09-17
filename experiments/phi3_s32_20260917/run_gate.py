#!/usr/bin/env python3
"""Run the frozen Phi-3-mini-4K TailSpline S=32 RULER gate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.fixed_rope_three_interfaces_20260913 import tables


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
DEFAULT_SCALE = 32.0
THRESHOLD = 0.80


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def task_equal_score(rows: list[dict]) -> tuple[float, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task"))
        score = row.get("ruler_official_score")
        if task not in TASKS or score is None:
            raise ValueError("generation row lacks a registered task or official score")
        grouped[task].append(float(score))
    if set(grouped) != set(TASKS) or any(len(grouped[task]) != 10 for task in TASKS):
        raise ValueError("gate output must contain exactly Full-13 x 10")
    by_task = {task: sum(grouped[task]) / len(grouped[task]) for task in TASKS}
    return sum(by_task.values()) / len(TASKS), by_task


def freeze_table(model: Path, path: Path, *, scale: float) -> dict:
    config = json.loads((model / "config.json").read_text())
    geometry = tables.model_geometry(config)
    values, gain, construction = tables.build_analytic(
        config, method="tailspline", scale=scale,
        low=None, high=None, depth=1.0, gain=None,
    )
    receipt = tables.make_receipt(
        candidate_id=f"phi3mini4k_tailspline_s{int(scale)}",
        model_id="phi3mini4k", role="candidate", scale=scale,
        geometry=geometry, values=values, gain=gain,
        construction=construction, source="analytic:tailspline",
        changed_variables=["internal_frequency_allocation"],
    )
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != receipt:
            raise ValueError("existing Phi TailSpline table drift")
    else:
        atomic_json(path, receipt)
    return receipt


def validate_panel(path: Path, *, length: int) -> str:
    rows = read_rows(path)
    counts = defaultdict(int)
    for row in rows:
        counts[str(row.get("task"))] += 1
        if (
            int(row.get("length_cap", -1)) != length
            or len(row.get("prompt_ids") or []) + int(row.get("max_new_tokens", 0)) > length
            or row.get("selection_mode") != "source-order"
            or row.get("selection_uses_model_outputs") is not False
        ):
            raise ValueError("Phi RULER panel contract drift")
    if len(rows) != 130 or counts != {task: 10 for task in TASKS}:
        raise ValueError("Phi panel is not Full-13 x 10")
    return sha256(path)


def prepare_32k(args: argparse.Namespace) -> Path:
    root = args.root / "assets32"
    command = [
        str(args.python), "-m",
        "experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer",
        "--model", str(args.model), "--model-id", f"phi3mini4k_s{int(args.scale)}_32k",
        "--data-root", str(args.ruler), "--out", str(root),
        "--scale", str(int(args.scale)), "--lengths", "32768", "--rows-per-task", "10",
        "--seed", "20261101", "--qa-offset", "5600",
    ]
    subprocess.run(command, cwd=args.repo, check=True)
    return root / "panels/32768/inputs.jsonl"


def run_length(
    args: argparse.Namespace, *, length: int, panel: Path, table: Path, scale: float,
) -> dict:
    panel_sha = validate_panel(panel, length=length)
    out = args.root / "runs" / f"tailspline_s{int(scale)}_{length}"
    command = [
        str(args.python), "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data", str(args.root / "minimal_eval_manifest.json"),
        "--model", str(args.model), "--arm", "Native",
        "--extra-panel", str(panel), "--only-extra-panels", "--skip-lm",
        "--length-cap", str(length), "--batch-size", "1",
        "--unmasked-unpadded-generate", "--phi3-sliding-flex",
        "--static-table-json", str(table),
        "--table-label", f"phi3mini4k_tailspline_s{int(scale)}_{length}",
        "--out", str(out), "--execute",
    ]
    status = out / "status.json"
    if not status.is_file() or json.loads(status.read_text()) != {
        "status": "COMPLETE", "rows": 130, "lm_rows": 0,
    }:
        subprocess.run(command, cwd=args.repo, check=True)
    rows = read_rows(out / "generations.jsonl")
    score, by_task = task_equal_score(rows)
    report = {
        "status": "PHI3_TAILSPLINE_RULER_GATE_COMPLETE_V1",
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "method": "TailSpline exact finite-grid",
        "scale": scale,
        "length": length,
        "rows": len(rows),
        "panel_sha256": panel_sha,
        "generations_sha256": sha256(out / "generations.jsonl"),
        "task_equal_official_score": score,
        "threshold": THRESHOLD,
        "passes_threshold": score >= THRESHOLD,
        "by_task": by_task,
    }
    atomic_json(args.root / "reports" / f"tailspline_s{int(scale)}_{length}.json", report)
    return report


def shutdown_after_report(args: argparse.Namespace, report: dict) -> None:
    atomic_json(args.root / "decision.json", {
        "status": "SHUTDOWN_AUTHORIZED_BELOW_THRESHOLD",
        "trigger_length": report["length"],
        "score": report["task_equal_official_score"],
        "threshold": THRESHOLD,
        "reports_preserved": True,
    })
    subprocess.run(["sync"], check=True)
    subprocess.run(["bash", "/usr/bin/shutdown"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel16", type=Path, required=True)
    parser.add_argument("--ruler", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--scale", type=float, choices=(4.0, 32.0), default=DEFAULT_SCALE)
    parser.add_argument("--stop-after-16k", action="store_true")
    parser.add_argument("--shutdown-below-threshold", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "status": "PLAN_ONLY" if not args.execute else "EXECUTING",
        "model": str(args.model), "scale": args.scale,
        "first_length": 16384, "conditional_second_length": 32768,
        "threshold": THRESHOLD,
        "shutdown_below_threshold": bool(args.shutdown_below_threshold),
    }
    print(json.dumps(plan, sort_keys=True))
    if not args.execute:
        return
    args.root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.root / "minimal_eval_manifest.json", {"evaluation_panels": {}, "lengths": []})
    table_path = args.root / "tables" / f"tailspline_s{int(args.scale)}.json"
    freeze_table(args.model, table_path, scale=args.scale)
    first = run_length(
        args, length=16384, panel=args.panel16, table=table_path, scale=args.scale,
    )
    if args.stop_after_16k:
        atomic_json(args.root / "decision.json", {
            "status": "STOP_AFTER_16K_NO_SHUTDOWN",
            "score": first["task_equal_official_score"],
            "threshold": THRESHOLD,
            "scale": args.scale,
        })
        return
    if not first["passes_threshold"]:
        if args.shutdown_below_threshold:
            shutdown_after_report(args, first)
        return
    panel32 = prepare_32k(args)
    second = run_length(
        args, length=32768, panel=panel32, table=table_path, scale=args.scale,
    )
    if not second["passes_threshold"] and args.shutdown_below_threshold:
        shutdown_after_report(args, second)
    elif second["passes_threshold"]:
        atomic_json(args.root / "decision.json", {
            "status": "HOLD_FOR_USER_AFTER_32K_PASS",
            "score": second["task_equal_official_score"],
            "threshold": THRESHOLD,
        })


if __name__ == "__main__":
    main()
