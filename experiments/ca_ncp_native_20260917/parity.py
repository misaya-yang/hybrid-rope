#!/usr/bin/env python3
"""Run eight-prompt Native and carrier-table identity parity canaries."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from .io_utils import atomic_json, file_sha256


def command(
    *, python: str, data: Path, model: Path, panel: Path, out: Path,
    table: Path | None, alignment: Path | None, label: str, tasks: list[str],
) -> list[str]:
    value = [
        python, "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data", str(data), "--model", str(model), "--arm", "Native",
        "--extra-panel", str(panel), "--only-extra-panels", "--skip-lm",
        "--batch-size", "1", "--out", str(out),
        "--limit-per-cell", "1",
    ]
    for task in tasks:
        value += ["--task", task]
    if table is not None:
        value += ["--static-table-json", str(table), "--table-label", label]
    if alignment is not None:
        value += ["--ca-ncp-alignment-npz", str(alignment), "--ca-ncp-alignment-label", label]
    return value


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def execute_parity(
    *, repo: Path, python: str, data: Path, model: Path, root: Path,
    execute_commands: bool,
) -> dict:
    pilot = json.loads((root / "assets/pilot/manifest.json").read_text())
    panel = Path(pilot["panel"]["source_inputs"])
    tasks = list(pilot["parity"]["tasks"])
    if pilot["parity"].get("limit_per_cell") != 1 or len(tasks) != 8:
        raise ValueError("parity selection contract differs")
    construction = root / "construction"
    identity = root / "alignment/identity_alignment.npz"
    runs = root / "parity/runs"
    specs = {
        "native_identity": (None, identity),
        "carrier_plain": (construction / "carrier_ncp.json", None),
        "carrier_identity": (construction / "carrier_ncp.json", identity),
    }
    commands = {
        name: command(
            python=python, data=data, model=model, panel=panel, out=runs / name,
            table=table, alignment=alignment, label=f"ca_ncp_parity_{name}", tasks=tasks,
        )
        for name, (table, alignment) in specs.items()
    }
    if not execute_commands:
        return {
            "status": "PLAN_ONLY", "commands_without_execute": commands,
            "rows_per_path": 8, "native_plain_reused_from_formal_arm": "N0",
            "parallel_workers": 3,
        }
    if not (root / "runs/N0/status.json").is_file():
        raise FileNotFoundError("parity reuses the completed N0 formal arm")
    pending = []
    for name, value in commands.items():
        status = runs / name / "status.json"
        if status.is_file() and json.loads(status.read_text()) == {"status": "COMPLETE", "rows": 8, "lm_rows": 0}:
            continue
        pending.append((name, value))
    processes = [(name, subprocess.Popen(value + ["--execute"], cwd=repo)) for name, value in pending]
    failures = []
    for name, process in processes:
        code = process.wait()
        if code:
            failures.append((name, code))
    if failures:
        raise subprocess.CalledProcessError(failures[0][1], failures[0][0])
    rows = {name: read_rows(runs / name / "generations.jsonl") for name in specs}
    for name, value in rows.items():
        if len(value) != 8:
            raise ValueError(f"parity path {name} did not produce eight rows")
    comparisons = {}
    comparisons_to_base = {
        "native_identity": root / "runs/N0/generations.jsonl",
        "carrier_identity": runs / "carrier_plain/generations.jsonl",
    }
    for identity_name, base_path in comparisons_to_base.items():
        base = {row["row_id"]: row for row in read_rows(base_path)}
        if any(row["row_id"] not in base for row in rows[identity_name]):
            raise ValueError("parity identity rows are absent from the reused formal arm")
        differing = [
            row["row_id"] for row in rows[identity_name]
            if row["generated_ids"] != base[row["row_id"]]["generated_ids"]
        ]
        base_name = "N0_formal" if identity_name == "native_identity" else "carrier_plain_canary"
        comparisons[f"{base_name}_formal_vs_{identity_name}"] = {
            "rows": 8, "different_generated_token_rows": differing,
        }
        if differing:
            raise RuntimeError(f"identity alignment changed generation: {base_name}/{identity_name}/{differing}")
    report = {
        "status": "CA_NCP_RUNTIME_PARITY_PASS_V1_1",
        "comparisons": comparisons,
        "panel_sha256": file_sha256(panel),
        "identity_alignment_sha256": file_sha256(identity),
        "identity_run_raw_sha256": {name: file_sha256(runs / name / "generations.jsonl") for name in specs},
        "plain_base_raw_sha256": {
            "N0_formal": file_sha256(root / "runs/N0/generations.jsonl"),
            "carrier_plain_canary": file_sha256(runs / "carrier_plain/generations.jsonl"),
        },
        "accuracy_role": "none",
        "scope": "Runtime identity canary only; not a model-quality result.",
    }
    atomic_json(root / "parity/PARITY_REPORT.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/hybrid-rope"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(execute_parity(
            repo=args.repo, python=args.python, data=args.data, model=args.model,
            root=args.root, execute_commands=False,
        ), indent=2))
        return
    with open("/tmp/hybrid-rope-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ], text=True).strip()
        if active:
            raise RuntimeError("a GPU process is active; parity does not co-run or stop it")
        report = execute_parity(
            repo=args.repo, python=args.python, data=args.data, model=args.model,
            root=args.root, execute_commands=True,
        )
    print(json.dumps({"status": report["status"]}))


if __name__ == "__main__":
    main()
