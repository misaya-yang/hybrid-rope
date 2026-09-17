#!/usr/bin/env python3
"""Plan or execute one frozen A/B Native follow-up wave."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from . import METHODS
from .contract import CURRENT_REPORT, DATA, MODEL, NCP_TABLE, PANEL, PANEL_SHA256, ROOT, WAVE1, WAVE2


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(method: str, out: Path, *, canary: bool) -> list[str]:
    value = [
        sys.executable, "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data", str(DATA), "--model", str(MODEL), "--arm", "Native",
        "--extra-panel", str(PANEL), "--only-extra-panels", "--skip-lm",
        "--batch-size", "1", "--native-followup-method", method,
        "--native-followup-candidate-table", str(NCP_TABLE), "--out", str(out),
    ]
    if canary:
        value += ["--limit-per-cell", "1"]
        for task in (
            "niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
            "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery",
        ):
            value += ["--task", task]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=("canary", "wave1", "wave2"), required=True)
    parser.add_argument("--parallel-workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    methods = METHODS if args.wave == "canary" else WAVE1 if args.wave == "wave1" else WAVE2
    canary = args.wave == "canary"
    expected = 8 if canary else 130
    root = ROOT / ("canary" if canary else "runs")
    commands = {method: command(method, root / method, canary=canary) for method in methods}
    readiness = {
        "panel_exists": PANEL.is_file(), "panel_sha256": sha(PANEL) if PANEL.is_file() else None,
        "panel_sha256_expected": PANEL_SHA256, "model_exists": MODEL.is_dir(),
        "candidate_table_exists": NCP_TABLE.is_file(),
        "current_round_complete": CURRENT_REPORT.is_file(),
    }
    if not args.execute:
        print(json.dumps({
            "status": "PLAN_ONLY", "wave": args.wave, "methods": methods,
            "expected_rows_per_method": expected, "parallel_workers": args.parallel_workers,
            "readiness": readiness, "commands_without_execute": commands,
        }, indent=2, default=str))
        return
    if not all((readiness["panel_exists"], readiness["model_exists"], readiness["candidate_table_exists"])):
        raise FileNotFoundError(f"execution assets incomplete: {readiness}")
    if readiness["panel_sha256"] != PANEL_SHA256:
        raise ValueError("panel identity drift")
    if not canary and not readiness["current_round_complete"]:
        raise RuntimeError("current authorized three-arm round must complete before a new formal wave")
    pending = []
    for method in methods:
        status = root / method / "status.json"
        if not status.is_file() or json.loads(status.read_text()) != {"status": "COMPLETE", "rows": expected, "lm_rows": 0}:
            pending.append(method)
    with open("/tmp/hybrid-rope-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ], text=True).strip()
        if active:
            raise RuntimeError("GPU is active")
        for start in range(0, len(pending), args.parallel_workers):
            batch = pending[start : start + args.parallel_workers]
            processes = [(method, subprocess.Popen(commands[method] + ["--execute"])) for method in batch]
            failures = [(method, process.wait()) for method, process in processes]
            failures = [(method, code) for method, code in failures if code]
            if failures:
                raise subprocess.CalledProcessError(failures[0][1], failures[0][0])
    print(json.dumps({"status": "WAVE_COMPLETE", "wave": args.wave, "methods": methods}))


if __name__ == "__main__":
    main()
