#!/usr/bin/env python3
"""Print or execute the three-arm CA-NCP safety follow-up."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from experiments.ca_ncp_native_20260917.run import arm_command

from . import ARMS


def commands(*, python: str, data: Path, model: Path, source: Path, root: Path) -> dict[str, list[str]]:
    panel = Path(json.loads((source / "assets/pilot/manifest.json").read_text())["panel"]["source_inputs"])
    carrier = source / "construction/carrier_ncp.json"
    cap = root / "alignments/operator_cap/alignment.npz"
    axis = root / "alignments/P_axis_consensus/alignment.npz"
    specs = {
        "N_operator_cap": (None, cap),
        "P_operator_cap": (carrier, cap),
        "P_axis_consensus": (carrier, axis),
    }
    return {
        arm: arm_command(
            python=python, data=data, model=model, panel=panel, out=root / "runs" / arm,
            table=specs[arm][0], alignment=specs[arm][1], arm=arm,
        )
        for arm in ARMS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/hybrid-rope"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parallel-workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    planned = commands(
        python=args.python, data=args.data, model=args.model,
        source=args.source_root, root=args.root,
    )
    required = [
        args.root / "METHOD_RECEIPT.json",
        args.root / "alignments/operator_cap/alignment.npz",
        args.root / "alignments/P_axis_consensus/alignment.npz",
        args.source_root / "parity/PARITY_REPORT.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("safe follow-up assets are incomplete: " + ", ".join(missing))
    pending = []
    for arm in ARMS:
        status = args.root / "runs" / arm / "status.json"
        if not status.is_file() or json.loads(status.read_text()) != {"status": "COMPLETE", "rows": 130, "lm_rows": 0}:
            pending.append(arm)
    if not args.execute:
        print(json.dumps({
            "status": "PLAN_ONLY",
            "gpu_execution": False,
            "arms": list(ARMS),
            "pending": pending,
            "parallel_workers": args.parallel_workers,
            "expected_new_generations": 130 * len(pending),
            "commands_without_execute": planned,
        }, indent=2))
        return
    with open("/tmp/hybrid-rope-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ], text=True).strip()
        if active:
            raise RuntimeError("a GPU process is active; safe follow-up does not co-run or stop it")
        for start in range(0, len(pending), args.parallel_workers):
            batch = pending[start : start + args.parallel_workers]
            processes = [
                (arm, subprocess.Popen(planned[arm] + ["--execute"], cwd=args.repo)) for arm in batch
            ]
            failures = []
            for arm, process in processes:
                code = process.wait()
                if code:
                    failures.append((arm, code))
            if failures:
                raise subprocess.CalledProcessError(failures[0][1], failures[0][0])
    subprocess.run([
        args.python, "-m", "experiments.ca_ncp_safe_followup_20260917.report",
        "--source-root", str(args.source_root), "--root", str(args.root),
        "--out", str(args.root / "reports/paired_report.json"),
    ], cwd=args.repo, check=True)
    print(json.dumps({"status": "CA_NCP_SAFE_FOLLOWUP_COMPLETE_V1", "arms": list(ARMS)}))


if __name__ == "__main__":
    main()
