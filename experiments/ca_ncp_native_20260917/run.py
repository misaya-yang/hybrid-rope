#!/usr/bin/env python3
"""Print or execute the frozen five-arm CA-NCP Native pilot.

Default is PLAN_ONLY. ``--execute`` acquires the shared GPU lock, runs parity,
then completes N0/C0/P0/N1/P1 without changing the frozen method.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from . import ARMS
from .parity import execute_parity


DEFAULT_DATA = Path(
    "/root/autodl-tmp/today_rope_plan_20260914/"
    "tailspline_olmo_s4_classic/assets/ppl46/manifest.json"
)


def arm_command(
    *, python: str, data: Path, model: Path, panel: Path, out: Path,
    table: Path | None, alignment: Path | None, arm: str,
) -> list[str]:
    command = [
        python, "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
        "--data", str(data), "--model", str(model), "--arm", "Native",
        "--extra-panel", str(panel), "--only-extra-panels", "--skip-lm",
        "--batch-size", "1", "--out", str(out),
    ]
    if table is not None:
        command += ["--static-table-json", str(table), "--table-label", f"ca_ncp_{arm}"]
    if alignment is not None:
        command += ["--ca-ncp-alignment-npz", str(alignment), "--ca-ncp-alignment-label", f"ca_ncp_{arm}"]
    return command


def commands(*, python: str, data: Path, model: Path, root: Path, arms: list[str]) -> dict[str, list[str]]:
    pilot_path = root / "assets/pilot/manifest.json"
    panel = (
        Path(json.loads(pilot_path.read_text())["panel"]["source_inputs"])
        if pilot_path.is_file() else root / "assets/pilot/REUSED_SOURCE_INPUTS.jsonl"
    )
    construction = root / "construction"
    alignment = root / "alignment/alignment.npz"
    specs = {
        "N0": (None, None),
        "C0": (construction / "ncp.json", None),
        "P0": (construction / "carrier_ncp.json", None),
        "N1": (None, alignment),
        "P1": (construction / "carrier_ncp.json", alignment),
    }
    return {
        arm: arm_command(
            python=python, data=data, model=model, panel=panel,
            out=root / "runs" / arm, table=specs[arm][0], alignment=specs[arm][1], arm=arm,
        )
        for arm in arms
    }


def validate_assets(root: Path, model: Path) -> None:
    required = (
        root / "construction/METHOD_RECEIPT.json",
        root / "statistics/STATISTICS_RECEIPT.json",
        root / "alignment/ALIGNMENT_RECEIPT.json",
        root / "alignment/alignment.npz",
        root / "alignment/identity_alignment.npz",
        root / "assets/pilot/manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("CA-NCP execution assets are incomplete: " + ", ".join(missing))
    method = json.loads((root / "construction/METHOD_RECEIPT.json").read_text())
    pilot = json.loads((root / "assets/pilot/manifest.json").read_text())
    if method["model_identity"]["config_sha256"] != pilot["model_identity"]["config_sha256"]:
        raise ValueError("pilot and construction use different checkpoints")
    if Path(method["model_identity"]["model_path"]).resolve() != model.resolve():
        raise ValueError("requested model path differs from METHOD_RECEIPT")
    panel = Path(pilot["panel"]["source_inputs"])
    if not panel.is_file():
        raise FileNotFoundError(f"reused pilot panel is unavailable: {panel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/hybrid-rope"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--parallel-workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("duplicate arms")
    planned = commands(
        python=args.python, data=args.data, model=args.model, root=args.root, arms=args.arms,
    )
    if not args.execute:
        incomplete = 0
        for arm in args.arms:
            status = args.root / "runs" / arm / "status.json"
            if not status.is_file() or json.loads(status.read_text()) != {"status": "COMPLETE", "rows": 130, "lm_rows": 0}:
                incomplete += 1
        parity_ready = (args.root / "parity/PARITY_REPORT.json").is_file()
        print(json.dumps({
            "status": "PLAN_ONLY",
            "model_loaded": False,
            "gpu_execution": False,
            "arms": args.arms,
            "expected_rows_per_arm": 130,
            "expected_new_formal_generations": 130 * incomplete,
            "expected_new_parity_generations": 0 if parity_ready else 24,
            "expected_new_generations": 130 * incomplete + (0 if parity_ready else 24),
            "baseline_reuse": [arm for arm in ("N0", "C0") if (args.root / "runs" / arm).is_symlink()],
            "parallel_workers": args.parallel_workers,
            "parity_rows": 8,
            "commands_without_execute": planned,
        }, indent=2))
        return
    validate_assets(args.root, args.model)
    with open("/tmp/hybrid-rope-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ], text=True).strip()
        if active:
            raise RuntimeError("a GPU process is active; CA-NCP does not co-run or stop it")
        parity = execute_parity(
            repo=args.repo, python=args.python, data=args.data, model=args.model,
            root=args.root, execute_commands=True,
        )
        if parity.get("status") != "CA_NCP_RUNTIME_PARITY_PASS_V1_1":
            raise RuntimeError("runtime parity did not pass")
        panel_path = Path(json.loads((args.root / "assets/pilot/manifest.json").read_text())["panel"]["source_inputs"])
        panel_rows = [
            json.loads(line) for line in panel_path.read_text().splitlines()
            if line.strip()
        ]
        expected_ids = [str(row["row_id"]) for row in panel_rows]
        pending = []
        for arm in args.arms:
            out = args.root / "runs" / arm
            status = out / "status.json"
            if status.is_file() and json.loads(status.read_text()) == {"status": "COMPLETE", "rows": 130, "lm_rows": 0}:
                rows = [json.loads(line) for line in (out / "generations.jsonl").read_text().splitlines() if line.strip()]
                if [str(row["row_id"]) for row in rows] != expected_ids:
                    raise ValueError(f"completed arm {arm} has another prompt identity")
                continue
            pending.append(arm)
        for start in range(0, len(pending), args.parallel_workers):
            batch = pending[start:start + args.parallel_workers]
            processes = [(arm, subprocess.Popen(planned[arm] + ["--execute"], cwd=args.repo)) for arm in batch]
            failures = []
            for arm, process in processes:
                code = process.wait()
                if code:
                    failures.append((arm, code))
            if failures:
                raise subprocess.CalledProcessError(failures[0][1], failures[0][0])
    if set(args.arms) == set(ARMS):
        subprocess.run([
            args.python, "-m", "experiments.ca_ncp_native_20260917.report",
            "--assets", str(args.root / "assets/pilot"),
            "--run-root", str(args.root / "runs"),
            "--construction", str(args.root / "construction"),
            "--alignment", str(args.root / "alignment"),
            "--out", str(args.root / "reports/paired_report.json"),
        ], cwd=args.repo, check=True)
    print(json.dumps({"status": "CA_NCP_QUEUE_COMPLETE", "arms": args.arms}))


if __name__ == "__main__":
    main()
