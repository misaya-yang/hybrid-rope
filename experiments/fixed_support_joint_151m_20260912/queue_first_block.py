#!/usr/bin/env python3
"""Wait for S6 closure, qualify full-z, then run one paired S1 trio."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("a") as handle:
        handle.write(json.dumps({"command": command}) + "\n")
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--s6-summary", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "queue_state.json"
    log_path = args.output_root / "queue.log"
    state = {
        "status": "WAITING_S6_CLOSE",
        "wait_pid": int(args.wait_pid),
        "started_at": time.time(),
    }
    atomic_json(state_path, state)
    try:
        while alive(args.wait_pid):
            time.sleep(15)
        summary = json.loads(args.s6_summary.read_text())
        if summary.get("status") != "COMPLETE":
            raise RuntimeError("S6 endpoint process ended without COMPLETE summary")

        python = "/root/miniconda3/bin/python"
        common = [
            "--original-root", str(args.original_root),
            "--data-manifest", str(args.data_manifest),
            "--support", "500000",
            "--seed", "42",
        ]
        state["status"] = "PREFLIGHT"
        atomic_json(state_path, state)
        run_logged(
            [
                python,
                str(args.code_root / "preflight.py"),
                *common,
                "--output", str(args.output_root / "preflight.json"),
            ],
            log_path,
        )

        state["status"] = "FULL_Z_PROBE"
        atomic_json(state_path, state)
        probe = args.output_root / "probe_full_z"
        run_logged(
            [
                python,
                str(args.code_root / "run.py"),
                *common,
                "--arm", "full_z",
                "--output", str(probe),
                "--probe",
            ],
            log_path,
        )
        probe_status = json.loads((probe / "status.json").read_text())
        if (
            probe_status.get("status") != "PROBE_PASS"
            or float(probe_status.get("steady_tokens_per_second", 0.0)) <= 0.0
            or float(probe_status.get("first_allocation_update_max_abs", 0.0)) <= 0.0
        ):
            raise RuntimeError("full-z discarded qualification did not pass")

        for arm in ("geo", "cosh", "full_z"):
            state.update({"status": "RUNNING_S1_BLOCK", "arm": arm})
            atomic_json(state_path, state)
            run_logged(
                [
                    python,
                    str(args.code_root / "run.py"),
                    *common,
                    "--arm", arm,
                    "--output", str(args.output_root / "runs" / arm),
                ],
                log_path,
            )
        state.update({"status": "COMPLETE", "arm": None, "finished_at": time.time()})
        atomic_json(state_path, state)
    except Exception as error:
        state.update(
            {
                "status": "FAILED",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "failed_at": time.time(),
            }
        )
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
