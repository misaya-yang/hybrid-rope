#!/usr/bin/env python3
"""Recover an interrupted paired S1 trio without overwriting completed arms."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path


MINIMUM_FREE_BYTES = 5 * 1024**3


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("a") as handle:
        handle.write(json.dumps({"command": command, "recovery": True}) + "\n")
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def arm_action(output: Path) -> str:
    if not output.exists() or not any(output.iterdir()):
        return "start"
    status_path = output / "status.json"
    if status_path.exists():
        status = read_json(status_path)
        if status.get("status") == "COMPLETE":
            if not (output / "model_1b.pt").is_file():
                raise RuntimeError(f"completed arm lacks model_1b.pt: {output}")
            return "skip"
    if (output / "resume.pt").is_file():
        return "resume"
    raise RuntimeError(f"nonempty interrupted arm lacks resume.pt: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--support", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reason", default="host outage")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "queue_state.json"
    log_path = args.output_root / "queue.log"
    state = {
        "status": "RECOVERING",
        "arm": None,
        "support": int(args.support),
        "seed": int(args.seed),
        "recovery_reason": str(args.reason),
        "recovery_started_at": time.time(),
    }
    atomic_json(state_path, state)
    try:
        probe_status = read_json(args.output_root / "probe_full_z" / "status.json")
        if probe_status.get("status") != "PROBE_PASS":
            raise RuntimeError("existing full-z qualification is not PROBE_PASS")

        python = "/root/miniconda3/bin/python"
        common = [
            "--original-root", str(args.original_root),
            "--data-manifest", str(args.data_manifest),
            "--support", str(args.support),
            "--seed", str(args.seed),
        ]
        for arm in ("geo", "cosh", "full_z"):
            output = args.output_root / "runs" / arm
            action = arm_action(output)
            if action == "skip":
                continue
            free_bytes = shutil.disk_usage(args.output_root).free
            if free_bytes < MINIMUM_FREE_BYTES:
                raise RuntimeError(
                    f"free-space gate failed before {arm}: {free_bytes} bytes"
                )
            state.update(
                {
                    "status": "RUNNING_S1_BLOCK",
                    "arm": arm,
                    "action": action,
                    "free_bytes_before_arm": free_bytes,
                }
            )
            atomic_json(state_path, state)
            command = [
                python,
                str(args.code_root / "run.py"),
                *common,
                "--arm", arm,
                "--output", str(output),
            ]
            if action == "resume":
                command.append("--resume")
            run_logged(command, log_path)
            completed = read_json(output / "status.json")
            if completed.get("status") != "COMPLETE":
                raise RuntimeError(f"arm did not finish cleanly: {arm}")

        state.update(
            {
                "status": "COMPLETE",
                "arm": None,
                "action": None,
                "finished_at": time.time(),
            }
        )
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
