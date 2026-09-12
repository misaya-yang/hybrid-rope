#!/usr/bin/env python3
"""Queue a preregistered paired seed block after an upstream GPU stage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
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
    parser.add_argument("--required-block-state", type=Path, required=True)
    parser.add_argument("--upstream-eval-state", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--support", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "queue_state.json"
    log_path = args.output_root / "queue.log"
    state = {"status": "WAITING_UPSTREAM", "wait_pid": args.wait_pid}
    atomic_json(state_path, state)
    try:
        while alive(args.wait_pid):
            time.sleep(20)
        required = json.loads(args.required_block_state.read_text())
        if required.get("status") != "COMPLETE":
            raise RuntimeError("required paired training block is not complete")
        upstream_eval = (
            json.loads(args.upstream_eval_state.read_text())
            if args.upstream_eval_state.exists()
            else {"status": "MISSING"}
        )
        state["upstream_evaluation_status"] = upstream_eval.get("status")
        python = "/root/miniconda3/bin/python"
        common = [
            "--original-root", str(args.original_root),
            "--data-manifest", str(args.data_manifest),
            "--support", str(args.support),
            "--seed", str(args.seed),
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
        for arm in ("geo", "cosh", "full_z"):
            free_bytes = shutil.disk_usage(args.output_root).free
            if free_bytes < 5 * 2**30:
                raise RuntimeError(
                    f"less than 5 GiB free before {arm}: {free_bytes} bytes"
                )
            state.update(
                {
                    "status": "RUNNING_S1_BLOCK",
                    "arm": arm,
                    "free_bytes_before_arm": free_bytes,
                }
            )
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
            }
        )
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
