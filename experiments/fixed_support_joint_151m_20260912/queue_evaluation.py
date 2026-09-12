#!/usr/bin/env python3
"""Wait for the first S1 training block and evaluate both common checkpoints."""

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
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--block-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--support", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    state_path = args.block_root / "evaluation_queue_state.json"
    log_path = args.block_root / "evaluation_queue.log"
    state = {"status": "WAITING_TRAINING_BLOCK", "wait_pid": args.wait_pid}
    atomic_json(state_path, state)
    try:
        while alive(args.wait_pid):
            time.sleep(20)
        training_state = json.loads((args.block_root / "queue_state.json").read_text())
        if training_state.get("status") != "COMPLETE":
            raise RuntimeError("training supervisor ended without a complete three-arm block")
        python = "/root/miniconda3/bin/python"
        for arm in ("geo", "cosh", "full_z"):
            for checkpoint_name, checkpoint_label in (
                ("model_500m.pt", "500m"),
                ("model_1b.pt", "1b"),
            ):
                checkpoint = args.block_root / "runs" / arm / checkpoint_name
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                output = args.block_root / "evaluation" / checkpoint_label / arm
                state.update({"status": "RUNNING_EVALUATION", "arm": arm, "checkpoint": checkpoint_label})
                atomic_json(state_path, state)
                command = [
                    python,
                    str(args.code_root / "evaluate.py"),
                    "--original-root", str(args.original_root),
                    "--checkpoint", str(checkpoint),
                    "--eval-manifest", str(args.eval_manifest),
                    "--output", str(output),
                    "--arm", arm,
                    "--support", str(args.support),
                    "--seed", str(args.seed),
                ]
                with log_path.open("a") as handle:
                    handle.write(json.dumps({"command": command}) + "\n")
                    handle.flush()
                    subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        state.update({"status": "COMPLETE", "arm": None, "checkpoint": None})
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
