#!/usr/bin/env python3
"""Power off the host after the complete seed-42 S1 block and evaluation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

import torch


ARMS = ("geo", "cosh", "full_z")
CHECKPOINTS = ("model_500m.pt", "model_1b.pt")
CHECKPOINT_LABELS = ("500m", "1b")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_in_use(path: Path) -> bool:
    target = path.resolve()
    for process in Path("/proc").glob("[0-9]*"):
        fd_root = process / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                if descriptor.resolve() == target:
                    return True
            except (FileNotFoundError, PermissionError):
                continue
    return False


def completion_ready(block_root: Path) -> bool:
    queue_state = block_root / "queue_state.json"
    evaluation_state = block_root / "evaluation_queue_state.json"
    if not queue_state.is_file() or not evaluation_state.is_file():
        return False
    if read_json(queue_state).get("status") != "COMPLETE":
        return False
    if read_json(evaluation_state).get("status") != "COMPLETE":
        return False
    for arm in ARMS:
        run_root = block_root / "runs" / arm
        status_path = run_root / "status.json"
        if not status_path.is_file() or read_json(status_path).get("status") != "COMPLETE":
            return False
        for checkpoint in CHECKPOINTS:
            if not (run_root / checkpoint).is_file():
                return False
        for label in CHECKPOINT_LABELS:
            eval_status = block_root / "evaluation" / label / arm / "status.json"
            if not eval_status.is_file() or read_json(eval_status).get("status") != "COMPLETE":
                return False
    return True


def validate_arm_checkpoints(run_root: Path, arm: str) -> list[dict]:
    verified = []
    expected_updates = {"model_500m.pt": 7_629, "model_1b.pt": 15_258}
    for checkpoint in CHECKPOINTS:
        path = run_root / checkpoint
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        completed_updates = int(
            payload.get(
                "completed_updates",
                payload.get("metadata", {}).get("completed_updates", 0),
            )
        )
        if completed_updates != expected_updates[checkpoint]:
            raise RuntimeError(
                f"unexpected completed_updates={completed_updates} for {path}"
            )
        verified.append(
            {
                "arm": arm,
                "checkpoint": checkpoint,
                "bytes": path.stat().st_size,
                "completed_updates": completed_updates,
            }
        )
    return verified


def cleanup_completed_resumes(block_root: Path, cleanup_log: Path) -> list[dict]:
    removed = []
    for arm in ARMS:
        run_root = block_root / "runs" / arm
        resume = run_root / "resume.pt"
        status_path = run_root / "status.json"
        if not resume.exists() or not status_path.is_file():
            continue
        if read_json(status_path).get("status") != "COMPLETE" or file_in_use(resume):
            continue
        validate_arm_checkpoints(run_root, arm)
        size = resume.stat().st_size
        resume.unlink()
        removed.append({"arm": arm, "path": str(resume), "bytes": size})
        with cleanup_log.open("a") as handle:
            handle.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                f"deleted completed {arm} resume.pt bytes={size} "
                "after checkpoint-readability verification\n"
            )
    return removed


def validate_all_checkpoints(block_root: Path) -> list[dict]:
    verified = []
    for arm in ARMS:
        verified.extend(validate_arm_checkpoints(block_root / "runs" / arm, arm))
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-root", type=Path, required=True)
    parser.add_argument("--cleanup-log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()

    args.block_root = args.block_root.resolve()
    lock_path = args.block_root / "shutdown_guardian.lock"
    receipt_path = args.block_root / "shutdown_receipt.json"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another shutdown guardian is already running") from error

        removed = []
        while not completion_ready(args.block_root):
            removed.extend(cleanup_completed_resumes(args.block_root, args.cleanup_log))
            time.sleep(args.poll_seconds)

        removed.extend(cleanup_completed_resumes(args.block_root, args.cleanup_log))
        receipt = {
            "status": "POWERING_OFF",
            "reason": "User authorized shutdown after complete seed42 S1 experiment",
            "block_root": str(args.block_root),
            "completed_at": time.time(),
            "verified_checkpoints": validate_all_checkpoints(args.block_root),
            "removed_resume_checkpoints": removed,
        }
        atomic_json(receipt_path, receipt)
        os.sync()
        subprocess.run(["/sbin/poweroff"], check=True)


if __name__ == "__main__":
    main()
