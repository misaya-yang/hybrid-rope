#!/usr/bin/env python3
"""Execute the frozen queue in stage order and score completed comparisons."""
from __future__ import annotations

import argparse
import atexit
import json
from pathlib import Path
import subprocess
import sys
import time

from .pipeline import atomic_json, exclusive_lock, plan_queue, read_json, score_comparison


def ordered_batches(queue: dict, contract: dict) -> list[tuple[str, str]]:
    stage_rank = {stage: index for index, stage in enumerate(contract["stage_order"])}
    pairs = {(job["stage"], job["model_id"]) for job in queue.get("queue", [])}
    return sorted(pairs, key=lambda pair: (stage_rank[pair[0]], pair[1]))


def score_ready(contract_path: Path, score_root: Path, *, draws: int) -> dict:
    contract = read_json(contract_path)
    results = {}
    for comparison in contract["comparisons"]:
        comparison_id = comparison["comparison_id"]
        try:
            result = score_comparison(
                contract_path, comparison_id, score_root / comparison_id,
                draws=draws, seed=20260913,
            )
        except ValueError as error:
            if not any(fragment in str(error) for fragment in (
                "is missing", "not post-run verified COMPLETE",
            )):
                raise
            results[comparison_id] = {"status": "PENDING_ROWS", "reason": str(error)}
        else:
            results[comparison_id] = {
                "status": "COMPLETE",
                "candidate": result["candidate"],
                "scope": result["scope"],
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()

    queue_path = args.queue.resolve()
    queue = read_json(queue_path)
    contract_path = Path(queue["contract_path"])
    contract = read_json(contract_path)
    batches = ordered_batches(queue, contract)
    plan = {
        "status": "READY_GPU" if batches else "COMPLETE_NOTHING_MISSING",
        "queue": str(queue_path),
        "batches": [{"stage": stage, "model_id": model_id} for stage, model_id in batches],
        "policy": "fixed order; no score-triggered cancellation or mid-run retuning",
    }
    if not args.execute or not batches:
        print(json.dumps(plan, sort_keys=True))
        return

    root = queue_path.parent.parent
    run_lock = exclusive_lock(
        root / ".night_run.lock",
        {"queue": str(queue_path), "contract": str(contract_path)},
    )
    run_lock.__enter__()
    release_lock = lambda: run_lock.__exit__(None, None, None)
    atexit.register(release_lock)
    status_path = root / "night_status.json"
    score_root = root / "scores"
    status = {
        **plan,
        "status": "RUNNING",
        "started_unix": time.time(),
        "completed_batches": [],
        "active_batch": None,
        "score_status": {},
    }
    atomic_json(status_path, status)
    try:
        for stage, model_id in batches:
            status["active_batch"] = {"stage": stage, "model_id": model_id, "started_unix": time.time()}
            atomic_json(status_path, status)
            command = [
                sys.executable, "-m",
                "experiments.fixed_rope_three_interfaces_20260913.resident_eval",
                "--queue", str(queue_path), "--model-id", model_id,
                "--stage", stage, "--execute",
            ]
            subprocess.run(command, cwd=contract["repo_root"], check=True)
            status["completed_batches"].append({
                "stage": stage, "model_id": model_id, "completed_unix": time.time(),
            })
            status["active_batch"] = None
            status["score_status"] = score_ready(
                contract_path, score_root, draws=args.bootstrap_draws,
            )
            atomic_json(status_path, status)
        refreshed = plan_queue(contract_path, queue_path.parent)
        coverage_after = queue_path.parent / "coverage_after.json"
        atomic_json(coverage_after, {"coverage": refreshed["coverage"]})
        status["coverage_after"] = str(coverage_after.resolve())
        status["remaining_jobs"] = len(refreshed["queue"])
        status["remaining_rows"] = sum(job["remaining_rows"] for job in refreshed["queue"])
        status["status"] = "COMPLETE" if not refreshed["queue"] else "INCOMPLETE_ROWS"
        status["completed_unix"] = time.time()
        atomic_json(status_path, status)
    except BaseException as error:
        status["status"] = "FAILED"
        status["failure"] = {"type": type(error).__name__, "message": str(error)}
        status["failed_unix"] = time.time()
        atomic_json(status_path, status)
        raise
    atexit.unregister(release_lock)
    release_lock()
    print(json.dumps({
        "status": status["status"], "remaining_rows": status.get("remaining_rows"),
        "status_path": str(status_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
