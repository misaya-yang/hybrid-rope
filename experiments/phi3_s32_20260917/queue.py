#!/usr/bin/env python3
"""Wait for the Phi-3 4K snapshot, repair its download, then run the gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_alive(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        os.kill(int(path.read_text().strip()), 0)
    except (OSError, ValueError):
        return False
    return True


def model_ready(model: Path) -> bool:
    index = model / "model.safetensors.index.json"
    if not index.is_file():
        return False
    payload = json.loads(index.read_text())
    shards = sorted(set(payload.get("weight_map", {}).values()))
    if not shards or any(not (model / name).is_file() for name in shards):
        return False
    # ``metadata.total_size`` is the tensor payload size, not the safetensors
    # file size (which also contains headers).  Validate each indexed shard by
    # opening it and matching its complete key set instead of comparing bytes.
    try:
        from safetensors import safe_open

        for name in shards:
            expected_keys = {
                key for key, shard in payload["weight_map"].items() if shard == name
            }
            with safe_open(model / name, framework="pt", device="cpu") as stream:
                observed_keys = set(stream.keys())
            if observed_keys != expected_keys:
                return False
    except (OSError, ValueError, RuntimeError):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel16", type=Path, required=True)
    parser.add_argument("--ruler", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--download-pid", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({
        "status": "PLAN_ONLY" if not args.execute else "WAITING_FOR_MODEL",
        "model_ready": model_ready(args.model),
        "poll_seconds": args.poll_seconds,
    }, sort_keys=True), flush=True)
    if not args.execute:
        return
    if args.poll_seconds < 10:
        raise ValueError("poll interval must be at least ten seconds")
    while not model_ready(args.model):
        if not process_alive(args.download_pid):
            command = [
                str(args.python), "-c",
                (
                    "from modelscope import snapshot_download; "
                    f"print(snapshot_download('LLM-Research/Phi-3-mini-4k-instruct', "
                    f"local_dir={str(args.model)!r}, max_workers=8))"
                ),
            ]
            process = subprocess.Popen(command)
            args.download_pid.write_text(str(process.pid) + "\n")
            process.wait()
            if process.returncode:
                time.sleep(args.poll_seconds)
        else:
            time.sleep(args.poll_seconds)
    command = [
        str(args.python), "-m", "experiments.phi3_s32_20260917.run_gate",
        "--repo", str(args.repo), "--root", str(args.root),
        "--model", str(args.model), "--panel16", str(args.panel16),
        "--ruler", str(args.ruler), "--python", str(args.python),
        "--shutdown-below-threshold", "--execute",
    ]
    raise SystemExit(subprocess.run(command, cwd=args.repo).returncode)


if __name__ == "__main__":
    main()
