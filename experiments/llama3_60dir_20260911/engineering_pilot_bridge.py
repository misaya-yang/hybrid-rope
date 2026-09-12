"""Run a bounded real-generation integration pilot, then resume the main queue."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return ""


def atomic_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queue-pid", required=True, type=int)
    ap.add_argument("--arms", default="D01a,D13a")
    ap.add_argument("--root", default="/root/autodl-tmp/llama3_planb_20260911")
    ap.add_argument("--code", default="/root/autodl-tmp/llama3_60dir_20260911")
    args = ap.parse_args(argv)
    root, code = Path(args.root), Path(args.code)
    if "llama_planb_queue.py" not in cmdline(args.queue_pid):
        raise SystemExit("REFUSING: queue PID identity mismatch")
    state_path = root / "engineering_pilot_bridge_state.json"
    os.kill(args.queue_pid, signal.SIGSTOP)
    try:
        panel = root / "data" / "P_pilot" / "rows.jsonl"
        command = [
            "/root/miniconda3/bin/python", str(code / "llama_runner.py"),
            "--model", "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct",
            "--panel", str(panel), "--expected-data", str(panel),
            "--out", str(root / "results" / "P_pilot_engineering_candidates"),
            "--arms", args.arms, "--lengths", "8192,16384,32768",
            "--native-npy", str(root / "native_inv_freq.npy"),
            "--scorer", ("/root/autodl-tmp/olmo_fast_screen_20260908/code/"
                         "scripts/experiments/olmo_fast_screen/ruler_bench.py"),
            "--authorized-scopes", "frequency,position_amplitude",
        ]
        state = {"status": "RUNNING", "role": "ENGINEERING_INTEGRATION_ONLY",
                 "claim_scope": "not S/V/H selection evidence", "command": command,
                 "started_at": time.time()}
        atomic_json(state_path, state)
        with (root / "engineering_pilot_bridge.log").open("a") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        state.update(status="COMPLETE" if result.returncode == 0 else "FAILED",
                     returncode=result.returncode, completed_at=time.time())
        atomic_json(state_path, state)
        return result.returncode
    finally:
        if "llama_planb_queue.py" in cmdline(args.queue_pid):
            os.kill(args.queue_pid, signal.SIGCONT)


if __name__ == "__main__":
    raise SystemExit(main())
