"""Keep the paid GPU useful while the full engineering pilot is prepared.

The already-started 32K operator-parity child is allowed to finish while its
queue controller is paused.  Then one frozen 8K row is evaluated with Native
and MR through the real generation/scorer path.  The main queue is resumed in
a ``finally`` block, whether the integration canary passes or fails.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process_cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return ""


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/autodl-tmp/llama3_planb_20260911")
    ap.add_argument("--code", default="/root/autodl-tmp/llama3_60dir_20260911")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--queue-pid", required=True, type=int)
    ap.add_argument("--parity-child-pid", type=int,
                    help="active parity child; omit when parity has already completed")
    args = ap.parse_args(argv)

    root, code = Path(args.root), Path(args.code)
    state_path = root / "integration_canary_state.json"
    if "llama_planb_queue.py" not in process_cmdline(args.queue_pid):
        raise SystemExit("REFUSING: queue PID identity mismatch")
    if args.parity_child_pid is not None and \
            "gpu_core_parity.py" not in process_cmdline(args.parity_child_pid):
        raise SystemExit("REFUSING: parity child PID identity mismatch")

    os.kill(args.queue_pid, signal.SIGSTOP)
    state = {"status": "WAITING_FOR_PARITY", "queue_pid": args.queue_pid,
             "parity_child_pid": args.parity_child_pid, "started_at": time.time()}
    atomic_json(state_path, state)
    try:
        if args.parity_child_pid is not None:
            while process_cmdline(args.parity_child_pid):
                time.sleep(0.5)
        partial = root / "data" / "P_pilot" / "rows.partial.jsonl"
        final = root / "data" / "P_pilot" / "rows.jsonl"
        while not ((partial.exists() and partial.read_text().strip()) or
                   (final.exists() and final.read_text().strip())):
            state.update(status="WAITING_FOR_FIRST_FROZEN_ROW", checked_at=time.time())
            atomic_json(state_path, state)
            time.sleep(0.5)
        source = final if final.exists() else partial
        rows = [json.loads(line) for line in source.read_text().splitlines() if line]
        row = next((r for r in rows if int(r["length_cap"]) == 8192), None)
        if row is None:
            raise RuntimeError("partial panel has no 8K row")
        panel = root / "data" / "integration_canary_row.jsonl"
        panel.write_text(json.dumps(row) + "\n")
        cmd = [
            "/root/miniconda3/bin/python", str(code / "llama_runner.py"),
            "--model", args.model, "--panel", str(panel),
            "--out", str(root / "results" / "integration_canary"),
            "--arms", "Native,MR",
            "--native-npy", str(root / "native_inv_freq.npy"),
            "--scorer", ("/root/autodl-tmp/olmo_fast_screen_20260908/code/"
                         "scripts/experiments/olmo_fast_screen/ruler_bench.py"),
        ]
        state.update(status="RUNNING_REAL_GENERATION", command=cmd, started_generation_at=time.time())
        atomic_json(state_path, state)
        with (root / "integration_canary.log").open("a") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        state.update(status="COMPLETE" if result.returncode == 0 else "FAILED",
                     returncode=result.returncode, completed_at=time.time())
        atomic_json(state_path, state)
        return result.returncode
    finally:
        if "llama_planb_queue.py" in process_cmdline(args.queue_pid):
            os.kill(args.queue_pid, signal.SIGCONT)


if __name__ == "__main__":
    raise SystemExit(main())
