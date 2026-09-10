"""Launch the two prepared research cores; Sol owns subsequent improvement.

There is no elapsed-time limit, watchdog kill, automatic shutdown, or new GPU
rental. Fixed control results are reused. Change --tag when editing a method.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/root/autodl-tmp/position_overnight_20260909")
    p.add_argument("--core", choices=("both", "pc2", "pm"), default="both")
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--per-task", type=int, default=4, help="0 selects the full fixed split")
    p.add_argument("--tag", default="initial")
    p.add_argument("--pc2-method", default="pc2", choices=("pc2", "pc2_unweighted", "pc2_rank1"))
    p.add_argument("--pm-query-policy", default="uniform_prefix", choices=("uniform_prefix", "recent_prefix"))
    p.add_argument("--pm-horizon", type=int, default=512)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = Path(args.root)
    model_view = root / "runs/pm_gpu_ready_20260909_v3/model_view"
    pc2 = [sys.executable, "-m", "experiments.nosa_position.run", "--model", "/root/autodl-tmp/NOSA-1B",
           "--data", str(root / "data/pc2/rows.jsonl"), "--output", str(root / f"runs/pc2_{args.split}_{args.tag}"),
           "--baseline-cache", str(root / "baselines/pc2"), "--split", args.split, "--lengths", "16384",
           "--tasks", "niah_single_1", "niah_multikey_1", "niah_multiquery", "vt",
           "--selectors", "native", "cobs_rank2", "split2", args.pc2_method,
           "--attention-query-chunk-size", "64", "--baseline-attention-query-chunk-size", "16"]
    pm = [sys.executable, "-m", "experiments.pm_keep.run", "--model", str(model_view),
          "--data", str(root / "data/pm_keep/rows.jsonl"), "--output", str(root / f"runs/pm_{args.split}_{args.tag}"),
          "--baseline-cache", str(root / "baselines/pm_keep"), "--split", args.split,
          "--arms", "F", "E", "E_author_policy", "C", "P", "U", "K",
          "--query-policy", args.pm_query_policy, "--horizon", str(args.pm_horizon)]
    if args.per_task:
        pc2 += ["--per-cell", str(args.per_task)]
        pm += ["--per-task", str(args.per_task)]
    commands = [("pc2", pc2), ("pm", pm)]
    if args.core != "both":
        commands = [(name, cmd) for name, cmd in commands if name == args.core]
    for name, command in commands:
        print(json.dumps({"core": name, "argv": command, "time_limit": None, "automatic_shutdown": False}), flush=True)
    if args.dry_run:
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "PM_KEEP_KVPRESS_ROOT": str(root / "vendor/kvpress")}
    status = {"status": "RUNNING", "pid": os.getpid(), "started_at": time.time(), "commands": commands,
              "finished": [], "time_limit": None, "automatic_shutdown": False}
    path = root / f"queue_{args.tag}.json"
    for name, command in commands:
        if (root / "STOP").exists():
            status["status"] = "STOPPED_BY_USER"
            break
        status["current_core"] = name
        path.write_text(json.dumps(status, indent=2) + "\n")
        log_path = root / f"runs/{name}_{args.split}_{args.tag}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, cwd=root / "code", env=env, stdout=log, stderr=subprocess.STDOUT)
            status["child_pid"] = process.pid
            path.write_text(json.dumps(status, indent=2) + "\n")
            code = process.wait()
        status["finished"].append({"core": name, "exit_code": code, "log": str(log_path)})
        # A failed core is reported, while independent work on the other core
        # remains useful. Sol reads the concrete exception and repairs it.
        path.write_text(json.dumps(status, indent=2) + "\n")
    if status["status"] == "RUNNING":
        status["status"] = "COMPLETE" if all(x["exit_code"] == 0 for x in status["finished"]) else "NEEDS_REPAIR"
    status["finished_at"] = time.time()
    path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
