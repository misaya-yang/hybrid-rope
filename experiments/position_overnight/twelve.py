"""Print or execute one of twelve PC2/PM experiments under the shared GPU lock."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def command(a):
    root = Path(a.root)
    pc = a.id <= 6
    split = "test" if a.id in (6, 12) else "dev"
    output = root / "runs" / f"e{a.id:02d}_{a.tag}"
    if pc:
        method = {1: "pc2", 2: "pc2_unweighted", 3: "pc2_rank1", 4: "pc2"}.get(a.id, a.pc2_method)
        if method is None:
            raise ValueError("E05/E06 require --pc2-method selected on DEV")
        selectors = ["native", "cobs_rank2", "split2", method]
        if a.id == 4:
            selectors = ["weighted_mean", "pc2"]
        cmd = [sys.executable, "-m", "experiments.nosa_position.run",
               "--model", "/root/autodl-tmp/NOSA-1B", "--data", str(root / "data/pc2/rows.jsonl"),
               "--output", str(output), "--baseline-cache", str(root / "baselines/pc2"),
               "--split", split, "--lengths", "16384", "--tasks",
               "niah_single_1", "niah_multikey_1", "niah_multiquery", "vt",
               "--selectors", *selectors, "--topk", str(48 if a.id == 5 else 64),
               "--chunk-size", str(a.chunk_size),
               "--attention-query-chunk-size", str(a.query_chunk)]
        if a.chunk_size == 128 and a.query_chunk == 64:
            cmd += ["--baseline-attention-query-chunk-size", "16"]
        if split == "dev" and a.per_task:
            cmd += ["--per-cell", str(a.per_task)]
    else:
        policy, horizon = {7: ("uniform_prefix", 512), 8: ("recent_prefix", 512),
                           9: ("uniform_prefix", 128), 10: ("recent_prefix", 128)}.get(
                               a.id, (a.pm_policy, a.pm_horizon))
        if policy is None or horizon is None:
            raise ValueError("E11/E12 require --pm-policy and --pm-horizon selected on DEV")
        cmd = [sys.executable, "-m", "experiments.pm_keep.run", "--model",
               str(root / "runs/pm_gpu_ready_20260909_v3/model_view"),
               "--data", str(root / "data/pm_keep/rows.jsonl"), "--output", str(output),
               "--baseline-cache", str(root / "baselines/pm_keep"), "--split", split,
               "--arms", "F", "E", "E_author_policy", "K", "P", "C", "U",
               "--query-policy", policy, "--horizon", str(horizon),
               "--keep-fraction", str(.125 if a.id == 11 else .25)]
        if split == "dev" and a.per_task:
            cmd += ["--per-task", str(a.per_task)]
    if a.reuse_from:
        cmd += ["--reuse-from", *a.reuse_from]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id", required=True, type=int, choices=range(1, 13))
    p.add_argument("--root", default="/root/autodl-tmp/position_overnight_20260909")
    p.add_argument("--tag", default="v1")
    p.add_argument("--per-task", type=int, default=4, help="0 expands DEV; TEST always uses full split")
    p.add_argument("--pc2-method", choices=("pc2", "pc2_unweighted", "pc2_rank1"))
    p.add_argument("--pm-policy", choices=("uniform_prefix", "recent_prefix"))
    p.add_argument("--pm-horizon", type=int, choices=(128, 512))
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--query-chunk", type=int, default=64)
    p.add_argument("--reuse-from", nargs="+", default=[], help="Explicit matching candidate runs to reuse")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--wait-for-lock", action="store_true", help="Queue behind the current GPU owner")
    a = p.parse_args()
    try:
        cmd = command(a)
    except ValueError as e:
        p.error(str(e))
    print(json.dumps({"experiment": a.id, "argv": cmd}, ensure_ascii=False), flush=True)
    if not a.execute:
        return
    root = Path(a.root)
    state_path = root / f"queue_e{a.id:02d}_{a.tag}.json"
    state = {"status": "WAITING", "pid": os.getpid(), "experiment": a.id, "command": cmd,
             "started_at": time.time()}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if a.wait_for_lock else fcntl.LOCK_NB))
        if (root / "STOP").exists():
            state.update(status="STOPPED_BY_USER")
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise SystemExit("User STOP exists; not launching")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "PM_KEEP_KVPRESS_ROOT": str(root / "vendor/kvpress")}
        process = subprocess.Popen(cmd, cwd=root / "code", env=env)
        state.update(status="RUNNING", child_pid=process.pid)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        code = process.wait()
        state.update(status="COMPLETE" if code == 0 else "NEEDS_REPAIR", exit_code=code,
                     finished_at=time.time())
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        raise SystemExit(code)


if __name__ == "__main__":
    main()
