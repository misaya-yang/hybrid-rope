"""Bridge the running P bootstrap queue into the resumable Llama-only S queue."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def pid_alive(pid, needle):
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return False
    return needle in cmd


def find_unique_pid(needle):
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle in command:
            found.append(int(proc.name))
    if len(found) != 1:
        raise RuntimeError(f"expected one {needle} process, found {found}")
    return found[0]


def run_logged(command, log_path):
    with Path(log_path).open("a", encoding="utf-8") as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/autodl-tmp/llama3_planb_20260911")
    ap.add_argument("--code", default="/root/autodl-tmp/llama3_60dir_20260911")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--scorer", default=(
        "/root/autodl-tmp/olmo_fast_screen_20260908/code/scripts/"
        "experiments/olmo_fast_screen/ruler_bench.py"))
    args = ap.parse_args(argv)
    root, code = Path(args.root), Path(args.code)
    state_path = root / "s_supervisor_state.json"
    pid_path = root / "queue.pid"
    p_pid = (int(pid_path.read_text().strip()) if pid_path.exists()
             else find_unique_pid("llama_planb_queue.py"))
    state = {"status": "WAITING_FOR_P_L0", "p_queue_pid": p_pid,
             "scope": "Meta-Llama-3-8B-Instruct only", "started_at": time.time()}
    atomic_json(state_path, state)

    while True:
        p_state = read_json(root / "queue_state.json")
        if p_state.get("status") == "COMPLETE_P_L0":
            break
        if not pid_alive(p_pid, "llama_planb_queue.py"):
            state.update(status="BLOCKED_P_QUEUE_EXITED",
                         observed_p_status=p_state.get("status"), updated_at=time.time())
            atomic_json(state_path, state)
            return 3
        state.update(observed_p_status=p_state.get("status"), checked_at=time.time())
        atomic_json(state_path, state)
        time.sleep(2)

    validation = root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = validation / "checkpoint_sha256.json"
    while read_json(checkpoint_manifest).get("status") != "COMPLETE":
        hash_pid_path = root / "checkpoint_hash.pid"
        hash_pid = int(hash_pid_path.read_text()) if hash_pid_path.exists() else -1
        if hash_pid < 0 or not pid_alive(hash_pid, "hash_checkpoint.py"):
            state.update(status="BLOCKED_CHECKPOINT_IDENTITY", completed_at=time.time())
            atomic_json(state_path, state)
            return 3
        state.update(status="WAITING_FOR_CHECKPOINT_IDENTITY", checked_at=time.time())
        atomic_json(state_path, state)
        time.sleep(2)
    repair_command = [
        "/root/miniconda3/bin/python", str(code / "repair_panel_source_ids.py"),
        "--root", str(root), "--stages", "P,S", "--repair-results", "P",
        "--out", str(validation / "source_identity_repair.json"),
    ]
    state.update(status="REPAIRING_AND_VALIDATING_DATA",
                 repair_command=repair_command, repair_started_at=time.time())
    atomic_json(state_path, state)
    if run_logged(repair_command, root / "llama_s_supervisor.log") != 0:
        state.update(status="BLOCKED_DATA_REPAIR", completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    validate_command = [
        "/root/miniconda3/bin/python", str(code / "validate_planb_data.py"),
        "--root", str(root), "--stages", "P,S",
        "--out", str(validation / "P_S_data_validation.json"),
    ]
    if run_logged(validate_command, root / "llama_s_supervisor.log") != 0:
        state.update(status="BLOCKED_DATA_VALIDATION", completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    readiness_command = [
        "/root/miniconda3/bin/python", str(code / "p_readiness_report.py"),
        "--root", str(root),
        "--out", str(validation / "P_readiness_report.json"),
    ]
    if run_logged(readiness_command, root / "llama_s_supervisor.log") != 0:
        state.update(status="BLOCKED_P_INSTRUMENT", completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    dry_run_command = [
        "/root/miniconda3/bin/python", str(code / "llama_svh_queue.py"),
        "--code", str(code), "--dry-run",
    ]
    with (validation / "s_queue_dry_run.json").open("w") as stream:
        dry = subprocess.run(dry_run_command, stdout=stream, stderr=subprocess.PIPE,
                             text=True)
    if dry.returncode != 0:
        state.update(status="BLOCKED_S_DRY_RUN", dry_run_stderr=dry.stderr,
                     completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    command = [
        "/root/miniconda3/bin/python", str(code / "llama_svh_queue.py"),
        "--root", str(root), "--code", str(code), "--model", args.model,
        "--scorer", args.scorer, "--execute",
    ]
    state.update(status="RUNNING_S", command=command, s_started_at=time.time())
    atomic_json(state_path, state)
    with (root / "llama_svh_queue.log").open("a") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    final = read_json(root / "sv_queue_state.json")
    state.update(status="COMPLETE_S" if result.returncode == 0 else "BLOCKED_S",
                 returncode=result.returncode, s_status=final.get("status"),
                 completed_at=time.time())
    atomic_json(state_path, state)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
