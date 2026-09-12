"""Promote the paper-faithful s=16 comparison to a compact 64K panel."""
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


def alive(pid):
    return Path(f"/proc/{pid}").exists()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wait-current", type=int, required=True)
    ap.add_argument("--wait-prep", type=int, required=True)
    ap.add_argument("--root", type=Path,
                    default=Path("/root/autodl-tmp/llama3_mrrope_s16_20260911"))
    ap.add_argument("--code", type=Path,
                    default=Path("/root/autodl-tmp/llama3_60dir_20260911"))
    ap.add_argument("--model", type=Path,
                    default=Path("/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"))
    ap.add_argument("--scorer", type=Path, default=Path(
        "/root/autodl-tmp/olmo_fast_screen_20260908/code/scripts/experiments/"
        "olmo_fast_screen/ruler_bench.py"))
    args = ap.parse_args(argv)

    root, code = args.root, args.code
    data = root / "data" / "64k_compact"
    state_path = root / "paper_s16_64k_state.json"
    state = {"status": "WAITING", "wait_current": args.wait_current,
             "wait_prep": args.wait_prep, "started_at": time.time()}
    atomic_json(state_path, state)
    while alive(args.wait_current) or alive(args.wait_prep):
        state.update(current_alive=alive(args.wait_current),
                     prep_alive=alive(args.wait_prep), checked_at=time.time())
        atomic_json(state_path, state)
        time.sleep(2)

    prior = read_json(root / "results" / "s16_16k" / "run_summary.json")
    manifest = read_json(data / "manifest.json")
    if prior.get("status") != "COMPLETE":
        state.update(status="BLOCKED_PRIOR_RUN", prior=prior, completed_at=time.time())
        atomic_json(state_path, state)
        return 3
    expected_cells = {f"{task}|65536": 4 for task in (
        "niah_single_2", "niah_multikey_2", "niah_multivalue",
        "niah_multiquery", "vt", "fwe", "qa_1", "qa_2")}
    if (manifest.get("status") != "COMPLETE" or manifest.get("rows") != 32 or
            manifest.get("caps") != [65536] or
            manifest.get("cell_counts") != expected_cells):
        state.update(status="BLOCKED_PANEL", manifest=manifest,
                     completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    common = [
        "/root/miniconda3/bin/python", str(code / "llama_runner.py"),
        "--model", str(args.model), "--panel", str(data / "rows.jsonl"),
        "--scale", "16", "--native-npy",
        "/root/autodl-tmp/llama3_planb_20260911/native_inv_freq.npy",
        "--scorer", str(args.scorer), "--authorized-scopes", "frequency",
        "--checkpoint-manifest",
        "/root/autodl-tmp/llama3_planb_20260911/validation/checkpoint_sha256.json",
    ]
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def run(label, command):
        state.update(status=f"RUNNING_{label}", command=command,
                     run_started_at=time.time())
        atomic_json(state_path, state)
        with (logs / f"{label.lower()}.log").open("a") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env)
            (root / "runner.pid").write_text(f"{child.pid}\n")
            rc = child.wait()
        state.update(last_label=label, last_returncode=rc,
                     last_completed_at=time.time())
        atomic_json(state_path, state)
        return rc

    probe = common + ["--out", str(root / "results" / "s16_64k_probe"),
                      "--arms", "BM", "--limit", "1"]
    if run("64K_PROBE", probe) != 0:
        state.update(status="BLOCKED_64K_FEASIBILITY", completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    full = common + ["--expected-data", str(data / "rows.jsonl"),
                     "--out", str(root / "results" / "s16_64k_compact"),
                     "--arms", "MR,BM,OfficialYaRN"]
    if run("64K_COMPACT", full) != 0:
        state.update(status="BLOCKED_64K_RUN", completed_at=time.time())
        atomic_json(state_path, state)
        return 3
    state.update(status="COMPLETE", completed_at=time.time())
    atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
