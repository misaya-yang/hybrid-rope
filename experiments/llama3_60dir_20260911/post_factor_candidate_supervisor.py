"""Run the preregistered MR-area smooth range candidate after attribution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


CAPS = (8192, 16384, 24576, 32768, 40960, 49152)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def gpu_pids() -> list[int]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], text=True)
    return [int(line) for line in text.splitlines() if line.strip().isdigit()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wait-current", required=True, type=int)
    parser.add_argument(
        "--root", type=Path,
        default=Path("/root/autodl-tmp/llama3_mrrope_s16_20260911"))
    parser.add_argument(
        "--code", type=Path,
        default=Path("/root/autodl-tmp/llama3_60dir_20260911"))
    args = parser.parse_args(argv)

    root, code = args.root, args.code
    state_path = root / "range_candidate_state.json"
    state = {
        "status": "WAITING_SCALE_GAIN_FACTORIAL",
        "wait_current": args.wait_current,
        "started_at": time.time(),
        "candidate": "MR_AREA_SMOOTH",
        "hypothesis": (
            "MR-matched exponent area plus two-boundary-smooth increments "
            "retains part of BM's prefix benefit while repairing its 48K tail"
        ),
    }
    atomic_json(state_path, state)
    while alive(args.wait_current):
        state.update(checked_at=time.time(), current_alive=True)
        atomic_json(state_path, state)
        time.sleep(2)

    attribution = read_json(root / "scale_gain_state.json")
    report = root / "results" / "scale_gain_factorial" / "factorial_report.json"
    if attribution.get("status") != "COMPLETE" or not report.is_file():
        state.update(
            status="BLOCKED_FACTORIAL_INCOMPLETE", attribution=attribution,
            completed_at=time.time())
        atomic_json(state_path, state)
        return 3
    active = gpu_pids()
    if active:
        state.update(status="BLOCKED_GPU_BUSY", gpu_pids=active,
                     completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    panel = root / "data" / "anytime_s6_d_seed20260914" / "rows.jsonl"
    manifest = read_json(panel.with_name("manifest.json"))
    if manifest.get("status") != "COMPLETE" or manifest.get("rows") != 192:
        state.update(status="BLOCKED_PANEL", manifest=manifest,
                     completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    command = [
        "/root/miniconda3/bin/python", str(code / "llama_runner.py"),
        "--model", "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct",
        "--panel", str(panel), "--expected-data", str(panel),
        "--scale", "6", "--arms", "MR_AREA_SMOOTH",
        "--lengths", ",".join(map(str, CAPS)),
        "--native-npy",
        "/root/autodl-tmp/llama3_planb_20260911/native_inv_freq.npy",
        "--scorer", "/root/autodl-tmp/olmo_fast_screen_20260908/code/"
        "scripts/experiments/olmo_fast_screen/ruler_bench.py",
        "--authorized-scopes", "frequency",
        "--checkpoint-manifest", "/root/autodl-tmp/llama3_planb_20260911/"
        "validation/checkpoint_sha256.json",
        "--out", str(root / "results" / "anytime_s6_d" / "mr_area_smooth"),
    ]
    state.update(status="RUNNING_MR_AREA_SMOOTH_S6", command=command,
                 run_started_at=time.time(), factorial_report=str(report))
    atomic_json(state_path, state)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path = root / "logs" / "mr_area_smooth_s6.log"
    with log_path.open("a") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env)
        (root / "runner.pid").write_text(f"{child.pid}\n")
        return_code = child.wait()
    state.update(
        status="COMPLETE" if return_code == 0 else "BLOCKED_RUN",
        returncode=return_code, completed_at=time.time(),
        output=str(root / "results" / "anytime_s6_d" / "mr_area_smooth"),
    )
    atomic_json(state_path, state)
    return 0 if return_code == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
