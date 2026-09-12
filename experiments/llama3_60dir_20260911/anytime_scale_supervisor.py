"""Finish the MR scale-policy matrix, then run the fixed-S=6 anytime curve.

This supervisor waits for the already-running Qwen confirmation. It launches
only preregistered Llama comparisons and never selects a method from outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


TASKS = (
    "niah_single_2", "niah_multikey_2", "niah_multivalue",
    "niah_multiquery", "vt", "fwe", "qa_1", "qa_2",
)
ANYTIME_CAPS = (8192, 16384, 24576, 32768, 40960, 49152)


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
    parser.add_argument("--wait-current", type=int, required=True)
    parser.add_argument(
        "--root", type=Path,
        default=Path("/root/autodl-tmp/llama3_mrrope_s16_20260911"))
    parser.add_argument(
        "--code", type=Path,
        default=Path("/root/autodl-tmp/llama3_60dir_20260911"))
    parser.add_argument(
        "--qwen-out", type=Path,
        default=Path("/root/autodl-tmp/bm_transfer_20260908/"
                     "run_qwen3_s2_confirm3_20260912"))
    args = parser.parse_args(argv)

    root, code = args.root, args.code
    state_path = root / "anytime_scale_state.json"
    state = {
        "status": "WAITING_QWEN", "wait_current": args.wait_current,
        "started_at": time.time(),
    }
    atomic_json(state_path, state)
    while alive(args.wait_current):
        state.update(checked_at=time.time(), current_alive=True)
        atomic_json(state_path, state)
        time.sleep(2)

    qwen = read_json(args.qwen_out / "decision.json")
    required_qwen = ["MrPro.json", "MrProBM.json", "MrProUni.json"]
    if (qwen.get("status") != "QUEUE_COMPLETE" or
            not all((args.qwen_out / name).is_file() for name in required_qwen)):
        state.update(status="BLOCKED_QWEN_INCOMPLETE", qwen=qwen,
                     completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    if gpu_pids():
        state.update(status="BLOCKED_GPU_BUSY", gpu_pids=gpu_pids(),
                     completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    s_panel = Path("/root/autodl-tmp/llama3_planb_20260911/data/S")
    s_manifest = read_json(s_panel / "manifest.json")
    if s_manifest.get("status") != "COMPLETE" or s_manifest.get("rows") != 160:
        state.update(status="BLOCKED_S_PANEL", manifest=s_manifest,
                     completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    anytime = root / "data" / "anytime_s6_d_seed20260914"
    manifest = read_json(anytime / "manifest.json")
    expected_cells = {f"{task}|{cap}": 4 for cap in ANYTIME_CAPS for task in TASKS}
    if (manifest.get("status") != "COMPLETE" or manifest.get("rows") != 192 or
            manifest.get("caps") != list(ANYTIME_CAPS) or
            manifest.get("cell_counts") != expected_cells):
        state.update(status="BLOCKED_ANYTIME_PANEL", manifest=manifest,
                     completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    common = [
        "/root/miniconda3/bin/python", str(code / "llama_runner.py"),
        "--model", "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct",
        "--native-npy", "/root/autodl-tmp/llama3_planb_20260911/native_inv_freq.npy",
        "--scorer", "/root/autodl-tmp/olmo_fast_screen_20260908/code/scripts/"
        "experiments/olmo_fast_screen/ruler_bench.py",
        "--authorized-scopes", "frequency",
        "--checkpoint-manifest", "/root/autodl-tmp/llama3_planb_20260911/"
        "validation/checkpoint_sha256.json",
    ]
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def run(label: str, command: list[str]) -> int:
        state.update(status=f"RUNNING_{label}", command=command,
                     run_started_at=time.time())
        atomic_json(state_path, state)
        with (logs / f"{label.lower()}.log").open("a") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env)
            (root / "runner.pid").write_text(f"{child.pid}\n")
            return_code = child.wait()
        state.update(last_label=label, last_returncode=return_code,
                     last_completed_at=time.time())
        atomic_json(state_path, state)
        return return_code

    matrix_root = root / "results" / "mr_scale_matrix"
    commands = [
        ("MR_SCALE2_8K", common + [
            "--panel", str(s_panel / "rows.jsonl"),
            "--expected-data", str(s_panel / "rows.jsonl"),
            "--out", str(matrix_root / "s2_8k"), "--arms", "MR",
            "--lengths", "8192", "--scale", "2"]),
        ("MR_SCALE8_8K32K", common + [
            "--panel", str(s_panel / "rows.jsonl"),
            "--expected-data", str(s_panel / "rows.jsonl"),
            "--out", str(matrix_root / "s8_8k32k"), "--arms", "MR",
            "--lengths", "8192,16384,32768", "--scale", "8"]),
        ("ANYTIME_S6_NATIVE", common + [
            "--panel", str(anytime / "rows.jsonl"),
            "--expected-data", str(anytime / "rows.jsonl"),
            "--out", str(root / "results" / "anytime_s6_d" / "native8k"),
            "--arms", "Native", "--lengths", "8192", "--scale", "6"]),
        ("ANYTIME_S6_FIXED", common + [
            "--panel", str(anytime / "rows.jsonl"),
            "--expected-data", str(anytime / "rows.jsonl"),
            "--out", str(root / "results" / "anytime_s6_d" / "fixed_s6"),
            "--arms", "MR,OfficialYaRN,BM",
            "--lengths", ",".join(map(str, ANYTIME_CAPS)), "--scale", "6"]),
    ]
    for label, command in commands:
        if run(label, command) != 0:
            state.update(status=f"BLOCKED_{label}", completed_at=time.time())
            atomic_json(state_path, state)
            return 3

    report_command = [
        "/root/miniconda3/bin/python", str(code / "coverage_report.py"),
        "--out", str(root / "results" / "anytime_s6_d" / "coverage_report.json"),
    ]
    state.update(status="BUILDING_REPORT", command=report_command,
                 report_started_at=time.time())
    atomic_json(state_path, state)
    report = subprocess.run(report_command, env=env)
    if report.returncode != 0:
        state.update(status="BLOCKED_REPORT", returncode=report.returncode,
                     completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    state.update(status="COMPLETE", completed_at=time.time())
    atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
