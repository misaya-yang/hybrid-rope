"""Run the fixed MR frequency-scale x gain-scale attribution after coverage."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


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


def load(path: Path, cap: int) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {row["row_id"]: row for row in rows if int(row["length_cap"]) == cap}


def macro(rows: dict[str, dict]) -> float:
    by_task = defaultdict(list)
    for row in rows.values():
        by_task[row["task"]].append(float(row["partial_score"]))
    if not by_task:
        raise ValueError("empty result cell")
    return float(np.mean([np.mean(values) for values in by_task.values()]))


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
    state_path = root / "scale_gain_state.json"
    state = {"status": "WAITING_COVERAGE", "wait_current": args.wait_current,
             "started_at": time.time()}
    atomic_json(state_path, state)
    while alive(args.wait_current):
        state.update(checked_at=time.time(), current_alive=True)
        atomic_json(state_path, state)
        time.sleep(2)

    coverage_state = read_json(root / "anytime_scale_state.json")
    if coverage_state.get("status") != "COMPLETE":
        state.update(status="BLOCKED_COVERAGE_INCOMPLETE",
                     coverage_state=coverage_state, completed_at=time.time())
        atomic_json(state_path, state)
        return 3
    active = gpu_pids()
    if active:
        state.update(status="BLOCKED_GPU_BUSY", gpu_pids=active,
                     completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    panel = Path("/root/autodl-tmp/llama3_planb_20260911/data/S/rows.jsonl")
    common = [
        "/root/miniconda3/bin/python", str(code / "llama_runner.py"),
        "--model", "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct",
        "--panel", str(panel), "--expected-data", str(panel), "--scale", "16",
        "--native-npy", "/root/autodl-tmp/llama3_planb_20260911/native_inv_freq.npy",
        "--scorer", "/root/autodl-tmp/olmo_fast_screen_20260908/code/scripts/"
        "experiments/olmo_fast_screen/ruler_bench.py",
        "--authorized-scopes", "frequency",
        "--checkpoint-manifest", "/root/autodl-tmp/llama3_planb_20260911/"
        "validation/checkpoint_sha256.json",
    ]
    result_root = root / "results" / "scale_gain_factorial"
    commands = [
        ("SCALE_GAIN_NATIVE8K", common + [
            "--out", str(result_root / "native8k"),
            "--arms", "MR_FS16_GS1,MR_FS1_GS16", "--lengths", "8192"]),
        ("SCALE_GAIN_SUBRANGE", common + [
            "--out", str(result_root / "subrange16k32k"),
            "--arms", "MR_FS16_GS4,MR_FS4_GS16",
            "--lengths", "16384,32768"]),
    ]
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    for label, command in commands:
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
        if return_code != 0:
            state.update(status=f"BLOCKED_{label}", completed_at=time.time())
            atomic_json(state_path, state)
            return 3

    planb = Path("/root/autodl-tmp/llama3_planb_20260911/results/S")
    existing = {
        (8192, "A_f16_g16"): root / "results/s16_8k32k/MR.jsonl",
        (8192, "B_f16_g1"): result_root / "native8k/MR_FS16_GS1.jsonl",
        (8192, "C_f1_g16"): result_root / "native8k/MR_FS1_GS16.jsonl",
        (8192, "D_f1_g1"): planb / "Native/Native.jsonl",
        (16384, "A_f16_g16"): root / "results/s16_16k/MR.jsonl",
        (16384, "B_f16_g4"): result_root / "subrange16k32k/MR_FS16_GS4.jsonl",
        (16384, "C_f4_g16"): result_root / "subrange16k32k/MR_FS4_GS16.jsonl",
        (16384, "D_f4_g4"): planb / "MR/MR.jsonl",
        (32768, "A_f16_g16"): root / "results/s16_8k32k/MR.jsonl",
        (32768, "B_f16_g4"): result_root / "subrange16k32k/MR_FS16_GS4.jsonl",
        (32768, "C_f4_g16"): result_root / "subrange16k32k/MR_FS4_GS16.jsonl",
        (32768, "D_f4_g4"): planb / "MR/MR.jsonl",
    }
    report = {"cells": {}, "effects": {}}
    for cap in (8192, 16384, 32768):
        cells = {name: load(path, cap) for (cell_cap, name), path in existing.items()
                 if cell_cap == cap}
        row_sets = [set(rows) for rows in cells.values()]
        if any(rows != row_sets[0] for rows in row_sets[1:]):
            raise ValueError(f"factorial row mismatch at {cap}")
        scores = {name: macro(rows) for name, rows in cells.items()}
        report["cells"][str(cap)] = scores
        names = sorted(scores)
        a, b, c, d = (scores[name] for name in names)
        report["effects"][str(cap)] = {
            "frequency_effect_at_high_gain_A_minus_C": a - c,
            "frequency_effect_at_gentle_gain_B_minus_D": b - d,
            "gain_effect_at_extreme_frequency_A_minus_B": a - b,
            "gain_effect_at_gentle_frequency_C_minus_D": c - d,
            "interaction_A_minus_B_minus_C_plus_D": a - b - c + d,
        }
    atomic_json(result_root / "factorial_report.json", report)
    state.update(status="COMPLETE", completed_at=time.time(),
                 report=str(result_root / "factorial_report.json"))
    atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
