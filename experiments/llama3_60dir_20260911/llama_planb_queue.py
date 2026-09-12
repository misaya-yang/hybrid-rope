"""Recoverable Llama-only Plan B bootstrap queue.

This controller closes the verified transition from GPU parity to the P pilot
and full P L0 controls.  It never launches the historical OLMo/Qwen queue.
Every child runs in the foreground of this controller, retries once under the
same contract, and writes resumable per-row outputs through ``llama_runner``.
"""
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
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def pid_alive(path):
    try:
        pid = int(Path(path).read_text().strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False


def manifest_complete(directory):
    path = Path(directory) / "manifest.json"
    try:
        return json.loads(path.read_text()).get("status") == "COMPLETE"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def run_checked(cmd, log_path, state, state_path):
    log_path = Path(log_path)
    for attempt in (1, 2):
        state.update(status="RUNNING", command=cmd, attempt=attempt,
                     started_at=time.time())
        atomic_json(state_path, state)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== attempt {attempt}: {' '.join(cmd)} ===\n")
            log.flush()
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            state.update(last_returncode=0, completed_at=time.time())
            atomic_json(state_path, state)
            return
        state.update(last_returncode=result.returncode, failed_at=time.time())
        atomic_json(state_path, state)
    raise RuntimeError(f"child failed twice: {' '.join(cmd)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/autodl-tmp/llama3_planb_20260911")
    ap.add_argument("--code", default="/root/autodl-tmp/llama3_60dir_20260911")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct")
    args = ap.parse_args(argv)

    root, code = Path(args.root), Path(args.code)
    py = "/root/miniconda3/bin/python"
    native = root / "native_inv_freq.npy"
    scorer = ("/root/autodl-tmp/olmo_fast_screen_20260908/code/scripts/"
              "experiments/olmo_fast_screen/ruler_bench.py")
    state_path = root / "queue_state.json"
    state = {"status": "STARTING", "scope": "Llama-3-only Plan B", "completed_units": []}
    atomic_json(state_path, state)

    # Do not race the already-running 8K extension parity probe.
    parity_pid = root / "gpu_extension_parity_8192.pid"
    while pid_alive(parity_pid):
        state.update(status="WAITING_FOR_8K_EXTENSION_PARITY", checked_at=time.time())
        atomic_json(state_path, state)
        time.sleep(2)

    pilot = root / "data" / "P_pilot"
    if not manifest_complete(pilot):
        # The 32K extension parity is useful target-length validation, not a
        # placeholder workload.  It also keeps the paid GPU productive while
        # the CPU-only upstream generators finish the first panel.
        extensions = ",".join(
            f"D{direction:02d}{policy}"
            for direction in [4, 5, *range(7, 21)] for policy in "abc")
        scopes = ("frequency,frequency_assignment,signed_frequency,dc_frequency,"
                  "spectral_amplitude,qk_phase_bias,qk_pair_metric,"
                  "position_amplitude,position_phase")
        run_checked([
            py, str(code / "gpu_core_parity.py"), "--model", args.model,
            "--native-npy", str(native), "--length", "32768", "--arms", extensions,
            "--authorized-scopes", scopes, "--out", str(root / "gpu_extension_parity_32768.json"),
        ], root / "gpu_extension_parity_32768.log", state, state_path)
        state["completed_units"].append("GPU_EXTENSION_PARITY_32768")

    while not manifest_complete(pilot):
        state.update(status="WAITING_FOR_P_PILOT_DATA", checked_at=time.time())
        atomic_json(state_path, state)
        time.sleep(2)

    runner = str(code / "llama_runner.py")

    def evaluate(unit, data_dir, output_dir, arms, lengths):
        panel = Path(data_dir) / "rows.jsonl"
        run_checked([
            py, runner, "--model", args.model, "--panel", str(panel),
            "--expected-data", str(panel), "--out", str(output_dir),
            "--arms", arms, "--lengths", lengths,
            "--native-npy", str(native), "--scorer", scorer,
        ], root / f"{unit}.log", state, state_path)
        state["completed_units"].append(unit)
        atomic_json(state_path, state)

    evaluate("P_PILOT_NATIVE", pilot, root / "results" / "P_pilot_native",
             "Native", "8192")
    evaluate("P_PILOT_L0", pilot, root / "results" / "P_pilot_l0",
             "MR,OfficialYaRN,BM,UNI", "8192,16384,32768")

    full_p = root / "data" / "P"
    while not manifest_complete(full_p):
        state.update(status="WAITING_FOR_FULL_P_DATA", checked_at=time.time())
        atomic_json(state_path, state)
        time.sleep(2)
    evaluate("P_NATIVE", full_p, root / "results" / "P_native",
             "Native", "8192")
    evaluate("P_L0", full_p, root / "results" / "P_l0",
             "MR,OfficialYaRN,BM,UNI", "8192,16384,32768")

    state.update(status="COMPLETE_P_L0", completed_at=time.time())
    atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
