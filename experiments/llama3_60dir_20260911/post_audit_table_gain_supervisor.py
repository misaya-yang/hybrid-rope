"""Launch MR/BM unit-gain endpoint controls after current-runner parity passes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def rows(path: Path, cap: int) -> dict[str, dict]:
    return {
        row["row_id"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line)
        if int(row["length_cap"]) == cap
    }


def gpu_pids() -> list[int]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], text=True)
    return [int(line) for line in text.splitlines() if line.strip().isdigit()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-current", required=True, type=int)
    parser.add_argument(
        "--root", type=Path,
        default=Path("/root/autodl-tmp/llama3_mrrope_s16_20260911"))
    parser.add_argument(
        "--code", type=Path,
        default=Path("/root/autodl-tmp/llama3_60dir_20260911"))
    args = parser.parse_args(argv)
    root, code = args.root, args.code
    state_path = root / "table_gain_state.json"
    state = {"status": "WAITING_CURRENT_RUNNER_AUDIT",
             "wait_current": args.wait_current, "started_at": time.time()}
    atomic_json(state_path, state)
    while Path(f"/proc/{args.wait_current}").exists():
        state.update(checked_at=time.time(), current_alive=True)
        atomic_json(state_path, state)
        time.sleep(2)

    audit = root / "results" / "anytime_s6_d" / "current_runner_48k_audit"
    old = root / "results" / "anytime_s6_d" / "fixed_s6"
    summary_path = audit / "run_summary.json"
    try:
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "COMPLETE":
            raise ValueError("current-runner audit summary is incomplete")
        parity = {}
        identity_fields = (
            "prompt_sha256", "source_id", "task", "length_cap", "references",
            "max_new_tokens", "input_tokens", "operator_sha256",
            "checkpoint_manifest_sha256", "scorer_sha256",
            "scoring_contract_sha256",
        )
        for arm in ("MR", "BM"):
            before = rows(old / f"{arm}.jsonl", 49152)
            after = rows(audit / f"{arm}.jsonl", 49152)
            if before.keys() != after.keys() or len(before) != 32:
                raise ValueError(f"{arm} row identity mismatch")
            for row_id in before:
                for field in identity_fields:
                    if before[row_id].get(field) != after[row_id].get(field):
                        raise ValueError(f"{arm} {row_id} field drift: {field}")
                if float(before[row_id]["partial_score"]) != float(
                        after[row_id]["partial_score"]):
                    raise ValueError(f"{arm} {row_id} score drift")
            parity[arm] = {
                "rows": len(before),
                "partial_score_sum": sum(
                    float(row["partial_score"]) for row in after.values()),
                "old_runner_sha256": next(iter(before.values()))["runner_sha256"],
                "new_runner_sha256": next(iter(after.values()))["runner_sha256"],
            }
    except Exception as error:
        state.update(status="BLOCKED_AUDIT_PARITY", error=str(error),
                     completed_at=time.time())
        atomic_json(state_path, state)
        return 3
    active = gpu_pids()
    if active:
        state.update(status="BLOCKED_GPU_BUSY", gpu_pids=active,
                     parity=parity, completed_at=time.time())
        atomic_json(state_path, state)
        return 3

    panel = root / "data" / "anytime_s6_d_seed20260914" / "rows.jsonl"
    output = root / "results" / "anytime_s6_d" / "table_gain_endpoints"
    command = [
        "/root/miniconda3/bin/python", str(code / "llama_runner.py"),
        "--model", "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct",
        "--panel", str(panel), "--expected-data", str(panel),
        "--scale", "6", "--arms", "MR_G1,BM_G1",
        "--lengths", "8192,49152",
        "--native-npy",
        "/root/autodl-tmp/llama3_planb_20260911/native_inv_freq.npy",
        "--scorer", "/root/autodl-tmp/olmo_fast_screen_20260908/code/"
        "scripts/experiments/olmo_fast_screen/ruler_bench.py",
        "--authorized-scopes", "frequency",
        "--checkpoint-manifest", "/root/autodl-tmp/llama3_planb_20260911/"
        "validation/checkpoint_sha256.json",
        "--out", str(output),
    ]
    state.update(status="RUNNING_TABLE_GAIN_ENDPOINTS", parity=parity,
                 command=command, run_started_at=time.time())
    atomic_json(state_path, state)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with (root / "logs" / "table_gain_endpoints.log").open("a") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env)
        (root / "runner.pid").write_text(f"{child.pid}\n")
        return_code = child.wait()
    state.update(
        status="COMPLETE" if return_code == 0 else "BLOCKED_RUN",
        returncode=return_code, output=str(output), completed_at=time.time())
    atomic_json(state_path, state)
    return 0 if return_code == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
