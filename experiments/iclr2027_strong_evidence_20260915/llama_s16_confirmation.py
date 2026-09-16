#!/usr/bin/env python3
"""Prepare/run four fixed Llama S16 confirmation blocks; default is plan-only.

No scheduler is changed. --prepare is CPU-only; --execute is a separate opt-in
after the already-authorized InfiniteBench/Qwen queue. Scores never control
whether subsequent blocks run. The historical gate is not independent evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .prepare_clean_transfer import TASKS, _atomic_json, _sha256


PLAN = Path("/root/autodl-tmp/today_rope_plan_20260914")
ARMS = ("tailspline", "mrpro")
LENGTH = 131072
BLOCKS = tuple({"block": i + 1, "seed": 20263701 + i * 10000,
                "qa_offset": 5810 + i * 10, "rows_per_task": 10} for i in range(4))
RUNTIME_KEYS = ("prefill_chunk_size", "generation_prefill_strategy", "batch_size",
                "generation_order", "runtime_versions")
IDENTITY_KEYS = ("row_id", "task", "length_cap", "input_tokens", "prompt_sha256", "references")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def compact_rows(path: Path) -> list[dict]:
    # A full 128K panel is large: do not keep four copies of all token IDs.
    with path.open() as stream:
        return [{key: row.get(key) for key in IDENTITY_KEYS}
                for line in stream if line.strip() for row in [json.loads(line)]]


def frozen_json(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"frozen confirmation identity differs: {path}")
    else:
        _atomic_json(path, value)


def validate_disjoint(panels: dict[str, list[dict]]) -> None:
    seen = set()
    for name, rows in panels.items():
        if (Counter(row["task"] for row in rows) != Counter({task: 10 for task in TASKS})
                or len({row["row_id"] for row in rows}) != 130
                or any(row["length_cap"] != LENGTH for row in rows)):
            raise ValueError(f"incomplete Full13 x 10 panel: {name}")
        hashes = {row["prompt_sha256"] for row in rows}
        if len(hashes) != 130 or seen.intersection(hashes):
            raise ValueError(f"gate/confirmation prompts overlap: {name}")
        seen.update(hashes)


def gate_configuration(gate: Path) -> dict:
    contracts = {arm: read_json(gate / "runs" / arm / "contract.json") for arm in ARMS}
    runtime = {arm: {key: contracts[arm].get(key) for key in RUNTIME_KEYS} for arm in ARMS}
    if runtime["tailspline"] != runtime["mrpro"]:
        raise ValueError("gate arms used different generation runtimes")
    tables = {}
    for arm in ARMS:
        receipt = read_json(gate / "tables" / f"{arm}.json")
        table = receipt.get("table", receipt)
        active = contracts[arm].get("static_table") or {}
        if (receipt.get("scale") != 16 or receipt.get("band_envelope") != [18, 35]
                or receipt.get("gain") != 1 + 0.1 * math.log(16)
                or any(active.get(key) != table.get(key) for key in ("values_float32", "gain"))):
            raise ValueError(f"gate does not contain the frozen S16 table: {arm}")
        tables[arm] = {key: table[key] for key in ("values_float32", "gain")}
    return {"generation_runtime": runtime["tailspline"], "tables": tables,
            "lm_prefill_chunk_size": contracts["tailspline"]["lm_prefill_chunk_size"]}


def prepare_commands(args) -> list[list[str]]:
    return [[args.python, "-m", "experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer",
             "--model", str(args.model), "--model-id", "llama3_8b", "--data-root", str(args.upstream),
             "--out", str(args.out / f"block{block['block']}" / "assets"),
             "--scale", "16", "--lengths", str(LENGTH), "--rows-per-task", "10",
             "--seed", str(block["seed"]), "--qa-offset", str(block["qa_offset"])]
            for block in BLOCKS]


def prepare(args) -> dict:
    frozen_json(args.out / "specification.json", {
        "status": "LLAMA_S16_CONFIRM40_FIXED_DESIGN_V1", "blocks": list(BLOCKS),
        "gate_qa_indices": [5800, 5809], "scale": 16, "length": LENGTH,
        "selection_uses_model_outputs": False, "adaptive_stopping": False,
        "primary": "independent confirm40", "secondary": "gate10 plus confirm40 cumulative50",
    })
    # The four blocks are independent and the Pro6000 host has ample CPU/RAM.
    # Prepare them concurrently so CPU tokenization finishes before the GPU queue.
    logs = args.out / "logs"
    logs.mkdir(exist_ok=True)
    processes = []
    for block, command in zip(BLOCKS, prepare_commands(args)):
        log = (logs / f"prepare_block{block['block']}.log").open("a")
        processes.append((block, log, subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
        )))
    failures = []
    for block, log, process in processes:
        returncode = process.wait()
        log.close()
        if returncode:
            failures.append((block["block"], returncode))
    if failures:
        raise RuntimeError(f"S16 block preparation failed: {failures}")
    panels = {"gate": compact_rows(args.gate / "assets/full13/inputs.jsonl")}
    receipts = []
    for block in BLOCKS:
        name = f"block{block['block']}"
        path = args.out / name / "assets/panels" / str(LENGTH) / "inputs.jsonl"
        panels[name] = compact_rows(path)
        manifest = read_json(args.out / name / "assets/manifest.json")
        if any(manifest.get(key) != block[key] for key in ("seed", "qa_offset", "rows_per_task")):
            raise ValueError(f"source seed/QA offset drift: {name}")
        receipts.append({**block, "inputs": str(path.relative_to(args.out)), "inputs_sha256": _sha256(path)})
    validate_disjoint(panels)
    ready = {"status": "LLAMA_S16_CONFIRM40_READY_V1", "blocks": receipts,
             "gate_inputs_sha256": _sha256(args.gate / "assets/full13/inputs.jsonl"),
             **gate_configuration(args.gate)}
    frozen_json(args.out / "ready.json", ready)
    return ready


def generation_command(args, block: dict, arm: str, ready: dict) -> list[str]:
    runtime = ready["generation_runtime"]
    command = [args.python, "-m", "experiments.olmo_recovery_20260912.recovery_v2_eval",
               "--data", str(args.gate / "assets/ppl10/manifest.json"),
               "--model", str(args.model), "--arm", "Native", "--only-extra-panels", "--skip-lm",
               "--extra-panel", str(args.out / block["inputs"]),
               "--length-cap", str(LENGTH), "--lm-length-cap", str(LENGTH),
               "--prefill-chunk-size", str(runtime["prefill_chunk_size"]),
               "--lm-prefill-chunk-size", str(ready["lm_prefill_chunk_size"]),
               "--batch-size", str(runtime["batch_size"]),
               "--static-table-json", str(args.gate / "tables" / f"{arm}.json"),
               "--table-label", f"llama3_8b_s16_128k_{arm}",
               "--out", str(args.out / f"block{block['block']}" / "runs" / arm)]
    if runtime["generation_order"] == "longest_first_shape_sorted_v1":
        command.append("--longest-first")
    return command


def validate_ready(args) -> tuple[dict, dict[str, list[dict]]]:
    ready = read_json(args.out / "ready.json")
    if (ready.get("status") != "LLAMA_S16_CONFIRM40_READY_V1"
            or [{key: value[key] for key in BLOCKS[0]} for value in ready["blocks"]] != list(BLOCKS)
            or any(ready.get(key) != value for key, value in gate_configuration(args.gate).items())):
        raise ValueError("confirmation preparation/runtime identity drift")
    paths = {"gate": (args.gate / "assets/full13/inputs.jsonl", ready["gate_inputs_sha256"])}
    paths.update({f"block{b['block']}": (args.out / b["inputs"], b["inputs_sha256"]) for b in ready["blocks"]})
    panels = {}
    for name, (path, digest) in paths.items():
        if _sha256(path) != digest:
            raise ValueError(f"confirmation input drift: {name}")
        panels[name] = compact_rows(path)
    validate_disjoint(panels)
    return ready, panels


def load_scores(path: Path, panel: list[dict], arm: str, ready: dict, *, gate=False) -> dict:
    expected_status = {"status": "COMPLETE", "rows": 130, "lm_rows": 10 if gate else 0}
    if read_json(path / "status.json") != expected_status:
        raise ValueError(f"incomplete S16 block: {path}")
    contract = read_json(path / "contract.json")
    if {key: contract.get(key) for key in RUNTIME_KEYS} != ready["generation_runtime"]:
        raise ValueError(f"S16 generation runtime differs: {path}")
    if any((contract.get("static_table") or {}).get(key) != value
           for key, value in ready["tables"][arm].items()):
        raise ValueError(f"S16 installed table differs: {path}")
    with (path / "generations.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    mapping = {row["row_id"]: row for row in rows}
    if len(rows) != 130 or len(mapping) != 130 or set(mapping) != {r["row_id"] for r in panel}:
        raise ValueError(f"S16 block row coverage differs: {path}")
    result = {}
    for source in panel:
        row = mapping[source["row_id"]]
        if any(row.get(key) != source[key] for key in IDENTITY_KEYS):
            raise ValueError(f"S16 prompt identity differs: {path}/{source['row_id']}")
        value = float(row["ruler_official_score"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid RULER score")
        result[(source["task"], source["prompt_sha256"])] = value
    return result


def paired_summary(cells: dict[str, dict], *, draws=20000, seed=20263701) -> dict:
    if set(cells[ARMS[0]]) != set(cells[ARMS[1]]):
        raise ValueError("S16 arms are not exactly paired")
    rng = np.random.default_rng(seed)
    sampled = np.zeros(draws)
    by_task = {}
    for task in TASKS:
        keys = sorted(key for key in cells[ARMS[0]] if key[0] == task)
        if not keys:
            raise ValueError("S16 summary is missing a task")
        values = np.asarray([[cells[arm][key] for arm in ARMS] for key in keys])
        delta = values[:, 0] - values[:, 1]
        sampled += delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1) / len(TASKS)
        by_task[task] = {"n": len(keys), **dict(zip(ARMS, map(float, values.mean(axis=0)))),
                         "delta": float(delta.mean())}
    return {"rows_per_arm": len(cells[ARMS[0]]), "by_task": by_task,
            "task_macro": {arm: float(np.mean([value[arm] for value in by_task.values()])) for arm in ARMS},
            "delta": float(np.mean([value["delta"] for value in by_task.values()])),
            "ci95": list(map(float, np.quantile(sampled, [0.025, 0.975]))),
            "bootstrap": {"draws": draws, "seed": seed, "unit": "paired rows within task; equal task weight"}}


def report(args, ready, panels) -> dict:
    blocks = {}
    combined = {arm: {} for arm in ARMS}
    for block in ready["blocks"]:
        name = f"block{block['block']}"
        cells = {arm: load_scores(args.out / name / "runs" / arm, panels[name], arm, ready) for arm in ARMS}
        blocks[name] = paired_summary(cells)
        for arm in ARMS:
            combined[arm].update(cells[arm])
    confirm = paired_summary(combined)
    for arm in ARMS:
        combined[arm].update(load_scores(args.gate / "runs" / arm, panels["gate"], arm, ready, gate=True))
    result = {"status": "LLAMA_S16_CONFIRM40_CUMULATIVE50_COMPLETE_V1", "scale": 16,
              "native_length": 8192, "evaluation_length": LENGTH, "blocks": blocks,
              "confirm40_primary": confirm, "gate10_plus_confirm40_cumulative50": paired_summary(combined),
              "adaptive_stopping": False, "generation_runtime": ready["generation_runtime"],
              "frozen_block_identity": ready["blocks"],
              "ppl": "Reused historical gate PPL10; not rerun or expanded by this confirmation.",
              "claim_boundary": "Independent confirm40 is primary. Cumulative50 includes the previously observed gate; it is not a new independent 50/task confirmation.",
              "sampling_boundary": "S4/32K subsampling intervals describe the finite completed 200/task population, not an external-population confidence interval. S4/32K versus S16/128K changes both scale and length; their difference cannot be attributed to sample size alone."}
    _atomic_json(args.out / "reports/confirm40_and_cumulative50.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=PLAN / "tailspline_llama_s16_128k_gate")
    parser.add_argument("--out", type=Path, default=PLAN / "tailspline_llama_s16_128k_confirm40")
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--upstream", type=Path, default=Path("/root/autodl-tmp/rope_qwen_baseline_20260907/ruler_upstream/RULER-c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"))
    parser.add_argument("--python", default=sys.executable)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not any((args.prepare, args.execute, args.report_only)):
        print(json.dumps({"status": "PLAN_ONLY", "blocks": list(BLOCKS), "new_generations": 1040,
                          "new_lm_rows": 0, "commands_cpu_prepare": prepare_commands(args),
                          "queue_policy": "Separate opt-in after the existing InfiniteBench and Qwen append queue; no scheduler mutation."}, indent=2))
        return
    if args.prepare:
        print(json.dumps({"status": prepare(args)["status"]}))
        return
    ready, panels = validate_ready(args)
    if args.execute:
        import fcntl
        with open(os.environ.get("GPU_LOCK_PATH", "/tmp/hybrid-rope-gpu0.lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for block in ready["blocks"]:
                name = f"block{block['block']}"
                started = time.monotonic()
                cells = {}
                for arm in ARMS:
                    run = args.out / name / "runs" / arm
                    if not (run / "status.json").exists():
                        logs = args.out / "logs"
                        logs.mkdir(exist_ok=True)
                        with (logs / f"{name}_{arm}.log").open("a") as log:
                            subprocess.run(generation_command(args, block, arm, ready) + ["--execute"],
                                           check=True, stdout=log, stderr=subprocess.STDOUT)
                    cells[arm] = load_scores(run, panels[name], arm, ready)
                _atomic_json(args.out / name / "report.json", {
                    **paired_summary(cells), "wall_seconds_this_invocation": time.monotonic() - started,
                    "block": block["block"], "subsequent_blocks": "fixed; never selected from this score"})
    print(json.dumps({"status": report(args, ready, panels)["status"]}))


if __name__ == "__main__":
    main()
