#!/usr/bin/env python3
"""Freeze CPU-side assets and an execution ledger for the ICLR sprint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
CAPS = (8192, 16384, 32768)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def iter_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


def state(path: Path, expected: dict | None = None) -> dict:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    value = read_json(path)
    return {
        "path": str(path), "exists": True, "sha256": sha256(path),
        "status": value.get("status"), "matches_expected": expected is None or value == expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = args.plan_root.resolve()
    output = args.out.resolve()
    classic = plan / "tailspline_llama_s4_classic"
    clean = plan / "tailspline_llama_s4_32k_ruler200_clean"
    natural = plan / "tailspline_llama_s4_naturalqa631"
    dose = plan / "tailspline_llama_s4_matched_dose_c"
    strong = plan / "tailspline_llama_s4_classic_strong_baselines"
    native_z = plan / "olmo_native_z5_enhancement"

    classic_panel = classic / "assets/full13/rows.jsonl"
    cells = Counter()
    selected = {}
    classic_count = 0
    for row in iter_jsonl(classic_panel):
        classic_count += 1
        key = (str(row["task"]), int(row["length_cap"]))
        cells[key] += 1
        identity = (str(row["row_id"]), str(row["prompt_sha256"]))
        if key not in selected or identity < selected[key][0]:
            selected[key] = (identity, row)
    expected_cells = {(task, cap) for task in TASKS for cap in CAPS}
    if classic_count != 390 or set(cells) != expected_cells or set(cells.values()) != {10}:
        raise ValueError("classic panel identity drift")
    runtime_probe = [
        selected[(task, cap)][1]
        for cap in CAPS for task in TASKS
    ]
    if len({row["prompt_sha256"] for row in runtime_probe}) != 39:
        raise ValueError("runtime probe prompt identity drift")

    clean_panel = clean / "assets/inputs.jsonl"
    clean_count = 0
    clean_tasks = Counter()
    clean_prompts = set()
    for row in iter_jsonl(clean_panel):
        clean_count += 1
        clean_tasks[row["task"]] += 1
        clean_prompts.add(row["prompt_sha256"])
    if clean_count != 2600 or len(clean_prompts) != 2600 or clean_tasks != Counter({task: 200 for task in TASKS}):
        raise ValueError("clean RULER-200 identity drift")
    clean_manifest = read_json(clean / "assets/manifest.json")
    if clean_manifest.get("content_padding") is not False or clean_manifest.get("selection_mode") != "source-order":
        raise ValueError("clean RULER-200 contract drift")

    natural_panel = natural / "assets/inputs.jsonl"
    natural_counts = Counter()
    natural_documents = defaultdict(set)
    natural_cluster_tasks = defaultdict(set)
    natural_rows = 0
    for row in iter_jsonl(natural_panel):
        natural_rows += 1
        key = (str(row["llama_native_stratum"]), str(row["task"]))
        cluster = str(row["document_cluster_id"])
        natural_counts[key] += 1
        natural_documents[key].add(cluster)
        natural_cluster_tasks[cluster].add(str(row["task"]))
    if natural_rows != 631 or {key[0] for key in natural_counts} != {"within_native", "extended"}:
        raise ValueError("Natural-QA631 identity drift")

    tables = {}
    for name, path in {
        "tailspline": classic / "tables/tailspline.json",
        "mrpro": classic / "tables/mrpro.json",
        "dose_control_c": dose / "tables/llama_s4_tailspline_dose_control.json",
        "yarn": strong / "tables/yarn.json",
    }.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        table = read_json(path)
        tables[name] = {
            "path": str(path), "sha256": sha256(path), "gain": table.get("gain"),
            "table_sha256_float32": table.get("table_sha256_float32"),
        }

    probe_path = output / "assets/classic_runtime_probe39.jsonl"
    atomic_jsonl(probe_path, runtime_probe)
    ledger = {
        "status": "ICLR2027_SPRINT_CPU_ASSETS_COMPLETE_V1",
        "policy": {
            "original_gpu_order": ["clean T/P completion", "Natural-QA631 T/P", "Native-Z5"],
            "clone_gpu_order": ["classic TailSpline batch sensitivity 39", "clean 32K YaRN 2600", "classic YaRN batch1"],
            "excluded": ["BM", "new curve search", "new model", "HELMET"],
        },
        "classic": {"path": str(classic_panel), "rows": 390, "sha256": sha256(classic_panel)},
        "runtime_probe": {"path": str(probe_path), "rows": 39, "sha256": sha256(probe_path)},
        "clean_ruler": {
            "path": str(clean_panel), "rows": 2600, "rows_per_task": 200,
            "sha256": sha256(clean_panel), "batching": "batch1 exact unpadded prompt_ids",
        },
        "natural_qa": {
            "panel": str(natural_panel),
            "panel_sha256": sha256(natural_panel),
            "rows": natural_rows,
            "rows_by_stratum_task": {
                f"{stratum}|{task}": count
                for (stratum, task), count in sorted(natural_counts.items())
            },
            "source_documents_by_stratum_task": {
                f"{stratum}|{task}": len(documents)
                for (stratum, task), documents in sorted(natural_documents.items())
            },
            "clusters_shared_across_tasks": sum(
                len(tasks) > 1 for tasks in natural_cluster_tasks.values()
            ),
            "panel_state": state(natural / "assets/manifest.json"),
            "report_state": state(natural / "reports/tailspline_vs_mrpro_naturalqa631.json"),
        },
        "completed_or_reusable": {
            "matched_dose_c": state(dose / "reports/tailspline_vs_dose_control_c_classic.json"),
            "llama_native_ppl": state(classic / "runs/native_ppl_8k/status.json"),
            "clean_tailspline": state(clean / "runs/tailspline/status.json", {"status": "COMPLETE", "rows": 2600, "lm_rows": 0}),
            "clean_mrpro": state(clean / "runs/mrpro/status.json", {"status": "COMPLETE", "rows": 2600, "lm_rows": 0}),
            "native_z": state(native_z / "optimization/status.json"),
        },
        "tables": tables,
        "claim_boundaries": [
            "The 39-row replay diagnoses runtime sensitivity; it is not a replacement benchmark.",
            "Clean YaRN must use the exact clean 2600 prompts and batch1 unpadded runtime.",
            "Classic YaRN must use batch1 to match the completed TailSpline/MrPro classic arms.",
            "No BM result is scheduled by this sprint queue.",
        ],
        "model_execution": False,
    }
    atomic_json(output / "assets/manifest.json", ledger)
    atomic_json(output / "status/cpu_ready.json", {
        "status": "READY_FOR_DATA_DISK_CLONE",
        "model_execution": False,
        "original_entrypoint": "experiments/iclr2027_three_track_sprint_20260915/run_original_gpu_queue.sh",
        "clone_entrypoint": "experiments/iclr2027_three_track_sprint_20260915/run_clone_gpu_queue.sh",
        "bm_scheduled": False,
        "frozen_manifest_sha256": sha256(output / "assets/manifest.json"),
    })
    print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
