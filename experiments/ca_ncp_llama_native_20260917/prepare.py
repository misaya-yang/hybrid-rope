#!/usr/bin/env python3
"""Prepare the frozen Llama Native-8K CA-NCP gate without model execution."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.ca_ncp_native_20260917.core import build_carrier_table, tensor_sha256
from experiments.ca_ncp_native_20260917.io_utils import atomic_json, file_sha256, model_identity
from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS
from scripts.experiments.cross_audit.tables import native_table

from . import (
    CARRIER_TABLE_SHA256,
    METHOD_ID,
    NATIVE_LENGTH,
    NCP_TABLE_SHA256,
    ROPE_BASE,
    ROTARY_PAIRS,
)


PILOT_CONTRACT = "CA_NCP_NATIVE_FULL13_X10_REUSE_V1_1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def resolve_panel(manifest_path: Path, manifest: dict) -> Path:
    panel = (manifest.get("panels") or {}).get(str(NATIVE_LENGTH)) or {}
    value = panel.get("inputs")
    if not value:
        raise ValueError("source asset lacks the Llama Native-8K panel")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def select_panel(source_path: Path, source: dict) -> list[dict]:
    rows = read_jsonl(resolve_panel(source_path, source))
    selected = []
    for task in TASKS:
        task_rows = [row for row in rows if row.get("task") == task]
        if len(task_rows) < 10:
            raise ValueError(f"source panel lacks ten rows for {task}")
        selected.extend(task_rows[:10])
    if len(selected) != 130 or len({str(row["row_id"]) for row in selected}) != 130:
        raise ValueError("selected Llama pilot does not contain 130 unique rows")
    return selected


def table_payload(values: np.ndarray, *, label: str, construction: dict) -> dict:
    return {
        "status": "FROZEN_CA_NCP_LLAMA_TABLE_V1",
        "candidate_id": label,
        "values_float32": np.asarray(values, dtype=np.float32).tolist(),
        "gain": 1.0,
        "table_sha256_float32": tensor_sha256(values),
        "construction": construction,
    }


def prepare_tables(model: Path, runtime_matrix: Path, out: Path) -> dict:
    # Reuse the already established checkpoint location without re-hashing the
    # 15GB weight shards while the OLMo GPU queue is active.
    identity = model_identity(model, include_checkpoint_files=False)
    if (
        identity["model_type"] != "llama"
        or identity["native_length"] != NATIVE_LENGTH
        or identity["rotary_pairs"] != ROTARY_PAIRS
        or identity["rope_theta"] != ROPE_BASE
    ):
        raise ValueError("Llama CA-NCP is frozen to Llama-3-8B Native-8192, K=64, base=500000")
    matrix = json.loads(runtime_matrix.read_text())
    matches = [table for table in matrix.get("tables", []) if table.get("model_id") == "llama3_8b"]
    if len(matches) != 1:
        raise ValueError("runtime matrix must contain exactly one llama3_8b NCP table")
    source = matches[0]
    ncp = np.asarray(source["values_float32"], dtype=np.float32)
    native = native_table(ROTARY_PAIRS * 2, ROPE_BASE).astype(np.float32)
    geometry = build_carrier_table(native, ncp, NATIVE_LENGTH)
    if geometry["ncp_table_sha256_float32"] != NCP_TABLE_SHA256:
        raise ValueError("Llama NCP table differs from the frozen public construction")
    if geometry["carrier_table_sha256_float32"] != CARRIER_TABLE_SHA256:
        raise ValueError("Llama carrier table differs from the frozen CPU reference")
    construction = {
        "method_id": METHOD_ID,
        "native_length": NATIVE_LENGTH,
        "rotary_pairs": ROTARY_PAIRS,
        "rope_base": ROPE_BASE,
        "gain": 1.0,
        "carrier_ratio": geometry["carrier_ratio"],
        "carrier_slot_zero_based": geometry["carrier_slot"],
        "carrier_local_zero_based": geometry["carrier_local"],
        "carrier_frequency_float64": geometry["carrier_frequency_float64"],
        "carrier_frequency_float32": geometry["carrier_frequency_float32"],
        "active_indices_zero_based": geometry["active_indices"].tolist(),
        "ncp_changed_indices_zero_based": geometry["changed_indices"].tolist(),
        "activity_rule": "NCP differs from Native, interior slot, and L*omega_native <= 2*pi",
        "ncp_prior_construction": source.get("construction", {}),
        "selection_uses_model_outputs": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "native.json", table_payload(native, label="N0_native", construction=construction))
    atomic_json(out / "ncp.json", table_payload(ncp, label="C0_llama_ncp", construction=construction))
    atomic_json(
        out / "carrier_ncp.json",
        table_payload(geometry["carrier_table"], label="P0_llama_carrier_ncp", construction=construction),
    )
    receipt = {
        "status": "METHOD_CONSTRUCTION_COMPLETE",
        "method_id": METHOD_ID,
        "model_identity": identity,
        "ncp_source_path": str(runtime_matrix.resolve()),
        "ncp_source_file_sha256": file_sha256(runtime_matrix),
        "native_table_sha256_float32": geometry["native_table_sha256_float32"],
        "ncp_table_sha256_float32": geometry["ncp_table_sha256_float32"],
        "carrier_table_sha256_float32": geometry["carrier_table_sha256_float32"],
        "construction": construction,
        "public_constants_only_for_tables": True,
        "model_execution": False,
        "task_outputs_read": False,
        "claim_boundary": "Llama frequency construction only; task quality remains contingent on the OLMo gate and a future GPU run.",
    }
    atomic_json(out / "METHOD_RECEIPT.json", receipt)
    return receipt


def prepare_panel(model: Path, source_manifest: Path, out: Path) -> tuple[dict, list[dict]]:
    source = json.loads(source_manifest.read_text())
    identity = model_identity(model, include_checkpoint_files=False)
    if (source.get("model_identity") or {}).get("config_sha256") != identity["config_sha256"]:
        raise ValueError("source Llama panel tokenizer/checkpoint differs")
    if (
        source.get("status") != "COMPLETE"
        or source.get("selection_mode") != "source-order"
        or source.get("selection_uses_model_outputs") is not False
        or source.get("content_padding") is not False
    ):
        raise ValueError("source Llama panel is not the clean output-blind unpadded asset")
    selected = select_panel(source_manifest, source)
    inputs = out / "inputs.jsonl"
    write_jsonl(inputs, selected)
    payload = {
        "status": "COMPLETE",
        "contract": PILOT_CONTRACT,
        "rows": 130,
        "rows_per_task": 10,
        "tasks": list(TASKS),
        "model_identity": identity,
        "selection_mode": "source-order",
        "selection_uses_model_outputs": False,
        "content_padding": False,
        "reuse": {
            "source_manifest": str(source_manifest.resolve()),
            "source_manifest_sha256": file_sha256(source_manifest),
            "source_rows_per_task": source.get("rows_per_task"),
            "selection": "first ten stored rows per task",
            "retokenized": False,
        },
        "panel": {"source_inputs": str(inputs.resolve()), "inputs_sha256": file_sha256(inputs), "rows": 130},
        "parity": {
            "source_inputs": str(inputs.resolve()),
            "tasks": list(TASKS[:8]),
            "limit_per_cell": 1,
            "rows": 8,
            "selection": "first stored row of the first eight frozen tasks",
            "accuracy_role": "none",
        },
        "scope": "Llama Native-8K first-ten-per-task CA-NCP transfer gate from the existing Full-13x50 asset.",
    }
    atomic_json(out / "manifest.json", payload)
    return payload, selected


def reuse_native(source_run: Path, selected: list[dict], out: Path, method: dict) -> dict:
    selected_ids = [str(row["row_id"]) for row in selected]
    source_rows = {str(row["row_id"]): row for row in read_jsonl(source_run / "generations.jsonl")}
    if any(row_id not in source_rows for row_id in selected_ids):
        raise ValueError("existing Llama Native run lacks selected pilot rows")
    rows = [source_rows[row_id] for row_id in selected_ids]
    if Counter(row["task"] for row in rows) != Counter({task: 10 for task in TASKS}):
        raise ValueError("reused Llama Native task counts differ")
    source_contract = json.loads((source_run / "contract.json").read_text())
    source_summary = json.loads((source_run / "summary.json").read_text())
    table = source_summary.get("table") or {}
    values = np.asarray(table.get("values_float32"), dtype=np.float32)
    if tensor_sha256(values) != method["native_table_sha256_float32"] or float(table.get("gain")) != 1.0:
        raise ValueError("existing Llama Native run installed another frequency table")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "generations.jsonl", rows)
    contract = {
        **source_contract,
        "arm": "ca_ncp_llama_N0",
        "base_arm": "Native",
        "row_ids": selected_ids,
        "ca_ncp_alignment": None,
        "reused_source_run": str(source_run.resolve()),
        "reused_source_generations_sha256": file_sha256(source_run / "generations.jsonl"),
    }
    summary = {
        **source_summary,
        "identity": contract,
        "ca_ncp_alignment": None,
        "rows": 130,
        "reused_source_run": str(source_run.resolve()),
    }
    atomic_json(out / "contract.json", contract)
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "status.json", {"status": "COMPLETE", "rows": 130, "lm_rows": 0})
    return {
        "source": str(source_run.resolve()),
        "source_generations_sha256": file_sha256(source_run / "generations.jsonl"),
        "subset_generations_sha256": file_sha256(out / "generations.jsonl"),
        "rows": 130,
        "reused_without_model_execution": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime-matrix", type=Path, required=True)
    parser.add_argument("--source-assets", type=Path, required=True)
    parser.add_argument("--source-native-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    method = prepare_tables(args.model, args.runtime_matrix, args.out / "construction")
    pilot, selected = prepare_panel(args.model, args.source_assets, args.out / "assets/pilot")
    reuse = reuse_native(args.source_native_run, selected, args.out / "runs/N0", method)
    receipt = {
        "status": "CA_NCP_LLAMA_CPU_BASE_READY",
        "method_receipt": str((args.out / "construction/METHOD_RECEIPT.json").resolve()),
        "pilot_manifest": str((args.out / "assets/pilot/manifest.json").resolve()),
        "panel_sha256": pilot["panel"]["inputs_sha256"],
        "native_baseline": reuse,
        "gpu_execution": False,
    }
    atomic_json(args.out / "CPU_BASE_READINESS.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
