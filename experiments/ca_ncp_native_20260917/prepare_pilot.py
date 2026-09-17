#!/usr/bin/env python3
"""Freeze a reference to the existing clean Native-4K Full-13 x 10 panel.

CA-NCP does not use task outputs to construct its alignment, so regenerating or
retokenizing another panel adds no control. This entry point validates the exact
existing source-order panel and records its hashes without copying prompt IDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.iclr2027_strong_evidence_20260915.prepare_clean_transfer import TASKS
from .io_utils import atomic_json, file_sha256, model_identity


CONTRACT = "CA_NCP_NATIVE_FULL13_X10_REUSE_V1_1"
DEFAULT_REUSE = Path(
    "/root/autodl-tmp/today_rope_plan_20260914/native_research_20260916/"
    "assets/ruler_confirm_13x10/manifest.json"
)


def resolve_panel(manifest_path: Path, manifest: dict) -> Path:
    panel = ((manifest.get("panels") or {}).get("4096") or manifest.get("panel") or {})
    value = panel.get("inputs")
    if not value:
        raise ValueError("reuse manifest lacks its 4096-token panel input")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows-per-task", type=int, default=10)
    parser.add_argument("--reuse-manifest", type=Path, default=DEFAULT_REUSE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.rows_per_task != 10:
        raise ValueError("the frozen first CA-NCP pilot is Full-13 x 10")
    identity = model_identity(args.model, include_checkpoint_files=False)
    manifest_path = args.out / "manifest.json"
    if manifest_path.is_file():
        previous_identity = json.loads(manifest_path.read_text()).get("model_identity") or {}
        if previous_identity.get("config_sha256") == identity["config_sha256"]:
            identity = previous_identity
    reuse_path = args.reuse_manifest.resolve()
    source = json.loads(reuse_path.read_text())
    panel_path = resolve_panel(reuse_path, source)
    rows = [json.loads(line) for line in panel_path.read_text().splitlines() if line.strip()]
    counts = {task: sum(str(row.get("task")) == task for row in rows) for task in TASKS}
    source_identity = source.get("model_identity") or {}
    if source.get("status") != "COMPLETE" or source.get("rows") != 130:
        raise ValueError("reuse panel is not a complete 130-row asset")
    if source.get("rows_per_task") != 10 or list(source.get("tasks") or []) != list(TASKS):
        raise ValueError("reuse panel is not Full-13 x 10 in the frozen task order")
    if counts != {task: 10 for task in TASKS} or len({str(row["row_id"]) for row in rows}) != 130:
        raise ValueError("reuse panel row identity/task counts differ")
    if (
        source.get("selection_mode") != "source-order"
        or source.get("selection_uses_model_outputs") is not False
        or source.get("content_padding") is not False
    ):
        raise ValueError("reuse panel is not the required output-blind, unpadded source-order asset")
    if source_identity.get("config_sha256") != identity["config_sha256"]:
        raise ValueError("reuse panel tokenizer checkpoint differs from CA-NCP")
    recorded = ((source.get("panels") or {}).get("4096") or source.get("panel") or {}).get("inputs_sha256")
    observed = file_sha256(panel_path)
    if recorded != observed:
        raise ValueError("reuse panel inputs hash drifted")
    payload = {
        "status": "COMPLETE",
        "contract": CONTRACT,
        "rows": 130,
        "rows_per_task": 10,
        "tasks": list(TASKS),
        "model_identity": identity,
        "selection_mode": "source-order",
        "selection_uses_model_outputs": False,
        "content_padding": False,
        "reuse": {
            "source_manifest": str(reuse_path),
            "source_manifest_sha256": file_sha256(reuse_path),
            "source_contract": source.get("contract_revision"),
            "source_seed": source.get("seed"),
            "source_qa_offset": source.get("qa_offset"),
            "retokenized": False,
            "prompt_data_copied": False,
        },
        "panel": {
            "source_inputs": str(panel_path),
            "inputs_sha256": observed,
            "rows": 130,
        },
        "parity": {
            "source_inputs": str(panel_path),
            "tasks": list(TASKS[:8]),
            "limit_per_cell": 1,
            "rows": 8,
            "selection": "first stored row of the first eight frozen tasks; no new tokenization",
            "accuracy_role": "none",
        },
        "scope": "Reference-only CA-NCP pilot asset; reuses exact tokenized prompts without copying them.",
    }
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != payload:
        raise ValueError("existing CA-NCP pilot reference differs")
    if args.out.exists() and any(args.out.iterdir()) and not manifest_path.is_file():
        raise ValueError("pilot output is nonempty without a valid reuse manifest")
    atomic_json(manifest_path, payload)
    print(json.dumps({"status": "COMPLETE_REUSED", "rows": 130, "sha256": observed}))


if __name__ == "__main__":
    main()
