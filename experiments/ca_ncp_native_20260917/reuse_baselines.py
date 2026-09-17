#!/usr/bin/env python3
"""Validate and reference the existing N0/C0 outputs without copying or rerunning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import tensor_sha256
from .io_utils import atomic_json, file_sha256


DEFAULT_NATIVE = Path(
    "/root/autodl-tmp/today_rope_plan_20260914/native_research_20260916/"
    "runs/confirm/ruler/native"
)
DEFAULT_NCP = Path(
    "/root/autodl-tmp/today_rope_plan_20260914/native_research_20260916/"
    "runs/confirm/ruler/ncp"
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_run(source: Path, *, arm: str, pilot: dict, expected_table_hash: str) -> dict:
    for name in ("contract.json", "summary.json", "status.json", "generations.jsonl"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    status = json.loads((source / "status.json").read_text())
    contract = json.loads((source / "contract.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    generated = rows(source / "generations.jsonl")
    panel = rows(Path(pilot["panel"]["source_inputs"]))
    if status != {"status": "COMPLETE", "rows": 130, "lm_rows": 0} or len(generated) != 130:
        raise ValueError(f"existing {arm} baseline is incomplete")
    if [row["row_id"] for row in generated] != [row["row_id"] for row in panel]:
        raise ValueError(f"existing {arm} row identity differs from the reused panel")
    for observed, reference in zip(generated, panel):
        # The historical evaluator stored the complete frozen panel hash and all
        # semantic input fields, but omitted max_new_tokens from each output row.
        for field in ("task", "prompt_sha256", "references", "input_tokens"):
            if observed.get(field) != reference.get(field):
                raise ValueError(f"existing {arm} input drift: {observed['row_id']}/{field}")
    runtime = contract.get("runtime_versions") or {}
    required = {
        "unadapted": True,
        "batch_size": 1,
        "prefill_chunk_size": 0,
        "generation_prefill_strategy": "direct_generate_v1",
        "generation_order": "panel_order_v1",
    }
    if any(contract.get(key) != value for key, value in required.items()):
        raise ValueError(f"existing {arm} execution contract differs: {required}")
    if (
        runtime.get("torch") != "2.8.0+cu128"
        or runtime.get("transformers") != "5.15.1"
        or runtime.get("model_dtype") != "bfloat16"
        or runtime.get("attention_backend") != "torch_sdpa_flash_only"
    ):
        raise ValueError(f"existing {arm} runtime software/precision differs")
    table = summary.get("table") or {}
    values = np.asarray(table.get("values_float32"), dtype=np.float32)
    if values.shape != (64,) or tensor_sha256(values) != expected_table_hash or float(table.get("gain")) != 1.0:
        raise ValueError(f"existing {arm} frequency table differs")
    if contract.get("ca_ncp_alignment") is not None or summary.get("ca_ncp_alignment") is not None:
        raise ValueError(f"existing {arm} unexpectedly contains CA-NCP alignment")
    return {
        "source": str(source.resolve()),
        "contract_sha256": file_sha256(source / "contract.json"),
        "summary_sha256": file_sha256(source / "summary.json"),
        "generations_sha256": file_sha256(source / "generations.jsonl"),
        "table_sha256_float32": expected_table_hash,
        "rows": 130,
        "runtime_versions": runtime,
        "reused_without_copy": True,
    }


def link(directory: Path, source: Path) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        if directory.resolve() != source.resolve():
            raise ValueError(f"baseline link points elsewhere: {directory}")
        return
    if directory.exists():
        raise ValueError(f"baseline output already exists and is not a reuse link: {directory}")
    directory.symlink_to(source.resolve(), target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--ncp-run", type=Path, default=DEFAULT_NCP)
    args = parser.parse_args()
    pilot = json.loads((args.root / "assets/pilot/manifest.json").read_text())
    method = json.loads((args.root / "construction/METHOD_RECEIPT.json").read_text())
    receipts = {
        "N0": validate_run(
            args.native_run, arm="N0", pilot=pilot,
            expected_table_hash=method["native_table_sha256_float32"],
        ),
        "C0": validate_run(
            args.ncp_run, arm="C0", pilot=pilot,
            expected_table_hash=method["ncp_table_sha256_float32"],
        ),
    }
    link(args.root / "runs/N0", args.native_run)
    link(args.root / "runs/C0", args.ncp_run)
    receipt = {
        "status": "CA_NCP_BASELINE_REUSE_READY",
        "panel_sha256": pilot["panel"]["inputs_sha256"],
        "runs": receipts,
        "new_baseline_generations": 0,
        "scope": "Exact same tokenized panel, checkpoint, table, precision, decoder and runtime; symlink references only.",
    }
    atomic_json(args.root / "BASELINE_REUSE_RECEIPT.json", receipt)
    print(json.dumps({"status": receipt["status"], "new_baseline_generations": 0}))


if __name__ == "__main__":
    main()
