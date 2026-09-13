#!/usr/bin/env python3
"""Combine the three frozen 778-row natural-QA batches for the E2 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.cross_audit.tables import native_table, tensor_sha, transform
from scripts.lib.rope.boundary_matched import boundary_matched_inv_freq


TASKS = ("hotpotqa", "2wikimqa", "qasper", "narrativeqa", "multifieldqa_en")
CAPS = {"hotpotqa": 32, "2wikimqa": 32, "qasper": 128, "narrativeqa": 128, "multifieldqa_en": 64}
EXPECTED_ALL = {"hotpotqa": 171, "2wikimqa": 199, "qasper": 197, "narrativeqa": 61, "multifieldqa_en": 150}
EXPECTED_LONG = {"hotpotqa": 166, "2wikimqa": 173, "qasper": 119, "narrativeqa": 61, "multifieldqa_en": 112}
MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
HISTORICAL_TABLE_SHA256 = {
    "Native": "dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34",
    "MrPro": "ed0120e3ba436199c71d2e25e923d7665a1e9c264fe18135f61f310296b94069",
    "MrProBM": "fc0f443b1c58601d51209adb7e2b26df7ba10058a9dbdf193eb1de49116d446a",
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def table_record(values: np.ndarray, gain: float, construction: dict) -> dict:
    values = np.ascontiguousarray(values, dtype=np.float32)
    return {"values_float32": values.tolist(), "tensor_sha256": tensor_sha(values),
            "gain": float(gain), "construction": construction}


def build_tables() -> dict[str, dict]:
    native = native_table(128, 500_000).astype(np.float32)
    bm, bm_gain, bm_meta = boundary_matched_inv_freq(
        torch.from_numpy(native.copy()), base=500_000, reference_length=4_096, scale=4)
    pro, pro_gain, pro_meta = transform(native, dim=128, base=500_000,
                                         reference_length=4_096, scale=4, method="mrpro")
    uni, uni_gain, uni_meta = transform(native, dim=128, base=500_000,
                                         reference_length=4_096, scale=4, method="mruni")
    yarn, yarn_gain, yarn_meta = transform(native, dim=128, base=500_000,
                                            reference_length=4_096, scale=4, method="yarn")
    g4 = 1.0 + 0.1 * math.log(4.0)
    if max(abs(x - g4) for x in (bm_gain, pro_gain, uni_gain, yarn_gain)) > 1e-15:
        raise AssertionError("matched gain construction drift")
    raw = {
        "bm_g4": (bm.numpy(), g4, bm_meta), "mrpro_g4": (pro, g4, pro_meta),
        "mruni_g4": (uni, g4, uni_meta), "official_yarn_g4": (yarn, g4, yarn_meta),
        "native_g1": (native, 1.0, {"method": "identity"}),
        "bm_g1": (bm.numpy(), 1.0, {**bm_meta, "gain_override": 1.0}),
        "mrpro_g1": (pro, 1.0, {**pro_meta, "gain_override": 1.0}),
    }
    return {name: table_record(*parts) for name, parts in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", nargs=3, type=Path, required=True,
                        help="prepared_natural_01, _02, and _extra_01 directories")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    model = args.model.resolve()
    if not (model / "tokenizer.json").is_file() or not (model / "config.json").is_file():
        raise FileNotFoundError("model config/tokenizer files are required")
    rows: list[dict] = []
    sources = []
    seen = set()
    generation = None
    for source in args.prepared:
        source = source.resolve()
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
            raise ValueError(f"model contract drift in {source}")
        current_generation = json.loads((source / "generation_config.json").read_text())
        if generation is None:
            generation = current_generation
        elif current_generation != generation:
            raise ValueError("generation configuration differs across frozen batches")
        batch = read_jsonl(source / "screen.jsonl")
        if len(batch) != int(manifest["screen_rows"]):
            raise ValueError("batch row count differs from manifest")
        for row in batch:
            required = {"row_id", "task", "prompt_ids", "prompt_sha256", "input_tokens", "max_new_tokens", "references"}
            if not required.issubset(row):
                raise ValueError(f"incomplete frozen row {row.get('row_id')}")
            if row["row_id"] in seen:
                raise ValueError(f"duplicate frozen row {row['row_id']}")
            if row["task"] not in TASKS or row["max_new_tokens"] != CAPS[row["task"]]:
                raise ValueError("task or original generation cap drift")
            if len(row["prompt_ids"]) != row["input_tokens"]:
                raise ValueError("prompt token length drift")
            if row["input_tokens"] + row["max_new_tokens"] > 16_384:
                raise ValueError("frozen prompt exceeds original physical cap")
            seen.add(row["row_id"])
            kept = {key: row[key] for key in required}
            kept["source_id"] = row.get("source_id")
            kept["source_row_index"] = row.get("source_row_index")
            kept["question_cluster_id"] = row["row_id"]
            kept["document_cluster_id"] = row.get("source_id") or row["row_id"]
            rows.append(kept)
        sources.append({"name": source.name, "rows": len(batch)})

    all_counts = {task: sum(row["task"] == task for row in rows) for task in TASKS}
    long_counts = {task: sum(row["task"] == task and row["input_tokens"] > 4_096 for row in rows) for task in TASKS}
    if len(rows) != 778 or all_counts != EXPECTED_ALL or long_counts != EXPECTED_LONG:
        raise ValueError(f"not the locked 778/631 pool: all={all_counts}, long={long_counts}")

    tables = build_tables()
    generation.update({"do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                       "use_cache": True, "min_new_tokens": 0, "forced_eos_token_id": None})
    args.output.mkdir(parents=True)
    inputs_path = args.output / "inputs.jsonl"
    with inputs_path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    atomic_json(args.output / "tables.json", tables)
    atomic_json(args.output / "generation_config.json", generation)
    manifest = {
        "schema_version": 1, "status": "READY_GPU_NOT_RUN", "experiment": "E2",
        "model": {"id": MODEL_ID, "revision": REVISION, "path": str(model)},
        "pool": {"rows": 778, "long_rows": 631, "long_rule": "input_tokens > 4096",
                 "all_by_task": all_counts, "long_by_task": long_counts,
                 "identity": "the original frozen eligible LongBench pool; no replacement or reselection"},
        "task_caps": CAPS, "sources": sources,
        "arms": list(tables), "primary_contrasts": ["bm_g4-mruni_g4", "bm_g4-official_yarn_g4"],
        "gain_interaction": "(bm_g4-mrpro_g4)-(bm_g1-mrpro_g1)",
        "scoring": "whole-response LongBench-normalized token F1; five-task equal macro on 631 long rows",
        "raw_output_contract": "preserve every generated token id including EOS; decode score text after removing terminal EOS only; never substring-grade",
        "decoder_contract": "tokenizer.decode(score_ids, skip_special_tokens=False), matching the original natural-QA runner",
        "cluster_contract": "document_cluster_id uses frozen source_id when present; otherwise question_cluster_id=row_id",
        "required_device": {"name_contains": "5090", "cuda_capability": [12, 0], "torch_arch": "sm_120", "precision": "bfloat16"},
        "prompt_collection_sha256": digest([[row["row_id"], row["prompt_sha256"]] for row in rows]),
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "rows": 778, "long_rows": 631, "arms": list(tables)}))


if __name__ == "__main__":
    main()
