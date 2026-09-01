#!/usr/bin/env python3
"""Conditional P3 CPU input builder; never evaluate models or construct RoPE tables.

Fixed protocol: first 16 unique eligible documents from source row 40000,
8 calibration / 8 confirmation, 32768 Native tokens, final 256 targets.
Actual use remains conditional on explicit P3 authorization after K32 confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.data import prepare_native_reference_calibration as common  # noqa: E402

NATIVE_LENGTH = 32768
START_ROW = 40000
TARGET_TOKENS = 256
SPLITS = ("calibration", "confirmation")
DOCUMENTS_PER_SPLIT = 8
MODEL_OUTPUT_FIELDS = {"decision", "selected_length", "metrics", "nll", "loss", "score",
                       "official_task_score", "prediction", "generated_token_ids", "exact_match"}


def load_input_exclusions(paths: list[Path]) -> tuple[set[str], list[dict]]:
    excluded, receipts = set(), []
    for path in paths:
        hashes, natural_rows, ignored_rows = set(), 0, 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or "family" not in row or MODEL_OUTPUT_FIELDS & row.keys():
                    raise ValueError("exclusions must be input JSONL, not model outputs or decisions")
                if row["family"] != "natural":
                    ignored_rows += 1
                    continue
                value = row.get("source_text_sha256")
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError("natural exclusion input lacks a valid source_text_sha256")
                natural_rows += 1
                hashes.add(value)
        excluded.update(hashes)
        receipts.append({"name": path.name, "sha256": common.sha256_file(path),
                         "natural_input_rows": natural_rows, "unique_source_documents": len(hashes),
                         "ignored_non_natural_rows": ignored_rows,
                         "source_set_sha256": common.canonical_hash(sorted(hashes))})
    return excluded, receipts


def choose_bos(tokenizer: Any, config: dict) -> dict:
    vocabulary_size = len(tokenizer)
    for source, value in (("tokenizer.bos_token_id", tokenizer.bos_token_id),
                          ("checkpoint_config.bos_token_id", config.get("bos_token_id"))):
        if type(value) is int and 0 <= value < vocabulary_size:
            return {"token_id": value, "source": source}
    raise ValueError("no valid BOS id in tokenizer or explicit checkpoint config")


def build_rows(texts: Iterable[tuple[int, str]], tokenizer: Any, bos: dict,
               excluded: set[str]) -> tuple[dict[str, list[dict]], dict]:
    selected, seen, encountered_exclusions = [], set(), set()
    stats = {"source_rows_inspected": 0, "excluded_source_rows": 0,
             "duplicate_selected_source_rows": 0, "short_source_rows": 0}
    for source_row, text in texts:
        if source_row < START_ROW:
            continue
        stats["source_rows_inspected"] += 1
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in excluded:
            stats["excluded_source_rows"] += 1
            encountered_exclusions.add(text_hash)
            continue
        if text_hash in seen:
            stats["duplicate_selected_source_rows"] += 1
            continue
        document_ids = tokenizer.encode(text, add_special_tokens=False,
                                        truncation=True, max_length=NATIVE_LENGTH)
        if len(document_ids) < NATIVE_LENGTH:
            stats["short_source_rows"] += 1
            continue
        seen.add(text_hash)
        split = SPLITS[len(selected) // DOCUMENTS_PER_SPLIT]
        local_index = len(selected) % DOCUMENTS_PER_SPLIT
        ids = [bos["token_id"]] + [int(value) for value in document_ids[:NATIVE_LENGTH - 1]]
        selected.append({
            "family": "natural", "variant": "natural", "split": split,
            "sample_id": f"native-qk-{split}-{local_index:03d}", "length": NATIVE_LENGTH,
            "input_ids": ids, "target_start": NATIVE_LENGTH - TARGET_TOKENS,
            "target_tokens": TARGET_TOKENS, "source_row": source_row,
            "source_text_sha256": text_hash, "input_ids_sha256": common.ids_hash(ids),
            "target_ids_sha256": common.ids_hash(ids[-TARGET_TOKENS:]),
        })
        if len(selected) == DOCUMENTS_PER_SPLIT * len(SPLITS):
            break
    if len(selected) != DOCUMENTS_PER_SPLIT * len(SPLITS):
        raise ValueError(f"need 16 unique nonexcluded >=32768-token documents; found {len(selected)}")
    stats["excluded_unique_source_documents_encountered"] = len(encountered_exclusions)
    stats["selected_documents"] = len(selected)
    return {split: [row for row in selected if row["split"] == split] for split in SPLITS}, stats


def prepare(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError("output directory must be fresh")
    if not args.exclude_inputs:
        raise ValueError("explicit exclusion input files are required")
    config_path = args.checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    if (not str(config.get("model_type", "")).startswith("qwen")
            or config.get("max_position_embeddings") != NATIVE_LENGTH):
        raise ValueError("requires a Qwen checkpoint with config Native length 32768")
    excluded, exclusion_receipts = load_input_exclusions(args.exclude_inputs)
    tokenizer = common.load_tokenizer(args.checkpoint)
    bos = choose_bos(tokenizer, config)
    token_files = [{"name": Path(row["name"]).name, "sha256": row["sha256"]}
                   for row in common.tokenizer_file_receipts(args.checkpoint)]
    source_hash = common.sha256_file(args.source)
    rows, scan = build_rows(common.iter_source_texts(args.source, START_ROW), tokenizer, bos, excluded)
    source_sets = {split: {row["source_text_sha256"] for row in values} for split, values in rows.items()}
    if (source_sets["calibration"] & source_sets["confirmation"]
            or any(values & excluded for values in source_sets.values())):
        raise ValueError("source document disjointness failed")
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split in SPLITS:
        path = args.output / f"{split}.jsonl"
        temporary = path.with_suffix(".jsonl.incomplete")
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows[split]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        temporary.replace(path)
        files[split] = {"path": path.name, "sha256": common.sha256_file(path), "rows": len(rows[split]),
                        "source_document_set_sha256": common.canonical_hash(sorted(source_sets[split]))}
    manifest = {
        "status": "NATIVE_QK_CALIBRATION_DATA_READY_V1", "native_length": NATIVE_LENGTH,
        "target_tokens": TARGET_TOKENS, "files": files, "bos": bos,
        "model_type": config["model_type"], "config_file": config_path.name,
        "config_sha256": common.sha256_file(config_path), "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_files": token_files, "tokenizer_files_sha256": common.canonical_hash(token_files),
        "source": {"name": args.source.name, "sha256": source_hash, "start_row_inclusive_zero_based": START_ROW},
        "exclusion_inputs": exclusion_receipts, "unique_excluded_source_documents": len(excluded),
        "source_scan": scan,
        "selection": "first 16 unique nonexcluded documents with at least 32768 tokens in source order; first 8 calibration, next 8 confirmation; no loss or model-output filtering",
        "token_contract": "BOS plus first 32767 document tokens; final 256 tokens are targets; no chat template and no appended EOS",
        "disjointness": {"calibration_confirmation_documents": True,
                         "selected_vs_explicit_exclusions": True,
                         "historical_scope": "only source hashes in supplied natural-input JSONL; no claim about unsupplied historical data"},
        "model_evaluation_status": "NOT_RUN", "rope_tables_constructed": False,
        "p3_scope": "conditional asset; explicit P3 opening after K32 crossing confirmation is required before model execution",
        "input_ids_sha256_encoding": "contiguous little-endian signed int64 token bytes",
        "script_sha256": common.sha256_file(Path(__file__)),
        "utility_script_sha256": common.sha256_file(Path(common.__file__)),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exclude-inputs", type=Path, nargs="+", action="extend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
