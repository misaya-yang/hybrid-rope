#!/usr/bin/env python3
"""Prepare a fresh paired natural-text holdout, excluding all P0 input documents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.prepare_native_reference_calibration import (
    canonical_hash, ids_hash, iter_source_texts, load_tokenizer, sha256_file,
    tokenizer_file_receipts,
)

GRID = (4096, 8192, 16384)
DOCUMENTS = 32
START_ROW = 20000
TARGET_TOKENS = 256
STATUS = "REFERENCE_COUPLING_NLL_DATA_READY_V1"


def child_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("manifest path leaves its artifact directory") from error
    return path


def load_p0_exclusions(root: Path) -> tuple[set[str], dict, dict]:
    """Inspect original INPUT rows only; no decisions or model results are read."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "NATIVE_REFERENCE_CALIBRATION_DATA_READY_V1":
        raise ValueError("P0 requires its original READY input-data manifest")
    excluded, files = set(), {}
    for split, count in (("calibration", 32), ("confirmation", 64)):
        entry = manifest["files"][split]
        path = child_path(root, entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"P0 input file hash mismatch: {split}")
        samples, seen, total = {}, set(), 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                total += 1
                if row.get("split") != split:
                    raise ValueError("P0 input split drift")
                if row.get("family") != "natural":
                    continue
                sample_id, length = row["sample_id"], row["length"]
                source_hash = row["source_text_sha256"]
                if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
                    raise ValueError("invalid P0 source-text hash")
                if (sample_id, length) in seen or length not in (1024, 2048, 4096, 8192):
                    raise ValueError("P0 duplicate or unregistered natural input cell")
                seen.add((sample_id, length))
                if sample_id in samples and samples[sample_id] != source_hash:
                    raise ValueError("P0 sample identity changes across lengths")
                samples[sample_id] = source_hash
        expected_ids = {f"natural-{split}-{index:03d}" for index in range(count)}
        hashes = set(samples.values())
        if (total != entry["rows"] or entry["natural_documents"] != count
                or set(samples) != expected_ids or len(seen) != count * 4 or len(hashes) != count
                or excluded & hashes):
            raise ValueError("P0 complete, disjoint 32/64-document input identity failed")
        excluded.update(hashes)
        files[split] = {"sha256": entry["sha256"], "source_documents": count}
    return excluded, {"manifest_sha256": sha256_file(manifest_path), "files": files,
                      "excluded_documents": len(excluded),
                      "excluded_source_set_sha256": canonical_hash(sorted(excluded)),
                      "model_outcomes_read": False}, manifest


def select_documents(rows, tokenizer, excluded: set[str]) -> list[dict]:
    selected, seen = [], set(excluded)
    for source_row, text in rows:
        if source_row < START_ROW:
            continue
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if source_hash in seen:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max(GRID))
        if len(ids) < max(GRID):
            continue
        seen.add(source_hash)
        selected.append({"source_row": source_row, "source_text_sha256": source_hash,
                         "document_ids": [int(token) for token in ids[:max(GRID)]]})
        if len(selected) == DOCUMENTS:
            return selected
    raise ValueError(f"need {DOCUMENTS} fresh >=16384-token documents; found {len(selected)}")


def natural_rows(document: dict, index: int, bos_id: int):
    full = [bos_id] + document["document_ids"][:max(GRID) - 1]
    if len(full) != max(GRID):
        raise ValueError("document is shorter than the frozen holdout grid")
    for length in GRID:
        ids = [bos_id] + full[-(length - 1):]
        yield {"family": "natural", "variant": "natural", "split": "holdout",
               "sample_id": f"reference-coupling-natural-{index:03d}", "length": length,
               "input_ids": ids, "target_start": length - TARGET_TOKENS,
               "target_tokens": TARGET_TOKENS, "source_row": document["source_row"],
               "source_text_sha256": document["source_text_sha256"],
               "prompt_ids_sha256": ids_hash(ids), "target_ids_sha256": ids_hash(ids[-TARGET_TOKENS:])}


def prepare(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError("holdout output directory must be fresh")
    excluded, p0_receipt, p0_manifest = load_p0_exclusions(args.p0_data_root)
    config_path = args.checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    tokenizer_files = tokenizer_file_receipts(args.checkpoint)
    if (sha256_file(config_path) != p0_manifest["config_sha256"]
            or tokenizer_files != p0_manifest["tokenizer_files"]):
        raise ValueError("checkpoint config/tokenizer differs from original P0 input contract")
    tokenizer = load_tokenizer(args.checkpoint)
    if tokenizer.bos_token_id is None:
        raise ValueError("checkpoint tokenizer must provide BOS")
    documents = select_documents(iter_source_texts(args.source, START_ROW), tokenizer, excluded)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "holdout.jsonl"
    temporary = path.with_suffix(".jsonl.incomplete")
    with temporary.open("x") as handle:
        for index, document in enumerate(documents):
            for row in natural_rows(document, index, int(tokenizer.bos_token_id)):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)
    manifest = {
        "status": STATUS, "model_evaluation_status": "NOT_RUN", "grid": list(GRID),
        "natural_documents": DOCUMENTS, "rows": DOCUMENTS * len(GRID), "target_tokens": TARGET_TOKENS,
        "file": {"path": path.name, "sha256": sha256_file(path)},
        "config_sha256": sha256_file(config_path), "L_config": int(config["max_position_embeddings"]),
        "tokenizer_files": tokenizer_files, "bos_token_id": int(tokenizer.bos_token_id),
        "source": {"name": args.source.name, "sha256": sha256_file(args.source), "start_row": START_ROW},
        "p0_inputs": p0_receipt,
        "source_document_set_sha256": canonical_hash(sorted(d["source_text_sha256"] for d in documents)),
        "selection": "First 32 unique >=16384-token documents from source row 20000, excluding all 96 P0 natural document hashes; no loss selection",
        "target_contract": "BOS plus first 16383 document tokens; each shorter input is BOS plus final L-1 tokens; identical final 256 targets; no EOS appended",
        "disjoint_from_all_p0_natural_documents": True, "ruler_is_separate": True,
        "script_sha256": sha256_file(Path(__file__)),
        "p0_helper_sha256": sha256_file(ROOT / "scripts/data/prepare_native_reference_calibration.py"),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--p0-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
