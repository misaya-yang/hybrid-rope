#!/usr/bin/env python3
"""Prepare ten frozen 128K ProofPile documents for the Llama S=16 gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np


LENGTH = 131072
DOCUMENTS = 10
SEED = 20260930


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_eligible(indices: list[int], *, count=DOCUMENTS, seed=SEED) -> list[int]:
    if len(indices) < count:
        raise ValueError(f"only {len(indices)} ProofPile documents reach 128K; need {count}")
    shuffled = list(indices)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:count])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text())
        if value.get("status") == "COMPLETE" and value.get("documents") == DOCUMENTS:
            print(json.dumps({"status": "SKIP_COMPLETE", "documents": DOCUMENTS}))
            return
        raise FileExistsError("PPL10 output contains a different or incomplete manifest")

    from transformers import AutoTokenizer

    config = json.loads((args.model / "config.json").read_text())
    identity = (
        config.get("model_type"), config.get("hidden_size"),
        config.get("num_hidden_layers"), config.get("num_attention_heads"),
        config.get("num_key_value_heads"), config.get("rope_theta"),
        config.get("max_position_embeddings"), config.get("rope_scaling"),
    )
    expected = ("llama", 4096, 32, 32, 8, 500000.0, 8192, None)
    if identity != expected:
        raise ValueError(f"checkpoint identity {identity!r} != {expected!r}")
    source_payload = json.loads(args.source_manifest.read_text())
    records = source_payload.get("docs")
    if not isinstance(records, list) or Counter(row.get("dataset") for row in records)["proofpile"] != 32:
        raise ValueError("source manifest must preserve the frozen ProofPile32 pool")

    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    required = LENGTH + 1
    tokenized = {}
    eligible = []
    for index, record in enumerate(records):
        if record.get("dataset") != "proofpile":
            continue
        path = args.source_root / str(record["file"])
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError(f"ProofPile source identity drift: {path}")
        values = tokenizer.encode(
            path.read_text(encoding="utf-8", errors="ignore"), add_special_tokens=False,
        )
        if len(values) >= required:
            eligible.append(index)
            tokenized[index] = values
    selected = select_eligible(eligible)
    array = np.stack([
        np.asarray(tokenized[index][:required], dtype=np.int64) for index in selected
    ])
    output.mkdir(parents=True)
    array_path = output / "lm.npy"
    np.save(array_path, array, allow_pickle=False)
    selected_records = []
    for document, index in enumerate(selected):
        record = records[index]
        selected_records.append({
            "document": document,
            "source_manifest_index": index,
            "dataset": "proofpile",
            "split": record.get("split"),
            "file": record["file"],
            "source_sha256": record["sha256"],
            "available_llama_tokens": len(tokenized[index]),
            "used_prefix_tokens": required,
        })
    manifest = {
        "status": "COMPLETE",
        "contract": "TAILSPLINE_LLAMA_S16_PROOFPILE10_128K_V1",
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256(args.model / "config.json"),
        "tokenizer_sha256": sha256(args.model / "tokenizer.json"),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "eligible_proofpile_documents": len(eligible),
        "selection_seed": SEED,
        "selection_uses_model_outputs": False,
        "documents": DOCUMENTS,
        "lengths": [LENGTH],
        "array_shape": list(array.shape),
        "array_dtype": str(array.dtype),
        "lm_array_sha256": sha256(array_path),
        "lm_evaluation": {"dev": str(array_path)},
        "evaluation_panels": {"dev": []},
        "document_records": selected_records,
        "aggregation": "token-weighted 128K NLL/PPL; paired document bootstrap",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "documents": DOCUMENTS, "eligible": len(eligible)}))


if __name__ == "__main__":
    main()
