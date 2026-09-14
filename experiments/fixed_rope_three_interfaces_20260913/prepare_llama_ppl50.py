#!/usr/bin/env python3
"""Tokenize the existing 50-document held-out corpus for Llama PPL curves.

This is CPU-only data preparation.  It preserves the 36 ProofPile-test and 14
PG19-test source identities already present on the server and creates one
prefix-aligned 8K/16K/32K evaluation array.  It never reads model outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


LENGTHS = (8192, 16384, 32768)
DOCUMENTS = 50
DATASET_COUNTS = {"proofpile": 36, "pg19": 14}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_sources(source_root: Path, source_manifest: Path) -> list[dict]:
    payload = json.loads(source_manifest.read_text())
    docs = payload.get("docs")
    if not isinstance(docs, list) or len(docs) != DOCUMENTS:
        raise ValueError("held-out source manifest must contain exactly 50 documents")
    if Counter(record.get("dataset") for record in docs) != Counter(DATASET_COUNTS):
        raise ValueError("held-out corpus must preserve the frozen 36 ProofPile/14 PG19 split")
    seen = set()
    validated = []
    for index, record in enumerate(docs):
        name = str(record.get("file", ""))
        path = source_root / name
        if not name or name in seen or not path.is_file():
            raise ValueError(f"missing or repeated held-out source at index {index}: {name}")
        seen.add(name)
        actual_sha = sha_file(path)
        if actual_sha != record.get("sha256"):
            raise ValueError(f"held-out source hash drift: {name}")
        validated.append({**record, "path": str(path), "verified_sha256": actual_sha})
    return validated


def prepare(
    *, model: Path, source_root: Path, source_manifest: Path, out: Path,
) -> dict:
    if out.exists():
        raise FileExistsError(out)
    from transformers import AutoTokenizer

    config = json.loads((model / "config.json").read_text())
    identity = (
        config.get("model_type"), config.get("hidden_size"),
        config.get("num_hidden_layers"), config.get("num_attention_heads"),
        config.get("num_key_value_heads"), config.get("rope_theta"),
        config.get("max_position_embeddings"), config.get("rope_scaling"),
    )
    expected = ("llama", 4096, 32, 32, 8, 500000.0, 8192, None)
    if identity != expected:
        raise ValueError(f"checkpoint identity {identity!r} != {expected!r}")
    docs = validate_sources(source_root, source_manifest)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    arrays = []
    records = []
    required_tokens = max(LENGTHS) + 1
    for index, record in enumerate(docs):
        path = Path(record["path"])
        ids = tokenizer.encode(
            path.read_text(encoding="utf-8", errors="ignore"),
            add_special_tokens=False,
        )
        if len(ids) < required_tokens:
            raise ValueError(
                f"held-out source {path.name} has {len(ids)} Llama tokens; "
                f"need {required_tokens}"
            )
        arrays.append(np.asarray(ids[:required_tokens], dtype=np.int64))
        records.append({
            "document": index,
            "dataset": record["dataset"],
            "split": record["split"],
            "file": path.name,
            "source_sha256": record["verified_sha256"],
            "source_row": record.get("source_row"),
            "available_llama_tokens": len(ids),
            "used_prefix_tokens": required_tokens,
        })
    out.mkdir(parents=True)
    array_path = out / "lm.npy"
    np.save(array_path, np.stack(arrays), allow_pickle=False)
    manifest = {
        "status": "COMPLETE",
        "contract": "TAILSPLINE_LLAMA_PPL50_V1",
        "model": str(model),
        "model_config_sha256": sha_file(model / "config.json"),
        "tokenizer_sha256": sha_file(model / "tokenizer.json"),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha_file(source_manifest),
        "documents": DOCUMENTS,
        "dataset_counts": DATASET_COUNTS,
        "lengths": list(LENGTHS),
        "array_shape": [DOCUMENTS, required_tokens],
        "array_dtype": "int64",
        "lm_array_sha256": sha_file(array_path),
        "lm_evaluation": {"dev": str(array_path)},
        "evaluation_panels": {"dev": []},
        "document_records": records,
        "aggregation": (
            "token-weighted NLL per length; report ProofPile and PG19 separately, "
            "with their preregistered source-equal diagnostic"
        ),
        "selection_uses_model_outputs": False,
        "model_execution": False,
    }
    atomic_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        model=args.model.resolve(), source_root=args.source_root.resolve(),
        source_manifest=args.source_manifest.resolve(), out=args.out.resolve(),
    )
    print(json.dumps({
        "status": result["status"], "documents": result["documents"],
        "lengths": result["lengths"], "out": str(args.out.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
