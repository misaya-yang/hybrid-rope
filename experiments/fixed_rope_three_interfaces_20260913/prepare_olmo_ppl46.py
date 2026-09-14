#!/usr/bin/env python3
"""Tokenize the frozen 46-document corpus for OLMo 4K/8K/16K PPL curves."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .prepare_llama_ppl46 import (
    DATASET_COUNTS,
    DOCUMENTS,
    atomic_json,
    sha_file,
    validate_sources,
)


LENGTHS = (4096, 8192, 16384)


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
    expected = ("olmo2", 2048, 16, 16, 16, 500000, 4096, None)
    if identity != expected:
        raise ValueError(f"checkpoint identity {identity!r} != {expected!r}")
    docs = validate_sources(source_root, source_manifest)
    if Counter(record["dataset"] for record in docs) != Counter(DATASET_COUNTS):
        raise ValueError("held-out source split drift")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    required_tokens = max(LENGTHS) + 1
    arrays = []
    records = []
    for index, record in enumerate(docs):
        path = Path(record["path"])
        ids = tokenizer.encode(
            path.read_text(encoding="utf-8", errors="ignore"),
            add_special_tokens=False,
        )
        if len(ids) < required_tokens:
            raise ValueError(
                f"held-out source {path.name} has {len(ids)} OLMo tokens; "
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
            "available_olmo_tokens": len(ids),
            "used_prefix_tokens": required_tokens,
        })
    out.mkdir(parents=True)
    array_path = out / "lm.npy"
    np.save(array_path, np.stack(arrays), allow_pickle=False)
    manifest = {
        "status": "COMPLETE",
        "contract": "TAILSPLINE_OLMO_PPL46_V1",
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
