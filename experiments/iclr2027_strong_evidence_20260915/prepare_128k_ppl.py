#!/usr/bin/env python3
"""Freeze a small model-tokenized long-document panel for exact NLL/PPL."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--documents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--dataset", action="append",
                        help="source-manifest dataset label to include; default proofpile")
    args = parser.parse_args()
    if args.length <= 0 or args.documents <= 0 or args.seed < 0:
        raise ValueError("length/documents must be positive and seed nonnegative")

    output = args.out.resolve()
    manifest_path = output / "manifest.json"
    selected_datasets = tuple(args.dataset or ("proofpile",))
    request = {
        "contract": (
            "MODEL_TOKENIZED_LONG_CONTEXT_PPL_V1"
            if args.dataset else "MODEL_TOKENIZED_PROOFPILE_128K_PPL_V1"
        ),
        "model_id": args.model_id,
        "model_config_sha256": sha256(args.model / "config.json"),
        "source_manifest_sha256": sha256(args.source_manifest),
        "selection_seed": args.seed,
        "documents": args.documents,
        "lengths": [args.length],
    }
    if args.dataset:
        request["datasets"] = list(selected_datasets)
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text())
        if value.get("status") == "COMPLETE" and all(value.get(k) == v for k, v in request.items()):
            array_path = output / "lm.npy"
            if array_path.is_file() and value.get("lm_array_sha256") == sha256(array_path):
                print(json.dumps({"status": "SKIP_COMPLETE", "documents": args.documents}))
                return
        raise FileExistsError("PPL output contains a different or incomplete manifest")

    from transformers import AutoTokenizer

    source_payload = json.loads(args.source_manifest.read_text())
    records = source_payload.get("docs")
    counts = Counter(row.get("dataset") for row in records) if isinstance(records, list) else Counter()
    if not isinstance(records, list) or sum(counts[label] for label in selected_datasets) < args.documents:
        raise ValueError("source manifest lacks enough documents for the requested datasets")
    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True)
    required = args.length + 1
    tokenized: dict[int, list[int]] = {}
    eligible = []
    for index, record in enumerate(records):
        if record.get("dataset") not in selected_datasets:
            continue
        path = args.source_root / str(record["file"])
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError(f"long-document source identity drift: {path}")
        values = tokenizer.encode(
            path.read_text(encoding="utf-8", errors="ignore"), add_special_tokens=False,
        )
        if len(values) >= required:
            eligible.append(index)
            tokenized[index] = values
    if len(eligible) < args.documents:
        raise ValueError(f"only {len(eligible)} source documents reach {args.length}")
    shuffled = list(eligible)
    random.Random(args.seed).shuffle(shuffled)
    selected = sorted(shuffled[:args.documents])
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
            "dataset": record.get("dataset"),
            "file": record["file"],
            "source_sha256": record["sha256"],
            "available_tokens": len(tokenized[index]),
            "used_prefix_tokens": required,
        })
    tokenizer_path = args.model / "tokenizer.json"
    manifest = {
        "status": "COMPLETE",
        **request,
        "model_artifact_name": args.model.name,
        "tokenizer_sha256": sha256(tokenizer_path) if tokenizer_path.is_file() else None,
        "source_manifest_artifact": args.source_manifest.name,
        "eligible_source_documents": len(eligible),
        "source_corpus": ",".join(selected_datasets),
        "selection_uses_model_outputs": False,
        "array_shape": list(array.shape),
        "array_dtype": str(array.dtype),
        "lm_array_sha256": sha256(array_path),
        "lm_evaluation": {"dev": str(array_path)},
        "evaluation_panels": {"dev": []},
        "document_records": selected_records,
        "aggregation": f"token-weighted {args.length}-token NLL/PPL; paired document bootstrap",
    }
    temporary = manifest_path.with_name(manifest_path.name + ".incomplete")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps({"status": "COMPLETE", "documents": args.documents,
                      "eligible": len(eligible)}))


if __name__ == "__main__":
    main()
