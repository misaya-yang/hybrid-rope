#!/usr/bin/env python3
"""Build a deterministic, document-disjoint 32K PG19 NLL anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    SEED,
    sha256_file,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_eval_data import (
    PG19_FILES,
    PG19_REPOSITORY,
    PG19_REVISION,
)


LENGTH = 32_768
DEFAULT_COUNT = 64
SELECTION_SEED = SEED + 3
TOKENIZER_FILES = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def tokenizer_receipt(root: Path) -> dict[str, Any]:
    files = {
        name: sha256_file(root / name)
        for name in TOKENIZER_FILES
    }
    return {
        "files": files,
        "combined_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    args = parser.parse_args()

    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    raw_root = args.raw_root.resolve()
    tokenizer_root = args.tokenizer.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    candidates: list[tuple[str, np.ndarray]] = []
    sources = []
    for source in PG19_FILES:
        path = raw_root / source["path"]
        if path.stat().st_size != source["bytes"]:
            raise RuntimeError(f"PG19 byte-size drift: {path}")
        digest = sha256_file(path)
        if digest != source["sha256"]:
            raise RuntimeError(f"PG19 SHA-256 drift: {path}")
        eligible = 0
        rows = 0
        for batch in parquet.ParquetFile(path).iter_batches(
            batch_size=1,
            columns=["text"],
        ):
            token_ids = tokenizer.encode(
                batch.column(0)[0].as_py(),
                add_special_tokens=False,
            )
            if len(token_ids) >= LENGTH:
                candidates.append(
                    (
                        f"pg19/{source['split']}/{rows:06d}",
                        np.asarray(token_ids[:LENGTH], dtype=np.uint32),
                    )
                )
                eligible += 1
            rows += 1
        sources.append(
            {
                "split": source["split"],
                "path": source["path"],
                "bytes": source["bytes"],
                "sha256": digest,
                "rows": rows,
                "eligible_32k_documents": eligible,
            }
        )
    random.Random(SELECTION_SEED).shuffle(candidates)
    if len(candidates) < args.count:
        raise RuntimeError(
            f"only {len(candidates)} PG19 documents contain {LENGTH} tokens"
        )
    selected = candidates[: args.count]
    array = np.stack([tokens for _, tokens in selected])
    anchor_path = output / "long_documents_32k.uint32.npy"
    np.save(anchor_path, array, allow_pickle=False)
    metadata_path = output / "long_documents_32k.uint32.metadata.json"
    write_json(
        metadata_path,
        [
            {"row": index, "source": source, "token_offset": 0}
            for index, (source, _) in enumerate(selected)
        ],
    )
    manifest = {
        "status": "EVAL32K_VERIFIED",
        "held_out": True,
        "repository": PG19_REPOSITORY,
        "revision": PG19_REVISION,
        "selection_seed": SELECTION_SEED,
        "one_window_per_document": True,
        "document_boundary_crossing": False,
        "tokenizer": tokenizer_receipt(tokenizer_root),
        "sources": sources,
        "anchor": {
            "path": anchor_path.name,
            "sha256": sha256_file(anchor_path),
            "metadata_path": metadata_path.name,
            "metadata_sha256": sha256_file(metadata_path),
            "rows": args.count,
            "length": LENGTH,
        },
    }
    write_json(output / "eval32k_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
