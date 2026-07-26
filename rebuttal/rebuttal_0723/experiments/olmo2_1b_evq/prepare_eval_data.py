#!/usr/bin/env python3
"""Prepare fixed held-out OLMo/Dolma2 natural-text evaluation anchors."""

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
    TOKENIZER_MARKERS,
    sha256_file,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.prepare_official_stream import (
    range_get,
    remote_size,
)


EVAL_ROOT = (
    "https://olmo-data.org/eval-data/perplexity/"
    "v3_small_dolma2-tokenizer"
)
SOURCES = (
    "c4_en",
    "dolma_books",
    "dolma_common-crawl",
    "dolma_pes2o",
    "dolma_reddit",
    "dolma_stack",
    "dolma_wiki",
    "ice",
    "m2d2_s2orc",
    "pile",
    "wikitext_103",
)
LONG_LENGTH = 16_384
SHORT_LENGTH = 4_096
PG19_REPOSITORY = "emozilla/pg19"
PG19_REVISION = "b7bca68072ef1d86348f080bbda0996648d94315"
PG19_FILES = (
    {
        "split": "test",
        "path": "data/test-00000-of-00001-29a571947c0b5ccc.parquet",
        "bytes": 24_874_679,
        "sha256": "9aae5ddf035760257458cff08d2575d78a15f84eff867af7a87eff0681b01bfc",
    },
    {
        "split": "validation",
        "path": "data/validation-00000-of-00001-0f92e2337f79aeac.parquet",
        "bytes": 10_803_864,
        "sha256": "81680529564d4ead1c0e3859509a62d86c7126c32afc95dce6bd98e729e491ef",
    },
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def source_url(name: str) -> str:
    return f"{EVAL_ROOT}/{name}/val/part-0-00000.npy"


def download_source(name: str, output: Path) -> dict[str, Any]:
    url = source_url(name)
    size = remote_size(url)
    if size % np.dtype(np.uint32).itemsize:
        raise RuntimeError(f"{url}: invalid uint32 size")
    if not output.exists():
        payload = range_get(url, 0, size - 1)
        temporary = output.with_suffix(output.suffix + ".incomplete")
        temporary.write_bytes(payload)
        temporary.replace(output)
    if output.stat().st_size != size:
        raise RuntimeError(f"{output}: size mismatch")
    return {
        "name": name,
        "url": url,
        "path": output.name,
        "bytes": size,
        "tokens": size // np.dtype(np.uint32).itemsize,
        "sha256": sha256_file(output),
    }


def document_spans(tokens: np.ndarray) -> list[tuple[int, int]]:
    eos = TOKENIZER_MARKERS["eos_token_id"]
    boundaries = np.flatnonzero(tokens == eos)
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in boundaries:
        end = int(boundary) + 1
        if end > start:
            spans.append((start, end))
        start = end
    if start < len(tokens):
        spans.append((start, len(tokens)))
    return spans


def build_long_candidates(
    sources: list[tuple[str, np.ndarray]],
) -> list[tuple[str, int, np.ndarray]]:
    primary: list[tuple[str, int, np.ndarray]] = []
    secondary: list[tuple[str, int, np.ndarray]] = []
    for name, tokens in sources:
        for start, end in document_spans(tokens):
            if end - start < LONG_LENGTH:
                continue
            primary.append(
                (name, start, np.asarray(tokens[start : start + LONG_LENGTH]))
            )
            for offset in range(start + LONG_LENGTH, end - LONG_LENGTH + 1, LONG_LENGTH):
                secondary.append(
                    (
                        name,
                        offset,
                        np.asarray(tokens[offset : offset + LONG_LENGTH]),
                    )
                )
    rng = random.Random(SEED)
    rng.shuffle(primary)
    rng.shuffle(secondary)
    return primary + secondary


def build_short_candidates(
    sources: list[tuple[str, np.ndarray]],
) -> list[tuple[str, int, np.ndarray]]:
    candidates: list[tuple[str, int, np.ndarray]] = []
    for name, tokens in sources:
        usable = len(tokens) // SHORT_LENGTH
        for index in range(usable):
            start = index * SHORT_LENGTH
            candidates.append(
                (
                    name,
                    start,
                    np.asarray(tokens[start : start + SHORT_LENGTH]),
                )
            )
    random.Random(SEED + 1).shuffle(candidates)
    return candidates


def build_pg19_long_candidates(
    output: Path,
    tokenizer_path: Path,
) -> tuple[list[tuple[str, int, np.ndarray]], list[dict[str, Any]]]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    raw_root = output / "raw" / "pg19"
    raw_root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    candidates: list[tuple[str, int, np.ndarray]] = []
    raw_sources: list[dict[str, Any]] = []
    for source in PG19_FILES:
        downloaded = Path(
            hf_hub_download(
                repo_id=PG19_REPOSITORY,
                repo_type="dataset",
                revision=PG19_REVISION,
                filename=source["path"],
                local_dir=raw_root,
            )
        )
        if downloaded.stat().st_size != source["bytes"]:
            raise RuntimeError(f"{downloaded}: PG19 byte-size drift")
        if sha256_file(downloaded) != source["sha256"]:
            raise RuntimeError(f"{downloaded}: PG19 SHA-256 drift")
        rows = 0
        eligible = 0
        parquet_file = parquet.ParquetFile(downloaded)
        for batch in parquet_file.iter_batches(
            batch_size=1, columns=["text"]
        ):
            text = batch.column(0)[0].as_py()
            token_ids = tokenizer.encode(
                text, add_special_tokens=False
            )
            if len(token_ids) >= LONG_LENGTH:
                document_id = (
                    f"pg19/{source['split']}/{rows:06d}"
                )
                candidates.append(
                    (
                        document_id,
                        0,
                        np.asarray(
                            token_ids[:LONG_LENGTH], dtype=np.uint32
                        ),
                    )
                )
                eligible += 1
            rows += 1
        raw_sources.append(
            {
                "name": f"pg19_{source['split']}",
                "repository": PG19_REPOSITORY,
                "revision": PG19_REVISION,
                "split": source["split"],
                "path": str(downloaded.relative_to(output)),
                "bytes": source["bytes"],
                "sha256": source["sha256"],
                "rows": rows,
                "eligible_unique_documents": eligible,
            }
        )
    random.Random(SEED + 2).shuffle(candidates)
    return candidates, raw_sources


def save_anchors(
    output: Path,
    candidates: list[tuple[str, int, np.ndarray]],
    *,
    count: int,
    length: int,
) -> dict[str, Any]:
    if len(candidates) < count:
        raise RuntimeError(
            f"only {len(candidates)} candidate windows for requested {count}"
        )
    selected = candidates[:count]
    array = np.stack([row[2] for row in selected]).astype(np.uint32)
    if array.shape != (count, length):
        raise RuntimeError(f"anchor shape drift: {array.shape}")
    np.save(output, array, allow_pickle=False)
    metadata = [
        {"row": index, "source": source, "token_offset": offset}
        for index, (source, offset, _) in enumerate(selected)
    ]
    metadata_path = output.with_suffix(".metadata.json")
    write_json(metadata_path, metadata)
    return {
        "path": output.name,
        "sha256": sha256_file(output),
        "metadata_path": metadata_path.name,
        "metadata_sha256": sha256_file(metadata_path),
        "rows": count,
        "length": length,
        "source_counts": {
            name: sum(row["source"] == name for row in metadata)
            for name in sorted({row["source"] for row in metadata})
        },
    }


def validate_existing(
    manifest_path: Path,
    *,
    long_count: int,
    short_count: int,
) -> dict[str, Any]:
    output = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "EVAL_DATA_VERIFIED":
        raise RuntimeError("evaluation manifest is not verified")
    for row in manifest["sources"] + manifest.get("raw_sources", []):
        path = output / row["path"]
        if path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"{path}: byte-size drift")
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"{path}: SHA-256 drift")
    expected = {
        "long_documents": (long_count, LONG_LENGTH),
        "official_validation": (short_count, SHORT_LENGTH),
    }
    for name, (rows, length) in expected.items():
        anchor = manifest["anchors"][name]
        path = output / anchor["path"]
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if array.dtype != np.uint32 or array.shape != (rows, length):
            raise RuntimeError(f"{name}: anchor shape/dtype drift")
        if sha256_file(path) != anchor["sha256"]:
            raise RuntimeError(f"{name}: anchor SHA-256 drift")
        metadata_path = output / anchor["metadata_path"]
        if sha256_file(metadata_path) != anchor["metadata_sha256"]:
            raise RuntimeError(f"{name}: metadata SHA-256 drift")
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if len(metadata) != rows:
            raise RuntimeError(f"{name}: metadata row-count drift")
        if name == "long_documents":
            documents = [row["source"] for row in metadata]
            if len(set(documents)) != rows:
                raise RuntimeError(
                    "long anchors are not one-window-per-document"
                )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--long-count", type=int, default=128)
    parser.add_argument("--short-count", type=int, default=256)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    manifest_path = output / "eval_manifest.json"
    if args.validate_only:
        manifest = validate_existing(
            manifest_path,
            long_count=args.long_count,
            short_count=args.short_count,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    source_dir = output / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, Any]] = []
    arrays: list[tuple[str, np.ndarray]] = []
    for name in SOURCES:
        path = source_dir / f"{name}.uint32.bin"
        row = download_source(name, path)
        row["path"] = str(path.relative_to(output))
        if row["bytes"] % 4 or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"{path}: source validation failed")
        source_rows.append(row)
        arrays.append(
            (
                name,
                np.memmap(path, dtype=np.uint32, mode="r"),
            )
        )

    long_path = output / "long_documents_16k.uint32.npy"
    short_path = output / "official_validation_4k.uint32.npy"
    pg19_candidates, raw_sources = build_pg19_long_candidates(
        output, args.tokenizer_path.resolve()
    )
    anchor_rows = {
        "long_documents": save_anchors(
            long_path,
            pg19_candidates,
            count=args.long_count,
            length=LONG_LENGTH,
        ),
        "official_validation": save_anchors(
            short_path,
            build_short_candidates(arrays),
            count=args.short_count,
            length=SHORT_LENGTH,
        ),
    }
    manifest = {
        "status": "EVAL_DATA_VERIFIED",
        "tokenizer": "allenai_dolma2",
        "eos_token_id": TOKENIZER_MARKERS["eos_token_id"],
        "held_out": True,
        "selection_seed": SEED,
        "sources": source_rows,
        "raw_sources": raw_sources,
        "anchors": anchor_rows,
        "long_document_contract": {
            "corpus": PG19_REPOSITORY,
            "revision": PG19_REVISION,
            "splits": ["test", "validation"],
            "one_window_per_document": True,
            "document_boundary_crossing": False,
        },
        "primary_endpoint": (
            "paired tail-NLL difference on 8K and 16K prefixes of "
            "long_documents"
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
