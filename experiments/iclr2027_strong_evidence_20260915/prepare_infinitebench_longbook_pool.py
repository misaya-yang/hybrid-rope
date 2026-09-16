#!/usr/bin/env python3
"""Freeze the longest unique official InfiniteBench LongBook contexts for PPL."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path


SOURCE_URL = "https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=32)
    args = parser.parse_args()
    if args.documents <= 0:
        raise ValueError("documents must be positive")
    source_sha = sha256(args.source)
    manifest_path = args.out / "sources.json"
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text())
        if (
            value.get("status") == "INFINITEBENCH_LONGBOOK_LONGEST_POOL_READY_V1"
            and value.get("source_sha256") == source_sha
            and len(value.get("docs", [])) == args.documents
            and all((args.out / row["file"]).is_file() for row in value["docs"])
        ):
            print(json.dumps({"status": "SKIP_COMPLETE", "documents": args.documents}))
            return
        raise FileExistsError("existing LongBook context pool differs")

    unique: dict[str, tuple[int, str]] = {}
    with args.source.open(encoding="utf-8") as stream:
        for source_row, line in enumerate(stream):
            row = json.loads(line)
            context = row.get("context")
            if not isinstance(context, str) or not context:
                continue
            digest = text_sha256(context)
            unique.setdefault(digest, (source_row, context))
    heap: list[tuple[int, int, str, str]] = []
    for digest, (source_row, context) in unique.items():
        item = (len(context), source_row, digest, context)
        if len(heap) < args.documents:
            heapq.heappush(heap, item)
        elif item[:3] > heap[0][:3]:
            heapq.heapreplace(heap, item)
    if len(heap) != args.documents:
        raise ValueError(f"only {len(heap)} unique LongBook contexts are available")
    selected = sorted(heap, key=lambda item: item[1])
    args.out.mkdir(parents=True, exist_ok=True)
    docs = []
    for char_count, source_row, digest, context in selected:
        name = f"longbook_context_{source_row:04d}_{digest[:12]}.txt"
        path = args.out / name
        temporary = path.with_name(path.name + ".incomplete")
        temporary.write_text(context, encoding="utf-8")
        os.replace(temporary, path)
        docs.append({
            "dataset": "infinitebench_longbook", "split": "official",
            "file": name, "sha256": sha256(path), "source_row": source_row,
            "full_text_sha256": digest, "full_text_chars": char_count,
            "source_url": SOURCE_URL, "source_file_sha256": source_sha,
        })
    manifest = {
        "status": "INFINITEBENCH_LONGBOOK_LONGEST_POOL_READY_V1",
        "selection": (
            f"The {args.documents} longest unique official longbook_qa_eng contexts by Unicode "
            "character count; context hash deduplication; no question, answer, packing or concatenation."
        ),
        "no_packing": True,
        "source_sha256": source_sha,
        "source_url": SOURCE_URL,
        "unique_contexts": len(unique),
        "docs": docs,
    }
    temporary = manifest_path.with_name(manifest_path.name + ".incomplete")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps({
        "status": manifest["status"], "documents": len(docs),
        "unique_contexts": len(unique),
        "minimum_chars": min(row["full_text_chars"] for row in docs),
        "maximum_chars": max(row["full_text_chars"] for row in docs),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
