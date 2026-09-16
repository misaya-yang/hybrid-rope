#!/usr/bin/env python3
"""Extract the 32 longest natural ProofPile test documents for 256K PPL."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import os
from pathlib import Path


EXPECTED_ARCHIVE_SHA256 = "b1bc923aa34b2b03db08e2f451d8442d9ca7aad1c857a8835382c94a8bb1d835"
SOURCE_URL = "https://huggingface.co/datasets/hoskinson-center/proof-pile/resolve/main/test/proofpile_test.jsonl.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=32)
    parser.add_argument("--split", default="test")
    parser.add_argument("--expected-archive-sha256", default=EXPECTED_ARCHIVE_SHA256)
    parser.add_argument("--source-url", default=SOURCE_URL)
    args = parser.parse_args()
    if args.documents <= 0:
        raise ValueError("documents must be positive")
    archive_sha = sha256(args.archive)
    if archive_sha != args.expected_archive_sha256:
        raise ValueError(f"ProofPile archive SHA256 drift: {archive_sha}")
    manifest_path = args.out / "sources.json"
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text())
        if (
            value.get("status") == "PROOFPILE256_LONGEST_POOL_READY_V1"
            and value.get("proofpile_archive_sha256") == archive_sha
            and value.get("split", "test") == args.split
            and len(value.get("docs", [])) == args.documents
            and all((args.out / row["file"]).is_file() for row in value["docs"])
        ):
            print(json.dumps({"status": "SKIP_COMPLETE", "documents": args.documents}))
            return
        raise FileExistsError("existing 256K ProofPile pool differs")

    heap: list[tuple[int, int, str, dict]] = []
    with gzip.open(args.archive, "rt", encoding="utf-8", errors="replace") as stream:
        for source_row, line in enumerate(stream):
            value = json.loads(line)
            text = value.get("text")
            if not isinstance(text, str) or not text:
                continue
            item = (len(text), source_row, text, value.get("meta") or {})
            if len(heap) < args.documents:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
    if len(heap) != args.documents:
        raise ValueError("ProofPile archive has too few nonempty documents")
    selected = sorted(heap, key=lambda item: item[1])
    args.out.mkdir(parents=True, exist_ok=True)
    docs = []
    for char_count, source_row, text, metadata in selected:
        name = f"proofpile_{args.split}_{source_row:06d}.txt"
        path = args.out / name
        temporary = path.with_name(path.name + ".incomplete")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        digest = sha256(path)
        docs.append({
            "dataset": "proofpile", "split": args.split, "file": name,
            "sha256": digest, "source_row": source_row,
            "source_archive_sha256": archive_sha, "full_text_sha256": digest,
            "full_text_chars": char_count, "source_metadata": {"meta": metadata},
            "source_url": args.source_url,
        })
    manifest = {
        "status": "PROOFPILE256_LONGEST_POOL_READY_V1",
        "selection": (
            f"The {args.documents} longest nonempty ProofPile test rows by Unicode character count; "
            "model-independent selection; no packing or concatenation."
        ),
        "no_packing": True,
        "proofpile_archive_sha256": archive_sha,
        "source_url": args.source_url,
        "split": args.split,
        "docs": docs,
    }
    temporary = manifest_path.with_name(manifest_path.name + ".incomplete")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps({
        "status": manifest["status"], "documents": len(docs),
        "minimum_chars": min(row["full_text_chars"] for row in docs),
        "maximum_chars": max(row["full_text_chars"] for row in docs),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
