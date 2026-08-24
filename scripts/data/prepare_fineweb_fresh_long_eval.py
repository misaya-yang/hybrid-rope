#!/usr/bin/env python3
"""Prepare deterministic fresh 4K/8K/16K FineWeb-Edu NLL rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer


STATUS = "FINEWEB_EDU_FRESH_LONG_EVAL_READY_V1"
EXCLUDED = {
    "23b0db7860e2ad9bafc74179007831877fd6f6f9aa626b6ec25754685d93e1e3",
    "d2496bb13d60ea3d53676ce0a92eb347bb7dc63cb98ce6b29ddfbd6befc29c18",
    "d2b1197b8a292a64e335715ea342e4e157cdb156fc7406eb52c5ffee4e6d9b5b",
    "761a5525725a62e6526c917daa572e1041b530268a29034abe9b2f67f442e2f8",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".incomplete",
        mode="w", encoding="utf-8", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def tokenizer_digest(checkpoint: Path) -> str:
    files = [
        path for path in sorted(checkpoint.iterdir())
        if path.name.startswith("tokenizer") or path.name == "special_tokens_map.json"
    ]
    return canonical_sha256([
        {"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in files if path.is_file()
    ])


def manifest_rows(manifest_path: Path) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_path = (manifest_path.parent / manifest["pg19"]["rows_path"]).resolve()
    if sha256_file(rows_path) != manifest["rows_sha256"]:
        raise RuntimeError(f"prior rows hash drift: {manifest_path}")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    return manifest, rows_path, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=32)
    parser.add_argument("--skip-eligible-documents", type=int, default=0)
    parser.add_argument("--tail-tokens", type=int, default=1024)
    parser.add_argument(
        "--prior-manifest", type=Path, action="append", default=[],
        help="Manifest whose source documents must be excluded from this split (repeatable).",
    )
    args = parser.parse_args()
    source = args.source.resolve(); checkpoint = args.checkpoint.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"; rows_path = output / "rows.jsonl"
    observed_sha = sha256_file(source)
    if observed_sha != str(args.expected_source_sha256):
        raise RuntimeError("source shard hash drift")
    tokenizer_sha = tokenizer_digest(checkpoint)
    prior_text_sha256: set[str] = set()
    prior_receipts: list[dict[str, Any]] = []
    for prior_path_arg in args.prior_manifest:
        prior_path = prior_path_arg.expanduser().resolve()
        _, _, prior_rows = manifest_rows(prior_path)
        prior_hashes = {str(row["source_text_sha256"]) for row in prior_rows}
        prior_text_sha256.update(prior_hashes)
        prior_receipts.append({
            "manifest_sha256": sha256_file(prior_path),
            "selected_document_set_sha256": canonical_sha256(sorted(prior_hashes)),
        })
    if manifest_path.exists():
        manifest, existing_rows_path, existing_rows = manifest_rows(manifest_path)
        expected = {
            "status": STATUS,
            "source_shard_sha256": observed_sha,
            "tokenizer_sha256": tokenizer_sha,
            "eval_row_schema": "fineweb_pg19_tail_nll_v1",
            "append_eos": False,
            "skip_eligible_documents": int(args.skip_eligible_documents),
            "documents": int(args.documents),
            "tail_tokens": int(args.tail_tokens),
            "prior_split_receipts": prior_receipts,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"existing manifest identity drift for {key}")
        if existing_rows_path != rows_path or len(existing_rows) != int(args.documents) * 3:
            raise RuntimeError("existing manifest row-count/path drift")
        existing_hashes = {str(row["source_text_sha256"]) for row in existing_rows}
        if (
            existing_hashes & prior_text_sha256
            or manifest.get("selected_document_set_sha256")
            != canonical_sha256(sorted(existing_hashes))
        ):
            raise RuntimeError("existing manifest split-disjointness drift")
        print(manifest_path.read_text(), end="")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    selected = 0; source_row = 0; eligible_seen = 0; selected_hashes: set[str] = set()
    with rows_path.open("w", encoding="utf-8") as handle:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=16, columns=["text"]):
            texts = [str(value) for value in batch.column(0).to_pylist()]
            encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            for text, ids in zip(texts, encoded):
                text_sha = hashlib.sha256(text.encode()).hexdigest()
                row_index = source_row; source_row += 1
                if text_sha in EXCLUDED or text_sha in prior_text_sha256 or len(ids) < 16384:
                    continue
                if eligible_seen < int(args.skip_eligible_documents):
                    eligible_seen += 1
                    continue
                eligible_seen += 1
                if text_sha in selected_hashes:
                    continue
                for multiplier in (1, 2, 4):
                    length = 4096 * multiplier
                    input_ids = [int(value) for value in ids[:length]]
                    row = {
                        "task": "pg19", "family": "pg19", "multiplier": multiplier,
                        "input_ids": input_ids,
                        "nll_target_start": length - int(args.tail_tokens),
                        "nll_target_tokens": int(args.tail_tokens),
                        "source_row": row_index, "source_text_sha256": text_sha,
                        "input_sha256": canonical_sha256(input_ids),
                    }
                    row["row_sha256"] = canonical_sha256({k: v for k, v in row.items() if k != "input_ids"})
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                selected_hashes.add(text_sha)
                selected += 1
                if selected >= int(args.documents): break
            if selected >= int(args.documents): break
        handle.flush(); os.fsync(handle.fileno())
    if selected != int(args.documents):
        raise RuntimeError("not enough 16K documents")
    atomic_json(manifest_path, {
        "status": STATUS, "tokenization_executed": True,
        "native_context_length": 4096,
        "source_identity": f"FineWeb-Edu/{source.name}",
        "source_shard_sha256": observed_sha,
        "tokenizer_sha256": tokenizer_sha,
        "eval_row_schema": "fineweb_pg19_tail_nll_v1",
        "append_eos": False,
        "excluded_prior_text_sha256": sorted(EXCLUDED),
        "prior_split_receipts": prior_receipts,
        "prior_selected_documents": len(prior_text_sha256),
        "skip_eligible_documents": int(args.skip_eligible_documents),
        "documents": selected, "tail_tokens": int(args.tail_tokens),
        "selected_document_set_sha256": canonical_sha256(sorted(selected_hashes)),
        "rows_sha256": sha256_file(rows_path),
        "pg19": {"rows_path": rows_path.name}, "longbench": {"cells": {}},
    })
    print(manifest_path.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
