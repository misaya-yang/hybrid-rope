#!/usr/bin/env python3
"""Materialise the firewall-disjoint D/S/T splits for the success-first tournament.

Builds three development/selection/confirmation splits from one FineWeb-Edu
shard (``ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830`` §5): 64 development,
64 selection, and 128 final-confirmation documents, each tokenised into physical
``1x/2x/4x`` rows so a static table's ``4x`` forward yields every endpoint and a
standalone ``1x`` forward backs the parity smoke.

Disjointness is enforced by construction and by receipt:

*   documents are consumed in stream order and assigned to exactly one split;
*   every historical prior manifest's selected documents are excluded;
*   any extra exclusion-hash files (for example recovered R0 or learned-direction
    source documents) are excluded;
*   documents shorter than 16384 tokens are skipped.

If the corpus cannot supply the required disjoint documents, the script stops
rather than silently reusing outcome-seen rows.
"""

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


STATUS = "SUCCESS_FIRST_SPLITS_READY_V1"
SPLIT_SIZES = {"D": 64, "S": 64, "T": 128}
NATIVE_LENGTH = 4096
TAIL_TOKENS = 1024
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


def load_prior_hashes(prior_manifests: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    hashes: set[str] = set()
    receipts = []
    for prior_path_arg in prior_manifests:
        prior_path = prior_path_arg.expanduser().resolve()
        manifest = json.loads(prior_path.read_text(encoding="utf-8"))
        rows_rel = manifest.get("pg19", {}).get("rows_path") or manifest.get("rows_path")
        rows_path = (prior_path.parent / rows_rel).resolve() if rows_rel else None
        if rows_path is None or not rows_path.is_file():
            raise RuntimeError(f"prior manifest has no resolvable rows: {prior_path}")
        if manifest.get("rows_sha256") and sha256_file(rows_path) != manifest["rows_sha256"]:
            raise RuntimeError(f"prior rows hash drift: {prior_path}")
        prior_hashes = set()
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prior_hashes.add(str(json.loads(line)["source_text_sha256"]))
        hashes.update(prior_hashes)
        receipts.append({
            "manifest": str(prior_path),
            "manifest_sha256": sha256_file(prior_path),
            "documents": len(prior_hashes),
            "selected_document_set_sha256": canonical_sha256(sorted(prior_hashes)),
        })
    return hashes, receipts


def load_extra_exclusions(files: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in files:
        resolved = path.expanduser().resolve()
        for line in resolved.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if token and not token.startswith("#"):
                hashes.add(token)
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-text-sha256-file", type=Path, action="append", default=[])
    parser.add_argument("--skip-eligible-documents", type=int, default=0)
    return parser.parse_args()


def write_split_rows(rows_path: Path, docs: list[dict[str, Any]]) -> None:
    with rows_path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            for multiplier in (1, 2, 4):
                length = NATIVE_LENGTH * multiplier
                input_ids = doc["input_ids"][:length]
                row = {
                    "task": "pg19",
                    "family": "pg19",
                    "multiplier": multiplier,
                    "split": doc["split"],
                    "input_ids": input_ids,
                    "nll_target_start": length - TAIL_TOKENS,
                    "nll_target_tokens": TAIL_TOKENS,
                    "source_row": doc["source_row"],
                    "source_text_sha256": doc["source_text_sha256"],
                    "input_sha256": canonical_sha256(input_ids),
                }
                row["row_sha256"] = canonical_sha256(
                    {k: v for k, v in row.items() if k != "input_ids"}
                )
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    observed_sha = sha256_file(source)
    if observed_sha != str(args.expected_source_sha256):
        raise RuntimeError("source shard hash drift")

    prior_hashes, prior_receipts = load_prior_hashes(args.prior_manifest)
    extra_hashes = load_extra_exclusions(args.exclude_text_sha256_file)
    tokenizer_sha = tokenizer_digest(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )

    total_needed = sum(SPLIT_SIZES.values())
    excluded = EXCLUDED | prior_hashes | extra_hashes
    docs: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    source_row = 0
    eligible_seen = 0

    for batch in pq.ParquetFile(source).iter_batches(batch_size=16, columns=["text"]):
        texts = [str(value) for value in batch.column(0).to_pylist()]
        encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for text, ids in zip(texts, encoded):
            text_sha = hashlib.sha256(text.encode()).hexdigest()
            row_index = source_row
            source_row += 1
            if text_sha in excluded or len(ids) < NATIVE_LENGTH * 4:
                continue
            if eligible_seen < int(args.skip_eligible_documents):
                eligible_seen += 1
                continue
            eligible_seen += 1
            if text_sha in selected_hashes:
                continue
            selected_hashes.add(text_sha)
            docs.append({
                "split": None,
                "source_row": row_index,
                "source_text_sha256": text_sha,
                "input_ids": [int(v) for v in ids],
            })
            if len(docs) >= total_needed:
                break
        if len(docs) >= total_needed:
            break

    if len(docs) < total_needed:
        raise RuntimeError(
            f"corpus cannot supply {total_needed} disjoint documents (found {len(docs)}); "
            "stop and acquire a new owner-backed shard"
        )

    splits: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for name, size in SPLIT_SIZES.items():
        chunk = docs[cursor : cursor + size]
        for doc in chunk:
            doc["split"] = name
        splits[name] = chunk
        cursor += size

    firewall: dict[str, Any] = {}
    for name, chunk in splits.items():
        split_hashes = sorted(doc["source_text_sha256"] for doc in chunk)
        rows_path = output / f"rows_{name}.jsonl"
        write_split_rows(rows_path, chunk)
        manifest = {
            "status": STATUS,
            "split": name,
            "documents": len(chunk),
            "native_context_length": NATIVE_LENGTH,
            "tail_tokens": TAIL_TOKENS,
            "eval_row_schema": "fineweb_pg19_tail_nll_v1",
            "source_shard_sha256": observed_sha,
            "tokenizer_sha256": tokenizer_sha,
            "selected_document_set_sha256": canonical_sha256(split_hashes),
            "rows_sha256": sha256_file(rows_path),
            "pg19": {"rows_path": rows_path.name},
        }
        atomic_json(output / f"manifest_{name}.json", manifest)
        firewall[name] = {
            "manifest": str(output / f"manifest_{name}.json"),
            "rows": str(rows_path),
            "documents": len(chunk),
            "selected_document_set_sha256": manifest["selected_document_set_sha256"],
        }

    # Pairwise disjointness is guaranteed by construction; assert it for the receipt.
    sets = {name: {doc["source_text_sha256"] for doc in chunk} for name, chunk in splits.items()}
    for a in sets:
        for b in sets:
            if a != b and sets[a] & sets[b]:
                raise RuntimeError(f"splits {a} and {b} are not disjoint")

    receipt = {
        "status": STATUS,
        "source_shard_sha256": observed_sha,
        "tokenizer_sha256": tokenizer_sha,
        "split_sizes": SPLIT_SIZES,
        "prior_split_receipts": prior_receipts,
        "extra_exclusion_hashes": len(extra_hashes),
        "skip_eligible_documents": int(args.skip_eligible_documents),
        "splits": firewall,
        "pairwise_disjoint": True,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "firewall_receipt.json", receipt)
    print(json.dumps({
        "status": STATUS,
        "splits": {name: info["documents"] for name, info in firewall.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
