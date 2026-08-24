#!/usr/bin/env python3
"""Tokenize a receipt-bound new FineWeb-Edu parquet shard to exactly 1B tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


STATUS = "FINEWEB_EDU_NEW_SHARD_1B_TOKENS_COMPLETE_V1"
EVAL_STATUS = "FINEWEB_EDU_NEW_SHARD_FRESH_EVAL_READY_V1"
EXCLUDED_TEXT_SHA256 = {
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
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".incomplete",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def tokenizer_digest(checkpoint: Path) -> str:
    files = [
        path for path in sorted(checkpoint.iterdir())
        if path.name.startswith("tokenizer") or path.name in {"special_tokens_map.json"}
    ]
    return canonical_sha256([
        {"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in files if path.is_file()
    ])


def eval_row(
    ids: list[int], *, source_index: int, source_row: int, text_sha: str, multiplier: int,
) -> dict[str, Any]:
    length = 4096 * int(multiplier)
    selected = [int(value) for value in ids[:length]]
    target_tokens = 512
    row = {
        "task": "pg19",
        "family": "pg19",
        "multiplier": int(multiplier),
        "input_ids": selected,
        "nll_target_start": length - target_tokens,
        "nll_target_tokens": target_tokens,
        "source_index": int(source_index),
        "source_row": int(source_row),
        "source_text_sha256": text_sha,
        "input_sha256": canonical_sha256(selected),
    }
    row["row_sha256"] = canonical_sha256({key: value for key, value in row.items() if key != "input_ids"})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-source-sha256", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--eval-documents", type=int, default=20)
    parser.add_argument("--batch-rows", type=int, default=16)
    args = parser.parse_args()

    sources = [path.expanduser().resolve() for path in args.source]
    expected_source_hashes = [str(value) for value in args.expected_source_sha256]
    if len(sources) != len(expected_source_hashes):
        raise ValueError("every source requires one expected SHA-256")
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    observed_source_hashes = [sha256_file(source) for source in sources]
    if observed_source_hashes != expected_source_hashes:
        raise RuntimeError("new FineWeb-Edu source shard hash drift")
    tokenizer_sha = tokenizer_digest(checkpoint)
    final_receipt = output / "receipt.json"
    if final_receipt.is_file():
        receipt = json.loads(final_receipt.read_text(encoding="utf-8"))
        expected_receipt = {
            "status": STATUS,
            "source_sha256": observed_source_hashes,
            "checkpoint_tokenizer_sha256": tokenizer_sha,
            "tokens": int(args.target_tokens),
            "fresh_eval_documents": int(args.eval_documents),
            "fresh_eval_row_schema": "fineweb_pg19_tail_nll_v1",
            "fresh_eval_append_eos": False,
        }
        for key, value in expected_receipt.items():
            if receipt.get(key) != value:
                raise RuntimeError(f"existing final receipt identity drift for {key}")
        artifact_receipts = {
            "tokens_file_sha256": output / "tokens.i32.bin",
            "offsets_file_sha256": output / "document_offsets.u64.bin",
            "fresh_eval_rows_sha256": output / "fresh_eval_rows.jsonl",
            "fresh_eval_manifest_sha256": output / "fresh_eval_manifest.json",
        }
        for key, path in artifact_receipts.items():
            if not path.is_file() or sha256_file(path) != receipt.get(key):
                raise RuntimeError(f"existing final artifact hash drift for {key}")
        eval_manifest = json.loads(
            artifact_receipts["fresh_eval_manifest_sha256"].read_text(encoding="utf-8")
        )
        expected_eval_manifest = {
            "status": EVAL_STATUS,
            "tokenizer_sha256": tokenizer_sha,
            "source_shard_sha256": observed_source_hashes,
            "eval_documents": int(args.eval_documents),
            "eval_rows": int(args.eval_documents) * 2,
            "eval_row_schema": "fineweb_pg19_tail_nll_v1",
            "append_eos": False,
            "rows_sha256": receipt["fresh_eval_rows_sha256"],
        }
        for key, value in expected_eval_manifest.items():
            if eval_manifest.get(key) != value:
                raise RuntimeError(f"existing fresh-eval manifest identity drift for {key}")
        print(final_receipt.read_text(encoding="utf-8"), end="")
        return 0
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer has no EOS token")

    tokens_path = output / "tokens.i32.bin"
    offsets_path = output / "document_offsets.u64.bin"
    eval_rows_path = output / "fresh_eval_rows.jsonl"
    state_path = output / "progress.json"
    state_identity = {
        "source_sha256": observed_source_hashes,
        "checkpoint_tokenizer_sha256": tokenizer_sha,
        "target_tokens": int(args.target_tokens),
        "eval_documents": int(args.eval_documents),
        "eval_row_schema": "fineweb_pg19_tail_nll_v1",
        "eval_append_eos": False,
    }
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {
            "next_source_row": 0,
            "source_index": 0,
            "tokens_written": 0,
            "documents_written": 0,
            "rolling_document_sha256": "0" * 64,
            "eval_documents_written": 0,
            **state_identity,
        }
    )
    state.setdefault("source_index", 0)
    for key, value in state_identity.items():
        if state.get(key) != value:
            raise RuntimeError(f"tokenization resume identity drift for {key}")
    expected_token_bytes = int(state["tokens_written"]) * 4
    expected_offset_bytes = int(state["documents_written"]) * 8
    for path, expected_bytes in (
        (tokens_path, expected_token_bytes), (offsets_path, expected_offset_bytes),
    ):
        observed_bytes = path.stat().st_size if path.exists() else 0
        if observed_bytes < expected_bytes:
            raise RuntimeError(f"tokenization resume file is shorter than committed state: {path}")
        if observed_bytes > expected_bytes:
            with path.open("r+b") as handle:
                handle.truncate(expected_bytes)

    complete_eval_hashes: set[str] = set()
    if eval_rows_path.is_file():
        parsed_rows = [
            json.loads(line) for line in eval_rows_path.read_text().splitlines() if line.strip()
        ]
        by_document: dict[str, dict[int, dict[str, Any]]] = {}
        for row in parsed_rows:
            text_sha = str(row["source_text_sha256"])
            multiplier = int(row["multiplier"])
            if multiplier in by_document.setdefault(text_sha, {}):
                raise RuntimeError("duplicate fresh-eval row identity during resume")
            by_document[text_sha][multiplier] = row
        complete = [
            row
            for text_sha, cells in by_document.items()
            if set(cells) == {1, 2}
            for row in (cells[1], cells[2])
        ]
        complete_eval_hashes = {
            text_sha for text_sha, cells in by_document.items() if set(cells) == {1, 2}
        }
        with tempfile.NamedTemporaryFile(
            dir=eval_rows_path.parent, prefix=eval_rows_path.name + ".",
            suffix=".incomplete", mode="w", encoding="utf-8", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in complete:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        temporary.replace(eval_rows_path)
    state["eval_documents_written"] = len(complete_eval_hashes)

    done = False
    with tokens_path.open("ab") as token_handle, offsets_path.open("ab") as offset_handle:
        for source_index in range(int(state["source_index"]), len(sources)):
            source = sources[source_index]
            parquet = pq.ParquetFile(source)
            source_row = 0
            resume_row = int(state["next_source_row"]) if source_index == int(state["source_index"]) else 0
            for batch in parquet.iter_batches(batch_size=int(args.batch_rows), columns=["text"]):
                texts = batch.column(0).to_pylist()
                batch_start = source_row
                source_row += len(texts)
                if source_row <= resume_row:
                    continue
                selected_indices = [index for index in range(len(texts)) if batch_start + index >= resume_row]
                selected_texts = [str(texts[index]) for index in selected_indices]
                encoded = tokenizer(
                    selected_texts, add_special_tokens=False,
                    return_attention_mask=False, return_token_type_ids=False,
                )["input_ids"]
                for local_index, text, ids in zip(selected_indices, selected_texts, encoded):
                    absolute_row = batch_start + local_index
                    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    state["source_index"] = source_index
                    state["next_source_row"] = absolute_row + 1
                    if text_sha in EXCLUDED_TEXT_SHA256:
                        continue
                    raw_ids = [int(value) for value in ids]
                    if (
                        int(state["eval_documents_written"]) < int(args.eval_documents)
                        and text_sha not in complete_eval_hashes
                        and len(raw_ids) >= 8192
                    ):
                        append_jsonl(eval_rows_path, eval_row(
                            raw_ids, source_index=source_index,
                            source_row=absolute_row, text_sha=text_sha, multiplier=1,
                        ))
                        append_jsonl(eval_rows_path, eval_row(
                            raw_ids, source_index=source_index,
                            source_row=absolute_row, text_sha=text_sha, multiplier=2,
                        ))
                        complete_eval_hashes.add(text_sha)
                        state["eval_documents_written"] = int(state["eval_documents_written"]) + 1
                    document_ids = raw_ids + [int(tokenizer.eos_token_id)]
                    remaining = int(args.target_tokens) - int(state["tokens_written"])
                    if remaining <= 0:
                        done = True; break
                    document_ids = document_ids[:remaining]
                    np.asarray([int(state["tokens_written"])], dtype="<u8").tofile(offset_handle)
                    np.asarray(document_ids, dtype="<i4").tofile(token_handle)
                    state["tokens_written"] = int(state["tokens_written"]) + len(document_ids)
                    state["documents_written"] = int(state["documents_written"]) + 1
                    state["rolling_document_sha256"] = hashlib.sha256(
                        (str(state["rolling_document_sha256"]) + text_sha).encode()
                    ).hexdigest()
                    if int(state["documents_written"]) % 100 == 0:
                        token_handle.flush(); offset_handle.flush()
                        os.fsync(token_handle.fileno()); os.fsync(offset_handle.fileno())
                        atomic_json(state_path, state)
                    if int(state["tokens_written"]) >= int(args.target_tokens):
                        done = True; break
                if done:
                    break
            if done:
                break
            state["source_index"] = source_index + 1
            state["next_source_row"] = 0
            atomic_json(state_path, state)
        token_handle.flush(); offset_handle.flush()
        os.fsync(token_handle.fileno()); os.fsync(offset_handle.fileno())
    atomic_json(state_path, state)
    if not done or int(state["tokens_written"]) != int(args.target_tokens):
        raise RuntimeError("source shard ended before the 1B-token target")
    if int(state["eval_documents_written"]) != int(args.eval_documents):
        raise RuntimeError("insufficient fresh 8K evaluation documents")
    eval_manifest = output / "fresh_eval_manifest.json"
    eval_manifest_value = {
        "status": EVAL_STATUS,
        "tokenization_executed": True,
        "native_context_length": 4096,
        "tokenizer_sha256": tokenizer_sha,
        "source_shard_sha256": observed_source_hashes,
        "source_shard_identity": [
            f"FineWeb-Edu sample/10BT/{source.name}" for source in sources
        ],
        "excluded_prior_text_sha256": sorted(EXCLUDED_TEXT_SHA256),
        "eval_documents": int(args.eval_documents),
        "eval_rows": int(args.eval_documents) * 2,
        "eval_row_schema": "fineweb_pg19_tail_nll_v1",
        "append_eos": False,
        "rows_sha256": sha256_file(eval_rows_path),
        "pg19": {"rows_path": eval_rows_path.name},
        "longbench": {"cells": {}},
    }
    if eval_manifest.is_file():
        existing_eval_manifest = json.loads(eval_manifest.read_text(encoding="utf-8"))
        if existing_eval_manifest != eval_manifest_value:
            raise RuntimeError("fresh-eval manifest identity drift")
    else:
        atomic_json(eval_manifest, eval_manifest_value)
    receipt = {
        "status": STATUS,
        "source_identity": [
            f"HuggingFaceFW/fineweb-edu sample/10BT/{source.name}" for source in sources
        ],
        "source_sha256": observed_source_hashes,
        "source_bytes": [source.stat().st_size for source in sources],
        "distinct_from_existing_shards": ["000_00000", "001_00000", "004_00000"],
        "excluded_prior_text_sha256": sorted(EXCLUDED_TEXT_SHA256),
        "checkpoint_tokenizer_sha256": tokenizer_sha,
        "tokens": int(state["tokens_written"]),
        "documents": int(state["documents_written"]),
        "tokens_file_sha256": sha256_file(tokens_path),
        "offsets_file_sha256": sha256_file(offsets_path),
        "fresh_eval_rows_sha256": sha256_file(eval_rows_path),
        "fresh_eval_manifest_sha256": sha256_file(eval_manifest),
        "fresh_eval_documents": int(args.eval_documents),
        "fresh_eval_row_schema": "fineweb_pg19_tail_nll_v1",
        "fresh_eval_append_eos": False,
        "rolling_document_sha256": state["rolling_document_sha256"],
        "training_or_parameter_updates": False,
    }
    atomic_json(final_receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
