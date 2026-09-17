#!/usr/bin/env python3
"""Freeze source-disjoint unlabeled Native windows and causal Q/K pairs.

The input corpus manifest must contain PG19-train and ProofPile-train rows. Each
row needs ``source``, ``source_split``, ``doc_id`` and exactly one of
``token_file``, ``text_path`` or ``text``. Model outputs and task scores are not
read. The first 16 eligible documents per source in SHA256(doc_id) order are fit;
the next four are report-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import (
    FIT_DOCUMENTS_PER_SOURCE,
    METHOD_ID,
    NATIVE_LENGTH,
    PAIRS_PER_DOCUMENT,
    REPORT_DOCUMENTS_PER_SOURCE,
    SOURCES,
)
from .core import sample_causal_pairs
from .io_utils import atomic_json, canonical_sha256, file_sha256, model_identity, read_json_or_jsonl


CONTRACT = "CA_NCP_UNLABELED_NATIVE_QK_STATISTICS_V1_1"


def normalize_source(value: str) -> str:
    value = str(value).strip().lower().replace("-", "").replace("_", "")
    aliases = {"pg19": "pg19", "proofpile": "proofpile", "theproofpile": "proofpile"}
    if value not in aliases:
        raise ValueError(f"unsupported source {value!r}; expected PG19 or ProofPile")
    return aliases[value]


def _load_tokens(row: dict, *, tokenizer, base: Path) -> tuple[np.ndarray, str, Path | None]:
    choices = [name for name in ("token_file", "text_path", "text") if row.get(name) is not None]
    if len(choices) != 1:
        raise ValueError("each corpus row needs exactly one of token_file, text_path, or text")
    if choices[0] == "token_file":
        path = Path(row["token_file"])
        if not path.is_absolute():
            path = base / path
        tokens = np.load(path, allow_pickle=False)
        if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError(f"token_file must be a one-dimensional integer array: {path}")
        return np.asarray(tokens, dtype=np.int64), file_sha256(path), path.resolve()
    if choices[0] == "text_path":
        path = Path(row["text_path"])
        if not path.is_absolute():
            path = base / path
        text = path.read_text(errors="strict")
        source_hash = file_sha256(path)
    else:
        text = str(row["text"])
        source_hash = hashlib.sha256(text.encode()).hexdigest()
    tokens = tokenizer(text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    return np.asarray(tokens, dtype=np.int64), source_hash, None


def _window(
    tokens: np.ndarray, *, doc_id: str, source_hash: str, native_length: int,
) -> tuple[np.ndarray, int]:
    if tokens.size < native_length:
        raise ValueError("document is shorter than the Native window")
    digest = hashlib.sha256(
        f"CA_NCP_WINDOW_V1.1|{doc_id}|{source_hash}".encode()
    ).digest()
    choices = tokens.size - native_length + 1
    offset = int.from_bytes(digest[:8], "little") % choices
    return np.ascontiguousarray(tokens[offset : offset + native_length], dtype=np.int64), offset


def _pair_seed(doc_id: str, token_sha256: str) -> int:
    digest = hashlib.sha256(f"CA_NCP_V1.1|{doc_id}|{token_sha256}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _verify_complete(out: Path, request: dict) -> dict | None:
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("request") != request or manifest.get("status") != "CPU_PREPARED":
        raise ValueError("existing statistics assets have another request or are incomplete")
    for row in manifest["documents"]:
        for field in ("token_file", "pair_file"):
            path = Path(row[field])
            if not path.is_absolute():
                path = out / path
            if not path.is_file() or file_sha256(path) != row[field + "_sha256"]:
                raise ValueError(f"prepared statistics asset drifted: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--fit-documents", type=int, default=32)
    parser.add_argument("--report-documents", type=int, default=8)
    parser.add_argument("--pairs-per-document", type=int, default=PAIRS_PER_DOCUMENT)
    parser.add_argument("--native-length", type=int, default=NATIVE_LENGTH)
    parser.add_argument("--rotary-pairs", type=int, default=64)
    parser.add_argument("--method-id", default=METHOD_ID)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.fit_documents != 2 * FIT_DOCUMENTS_PER_SOURCE or args.report_documents != 2 * REPORT_DOCUMENTS_PER_SOURCE:
        raise ValueError("v1.1 freezes 16 fit plus 4 report documents per source")
    if args.pairs_per_document != PAIRS_PER_DOCUMENT:
        raise ValueError("v1.1 freezes 512 causal pairs per document")
    identity = model_identity(args.model, include_checkpoint_files=False)
    if identity["native_length"] != args.native_length or identity["rotary_pairs"] != args.rotary_pairs:
        raise ValueError("CA-NCP model geometry differs from the requested Native length or rotary pairs")
    corpus_path = args.corpus_manifest.resolve()
    request = {
        "contract": CONTRACT,
        "method_id": args.method_id,
        "model_config_sha256": identity["config_sha256"],
        "corpus_manifest_sha256": file_sha256(corpus_path),
        "fit_documents": args.fit_documents,
        "report_documents": args.report_documents,
        "pairs_per_document": args.pairs_per_document,
        "native_length": args.native_length,
        "rotary_pairs": args.rotary_pairs,
        "selection": "first eligible documents by SHA256(doc_id) within each source; deterministic hash window",
    }
    existing = _verify_complete(args.out, request)
    if existing is not None:
        print(json.dumps({"status": "SKIP_COMPLETE", "documents": len(existing["documents"])}))
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("output directory is nonempty without a valid complete manifest")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    special = {int(value) for value in tokenizer.all_special_ids}
    rows = read_json_or_jsonl(corpus_path)
    candidates: dict[str, list[dict]] = {source: [] for source in SOURCES}
    for row in rows:
        source = normalize_source(row.get("source"))
        if str(row.get("source_split", "")).lower() != "train":
            continue
        doc_id = str(row.get("doc_id", ""))
        if not doc_id:
            raise ValueError("corpus row lacks doc_id")
        candidates[source].append({**row, "source": source, "doc_id": doc_id})
    required = FIT_DOCUMENTS_PER_SOURCE + REPORT_DOCUMENTS_PER_SOURCE
    selected: list[tuple[dict, np.ndarray, str, int]] = []
    for source in SOURCES:
        eligible = []
        ordered = sorted(candidates[source], key=lambda row: (hashlib.sha256(row["doc_id"].encode()).hexdigest(), row["doc_id"]))
        for row in ordered:
            try:
                tokens, raw_hash, token_source = _load_tokens(row, tokenizer=tokenizer, base=corpus_path.parent)
                window, offset = _window(
                    tokens, doc_id=row["doc_id"], source_hash=raw_hash,
                    native_length=args.native_length,
                )
            except ValueError as error:
                if "shorter than" in str(error):
                    continue
                raise
            eligible.append((row, window, raw_hash, offset, token_source, int(tokens.size)))
            if len(eligible) == required:
                break
        if len(eligible) != required:
            raise ValueError(f"source {source} has {len(eligible)} eligible train documents; need {required}")
        selected.extend(eligible)

    args.out.mkdir(parents=True)
    documents = []
    for source in SOURCES:
        source_rows = [item for item in selected if item[0]["source"] == source]
        for index, (row, window, raw_hash, offset, token_source, source_token_count) in enumerate(source_rows):
            role = "fit" if index < FIT_DOCUMENTS_PER_SOURCE else "report"
            safe = hashlib.sha256(f"{source}|{row['doc_id']}".encode()).hexdigest()[:16]
            token_rel = Path("tokens") / f"{source}_{safe}.npy"
            pair_rel = Path("pairs") / f"{source}_{safe}.npz"
            pair_path = args.out / pair_rel
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            if token_source is not None and source_token_count == args.native_length and offset == 0:
                token_path = token_source
                token_value = str(token_path)
                token_external = True
            else:
                token_path = args.out / token_rel
                token_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(token_path, window.astype(np.int32), allow_pickle=False)
                token_value = str(token_rel)
                token_external = False
            token_hash = file_sha256(token_path)
            valid = np.asarray([i for i, token in enumerate(window) if int(token) not in special], dtype=np.int64)
            query, key = sample_causal_pairs(valid, args.pairs_per_document, _pair_seed(row["doc_id"], token_hash))
            np.savez_compressed(pair_path, query_pos=query, key_pos=key)
            documents.append({
                "source": source,
                "source_split": "train",
                "doc_id": row["doc_id"],
                "source_content_sha256": raw_hash,
                "token_file": token_value,
                "token_file_sha256": token_hash,
                "token_file_external_reuse": token_external,
                "pair_file": str(pair_rel),
                "pair_file_sha256": file_sha256(pair_path),
                "window_offset": offset,
                "num_tokens": int(window.size),
                "valid_sample_positions": int(valid.size),
                "role": role,
                "tokenizer_hash": canonical_sha256(identity["tokenizer_files_sha256"]),
            })
    if sum(row["role"] == "fit" for row in documents) != 32 or sum(row["role"] == "report" for row in documents) != 8:
        raise AssertionError("fit/report allocation drifted")
    manifest = {
        "status": "CPU_PREPARED",
        "request": request,
        "model_identity": identity,
        "documents": documents,
        "document_counts": {"fit": 32, "report": 8, "total": 40},
        "source_counts": {source: sum(row["source"] == source for row in documents) for source in SOURCES},
        "selection_uses_model_outputs": False,
        "uses_task_answers": False,
        "gpu_execution": False,
        "scope": "Unlabeled Native Q/K statistics input; no model quality result.",
    }
    atomic_json(args.out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "documents": len(documents), "out": str(args.out)}))


if __name__ == "__main__":
    main()
