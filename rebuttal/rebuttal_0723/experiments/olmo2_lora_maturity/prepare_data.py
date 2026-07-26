#!/usr/bin/env python3
"""Freeze official OLMo-2 SFT and long-context data for LoRA screening.

The output intentionally contains two complementary data families:

* one paired set of official LongAlign examples rendered at 4K/8K/16K;
* one 4K replay set from the official OLMo-2 Tulu-3 SFT mixture.

LongAlign examples are selected only when their full OLMo-tokenized form fits
within 16K.  The 16K view is therefore untruncated.  Its 4K/8K views preserve
the beginning-of-sequence token and the tail containing the query/assistant
answer.  This makes the three views row-paired while keeping answer supervision
available; the manifest records every cropped-token count.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


MODEL_ID = "allenai/OLMo-2-0425-1B-early-training"
MODEL_REVISIONS = {
    "step10000_21B": "daa674482460ee4a0730711e86ef9a834c41c3d0",
    "step20000_42B": "f9dd86fb2eee6a7f0c79dc6fc2f671b58523cddb",
    "step30000_63B": "6251e24cf3f303f9d64c78456a155a5dbe2a35e8",
}
LONGALIGN_ID = "zai-org/LongAlign-10k"
LONGALIGN_REVISION = "12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc"
TULU_ID = "allenai/tulu-3-sft-olmo-2-mixture-0225"
TULU_REVISION = "d91a0785ade02942520280fb484866fce41e448f"
DEFAULT_LENGTHS = (4_096, 8_192, 16_384)

# This is the official OLMo-2 SFT template, with generation tags added only to
# recover an exact assistant-token mask from Transformers.  The rendered text
# is unchanged by the generation tags.
OLMO2_SFT_MASK_TEMPLATE = r"""{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>
' + message['content'] + '
' }}{% elif message['role'] == 'user' %}{{ '<|user|>
' + message['content'] + '
' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>
' }}{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}{% if not loop.last %}{{
'
' }}{% endif %}{% endif %}{% endfor %}"""


@dataclass(frozen=True)
class TokenizedRecord:
    source_row: int
    source_id: str
    input_ids: np.ndarray
    assistant_mask: np.ndarray
    source_kind: str


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    with Path(path).open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{row_index + 1} is not an object")
            yield row_index, value


def normalize_messages(value: Mapping[str, Any]) -> list[dict[str, str]] | None:
    raw = value.get("messages")
    if not isinstance(raw, list) or not raw:
        return None
    messages: list[dict[str, str]] = []
    for message in raw:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        messages.append({"role": role, "content": content})
    if not any(message["role"] == "assistant" for message in messages):
        return None
    return messages


def tokenize_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        chat_template=OLMO2_SFT_MASK_TEMPLATE,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int32)
    assistant = encoded.get("assistant_masks")
    if assistant is None:
        raise RuntimeError("Transformers did not return assistant_masks")
    assistant_mask = np.asarray(assistant, dtype=np.uint8)
    if len(input_ids) != len(assistant_mask):
        raise RuntimeError("assistant mask length does not match input IDs")
    if input_ids.size == 0 or not bool(assistant_mask.any()):
        raise ValueError("rendered example has no supervised assistant tokens")
    if not bool(np.logical_or(
        assistant_mask == 0, assistant_mask == 1
    ).all()):
        raise RuntimeError("assistant mask is not binary")
    return input_ids, assistant_mask


def tokenize_message_batch(
    tokenizer: Any,
    messages: Sequence[Sequence[Mapping[str, str]]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    encoded = tokenizer.apply_chat_template(
        [list(value) for value in messages],
        chat_template=OLMO2_SFT_MASK_TEMPLATE,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        padding=False,
    )
    masks = encoded.get("assistant_masks")
    if masks is None:
        raise RuntimeError("Transformers did not return assistant_masks")
    if len(encoded["input_ids"]) != len(messages) or len(masks) != len(
        messages
    ):
        raise RuntimeError("batched tokenizer row-count drift")
    output = []
    for input_values, mask_values in zip(encoded["input_ids"], masks):
        input_ids = np.asarray(input_values, dtype=np.int32)
        assistant_mask = np.asarray(mask_values, dtype=np.uint8)
        if input_ids.shape != assistant_mask.shape:
            raise RuntimeError("assistant mask length does not match input IDs")
        if input_ids.size == 0 or not bool(assistant_mask.any()):
            raise ValueError(
                "rendered batch example has no supervised assistant tokens"
            )
        output.append((input_ids, assistant_mask))
    return output


def stable_score(seed: int, source_kind: str, source_id: str) -> int:
    payload = f"{seed}\0{source_kind}\0{source_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def select_longalign(
    *,
    path: Path,
    tokenizer: Any,
    count: int,
    maximum_full_length: int,
    minimum_full_length: int,
    seed: int,
) -> tuple[list[TokenizedRecord], dict[str, int]]:
    # Python's heap is a min-heap.  Negative scores keep the largest selected
    # score at the root so it can be replaced by a better (smaller) score.
    heap: list[tuple[int, int, TokenizedRecord]] = []
    stats = {
        "source_rows": 0,
        "unsupported_rows": 0,
        "tokenization_failures": 0,
        "too_short_rows": 0,
        "too_long_rows": 0,
        "eligible_rows": 0,
    }
    batch: list[
        tuple[int, Mapping[str, Any], list[dict[str, str]]]
    ] = []

    def consume(
        values: Sequence[
            tuple[int, Mapping[str, Any], list[dict[str, str]]]
        ],
    ) -> None:
        try:
            tokenized = tokenize_message_batch(
                tokenizer, [messages for _, _, messages in values]
            )
        except (RuntimeError, ValueError):
            stats["tokenization_failures"] += len(values)
            return
        for (source_row, value, _), (
            input_ids,
            assistant_mask,
        ) in zip(values, tokenized):
            length = len(input_ids)
            if length < int(minimum_full_length):
                stats["too_short_rows"] += 1
                continue
            if length > int(maximum_full_length):
                stats["too_long_rows"] += 1
                continue
            stats["eligible_rows"] += 1
            source_id = str(value.get("id", source_row))
            record = TokenizedRecord(
                source_row=source_row,
                source_id=source_id,
                input_ids=input_ids,
                assistant_mask=assistant_mask,
                source_kind="longalign",
            )
            score = stable_score(seed, record.source_kind, record.source_id)
            item = (-score, -source_row, record)
            if len(heap) < int(count):
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    for source_row, value in iter_jsonl(path):
        stats["source_rows"] += 1
        messages = normalize_messages(value)
        if messages is None:
            stats["unsupported_rows"] += 1
            continue
        batch.append((source_row, value, messages))
        if len(batch) == 32:
            consume(batch)
            batch.clear()
    if batch:
        consume(batch)
    if len(heap) != int(count):
        raise RuntimeError(
            f"only {len(heap)} eligible LongAlign rows; requested {count}"
        )
    selected = [item[2] for item in heap]
    selected.sort(
        key=lambda record: stable_score(
            seed, record.source_kind, record.source_id
        )
    )
    stats["selected_rows"] = len(selected)
    return selected, stats


def parquet_row_count(paths: Sequence[Path]) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def take_parquet_rows(
    paths: Sequence[Path], indices: Sequence[int]
) -> list[Mapping[str, Any]]:
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    table = dataset.take(pa.array(list(indices), type=pa.int64()))
    return table.to_pylist()


def select_tulu(
    *,
    paths: Sequence[Path],
    tokenizer: Any,
    count: int,
    maximum_length: int,
    seed: int,
) -> tuple[list[TokenizedRecord], dict[str, int]]:
    total_rows = parquet_row_count(paths)
    candidate_count = min(total_rows, max(int(count) * 4, int(count) + 1024))
    rng = random.Random(int(seed))
    candidate_indices = rng.sample(range(total_rows), candidate_count)
    values = take_parquet_rows(paths, candidate_indices)
    selected: list[TokenizedRecord] = []
    stats = {
        "source_rows": total_rows,
        "sampled_candidate_rows": candidate_count,
        "unsupported_rows": 0,
        "tokenization_failures": 0,
        "no_assistant_after_cap": 0,
    }
    candidates: list[tuple[int, TokenizedRecord]] = []
    supported: list[
        tuple[int, Mapping[str, Any], list[dict[str, str]]]
    ] = []
    for source_row, value in zip(candidate_indices, values):
        messages = normalize_messages(value)
        if messages is None:
            stats["unsupported_rows"] += 1
            continue
        supported.append((int(source_row), value, messages))
    for batch_start in range(0, len(supported), 64):
        batch = supported[batch_start : batch_start + 64]
        try:
            tokenized = tokenize_message_batch(
                tokenizer, [messages for _, _, messages in batch]
            )
        except (RuntimeError, ValueError):
            stats["tokenization_failures"] += len(batch)
            continue
        for (source_row, value, _), (
            input_ids,
            assistant_mask,
        ) in zip(batch, tokenized):
            record = TokenizedRecord(
                source_row=int(source_row),
                source_id=str(value.get("id", source_row)),
                input_ids=input_ids,
                assistant_mask=assistant_mask,
                source_kind="tulu3",
            )
            capped = cap_record(
                record,
                maximum_length=int(maximum_length),
                bos_token_id=int(tokenizer.bos_token_id),
            )
            if not bool(capped.assistant_mask.any()):
                stats["no_assistant_after_cap"] += 1
                continue
            candidates.append(
                (
                    stable_score(
                        seed, record.source_kind, record.source_id
                    ),
                    record,
                )
            )
    candidates.sort(key=lambda item: item[0])
    selected = [item[1] for item in candidates[: int(count)]]
    if len(selected) != int(count):
        raise RuntimeError(
            f"only {len(selected)} valid Tulu rows; requested {count}"
        )
    stats["selected_rows"] = len(selected)
    return selected, stats


def cap_record(
    record: TokenizedRecord,
    *,
    maximum_length: int,
    bos_token_id: int,
) -> TokenizedRecord:
    if len(record.input_ids) <= int(maximum_length):
        return record
    if maximum_length < 2:
        raise ValueError("maximum_length must be at least two")
    has_bos = record.input_ids[0] == int(bos_token_id)
    prefix_ids = (
        record.input_ids[:1]
        if has_bos
        else np.empty(0, dtype=np.int32)
    )
    prefix_mask = (
        np.zeros(1, dtype=np.uint8)
        if has_bos
        else np.empty(0, dtype=np.uint8)
    )
    tail_length = int(maximum_length) - len(prefix_ids)
    input_ids = np.concatenate(
        (prefix_ids, record.input_ids[-tail_length:])
    )
    assistant_mask = np.concatenate(
        (prefix_mask, record.assistant_mask[-tail_length:])
    )
    if len(input_ids) != int(maximum_length):
        raise RuntimeError("capped sequence length drift")
    return TokenizedRecord(
        source_row=record.source_row,
        source_id=record.source_id,
        input_ids=input_ids,
        assistant_mask=assistant_mask,
        source_kind=record.source_kind,
    )


def split_mask(count: int, validation_rows: int, seed: int) -> np.ndarray:
    if not 0 < int(validation_rows) < int(count):
        raise ValueError("validation_rows must lie inside dataset size")
    indices = list(range(int(count)))
    random.Random(int(seed)).shuffle(indices)
    output = np.zeros(int(count), dtype=np.uint8)
    output[indices[: int(validation_rows)]] = 1
    return output


def write_view(
    *,
    records: Sequence[TokenizedRecord],
    output_dir: Path,
    maximum_length: int,
    pad_token_id: int,
    bos_token_id: int,
    validation_rows: int,
    split_seed: int,
    source_receipt: Mapping[str, Any],
    tokenizer_receipt: Mapping[str, Any],
    view_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    count = len(records)
    input_path = output_dir / "input_ids.npy"
    mask_path = output_dir / "assistant_mask.npy"
    lengths_path = output_dir / "lengths.npy"
    split_path = output_dir / "split.npy"
    metadata_path = output_dir / "rows.jsonl"
    input_tmp = input_path.with_name(input_path.name + ".incomplete")
    mask_tmp = mask_path.with_name(mask_path.name + ".incomplete")
    inputs = np.lib.format.open_memmap(
        input_tmp,
        mode="w+",
        dtype=np.int32,
        shape=(count, int(maximum_length)),
    )
    assistant = np.lib.format.open_memmap(
        mask_tmp,
        mode="w+",
        dtype=np.uint8,
        shape=(count, int(maximum_length)),
    )
    inputs[:] = int(pad_token_id)
    assistant[:] = 0
    lengths = np.empty(count, dtype=np.int32)
    metadata_tmp = metadata_path.with_name(
        metadata_path.name + ".incomplete"
    )
    with metadata_tmp.open("w", encoding="utf-8") as metadata:
        for index, original in enumerate(records):
            capped = cap_record(
                original,
                maximum_length=int(maximum_length),
                bos_token_id=int(bos_token_id),
            )
            length = len(capped.input_ids)
            supervised = int(capped.assistant_mask.sum())
            if supervised <= 0:
                raise RuntimeError(
                    f"{view_name} row {index} lost all assistant supervision"
                )
            inputs[index, :length] = np.asarray(
                capped.input_ids, dtype=np.int32
            )
            assistant[index, :length] = np.asarray(
                capped.assistant_mask, dtype=np.uint8
            )
            lengths[index] = length
            metadata.write(
                json.dumps(
                    {
                        "row": index,
                        "source_kind": original.source_kind,
                        "source_row": original.source_row,
                        "source_id": original.source_id,
                        "original_tokens": len(original.input_ids),
                        "stored_tokens": length,
                        "cropped_left_tokens": max(
                            0, len(original.input_ids) - length
                        ),
                        "assistant_tokens": supervised,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    inputs.flush()
    assistant.flush()
    del inputs, assistant
    os.replace(input_tmp, input_path)
    os.replace(mask_tmp, mask_path)
    os.replace(metadata_tmp, metadata_path)
    with lengths_path.with_name(
        lengths_path.name + ".incomplete"
    ).open("wb") as handle:
        np.save(handle, lengths, allow_pickle=False)
    os.replace(
        lengths_path.with_name(lengths_path.name + ".incomplete"),
        lengths_path,
    )
    splits = split_mask(count, validation_rows, split_seed)
    with split_path.with_name(split_path.name + ".incomplete").open(
        "wb"
    ) as handle:
        np.save(handle, splits, allow_pickle=False)
    os.replace(
        split_path.with_name(split_path.name + ".incomplete"),
        split_path,
    )
    files = {}
    for path in (
        input_path,
        mask_path,
        lengths_path,
        split_path,
        metadata_path,
    ):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "format_version": 1,
        "view": view_name,
        "source": dict(source_receipt),
        "tokenizer": dict(tokenizer_receipt),
        "rendering": {
            "template": "official_olmo2_sft_with_generation_mask",
            "template_sha256": hashlib.sha256(
                OLMO2_SFT_MASK_TEMPLATE.encode("utf-8")
            ).hexdigest(),
            "labels": "assistant_content_and_eos_only",
            "padding": "right_to_fixed_storage_shape",
            "truncation": (
                "none_when_full_render_fits; otherwise preserve BOS and "
                "rightmost tokens containing query/assistant answer"
            ),
        },
        "shape": [count, int(maximum_length)],
        "pad_token_id": int(pad_token_id),
        "validation_rows": int(splits.sum()),
        "training_rows": int(count - splits.sum()),
        "length_statistics": {
            "minimum": int(lengths.min()),
            "median": float(np.median(lengths)),
            "maximum": int(lengths.max()),
        },
        "assistant_token_statistics": {
            "minimum": int(
                min(int(cap_record(
                    record,
                    maximum_length=int(maximum_length),
                    bos_token_id=int(bos_token_id),
                ).assistant_mask.sum()) for record in records)
            ),
            "maximum": int(
                max(int(cap_record(
                    record,
                    maximum_length=int(maximum_length),
                    bos_token_id=int(bos_token_id),
                ).assistant_mask.sum()) for record in records)
            ),
        },
        "files": files,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    manifest["manifest_sha256"] = sha256_file(
        output_dir / "manifest.json"
    )
    return manifest


def tokenizer_receipt(checkpoint: Path, tokenizer: Any) -> dict[str, Any]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    files = {}
    for name in names:
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "checkpoint": checkpoint.name,
        "class": type(tokenizer).__name__,
        "bos_token_id": int(tokenizer.bos_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "vocab_size": int(tokenizer.vocab_size),
        "files": files,
    }


def source_files_receipt(
    source_id: str, revision: str, paths: Iterable[Path]
) -> dict[str, Any]:
    return {
        "id": source_id,
        "revision": revision,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(paths)
        },
    }


def verify_collection(path: Path) -> dict[str, Any]:
    collection = json.loads(path.read_text(encoding="utf-8"))
    for view in collection["views"]:
        view_dir = path.parent / view["relative_path"]
        manifest_path = view_dir / "manifest.json"
        actual_manifest_sha = sha256_file(manifest_path)
        if actual_manifest_sha != view["manifest_sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, receipt in manifest["files"].items():
            candidate = view_dir / name
            if candidate.stat().st_size != int(receipt["bytes"]):
                raise RuntimeError(f"size mismatch: {candidate}")
            if sha256_file(candidate) != receipt["sha256"]:
                raise RuntimeError(f"hash mismatch: {candidate}")
        inputs = np.load(view_dir / "input_ids.npy", mmap_mode="r")
        masks = np.load(view_dir / "assistant_mask.npy", mmap_mode="r")
        lengths = np.load(view_dir / "lengths.npy", mmap_mode="r")
        splits = np.load(view_dir / "split.npy", mmap_mode="r")
        expected = tuple(int(value) for value in manifest["shape"])
        if inputs.dtype != np.int32 or tuple(inputs.shape) != expected:
            raise RuntimeError(f"input shape/dtype mismatch: {view_dir}")
        if masks.dtype != np.uint8 or tuple(masks.shape) != expected:
            raise RuntimeError(f"mask shape/dtype mismatch: {view_dir}")
        if lengths.dtype != np.int32 or lengths.shape != (expected[0],):
            raise RuntimeError(f"length shape/dtype mismatch: {view_dir}")
        if splits.dtype != np.uint8 or splits.shape != (expected[0],):
            raise RuntimeError(f"split shape/dtype mismatch: {view_dir}")
        sample_indices = sorted(
            {0, expected[0] // 3, 2 * expected[0] // 3, expected[0] - 1}
        )
        for index in sample_indices:
            length = int(lengths[index])
            if not 0 < length <= expected[1]:
                raise RuntimeError(f"invalid length in {view_dir}: {length}")
            if not bool(masks[index, :length].any()):
                raise RuntimeError(f"no assistant target in {view_dir}:{index}")
            if bool(masks[index, length:].any()):
                raise RuntimeError(f"assistant mask extends into padding: {view_dir}")
    return {
        "status": "OLMO2_LORA_DATA_VERIFIED",
        "collection": str(path),
        "collection_sha256": sha256_file(path),
        "views": len(collection["views"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--longalign-jsonl", type=Path)
    parser.add_argument("--tulu-parquet-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS)
    )
    parser.add_argument("--longalign-rows", type=int, default=2_048)
    parser.add_argument("--tulu-replay-rows", type=int, default=4_096)
    parser.add_argument("--validation-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection_path = args.output_root / "collection_manifest.json"
    if args.verify_only:
        print(json.dumps(verify_collection(collection_path), indent=2))
        return
    if tuple(sorted(set(args.lengths))) != DEFAULT_LENGTHS:
        raise ValueError(f"lengths must be exactly {DEFAULT_LENGTHS}")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.checkpoint is None or args.longalign_jsonl is None:
        raise ValueError("checkpoint and LongAlign JSONL are required")
    if args.tulu_parquet_dir is None:
        raise ValueError("Tulu parquet directory is required")
    checkpoint = args.checkpoint.resolve()
    longalign_path = args.longalign_jsonl.resolve()
    parquet_paths = sorted(args.tulu_parquet_dir.resolve().glob("*.parquet"))
    if len(parquet_paths) != 6:
        raise RuntimeError(
            f"expected six official Tulu shards, found {len(parquet_paths)}"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, use_fast=True
    )
    if (
        int(tokenizer.bos_token_id) != 100_257
        or int(tokenizer.eos_token_id) != 100_257
        or int(tokenizer.pad_token_id) != 100_277
    ):
        raise RuntimeError("OLMo tokenizer special-token contract drift")
    tokenizer_info = tokenizer_receipt(checkpoint, tokenizer)
    longalign_source = source_files_receipt(
        LONGALIGN_ID, LONGALIGN_REVISION, [longalign_path]
    )
    tulu_source = source_files_receipt(
        TULU_ID, TULU_REVISION, parquet_paths
    )
    longalign, longalign_stats = select_longalign(
        path=longalign_path,
        tokenizer=tokenizer,
        count=int(args.longalign_rows),
        maximum_full_length=max(args.lengths),
        minimum_full_length=8_000,
        seed=int(args.seed),
    )
    tulu, tulu_stats = select_tulu(
        paths=parquet_paths,
        tokenizer=tokenizer,
        count=int(args.tulu_replay_rows),
        maximum_length=min(args.lengths),
        seed=int(args.seed) + 1,
    )
    args.output_root.mkdir(parents=True)
    views = []
    for length in sorted(args.lengths):
        relative = Path(f"longalign_paired_L{int(length)}")
        manifest = write_view(
            records=longalign,
            output_dir=args.output_root / relative,
            maximum_length=int(length),
            pad_token_id=int(tokenizer.pad_token_id),
            bos_token_id=int(tokenizer.bos_token_id),
            validation_rows=int(args.validation_rows),
            split_seed=int(args.seed) + 11,
            source_receipt=longalign_source,
            tokenizer_receipt=tokenizer_info,
            view_name=relative.name,
        )
        views.append(
            {
                "relative_path": str(relative),
                "manifest_sha256": manifest["manifest_sha256"],
            }
        )
    replay_relative = Path("tulu3_replay_L4096")
    replay_manifest = write_view(
        records=tulu,
        output_dir=args.output_root / replay_relative,
        maximum_length=min(args.lengths),
        pad_token_id=int(tokenizer.pad_token_id),
        bos_token_id=int(tokenizer.bos_token_id),
        validation_rows=int(args.validation_rows),
        split_seed=int(args.seed) + 12,
        source_receipt=tulu_source,
        tokenizer_receipt=tokenizer_info,
        view_name=replay_relative.name,
    )
    views.append(
        {
            "relative_path": str(replay_relative),
            "manifest_sha256": replay_manifest["manifest_sha256"],
        }
    )
    collection = {
        "format_version": 1,
        "status": "OLMO2_LORA_DATA_PREPARED",
        "purpose": (
            "maturity-by-EVQ-LoRA screening; paired LongAlign length views "
            "plus official OLMo-2 Tulu replay"
        ),
        "model_id": MODEL_ID,
        "model_revisions": MODEL_REVISIONS,
        "seed": int(args.seed),
        "tokenizer": tokenizer_info,
        "sources": {
            "longalign": longalign_source,
            "tulu3_olmo2": tulu_source,
        },
        "selection": {
            "longalign": longalign_stats,
            "tulu3": tulu_stats,
        },
        "views": views,
    }
    atomic_json(collection_path, collection)
    print(json.dumps(verify_collection(collection_path), indent=2))


if __name__ == "__main__":
    main()
