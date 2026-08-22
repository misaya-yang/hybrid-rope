"""CPU-only physical context-stretch data builder for phase-chord retrofit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


STATUS = "OLMO2_PHASE_CHORD_CONTEXT_DATA_PREPARED_V1"
SOURCE_BLOCK = 1_024
TARGET_BLOCK = 16
TRAIN_LENGTH = 8_192
TRAIN_DISTRACTOR_BLOCK = 1_192
TRAIN_DISTRACTOR_COUNT = 6
VALIDATION_LENGTH = 16_384
VALIDATION_DISTRACTOR_COUNT = 16
VALIDATION_DISTRACTOR_BLOCK = 959


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must be a rank-two integer tensor")
    return np.asarray(result, dtype=np.uint32)


def _rows_from_receipt(receipt: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    rows = receipt.get("rows", receipt.get("row_provenance"))
    if not isinstance(rows, list) and isinstance(receipt.get("documents"), list):
        documents = receipt["documents"]
        if len(documents) != count:
            raise ValueError("token receipt documents length does not match tensor")
        validation_start = count - 128
        rows = [
            {
                "row": index,
                "split": "validation" if index >= validation_start else "train",
                "document_id": str(item.get("text_sha256", "")),
                "parquet_row": item.get("parquet_row"),
                "source_tokens": item.get("source_tokens"),
            }
            for index, item in enumerate(documents)
            if isinstance(item, Mapping)
        ]
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("token receipt must contain one rows entry per tensor row")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping):
            raise ValueError("token receipt row is not an object")
        row = dict(item)
        row.setdefault("row", index)
        if int(row["row"]) != index:
            raise ValueError("token receipt rows must be ordered by tensor row")
        split = str(row.get("split", ""))
        document = row.get("document_id", row.get("doc_id"))
        if split not in {"train", "validation", "val"} or document is None:
            raise ValueError("each receipt row needs split and document_id")
        row["split"] = "validation" if split == "val" else split
        row["document_id"] = str(document)
        normalized.append(row)
    documents = {row["document_id"] for row in normalized}
    if len(documents) != count:
        raise ValueError("token receipt must bind distinct parquet documents")
    train = {row["document_id"] for row in normalized if row["split"] == "train"}
    validation = {
        row["document_id"] for row in normalized if row["split"] == "validation"
    }
    if not train or not validation or train & validation:
        raise ValueError("train/validation document sets must be non-empty and disjoint")
    return normalized


def _load_tensor(path: Path) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("loading the pure tensor requires CPU PyTorch") from exc
    value = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    if isinstance(value, Mapping):
        value = value.get("input_ids", value.get("tokens"))
    if value is None:
        raise ValueError("pure tensor does not contain input_ids or tokens")
    return _array(value, "pure tensor")


def load_source(tensor_path: Path, receipt_path: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, str]]:
    tokens = _load_tensor(tensor_path)
    receipt = json.loads(receipt_path.resolve().read_text(encoding="utf-8"))
    observed_tensor_file_sha = sha256_file(tensor_path.resolve())
    observed_tensor_array_sha = sha256_array(tokens)
    expected_tensor_sha = receipt.get("tensor_sha256")
    if expected_tensor_sha is not None and str(expected_tensor_sha) not in {
        observed_tensor_file_sha,
        observed_tensor_array_sha,
    }:
        raise ValueError("pure tensor SHA-256 does not match token receipt")
    rows = _rows_from_receipt(receipt, tokens.shape[0])
    if tokens.shape[1] != 4_096:
        raise ValueError("production source tensor must contain 4096-token rows")
    return tokens, rows, {
        "tensor_file_sha256": observed_tensor_file_sha,
        "tensor_array_sha256": observed_tensor_array_sha,
        "token_receipt_sha256": sha256_file(receipt_path.resolve()),
    }


def _contains(haystack: np.ndarray, needle: np.ndarray) -> bool:
    if len(needle) > len(haystack):
        return False
    if len(needle) == 0:
        return True
    # Both arrays are contiguous uint32 token IDs.  Byte search runs in C and
    # is orders of magnitude faster than a Python loop over 1K-token natural
    # blocks.  Only item-aligned matches count as token-sequence matches.
    haystack_bytes = np.ascontiguousarray(haystack, dtype="<u4").tobytes()
    needle_bytes = np.ascontiguousarray(needle, dtype="<u4").tobytes()
    offset = 0
    while True:
        match = haystack_bytes.find(needle_bytes, offset)
        if match < 0:
            return False
        if match % np.dtype("<u4").itemsize == 0:
            return True
        offset = match + 1


def _safe_block(
    tokens: np.ndarray,
    row: int,
    offset: int,
    length: int,
    forbidden: Sequence[np.ndarray],
) -> np.ndarray | None:
    if offset < 0 or offset + length > tokens.shape[1]:
        return None
    block = np.asarray(tokens[row, offset : offset + length], dtype=np.uint32).copy()
    if any(_contains(block, value) or _contains(value, block) for value in forbidden):
        return None
    return block


def _split_indices(rows: Sequence[Mapping[str, Any]], split: str) -> list[int]:
    return [index for index, row in enumerate(rows) if row["split"] == split]


def _candidate_rows(
    rows: Sequence[Mapping[str, Any]], source_row: int, *, split: str
) -> list[int]:
    return [index for index in _split_indices(rows, split) if index != source_row]


def _row_offsets(row: int, seed: int) -> tuple[int, ...]:
    rng = np.random.default_rng(int(seed) + int(row) * 1_000_003)
    values = rng.choice(4_096 - SOURCE_BLOCK - TARGET_BLOCK + 1, size=4, replace=False)
    return tuple(int(value) for value in sorted(values.tolist()))


def _build_one(
    tokens: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_row: int,
    source_offset: int,
    length: int,
    distractor_count: int,
    distractor_block: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = np.asarray(tokens[source_row, source_offset : source_offset + SOURCE_BLOCK], dtype=np.uint32)
    target_offset = source_offset + SOURCE_BLOCK
    target = np.asarray(tokens[source_row, target_offset : target_offset + TARGET_BLOCK], dtype=np.uint32)
    if len(source) != SOURCE_BLOCK or len(target) != TARGET_BLOCK:
        raise ValueError("source/target offset exceeds the pure tensor row")
    candidates = _candidate_rows(rows, source_row, split=str(rows[source_row]["split"]))
    if len(candidates) < int(distractor_count) + 1:
        raise ValueError("split has too few distinct rows for source and distractors")
    rng = np.random.default_rng(int(seed) + int(source_row) * 1_000_003 + int(source_offset))
    permutation = rng.permutation(candidates).tolist()
    other_row = None
    other_source = None
    for row in permutation:
        block = _safe_block(tokens, row, source_offset, SOURCE_BLOCK, (source, target))
        if block is not None:
            other_row, other_source = int(row), block
            break
    if other_row is None or other_source is None:
        raise ValueError("could not find a non-overlapping replacement source")
    distractor_ids: list[int] = []
    distractor_offsets: list[int] = []
    distractors: list[np.ndarray] = []
    forbidden = (source, target, other_source)
    for row in permutation:
        if row == other_row:
            continue
        for attempt in range(tokens.shape[1]):
            offset = int((int(source_offset) * 131 + int(row) * 17 + attempt * 97) % (tokens.shape[1] - distractor_block + 1))
            block = _safe_block(tokens, int(row), offset, distractor_block, forbidden)
            if block is not None:
                distractor_ids.append(int(row))
                distractor_offsets.append(offset)
                distractors.append(block)
                forbidden = (*forbidden, block)
                break
        if len(distractors) == int(distractor_count):
            break
    if len(distractors) != int(distractor_count):
        raise ValueError("could not construct enough deterministic distractors")
    source_slot = int(rng.integers(0, int(distractor_count) + 1))
    correct_blocks = list(distractors)
    swapped_blocks = list(distractors)
    correct_blocks.insert(source_slot, source)
    swapped_blocks.insert(source_slot, other_source)
    correct = np.concatenate((*correct_blocks, target))
    swapped = np.concatenate((*swapped_blocks, target))
    if len(correct) != int(length) or len(swapped) != int(length):
        raise ValueError("constructed context length drift")
    target_start = int(length) - TARGET_BLOCK
    mask = np.zeros(length, dtype=np.bool_)
    mask[target_start:] = True
    short_correct = np.concatenate((source, target))
    short_swapped = np.concatenate((other_source, target))
    short_mask = np.zeros(SOURCE_BLOCK + TARGET_BLOCK, dtype=np.bool_)
    short_mask[SOURCE_BLOCK:] = True
    arrays = {
        "correct_input_ids": correct,
        "correct_target_mask": mask.copy(),
        "swapped_input_ids": swapped,
        "swapped_target_mask": mask,
        "short_correct_input_ids": short_correct,
        "short_correct_target_mask": short_mask.copy(),
        "short_swapped_input_ids": short_swapped,
        "short_swapped_target_mask": short_mask,
    }
    provenance = {
        "source_row": int(source_row),
        "source_split": str(rows[source_row]["split"]),
        "source_document_id": str(rows[source_row]["document_id"]),
        "other_source_row": int(other_row),
        "other_source_document_id": str(rows[other_row]["document_id"]),
        "source_offset": int(source_offset),
        "target_offset_in_source_row": int(target_offset),
        "source_position": [source_slot * int(distractor_block), source_slot * int(distractor_block) + SOURCE_BLOCK],
        "source_slot": source_slot,
        "target_position": [target_start, int(length)],
        "distractor_rows": distractor_ids,
        "distractor_document_ids": [str(rows[row]["document_id"]) for row in distractor_ids],
        "distractor_offsets": distractor_offsets,
        "length": int(length),
        "same_positions_correct_swapped": True,
        "source_effect_target_only": True,
    }
    return arrays, provenance


def build_split(
    tokens: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    length: int,
    distractor_count: int,
    distractor_block: int,
    seed: int = 20260821,
) -> dict[str, Any]:
    tokens = _array(tokens, "tokens")
    if length == TRAIN_LENGTH and (distractor_count, distractor_block) != (6, 1192):
        raise ValueError("production 8K requires six 1192-token distractor blocks")
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    row_ids = _split_indices(rows, split)
    if not row_ids:
        raise ValueError(f"receipt has no {split} rows")
    all_arrays: dict[str, list[np.ndarray]] = {name: [] for name in (
        "correct_input_ids", "correct_target_mask",
        "swapped_input_ids", "swapped_target_mask",
        "short_correct_input_ids", "short_correct_target_mask",
        "short_swapped_input_ids", "short_swapped_target_mask",
    )}
    provenance: list[dict[str, Any]] = []
    for row in row_ids:
        for offset in _row_offsets(row, seed):
            arrays, receipt = _build_one(
                tokens, rows, source_row=row, source_offset=offset,
                length=length, distractor_count=distractor_count,
                distractor_block=distractor_block, seed=seed,
            )
            for name, value in arrays.items():
                all_arrays[name].append(value)
            provenance.append({
                **receipt,
                "example_index": len(provenance),
                "branches": ["correct", "swapped"],
                "same_target_position": True,
            })
    materialized = {name: np.stack(values) for name, values in all_arrays.items()}
    return {
        "arrays": materialized,
        "provenance": provenance,
        "manifest": {
            "status": STATUS,
            "split": split,
            "length": int(length),
            "source_rows": len(row_ids),
            "examples_per_source_row": 4,
            "examples": len(provenance),
            "source_block": SOURCE_BLOCK,
            "target_block": TARGET_BLOCK,
            "distractor_count": int(distractor_count),
            "distractor_block": int(distractor_block),
            "target_position": [int(length - TARGET_BLOCK), int(length)],
            "short_shape": [len(provenance), SOURCE_BLOCK + TARGET_BLOCK],
            "short_target_position": [SOURCE_BLOCK, SOURCE_BLOCK + TARGET_BLOCK],
            "cross_split_rows": False,
        },
    }


def write_split(result: Mapping[str, Any], output: Path, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    output = output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output if output.exists() else incomplete)
    incomplete.mkdir(parents=True)
    for name, value in result["arrays"].items():
        np.save(incomplete / f"{name}.npy", value, allow_pickle=False)
    provenance_path = incomplete / "provenance.jsonl"
    with provenance_path.open("w", encoding="utf-8") as handle:
        for row in result["provenance"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = dict(result["manifest"])
    manifest["source_hashes"] = dict(source_hashes)
    manifest["output_hashes"] = {
        path.name: sha256_file(path)
        for path in incomplete.glob("*.npy")
    }
    manifest["output_hashes"]["provenance.jsonl"] = sha256_file(provenance_path)
    (incomplete / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.replace(output)
    return manifest


def write_split_streaming(
    tokens: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    length: int,
    distractor_count: int,
    distractor_block: int,
    output: Path,
    source_hashes: Mapping[str, str],
    seed: int = 20260821,
) -> dict[str, Any]:
    """Build one production split directly into atomic NPY memmaps.

    The no-GPU container cannot hold both thousands of per-row arrays and the
    final stacked copies.  Streaming keeps only one correct/swapped pair live.
    """

    output = output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output if output.exists() else incomplete)
    row_ids = _split_indices(rows, split)
    if not row_ids:
        raise ValueError(f"receipt has no {split} rows")
    if length == TRAIN_LENGTH and (distractor_count, distractor_block) != (
        TRAIN_DISTRACTOR_COUNT,
        TRAIN_DISTRACTOR_BLOCK,
    ):
        raise ValueError("production 8K geometry drift")
    examples = len(row_ids) * 4
    incomplete.mkdir(parents=True)
    specs = {
        "correct_input_ids": (np.uint32, (examples, length)),
        "correct_target_mask": (np.bool_, (examples, length)),
        "swapped_input_ids": (np.uint32, (examples, length)),
        "swapped_target_mask": (np.bool_, (examples, length)),
        "short_correct_input_ids": (
            np.uint32,
            (examples, SOURCE_BLOCK + TARGET_BLOCK),
        ),
        "short_correct_target_mask": (
            np.bool_,
            (examples, SOURCE_BLOCK + TARGET_BLOCK),
        ),
        "short_swapped_input_ids": (
            np.uint32,
            (examples, SOURCE_BLOCK + TARGET_BLOCK),
        ),
        "short_swapped_target_mask": (
            np.bool_,
            (examples, SOURCE_BLOCK + TARGET_BLOCK),
        ),
    }
    mapped = {
        name: np.lib.format.open_memmap(
            incomplete / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        for name, (dtype, shape) in specs.items()
    }
    provenance_path = incomplete / "provenance.jsonl"
    index = 0
    with provenance_path.open("w", encoding="utf-8") as handle:
        for row in row_ids:
            for offset in _row_offsets(row, seed):
                arrays, receipt = _build_one(
                    tokens,
                    rows,
                    source_row=row,
                    source_offset=offset,
                    length=length,
                    distractor_count=distractor_count,
                    distractor_block=distractor_block,
                    seed=seed,
                )
                for name, value in arrays.items():
                    mapped[name][index] = value
                handle.write(
                    json.dumps(
                        {
                            **receipt,
                            "example_index": index,
                            "branches": ["correct", "swapped"],
                            "same_target_position": True,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                index += 1
                if index % 256 == 0:
                    for value in mapped.values():
                        value.flush()
    if index != examples:
        raise RuntimeError("streamed example count drift")
    for value in mapped.values():
        value.flush()
    del mapped
    manifest = {
        "status": STATUS,
        "split": split,
        "length": int(length),
        "source_rows": len(row_ids),
        "examples_per_source_row": 4,
        "examples": examples,
        "source_block": SOURCE_BLOCK,
        "target_block": TARGET_BLOCK,
        "distractor_count": int(distractor_count),
        "distractor_block": int(distractor_block),
        "target_position": [int(length - TARGET_BLOCK), int(length)],
        "short_shape": [examples, SOURCE_BLOCK + TARGET_BLOCK],
        "short_target_position": [SOURCE_BLOCK, SOURCE_BLOCK + TARGET_BLOCK],
        "cross_split_rows": False,
        "source_hashes": dict(source_hashes),
        "output_hashes": {
            path.name: sha256_file(path)
            for path in incomplete.glob("*.npy")
        },
    }
    manifest["output_hashes"]["provenance.jsonl"] = sha256_file(
        provenance_path
    )
    (incomplete / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(output)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--token-receipt", type=Path, required=True)
    parser.add_argument("--train8k-output", type=Path, required=True)
    parser.add_argument("--val8k-output", type=Path, required=True)
    parser.add_argument("--val16k-output", type=Path, required=True)
    args = parser.parse_args(argv)
    tokens, rows, hashes = load_source(args.tensor, args.token_receipt)
    specs = (
        ("train", TRAIN_LENGTH, TRAIN_DISTRACTOR_COUNT,
         TRAIN_DISTRACTOR_BLOCK, args.train8k_output),
        ("validation", TRAIN_LENGTH, TRAIN_DISTRACTOR_COUNT,
         TRAIN_DISTRACTOR_BLOCK, args.val8k_output),
        ("validation", VALIDATION_LENGTH, VALIDATION_DISTRACTOR_COUNT,
         VALIDATION_DISTRACTOR_BLOCK, args.val16k_output),
    )
    for split, length, count, block, output in specs:
        write_split_streaming(
            tokens,
            rows,
            split=split,
            length=length,
            distractor_count=count,
            distractor_block=block,
            output=output,
            source_hashes=hashes,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
