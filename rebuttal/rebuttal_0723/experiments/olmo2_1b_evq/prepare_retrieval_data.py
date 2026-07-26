#!/usr/bin/env python3
"""Prepare deterministic teacher-forced long-range retrieval probes."""

from __future__ import annotations

import argparse
import json
import random
import string
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    SEED,
    sha256_file,
)


LENGTHS = (4_096, 8_192, 16_384)
SOURCE_FRACTIONS = (0.1, 0.5, 0.9)
DISTRACTOR_COUNTS = (0, 8)
ANSWER_WORDS = (
    " blue",
    " green",
    " red",
    " black",
    " white",
    " orange",
    " purple",
    " yellow",
    " seven",
    " nine",
    " four",
    " six",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def one_token_answers(tokenizer: Any) -> list[tuple[str, int]]:
    answers: list[tuple[str, int]] = []
    for word in ANSWER_WORDS:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) == 1:
            answers.append((word, int(token_ids[0])))
    if len(answers) < 8:
        raise RuntimeError(
            f"only {len(answers)} registered answers are single-token"
        )
    return answers


def encode(tokenizer: Any, text: str) -> np.ndarray:
    values = tokenizer.encode(text, add_special_tokens=False)
    return np.asarray(values, dtype=np.uint32)


def random_key(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(8))


def insert_nonoverlapping(
    base: np.ndarray,
    replacements: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    output = base.copy()
    occupied = np.zeros(len(base), dtype=np.bool_)
    for start, value in replacements:
        end = start + len(value)
        if start < 0 or end > len(base) or occupied[start:end].any():
            raise RuntimeError("retrieval insertion overlap")
        output[start:end] = value
        occupied[start:end] = True
    return output


def choose_distractor_starts(
    *,
    usable: int,
    source_start: int,
    source_length: int,
    lengths: list[int],
) -> list[int]:
    if not lengths:
        return []
    candidates = np.linspace(128, usable - 128, len(lengths) + 2).astype(int)[
        1:-1
    ]
    if len(candidates) != len(lengths):
        raise RuntimeError("distractor candidate/length count mismatch")
    starts: list[int] = []
    for candidate, length in zip(candidates, lengths):
        start = min(max(0, int(candidate) - length // 2), usable - length)
        source_end = source_start + source_length
        if start < source_end and start + length > source_start:
            start = source_end + 16
        if start + length > usable:
            start = max(0, source_start - length - 16)
        starts.append(start)
    return starts


def build_example(
    *,
    tokenizer: Any,
    filler: np.ndarray,
    length: int,
    source_fraction: float,
    distractor_count: int,
    gold: tuple[str, int],
    distractors: list[tuple[str, int]],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    key = random_key(rng)
    query = encode(
        tokenizer, f"\nQuestion: The value for code {key} is"
    )
    source = encode(
        tokenizer, f"\nMemory: The value for code {key} is{gold[0]}.\n"
    )
    usable = length - len(query)
    if usable <= len(source) + 256:
        raise RuntimeError("retrieval prompt is too short")
    filler_start = rng.randrange(0, max(1, len(filler) - usable))
    neutral = np.asarray(
        filler[filler_start : filler_start + usable], dtype=np.uint32
    )
    source_start = min(
        max(64, int(usable * source_fraction)),
        usable - len(source) - 64,
    )

    distractor_phrases: list[np.ndarray] = []
    distractor_token_ids: list[int] = []
    for index in range(distractor_count):
        answer = distractors[index % len(distractors)]
        distractor_key = random_key(rng)
        distractor_phrases.append(
            encode(
                tokenizer,
                (
                    f"\nMemory: The value for code {distractor_key} "
                    f"is{answer[0]}.\n"
                ),
            )
        )
        distractor_token_ids.append(answer[1])
    starts = choose_distractor_starts(
        usable=usable,
        source_start=source_start,
        source_length=len(source),
        lengths=[len(value) for value in distractor_phrases],
    )
    if len(starts) != len(distractor_phrases):
        raise RuntimeError("distractor start/phrase count mismatch")
    shared_replacements = list(zip(starts, distractor_phrases))
    deleted_prefix = insert_nonoverlapping(neutral, shared_replacements)
    sourced_prefix = insert_nonoverlapping(
        neutral,
        shared_replacements + [(source_start, source)],
    )
    sourced = np.concatenate([sourced_prefix, query]).astype(np.uint32)
    deleted = np.concatenate([deleted_prefix, query]).astype(np.uint32)
    if sourced.shape != (length,) or deleted.shape != (length,):
        raise RuntimeError("retrieval prompt length drift")
    if not np.array_equal(
        np.delete(sourced, np.s_[source_start : source_start + len(source)]),
        np.delete(deleted, np.s_[source_start : source_start + len(source)]),
    ):
        raise RuntimeError("source/deletion pair differs outside source span")
    metadata = {
        "length": length,
        "source_fraction": source_fraction,
        "source_start": source_start,
        "source_token_length": len(source),
        "distractor_count": distractor_count,
        "gold_word": gold[0],
        "gold_token_id": gold[1],
        "distractor_token_ids": sorted(
            {token_id for _, token_id in distractors}
        ),
        "in_context_distractor_token_ids": sorted(
            set(distractor_token_ids)
        ),
        "key": key,
    }
    return sourced, deleted, metadata


def load_filler(eval_manifest: Path) -> tuple[np.ndarray, str]:
    manifest = json.loads(eval_manifest.read_text(encoding="utf-8"))
    root = eval_manifest.parent
    arrays: list[np.ndarray] = []
    for row in manifest["sources"]:
        path = root / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"evaluation filler hash drift: {path}")
        arrays.append(np.memmap(path, dtype=np.uint32, mode="r"))
    return np.concatenate(arrays), sha256_file(eval_manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-cell", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    manifest_path = output / "retrieval_manifest.json"
    if args.validate_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "RETRIEVAL_DATA_VERIFIED":
            raise RuntimeError("retrieval dataset is not verified")
        if manifest.get("examples_per_cell") != args.examples_per_cell:
            raise RuntimeError("retrieval examples-per-cell drift")
        if manifest.get("lengths") != list(LENGTHS):
            raise RuntimeError("retrieval lengths drift")
        if manifest.get("source_fractions") != list(SOURCE_FRACTIONS):
            raise RuntimeError("retrieval source fractions drift")
        if manifest.get("distractor_counts") != list(DISTRACTOR_COUNTS):
            raise RuntimeError("retrieval distractor counts drift")
        expected_rows = (
            len(SOURCE_FRACTIONS)
            * len(DISTRACTOR_COUNTS)
            * args.examples_per_cell
        )
        expected_pairs = {
            (length, variant)
            for length in LENGTHS
            for variant in ("sourced", "source_deleted", "metadata")
        }
        actual_pairs = {
            (int(row["length"]), str(row["variant"]))
            for row in manifest["arrays"]
        }
        if actual_pairs != expected_pairs or len(manifest["arrays"]) != len(
            expected_pairs
        ):
            raise RuntimeError(
                "retrieval artifact matrix is incomplete or duplicated"
            )
        for row in manifest["arrays"]:
            path = output / row["path"]
            if int(row["rows"]) != expected_rows:
                raise RuntimeError(f"retrieval row-count drift: {path}")
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"retrieval array hash drift: {path}")
            if row["variant"] == "metadata":
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if len(metadata) != expected_rows:
                    raise RuntimeError(
                        f"retrieval metadata row-count drift: {path}"
                    )
            else:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.dtype != np.uint32 or array.shape != (
                    expected_rows,
                    int(row["length"]),
                ):
                    raise RuntimeError(
                        f"retrieval array shape/dtype drift: {path}"
                    )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer_path.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    answers = one_token_answers(tokenizer)
    filler, eval_manifest_sha = load_filler(args.eval_manifest.resolve())
    rng = random.Random(SEED + 2_026)
    arrays: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []
    for length in LENGTHS:
        sourced_rows: list[np.ndarray] = []
        deleted_rows: list[np.ndarray] = []
        local_metadata: list[dict[str, Any]] = []
        for source_fraction in SOURCE_FRACTIONS:
            for distractor_count in DISTRACTOR_COUNTS:
                for _ in range(args.examples_per_cell):
                    chosen = answers[rng.randrange(len(answers))]
                    other = [row for row in answers if row != chosen]
                    rng.shuffle(other)
                    sourced, deleted, metadata = build_example(
                        tokenizer=tokenizer,
                        filler=filler,
                        length=length,
                        source_fraction=source_fraction,
                        distractor_count=distractor_count,
                        gold=chosen,
                        distractors=other,
                        rng=rng,
                    )
                    metadata["row"] = len(sourced_rows)
                    sourced_rows.append(sourced)
                    deleted_rows.append(deleted)
                    local_metadata.append(metadata)
                    all_metadata.append(metadata)
        for variant, rows in (
            ("sourced", sourced_rows),
            ("source_deleted", deleted_rows),
        ):
            path = output / f"{variant}_{length}.uint32.npy"
            np.save(path, np.stack(rows), allow_pickle=False)
            arrays.append(
                {
                    "variant": variant,
                    "length": length,
                    "rows": len(rows),
                    "path": path.name,
                    "sha256": sha256_file(path),
                }
            )
        metadata_path = output / f"metadata_{length}.json"
        write_json(metadata_path, local_metadata)
        arrays.append(
            {
                "variant": "metadata",
                "length": length,
                "rows": len(local_metadata),
                "path": metadata_path.name,
                "sha256": sha256_file(metadata_path),
            }
        )

    manifest = {
        "status": "RETRIEVAL_DATA_VERIFIED",
        "seed": SEED + 2_026,
        "tokenizer_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
        "natural_eval_manifest_sha256": eval_manifest_sha,
        "lengths": list(LENGTHS),
        "source_fractions": list(SOURCE_FRACTIONS),
        "distractor_counts": list(DISTRACTOR_COUNTS),
        "examples_per_cell": args.examples_per_cell,
        "answer_token_ids": [token_id for _, token_id in answers],
        "arrays": arrays,
        "counts": {
            str(key): value
            for key, value in Counter(row["length"] for row in all_metadata).items()
        },
        "metric_contract": [
            "answer_token_nll",
            "answer_token_rank",
            "gold_minus_best_distractor_logit",
            "next_token_exact_match",
            "source_deletion_nll_gap",
        ],
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
