#!/usr/bin/env python3
"""Prepare disjoint physical-4K 2Wiki QA training and LongBench evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer

from .phase_adaptation import FORMAT_VERSION, LENGTH
from .prepare_data import atomic_json, sha256_file


STATUS = "OLMO2_4K_2WIKI_PHASE_DATA_PREPARED_V1"
SOURCE_DATASET = "framolfese/2WikiMultihopQA"
EVALUATION_DATASET = "THUDM/LongBench:2wikimqa"
OFFICIAL_PROMPT = (
    "Answer the question based on the given passages. Only give me the "
    "answer and do not output any other words.\n\n"
    "The following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. Only give me the "
    "answer and do not output any other words.\n\n"
    "Question: {question}\nAnswer:"
)
QUERY_MARKER = "\nQuestion:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-api-dir", type=Path, required=True)
    parser.add_argument("--longbench-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-rows", type=int, default=1_536)
    parser.add_argument("--validation-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20_260_728)
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def qa_identity(question: str, answers: Iterable[str]) -> str:
    return canonical_sha256(
        {
            "question": " ".join(str(question).lower().split()),
            "answers": sorted(
                " ".join(str(answer).lower().split())
                for answer in answers
            ),
        }
    )


def chat_tokens_and_query_start(
    tokenizer: Any,
    prompt: str,
) -> tuple[list[int], int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    marker_start = rendered.rfind(QUERY_MARKER)
    if marker_start < 0:
        raise RuntimeError("2Wiki query marker is absent from rendered chat")
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    chat_ids = list(encoded.input_ids)
    offsets = list(encoded.offset_mapping)
    direct_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
    )
    if chat_ids != direct_ids:
        raise RuntimeError("rendered-chat tokenization contract drift")
    query_tokens = [
        index
        for index, (start, end) in enumerate(offsets)
        if int(end) > marker_start and int(start) < len(rendered)
    ]
    if not query_tokens:
        raise RuntimeError("2Wiki query marker is not recoverable")
    return chat_ids, int(query_tokens[0])


def context_text(row: dict[str, Any]) -> str:
    context = row["context"]
    titles = list(context["title"])
    sentences = list(context["sentences"])
    if len(titles) != len(sentences) or not titles:
        raise RuntimeError("2Wiki context geometry drift")
    blocks = []
    for title, rows in zip(titles, sentences):
        blocks.append(
            f"Title: {title}\n" + " ".join(str(value) for value in rows)
        )
    return "\n\n".join(blocks)


def load_training_rows(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("chunk_*.json"))
    if not paths:
        raise FileNotFoundError(f"no API chunks under {root}")
    rows: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload["rows"]:
            index = int(entry["row_idx"])
            if index in seen_indices:
                raise RuntimeError("duplicate 2Wiki source index")
            seen_indices.add(index)
            if entry.get("truncated_cells"):
                raise RuntimeError("dataset-server returned truncated cells")
            row = dict(entry["row"])
            row["_source_index"] = index
            row["_source_chunk_sha256"] = sha256_file(path)
            rows.append(row)
    return rows


def load_longbench_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.endswith("/2wikimqa.jsonl")
            or name == "2wikimqa.jsonl"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one LongBench 2wikimqa member, got {candidates}"
            )
        member = candidates[0]
        rows = [
            json.loads(line)
            for line in archive.read(member).decode("utf-8").splitlines()
            if line.strip()
        ]
    if len(rows) != 200:
        raise RuntimeError(
            f"LongBench 2wikimqa row-count drift: {len(rows)}"
        )
    return member, rows


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    train_api_dir = args.train_api_dir.resolve()
    longbench_zip = args.longbench_zip.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    requested = int(args.training_rows) + int(args.validation_rows)
    if requested <= 0 or int(args.validation_rows) < 32:
        raise ValueError("invalid train/validation row request")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer requires EOS and PAD ids")
    member, evaluation_source_rows = load_longbench_rows(longbench_zip)
    evaluation_identities: set[str] = set()
    evaluation_rows: list[dict[str, Any]] = []
    for index, row in enumerate(evaluation_source_rows):
        answers = [str(value) for value in row["answers"]]
        identity = qa_identity(str(row["input"]), answers)
        if identity in evaluation_identities:
            raise RuntimeError("duplicate LongBench QA identity")
        evaluation_identities.add(identity)
        prompt = OFFICIAL_PROMPT.format(
            context=str(row["context"]),
            question=str(row["input"]),
        )
        evaluation_rows.append(
            {
                "index": index,
                "source_id": str(row.get("_id", index)),
                "question": str(row["input"]),
                "answers": answers,
                "prompt": prompt,
                "source_row_sha256": canonical_sha256(row),
                "qa_identity_sha256": identity,
                "declared_length": row.get("length"),
            }
        )

    source_rows = load_training_rows(train_api_dir)
    generator = np.random.default_rng(int(args.seed))
    order = generator.permutation(len(source_rows)).tolist()
    prepared: list[dict[str, Any]] = []
    source_identities: set[str] = set()
    rejected_too_long = 0
    for source_slot in order:
        row = source_rows[int(source_slot)]
        answer = str(row["answer"])
        identity = qa_identity(str(row["question"]), [answer])
        if identity in evaluation_identities:
            raise RuntimeError("2Wiki train/LongBench evaluation overlap")
        if identity in source_identities:
            continue
        prompt = OFFICIAL_PROMPT.format(
            context=context_text(row),
            question=str(row["question"]),
        )
        chat_ids, query_start = chat_tokens_and_query_start(
            tokenizer,
            prompt,
        )
        answer_text = " " + answer.strip()
        answer_ids = list(
            tokenizer(answer_text, add_special_tokens=False).input_ids
        )
        if not answer_ids:
            raise RuntimeError("empty 2Wiki answer tokenization")
        combined = chat_ids + answer_ids + [int(tokenizer.eos_token_id)]
        if len(combined) > LENGTH:
            rejected_too_long += 1
            continue
        answer_start = len(chat_ids)
        if not 0 < query_start < answer_start:
            raise RuntimeError("2Wiki query/answer order drift")
        source_identities.add(identity)
        prepared.append(
            {
                "source": row,
                "input_ids": combined,
                "query_start": query_start,
                "answer_start": answer_start,
                "answer_ids": answer_ids,
                "answer_text": answer_text,
                "qa_identity_sha256": identity,
            }
        )
        if len(prepared) == requested:
            break
    if len(prepared) != requested:
        raise RuntimeError(
            f"only {len(prepared)} complete physical-4K rows available"
        )

    input_ids = np.full(
        (requested, LENGTH),
        int(tokenizer.pad_token_id),
        dtype=np.uint32,
    )
    labels = np.full((requested, LENGTH), -100, dtype=np.int32)
    query_starts = np.zeros(requested, dtype=np.int32)
    active_lengths = np.zeros(requested, dtype=np.int32)
    split = np.zeros(requested, dtype=np.uint8)
    metadata: list[dict[str, Any]] = []
    for index, entry in enumerate(prepared):
        ids = np.asarray(entry["input_ids"], dtype=np.uint32)
        active = len(ids)
        answer_start = int(entry["answer_start"])
        input_ids[index, :active] = ids
        labels[index, answer_start:active] = ids[
            answer_start:active
        ].astype(np.int32)
        query_starts[index] = int(entry["query_start"])
        active_lengths[index] = active
        if index >= int(args.training_rows):
            split[index] = 1
        source = entry["source"]
        metadata.append(
            {
                "row": index,
                "split": "train" if split[index] == 0 else "validation",
                "source_dataset": SOURCE_DATASET,
                "source_split": "train",
                "source_index": int(source["_source_index"]),
                "source_id": str(source["id"]),
                "source_row_sha256": canonical_sha256(
                    {
                        name: value
                        for name, value in source.items()
                        if not name.startswith("_")
                    }
                ),
                "source_chunk_sha256": source[
                    "_source_chunk_sha256"
                ],
                "qa_identity_sha256": entry["qa_identity_sha256"],
                "question": str(source["question"]),
                "answer": str(source["answer"]),
                "query_start": int(entry["query_start"]),
                "answer_start": answer_start,
                "answer_tokens": len(entry["answer_ids"]),
                "eos_position": active - 1,
                "active_length": active,
            }
        )

    output.mkdir(parents=True)
    np.save(output / "input_ids.npy", input_ids, allow_pickle=False)
    np.save(output / "labels.npy", labels, allow_pickle=False)
    np.save(
        output / "query_starts.npy",
        query_starts,
        allow_pickle=False,
    )
    np.save(
        output / "active_lengths.npy",
        active_lengths,
        allow_pickle=False,
    )
    np.save(output / "split.npy", split, allow_pickle=False)
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "evaluation_rows.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in evaluation_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    files: dict[str, dict[str, Any]] = {}
    for name in (
        "input_ids.npy",
        "labels.npy",
        "query_starts.npy",
        "active_lengths.npy",
        "split.npy",
        "rows.jsonl",
        "evaluation_rows.jsonl",
    ):
        path = output / name
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "status": STATUS,
        "format_version": FORMAT_VERSION,
        "shape": [requested, LENGTH],
        "physical_storage_length": LENGTH,
        "hard_maximum_training_length": LENGTH,
        "hard_maximum_training_position_id": LENGTH - 1,
        "labels_only_cover_answer_and_final_eos": True,
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "checkpoint": str(checkpoint),
        "tokenizer_sha256": sha256_file(
            checkpoint / "tokenizer.json"
        ),
        "seed": int(args.seed),
        "training_rows": int(args.training_rows),
        "validation_rows": int(args.validation_rows),
        "rejected_too_long": rejected_too_long,
        "train_eval_qa_identity_overlap": 0,
        "source": {
            "dataset": SOURCE_DATASET,
            "api_chunk_files": len(
                list(train_api_dir.glob("chunk_*.json"))
            ),
            "api_chunk_sha256": {
                path.name: sha256_file(path)
                for path in sorted(train_api_dir.glob("chunk_*.json"))
            },
        },
        "evaluation": {
            "dataset": EVALUATION_DATASET,
            "rows": len(evaluation_rows),
            "longbench_zip_sha256": sha256_file(longbench_zip),
            "zip_member": member,
            "official_prompt": OFFICIAL_PROMPT,
            "official_max_new_tokens": 32,
        },
        "phase_curriculum": {
            "physical_sequences": "<=4096",
            "semantic_block": "final_question_and_answer",
            "micro_batch_bucket_ratio": {
                "contiguous_4k": 1,
                "phase_to_8k": 1,
                "phase_to_16k": 2,
            },
        },
        "files": files,
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
