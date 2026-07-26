#!/usr/bin/env python3
"""Matched Geo/EVQ LoRA generalization probe for released OLMo-2 step 5000.

Both schedule arms start from identical released model parameters.  The EVQ
arm replaces only the non-persistent endpoint-grid rotary frequencies before
LoRA adaptation.  Training and formal evaluation use disjoint values, key
namespaces, source-position bins, templates, and PG19 documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    TAU,
    assert_frequency_contract,
    endpoint_geo_inv_freq,
    patch_endpoint_evq,
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    ADAPTATIONS,
    evaluate,
    install_adaptation,
    load_model,
    save_adapter,
    train,
    trainable_named_parameters,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
    place_phrase,
    seed_everything,
    sha256_arrays,
    sha256_file,
)
from transformers import AutoTokenizer


VALUE_WORDS = (
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
    " pink",
    " brown",
    " gray",
    " silver",
    " gold",
    " bronze",
    " cyan",
    " teal",
    " navy",
    " lime",
    " coral",
    " violet",
    " azure",
    " crimson",
    " amber",
    " olive",
    " peach",
    " beige",
    " one",
    " two",
    " three",
    " five",
    " eight",
    " ten",
    " eleven",
    " twelve",
    " thirteen",
    " fourteen",
    " fifteen",
    " sixteen",
    " apple",
    " pear",
    " mango",
    " lemon",
    " grape",
    " cherry",
    " berry",
    " tiger",
    " lion",
    " horse",
    " sheep",
    " eagle",
    " shark",
    " whale",
    " river",
    " ocean",
    " forest",
    " mountain",
    " valley",
    " desert",
    " winter",
    " summer",
    " spring",
    " autumn",
    " north",
    " south",
    " east",
    " west",
    " alpha",
    " beta",
    " gamma",
    " delta",
    " sigma",
    " omega",
    " circle",
    " square",
    " triangle",
    " star",
    " moon",
    " sun",
    " earth",
    " mars",
    " saturn",
)
TRAIN_VALUE_COUNT = 48
EVAL_VALUE_COUNT = 24
TRAIN_SOURCE_FRACTIONS = (0.2, 0.5, 0.8)
EVAL_SOURCE_FRACTIONS = (0.1, 0.35, 0.65, 0.9)
TRAIN_DISTRACTOR_COUNTS = (0, 4, 8)
EVAL_DISTRACTOR_COUNTS = (0, 8)
TRAIN_TEMPLATE_SPECS = (
    (
        "\nRecord {key}: the stored marker is{value}.\n",
        "\nRetrieve the marker stored for {key}. Marker:",
        "train_record_retrieve_v1",
    ),
    (
        "\nIdentifier {key} has code{value}.\n",
        "\nReturn the code for identifier {key}. Code:",
        "train_identifier_code_v2",
    ),
    (
        "\nThe lookup table maps {key} to{value}.\n",
        "\nLook up {key} in the table. Result:",
        "train_lookup_table_v3",
    ),
    (
        "\nNote for {key}: its label is{value}.\n",
        "\nWhat label belongs to {key}? Label:",
        "train_note_label_v4",
    ),
    (
        "\nRegistry key {key} carries token{value}.\n",
        "\nGive the token for registry key {key}. Token:",
        "train_registry_token_v5",
    ),
    (
        "\nItem {key} uses tag{value}.\n",
        "\nRecall the tag used by item {key}. Tag:",
        "train_item_tag_v6",
    ),
    (
        "\nLedger row {key} points to marker{value}.\n",
        "\nRead the marker for ledger row {key}. Marker:",
        "train_ledger_marker_v7",
    ),
    (
        "\nFor reference {key}, the codeword is{value}.\n",
        "\nSupply the codeword for reference {key}. Codeword:",
        "train_reference_codeword_v8",
    ),
)


@dataclass
class FormalProbeSet:
    sourced: np.ndarray
    deleted: np.ndarray
    swapped: np.ndarray
    gold: np.ndarray
    alternate: np.ndarray
    source_fraction: np.ndarray
    distractor_count: np.ndarray
    keys: list[str]
    value_words: list[str]
    document_sources: list[str]
    document_rows: np.ndarray
    crop_offsets: np.ndarray
    template_ids: list[str]

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            sha256_arrays(
                (
                    self.sourced,
                    self.deleted,
                    self.swapped,
                    self.gold,
                    self.alternate,
                    self.source_fraction,
                    self.distractor_count,
                    self.document_rows,
                    self.crop_offsets,
                )
            ).encode("ascii")
        )
        for values in (
            self.keys,
            self.value_words,
            self.document_sources,
            self.template_ids,
        ):
            digest.update(
                json.dumps(
                    values,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        return digest.hexdigest()


def one_token_values(tokenizer: Any) -> tuple[
    list[tuple[str, int]], list[tuple[str, int]]
]:
    values: list[tuple[str, int]] = []
    for word in VALUE_WORDS:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) == 1:
            values.append((word, int(token_ids[0])))
    required = TRAIN_VALUE_COUNT + EVAL_VALUE_COUNT
    if len(values) < required:
        raise RuntimeError(
            f"need at least {required} one-token values, got {len(values)}"
        )
    train_values = values[:TRAIN_VALUE_COUNT]
    eval_values = values[
        TRAIN_VALUE_COUNT : TRAIN_VALUE_COUNT + EVAL_VALUE_COUNT
    ]
    train_ids = {token_id for _, token_id in train_values}
    eval_ids = {token_id for _, token_id in eval_values}
    if train_ids & eval_ids:
        raise RuntimeError("train and evaluation value-token pools overlap")
    return train_values, eval_values


def broad_vocabulary_train_values(
    tokenizer: Any,
    *,
    excluded_token_ids: set[int],
    count: int,
    seed: int,
) -> list[tuple[str, int]]:
    """Select a broad, deterministic one-token value pool."""

    candidates: list[tuple[str, int]] = []
    special_ids = set(tokenizer.all_special_ids)
    for token_id in range(len(tokenizer)):
        if token_id in special_ids or token_id in excluded_token_ids:
            continue
        value = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if re.fullmatch(r" [A-Za-z]{3,20}", value) is None:
            continue
        if tokenizer.encode(
            value, add_special_tokens=False
        ) != [token_id]:
            continue
        train_render = tokenizer.encode(
            f" marker is{value}.",
            add_special_tokens=False,
        )
        eval_render = tokenizer.encode(
            f" symbol{value}.",
            add_special_tokens=False,
        )
        if (
            len(train_render) < 2
            or len(eval_render) < 2
            or train_render[-2] != token_id
            or eval_render[-2] != token_id
        ):
            continue
        candidates.append((value, token_id))
    if len(candidates) < int(count):
        raise RuntimeError(
            f"need {count} broad vocabulary values, got "
            f"{len(candidates)}"
        )
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    return candidates[: int(count)]


def encode(tokenizer: Any, text: str) -> np.ndarray:
    return np.asarray(
        tokenizer.encode(text, add_special_tokens=False),
        dtype=np.int64,
    )


def random_key(rng: random.Random, namespace: str) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return namespace + "".join(rng.choice(alphabet) for _ in range(10))


def phrases(
    *,
    tokenizer: Any,
    phase: str,
    key: str,
    value: str,
    train_template_count: int = 1,
) -> tuple[np.ndarray, np.ndarray, str]:
    if phase == "train":
        count = int(train_template_count)
        if count < 1 or count > len(TRAIN_TEMPLATE_SPECS):
            raise ValueError(
                f"train_template_count must be in "
                f"[1, {len(TRAIN_TEMPLATE_SPECS)}]"
            )
        if count == 1:
            template_index = 0
        else:
            template_index = int.from_bytes(
                hashlib.sha256(key.encode("utf-8")).digest()[:8],
                byteorder="big",
            ) % count
        source_format, query_format, template_id = (
            TRAIN_TEMPLATE_SPECS[template_index]
        )
        source_text = source_format.format(key=key, value=value)
        query_text = query_format.format(key=key)
    elif phase == "eval":
        source_text = (
            f"\nArchive entry [{key}] assigns the symbol{value}.\n"
        )
        query_text = (
            f"\nWhich symbol belongs to archive identifier [{key}]? "
            "Answer:"
        )
        template_id = "eval_archive_question_v1"
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return (
        encode(tokenizer, source_text),
        encode(tokenizer, query_text),
        template_id,
    )


def load_documents(
    array_path: Path,
    metadata_path: Path,
    *,
    expected_length: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    documents = np.load(
        array_path, mmap_mode="r", allow_pickle=False
    )
    if (
        documents.dtype != np.uint32
        or documents.ndim != 2
        or documents.shape[1] != int(expected_length)
    ):
        raise RuntimeError(
            f"document array drift at {array_path}: "
            f"{documents.dtype} {documents.shape}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list) or len(metadata) != len(documents):
        raise RuntimeError("document metadata row count drift")
    for index, row in enumerate(metadata):
        if int(row["row"]) != index or not str(row["source"]):
            raise RuntimeError("document metadata ordering drift")
    return documents, metadata


def select_document_rows(
    metadata: list[dict[str, Any]],
    *,
    split: str,
) -> list[int]:
    marker = f"/{split}/"
    rows = [
        int(row["row"])
        for row in metadata
        if marker in str(row["source"])
    ]
    if not rows:
        raise RuntimeError(f"no documents found for split {split!r}")
    sources = [str(metadata[row]["source"]) for row in rows]
    if len(sources) != len(set(sources)):
        raise RuntimeError("document source names are not unique")
    return rows


def build_probe_set(
    *,
    tokenizer: Any,
    documents: np.ndarray,
    document_metadata: list[dict[str, Any]],
    document_rows: list[int],
    values: list[tuple[str, int]],
    length: int,
    count: int,
    seed: int,
    source_fractions: tuple[float, ...],
    distractor_counts: tuple[int, ...],
    phase: str,
    train_template_count: int = 1,
) -> FormalProbeSet:
    rng = random.Random(seed)
    sourced_rows: list[np.ndarray] = []
    deleted_rows: list[np.ndarray] = []
    swapped_rows: list[np.ndarray] = []
    gold_ids: list[int] = []
    alternate_ids: list[int] = []
    fractions: list[float] = []
    densities: list[int] = []
    keys: list[str] = []
    value_words: list[str] = []
    document_sources: list[str] = []
    selected_document_rows: list[int] = []
    crop_offsets: list[int] = []
    template_ids: list[str] = []
    context_length = int(length) - 1
    namespace = "TR-" if phase == "train" else "EV-"

    for row_index in range(int(count)):
        requested_fraction = source_fractions[
            row_index % len(source_fractions)
        ]
        density = distractor_counts[
            (row_index // len(source_fractions))
            % len(distractor_counts)
        ]
        gold_index = rng.randrange(len(values))
        alternate_index = rng.randrange(len(values) - 1)
        if alternate_index >= gold_index:
            alternate_index += 1
        gold_word, gold_id = values[gold_index]
        alternate_word, alternate_id = values[alternate_index]
        key = random_key(rng, namespace)
        source, query, template_id = phrases(
            tokenizer=tokenizer,
            phase=phase,
            key=key,
            value=gold_word,
            train_template_count=int(train_template_count),
        )
        swapped_source, swapped_query, swapped_template_id = phrases(
            tokenizer=tokenizer,
            phase=phase,
            key=key,
            value=alternate_word,
            train_template_count=int(train_template_count),
        )
        if (
            len(source) != len(swapped_source)
            or not np.array_equal(query, swapped_query)
            or template_id != swapped_template_id
        ):
            raise RuntimeError("source-value swap changed prompt geometry")
        usable = context_length - len(query)
        if usable <= len(source) + 64:
            raise RuntimeError("probe is too short for source and query")

        document_row = document_rows[
            rng.randrange(len(document_rows))
        ]
        document = documents[document_row]
        if len(document) < usable:
            raise RuntimeError("document is shorter than probe prefix")
        crop_limit = len(document) - usable
        crop_start = 0 if crop_limit == 0 else rng.randrange(crop_limit + 1)
        neutral = np.asarray(
            document[crop_start : crop_start + usable],
            dtype=np.int64,
        ).copy()
        deleted_prefix = neutral.copy()
        occupied = np.zeros(usable, dtype=np.bool_)

        distractor_values = [
            row for index, row in enumerate(values) if index != gold_index
        ]
        rng.shuffle(distractor_values)
        for distractor_index in range(int(density)):
            distractor_word, _ = distractor_values[
                distractor_index % len(distractor_values)
            ]
            distractor_key = random_key(rng, namespace)
            distractor, _, _ = phrases(
                tokenizer=tokenizer,
                phase=phase,
                key=distractor_key,
                value=distractor_word,
            )
            preferred = int(
                (distractor_index + 1)
                * usable
                / (int(density) + 1)
            )
            place_phrase(
                deleted_prefix,
                occupied,
                distractor,
                preferred,
            )

        sourced_prefix = deleted_prefix.copy()
        source_occupied = occupied.copy()
        preferred_source = min(
            max(24, int(usable * float(requested_fraction))),
            usable - len(source) - 24,
        )
        source_start = place_phrase(
            sourced_prefix,
            source_occupied,
            source,
            preferred_source,
        )
        if occupied[source_start : source_start + len(source)].any():
            raise RuntimeError("source placement overlaps a distractor")
        swapped_prefix = deleted_prefix.copy()
        swapped_prefix[
            source_start : source_start + len(swapped_source)
        ] = swapped_source

        sourced = np.concatenate((sourced_prefix, query))
        deleted = np.concatenate((deleted_prefix, query))
        swapped = np.concatenate((swapped_prefix, query))
        expected = (context_length,)
        if (
            sourced.shape != expected
            or deleted.shape != expected
            or swapped.shape != expected
        ):
            raise RuntimeError("probe context length drift")
        sourced_rows.append(sourced)
        deleted_rows.append(deleted)
        swapped_rows.append(swapped)
        gold_ids.append(gold_id)
        alternate_ids.append(alternate_id)
        fractions.append(float(requested_fraction))
        densities.append(int(density))
        keys.append(key)
        value_words.append(gold_word)
        document_sources.append(
            str(document_metadata[document_row]["source"])
        )
        selected_document_rows.append(document_row)
        crop_offsets.append(crop_start)
        template_ids.append(template_id)

    return FormalProbeSet(
        sourced=np.stack(sourced_rows),
        deleted=np.stack(deleted_rows),
        swapped=np.stack(swapped_rows),
        gold=np.asarray(gold_ids, dtype=np.int64),
        alternate=np.asarray(alternate_ids, dtype=np.int64),
        source_fraction=np.asarray(fractions, dtype=np.float64),
        distractor_count=np.asarray(densities, dtype=np.int64),
        keys=keys,
        value_words=value_words,
        document_sources=document_sources,
        document_rows=np.asarray(
            selected_document_rows, dtype=np.int64
        ),
        crop_offsets=np.asarray(crop_offsets, dtype=np.int64),
        template_ids=template_ids,
    )


def apply_schedule(model: Any, schedule: str) -> dict[str, Any]:
    receipt = assert_frequency_contract()
    if schedule in {"native", "geo"}:
        active = (
            model.model.rotary_emb.inv_freq.detach().cpu().float()
        )
        if not torch.equal(active, endpoint_geo_inv_freq()):
            raise RuntimeError(
                "released checkpoint native endpoint frequency drift"
            )
        receipt["active_schedule"] = (
            "native_endpoint_rope" if schedule == "native" else "geo"
        )
        receipt["active_sha256_float32"] = tensor_sha256(active)
    elif schedule == "evq":
        receipt = patch_endpoint_evq(model, tau=TAU)
        receipt["active_schedule"] = "evq"
        receipt["active_sha256_float32"] = receipt[
            "evq_sha256_float32"
        ]
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return receipt


def assert_generalization_split(
    *,
    train_data: FormalProbeSet,
    eval_data: dict[int, FormalProbeSet],
    train_values: list[tuple[str, int]],
    eval_values: list[tuple[str, int]],
    train_source_fractions: tuple[float, ...],
) -> dict[str, Any]:
    eval_keys = {
        key for data in eval_data.values() for key in data.keys
    }
    eval_documents = {
        source
        for data in eval_data.values()
        for source in data.document_sources
    }
    eval_templates = {
        template
        for data in eval_data.values()
        for template in data.template_ids
    }
    train_keys = set(train_data.keys)
    train_documents = set(train_data.document_sources)
    train_templates = set(train_data.template_ids)
    train_value_ids = {token_id for _, token_id in train_values}
    eval_value_ids = {token_id for _, token_id in eval_values}
    checks = {
        "keys_disjoint": not bool(train_keys & eval_keys),
        "value_token_ids_disjoint": not bool(
            train_value_ids & eval_value_ids
        ),
        "document_sources_disjoint": not bool(
            train_documents & eval_documents
        ),
        "templates_disjoint": not bool(
            train_templates & eval_templates
        ),
        "requested_position_bins_disjoint": not bool(
            set(train_source_fractions)
            & set(EVAL_SOURCE_FRACTIONS)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"generalization split failed: {checks}")
    return {
        "checks": checks,
        "train": {
            "keys": len(train_keys),
            "value_token_ids": sorted(train_value_ids),
            "value_words": [word for word, _ in train_values],
            "document_sources": sorted(train_documents),
            "templates": sorted(train_templates),
            "source_fraction_bins": list(train_source_fractions),
        },
        "eval": {
            "keys": len(eval_keys),
            "value_token_ids": sorted(eval_value_ids),
            "value_words": [word for word, _ in eval_values],
            "document_sources": sorted(eval_documents),
            "templates": sorted(eval_templates),
            "source_fraction_bins": list(EVAL_SOURCE_FRACTIONS),
        },
    }


def annotate_rows(
    rows: list[dict[str, Any]],
    data_by_length: dict[int, FormalProbeSet],
) -> None:
    for row in rows:
        data = data_by_length[int(row["length"])]
        index = int(row["row"])
        row["value_word"] = data.value_words[index]
        row["document_source"] = data.document_sources[index]
        row["document_row"] = int(data.document_rows[index])
        row["document_crop_offset"] = int(data.crop_offsets[index])
        row["template_id"] = data.template_ids[index]


def hash_paths(paths: Iterable[Path]) -> dict[str, str]:
    return {
        str(path.resolve()): sha256_file(path.resolve())
        for path in paths
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--documents-16k", type=Path, required=True)
    parser.add_argument(
        "--documents-16k-metadata", type=Path, required=True
    )
    parser.add_argument("--documents-32k", type=Path)
    parser.add_argument("--documents-32k-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schedule", choices=("native", "geo", "evq"), required=True
    )
    parser.add_argument(
        "--adaptation",
        choices=(
            "baseline",
            "qkvo_answer",
            "qkvo_causal_margin",
            "qkvo_full",
        ),
        required=True,
    )
    parser.add_argument("--train-length", type=int, default=16_384)
    parser.add_argument(
        "--eval-lengths",
        type=int,
        nargs="+",
        default=[4_096, 8_192, 16_384, 32_768],
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "none",
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        ),
        default="none",
    )
    parser.add_argument("--train-examples", type=int, default=1_024)
    parser.add_argument(
        "--eval-examples-per-cell", type=int, default=15
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--canary-count", type=int, default=8)
    parser.add_argument("--resume-adapter", type=Path)
    parser.add_argument(
        "--train-value-pool",
        choices=("registered", "broad_vocab"),
        default="registered",
    )
    parser.add_argument(
        "--broad-train-value-count", type=int, default=512
    )
    parser.add_argument(
        "--train-template-count",
        type=int,
        choices=range(1, len(TRAIN_TEMPLATE_SPECS) + 1),
        default=1,
    )
    parser.add_argument(
        "--train-source-fractions",
        type=float,
        nargs="+",
        default=list(TRAIN_SOURCE_FRACTIONS),
    )
    parser.add_argument("--margin-loss-weight", type=float, default=0.25)
    parser.add_argument("--top1-margin", type=float, default=1.0)
    parser.add_argument(
        "--counterfactual-margin", type=float, default=1.0
    )
    args = parser.parse_args()

    if args.adaptation not in ADAPTATIONS:
        raise RuntimeError("adaptation registry drift")
    if int(args.train_length) != 16_384:
        raise RuntimeError("formal matched protocol requires 16K training")
    if (
        args.compile_mode != "none"
        and bool(args.gradient_checkpointing)
    ):
        raise RuntimeError(
            "compiled formal candidate requires checkpointing off"
        )
    if int(args.eval_examples_per_cell) < 13:
        raise RuntimeError(
            "formal evaluation requires at least 104 examples per length"
        )
    train_source_fractions = tuple(
        float(value) for value in args.train_source_fractions
    )
    if not train_source_fractions or any(
        value <= 0.0 or value >= 1.0
        for value in train_source_fractions
    ):
        raise ValueError(
            "train source fractions must be strictly between 0 and 1"
        )
    if set(train_source_fractions) & set(EVAL_SOURCE_FRACTIONS):
        raise ValueError(
            "train source fractions must remain disjoint from formal "
            "evaluation bins"
        )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    seed_everything(int(args.seed))
    runtime = configure_cuda()
    checkpoint = args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True
    )
    train_values, eval_values = one_token_values(tokenizer)
    if args.train_value_pool == "broad_vocab":
        train_values = broad_vocabulary_train_values(
            tokenizer,
            excluded_token_ids={
                token_id for _, token_id in eval_values
            },
            count=int(args.broad_train_value_count),
            seed=int(args.seed) + 70_001,
        )
    documents_16k_path = args.documents_16k.resolve()
    metadata_16k_path = args.documents_16k_metadata.resolve()
    documents_16k, metadata_16k = load_documents(
        documents_16k_path,
        metadata_16k_path,
        expected_length=16_384,
    )
    train_document_rows = select_document_rows(
        metadata_16k, split="validation"
    )
    eval_document_rows_16k = select_document_rows(
        metadata_16k, split="test"
    )
    documents_by_length: dict[int, tuple[
        np.ndarray, list[dict[str, Any]], list[int]
    ]] = {
        length: (
            documents_16k,
            metadata_16k,
            eval_document_rows_16k,
        )
        for length in args.eval_lengths
        if int(length) <= 16_384
    }
    input_paths = [documents_16k_path, metadata_16k_path]
    if any(int(length) > 16_384 for length in args.eval_lengths):
        if (
            args.documents_32k is None
            or args.documents_32k_metadata is None
        ):
            raise RuntimeError("32K evaluation assets are required")
        documents_32k_path = args.documents_32k.resolve()
        metadata_32k_path = args.documents_32k_metadata.resolve()
        documents_32k, metadata_32k = load_documents(
            documents_32k_path,
            metadata_32k_path,
            expected_length=32_768,
        )
        eval_document_rows_32k = select_document_rows(
            metadata_32k, split="test"
        )
        for length in args.eval_lengths:
            if int(length) > 16_384:
                documents_by_length[int(length)] = (
                    documents_32k,
                    metadata_32k,
                    eval_document_rows_32k,
                )
        input_paths.extend([documents_32k_path, metadata_32k_path])

    train_data = build_probe_set(
        tokenizer=tokenizer,
        documents=documents_16k,
        document_metadata=metadata_16k,
        document_rows=train_document_rows,
        values=train_values,
        length=int(args.train_length),
        count=int(args.train_examples),
        seed=int(args.seed) + 1,
        source_fractions=train_source_fractions,
        distractor_counts=TRAIN_DISTRACTOR_COUNTS,
        phase="train",
        train_template_count=int(args.train_template_count),
    )
    canary_data = build_probe_set(
        tokenizer=tokenizer,
        documents=documents_16k,
        document_metadata=metadata_16k,
        document_rows=train_document_rows,
        values=train_values,
        length=int(args.train_length),
        count=max(8, int(args.canary_count)),
        seed=int(args.seed) + 50_000,
        source_fractions=train_source_fractions,
        distractor_counts=(0, 8),
        phase="train",
        train_template_count=int(args.train_template_count),
    )
    eval_count = (
        len(EVAL_SOURCE_FRACTIONS)
        * len(EVAL_DISTRACTOR_COUNTS)
        * int(args.eval_examples_per_cell)
    )
    eval_data: dict[int, FormalProbeSet] = {}
    for length in args.eval_lengths:
        documents, metadata, document_rows = documents_by_length[
            int(length)
        ]
        eval_data[int(length)] = build_probe_set(
            tokenizer=tokenizer,
            documents=documents,
            document_metadata=metadata,
            document_rows=document_rows,
            values=eval_values,
            length=int(length),
            count=eval_count,
            seed=int(args.seed) + 100_000 + int(length),
            source_fractions=EVAL_SOURCE_FRACTIONS,
            distractor_counts=EVAL_DISTRACTOR_COUNTS,
            phase="eval",
        )
    split_receipt = assert_generalization_split(
        train_data=train_data,
        eval_data=eval_data,
        train_values=train_values,
        eval_values=eval_values,
        train_source_fractions=train_source_fractions,
    )

    model = load_model(checkpoint)
    frequency = apply_schedule(model, args.schedule)
    readout = install_adaptation(
        model,
        args.adaptation,
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    parent_adapter_metadata: dict[str, Any] | None = None
    parent_adapter_sha256: str | None = None
    if args.resume_adapter is not None:
        if args.adaptation == "baseline":
            raise RuntimeError("baseline cannot resume an adapter")
        parent_path = args.resume_adapter.resolve()
        payload = torch.load(
            parent_path, map_location="cpu", weights_only=True
        )
        state = payload["state"]
        expected = dict(trainable_named_parameters(model, readout))
        if set(state) != set(expected):
            raise RuntimeError("resume adapter parameter set mismatch")
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(
                    state[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
        parent_adapter_metadata = dict(payload.get("metadata", {}))
        expected_parent = {
            "schedule": args.schedule,
            "adaptation": args.adaptation,
            "rank": int(args.rank),
            "alpha": float(args.alpha),
            "train_length": int(args.train_length),
            "train_dataset_sha256": train_data.digest(),
        }
        for key, expected_value in expected_parent.items():
            if parent_adapter_metadata.get(key) != expected_value:
                raise RuntimeError(
                    f"resume adapter metadata mismatch for {key}"
                )
        parent_adapter_sha256 = sha256_file(parent_path)
        input_paths.append(parent_path)
    model.to("cuda")
    if readout is not None:
        readout.to("cuda")
    effective_steps = (
        0 if args.adaptation == "baseline" else int(args.steps)
    )
    training = train(
        model=model,
        readout=readout,
        data=train_data,
        canary_data=canary_data,
        adaptation=args.adaptation,
        steps=effective_steps,
        micro_batch_size=int(args.micro_batch_size),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps
        ),
        learning_rate=float(args.learning_rate),
        warmup_steps=int(args.warmup_steps),
        seed=int(args.seed),
        log_path=output / "train_log.jsonl",
        canary_count=int(args.canary_count),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        compile_mode=args.compile_mode,
        margin_loss_weight=float(args.margin_loss_weight),
        top1_margin=float(args.top1_margin),
        counterfactual_margin=float(args.counterfactual_margin),
    )
    rows, summary = evaluate(
        model=model,
        readout=readout,
        data_by_length=eval_data,
        answer_token_ids=[
            token_id for _, token_id in eval_values
        ],
        batch_size=int(args.eval_batch_size),
    )
    annotate_rows(rows, eval_data)
    adapter_metadata = {
        "base_checkpoint_sha256": ":".join(
            sha256_file(path)
            for path in sorted(checkpoint.glob("model-*.safetensors"))
        ),
        "schedule": args.schedule,
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": args.adaptation,
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "train_length": int(args.train_length),
        "train_dataset_sha256": train_data.digest(),
        "parent_adapter_sha256": parent_adapter_sha256,
        "parent_adapter_metadata": parent_adapter_metadata,
    }
    adapter_sha = save_adapter(
        output / "adapter.pt",
        model,
        readout,
        adapter_metadata,
    )
    receipt = {
        "status": "OLMO2_MATCHED_LORA_GENERALIZATION_COMPLETE",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "training_implementation_sha256": sha256_file(
            Path(train.__code__.co_filename).resolve()
        ),
        "metric_boundary": (
            "fixed held-out paraphrased one-token associative retrieval "
            "with disjoint supervised values and PG19 documents; "
            "task-specific mapping evidence, not broad downstream ability"
        ),
        "checkpoint": str(checkpoint),
        "base_checkpoint_sha256": adapter_metadata[
            "base_checkpoint_sha256"
        ],
        "tokenizer_sha256": sha256_file(
            checkpoint / "tokenizer.json"
        ),
        "input_files_sha256": hash_paths(input_paths),
        "schedule": args.schedule,
        "frequency": frequency,
        "adaptation": args.adaptation,
        "adapter_sha256": adapter_sha,
        "seed": int(args.seed),
        "runtime": runtime,
        "split_contract": split_receipt,
        "protocol": {
            "train_length": int(args.train_length),
            "train_examples": int(args.train_examples),
            "eval_lengths": [
                int(value) for value in args.eval_lengths
            ],
            "eval_examples_per_length": eval_count,
            "eval_examples_per_cell": int(
                args.eval_examples_per_cell
            ),
            "train_source_fractions": list(
                TRAIN_SOURCE_FRACTIONS
            ),
            "eval_source_fractions": list(
                EVAL_SOURCE_FRACTIONS
            ),
            "train_distractor_counts": list(
                TRAIN_DISTRACTOR_COUNTS
            ),
            "eval_distractor_counts": list(
                EVAL_DISTRACTOR_COUNTS
            ),
            "rank": int(args.rank),
            "alpha": float(args.alpha),
            "train_value_pool": args.train_value_pool,
            "broad_train_value_count": (
                int(args.broad_train_value_count)
                if args.train_value_pool == "broad_vocab"
                else None
            ),
            "train_template_count": int(
                args.train_template_count
            ),
            "train_source_fractions": list(
                train_source_fractions
            ),
            "learning_rate": float(args.learning_rate),
            "micro_batch_size": int(args.micro_batch_size),
            "gradient_accumulation_steps": int(
                args.gradient_accumulation_steps
            ),
            "gradient_checkpointing": bool(
                args.gradient_checkpointing
            ),
            "compile_mode": args.compile_mode,
            "margin_loss_weight": float(args.margin_loss_weight),
            "top1_margin": float(args.top1_margin),
            "counterfactual_margin": float(
                args.counterfactual_margin
            ),
            "warmup_steps": int(args.warmup_steps),
            "fixed_steps": effective_steps,
            "resumed_from_adapter_sha256": parent_adapter_sha256,
            "train_dataset_sha256": train_data.digest(),
            "canary_dataset_sha256": canary_data.digest(),
            "eval_dataset_sha256": {
                str(length): data.digest()
                for length, data in sorted(eval_data.items())
            },
        },
        "training": training,
        "summary": summary,
        "rows": rows,
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "schedule": args.schedule,
                "adaptation": args.adaptation,
                "training": training,
                "summary": {
                    key: value
                    for key, value in summary.items()
                    if key in {
                        f"L{int(length)}"
                        for length in args.eval_lengths
                    }
                },
                "adapter_sha256": adapter_sha,
                "output": str(output / "results.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
