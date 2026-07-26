"""Source-causal protocol with continuous positions and template diversity."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Sequence

import numpy as np

from rebuttal.rebuttal_0723.experiments.olmo2_lora_generalization import (
    FormalProbeSet,
    encode,
    random_key,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    place_phrase,
)


TRAIN_SOURCE_TEMPLATES = (
    ("\nRecord {key} stores{value}.\n", "record_stores"),
    ("\nEntry [{key}] contains{value}.\n", "entry_contains"),
    ("\nThe value assigned to {key} is{value}.\n", "value_assigned"),
    ("\nReference <{key}> maps to{value}.\n", "reference_maps"),
    ("\nFor identifier {key}, remember{value}.\n", "identifier_remember"),
    ("\nLookup({key}) returns{value}.\n", "lookup_returns"),
    ("\nCatalog item {key}: value{value}.\n", "catalog_value"),
    ("\nKey={key}; stored answer={value}.\n", "key_answer"),
)

TRAIN_QUERY_TEMPLATES = (
    ("\nWhat value is stored for {key}? Answer:", "what_value"),
    ("\nReturn the value associated with [{key}]. Value:", "return_value"),
    ("\nResolve reference <{key}>. Result:", "resolve_reference"),
    ("\nLook up {key} and give its value:", "look_up"),
    ("\nWhich answer belongs to identifier {key}? Answer:", "which_answer"),
    ("\nRead Lookup({key}). Output:", "read_lookup"),
    ("\nRetrieve catalog item {key}. Value:", "retrieve_catalog"),
    ("\nFor key={key}, stored answer:", "for_key"),
)

EVAL_SOURCE_TEMPLATE = (
    "\nArchive entry [{key}] assigns the symbol{value}.\n"
)
EVAL_QUERY_TEMPLATE = (
    "\nWhich symbol belongs to archive identifier [{key}]? Answer:"
)


def _hash_index(key: str, purpose: str, size: int) -> int:
    digest = hashlib.sha256(
        f"{purpose}\0{key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % int(size)


def continuous_source_fractions(
    count: int,
    *,
    seed: int,
    lower: float = 0.05,
    upper: float = 0.95,
) -> tuple[float, ...]:
    """Return a deterministic low-discrepancy position sequence.

    Each row receives its own position.  A seed-dependent phase and an
    irrational rotation avoid the small repeated bins that failed in the
    original conversion probe.
    """

    if int(count) <= 0:
        raise ValueError("count must be positive")
    if not 0.0 < float(lower) < float(upper) < 1.0:
        raise ValueError("position interval must lie strictly inside (0, 1)")
    seed_digest = hashlib.sha256(str(int(seed)).encode("ascii")).digest()
    phase = int.from_bytes(seed_digest[:8], "big") / float(2**64)
    rotation = (math.sqrt(5.0) - 1.0) / 2.0
    width = float(upper) - float(lower)
    return tuple(
        float(lower) + width * ((phase + index * rotation) % 1.0)
        for index in range(int(count))
    )


def render_phrases(
    *,
    tokenizer: Any,
    key: str,
    value: str,
    template_mode: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if template_mode == "train_anchor":
        source_format, source_id = TRAIN_SOURCE_TEMPLATES[0]
        query_format, query_id = TRAIN_QUERY_TEMPLATES[0]
    elif template_mode == "train_compositional":
        source_format, source_id = TRAIN_SOURCE_TEMPLATES[
            _hash_index(key, "source-template", len(TRAIN_SOURCE_TEMPLATES))
        ]
        query_format, query_id = TRAIN_QUERY_TEMPLATES[
            _hash_index(key, "query-template", len(TRAIN_QUERY_TEMPLATES))
        ]
    elif template_mode == "eval_unseen":
        source_format, source_id = (
            EVAL_SOURCE_TEMPLATE,
            "archive_assigns",
        )
        query_format, query_id = (
            EVAL_QUERY_TEMPLATE,
            "archive_question",
        )
    else:
        raise ValueError(f"unknown template mode {template_mode!r}")
    source = encode(
        tokenizer, source_format.format(key=key, value=value)
    )
    query = encode(tokenizer, query_format.format(key=key))
    return source, query, f"{source_id}+{query_id}"


def build_source_probe_set(
    *,
    tokenizer: Any,
    documents: np.ndarray,
    document_metadata: list[dict[str, Any]],
    document_rows: Sequence[int],
    values: list[tuple[str, int]],
    length: int,
    count: int,
    seed: int,
    source_fractions: Sequence[float] | None,
    distractor_counts: Sequence[int],
    template_mode: str,
    namespace: str,
) -> FormalProbeSet:
    """Build paired source/deletion/swap contexts.

    ``source_fractions=None`` activates deterministic continuous-uniform
    placement.  Explicit fractions retain the old factorial anchors.
    """

    if not document_rows:
        raise ValueError("document_rows is empty")
    if len(values) < 2:
        raise ValueError("at least two answer values are required")
    if not distractor_counts:
        raise ValueError("distractor_counts is empty")
    continuous_positions = source_fractions is None
    fractions_requested = (
        continuous_source_fractions(count, seed=seed + 91)
        if continuous_positions
        else tuple(float(value) for value in source_fractions)
    )
    if not fractions_requested:
        raise ValueError("source_fractions is empty")
    if any(value <= 0.0 or value >= 1.0 for value in fractions_requested):
        raise ValueError("source fractions must lie strictly inside (0, 1)")

    rng = random.Random(int(seed))
    sourced_rows: list[np.ndarray] = []
    deleted_rows: list[np.ndarray] = []
    swapped_rows: list[np.ndarray] = []
    gold_ids: list[int] = []
    alternate_ids: list[int] = []
    actual_fractions: list[float] = []
    densities: list[int] = []
    keys: list[str] = []
    value_words: list[str] = []
    document_sources: list[str] = []
    selected_document_rows: list[int] = []
    crop_offsets: list[int] = []
    template_ids: list[str] = []
    context_length = int(length) - 1

    for row_index in range(int(count)):
        requested_fraction = fractions_requested[
            row_index % len(fractions_requested)
        ]
        density_index = (
            row_index
            if continuous_positions
            else row_index // len(fractions_requested)
        )
        density = int(
            distractor_counts[density_index % len(distractor_counts)]
        )
        gold_index = rng.randrange(len(values))
        alternate_index = rng.randrange(len(values) - 1)
        if alternate_index >= gold_index:
            alternate_index += 1
        gold_word, gold_id = values[gold_index]
        alternate_word, alternate_id = values[alternate_index]
        key = random_key(rng, namespace)
        source, query, template_id = render_phrases(
            tokenizer=tokenizer,
            key=key,
            value=gold_word,
            template_mode=template_mode,
        )
        swapped_source, swapped_query, swapped_template_id = render_phrases(
            tokenizer=tokenizer,
            key=key,
            value=alternate_word,
            template_mode=template_mode,
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

        document_row = int(document_rows[rng.randrange(len(document_rows))])
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
        for distractor_index in range(density):
            distractor_word, _ = distractor_values[
                distractor_index % len(distractor_values)
            ]
            distractor_key = random_key(rng, namespace)
            distractor, _, _ = render_phrases(
                tokenizer=tokenizer,
                key=distractor_key,
                value=distractor_word,
                template_mode=template_mode,
            )
            preferred = int(
                (distractor_index + 1) * usable / (density + 1)
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
        gold_ids.append(int(gold_id))
        alternate_ids.append(int(alternate_id))
        actual_fractions.append(float(source_start / usable))
        densities.append(density)
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
        source_fraction=np.asarray(actual_fractions, dtype=np.float64),
        distractor_count=np.asarray(densities, dtype=np.int64),
        keys=keys,
        value_words=value_words,
        document_sources=document_sources,
        document_rows=np.asarray(selected_document_rows, dtype=np.int64),
        crop_offsets=np.asarray(crop_offsets, dtype=np.int64),
        template_ids=template_ids,
    )


def position_coverage(data: FormalProbeSet) -> dict[str, Any]:
    fractions = np.asarray(data.source_fraction, dtype=np.float64)
    deciles = np.minimum(9, np.floor(fractions * 10.0).astype(np.int64))
    rounded_positions = np.round(fractions, decimals=6)
    return {
        "minimum": float(fractions.min()),
        "maximum": float(fractions.max()),
        "unique_fraction_count_1e6": int(len(np.unique(rounded_positions))),
        "decile_counts": {
            str(index): int((deciles == index).sum())
            for index in range(10)
        },
    }
