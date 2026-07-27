#!/usr/bin/env python3
"""Repair EVQ routing with paired counterfactual supervision at 4K only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    TrainingBackbone,
    install_adaptation,
    load_model,
    save_adapter,
    trainable_named_parameters,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    append_jsonl,
    atomic_json,
    configure_cuda,
    cosine_lr,
    rank_of,
    seed_everything,
    sha256_file,
)

from .prepare_4k_routing_pairs import (
    LENGTH,
    OLMO2_EOS_TOKEN_ID,
    ROOT_STATUS,
    SET_STATUS,
    SUPERVISION_CONTRACT,
)
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import (
    apply_frequency,
    evaluate_natural_nll,
    fused_loss_module,
    load_fixed_view,
    natural_batch,
)


FAMILY_PATTERN = ("routing", "routing", "natural")
PAIR_SET_STATUSES = {
    SET_STATUS,
    "OLMO2_4K_COUNTERFACTUAL_NATURAL_MULTIQUERY_SET_PREPARED",
}
PAIR_COLLECTION_STATUSES = {
    ROOT_STATUS,
    "OLMO2_4K_COUNTERFACTUAL_NATURAL_MULTIQUERY_DATA_PREPARED",
}
NATURAL_MULTIQUERY_READY_STATUS = "OLMO2_4K_NATURAL_MULTIQUERY_READY"
VIRTUAL_QUERY_GAP_READY_STATUS = (
    "OLMO2_4K_QUERY_GAP_EOS_REPAIR_READY_V1"
)
TRAIN_QUERY_MARKER = "Which access code belongs to"
CALIBRATION_QUERY_MARKER = "Return the identifier assigned to"


def bound_code_sha256() -> dict[str, str]:
    trainer = Path(__file__).resolve()
    maturity_root = trainer.parent
    experiments_root = maturity_root.parent
    paths = {
        "trainer": trainer,
        "routing_data_contract": (
            maturity_root / "prepare_4k_routing_pairs.py"
        ),
        "training_primitives": maturity_root / "train_screen.py",
        "checkpoint_contract": maturity_root / "train_4k_stage_a.py",
        "lora_conversion": experiments_root / "olmo2_lora_conversion.py",
        "adapter_loader": (
            experiments_root / "olmo2_lora_ood_factorial.py"
        ),
        "shared_training_utils": (
            experiments_root / "small_model_lora_conversion.py"
        ),
        "evq_contract": experiments_root / "olmo2_1b_evq" / "contract.py",
    }
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }


class RoutingPairView:
    def __init__(
        self,
        path: Path,
        *,
        require_virtual_geometry: bool = False,
        query_marker_token_ids: tuple[int, ...] | None = None,
    ) -> None:
        self.path = path
        self.manifest = json.loads(
            (path / "manifest.json").read_text(encoding="utf-8")
        )
        status = self.manifest.get("status")
        if status not in PAIR_SET_STATUSES:
            raise RuntimeError(f"routing set is not prepared: {path}")
        if (
            int(self.manifest["maximum_training_length"]) != LENGTH
            or int(self.manifest["maximum_training_position_id"])
            != LENGTH - 1
            or int(self.manifest["queries_per_sequence"]) < 1
        ):
            raise RuntimeError("routing set violates the 4K contract")
        self.single_query_v2 = status == SET_STATUS
        if self.single_query_v2 and (
            int(self.manifest.get("format_version", -1)) != 2
            or self.manifest.get("supervision_contract")
            != SUPERVISION_CONTRACT
            or int(self.manifest.get("eos_token_id", -1))
            != OLMO2_EOS_TOKEN_ID
            or self.manifest.get("final_eos_supervised") is not True
            or self.manifest.get(
                "labels_only_cover_answer_and_final_eos"
            )
            is not True
        ):
            raise RuntimeError(
                "routing set lacks the answer-plus-immediate-EOS contract"
            )
        if not self.single_query_v2 and (
            int(self.manifest.get("format_version", -1)) != 1
            or self.manifest.get("final_eos_supervised") is not True
            or self.manifest.get(
                "labels_only_cover_answers_and_final_eos"
            )
            is not True
        ):
            raise RuntimeError("natural multi-query contract drift")
        self.eos_token_id = OLMO2_EOS_TOKEN_ID
        self.input_ids = np.load(
            path / "input_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.labels = np.load(
            path / "labels.npy", mmap_mode="r", allow_pickle=False
        )
        expected = tuple(int(value) for value in self.manifest["shape"])
        if (
            tuple(self.input_ids.shape) != expected
            or tuple(self.labels.shape) != expected
            or expected[1:] != (2, LENGTH)
        ):
            raise RuntimeError("routing set shape drift")
        if self.input_ids.dtype != np.uint32:
            raise RuntimeError("routing input dtype drift")
        if self.labels.dtype != np.int32:
            raise RuntimeError("routing label dtype drift")
        for name, expected_digest in self.manifest["files"].items():
            if sha256_file(path / name) != expected_digest:
                raise RuntimeError(f"routing data hash drift: {path / name}")
        self.rows = [
            json.loads(line)
            for line in (path / "rows.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if len(self.rows) != expected[0]:
            raise RuntimeError("routing metadata row-count drift")
        if self.single_query_v2:
            answer_token_total = 0
            eos_token_total = 0
            supervised_token_total = 0
            for index, row in enumerate(self.rows):
                answer_start = int(row["answer_start"])
                answer_tokens = int(row["answer_tokens"])
                eos_position = answer_start + answer_tokens
                active_length = eos_position + 1
                if not 0 < answer_start < active_length <= LENGTH:
                    raise RuntimeError(
                        "routing active-length metadata drift"
                    )
                if int(row.get("eos_position", -1)) != eos_position:
                    raise RuntimeError(
                        "routing EOS-position metadata drift"
                    )
                if int(row.get("supervised_tokens", -1)) != (
                    answer_tokens + 1
                ):
                    raise RuntimeError(
                        "routing supervised-token metadata drift"
                    )
                label_masks = self.labels[index] != -100
                expected_mask = np.zeros((2, LENGTH), dtype=bool)
                expected_mask[:, answer_start:active_length] = True
                if not np.array_equal(label_masks, expected_mask):
                    raise RuntimeError(
                        "routing answer-plus-EOS label geometry drift"
                    )
                for variant, token_key in (
                    (0, "gold_token_ids"),
                    (1, "alternate_token_ids"),
                ):
                    answer_ids = np.asarray(
                        [int(value) for value in row[token_key]],
                        dtype=np.int32,
                    )
                    if len(answer_ids) != answer_tokens:
                        raise RuntimeError(
                            "routing answer-token metadata drift"
                        )
                    observed_labels = self.labels[
                        index,
                        variant,
                        answer_start:eos_position,
                    ]
                    observed_inputs = self.input_ids[
                        index,
                        variant,
                        answer_start:eos_position,
                    ]
                    if (
                        not np.array_equal(observed_labels, answer_ids)
                        or not np.array_equal(
                            observed_inputs,
                            answer_ids.astype(np.uint32),
                        )
                    ):
                        raise RuntimeError(
                            "routing answer labels drift"
                        )
                eos_labels = self.labels[index, :, eos_position]
                eos_inputs = self.input_ids[index, :, eos_position]
                if (
                    not np.all(eos_labels == self.eos_token_id)
                    or not np.all(eos_inputs == self.eos_token_id)
                ):
                    raise RuntimeError("routing final-EOS label drift")
                answer_token_total += 2 * answer_tokens
                eos_token_total += 2
                supervised_token_total += 2 * (answer_tokens + 1)
            if (
                answer_token_total
                != int(self.manifest["supervised_answer_tokens"])
                or eos_token_total
                != int(self.manifest["supervised_eos_tokens"])
                or supervised_token_total
                != int(
                    self.manifest[
                        "supervised_answer_and_eos_tokens"
                    ]
                )
            ):
                raise RuntimeError(
                    "routing supervised-token aggregate drift"
                )
        self.virtual_geometry_ready = False
        if not require_virtual_geometry:
            return
        if (
            not self.single_query_v2
            or int(self.manifest["queries_per_sequence"]) != 1
        ):
            raise RuntimeError(
                "virtual query-gap training supports one query per sequence"
            )
        if not query_marker_token_ids:
            raise RuntimeError("virtual query-gap marker tokens are required")
        marker = list(query_marker_token_ids)
        self.source_starts = np.empty(expected[0], dtype=np.int64)
        self.source_stops = np.empty(expected[0], dtype=np.int64)
        self.query_starts = np.empty(expected[0], dtype=np.int64)
        self.answer_starts = np.empty(expected[0], dtype=np.int64)
        self.active_lengths = np.empty(expected[0], dtype=np.int64)
        for index, row in enumerate(self.rows):
            answer_start = int(row["answer_start"])
            answer_tokens = int(row["answer_tokens"])
            active_length = answer_start + answer_tokens + 1
            if not 0 < answer_start < active_length <= LENGTH:
                raise RuntimeError("routing active-length metadata drift")
            starts = []
            stops = []
            for variant, token_key in (
                (0, "gold_token_ids"),
                (1, "alternate_token_ids"),
            ):
                pattern = [int(value) for value in row[token_key]]
                sequence = self.input_ids[index, variant, :answer_start]
                occurrences = [
                    start
                    for start in range(
                        0, answer_start - len(pattern) + 1
                    )
                    if sequence[
                        start : start + len(pattern)
                    ].tolist()
                    == pattern
                ]
                if len(occurrences) != 1:
                    raise RuntimeError(
                        "routing source value is not uniquely recoverable"
                    )
                starts.append(int(occurrences[0]))
                stops.append(int(occurrences[0] + len(pattern)))
            if starts[0] != starts[1] or stops[0] != stops[1]:
                raise RuntimeError(
                    "counterfactual pair source geometry drift"
                )
            if (
                not np.array_equal(
                    self.input_ids[index, 0, : starts[0]],
                    self.input_ids[index, 1, : starts[1]],
                )
                or not np.array_equal(
                    self.input_ids[index, 0, stops[0] : answer_start],
                    self.input_ids[index, 1, stops[1] : answer_start],
                )
            ):
                raise RuntimeError(
                    "counterfactual prompt differs outside source value"
                )
            if not stops[0] < answer_start:
                raise RuntimeError("source does not precede answer query")
            query_starts = []
            for variant in range(2):
                sequence = self.input_ids[
                    index, variant, :answer_start
                ]
                occurrences = [
                    start
                    for start in range(
                        0, answer_start - len(marker) + 1
                    )
                    if sequence[
                        start : start + len(marker)
                    ].tolist()
                    == marker
                ]
                if len(occurrences) != 1:
                    raise RuntimeError(
                        "routing query boundary is not uniquely recoverable"
                    )
                query_starts.append(int(occurrences[0]))
            if query_starts[0] != query_starts[1]:
                raise RuntimeError(
                    "counterfactual pair query geometry drift"
                )
            if not stops[0] <= query_starts[0] < answer_start:
                raise RuntimeError(
                    "routing source/query semantic order drift"
                )
            self.source_starts[index] = starts[0]
            self.source_stops[index] = stops[0]
            self.query_starts[index] = query_starts[0]
            self.answer_starts[index] = answer_start
            self.active_lengths[index] = active_length
        self.virtual_geometry_ready = True


def query_gap_position_array(
    *,
    view: RoutingPairView,
    row_indices: np.ndarray,
    query_offsets: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Shift the semantic query block without revealing the gold source."""

    row_indices = np.asarray(row_indices, dtype=np.int64)
    query_offsets = np.asarray(query_offsets, dtype=np.int64)
    if row_indices.shape != query_offsets.shape:
        raise RuntimeError("virtual query-offset shape drift")
    if not view.virtual_geometry_ready:
        raise RuntimeError("virtual query-gap geometry is unavailable")
    physical = np.arange(LENGTH - 1, dtype=np.int64)
    pair_positions = np.broadcast_to(
        physical, (len(row_indices), LENGTH - 1)
    ).copy()
    receipts: list[dict[str, int]] = []
    for local_index, (row_index, query_offset) in enumerate(
        zip(row_indices.tolist(), query_offsets.tolist())
    ):
        active_length = int(view.active_lengths[row_index])
        active_context = min(active_length, LENGTH - 1)
        source_start = int(view.source_starts[row_index])
        source_stop = int(view.source_stops[row_index])
        query_start = int(view.query_starts[row_index])
        answer_start = int(view.answer_starts[row_index])
        if not 0 <= query_offset <= 3 * LENGTH + 1:
            raise RuntimeError("virtual query offset is outside 16K support")
        if query_offset:
            pair_positions[
                local_index, query_start:
            ] += int(query_offset)
        positions = pair_positions[local_index]
        active = positions[:active_context]
        if (
            int(active[0]) != 0
            or np.any(np.diff(positions) <= 0)
            or np.any(np.diff(positions[:query_start]) != 1)
            or np.any(np.diff(positions[query_start:]) != 1)
            or int(positions[-1]) >= 4 * LENGTH
        ):
            raise RuntimeError("virtual query-gap position contract drift")
        jump = int(positions[query_start] - positions[query_start - 1])
        if jump != int(query_offset) + 1:
            raise RuntimeError("virtual query-boundary jump drift")
        prediction_index = answer_start - 1
        physical_near = prediction_index - (source_stop - 1)
        physical_far = prediction_index - source_start
        virtual_near = (
            int(positions[prediction_index])
            - int(positions[source_stop - 1])
        )
        virtual_far = (
            int(positions[prediction_index])
            - int(positions[source_start])
        )
        if (
            virtual_near != physical_near + int(query_offset)
            or virtual_far != physical_far + int(query_offset)
        ):
            raise RuntimeError("realized query/source gap drift")
        receipts.append(
            {
                "row": int(row_index),
                "physical_sequence_length": LENGTH,
                "query_offset": int(query_offset),
                "active_length": active_length,
                "source_start": source_start,
                "source_stop": source_stop,
                "query_start": query_start,
                "answer_prediction_index": prediction_index,
                "physical_gap_near": physical_near,
                "physical_gap_far": physical_far,
                "virtual_gap_near": virtual_near,
                "virtual_gap_far": virtual_far,
                "virtual_query_prediction_position": int(
                    positions[prediction_index]
                ),
                "maximum_active_position_id": int(
                    positions[active_context - 1]
                ),
                "position_array_sha256": hashlib.sha256(
                    positions.astype("<i8", copy=False).tobytes(order="C")
                ).hexdigest(),
            }
        )
    return pair_positions, receipts


def query_gap_position_ids(
    *,
    view: RoutingPairView,
    row_indices: np.ndarray,
    query_offsets: np.ndarray,
) -> tuple[torch.Tensor, list[dict[str, int]], bytes]:
    pair_positions, receipts = query_gap_position_array(
        view=view,
        row_indices=row_indices,
        query_offsets=query_offsets,
    )
    flattened = np.repeat(pair_positions, 2, axis=0)
    if not np.array_equal(flattened[0::2], flattened[1::2]):
        raise RuntimeError(
            "counterfactual variants do not share query positions"
        )
    payload = flattened.tobytes(order="C")
    return (
        torch.from_numpy(flattened).to("cuda", non_blocking=True),
        receipts,
        payload,
    )


def _deterministic_band_values(
    *,
    low: int,
    high: int,
    count: int,
    seed: int,
    label: str,
) -> np.ndarray:
    if not 0 <= low <= high or count < 0:
        raise ValueError("invalid deterministic offset band")
    width = int(high - low + 1)
    digest = hashlib.sha256(
        f"evq-query-gap-v1\0{int(seed)}\0{label}".encode("ascii")
    ).digest()
    start = int.from_bytes(digest[:8], "little") % width
    stride = 1 + int.from_bytes(digest[8:16], "little") % width
    while math.gcd(stride, width) != 1:
        stride = 1 if stride == width else stride + 1
    values = (
        low
        + (
            start
            + stride * np.arange(int(count), dtype=np.int64)
        )
        % width
    )
    if len(np.unique(values)) != min(int(count), width):
        raise RuntimeError("deterministic offset band coverage drift")
    return values.astype("<i8", copy=False)


def deterministic_query_offset_stream(
    *,
    seed: int,
    routing_steps: int,
) -> np.ndarray:
    """Return four row-independent offsets per routing optimizer step."""

    routing_steps = int(routing_steps)
    if routing_steps <= 0:
        raise ValueError("routing steps must be positive")
    transition_count = routing_steps // 2
    transition = _deterministic_band_values(
        low=1,
        high=LENGTH,
        count=transition_count,
        seed=seed,
        label="transition",
    )
    middle = _deterministic_band_values(
        low=LENGTH + 1,
        high=2 * LENGTH,
        count=routing_steps,
        seed=seed,
        label="middle",
    )
    far = _deterministic_band_values(
        low=2 * LENGTH + 1,
        high=3 * LENGTH + 1,
        count=2 * routing_steps,
        seed=seed,
        label="far",
    )
    stream: list[int] = []
    transition_cursor = 0
    for routing_ordinal in range(routing_steps):
        low_offset = 0
        if routing_ordinal % 2:
            low_offset = int(transition[transition_cursor])
            transition_cursor += 1
        local = [
            low_offset,
            int(middle[routing_ordinal]),
            int(far[2 * routing_ordinal]),
            int(far[2 * routing_ordinal + 1]),
        ]
        digest = hashlib.sha256(
            (
                "evq-query-gap-order-v1\0"
                f"{int(seed)}\0{routing_ordinal}"
            ).encode("ascii")
        ).digest()
        order = sorted(range(4), key=lambda index: (digest[index], index))
        stream.extend(local[index] for index in order)
    if transition_cursor != transition_count:
        raise RuntimeError("deterministic transition cursor drift")
    values = np.asarray(stream, dtype="<i8")
    expected = {
        "contiguous": (routing_steps + 1) // 2,
        "transition": routing_steps // 2,
        "middle": routing_steps,
        "far": 2 * routing_steps,
    }
    actual = {
        "contiguous": int((values == 0).sum()),
        "transition": int(
            ((values >= 1) & (values <= LENGTH)).sum()
        ),
        "middle": int(
            (
                (values >= LENGTH + 1)
                & (values <= 2 * LENGTH)
            ).sum()
        ),
        "far": int(
            (
                (values >= 2 * LENGTH + 1)
                & (values <= 3 * LENGTH + 1)
            ).sum()
        ),
    }
    if actual != expected or len(values) != 4 * routing_steps:
        raise RuntimeError("deterministic query-offset quota drift")
    return values


def query_offset_band(offset: int) -> str:
    if offset == 0:
        return "contiguous"
    if 1 <= offset <= LENGTH:
        return "transition"
    if LENGTH + 1 <= offset <= 2 * LENGTH:
        return "middle"
    if 2 * LENGTH + 1 <= offset <= 3 * LENGTH + 1:
        return "far"
    raise RuntimeError("query offset lies outside registered bands")


def routing_batch(
    *,
    view: RoutingPairView,
    row_indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    host_ids = np.asarray(
        view.input_ids[row_indices], dtype=np.int64
    )
    host_labels = np.asarray(
        view.labels[row_indices], dtype=np.int64
    )
    alternate = host_labels[:, ::-1, :].copy()
    contexts = torch.from_numpy(
        host_ids[:, :, :-1].reshape(-1, LENGTH - 1).copy()
    ).to("cuda", non_blocking=True)
    labels = torch.from_numpy(
        host_labels[:, :, 1:].reshape(-1, LENGTH - 1).copy()
    ).to("cuda", non_blocking=True)
    alternate_labels = torch.from_numpy(
        alternate[:, :, 1:].reshape(-1, LENGTH - 1)
    ).to("cuda", non_blocking=True)
    mask = labels != -100
    if not torch.equal(mask, alternate_labels != -100):
        raise RuntimeError("paired routing answer geometry drift")
    supervised = int(mask.sum())
    if supervised <= 0:
        raise RuntimeError("routing batch has no supervised tokens")
    return contexts, labels, alternate_labels, supervised


def routing_objective(
    *,
    model: Any,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    alternate_labels: torch.Tensor,
    eos_token_id: int,
    margin: float,
    margin_weight: float,
    termination_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = labels != -100
    selected_hidden = hidden[mask]
    gold = labels[mask]
    alternate = alternate_labels[mask]
    logits = model.lm_head(selected_hidden).float()
    termination_mask = gold.eq(int(eos_token_id))
    answer_mask = ~termination_mask
    if not torch.any(answer_mask) or not torch.any(termination_mask):
        raise RuntimeError(
            "routing batch must contain answer and terminal-EOS targets"
        )
    answer_logits = logits[answer_mask]
    answer_gold = gold[answer_mask]
    termination_logits = logits[termination_mask]
    termination_gold = gold[termination_mask]
    eos_id = int(eos_token_id)
    answer_eos_logits = answer_logits[:, eos_id]
    answer_gold_logits = answer_logits.gather(
        1, answer_gold[:, None]
    ).squeeze(1)
    termination_eos_logits = termination_logits[:, eos_id]
    termination_max_non_eos = torch.maximum(
        termination_logits[:, :eos_id].amax(dim=-1),
        termination_logits[:, eos_id + 1 :].amax(dim=-1),
    )
    answer_ce = F.cross_entropy(answer_logits, answer_gold)
    termination_ce = F.cross_entropy(
        termination_logits, termination_gold
    )
    combined_ce = F.cross_entropy(logits, gold)
    source_mask = answer_mask & gold.ne(alternate)
    if not torch.any(source_mask):
        raise RuntimeError("paired routing batch has no source-dependent labels")
    if torch.any(termination_mask & gold.ne(alternate)):
        raise RuntimeError("terminal EOS leaked into counterfactual margin")
    source_logits = logits[source_mask]
    source_gold = gold[source_mask]
    source_alternate = alternate[source_mask]
    gold_logits = source_logits.gather(
        1, source_gold[:, None]
    ).squeeze(1)
    alternate_logits = source_logits.gather(
        1, source_alternate[:, None]
    ).squeeze(1)
    preference = gold_logits - alternate_logits
    counterfactual = F.softplus(float(margin) - preference).mean()
    total = (
        answer_ce
        + float(termination_weight) * termination_ce
        + float(margin_weight) * counterfactual
    )
    with torch.no_grad():
        metrics = {
            "combined_target_ce": float(combined_ce),
            "answer_ce": float(answer_ce),
            "termination_eos_ce": float(termination_ce),
            "counterfactual_loss": float(counterfactual),
            "preference_mean": float(preference.mean()),
            "preference_positive_fraction": float(
                (preference > 0).float().mean()
            ),
            "combined_target_token_exact": float(
                logits.argmax(dim=-1).eq(gold).float().mean()
            ),
            "answer_token_exact": float(
                answer_logits.argmax(dim=-1)
                .eq(answer_gold)
                .float()
                .mean()
            ),
            "termination_eos_exact": float(
                termination_logits.argmax(dim=-1)
                .eq(termination_gold)
                .float()
                .mean()
            ),
            "answer_gold_minus_eos_margin_mean": float(
                (answer_gold_logits - answer_eos_logits).mean()
            ),
            "answer_gold_minus_eos_margin_min": float(
                (answer_gold_logits - answer_eos_logits).min()
            ),
            "termination_eos_minus_max_non_eos_margin_mean": float(
                (
                    termination_eos_logits
                    - termination_max_non_eos
                ).mean()
            ),
            "termination_eos_minus_max_non_eos_margin_min": float(
                (
                    termination_eos_logits
                    - termination_max_non_eos
                ).min()
            ),
            "source_token_exact": float(
                source_logits.argmax(dim=-1)
                .eq(source_gold)
                .float()
                .mean()
            ),
            "answer_tokens": float(answer_gold.numel()),
            "termination_eos_tokens": float(
                termination_gold.numel()
            ),
            "source_tokens": float(source_gold.numel()),
        }
    return total, metrics


@torch.inference_mode()
def evaluate_routing(
    *,
    model: Any,
    view: RoutingPairView,
    rows: int,
    pair_batch_size: int,
    margin: float,
    query_offset: int = 0,
) -> dict[str, Any]:
    n = min(int(rows), len(view.input_ids))
    totals = {
        "combined_tokens": 0,
        "combined_nll_sum": 0.0,
        "combined_exact": 0,
        "answer_tokens": 0,
        "answer_nll_sum": 0.0,
        "answer_exact": 0,
        "answer_ranks": [],
        "termination_tokens": 0,
        "termination_nll_sum": 0.0,
        "termination_exact": 0,
        "source_tokens": 0,
        "source_exact": 0,
        "preference_sum": 0.0,
        "preference_positive": 0,
        "margin_satisfied": 0,
        "virtual_gap_near_min": None,
        "virtual_gap_far_max": None,
    }
    model.eval()
    for start in range(0, n, int(pair_batch_size)):
        indices = np.arange(start, min(n, start + pair_batch_size))
        contexts, labels, alternate_labels, _ = routing_batch(
            view=view, row_indices=indices
        )
        position_ids = None
        exposures: list[dict[str, int]] = []
        if int(query_offset):
            position_ids, exposures, _ = query_gap_position_ids(
                view=view,
                row_indices=indices,
                query_offsets=np.full(
                    len(indices), int(query_offset), dtype=np.int64
                ),
            )
            batch_near = min(
                row["virtual_gap_near"] for row in exposures
            )
            batch_far = max(
                row["virtual_gap_far"] for row in exposures
            )
            totals["virtual_gap_near_min"] = (
                batch_near
                if totals["virtual_gap_near_min"] is None
                else min(totals["virtual_gap_near_min"], batch_near)
            )
            totals["virtual_gap_far_max"] = (
                batch_far
                if totals["virtual_gap_far_max"] is None
                else max(totals["virtual_gap_far_max"], batch_far)
            )
        mask = labels != -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(
                input_ids=contexts,
                position_ids=position_ids,
                use_cache=False,
                return_dict=False,
            )[0]
            logits = model.lm_head(hidden[mask]).float()
        gold = labels[mask]
        alternate = alternate_labels[mask]
        nll = F.cross_entropy(logits, gold, reduction="none")
        termination_mask = gold.eq(int(view.eos_token_id))
        answer_mask = ~termination_mask
        if not torch.any(answer_mask) or not torch.any(termination_mask):
            raise RuntimeError(
                "routing calibration lacks answer or EOS targets"
            )
        answer_logits = logits[answer_mask]
        answer_gold = gold[answer_mask]
        answer_nll = nll[answer_mask]
        answer_ranks = rank_of(answer_logits, answer_gold)
        termination_logits = logits[termination_mask]
        termination_gold = gold[termination_mask]
        termination_nll = nll[termination_mask]
        source_mask = answer_mask & gold.ne(alternate)
        if not torch.any(source_mask):
            raise RuntimeError(
                "paired routing calibration has no source-dependent labels"
            )
        source_logits = logits[source_mask]
        source_gold = gold[source_mask]
        source_alternate = alternate[source_mask]
        preference = (
            source_logits.gather(
                1, source_gold[:, None]
            ).squeeze(1)
            - source_logits.gather(
                1, source_alternate[:, None]
            ).squeeze(1)
        )
        count = int(gold.numel())
        answer_count = int(answer_gold.numel())
        termination_count = int(termination_gold.numel())
        totals["combined_tokens"] += count
        totals["combined_nll_sum"] += float(nll.sum())
        totals["combined_exact"] += int(
            logits.argmax(dim=-1).eq(gold).sum()
        )
        totals["answer_tokens"] += answer_count
        totals["answer_nll_sum"] += float(answer_nll.sum())
        totals["answer_exact"] += int(
            answer_logits.argmax(dim=-1).eq(answer_gold).sum()
        )
        totals["termination_tokens"] += termination_count
        totals["termination_nll_sum"] += float(
            termination_nll.sum()
        )
        totals["termination_exact"] += int(
            termination_logits.argmax(dim=-1)
            .eq(termination_gold)
            .sum()
        )
        totals["source_tokens"] += int(source_gold.numel())
        totals["source_exact"] += int(
            source_logits.argmax(dim=-1).eq(source_gold).sum()
        )
        totals["answer_ranks"].extend(
            int(value)
            for value in answer_ranks.detach().cpu().tolist()
        )
        totals["preference_sum"] += float(preference.sum())
        totals["preference_positive"] += int((preference > 0).sum())
        totals["margin_satisfied"] += int(
            (preference >= float(margin)).sum()
        )
        del contexts, labels, alternate_labels, hidden, logits, position_ids
    count = int(totals["combined_tokens"])
    answer_count = int(totals["answer_tokens"])
    termination_count = int(totals["termination_tokens"])
    source_count = int(totals["source_tokens"])
    return {
        "pairs": n,
        "supervised_answer_and_eos_tokens": count,
        "mean_supervised_answer_and_eos_nll": (
            totals["combined_nll_sum"] / count
        ),
        "supervised_answer_and_eos_token_exact": (
            totals["combined_exact"] / count
        ),
        "answer_tokens": answer_count,
        "mean_answer_nll": totals["answer_nll_sum"] / answer_count,
        "answer_token_exact": totals["answer_exact"] / answer_count,
        "termination_eos_tokens": termination_count,
        "mean_termination_eos_nll": (
            totals["termination_nll_sum"] / termination_count
        ),
        "termination_eos_exact": (
            totals["termination_exact"] / termination_count
        ),
        "source_answer_tokens": source_count,
        "source_token_exact": totals["source_exact"] / source_count,
        "median_full_vocab_rank": float(
            np.median(totals["answer_ranks"])
        ),
        "mean_counterfactual_preference": (
            totals["preference_sum"] / source_count
        ),
        "preference_positive_fraction": (
            totals["preference_positive"] / source_count
        ),
        "margin_satisfied_fraction": (
            totals["margin_satisfied"] / source_count
        ),
        "position_policy": (
            "contiguous"
            if not int(query_offset)
            else "semantic_query_block_gap"
        ),
        "physical_sequence_length": LENGTH,
        "query_offset": int(query_offset),
        "virtual_gap_near_min": totals["virtual_gap_near_min"],
        "virtual_gap_far_max": totals["virtual_gap_far_max"],
        "capability_endpoint": False,
    }


def train(
    *,
    model: Any,
    routing_view: RoutingPairView,
    calibration_view: RoutingPairView,
    natural_view_path: Path,
    steps: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    margin: float,
    margin_weight: float,
    termination_weight: float,
    compile_mode: str,
    seed: int,
    log_path: Path,
    virtual_target_length: int = 0,
    virtual_bucket_weights: tuple[int, int, int] = (1, 1, 2),
    family_pattern: tuple[str, ...] = FAMILY_PATTERN,
    calibration_rows_during_training: int = 8,
    checkpoint_steps: tuple[int, ...] = (),
    checkpoint_callback: (
        Callable[[int, Any], dict[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    if int(micro_batch_size) % 2:
        raise ValueError("micro-batch-size must be even")
    if (
        not family_pattern
        or any(value not in {"routing", "natural"} for value in family_pattern)
        or int(calibration_rows_during_training) < 0
    ):
        raise ValueError("invalid training family/calibration contract")
    checkpoint_steps = tuple(int(value) for value in checkpoint_steps)
    if (
        tuple(sorted(set(checkpoint_steps))) != checkpoint_steps
        or any(
            value < 1 or value > int(steps)
            for value in checkpoint_steps
        )
        or bool(checkpoint_steps) != (checkpoint_callback is not None)
    ):
        raise ValueError("invalid checkpoint-selection contract")
    pair_batch_size = int(micro_batch_size) // 2
    uses_natural = "natural" in family_pattern
    natural_view = (
        load_fixed_view(natural_view_path) if uses_natural else None
    )
    if (
        natural_view is not None
        and natural_view.input_ids.shape[1] != LENGTH
    ):
        raise RuntimeError("natural replay violates the 4K contract")
    natural_rows = (
        None
        if natural_view is None
        else torch.from_numpy(natural_view.training_rows.copy())
    )
    parameters = [
        parameter
        for _, parameter in trainable_named_parameters(model, None)
    ]
    if not parameters:
        raise RuntimeError("counterfactual routing has no parameters")
    model.gradient_checkpointing_disable()
    backbone = torch.compile(
        TrainingBackbone(model.model),
        fullgraph=True,
        dynamic=False,
        mode=compile_mode,
    )
    natural_loss_module = fused_loss_module() if uses_natural else None
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 71_001)
    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = 0
    processed_tokens = 0
    family_steps = {"routing": 0, "natural": 0}
    supervised_tokens = {"routing": 0, "natural": 0}
    recent_losses: list[float] = []
    position_hash = hashlib.sha256()
    exposure_hash = hashlib.sha256()
    position_bucket_counts = {
        "contiguous": 0,
        "transition": 0,
        "middle": 0,
        "far": 0,
    }
    virtual_gap_near_min: int | None = None
    virtual_gap_far_max: int | None = None
    maximum_observed_position_id = LENGTH - 2
    query_offset_stream: np.ndarray | None = None
    query_offset_stream_sha256: str | None = None
    query_offset_cursor = 0
    checkpoint_history: list[dict[str, Any]] = []
    selected_step: int | None = None
    actual_steps = 0
    if int(virtual_target_length):
        if (
            int(virtual_target_length) != 4 * LENGTH
            or tuple(int(value) for value in virtual_bucket_weights)
            != (1, 1, 2)
            or pair_batch_size != 2
            or int(gradient_accumulation_steps) != 2
        ):
            raise RuntimeError("locked virtual query-gap contract drift")
        routing_steps = sum(
            family_pattern[(step - 1) % len(family_pattern)]
            == "routing"
            for step in range(1, int(steps) + 1)
        )
        query_offset_stream = deterministic_query_offset_stream(
            seed=int(seed),
            routing_steps=routing_steps,
        )
        query_offset_stream_sha256 = hashlib.sha256(
            query_offset_stream.tobytes(order="C")
        ).hexdigest()
    model.train()
    torch.cuda.reset_peak_memory_stats()

    for step in range(1, int(steps) + 1):
        family = family_pattern[(step - 1) % len(family_pattern)]
        family_steps[family] += 1
        lr = cosine_lr(
            step, int(steps), int(warmup_steps), float(learning_rate)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        raw_losses = []
        metric_rows: list[dict[str, float]] = []
        for accumulation_index in range(
            int(gradient_accumulation_steps)
        ):
            if family == "routing":
                indices = torch.randint(
                    len(routing_view.input_ids),
                    (pair_batch_size,),
                    generator=generator,
                ).numpy()
                (
                    contexts,
                    labels,
                    alternate_labels,
                    supervised,
                ) = routing_batch(
                    view=routing_view, row_indices=indices
                )
                position_ids = None
                if int(virtual_target_length):
                    if query_offset_stream is None:
                        raise RuntimeError(
                            "virtual query-offset stream is missing"
                        )
                    stop = query_offset_cursor + pair_batch_size
                    query_offsets = np.asarray(
                        query_offset_stream[
                            query_offset_cursor:stop
                        ],
                        dtype=np.int64,
                    )
                    if len(query_offsets) != pair_batch_size:
                        raise RuntimeError(
                            "virtual query-offset stream exhausted early"
                        )
                    query_offset_cursor = stop
                    position_ids, exposures, payload = (
                        query_gap_position_ids(
                            view=routing_view,
                            row_indices=indices,
                            query_offsets=query_offsets,
                        )
                    )
                    position_hash.update(payload)
                    for pair_slot, exposure in enumerate(exposures):
                        bucket = query_offset_band(
                            int(exposure["query_offset"])
                        )
                        position_bucket_counts[bucket] += 1
                        near = int(exposure["virtual_gap_near"])
                        far = int(exposure["virtual_gap_far"])
                        virtual_gap_near_min = (
                            near
                            if virtual_gap_near_min is None
                            else min(virtual_gap_near_min, near)
                        )
                        virtual_gap_far_max = (
                            far
                            if virtual_gap_far_max is None
                            else max(virtual_gap_far_max, far)
                        )
                        maximum_observed_position_id = max(
                            maximum_observed_position_id,
                            int(exposure["maximum_active_position_id"]),
                        )
                        exposure_hash.update(
                            np.asarray(
                                [
                                    step,
                                    accumulation_index,
                                    pair_slot,
                                    int(indices[pair_slot]),
                                    int(exposure["query_offset"]),
                                    int(exposure["source_start"]),
                                    int(exposure["source_stop"]),
                                    int(exposure["query_start"]),
                                    int(
                                        exposure[
                                            "answer_prediction_index"
                                        ]
                                    ),
                                    near,
                                    far,
                                ],
                                dtype="<i8",
                            ).tobytes(order="C")
                        )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = backbone(contexts, position_ids)
                    raw_loss, metrics = routing_objective(
                        model=model,
                        hidden=hidden,
                        labels=labels,
                        alternate_labels=alternate_labels,
                        eos_token_id=routing_view.eos_token_id,
                        margin=float(margin),
                        margin_weight=float(margin_weight),
                        termination_weight=float(termination_weight),
                    )
                metric_rows.append(metrics)
            else:
                if (
                    natural_view is None
                    or natural_rows is None
                    or natural_loss_module is None
                ):
                    raise RuntimeError(
                        "natural family selected without a natural view"
                    )
                indices = natural_rows[
                    torch.randint(
                        len(natural_rows),
                        (int(micro_batch_size),),
                        generator=generator,
                    )
                ].numpy()
                contexts, labels, supervised = natural_batch(
                    view=natural_view,
                    indices=indices,
                    objective="full",
                )
                alternate_labels = None
                position_ids = None
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = backbone(contexts, None)
                    raw_loss = natural_loss_module(
                        model.lm_head.weight,
                        hidden.reshape(-1, hidden.shape[-1]),
                        labels.reshape(-1),
                    )
                    if hasattr(raw_loss, "loss"):
                        raw_loss = raw_loss.loss
            loss = raw_loss / float(gradient_accumulation_steps)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite routing loss at step {step}"
                )
            loss.backward()
            raw_losses.append(float(raw_loss.detach()))
            supervised_tokens[family] += int(supervised)
            processed_tokens += int(contexts.numel())
            del contexts, labels, hidden, raw_loss, loss
            if alternate_labels is not None:
                del alternate_labels
            if position_ids is not None:
                del position_ids
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        actual_steps = step
        mean_loss = float(np.mean(raw_losses))
        recent_losses.append(mean_loss)

        checkpoint_outcome = None
        if step in checkpoint_steps:
            if checkpoint_callback is None:
                raise RuntimeError("checkpoint callback disappeared")
            checkpoint_outcome = dict(checkpoint_callback(step, model))
            if (
                int(checkpoint_outcome.get("step", -1)) != step
                or not isinstance(
                    checkpoint_outcome.get("passed"), bool
                )
            ):
                raise RuntimeError("checkpoint callback contract drift")
            checkpoint_history.append(checkpoint_outcome)
            model.train()
            if checkpoint_outcome["passed"]:
                selected_step = step

        if (
            step == 1
            or step % 25 == 0
            or step == int(steps)
            or step in checkpoint_steps
        ):
            torch.cuda.synchronize()
            now = time.perf_counter()
            row: dict[str, Any] = {
                "step": step,
                "family": family,
                "loss": mean_loss,
                "mean_loss_last_25": float(
                    np.mean(recent_losses[-25:])
                ),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "processed_input_tokens": processed_tokens,
                "supervised_tokens": dict(supervised_tokens),
                "family_steps": dict(family_steps),
                "elapsed_seconds": now - started,
                "interval_tokens_per_second": (
                    (processed_tokens - last_log_tokens)
                    / max(now - last_log_time, 1e-9)
                ),
                "peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
                "peak_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved()
                ),
            }
            if metric_rows:
                for name in metric_rows[0]:
                    row[name] = float(
                        np.mean(
                            [metrics[name] for metrics in metric_rows]
                        )
                    )
            if checkpoint_outcome is not None:
                row["checkpoint_selection"] = checkpoint_outcome
            if (
                int(calibration_rows_during_training) > 0
                and step in {100, 200, int(steps)}
            ):
                row["routing_calibration"] = evaluate_routing(
                    model=model,
                    view=calibration_view,
                    rows=int(calibration_rows_during_training),
                    pair_batch_size=2,
                    margin=float(margin),
                )
                model.train()
            append_jsonl(log_path, row)
            last_log_time = now
            last_log_tokens = processed_tokens
        if selected_step == step:
            break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if query_offset_stream is not None:
        expected_cursor = (
            int(family_steps["routing"])
            * pair_batch_size
            * int(gradient_accumulation_steps)
        )
        if query_offset_cursor != expected_cursor:
            raise RuntimeError(
                "virtual query-offset stream prefix consumption drift"
            )
        realized_total = sum(position_bucket_counts.values())
        if realized_total != query_offset_cursor:
            raise RuntimeError("virtual query-offset count drift")
    consumed_query_offset_prefix_sha256 = (
        None
        if query_offset_stream is None
        else hashlib.sha256(
            query_offset_stream[:query_offset_cursor].tobytes(order="C")
        ).hexdigest()
    )
    return {
        "steps": int(actual_steps),
        "maximum_steps": int(steps),
        "actual_steps": int(actual_steps),
        "selected_step": selected_step,
        "stopped_early": bool(
            selected_step is not None and selected_step < int(steps)
        ),
        "checkpoint_steps": list(checkpoint_steps),
        "checkpoint_history": checkpoint_history,
        "checkpoint_selection_required": bool(checkpoint_steps),
        "checkpoint_selection_passed": (
            selected_step is not None if checkpoint_steps else None
        ),
        "family_pattern": list(family_pattern),
        "calibration_rows_during_training": int(
            calibration_rows_during_training
        ),
        "family_steps": family_steps,
        "supervised_tokens": supervised_tokens,
        "processed_input_tokens": processed_tokens,
        "micro_batch_size": int(micro_batch_size),
        "gradient_accumulation_steps": int(
            gradient_accumulation_steps
        ),
        "global_batch_size": int(
            micro_batch_size * gradient_accumulation_steps
        ),
        "learning_rate": float(learning_rate),
        "warmup_steps": int(warmup_steps),
        "termination_weight": float(termination_weight),
        "counterfactual_margin": float(margin),
        "counterfactual_margin_weight": float(margin_weight),
        "compile_mode": compile_mode,
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "elapsed_seconds": elapsed,
        "tokens_per_second": processed_tokens / elapsed,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "peak_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved()
        ),
        "precision": "bf16_autocast",
        "optimizer": "fused_adamw",
        "natural_loss_backend": (
            "liger_fused_linear_cross_entropy"
            if uses_natural
            else "not_used"
        ),
        "position_policy": (
            "contiguous"
            if not int(virtual_target_length)
            else "semantic_query_block_continuous_gap"
        ),
        "virtual_target_length": int(virtual_target_length),
        "virtual_bucket_weights": list(virtual_bucket_weights),
        "position_bucket_counts": position_bucket_counts,
        "virtual_gap_near_min": virtual_gap_near_min,
        "virtual_gap_far_max": virtual_gap_far_max,
        "maximum_observed_position_id": int(
            maximum_observed_position_id
        ),
        "query_offset_stream_sha256": query_offset_stream_sha256,
        "consumed_query_offset_prefix_sha256": (
            consumed_query_offset_prefix_sha256
        ),
        "consumed_query_offset_values": int(query_offset_cursor),
        "realized_position_stream_sha256": (
            None
            if query_offset_stream is None
            else position_hash.hexdigest()
        ),
        "realized_exposure_stream_sha256": (
            None
            if query_offset_stream is None
            else exposure_hash.hexdigest()
        ),
    }


def registered_protocol(
    args: argparse.Namespace,
    *,
    virtual_query_gap: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "frequency": args.frequency,
        "steps": int(args.steps),
        "hard_maximum_training_length": LENGTH,
        "maximum_physical_training_sequence_length": LENGTH,
        "maximum_physical_token_index": LENGTH - 1,
        "family_pattern": list(FAMILY_PATTERN),
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "global_batch_size": int(
            args.micro_batch_size
            * args.gradient_accumulation_steps
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "counterfactual_margin": float(
            args.counterfactual_margin
        ),
        "counterfactual_margin_weight": float(
            args.counterfactual_margin_weight
        ),
        "termination_weight": float(args.termination_weight),
        "compile_mode": args.compile_mode,
        "natural_eval_rows": int(args.natural_eval_rows),
        "natural_retention_lengths": [LENGTH],
        "natural_retention_tail_tokens": 1_024,
        "natural_retention_baseline": (
            "same_loaded_parent_before_any_repair_step"
        ),
        "seed": int(args.seed),
    }
    if virtual_query_gap:
        routing_steps = sum(
            FAMILY_PATTERN[(step - 1) % len(FAMILY_PATTERN)]
            == "routing"
            for step in range(1, int(args.steps) + 1)
        )
        query_offset_stream = deterministic_query_offset_stream(
            seed=int(args.seed),
            routing_steps=routing_steps,
        )
        result.update(
            {
                "position_policy": (
                    "semantic_query_block_continuous_gap"
                ),
                "virtual_target_length": int(
                    args.virtual_target_length
                ),
                "virtual_bucket_weights": [
                    int(value)
                    for value in args.virtual_bucket_weights
                ],
                "maximum_allowed_position_id": 4 * LENGTH - 1,
                "maximum_realized_position_id": 4 * LENGTH - 1,
                "routing_data_format_version": 2,
                "supervision_contract": SUPERVISION_CONTRACT,
                "supervision": (
                    "answer_ce_plus_weighted_immediate_eos_ce"
                ),
                "eos_token_id": OLMO2_EOS_TOKEN_ID,
                "final_eos_supervised": True,
                "counterfactual_margin_scope": (
                    "answer_tokens_where_gold_differs"
                ),
                "routing_optimizer_steps": routing_steps,
                "routing_pair_exposures": int(
                    len(query_offset_stream)
                ),
                "query_offset_stream_sha256": hashlib.sha256(
                    query_offset_stream.tobytes(order="C")
                ).hexdigest(),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--routing-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--experiment-ready-receipt", type=Path)
    parser.add_argument("--parent-exact-baseline-result", type=Path)
    parser.add_argument("--parent-exact-baseline-examples", type=Path)
    parser.add_argument("--parent-exact-baseline-run-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frequency", choices=("native", "evq"), required=True
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=2
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--counterfactual-margin", type=float, default=1.0)
    parser.add_argument(
        "--counterfactual-margin-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--termination-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on immediate terminal-EOS CE, separate from answer CE."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--natural-eval-rows", type=int, default=16)
    parser.add_argument("--virtual-target-length", type=int, default=0)
    parser.add_argument(
        "--virtual-bucket-weights",
        type=int,
        nargs=3,
        default=(1, 1, 2),
        metavar=("W4K", "W8K", "W16K"),
    )
    parser.add_argument("--seed", type=int, default=20_260_725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.steps) <= 0:
        raise ValueError("steps must be positive")
    if float(args.counterfactual_margin) <= 0:
        raise ValueError("counterfactual margin must be positive")
    if float(args.counterfactual_margin_weight) <= 0:
        raise ValueError("counterfactual margin weight must be positive")
    if float(args.termination_weight) <= 0:
        raise ValueError("termination weight must be positive")
    if int(args.virtual_target_length) not in {0, 4 * LENGTH}:
        raise ValueError("virtual target length must be 0 or 16384")
    if (
        len(args.virtual_bucket_weights) != 3
        or any(int(value) < 0 for value in args.virtual_bucket_weights)
        or sum(int(value) for value in args.virtual_bucket_weights) <= 0
    ):
        raise ValueError("invalid virtual bucket weights")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    seed_everything(int(args.seed))
    checkpoint = args.checkpoint.resolve()
    ready_receipt = args.ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, ready_receipt
    )
    routing_root = args.routing_data.resolve()
    routing_manifest = json.loads(
        (routing_root / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        routing_manifest.get("status") not in PAIR_COLLECTION_STATUSES
        or int(routing_manifest["hard_maximum_training_length"])
        != LENGTH
        or int(routing_manifest["hard_maximum_training_position_id"])
        != LENGTH - 1
        or (
            routing_manifest.get("status") == ROOT_STATUS
            and (
                int(routing_manifest.get("format_version", -1)) != 2
                or routing_manifest.get("supervision_contract")
                != SUPERVISION_CONTRACT
                or int(routing_manifest.get("eos_token_id", -1))
                != OLMO2_EOS_TOKEN_ID
                or routing_manifest.get("final_eos_supervised") is not True
                or routing_manifest.get(
                    "labels_only_cover_answer_and_final_eos"
                )
                is not True
                or routing_manifest.get(
                    "answer_string_tokenizer_roundtrip_exact"
                )
                is not True
            )
        )
    ):
        raise RuntimeError("routing collection violates the 4K contract")
    experiment_ready = None
    experiment_ready_path = None
    natural_multiquery = (
        routing_manifest.get("status")
        == "OLMO2_4K_COUNTERFACTUAL_NATURAL_MULTIQUERY_DATA_PREPARED"
    )
    virtual_query_gap = bool(int(args.virtual_target_length))
    parent_exact_baseline_receipt = None
    if natural_multiquery or virtual_query_gap:
        if args.experiment_ready_receipt is None:
            raise RuntimeError(
                "this experiment requires a matching READY receipt"
            )
        experiment_ready_path = args.experiment_ready_receipt.resolve()
        experiment_ready = json.loads(
            experiment_ready_path.read_text(encoding="utf-8")
        )
        expected_ready_status = (
            VIRTUAL_QUERY_GAP_READY_STATUS
            if virtual_query_gap
            else NATURAL_MULTIQUERY_READY_STATUS
        )
        if experiment_ready.get("status") != expected_ready_status:
            raise RuntimeError("experiment READY status drift")
        if Path(experiment_ready["run_output"]).resolve() != output:
            raise RuntimeError("experiment READY output path drift")
        if (
            experiment_ready["trainer"]["sha256"]
            != sha256_file(Path(__file__).resolve())
        ):
            raise RuntimeError("experiment READY trainer hash drift")
        if (
            experiment_ready["inputs"]["routing_data"]["manifest"]["sha256"]
            != sha256_file(routing_root / "manifest.json")
        ):
            raise RuntimeError("experiment READY routing-data hash drift")
        if virtual_query_gap and (
            experiment_ready["inputs"]["conversion"]["sha256"]
            != sha256_file(
                Path(__file__).resolve().parents[1]
                / "olmo2_lora_conversion.py"
            )
        ):
            raise RuntimeError("experiment READY conversion hash drift")
        if (
            experiment_ready.get("bound_code_sha256")
            != bound_code_sha256()
        ):
            raise RuntimeError("experiment READY bound-code hash drift")
        if (
            experiment_ready["inputs"]["checkpoint"][
                "composite_sha256"
            ]
            != checkpoint_digest
            or experiment_ready["inputs"]["checkpoint"][
                "ready_receipt"
            ]["sha256"]
            != sha256_file(ready_receipt)
        ):
            raise RuntimeError("experiment READY checkpoint input drift")
        natural_path = (
            args.prepared_data.resolve() / "longalign_paired_L4096"
        )
        natural_ready = experiment_ready["inputs"]["natural_replay"]
        if Path(natural_ready["path"]).resolve() != natural_path:
            raise RuntimeError("experiment READY natural-replay path drift")
        for filename, entry in natural_ready["files"].items():
            if sha256_file(natural_path / filename) != entry["sha256"]:
                raise RuntimeError(
                    f"experiment READY natural-replay drift: {filename}"
                )
        background_path = args.background_dir.resolve()
        background_ready = experiment_ready["inputs"]["background"]
        if Path(background_ready["path"]).resolve() != background_path:
            raise RuntimeError("experiment READY background path drift")
        for filename, entry in background_ready["files"].items():
            if sha256_file(background_path / filename) != entry["sha256"]:
                raise RuntimeError(
                    f"experiment READY background drift: {filename}"
                )
        baseline_paths = (
            args.parent_exact_baseline_result,
            args.parent_exact_baseline_examples,
            args.parent_exact_baseline_run_manifest,
        )
        if virtual_query_gap and any(
            path is None for path in baseline_paths
        ):
            raise RuntimeError(
                "query-gap EOS repair requires the registered parent "
                "full-string exact baseline before training"
            )
        if virtual_query_gap:
            baseline_result_path = (
                args.parent_exact_baseline_result.resolve()
            )
            baseline_examples_path = (
                args.parent_exact_baseline_examples.resolve()
            )
            baseline_manifest_path = (
                args.parent_exact_baseline_run_manifest.resolve()
            )
            baseline = json.loads(
                baseline_result_path.read_text(encoding="utf-8")
            )
            baseline_manifest = json.loads(
                baseline_manifest_path.read_text(encoding="utf-8")
            )
            ready_sha = sha256_file(experiment_ready_path)
            parent_sha = sha256_file(args.parent_adapter.resolve())
            if (
                baseline.get("status")
                != "OLMO2_INSTRUCT_RULER_EXACT_SCREEN_COMPLETE_V2"
                or baseline.get("experiment_ready_receipt_sha256")
                != ready_sha
                or baseline.get("adapter", {}).get("sha256")
                != parent_sha
                or baseline.get("run_manifest_sha256")
                != sha256_file(baseline_manifest_path)
                or baseline.get("results", {}).get("examples_sha256")
                != sha256_file(baseline_examples_path)
                or baseline_manifest.get("experiment_role")
                != "parent_exact_baseline"
                or baseline_manifest.get(
                    "experiment_ready_receipt_sha256"
                )
                != ready_sha
                or baseline_manifest.get("adapter_sha256") != parent_sha
            ):
                raise RuntimeError(
                    "registered parent exact baseline receipt drift"
                )
            parent_exact_baseline_receipt = {
                "result_sha256": sha256_file(baseline_result_path),
                "examples_sha256": sha256_file(
                    baseline_examples_path
                ),
                "run_manifest_sha256": sha256_file(
                    baseline_manifest_path
                ),
                "adapter_sha256": parent_sha,
            }
    if (
        sha256_file(checkpoint / "tokenizer.json")
        != routing_manifest["tokenizer_sha256"]
    ):
        raise RuntimeError("routing collection tokenizer drift")
    for name in ("train", "calibration"):
        entry = routing_manifest["sets"][name]
        if (
            sha256_file(routing_root / name / "manifest.json")
            != entry["manifest_sha256"]
        ):
            raise RuntimeError(
                f"routing collection manifest drift: {name}"
            )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if (
        tokenizer.eos_token_id != OLMO2_EOS_TOKEN_ID
        or (
            routing_manifest.get("status") == ROOT_STATUS
            and int(routing_manifest["eos_token_id"])
            != int(tokenizer.eos_token_id)
        )
    ):
        raise RuntimeError("routing/tokenizer EOS contract drift")
    require_virtual_geometry = bool(int(args.virtual_target_length))
    routing_view = RoutingPairView(
        routing_root / "train",
        require_virtual_geometry=require_virtual_geometry,
        query_marker_token_ids=tuple(
            int(value)
            for value in tokenizer(
                TRAIN_QUERY_MARKER,
                add_special_tokens=False,
            ).input_ids
        ),
    )
    calibration_view = RoutingPairView(
        routing_root / "calibration",
        require_virtual_geometry=require_virtual_geometry,
        query_marker_token_ids=tuple(
            int(value)
            for value in tokenizer(
                CALIBRATION_QUERY_MARKER,
                add_special_tokens=False,
            ).input_ids
        ),
    )

    model = load_model(checkpoint)
    frequency = apply_frequency(model, args.frequency)
    readout = install_adaptation(
        model,
        "qkvo_answer",
        rank=int(args.rank),
        alpha=float(args.alpha),
    )
    if readout is not None:
        raise RuntimeError("counterfactual routing forbids a readout")
    parent_adapter = args.parent_adapter.resolve()
    if (
        experiment_ready is not None
        and experiment_ready["inputs"]["parent_adapter"]["sha256"]
        != sha256_file(parent_adapter)
    ):
        raise RuntimeError("experiment READY parent-adapter hash drift")
    parent_metadata = load_adapter(parent_adapter, model, None)
    expected_parent = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": args.frequency,
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": "qkvo_answer",
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
    }
    for name, expected in expected_parent.items():
        if parent_metadata.get(name) != expected:
            raise RuntimeError(
                f"parent adapter metadata drift for {name}"
            )
    run_protocol = registered_protocol(
        args,
        virtual_query_gap=virtual_query_gap,
    )
    if (
        experiment_ready is not None
        and experiment_ready["protocol"] != run_protocol
    ):
        raise RuntimeError("experiment READY protocol drift")
    output.mkdir(parents=True)
    runtime = configure_cuda()
    model.to("cuda")
    initial_natural_nll = evaluate_natural_nll(
        model=model,
        background_dir=args.background_dir.resolve(),
        lengths=(LENGTH,),
        rows=int(args.natural_eval_rows),
        tail_tokens=1_024,
    )
    initial_calibration = evaluate_routing(
        model=model,
        view=calibration_view,
        rows=16,
        pair_batch_size=2,
        margin=float(args.counterfactual_margin),
    )
    initial_virtual_calibration = None
    if int(args.virtual_target_length):
        initial_virtual_calibration = {
            f"query_offset_{offset}": evaluate_routing(
                model=model,
                view=calibration_view,
                rows=16,
                pair_batch_size=2,
                margin=float(args.counterfactual_margin),
                query_offset=offset,
            )
            for offset in (LENGTH, 2 * LENGTH, 3 * LENGTH + 1)
        }
    training = train(
        model=model,
        routing_view=routing_view,
        calibration_view=calibration_view,
        natural_view_path=(
            args.prepared_data.resolve() / "longalign_paired_L4096"
        ),
        steps=int(args.steps),
        micro_batch_size=int(args.micro_batch_size),
        gradient_accumulation_steps=int(
            args.gradient_accumulation_steps
        ),
        learning_rate=float(args.learning_rate),
        warmup_steps=int(args.warmup_steps),
        margin=float(args.counterfactual_margin),
        margin_weight=float(args.counterfactual_margin_weight),
        termination_weight=float(args.termination_weight),
        compile_mode=args.compile_mode,
        seed=int(args.seed),
        log_path=output / "train_log.jsonl",
        virtual_target_length=int(args.virtual_target_length),
        virtual_bucket_weights=tuple(
            int(value) for value in args.virtual_bucket_weights
        ),
    )
    final_calibration = evaluate_routing(
        model=model,
        view=calibration_view,
        rows=16,
        pair_batch_size=2,
        margin=float(args.counterfactual_margin),
    )
    final_virtual_calibration = None
    if int(args.virtual_target_length):
        final_virtual_calibration = {
            f"query_offset_{offset}": evaluate_routing(
                model=model,
                view=calibration_view,
                rows=16,
                pair_batch_size=2,
                margin=float(args.counterfactual_margin),
                query_offset=offset,
            )
            for offset in (LENGTH, 2 * LENGTH, 3 * LENGTH + 1)
        }
    final_natural_nll = evaluate_natural_nll(
        model=model,
        background_dir=args.background_dir.resolve(),
        lengths=(LENGTH,),
        rows=int(args.natural_eval_rows),
        tail_tokens=1_024,
    )
    natural_nll = {
        "baseline": {
            "adapter_sha256": sha256_file(parent_adapter),
            "timing": "before_any_repair_optimizer_step",
            "cells": initial_natural_nll,
        },
        "candidate": {
            "timing": "after_final_repair_optimizer_step",
            "cells": final_natural_nll,
        },
        "delta_candidate_minus_baseline": {
            "L4096_mean_nll": (
                float(final_natural_nll["L4096"]["mean_nll"])
                - float(initial_natural_nll["L4096"]["mean_nll"])
            ),
            "L4096_tail_mean_nll": (
                float(final_natural_nll["L4096"]["tail_mean_nll"])
                - float(
                    initial_natural_nll["L4096"]["tail_mean_nll"]
                )
            ),
        },
        "capability_endpoint": False,
    }
    adapter_metadata = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": args.frequency,
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": "qkvo_answer",
        "adaptation_description": (
            f"qkvo_r{int(args.rank)}_alpha{float(args.alpha):g}"
        ),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "training_sequence_length": LENGTH,
        "stage": (
            "counterfactual_routing_semantic_query_gap_16k_eos_v2"
            if int(args.virtual_target_length)
            else "counterfactual_source_routing_4k_eos_v2"
        ),
        "parent_adapter_sha256": sha256_file(parent_adapter),
        "routing_data_sha256": sha256_file(
            routing_root / "manifest.json"
        ),
        "seed": int(args.seed),
        "position_policy": (
            "contiguous"
            if not int(args.virtual_target_length)
            else "semantic_query_block_continuous_gap"
        ),
        "virtual_target_length": int(args.virtual_target_length),
        "supervision_contract": SUPERVISION_CONTRACT,
        "eos_token_id": OLMO2_EOS_TOKEN_ID,
        "final_eos_supervised": True,
        "termination_weight": float(args.termination_weight),
    }
    adapter_sha = save_adapter(
        output / "adapter.pt", model, None, adapter_metadata
    )
    receipt = {
        "status": (
            "OLMO2_4K_QUERY_GAP_EOS_REPAIR_COMPLETE_V1"
            if int(args.virtual_target_length)
            else "OLMO2_4K_COUNTERFACTUAL_ROUTING_EOS_V2_COMPLETE"
        ),
        "metric_boundary": (
            "paired source-content swaps use physical 4K sequences and a "
            "gold-independent semantic query-block offset spanning "
            "positions through 16K; natural replay stays contiguous 4K. "
            "Virtual calibration is teacher-forced and is not a capability "
            "endpoint; strict capability requires separate real physical "
            "8K/16K autoregressive evaluation"
            if int(args.virtual_target_length)
            else "paired source-content swaps and natural replay use only "
            "positions 0..4095; 8K/16K remain evaluation-only"
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "bound_code_sha256": bound_code_sha256(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "ready_receipt_sha256": sha256_file(ready_receipt),
        "experiment_ready_receipt_sha256": (
            None
            if experiment_ready_path is None
            else sha256_file(experiment_ready_path)
        ),
        "frequency": frequency,
        "parent_adapter": str(parent_adapter),
        "parent_adapter_sha256": sha256_file(parent_adapter),
        "adapter_sha256": adapter_sha,
        "parent_exact_baseline": parent_exact_baseline_receipt,
        "routing_data": {
            "path": str(routing_root),
            "manifest_sha256": sha256_file(
                routing_root / "manifest.json"
            ),
            "format_version": int(routing_manifest["format_version"]),
            "status": routing_manifest["status"],
            "supervision_contract": routing_manifest.get(
                "supervision_contract"
            ),
            "eos_token_id": routing_manifest.get("eos_token_id"),
            "final_eos_supervised": True,
            "labels_only_cover_answer_and_final_eos": True,
            "answer_string_tokenizer_roundtrip_exact": (
                routing_manifest.get(
                    "answer_string_tokenizer_roundtrip_exact"
                )
            ),
        },
        "runtime": runtime,
        "initial_routing_calibration": initial_calibration,
        "initial_virtual_routing_calibration": (
            initial_virtual_calibration
        ),
        "training": training,
        "final_routing_calibration": final_calibration,
        "final_virtual_routing_calibration": (
            final_virtual_calibration
        ),
        "natural_nll": natural_nll,
        "protocol": run_protocol,
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output / "results.json"),
                "initial_routing_calibration": initial_calibration,
                "training": training,
                "final_routing_calibration": final_calibration,
                "natural_nll": natural_nll,
                "adapter_sha256": adapter_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
