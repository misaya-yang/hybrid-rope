"""Pure contracts for the EVQ seed-42 retrieval-repair experiment."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


SEGMENT_STEPS = 32
LEARNING_RATE = 2e-5
WARMUP_STEPS = 4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


@dataclass(frozen=True)
class StageSpec:
    """One registered sequence-length and runtime-frequency stage."""

    name: str
    seq_len: int
    min_distance: int
    max_distance: int
    accumulation: int
    factor: float
    pair_threshold: float

    def __post_init__(self) -> None:
        if self.name not in {"r8", "r16"}:
            raise ValueError("stage name must be r8 or r16")
        if self.seq_len <= 0 or self.accumulation <= 0:
            raise ValueError("sequence length and accumulation must be positive")
        if not 0 < self.min_distance <= self.max_distance < self.seq_len:
            raise ValueError("distance bounds must fit inside the sequence")
        if self.factor <= 0:
            raise ValueError("runtime factor must be positive")
        if not 0.0 <= self.pair_threshold <= 1.0:
            raise ValueError("pair threshold must be a probability")

    @property
    def examples_per_segment(self) -> int:
        return SEGMENT_STEPS * self.accumulation

    @property
    def tokens_per_segment(self) -> int:
        return self.seq_len * self.examples_per_segment


_STAGES = {
    "r8": StageSpec(
        name="r8",
        seq_len=8192,
        min_distance=2048,
        max_distance=6144,
        accumulation=4,
        factor=1.0,
        pair_threshold=0.80,
    ),
    "r16": StageSpec(
        name="r16",
        seq_len=16384,
        min_distance=6144,
        max_distance=14336,
        accumulation=2,
        factor=2.0,
        pair_threshold=0.50,
    ),
}

_FACTORS_BY_LENGTH = {8192: 1.0, 16384: 2.0, 32768: 4.0}
_PASSKEY_PATTERN = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def get_stage(name: str) -> StageSpec:
    """Return one registered repair stage or fail closed."""
    try:
        return _STAGES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown stage {name!r}; expected r8 or r16") from exc


def registered_factor_for_length(context_length: int) -> float:
    """Return the only registered identity/YaRN factor for a context length."""
    try:
        return _FACTORS_BY_LENGTH[int(context_length)]
    except KeyError as exc:
        raise ValueError(
            f"unregistered context length {context_length!r}; expected 8192, 16384, or 32768"
        ) from exc


def segment_contract(stage: str, segment: int) -> dict[str, Any]:
    """Return the immutable optimizer and row budget for one segment."""
    spec = get_stage(stage)
    segment = int(segment)
    if segment not in (1, 2):
        raise ValueError("segment must be 1 or 2")
    row_start = (segment - 1) * spec.examples_per_segment
    row_stop = segment * spec.examples_per_segment
    return {
        "stage": spec.name,
        "segment": segment,
        "steps": SEGMENT_STEPS,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": spec.accumulation,
        "examples": spec.examples_per_segment,
        "row_range": [row_start, row_stop],
        "tokens": spec.tokens_per_segment,
        "optimizer_state": "fresh",
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "lr_scheduler": "cosine",
    }


def extract_first_passkey(text: str) -> str | None:
    """Extract the first standalone eight-digit passkey."""
    match = _PASSKEY_PATTERN.search(str(text))
    return None if match is None else match.group(1)


def _normalized_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def score_text_answer(
    prediction: str,
    gold: str,
    eos_terminated: bool,
) -> dict[str, object]:
    """Keep answer retrieval, full-output formatting, and stopping separate."""
    normalized = _normalized_text(prediction)
    expected = _normalized_text(gold)
    extracted = extract_first_passkey(normalized)
    bounded_gold = bool(
        re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", normalized)
    )
    return {
        "strict_exact": normalized == expected,
        "first_value_exact": extracted == expected,
        "gold_containment": bounded_gold,
        "extracted_value": extracted,
        "eos_terminated": bool(eos_terminated),
    }


def _finite_number(summary: Mapping[str, object], field: str) -> tuple[float, bool]:
    try:
        value = float(summary[field])
    except (KeyError, TypeError, ValueError):
        return math.nan, False
    return value, math.isfinite(value)


def decide_gate(
    stage: str,
    summary: Mapping[str, object],
    parent_summary: Mapping[str, object],
    *,
    segment: int,
) -> dict[str, object]:
    """Apply the pre-registered capability, retention, and rescue rules."""
    spec = get_stage(stage)
    if int(segment) not in (1, 2):
        raise ValueError("segment must be 1 or 2")

    pair, pair_finite = _finite_number(summary, "pair_consistency")
    removal, removal_finite = _finite_number(
        summary, "source_removal_positive_fraction"
    )
    passkey, passkey_finite = _finite_number(summary, "passkey_containment")
    temporal, temporal_finite = _finite_number(summary, "temporal_delta_nll")
    parent_pair, parent_finite = _finite_number(parent_summary, "pair_consistency")
    numeric_finite = all(
        (pair_finite, removal_finite, passkey_finite, temporal_finite, parent_finite)
    )

    task_types = summary.get("task_types")
    if isinstance(task_types, (str, bytes)):
        normalized_tasks: set[str] = set()
    else:
        try:
            normalized_tasks = {str(value) for value in task_types}  # type: ignore[arg-type]
        except TypeError:
            normalized_tasks = set()

    finite_check = bool(summary.get("finite")) and numeric_finite
    checks = {
        "pair_consistency": pair_finite and pair >= spec.pair_threshold,
        "source_removal_positive_fraction": removal_finite and removal >= 0.75,
        "passkey_containment": passkey_finite and passkey >= 0.50,
        "temporal_delta_nll": temporal_finite and temporal <= 0.20,
        "finite": finite_check,
        "task_types": normalized_tasks == {"kv", "update"},
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if not failed:
        status = "pass"
    elif (
        int(segment) == 1
        and checks["finite"]
        and checks["temporal_delta_nll"]
        and checks["task_types"]
        and pair - parent_pair >= 0.10
    ):
        status = "rescue_allowed"
    else:
        status = "stop"
    return {
        "stage": spec.name,
        "segment": int(segment),
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "thresholds": {
            "pair_consistency": spec.pair_threshold,
            "source_removal_positive_fraction": 0.75,
            "passkey_containment": 0.50,
            "temporal_delta_nll_max": 0.20,
            "rescue_pair_gain": 0.10,
        },
    }
