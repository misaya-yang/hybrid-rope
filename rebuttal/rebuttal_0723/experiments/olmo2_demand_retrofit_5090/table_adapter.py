"""Protected/conditional frequency-table adapter.

The old protected proposal performed same-index Native/EVQ splicing.  This
module treats that splice as an unsafe candidate and fails closed when it
creates near-collisions.  A conditional candidate must be supplied by an
offline solver; this bundle does not silently invent one or make Theorem 3
vacuous.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

from .protocol import ContractError, ROTARY_PAIRS


class TableCollisionError(ContractError):
    """Raised when a protected/conditional table has a collision risk."""

    def __init__(self, message: str, report: "CollisionReport") -> None:
        super().__init__(message)
        self.report = report


class ConditionalCandidateRequired(ContractError):
    """Raised when only an unsafe same-index splice was supplied."""

    def __init__(self, message: str, report: "CollisionReport") -> None:
        super().__init__(message)
        self.report = report


def canonical_pair_indices(
    values: Iterable[int],
    *,
    pair_count: int = ROTARY_PAIRS,
    max_pairs: int = 16,
) -> tuple[int, ...]:
    pairs = tuple(sorted(int(value) for value in values))
    if len(set(pairs)) != len(pairs):
        raise ContractError("protected pair indices contain duplicates")
    if len(pairs) > int(max_pairs):
        raise ContractError(
            f"protected set has {len(pairs)} pairs; max is {max_pairs}"
        )
    if any(index < 0 or index >= int(pair_count) for index in pairs):
        raise ContractError("protected pair index is out of range")
    return pairs


def _validate_inv_freq(
    values: Sequence[float],
    *,
    name: str,
    require_decreasing: bool = True,
) -> tuple[float, ...]:
    table = tuple(float(value) for value in values)
    if len(table) != ROTARY_PAIRS:
        raise ContractError(
            f"{name} must contain {ROTARY_PAIRS} rotary pairs, got {len(table)}"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in table):
        raise ContractError(f"{name} must contain finite positive frequencies")
    if require_decreasing and any(
        left <= right for left, right in zip(table, table[1:])
    ):
        raise ContractError(f"{name} must be strictly decreasing inverse frequencies")
    return table


def _float32_digest(values: Sequence[float]) -> str:
    payload = b"".join(struct.pack("<f", float(value)) for value in values)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CollisionReport:
    pair_count: int
    minimum_log_gap: float
    threshold_log_gap: float
    collision_pairs: tuple[tuple[int, int], ...]
    strictly_decreasing: bool
    finite_positive: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite_positive
            and self.strictly_decreasing
            and not self.collision_pairs
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_count": self.pair_count,
            "minimum_log_gap": self.minimum_log_gap,
            "threshold_log_gap": self.threshold_log_gap,
            "collision_pairs": [list(pair) for pair in self.collision_pairs],
            "strictly_decreasing": self.strictly_decreasing,
            "finite_positive": self.finite_positive,
            "passed": self.passed,
        }


def collision_report(
    values: Sequence[float],
    *,
    native_reference: Sequence[float] | None = None,
    gap_fraction_of_native_spacing: float = 0.20,
) -> CollisionReport:
    """Check all pairwise log-frequency gaps, not just adjacent entries."""

    table = tuple(float(value) for value in values)
    finite_positive = all(math.isfinite(value) and value > 0.0 for value in table)
    strictly_decreasing = (
        len(table) > 1
        and all(left > right for left, right in zip(table, table[1:]))
    )
    if native_reference is not None:
        native = _validate_inv_freq(native_reference, name="native_reference")
        native_gaps = [
            abs(math.log(left) - math.log(right))
            for left, right in zip(native, native[1:])
        ]
        native_spacing = sorted(native_gaps)[len(native_gaps) // 2]
    else:
        native_spacing = 0.0
    if not math.isfinite(gap_fraction_of_native_spacing) or (
        gap_fraction_of_native_spacing < 0.0
    ):
        raise ContractError("collision gap fraction must be finite and non-negative")
    threshold = float(gap_fraction_of_native_spacing) * native_spacing
    if threshold <= 0.0:
        threshold = 1e-12
    gaps = (
        [
            (abs(math.log(table[left]) - math.log(table[right])), left, right)
            for left, right in itertools.combinations(range(len(table)), 2)
        ]
        if finite_positive
        else []
    )
    minimum = min((gap for gap, _, _ in gaps), default=float("inf"))
    collisions = tuple(
        (left, right)
        for gap, left, right in gaps
        if gap < threshold
    )
    return CollisionReport(
        pair_count=len(table),
        minimum_log_gap=minimum,
        threshold_log_gap=threshold,
        collision_pairs=collisions,
        strictly_decreasing=strictly_decreasing,
        finite_positive=finite_positive,
    )


@dataclass(frozen=True)
class ConditionalTableResult:
    table: tuple[float, ...]
    protected_pairs: tuple[int, ...]
    unprotected_pairs: tuple[int, ...]
    mode: str
    frequency_sha256_float32: str
    collision: CollisionReport

    def as_dict(self) -> dict[str, object]:
        return {
            "table": list(self.table),
            "protected_pairs": list(self.protected_pairs),
            "unprotected_pairs": list(self.unprotected_pairs),
            "mode": self.mode,
            "frequency_sha256_float32": self.frequency_sha256_float32,
            "collision": self.collision.as_dict(),
        }


class ConditionalTableAdapter:
    """Build a table only from an explicitly supplied conditional candidate."""

    def __init__(
        self,
        *,
        max_protected_pairs: int = 16,
        gap_fraction_of_native_spacing: float = 0.20,
    ) -> None:
        self.max_protected_pairs = int(max_protected_pairs)
        self.gap_fraction_of_native_spacing = float(
            gap_fraction_of_native_spacing
        )
        if not 0 <= self.max_protected_pairs <= ROTARY_PAIRS:
            raise ContractError("invalid protected-pair budget")
        if not math.isfinite(self.gap_fraction_of_native_spacing) or (
            self.gap_fraction_of_native_spacing <= 0.0
        ):
            raise ContractError("collision threshold fraction must be positive")

    def apply(
        self,
        *,
        native_inv_freq: Sequence[float],
        evq_inv_freq: Sequence[float],
        protected_pairs: Iterable[int],
        conditional_candidate: Sequence[float] | None = None,
    ) -> ConditionalTableResult:
        """Return a collision-free candidate or fail closed.

        The conditional_candidate must be a complete table produced offline.
        If it is omitted, same-index splicing is inspected but never accepted.
        """

        native = _validate_inv_freq(native_inv_freq, name="native_inv_freq")
        evq = _validate_inv_freq(evq_inv_freq, name="evq_inv_freq")
        protected = canonical_pair_indices(
            protected_pairs,
            max_pairs=self.max_protected_pairs,
        )
        protected_set = set(protected)
        unprotected = tuple(
            index for index in range(ROTARY_PAIRS) if index not in protected_set
        )
        if conditional_candidate is None:
            candidate = list(evq)
            for index in protected:
                candidate[index] = native[index]
            mode = "unsafe_same_index_splice_rejected"
        else:
            candidate = list(conditional_candidate)
            mode = "offline_conditional_candidate"
        table = _validate_inv_freq(
            candidate,
            name="conditional_candidate",
            require_decreasing=False,
        )
        for index in protected:
            if table[index] != native[index]:
                raise ContractError(
                    f"protected Native frequency changed at pair {index}"
                )
        collision = collision_report(
            table,
            native_reference=native,
            gap_fraction_of_native_spacing=self.gap_fraction_of_native_spacing,
        )
        if conditional_candidate is None:
            raise ConditionalCandidateRequired(
                "same-index Native/EVQ splice is diagnostic only; "
                "a complete offline conditional candidate is required",
                collision,
            )
        result = ConditionalTableResult(
            table=table,
            protected_pairs=protected,
            unprotected_pairs=unprotected,
            mode=mode,
            frequency_sha256_float32=_float32_digest(table),
            collision=collision,
        )
        if not collision.passed:
            raise TableCollisionError(
                "conditional table failed strict monotonicity/collision check",
                collision,
            )
        return result
