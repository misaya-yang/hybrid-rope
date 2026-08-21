"""Capability gate evaluators for the R4' preparation contract.

These functions consume already-produced metrics only.  They never load a
model, decode text, or infer capability from NLL/PPL alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .protocol import ContractError, GateContract


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    observed: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "observed": dict(self.observed),
        }


def _number(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"missing numeric gate metric: {key}")
    return float(value)


def evaluate_4k_gate(
    *,
    native: Mapping[str, Any],
    candidate: Mapping[str, Any],
    contract: GateContract | None = None,
) -> GateResult:
    """Evaluate the full 4K retention gate against freshly matched Native."""

    contract = contract or GateContract()
    contract.validate()
    reasons: list[str] = []
    native_f1 = _number(native, "two_wiki_token_f1")
    candidate_f1 = _number(candidate, "two_wiki_token_f1")
    f1_drop = native_f1 - candidate_f1
    if f1_drop > contract.two_wiki_token_f1_drop_max_points:
        reasons.append(
            f"2Wiki token-F1 drop {f1_drop:.4f} exceeds "
            f"{contract.two_wiki_token_f1_drop_max_points:.4f}"
        )

    native_ruler = _number(native, "ruler_macro")
    candidate_ruler = _number(candidate, "ruler_macro")
    ruler_drop = native_ruler - candidate_ruler
    if ruler_drop > contract.ruler_macro_drop_max_points:
        reasons.append(
            f"RULER macro drop {ruler_drop:.4f} exceeds "
            f"{contract.ruler_macro_drop_max_points:.4f}"
        )

    native_nll = _number(native, "natural_nll")
    candidate_nll = _number(candidate, "natural_nll")
    nll_delta = candidate_nll - native_nll
    if nll_delta > contract.natural_nll_delta_max:
        reasons.append(
            f"natural NLL delta {nll_delta:.4f} exceeds "
            f"{contract.natural_nll_delta_max:.4f}"
        )

    native_families = native.get("ruler_family_scores")
    candidate_families = candidate.get("ruler_family_scores")
    if not isinstance(native_families, Mapping) or not isinstance(
        candidate_families, Mapping
    ):
        reasons.append("family-level RULER scores are missing")
    else:
        for family, native_score in native_families.items():
            if not isinstance(native_score, (int, float)):
                reasons.append(f"Native family score is non-numeric: {family}")
                continue
            candidate_score = candidate_families.get(family)
            if not isinstance(candidate_score, (int, float)):
                reasons.append(f"candidate family score is missing: {family}")
                continue
            if float(native_score) > 0.0 and float(candidate_score) <= 0.0:
                reasons.append(
                    f"Native-positive RULER family collapsed to zero: {family}"
                )

    if candidate.get("independent_retention_pass") is not True:
        reasons.append("independent retention slice did not pass")
    if candidate.get("strict_autoregressive_evaluation") is not True:
        reasons.append("4K endpoint is not marked strict autoregressive")

    return GateResult(
        gate="4k_retention",
        passed=not reasons,
        reasons=tuple(reasons),
        observed={
            "two_wiki_f1_drop": f1_drop,
            "ruler_macro_drop": ruler_drop,
            "natural_nll_delta": nll_delta,
        },
    )


def evaluate_8k_gate(
    *,
    four_k: GateResult,
    native: Mapping[str, Any],
    candidate: Mapping[str, Any],
    contract: GateContract | None = None,
) -> GateResult:
    """Evaluate 8K only after the complete 4K gate passes."""

    contract = contract or GateContract()
    contract.validate()
    reasons: list[str] = []
    if contract.eight_k_requires_four_k_pass and not four_k.passed:
        reasons.append("8K gate is blocked because the 4K gate failed")
    if candidate.get("strict_autoregressive_evaluation") is not True:
        reasons.append("8K endpoint is not marked strict autoregressive")
    native_score = _number(native, "strict_autoregressive_score")
    candidate_score = _number(candidate, "strict_autoregressive_score")
    if candidate_score <= native_score:
        reasons.append(
            f"candidate strict AR score {candidate_score:.4f} does not exceed "
            f"Native {native_score:.4f}"
        )
    return GateResult(
        gate="8k_length_transfer",
        passed=not reasons,
        reasons=tuple(reasons),
        observed={
            "native_strict_autoregressive_score": native_score,
            "candidate_strict_autoregressive_score": candidate_score,
            "four_k_passed": four_k.passed,
        },
    )
