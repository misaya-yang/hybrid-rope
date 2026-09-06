#!/usr/bin/env python3
"""Frozen selection and confirmation logic for the success-first tournament.

Pure CPU functions implementing the lexicographic development/selection rule and
the final confirmation verdict from
``ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830`` §6.  No model, no data: the
functions consume already-computed endpoint summaries and bootstrap intervals, so
they can be unit-tested without GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


GUARD_MARGIN = 0.01
FARTAIL_TIE_EPS = 0.01
FEASIBILITY_MODES = ("ABSOLUTE", "ANCHORED")


def _anchored_reference(candidate: dict[str, Any]) -> tuple[float, float]:
    try:
        return (
            float(candidate["yarn_native_prefix_delta"]),
            float(candidate["yarn_long_dense_delta"]),
        )
    except KeyError as error:
        raise ValueError(
            "ANCHORED feasibility needs the official YaRN factor-four arm measured "
            f"on the same rows; candidate {candidate.get('name')!r} lacks {error.args[0]}"
        ) from error


def is_feasible(
    candidate: dict[str, Any],
    margin: float = GUARD_MARGIN,
    mode: str = "ABSOLUTE",
) -> bool:
    """A candidate is feasible when both no-harm guards hold.

    ``ABSOLUTE`` compares against Native with ``margin``.  ``ANCHORED`` additionally
    accepts a candidate whose Native-prefix and long-dense costs are no worse than
    the official YaRN factor-four costs measured on the same rows.
    """

    if mode not in FEASIBILITY_MODES:
        raise ValueError(f"unknown feasibility mode {mode!r}")
    native_prefix_delta = float(candidate["native_prefix_delta"])
    long_dense_delta = float(candidate["long_dense_delta"])
    if native_prefix_delta <= margin and long_dense_delta <= margin:
        return True
    if mode == "ABSOLUTE":
        return False
    yarn_prefix, yarn_long_dense = _anchored_reference(candidate)
    return native_prefix_delta <= yarn_prefix and long_dense_delta <= yarn_long_dense


def _tiebreak_key(candidate: dict[str, Any]) -> tuple:
    """Smaller is preferred, in the predeclared tie-break order."""

    return (
        int(candidate["calibrated_dof"]),
        0 if not candidate.get("used_long_range_in_construction") else 1,
        float(candidate["chord_displacement_rms"]),
        float(candidate["support_movement"]),
    )


def apply_selection_rule(
    candidates: Iterable[dict[str, Any]],
    margin: float = GUARD_MARGIN,
    mode: str = "ABSOLUTE",
) -> dict[str, Any]:
    """Choose one global winner by the frozen lexicographic rule.

    Returns the winning candidate together with the feasibility ledger.  Raises
    ``ValueError`` if there are no candidates at all; the caller maps an empty
    feasible set to the family/programme stop rule.
    """

    rows = [dict(candidate) for candidate in candidates]
    if not rows:
        raise ValueError("selection rule received no candidates")
    feasible = [row for row in rows if is_feasible(row, margin, mode)]
    ledger = {
        "total": len(rows),
        "feasible": len(feasible),
        "feasibility_mode": mode,
        "rejected": [
            {
                "name": row["name"],
                "family": row.get("family"),
                "native_prefix_delta": float(row["native_prefix_delta"]),
                "long_dense_delta": float(row["long_dense_delta"]),
            }
            for row in rows
            if not is_feasible(row, margin, mode)
        ],
    }
    if not feasible:
        return {"winner": None, "ledger": ledger}

    best_far_tail = min(float(row["far_tail_nll"]) for row in feasible)
    near_best = [
        row
        for row in feasible
        if float(row["far_tail_nll"]) <= best_far_tail + FARTAIL_TIE_EPS
    ]
    near_best.sort(key=_tiebreak_key)
    return {"winner": near_best[0], "ledger": ledger}


def confirmation_verdict(
    intervals: dict[str, dict[str, float]],
    margin: float = GUARD_MARGIN,
    positive_control_passed: bool = True,
) -> str:
    """Issue exactly one verdict from paired-document bootstrap intervals.

    ``intervals`` maps endpoint -> {"mean_delta", "ci_low", "ci_high"}, where a
    negative delta favours the candidate.  ``positive_control_passed`` is the
    registered Native/determinism control on the confirmation split.  The optional
    ``native_prefix_vs_yarn`` and ``long_dense_vs_yarn`` entries carry
    candidate-minus-YaRN differences on the same rows and enable the anchored tier.
    """

    if not positive_control_passed:
        return "UNRESOLVED"

    prefix = intervals["native_prefix"]
    far_tail = intervals["far_tail"]
    long_dense = intervals["long_dense"]

    prefix_improves = prefix["ci_high"] < 0.0
    far_tail_improves = far_tail["ci_high"] < 0.0
    long_dense_guarded = long_dense["ci_high"] <= margin
    prefix_guarded = prefix["ci_high"] <= margin

    if prefix_improves and far_tail_improves and long_dense_guarded:
        return "JOINT_IMPROVEMENT"
    if prefix_guarded and long_dense_guarded and far_tail_improves:
        return "DEPLOYABLE_PARETO"

    prefix_vs_yarn = intervals.get("native_prefix_vs_yarn")
    long_dense_vs_yarn = intervals.get("long_dense_vs_yarn")
    if (
        far_tail_improves
        and prefix_vs_yarn is not None
        and long_dense_vs_yarn is not None
        and prefix_vs_yarn["ci_high"] <= 0.0
        and long_dense_vs_yarn["ci_high"] <= 0.0
    ):
        return "YARN_ANCHORED_PARETO"

    if far_tail_improves:
        return "MECHANISM_ONLY"
    if far_tail["mean_delta"] < 0.0:
        return "UNRESOLVED"
    return "FAIL"


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("select", "confirm"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=GUARD_MARGIN)
    parser.add_argument(
        "--feasibility-mode", choices=FEASIBILITY_MODES, default="ABSOLUTE"
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if args.mode == "select":
        outcome = apply_selection_rule(
            payload["candidates"], margin=args.margin, mode=args.feasibility_mode
        )
        print(json.dumps(outcome, indent=2, sort_keys=True))
        return 0 if outcome["winner"] is not None else 2
    verdict = confirmation_verdict(
        payload["intervals"],
        margin=args.margin,
        positive_control_passed=bool(payload.get("positive_control_passed", True)),
    )
    print(json.dumps({"verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
