"""Offline Pareto and layer-profile screening for R3-prime results.

This script consumes a future result JSON.  It never runs a model and does not
attach statistical significance to a Pareto or profile classification.  Its
``bimodal`` label is a deterministic shape screen only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        value = payload.get("results", payload.get("arms", payload.get("rows", [])))
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, Mapping)]
    raise ValueError("result JSON must be a list or an object with results/arms/rows")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric(row: Mapping[str, Any], key: str) -> float | None:
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping) and key in metrics:
        return _number(metrics[key])
    return _number(row.get(key))


def _profile(row: Mapping[str, Any]) -> list[float] | None:
    for key in ("layer_tau", "per_layer_tau", "layer_m", "per_layer_m"):
        value = row.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = [_number(item) for item in value]
            if values and all(item is not None for item in values):
                return [float(item) for item in values if item is not None]
    return None


def classify_profile(values: Sequence[float], *, epsilon: float = 1e-8) -> dict[str, Any]:
    """Classify a profile as flat, monotonic, bimodal, or other."""

    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        return {"label": "invalid", "peaks": [], "n_layers": len(numbers)}
    span = max(numbers) - min(numbers)
    if span <= epsilon:
        return {"label": "flat", "peaks": [], "n_layers": len(numbers), "range": span}
    tol = max(epsilon, span * 0.05)
    differences = [right - left for left, right in zip(numbers, numbers[1:])]
    if all(value >= -tol for value in differences) or all(value <= tol for value in differences):
        return {"label": "monotonic", "peaks": [], "n_layers": len(numbers), "range": span}

    peaks: list[int] = []
    for index in range(1, len(numbers) - 1):
        left, center, right = numbers[index - 1 : index + 2]
        if center - left >= tol and center - right >= tol:
            peaks.append(index)
    # Require separated, non-edge peaks so a noisy shoulder is not called a
    # two-mode profile.  The threshold is descriptive, not a model-selection test.
    separated = [
        (left, right)
        for left in peaks
        for right in peaks
        if right > left and right - left >= max(2, len(numbers) // 5)
    ]
    label = "bimodal" if separated else "other"
    return {
        "label": label,
        "peaks": peaks,
        "separated_peak_pairs": separated,
        "n_layers": len(numbers),
        "range": span,
    }


def pareto_frontier(
    rows: Sequence[Mapping[str, Any]],
    *,
    in_window_key: str = "in_window",
    ood_key: str = "ood",
    in_window_direction: str = "min",
    ood_direction: str = "min",
) -> list[int]:
    """Return row indices not dominated on two metrics."""

    if in_window_direction not in {"min", "max"} or ood_direction not in {"min", "max"}:
        raise ValueError("metric directions must be min or max")
    values: list[tuple[int, float, float]] = []
    for index, row in enumerate(rows):
        first = _metric(row, in_window_key)
        second = _metric(row, ood_key)
        if first is not None and second is not None:
            values.append((index, first, second))

    def no_worse(a: float, b: float, direction: str) -> bool:
        return a <= b if direction == "min" else a >= b

    def strictly_better(a: float, b: float, direction: str) -> bool:
        return a < b if direction == "min" else a > b

    frontier: list[int] = []
    for index, first, second in values:
        dominated = False
        for other_index, other_first, other_second in values:
            if other_index == index:
                continue
            if (
                no_worse(other_first, first, in_window_direction)
                and no_worse(other_second, second, ood_direction)
                and (
                    strictly_better(other_first, first, in_window_direction)
                    or strictly_better(other_second, second, ood_direction)
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(index)
    return frontier


def analyze_payload(
    payload: Any,
    *,
    in_window_key: str = "in_window",
    ood_key: str = "ood",
    in_window_direction: str = "min",
    ood_direction: str = "min",
) -> dict[str, Any]:
    rows = _rows(payload)
    profiles: list[dict[str, Any]] = []
    metric_rows = 0
    for index, row in enumerate(rows):
        profile = _profile(row)
        profile_result = classify_profile(profile) if profile is not None else {
            "label": "missing",
            "peaks": [],
            "n_layers": 0,
        }
        in_value = _metric(row, in_window_key)
        ood_value = _metric(row, ood_key)
        if in_value is not None and ood_value is not None:
            metric_rows += 1
        profiles.append(
            {
                "row_index": index,
                "arm_id": row.get("arm_id", row.get("name", f"row_{index}")),
                "in_window": in_value,
                "ood": ood_value,
                "profile": profile_result,
            }
        )
    frontier = pareto_frontier(
        rows,
        in_window_key=in_window_key,
        ood_key=ood_key,
        in_window_direction=in_window_direction,
        ood_direction=ood_direction,
    )
    frontier_profiles = [profiles[index] for index in frontier]
    bimodal_frontier = [row for row in frontier_profiles if row["profile"]["label"] == "bimodal"]
    bimodal_any = [row for row in profiles if row["profile"]["label"] == "bimodal"]
    if metric_rows == 0:
        decision = "UNDETERMINED_NO_METRICS"
    elif bimodal_frontier:
        decision = "BIMODAL_FRONTIER_SIGNAL"
    elif bimodal_any:
        decision = "BIMODAL_NOT_PARETO"
    else:
        decision = "NO_BIMODAL_FRONTIER_SIGNAL"
    return {
        "schema_version": 1,
        "status": "SCREENING_ONLY",
        "metric_keys": {
            "in_window": in_window_key,
            "ood": ood_key,
            "in_window_direction": in_window_direction,
            "ood_direction": ood_direction,
        },
        "row_count": len(rows),
        "metric_row_count": metric_rows,
        "pareto_frontier_indices": frontier,
        "bimodal_frontier_count": len(bimodal_frontier),
        "bimodal_any_count": len(bimodal_any),
        "screening_decision": decision,
        "rows": profiles,
        "limitations": [
            "No statistical test or training-seed uncertainty is computed.",
            "Bimodal is a descriptive local-peak screen and is not evidence of a causal regime.",
            "Metric directions must match the supplied endpoint (PPL/NLL usually min; F1/RULER usually max).",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--in-window-key", default="in_window")
    parser.add_argument("--ood-key", default="ood")
    parser.add_argument("--in-window-direction", choices=("min", "max"), default="min")
    parser.add_argument("--ood-direction", choices=("min", "max"), default="min")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.results_json.read_text(encoding="utf-8"))
    result = analyze_payload(
        payload,
        in_window_key=args.in_window_key,
        ood_key=args.ood_key,
        in_window_direction=args.in_window_direction,
        ood_direction=args.ood_direction,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()

