#!/usr/bin/env python3
"""Build portable model-by-length matrices from complete experiment summaries.

This module deliberately consumes a small, explicit summary contract instead of
guessing model, scale, or dataset identity from paths.  Each ``--source`` JSON
must have this shape (irrelevant optional fields are omitted)::

    {
      "schema": "STRONG_MATRIX_SOURCE_V1",
      "status": "complete",
      "experiment_id": "olmo_s4_clean_ruler50",
      "identity": {
        "model_id": "olmo2_1b",
        "benchmark_family": "ruler",
        "data_contract": "clean",
        "scale": 4,
        "native_length_tokens": 4096,
        "evaluation_contract": "full13-source-order-v1",
        "metric": {"name": "task_macro_official", "direction": "higher", "unit": "fraction"},
        "expected_length_multiples": [2, 4],
        "required_arms": ["tailspline", "mrpro"],
        "required_contrasts": [{"candidate": "tailspline", "baseline": "mrpro"}]
      },
      "cells": [{
        "length_tokens": 8192,
        "length_multiple": 2,
        "expected_rows_per_arm": 650,
        "paired_rows": 650,
        "arms": {
          "tailspline": {"status": "complete", "rows": 650, "score": 0.80},
          "mrpro": {"status": "complete", "rows": 650, "score": 0.70}
        },
        "contrasts": [{
          "candidate": "tailspline", "baseline": "mrpro", "delta": 0.10,
          "ci95": [0.05, 0.15], "uncertainty_unit": "paired_prompts"
        }]
      }]
    }

The output groups sources by the *entire* evaluation identity.  Consequently
clean/classic, S2/S4/S16, RULER/natural/PPL, metric, arm, and evaluation
contracts cannot be pooled accidentally.  Row-level paired intervals remain in
their cells.  Cross-model summaries contain descriptive means/ranges only and
never reinterpret row bootstraps as model-level uncertainty.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable


SOURCE_SCHEMA = "STRONG_MATRIX_SOURCE_V1"
SUMMARY_SCHEMA = "STRONG_MATRIX_SUMMARY_V1"
BENCHMARK_FAMILIES = {"ruler", "natural", "ppl"}
ALLOWED_SCALES = {2, 4, 16}
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _positive_int(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{label} must be a positive integer")
    return value


def _slug(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(_SLUG.fullmatch(value)), f"{label} must be a lowercase portable slug")
    return value


def _portable_text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value.strip() == value and bool(value), f"{label} must be non-empty text")
    _require("misaya." not in value.lower(), f"{label} contains a personal path fragment")
    _require(not Path(value).is_absolute(), f"{label} must not be an absolute path")
    return value


def _multiple(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    _require(result > 0, f"{label} must be positive")
    return result


def _multiple_key(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        return f"{rounded}L"
    return f"{value:g}L"


def _contrast_key(candidate: str, baseline: str) -> str:
    return f"{candidate}_minus_{baseline}"


def _normalize_required_contrasts(value: Any, required_arms: tuple[str, ...], label: str) -> tuple[tuple[str, str], ...]:
    _require(isinstance(value, list), f"{label} must be a list")
    normalized: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        _require(isinstance(item, dict), f"{label}[{index}] must be an object")
        candidate = _slug(item.get("candidate"), f"{label}[{index}].candidate")
        baseline = _slug(item.get("baseline"), f"{label}[{index}].baseline")
        _require(candidate != baseline, f"{label}[{index}] compares an arm with itself")
        _require(candidate in required_arms and baseline in required_arms, f"{label}[{index}] names an undeclared arm")
        normalized.append((candidate, baseline))
    _require(len(normalized) == len(set(normalized)), f"{label} contains duplicate contrasts")
    return tuple(normalized)


def _validate_arm_identity(arm: dict[str, Any], identity: dict[str, Any], label: str) -> None:
    """Reject optional arm-local metadata that contradicts the source identity."""
    expected = {
        "model_id": identity["model_id"],
        "benchmark_family": identity["benchmark_family"],
        "data_contract": identity["data_contract"],
        "scale": identity["scale"],
        "evaluation_contract": identity["evaluation_contract"],
    }
    for field, wanted in expected.items():
        if field in arm:
            _require(arm[field] == wanted, f"{label}.{field} mixes evaluation contracts")


def normalize_source(payload: Any, *, source_name: str = "source") -> dict[str, Any]:
    """Validate and normalize one complete experiment summary."""
    _require(isinstance(payload, dict), f"{source_name} must contain a JSON object")
    _require(payload.get("schema") == SOURCE_SCHEMA, f"{source_name} has unsupported schema")
    _require(payload.get("status") == "complete", f"{source_name} is not a complete experiment")
    experiment_id = _slug(payload.get("experiment_id"), f"{source_name}.experiment_id")

    raw_identity = payload.get("identity")
    _require(isinstance(raw_identity, dict), f"{source_name}.identity must be an object")
    model_id = _slug(raw_identity.get("model_id"), f"{source_name}.identity.model_id")
    benchmark = _slug(raw_identity.get("benchmark_family"), f"{source_name}.identity.benchmark_family")
    _require(benchmark in BENCHMARK_FAMILIES, f"{source_name} has unsupported benchmark_family")
    data_contract = _slug(raw_identity.get("data_contract"), f"{source_name}.identity.data_contract")
    scale = _positive_int(raw_identity.get("scale"), f"{source_name}.identity.scale")
    _require(scale in ALLOWED_SCALES, f"{source_name}.identity.scale must be one of S2/S4/S16")
    native_length = _positive_int(
        raw_identity.get("native_length_tokens"), f"{source_name}.identity.native_length_tokens"
    )
    evaluation_contract = _portable_text(
        raw_identity.get("evaluation_contract"), f"{source_name}.identity.evaluation_contract"
    )

    raw_metric = raw_identity.get("metric")
    _require(isinstance(raw_metric, dict), f"{source_name}.identity.metric must be an object")
    metric = {
        "name": _slug(raw_metric.get("name"), f"{source_name}.identity.metric.name"),
        "direction": _slug(raw_metric.get("direction"), f"{source_name}.identity.metric.direction"),
        "unit": _slug(raw_metric.get("unit"), f"{source_name}.identity.metric.unit"),
    }
    _require(metric["direction"] in {"higher", "lower"}, f"{source_name}.identity.metric.direction is invalid")

    raw_arms = raw_identity.get("required_arms")
    _require(isinstance(raw_arms, list) and raw_arms, f"{source_name}.identity.required_arms must be non-empty")
    required_arms = tuple(_slug(value, f"{source_name}.identity.required_arms") for value in raw_arms)
    _require(len(required_arms) == len(set(required_arms)), f"{source_name}.identity.required_arms has duplicates")
    required_contrasts = _normalize_required_contrasts(
        raw_identity.get("required_contrasts", []), required_arms, f"{source_name}.identity.required_contrasts"
    )

    raw_expected = raw_identity.get("expected_length_multiples")
    _require(isinstance(raw_expected, list) and raw_expected, f"{source_name}.identity.expected_length_multiples must be non-empty")
    expected_multiples = tuple(
        _multiple(value, f"{source_name}.identity.expected_length_multiples") for value in raw_expected
    )
    expected_keys = tuple(_multiple_key(value) for value in expected_multiples)
    _require(len(expected_keys) == len(set(expected_keys)), f"{source_name} has duplicate expected lengths")

    identity = {
        "model_id": model_id,
        "benchmark_family": benchmark,
        "data_contract": data_contract,
        "scale": scale,
        "native_length_tokens": native_length,
        "evaluation_contract": evaluation_contract,
        "metric": metric,
        "required_arms": list(required_arms),
        "required_contrasts": [
            {"candidate": candidate, "baseline": baseline}
            for candidate, baseline in required_contrasts
        ],
    }

    raw_cells = payload.get("cells")
    _require(isinstance(raw_cells, list) and raw_cells, f"{source_name}.cells must be non-empty")
    cells: list[dict[str, Any]] = []
    seen_lengths: set[str] = set()
    for cell_index, raw_cell in enumerate(raw_cells):
        cell_label = f"{source_name}.cells[{cell_index}]"
        _require(isinstance(raw_cell, dict), f"{cell_label} must be an object")
        length_tokens = _positive_int(raw_cell.get("length_tokens"), f"{cell_label}.length_tokens")
        length_multiple = _multiple(raw_cell.get("length_multiple"), f"{cell_label}.length_multiple")
        length_key = _multiple_key(length_multiple)
        _require(length_key not in seen_lengths, f"{source_name} repeats {length_key}")
        seen_lengths.add(length_key)
        expected_tokens = native_length * length_multiple
        tolerance = max(1.0, native_length * 0.005)
        _require(
            math.isclose(length_tokens, expected_tokens, rel_tol=0.0, abs_tol=tolerance),
            f"{cell_label} length_tokens disagrees with native_length_tokens * length_multiple",
        )
        expected_rows = _positive_int(
            raw_cell.get("expected_rows_per_arm"), f"{cell_label}.expected_rows_per_arm"
        )
        paired_rows = _positive_int(raw_cell.get("paired_rows"), f"{cell_label}.paired_rows")
        _require(paired_rows == expected_rows, f"{cell_label} is not complete on paired rows")

        raw_arm_entries = raw_cell.get("arms")
        _require(isinstance(raw_arm_entries, dict), f"{cell_label}.arms must be an object")
        _require(set(raw_arm_entries) == set(required_arms), f"{cell_label} does not contain exactly the required arms")
        arms: dict[str, dict[str, Any]] = {}
        for arm_name in required_arms:
            raw_arm = raw_arm_entries[arm_name]
            arm_label = f"{cell_label}.arms.{arm_name}"
            _require(isinstance(raw_arm, dict), f"{arm_label} must be an object")
            _require(raw_arm.get("status") == "complete", f"{arm_label} is incomplete")
            rows = _positive_int(raw_arm.get("rows"), f"{arm_label}.rows")
            _require(rows == expected_rows, f"{arm_label} row count is incomplete")
            _validate_arm_identity(raw_arm, identity, arm_label)
            arms[arm_name] = {"rows": rows, "score": _finite_number(raw_arm.get("score"), f"{arm_label}.score")}

        raw_contrasts = raw_cell.get("contrasts", [])
        _require(isinstance(raw_contrasts, list), f"{cell_label}.contrasts must be a list")
        contrasts: dict[str, dict[str, Any]] = {}
        for contrast_index, raw_contrast in enumerate(raw_contrasts):
            contrast_label = f"{cell_label}.contrasts[{contrast_index}]"
            _require(isinstance(raw_contrast, dict), f"{contrast_label} must be an object")
            candidate = _slug(raw_contrast.get("candidate"), f"{contrast_label}.candidate")
            baseline = _slug(raw_contrast.get("baseline"), f"{contrast_label}.baseline")
            pair = (candidate, baseline)
            _require(pair in required_contrasts, f"{contrast_label} is not a required contrast")
            key = _contrast_key(candidate, baseline)
            _require(key not in contrasts, f"{cell_label} repeats contrast {key}")
            delta = _finite_number(raw_contrast.get("delta"), f"{contrast_label}.delta")
            expected_delta = arms[candidate]["score"] - arms[baseline]["score"]
            _require(math.isclose(delta, expected_delta, rel_tol=1e-8, abs_tol=1e-10), f"{contrast_label}.delta disagrees with arm scores")
            raw_ci = raw_contrast.get("ci95")
            _require(isinstance(raw_ci, list) and len(raw_ci) == 2, f"{contrast_label}.ci95 must have two bounds")
            ci95 = [_finite_number(raw_ci[0], f"{contrast_label}.ci95[0]"), _finite_number(raw_ci[1], f"{contrast_label}.ci95[1]")]
            _require(ci95[0] <= ci95[1], f"{contrast_label}.ci95 is reversed")
            uncertainty_unit = _slug(raw_contrast.get("uncertainty_unit"), f"{contrast_label}.uncertainty_unit")
            _require(
                uncertainty_unit not in {"models", "cross_model", "pooled_models"},
                f"{contrast_label} incorrectly claims model-level uncertainty",
            )
            contrasts[key] = {
                "candidate": candidate,
                "baseline": baseline,
                "delta": delta,
                "ci95": ci95,
                "uncertainty_unit": uncertainty_unit,
            }
        _require(
            set(contrasts) == {_contrast_key(candidate, baseline) for candidate, baseline in required_contrasts},
            f"{cell_label} does not contain exactly the required contrasts",
        )
        cells.append({
            "length_tokens": length_tokens,
            "length_multiple": length_multiple,
            "length_key": length_key,
            "paired_rows": paired_rows,
            "arms": arms,
            "contrasts": contrasts,
        })

    _require(seen_lengths == set(expected_keys), f"{source_name} does not contain exactly its declared lengths")
    return {"experiment_id": experiment_id, "identity": identity, "cells": cells}


def load_source(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return normalize_source(json.load(stream), source_name=Path(path).name)


def _group_key(source: dict[str, Any]) -> tuple[Any, ...]:
    identity = source["identity"]
    metric = identity["metric"]
    return (
        identity["benchmark_family"],
        identity["data_contract"],
        identity["scale"],
        identity["evaluation_contract"],
        metric["name"],
        metric["direction"],
        metric["unit"],
        tuple(identity["required_arms"]),
        tuple((value["candidate"], value["baseline"]) for value in identity["required_contrasts"]),
    )


def build_summary(sources: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Create strictly partitioned matrices and descriptive model summaries."""
    normalized_sources = list(sources)
    _require(normalized_sources, "at least one source is required")
    experiment_ids = [source["experiment_id"] for source in normalized_sources]
    _require(len(experiment_ids) == len(set(experiment_ids)), "experiment_id values must be unique")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for source in normalized_sources:
        grouped.setdefault(_group_key(source), []).append(source)

    matrices: list[dict[str, Any]] = []
    for matrix_index, key in enumerate(sorted(grouped, key=lambda value: tuple(map(str, value))), start=1):
        group = grouped[key]
        first = group[0]["identity"]
        by_model: dict[str, dict[str, Any]] = {}
        used_cells: set[tuple[str, str]] = set()
        for source in sorted(group, key=lambda value: value["experiment_id"]):
            identity = source["identity"]
            model_id = identity["model_id"]
            row = by_model.setdefault(model_id, {
                "model_id": model_id,
                "native_length_tokens": identity["native_length_tokens"],
                "experiment_ids": [],
                "cells": {},
            })
            _require(
                row["native_length_tokens"] == identity["native_length_tokens"],
                f"model {model_id} has inconsistent native lengths within one matrix",
            )
            row["experiment_ids"].append(source["experiment_id"])
            for cell in source["cells"]:
                location = (model_id, cell["length_key"])
                _require(location not in used_cells, f"duplicate matrix cell {model_id}/{cell['length_key']}")
                used_cells.add(location)
                row["cells"][cell["length_key"]] = {
                    "length_tokens": cell["length_tokens"],
                    "paired_rows": cell["paired_rows"],
                    "arms": cell["arms"],
                    "contrasts": cell["contrasts"],
                }

        columns = sorted(
            {column for row in by_model.values() for column in row["cells"]},
            key=lambda value: float(value[:-1]),
        )
        rows = [by_model[model_id] for model_id in sorted(by_model)]
        contrast_names = [
            _contrast_key(value["candidate"], value["baseline"])
            for value in first["required_contrasts"]
        ]
        descriptive: dict[str, Any] = {}
        for column in columns:
            available = [row for row in rows if column in row["cells"]]
            contrast_summary: dict[str, Any] = {}
            for contrast_name in contrast_names:
                values = [row["cells"][column]["contrasts"][contrast_name]["delta"] for row in available]
                contrast_summary[contrast_name] = {
                    "n_models": len(values),
                    "models": [row["model_id"] for row in available],
                    "mean_delta": sum(values) / len(values),
                    "min_delta": min(values),
                    "max_delta": max(values),
                }
            descriptive[column] = {"contrasts": contrast_summary}

        matrices.append({
            "matrix_id": f"matrix_{matrix_index:02d}",
            "identity": {
                "benchmark_family": first["benchmark_family"],
                "data_contract": first["data_contract"],
                "scale": f"S{first['scale']}",
                "evaluation_contract": first["evaluation_contract"],
                "metric": first["metric"],
                "required_arms": first["required_arms"],
                "required_contrasts": first["required_contrasts"],
            },
            "columns": columns,
            "rows": rows,
            "cross_model_descriptive": descriptive,
            "cross_model_uncertainty": None,
            "cross_model_note": (
                "Unweighted model-level means and ranges are descriptive only. "
                "Cell-level paired intervals are not pooled or reinterpreted as cross-model confidence intervals."
            ),
        })

    return {
        "schema": SUMMARY_SCHEMA,
        "status": "complete_sources_only",
        "source_experiment_ids": sorted(experiment_ids),
        "matrices": matrices,
        "separation_policy": (
            "Matrices are partitioned by benchmark family, data contract, scale, evaluation contract, "
            "metric, required arms, and required contrasts."
        ),
    }


def _format_value(value: float, unit: str, *, signed: bool = False) -> str:
    if unit == "fraction":
        return f"{value * 100:+.2f}pp" if signed else f"{value * 100:.2f}%"
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value:.6g}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the normalized summary without embedding source filesystem paths."""
    _require(report.get("schema") == SUMMARY_SCHEMA, "unsupported summary schema")
    lines = [
        "# Strong evidence model-by-length matrices",
        "",
        "Only complete experiments are included. Each table has one evaluation identity; clean/classic, "
        "S2/S4/S16, and RULER/natural/PPL results are never pooled.",
        "",
    ]
    for matrix in report["matrices"]:
        identity = matrix["identity"]
        metric = identity["metric"]
        lines.extend([
            f"## {identity['benchmark_family']} · {identity['data_contract']} · {identity['scale']}",
            "",
            f"Contract: `{identity['evaluation_contract']}`. Metric: `{metric['name']}` "
            f"({metric['direction']} is better; unit `{metric['unit']}`).",
            "",
        ])
        columns = matrix["columns"]
        lines.append("| Model | Native tokens | " + " | ".join(columns) + " |")
        lines.append("|---|---:|" + "|".join("---" for _ in columns) + "|")
        for row in matrix["rows"]:
            rendered_cells = []
            for column in columns:
                cell = row["cells"].get(column)
                if cell is None:
                    rendered_cells.append("—")
                    continue
                arms = ", ".join(
                    f"{arm} {_format_value(summary['score'], metric['unit'])}"
                    for arm, summary in cell["arms"].items()
                )
                contrasts = []
                for contrast in cell["contrasts"].values():
                    low, high = contrast["ci95"]
                    contrasts.append(
                        f"{contrast['candidate']}−{contrast['baseline']} "
                        f"{_format_value(contrast['delta'], metric['unit'], signed=True)} "
                        f"[95% {_format_value(low, metric['unit'], signed=True)}, "
                        f"{_format_value(high, metric['unit'], signed=True)}]"
                    )
                rendered_cells.append(
                    f"{cell['length_tokens']} tok; {arms}; {'; '.join(contrasts)}; paired n={cell['paired_rows']}"
                )
            lines.append(
                f"| {row['model_id']} | {row['native_length_tokens']} | " + " | ".join(rendered_cells) + " |"
            )
        lines.extend(["", "Cross-model deltas below are descriptive model-level summaries only; no pooled confidence interval is reported.", ""])
        if matrix["columns"] and identity["required_contrasts"]:
            lines.append("| Length | Contrast | Models | Mean delta | Range |")
            lines.append("|---|---|---:|---:|---:|")
            for column in matrix["columns"]:
                for name, summary in matrix["cross_model_descriptive"][column]["contrasts"].items():
                    lines.append(
                        f"| {column} | {name} | {summary['n_models']} | "
                        f"{_format_value(summary['mean_delta'], metric['unit'], signed=True)} | "
                        f"{_format_value(summary['min_delta'], metric['unit'], signed=True)} to "
                        f"{_format_value(summary['max_delta'], metric['unit'], signed=True)} |"
                    )
            lines.append("")
    result = "\n".join(lines).rstrip() + "\n"
    _require("misaya." not in result.lower(), "rendered report contains a personal path fragment")
    return result


def _atomic_write(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True, help="Complete STRONG_MATRIX_SOURCE_V1 JSON; repeatable")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-markdown", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    _require(args.out_json != args.out_markdown, "JSON and Markdown outputs must differ")
    if not args.overwrite:
        _require(not args.out_json.exists() and not args.out_markdown.exists(), "output already exists; pass --overwrite to replace")
    report = build_summary(load_source(path) for path in args.source)
    markdown = render_markdown(report)
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _require("misaya." not in encoded.lower(), "JSON report contains a personal path fragment")
    _atomic_write(args.out_json, encoded)
    _atomic_write(args.out_markdown, markdown)
    print(json.dumps({
        "schema": SUMMARY_SCHEMA,
        "sources": len(report["source_experiment_ids"]),
        "matrices": len(report["matrices"]),
        # Keep even captured launcher logs free of user-specific absolute paths.
        "out_json": args.out_json.name,
        "out_markdown": args.out_markdown.name,
    }, indent=2))


if __name__ == "__main__":
    main()
