#!/usr/bin/env python3
"""Summarize the four-arm positional-distillation pilot against fixed gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .eval_positional_distill import representation_recovery, result_filename
except ImportError:
    from eval_positional_distill import representation_recovery, result_filename


REQUIRED_VARIANTS = (
    "base_geo",
    "base_evq",
    "geo_distill_s42",
    "evq_distill_s42",
)


def _ppl(row: dict, length: str) -> float:
    try:
        value = float(row["ppl"][length]["ppl"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing valid PPL@{length}") from error
    if value <= 0:
        raise ValueError(f"PPL@{length} must be positive")
    return value


def _hidden_error(row: dict) -> float:
    try:
        value = float(row["hidden_error"]["normalized_mse"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("missing valid hidden representation error") from error
    if value < 0:
        raise ValueError("hidden representation error must be non-negative")
    return value


def summarize_results(results: dict[str, dict]) -> dict:
    missing = [variant for variant in REQUIRED_VARIANTS if variant not in results]
    if missing:
        raise ValueError(f"missing required variants: {', '.join(missing)}")
    rows = {variant: results[variant] for variant in REQUIRED_VARIANTS}

    manifest_hashes = {row.get("data_manifest_sha256") for row in rows.values()}
    if len(manifest_hashes) != 1 or None in manifest_hashes:
        raise ValueError("all arms must share one data manifest")
    models = {row.get("model") for row in rows.values()}
    if len(models) != 1 or None in models:
        raise ValueError("all arms must share one model identity")
    fingerprint_by_json = {
        json.dumps(row.get("base_model_fingerprint"), sort_keys=True)
        for row in rows.values()
    }
    if len(fingerprint_by_json) != 1 or "null" in fingerprint_by_json:
        raise ValueError("all arms must share one model fingerprint")
    base_model_fingerprint = json.loads(next(iter(fingerprint_by_json)))

    base = rows["base_geo"]
    base_evq = rows["base_evq"]
    geo = rows["geo_distill_s42"]
    evq = rows["evq_distill_s42"]
    base_8k = _ppl(base, "8K")
    geo_8k_drift = (_ppl(geo, "8K") / base_8k - 1.0) * 100.0
    evq_8k_drift = (_ppl(evq, "8K") / base_8k - 1.0) * 100.0
    evq_16k_factor = _ppl(base, "16K") / _ppl(evq, "16K")
    evq_32k_factor = _ppl(base, "32K") / _ppl(evq, "32K")
    recovery = representation_recovery(
        injection_error=_hidden_error(base_evq),
        adapted_error=_hidden_error(evq),
    )

    metrics = {
        "geo_8k_drift_pct": geo_8k_drift,
        "evq_8k_drift_pct": evq_8k_drift,
        "evq_16k_improvement_factor": evq_16k_factor,
        "evq_32k_improvement_factor": evq_32k_factor,
        "representation_recovery": recovery,
        "base_evq_injection_error": _hidden_error(base_evq),
        "evq_distill_hidden_error": _hidden_error(evq),
    }
    gates = {
        "geo_8k_drift_at_most_1pct": geo_8k_drift <= 1.0,
        "evq_8k_drift_at_most_5pct": evq_8k_drift <= 5.0,
        "evq_16k_improvement_at_least_2x": evq_16k_factor >= 2.0,
        "evq_32k_improvement_at_least_4x": evq_32k_factor >= 4.0,
        "representation_recovery_at_least_90pct": (
            recovery is not None and recovery >= 0.9
        ),
    }
    return {
        "format_version": 1,
        "protocol": "llama8b_positional_hidden_distillation_seed42",
        "model": next(iter(models)),
        "base_model_fingerprint": base_model_fingerprint,
        "data_manifest_sha256": next(iter(manifest_hashes)),
        "metrics": metrics,
        "gates": gates,
        "pilot_pass": all(gates.values()),
        "scope": (
            "single-seed supporting pilot; quick capability metrics are not part "
            "of these gates"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {}
    for variant in REQUIRED_VARIANTS:
        path = args.result_dir / result_filename(variant)
        results[variant] = json.loads(path.read_text(encoding="utf-8"))
    summary = summarize_results(results)
    output_path = args.output or args.result_dir / "positional_distill_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["gates"], indent=2, sort_keys=True))
    print(f"pilot_pass={summary['pilot_pass']}")


if __name__ == "__main__":
    main()
