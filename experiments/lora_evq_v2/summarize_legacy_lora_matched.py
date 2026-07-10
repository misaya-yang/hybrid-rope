#!/usr/bin/env python3
"""Summarize the strict six-arm legacy LoRA matrix with paired seed deltas."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

try:
    from .legacy_lora_protocol import (
        LEGACY_METHODS,
        LEGACY_SEEDS,
        legacy_run_name,
        paired_metric_summary,
        validate_complete_matrix,
    )
except ImportError:
    from legacy_lora_protocol import (
        LEGACY_METHODS,
        LEGACY_SEEDS,
        legacy_run_name,
        paired_metric_summary,
        validate_complete_matrix,
    )


def summarize_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [dict(record) for record in records]
    by_variant = {row.get("variant"): row for row in rows}
    expected_variants = {"base_geo", "base_evq_tau1414"}
    expected_variants.update(
        legacy_run_name(method, seed)
        for method in LEGACY_METHODS
        for seed in LEGACY_SEEDS
    )
    if set(by_variant) != expected_variants or len(rows) != len(expected_variants):
        raise ValueError("summary requires exactly two baselines and six fresh adapter results")
    hex64 = re.compile(r"^[0-9a-f]{64}$")
    for field in ("eval_manifest_sha256", "model_manifest_sha256"):
        identities = {row.get(field) for row in rows}
        if len(identities) != 1 or not hex64.fullmatch(str(next(iter(identities), ""))):
            raise ValueError(f"all evaluations must share one {field}")
    expected_specs = {
        "base_geo": ("native_geo", None),
        "base_evq_tau1414": ("evq_cosh", None),
    }
    expected_specs.update({
        legacy_run_name(method, seed): (method, seed)
        for method in LEGACY_METHODS
        for seed in LEGACY_SEEDS
    })
    for variant, row in by_variant.items():
        method, seed = expected_specs[variant]
        if row.get("format_version") != 1 or row.get("method") != method or row.get("seed") != seed:
            raise ValueError(f"evaluation identity mismatch for {variant}")
        if row.get("frequency_provenance", {}).get("method") != method:
            raise ValueError(f"frequency provenance mismatch for {variant}")
        if seed is None:
            if row.get("adapter_sha256") is not None:
                raise ValueError(f"static baseline {variant} must not claim an adapter")
        else:
            for field in (
                "adapter_sha256",
                "training_data_manifest_sha256",
                "training_code_sha256",
            ):
                if not hex64.fullmatch(str(row.get(field, ""))):
                    raise ValueError(f"adapter evaluation {variant} lacks valid {field}")
        for length, context in (("8K", 8192), ("16K", 16384), ("32K", 32768)):
            metric = row.get("ppl", {}).get(length, {})
            nll = metric.get("nll")
            ppl = metric.get("ppl")
            if not isinstance(nll, (int, float)) or not math.isfinite(float(nll)):
                raise ValueError(f"non-finite NLL for {variant} at {length}")
            if not isinstance(ppl, (int, float)) or not math.isfinite(float(ppl)) or float(ppl) <= 0:
                raise ValueError(f"invalid PPL for {variant} at {length}")
            if metric.get("chunks") != 5 or metric.get("scored_tokens") != 5 * (context - 1):
                raise ValueError(f"evaluation coverage mismatch for {variant} at {length}")
            chunk_nll = metric.get("per_chunk_nll")
            if not isinstance(chunk_nll, list) or len(chunk_nll) != 5 or not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in chunk_nll
            ):
                raise ValueError(f"per-chunk NLL evidence missing for {variant} at {length}")
    train_rows = [
        {
            "method": row["method"],
            "seed": row["seed"],
            "data_manifest_sha256": row["training_data_manifest_sha256"],
            "model_manifest_sha256": row["model_manifest_sha256"],
            "code_sha256": row["training_code_sha256"],
            "status": "complete",
        }
        for row in rows
        if row.get("seed") is not None
    ]
    validate_complete_matrix(train_rows)
    metrics = {}
    for length in ("8K", "16K", "32K"):
        geo = {
            seed: float(by_variant[legacy_run_name("native_geo", seed)]["ppl"][length]["ppl"])
            for seed in LEGACY_SEEDS
        }
        evq = {
            seed: float(by_variant[legacy_run_name("evq_cosh", seed)]["ppl"][length]["ppl"])
            for seed in LEGACY_SEEDS
        }
        metrics[f"ppl@{length}"] = paired_metric_summary(geo, evq)
    return {
        "format_version": 1,
        "claim_scope": "protocol-matched multi-seed rerun on verified LongAlign-10k",
        "seed_scope": list(LEGACY_SEEDS),
        "static_baselines": {
            name: by_variant[name]["ppl"] for name in ("base_geo", "base_evq_tau1414")
        },
        "metrics": metrics,
        "provenance": {
            "eval_manifest_sha256": rows[0]["eval_manifest_sha256"],
            "model_manifest_sha256": rows[0]["model_manifest_sha256"],
        },
        "raw_records": rows,
        "statistical_note": "n=3; report raw seeds and paired descriptive statistics only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.results_dir.glob("eval_*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = summarize_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
