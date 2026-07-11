#!/usr/bin/env python3
"""Summarize Base-Geo versus a 300-step Geo+LoRA seed-42 control."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


HISTORICAL_BASE_8K_PPL = 7.42
HISTORICAL_EVQ_LORA_8K_PPL = 9.63
HISTORICAL_8K_TRADEOFF_PCT = (
    HISTORICAL_EVQ_LORA_8K_PPL / HISTORICAL_BASE_8K_PPL - 1.0
) * 100.0
TRADEOFF_MATCH_TOLERANCE_PERCENTAGE_POINTS = 5.0
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def _validate_record(
    record: Mapping[str, Any],
    *,
    variant: str,
    seed: int | None,
) -> Dict[str, Any]:
    expected = {
        "format_version": 1,
        "variant": variant,
        "method": "native_geo",
        "seed": seed,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{variant} identity mismatch for {key}")
    for field in ("eval_manifest_sha256", "model_manifest_sha256"):
        if not _HEX_64.fullmatch(str(record.get(field, ""))):
            raise ValueError(f"{variant} lacks valid {field}")
    if record.get("frequency_provenance", {}).get("method") != "native_geo":
        raise ValueError(f"{variant} frequency provenance is not native_geo")
    if seed is None:
        if record.get("adapter_sha256") is not None:
            raise ValueError("Base-Geo must not contain an adapter")
    else:
        for field in (
            "adapter_sha256",
            "training_data_manifest_sha256",
            "training_code_sha256",
        ):
            if not _HEX_64.fullmatch(str(record.get(field, ""))):
                raise ValueError(f"Geo+LoRA control lacks valid {field}")
    for length, context in (("8K", 8192), ("16K", 16384), ("32K", 32768)):
        metric = record.get("ppl", {}).get(length, {})
        nll = metric.get("nll")
        ppl = metric.get("ppl")
        if not isinstance(nll, (int, float)) or not math.isfinite(float(nll)):
            raise ValueError(f"{variant} has invalid NLL at {length}")
        if not isinstance(ppl, (int, float)) or not math.isfinite(float(ppl)) or float(ppl) <= 0:
            raise ValueError(f"{variant} has invalid PPL at {length}")
        if metric.get("chunks") != 5 or metric.get("scored_tokens") != 5 * (context - 1):
            raise ValueError(f"{variant} has incomplete coverage at {length}")
        per_chunk = metric.get("per_chunk_nll")
        if not isinstance(per_chunk, list) or len(per_chunk) != 5 or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in per_chunk
        ):
            raise ValueError(f"{variant} lacks per-chunk NLL at {length}")
    return dict(record)


def summarize_geo_control_records(
    records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = list(records)
    if len(rows) != 2:
        raise ValueError("Geo control requires exactly Base-Geo and Geo+LoRA-s42")
    by_variant = {row.get("variant"): row for row in rows}
    if set(by_variant) != {"base_geo", "geo_longalign_s42"}:
        raise ValueError("Geo control received an unexpected evaluation variant")
    base = _validate_record(by_variant["base_geo"], variant="base_geo", seed=None)
    geo = _validate_record(
        by_variant["geo_longalign_s42"],
        variant="geo_longalign_s42",
        seed=42,
    )
    for field in ("eval_manifest_sha256", "model_manifest_sha256"):
        if base[field] != geo[field]:
            raise ValueError(f"Geo control records do not share {field}")

    metrics = {}
    for length in ("8K", "16K", "32K"):
        base_ppl = float(base["ppl"][length]["ppl"])
        geo_ppl = float(geo["ppl"][length]["ppl"])
        ratio = geo_ppl / base_ppl
        metrics[f"ppl@{length}"] = {
            "base_geo": base_ppl,
            "geo_lora": geo_ppl,
            "ratio_geo_lora_over_base": ratio,
            "drift_pct": (ratio - 1.0) * 100.0,
            "nll_delta_geo_lora_minus_base": (
                float(geo["ppl"][length]["nll"])
                - float(base["ppl"][length]["nll"])
            ),
        }
    observed_8k_drift = metrics["ppl@8K"]["drift_pct"]
    distance = observed_8k_drift - HISTORICAL_8K_TRADEOFF_PCT
    return {
        "format_version": 1,
        "protocol": "legacy_longalign_geo_lora_seed42_control",
        "comparison": "Geo+LoRA-s42 minus Base-Geo",
        "seed": 42,
        "training_steps": 300,
        "metrics": metrics,
        "historical_reference": {
            "base_8k_ppl": HISTORICAL_BASE_8K_PPL,
            "evq_lora_8k_ppl": HISTORICAL_EVQ_LORA_8K_PPL,
            "tradeoff_pct": HISTORICAL_8K_TRADEOFF_PCT,
            "match_tolerance_percentage_points": (
                TRADEOFF_MATCH_TOLERANCE_PERCENTAGE_POINTS
            ),
        },
        "gate": {
            "observed_geo_8k_drift_pct": observed_8k_drift,
            "distance_from_historical_tradeoff_percentage_points": distance,
            "supports_finetuning_drift_explanation": (
                abs(distance) <= TRADEOFF_MATCH_TOLERANCE_PERCENTAGE_POINTS
            ),
        },
        "provenance": {
            "eval_manifest_sha256": base["eval_manifest_sha256"],
            "model_manifest_sha256": base["model_manifest_sha256"],
            "training_data_manifest_sha256": geo[
                "training_data_manifest_sha256"
            ],
            "training_code_sha256": geo["training_code_sha256"],
            "adapter_sha256": geo["adapter_sha256"],
        },
        "claim_boundary": (
            "A matched seed-42 control can attribute comparable 8K drift to "
            "the LongAlign/LoRA update, but remains single-seed evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads((args.results_dir / name).read_text(encoding="utf-8"))
        for name in ("eval_base_geo.json", "eval_geo_longalign_s42.json")
    ]
    summary = summarize_geo_control_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "gate": summary["gate"]}, indent=2))


if __name__ == "__main__":
    main()
