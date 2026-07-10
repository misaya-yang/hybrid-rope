#!/usr/bin/env python3
"""Summarize the four-arm positional-distillation pilot against fixed gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from .eval_positional_distill import (
        representation_recovery,
        result_filename,
        validate_eval_claim_protocol,
    )
except ImportError:
    from eval_positional_distill import (
        representation_recovery,
        result_filename,
        validate_eval_claim_protocol,
    )


REQUIRED_VARIANTS = (
    "base_geo",
    "base_evq",
    "geo_distill_s42",
    "evq_distill_s42",
)

ARM_CONTRACT = {
    "base_geo": ("native_geo", False),
    "base_evq": ("evq_cosh", False),
    "geo_distill_s42": ("native_geo", True),
    "evq_distill_s42": ("evq_cosh", True),
}


def _ppl(row: dict, length: str) -> float:
    try:
        value = float(row["ppl"][length]["ppl"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing valid PPL@{length}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"PPL@{length} must be positive")
    return value


def _hidden_error(row: dict) -> float:
    try:
        value = float(row["hidden_error"]["normalized_mse"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("missing valid hidden representation error") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError("hidden representation error must be non-negative")
    return value


def summarize_results(results: dict[str, dict]) -> dict:
    missing = [variant for variant in REQUIRED_VARIANTS if variant not in results]
    if missing:
        raise ValueError(f"missing required variants: {', '.join(missing)}")
    rows = {variant: results[variant] for variant in REQUIRED_VARIANTS}

    for variant, row in rows.items():
        method, requires_adapter = ARM_CONTRACT[variant]
        if row.get("variant") != variant:
            raise ValueError(f"variant mismatch for {variant}")
        if row.get("candidate_method") != method:
            raise ValueError(f"candidate method mismatch for {variant}")
        has_adapter = row.get("adapter") is not None
        if has_adapter != requires_adapter:
            raise ValueError(f"adapter presence mismatch for {variant}")
        if bool(row.get("adapter_protocol_validated")) != requires_adapter:
            raise ValueError(f"adapter protocol validation mismatch for {variant}")
        claim_ready_sha256 = row.get("claim_ready_sha256")
        adapter_run_protocol = row.get("adapter_run_protocol")
        if requires_adapter:
            if not isinstance(claim_ready_sha256, str) or len(claim_ready_sha256) != 64:
                raise ValueError(f"claim-ready evidence mismatch for {variant}")
            if not isinstance(adapter_run_protocol, dict):
                raise ValueError(f"adapter run protocol is missing for {variant}")
        elif claim_ready_sha256 is not None or adapter_run_protocol is not None:
            raise ValueError(f"base arm {variant} must not carry claim-ready evidence")
        protocol = row.get("adapter_training_protocol")
        if requires_adapter:
            expected_protocol = {
                "objective": "positional_hidden_distillation",
                "seed": 42,
                "max_steps": 1 if method == "native_geo" else 300,
                "warmup_steps": 0 if method == "native_geo" else 30,
                "effective_batch_size": 8,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "lora_r": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.0,
                "lora_targets": ["q_proj", "k_proj"],
                "data_manifest_sha256": row.get("data_manifest_sha256"),
            }
            if protocol != expected_protocol:
                raise ValueError(f"adapter training protocol mismatch for {variant}")
        elif protocol is not None:
            raise ValueError(f"base arm {variant} must not carry adapter training metadata")
        if int(row.get("seed", -1)) != 42:
            raise ValueError(f"seed mismatch for {variant}")
        frequency = row.get("frequency_provenance", {})
        if frequency.get("method") != method or not frequency.get("tensor_sha256"):
            raise ValueError(f"frequency provenance mismatch for {variant}")

    if (
        rows["base_geo"]["frequency_provenance"]["tensor_sha256"]
        != rows["geo_distill_s42"]["frequency_provenance"]["tensor_sha256"]
        or rows["base_evq"]["frequency_provenance"]["tensor_sha256"]
        != rows["evq_distill_s42"]["frequency_provenance"]["tensor_sha256"]
    ):
        raise ValueError("base and adapted arms must share exact schedule tensors")

    adapted_execution_protocols = {
        json.dumps(rows[variant]["adapter_run_protocol"], sort_keys=True)
        for variant in ("geo_distill_s42", "evq_distill_s42")
    }
    if len(adapted_execution_protocols) != 1:
        raise ValueError("adapted arms must share one execution protocol and runtime")

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
    corpus_by_json = {
        json.dumps(row.get("eval_corpus"), sort_keys=True) for row in rows.values()
    }
    if len(corpus_by_json) != 1 or "null" in corpus_by_json:
        raise ValueError("all arms must share one evaluation corpus")
    tokenizer_by_json = {
        json.dumps(row.get("tokenizer_fingerprint"), sort_keys=True)
        for row in rows.values()
    }
    if len(tokenizer_by_json) != 1 or "null" in tokenizer_by_json:
        raise ValueError("all arms must share one tokenizer fingerprint")
    eval_config_by_json = {
        json.dumps(row.get("eval_config"), sort_keys=True) for row in rows.values()
    }
    if len(eval_config_by_json) != 1 or "null" in eval_config_by_json:
        raise ValueError("all arms must share one evaluation configuration")
    eval_config = json.loads(next(iter(eval_config_by_json)))
    try:
        validate_eval_claim_protocol(
            ppl_lengths=tuple(int(item) for item in eval_config["ppl_lengths"]),
            ppl_chunks=int(eval_config["ppl_chunks"]),
            hidden_sequences=int(eval_config["hidden_sequences"]),
            hidden_batch_size=int(eval_config["hidden_batch_size"]),
            bf16=eval_config["bf16"] is True,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("results do not use the approved evaluation protocol") from error
    code_by_json = {
        json.dumps(row.get("code_sha256"), sort_keys=True) for row in rows.values()
    }
    if len(code_by_json) != 1 or "null" in code_by_json:
        raise ValueError("all arms must share one evaluator code identity")

    base = rows["base_geo"]
    base_evq = rows["base_evq"]
    geo = rows["geo_distill_s42"]
    evq = rows["evq_distill_s42"]
    base_8k = _ppl(base, "8K")
    geo_8k_drift = (_ppl(geo, "8K") / base_8k - 1.0) * 100.0
    evq_8k_drift = (_ppl(evq, "8K") / base_8k - 1.0) * 100.0
    geo_16k_drift = (_ppl(geo, "16K") / _ppl(base, "16K") - 1.0) * 100.0
    geo_32k_drift = (_ppl(geo, "32K") / _ppl(base, "32K") - 1.0) * 100.0
    evq_16k_factor = _ppl(base, "16K") / _ppl(evq, "16K")
    evq_32k_factor = _ppl(base, "32K") / _ppl(evq, "32K")
    matched_16k_factor = _ppl(geo, "16K") / _ppl(evq, "16K")
    matched_32k_factor = _ppl(geo, "32K") / _ppl(evq, "32K")
    recovery = representation_recovery(
        injection_error=_hidden_error(base_evq),
        adapted_error=_hidden_error(evq),
    )

    metrics = {
        "geo_8k_drift_pct": geo_8k_drift,
        "evq_8k_drift_pct": evq_8k_drift,
        "geo_16k_drift_pct": geo_16k_drift,
        "geo_32k_drift_pct": geo_32k_drift,
        "evq_16k_improvement_factor": evq_16k_factor,
        "evq_32k_improvement_factor": evq_32k_factor,
        "matched_16k_improvement_factor": matched_16k_factor,
        "matched_32k_improvement_factor": matched_32k_factor,
        "base_evq_8k_drift_pct": (_ppl(base_evq, "8K") / base_8k - 1.0) * 100.0,
        "evq_distill_vs_base_evq_8k_pct": (
            _ppl(evq, "8K") / _ppl(base_evq, "8K") - 1.0
        )
        * 100.0,
        "representation_recovery": recovery,
        "base_evq_injection_error": _hidden_error(base_evq),
        "evq_distill_hidden_error": _hidden_error(evq),
    }
    if any(
        value is not None and not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ValueError("summary metrics must all be finite")
    gates = {
        "geo_null_within_1pct_at_all_lengths": all(
            abs(value) <= 1.0
            for value in (geo_8k_drift, geo_16k_drift, geo_32k_drift)
        ),
        "evq_8k_drift_at_most_5pct": evq_8k_drift <= 5.0,
        "matched_16k_improvement_at_least_2x": matched_16k_factor >= 2.0,
        "matched_32k_improvement_at_least_4x": matched_32k_factor >= 4.0,
        "representation_recovery_at_least_90pct": (
            recovery is not None and recovery >= 0.9
        ),
    }
    return {
        "format_version": 1,
        "protocol": "llama8b_positional_hidden_distillation_seed42",
        "model": next(iter(models)),
        "base_model_fingerprint": base_model_fingerprint,
        "eval_corpus": json.loads(next(iter(corpus_by_json))),
        "tokenizer_fingerprint": json.loads(next(iter(tokenizer_by_json))),
        "eval_config": eval_config,
        "code_sha256": json.loads(next(iter(code_by_json))),
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["gates"], indent=2, sort_keys=True))
    print(f"pilot_pass={summary['pilot_pass']}")


if __name__ == "__main__":
    main()
