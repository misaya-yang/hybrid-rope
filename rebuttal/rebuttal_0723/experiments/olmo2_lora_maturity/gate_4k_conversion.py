#!/usr/bin/env python3
"""Fail-closed admission gates for the single 4K EVQ conversion arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def read_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def check_same_candidate(
    left: dict[str, Any], right: dict[str, Any]
) -> None:
    if left.get("checkpoint_sha256") != right.get("checkpoint_sha256"):
        raise RuntimeError("gate inputs use different base checkpoints")
    left_frequency = left.get("frequency", {})
    right_frequency = right.get("frequency", {})
    if (
        left_frequency.get("active_sha256_float32")
        != right_frequency.get("active_sha256_float32")
    ):
        raise RuntimeError("gate inputs use different frequency vectors")


def natural_metric(
    result: dict[str, Any],
    *,
    length: int,
    key: str = "tail_mean_nll",
) -> float:
    return finite(
        result["natural_nll"][f"L{int(length)}"][key],
        f"L{int(length)} {key}",
    )


def stage_a_gate(
    base: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    check_same_candidate(base, candidate)
    if base.get("mode") != "base" or candidate.get("mode") != "train":
        raise RuntimeError("Stage-A gate expects base and train receipts")
    if candidate.get("adaptation") != "qkvo_answer":
        raise RuntimeError("Stage-A candidate is not QKVO LoRA")
    training = candidate.get("training")
    if not isinstance(training, dict):
        raise RuntimeError("Stage-A candidate lacks a completed training receipt")
    if int(training.get("training_sequence_length", -1)) != 4_096:
        raise RuntimeError("Stage-A exceeded the 4K training contract")
    if int(training.get("actual_supervised_tokens", 0)) < 20_000_000:
        raise RuntimeError("Stage-A consumed fewer than 20M supervised tokens")
    throughput = finite(training["tokens_per_second"], "Stage-A throughput")
    if throughput <= 0:
        raise RuntimeError("Stage-A throughput is not positive")

    base_16k = natural_metric(base, length=16_384)
    candidate_16k = natural_metric(candidate, length=16_384)
    base_4k = natural_metric(base, length=4_096)
    candidate_4k = natural_metric(candidate, length=4_096)
    improvement_16k = base_16k - candidate_16k
    regression_4k = candidate_4k - base_4k
    checks = {
        "16k_tail_nll_improvement_ge_0p2": improvement_16k >= 0.2,
        "4k_tail_nll_regression_le_0p15": regression_4k <= 0.15,
        "finite_positive_throughput": throughput > 0,
    }
    return all(checks.values()), {
        "checks": checks,
        "metrics": {
            "base_16k_tail_nll": base_16k,
            "candidate_16k_tail_nll": candidate_16k,
            "improvement_16k_tail_nll": improvement_16k,
            "base_4k_tail_nll": base_4k,
            "candidate_4k_tail_nll": candidate_4k,
            "regression_4k_tail_nll": regression_4k,
            "tokens_per_second": throughput,
        },
    }


def binding_metrics(result: dict[str, Any], key: str) -> dict[str, float]:
    metrics = result.get(key)
    if not isinstance(metrics, dict):
        raise RuntimeError(f"binding receipt lacks {key}")
    return {
        name: finite(metrics[name], f"{key}.{name}")
        for name in (
            "full_vocab_exact",
            "mean_nll",
            "mean_source_deletion_nll_gap",
            "swap_follow_positive_fraction",
            "mean_swap_follow_score",
        )
    }


def stage_b1_gate(candidate: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if candidate.get("status") != "OLMO2_4K_STAGE_B1_COMPLETE":
        raise RuntimeError("Stage-B1 receipt status mismatch")
    if int(candidate["protocol"]["hard_maximum_training_length"]) != 4_096:
        raise RuntimeError("Stage-B1 exceeded the 4K training contract")
    metrics = binding_metrics(candidate, "binding_validation")
    checks = {
        "calibration_full_vocab_exact_ge_0p50": (
            metrics["full_vocab_exact"] >= 0.50
        ),
        "calibration_source_deletion_gap_positive": (
            metrics["mean_source_deletion_nll_gap"] > 0.0
        ),
        "calibration_swap_follow_fraction_ge_0p80": (
            metrics["swap_follow_positive_fraction"] >= 0.80
        ),
        "calibration_swap_follow_score_positive": (
            metrics["mean_swap_follow_score"] > 0.0
        ),
    }
    return all(checks.values()), {"checks": checks, "metrics": metrics}


def stage_b2_gate(
    stage_a: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    if candidate.get("status") != "OLMO2_4K_STAGE_B2_COMPLETE":
        raise RuntimeError("Stage-B2 receipt status mismatch")
    if int(candidate["protocol"]["hard_maximum_training_length"]) != 4_096:
        raise RuntimeError("Stage-B2 exceeded the 4K training contract")
    validation = binding_metrics(candidate, "binding_validation")
    final_test = binding_metrics(candidate, "binding_final_test")
    stage_a_4k = natural_metric(stage_a, length=4_096)
    stage_b2_4k = natural_metric(candidate, length=4_096)
    regression_4k = stage_b2_4k - stage_a_4k
    checks = {
        "validation_full_vocab_exact_ge_0p05": (
            validation["full_vocab_exact"] >= 0.05
        ),
        "final_full_vocab_exact_ge_0p05": (
            final_test["full_vocab_exact"] >= 0.05
        ),
        "validation_source_deletion_gap_positive": (
            validation["mean_source_deletion_nll_gap"] > 0.0
        ),
        "final_source_deletion_gap_positive": (
            final_test["mean_source_deletion_nll_gap"] > 0.0
        ),
        "validation_swap_follow_fraction_gt_0p50": (
            validation["swap_follow_positive_fraction"] > 0.50
        ),
        "final_swap_follow_fraction_gt_0p50": (
            final_test["swap_follow_positive_fraction"] > 0.50
        ),
        "4k_tail_nll_regression_le_0p10_from_stage_a": (
            regression_4k <= 0.10
        ),
    }
    return all(checks.values()), {
        "checks": checks,
        "metrics": {
            "validation": validation,
            "final_test": final_test,
            "stage_a_4k_tail_nll": stage_a_4k,
            "stage_b2_4k_tail_nll": stage_b2_4k,
            "regression_4k_tail_nll": regression_4k,
        },
    }


def causal_overall(result: dict[str, Any], set_name: str) -> dict[str, float]:
    overall = result["results"][set_name]["summary"]["overall"]
    return {
        name: finite(overall[name], f"{set_name}.{name}")
        for name in (
            "mean_answer_nll",
            "next_token_exact_match",
            "mean_source_deletion_nll_gap",
            "mean_swap_follow_score",
            "swap_follow_positive_fraction",
        )
    }


def causal_gate(
    base: dict[str, Any],
    candidate: dict[str, Any],
    set_name: str,
) -> tuple[bool, dict[str, Any]]:
    check_same_candidate(base, candidate)
    if candidate.get("adapter") is None or base.get("adapter") is not None:
        raise RuntimeError("causal gate expects base then adapted receipt")
    baseline = causal_overall(base, set_name)
    adapted = causal_overall(candidate, set_name)
    exact_gain = (
        adapted["next_token_exact_match"]
        - baseline["next_token_exact_match"]
    )
    nll_gain = baseline["mean_answer_nll"] - adapted["mean_answer_nll"]
    checks = {
        "16k_exact_strictly_improves": exact_gain > 0.0,
        "16k_exact_at_least_0p05": (
            adapted["next_token_exact_match"] >= 0.05
        ),
        "16k_answer_nll_improves": nll_gain > 0.0,
        "16k_source_deletion_gap_positive": (
            adapted["mean_source_deletion_nll_gap"] > 0.0
        ),
        "16k_swap_follow_score_positive": (
            adapted["mean_swap_follow_score"] > 0.0
        ),
        "16k_swap_follow_fraction_gt_0p50": (
            adapted["swap_follow_positive_fraction"] > 0.50
        ),
    }
    return all(checks.values()), {
        "checks": checks,
        "metrics": {
            "set": set_name,
            "baseline": baseline,
            "adapted": adapted,
            "exact_gain": exact_gain,
            "answer_nll_gain": nll_gain,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="gate", required=True)

    stage_a = subparsers.add_parser("stage-a")
    stage_a.add_argument("--base", type=Path, required=True)
    stage_a.add_argument("--candidate", type=Path, required=True)
    stage_a.add_argument("--output", type=Path, required=True)

    stage_b1 = subparsers.add_parser("stage-b1")
    stage_b1.add_argument("--candidate", type=Path, required=True)
    stage_b1.add_argument("--output", type=Path, required=True)

    stage_b2 = subparsers.add_parser("stage-b2")
    stage_b2.add_argument("--stage-a", type=Path, required=True)
    stage_b2.add_argument("--candidate", type=Path, required=True)
    stage_b2.add_argument("--output", type=Path, required=True)

    causal = subparsers.add_parser("causal")
    causal.add_argument("--base", type=Path, required=True)
    causal.add_argument("--candidate", type=Path, required=True)
    causal.add_argument("--set", default="canary_full_heldout")
    causal.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.gate == "stage-a":
        passed, details = stage_a_gate(
            read_result(args.base.resolve()),
            read_result(args.candidate.resolve()),
        )
    elif args.gate == "stage-b1":
        passed, details = stage_b1_gate(
            read_result(args.candidate.resolve())
        )
    elif args.gate == "stage-b2":
        passed, details = stage_b2_gate(
            read_result(args.stage_a.resolve()),
            read_result(args.candidate.resolve()),
        )
    elif args.gate == "causal":
        passed, details = causal_gate(
            read_result(args.base.resolve()),
            read_result(args.candidate.resolve()),
            str(args.set),
        )
    else:
        raise AssertionError(args.gate)

    receipt = {
        "status": "PASS" if passed else "STOP",
        "gate": args.gate,
        **details,
    }
    write_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
