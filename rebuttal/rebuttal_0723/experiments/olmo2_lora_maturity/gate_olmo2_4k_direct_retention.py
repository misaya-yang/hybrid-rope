#!/usr/bin/env python3
"""Fail-closed direct-parent 4K retention gate for an OLMo-2 EOS repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .gate_olmo2_4k_retention import (
    EXACT_GATE_NAME,
    MODEL_SHA256,
    NATURAL_NLL_DELTA_CEILING,
    TRAINING_STATUS,
    _comparison_gate,
    _finite,
    _load_examples,
    _validate_pairing,
    _validate_result,
)
from .prepare_data import atomic_json, sha256_file


GATE_NAME = "OLMO2_4K_DIRECT_RETENTION_V1"
READY_STATUS = "OLMO2_4K_RETENTION_READY_V1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--exact-gate", type=Path, required=True)
    parser.add_argument(
        "--experiment-ready-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--expected-frequency",
        choices=("native", "evq"),
        required=True,
    )
    for label in ("candidate", "parent"):
        parser.add_argument(f"--{label}-result", type=Path, required=True)
        parser.add_argument(f"--{label}-examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    training = _read_json(args.training_result)
    exact_gate = _read_json(args.exact_gate)
    ready_path = args.experiment_ready_receipt.resolve()
    ready_sha256 = sha256_file(ready_path)
    ready = _read_json(ready_path)
    exact_pass = (
        exact_gate.get("status") == "PASS"
        and exact_gate.get("expanded_evaluation_authorized") is True
    )
    exact_stop_diagnostic = (
        ready.get("post_stop_retention_diagnostic_authorized") is True
        and exact_gate.get("status") == "STOP"
        and exact_gate.get("expanded_evaluation_authorized") is False
    )
    if (
        training.get("status") != TRAINING_STATUS
        or exact_gate.get("gate") != EXACT_GATE_NAME
        or not (exact_pass or exact_stop_diagnostic)
    ):
        raise RuntimeError("direct retention lacks a passing exact gate")
    if (
        ready.get("status") != READY_STATUS
        or ready.get("gate", {}).get("sha256")
        != sha256_file(Path(__file__).resolve())
        or ready["inputs"]["checkpoint"]["composite_sha256"]
        != MODEL_SHA256
    ):
        raise RuntimeError("direct retention READY drift")

    checkpoint = Path(ready["inputs"]["checkpoint"]["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )

    def decode_generated(token_ids: list[int]) -> str:
        return str(
            tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

    candidate_result = _read_json(args.candidate_result)
    parent_result = _read_json(args.parent_result)
    expected_roles = {
        "candidate": candidate_result,
        "immediate_parent": parent_result,
    }
    for role, result in expected_roles.items():
        if (
            result.get("experiment_ready_receipt_sha256") != ready_sha256
            or result.get("experiment_role") != role
            or result.get("script_sha256")
            != ready["evaluator"]["sha256"]
            or result.get("bound_code_sha256")
            != ready.get("evaluator_bound_code_sha256")
            or not result.get("run_manifest_sha256")
        ):
            raise RuntimeError(f"{role} direct-retention binding drift")

    candidate_sha256 = str(training.get("adapter_sha256"))
    parent_sha256 = str(training.get("parent_adapter_sha256"))
    if len(candidate_sha256) != 64 or len(parent_sha256) != 64:
        raise RuntimeError("candidate/parent adapter SHA drift")

    candidate_rows = _load_examples(
        candidate_result,
        args.candidate_examples.resolve(),
        label="candidate",
    )
    parent_rows = _load_examples(
        parent_result,
        args.parent_examples.resolve(),
        label="immediate_parent",
    )
    candidate_scores = _validate_result(
        candidate_result,
        candidate_rows,
        adapter_sha256=candidate_sha256,
        label="candidate",
        decode_generated=decode_generated,
        expected_frequency=str(args.expected_frequency),
    )
    parent_scores = _validate_result(
        parent_result,
        parent_rows,
        adapter_sha256=parent_sha256,
        label="immediate_parent",
        decode_generated=decode_generated,
        expected_frequency=str(args.expected_frequency),
    )
    if (
        candidate_result["adapter"]["metadata"].get(
            "parent_adapter_sha256"
        )
        != parent_sha256
        or parent_result.get("data") != candidate_result.get("data")
        or parent_result.get("protocol") != candidate_result.get("protocol")
    ):
        raise RuntimeError("direct candidate/parent lineage or protocol drift")
    _validate_pairing(
        candidate_rows,
        parent_rows,
        label="candidate_vs_direct_parent",
    )

    task_passed, task_details = _comparison_gate(
        candidate_scores,
        parent_scores,
        label="direct_parent_retention",
    )
    natural = training.get("natural_nll", {})
    baseline = natural.get("baseline", {}).get("cells", {}).get("L4096", {})
    candidate_nll = (
        natural.get("candidate", {}).get("cells", {}).get("L4096", {})
    )
    if (
        natural.get("baseline", {}).get("adapter_sha256") != parent_sha256
        or int(baseline.get("rows", -1))
        != int(candidate_nll.get("rows", -2))
        or int(baseline.get("tokens", -1))
        != int(candidate_nll.get("tokens", -2))
        or int(baseline.get("tail_tokens_per_row", -1))
        != int(candidate_nll.get("tail_tokens_per_row", -2))
    ):
        raise RuntimeError("matched direct-parent natural-NLL drift")
    mean_delta = _finite(
        candidate_nll.get("mean_nll"), "candidate natural mean NLL"
    ) - _finite(baseline.get("mean_nll"), "parent natural mean NLL")
    tail_delta = _finite(
        candidate_nll.get("tail_mean_nll"),
        "candidate natural tail NLL",
    ) - _finite(
        baseline.get("tail_mean_nll"),
        "parent natural tail NLL",
    )
    natural_passed = (
        mean_delta <= NATURAL_NLL_DELTA_CEILING + 1e-12
        and tail_delta <= NATURAL_NLL_DELTA_CEILING + 1e-12
    )
    passed = task_passed and natural_passed
    receipt = {
        "status": "PASS" if passed else "STOP",
        "gate": GATE_NAME,
        "scope": (
            "fixed 13-task 4K RULER matrix and matched 4K natural NLL "
            "against the candidate's direct parent"
        ),
        "direct_parent_retention": task_details,
        "natural_nll_vs_direct_parent": {
            "mean_nll_delta": mean_delta,
            "tail_mean_nll_delta": tail_delta,
            "ceiling": NATURAL_NLL_DELTA_CEILING,
            "passed": natural_passed,
        },
        "overall_4k_retention_pass": passed,
        "capability_gate_status": exact_gate.get("status"),
        "post_stop_retention_diagnostic": exact_stop_diagnostic,
    }
    atomic_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(21)


if __name__ == "__main__":
    main()
