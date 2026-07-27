#!/usr/bin/env python3
"""Fail-closed minimal exact gate for the OLMo hybrid-EVQ candidate.

This only admits the frozen adapter to expanded evaluation. It is not the
final long-context capability or 4K no-forgetting gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .prepare_data import atomic_json, sha256_file


MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
HYBRID_SHA256 = (
    "54b8d165a612bcd3e08e0bab0adf68264a39240a51af9115092a35ec9d5f0695"
)
QK_MASK_SHA256 = (
    "eba79ecb7c38fc22b8e8560a6e6413061b4fabb740390982bcbb63130b63ae48"
)
TRAINING_STATUS = "OLMO2_4K_HYBRID_EXACT_TRAINING_COMPLETE_V1"
EVALUATION_STATUS = "OLMO2_INSTRUCT_RULER_EXACT_SCREEN_COMPLETE_V2"
SUPERVISION_CONTRACT = "numeric_answer_plus_immediate_eos_v1"
SCREEN_POLICY = {
    4_096: {"examples": 8, "minimum_exact_generation_passes": 8},
    8_192: {"examples": 8, "minimum_exact_generation_passes": 6},
    16_384: {"examples": 8, "minimum_exact_generation_passes": 2},
}


def _exact_count(value: Any, examples: int, label: str) -> int:
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise RuntimeError(f"{label} is not a finite rate in [0, 1]")
    raw = rate * int(examples)
    count = int(round(raw))
    if not math.isclose(raw, count, abs_tol=1e-8):
        raise RuntimeError(f"{label} is not an exact count ratio")
    return count


def exact_screen_gate(
    training: dict[str, Any],
    evaluation: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    if training.get("status") != TRAINING_STATUS:
        raise RuntimeError("hybrid training receipt status drift")
    if evaluation.get("status") != EVALUATION_STATUS:
        raise RuntimeError("evaluation predates exact/EOS scoring")
    if (
        training.get("checkpoint_sha256") != MODEL_SHA256
        or evaluation.get("checkpoint_sha256") != MODEL_SHA256
    ):
        raise RuntimeError("gate is not bound to OLMo-2 1.485B")
    bound_code = training.get("bound_code", {})
    if (
        evaluation.get("script_sha256")
        != bound_code.get("exact_evaluator")
        or sha256_file(Path(__file__).resolve())
        != bound_code.get("exact_gate")
    ):
        raise RuntimeError("exact evaluator or gate changed after READY")
    for label, frequency in (
        ("training", training.get("frequency", {})),
        ("evaluation", evaluation.get("frequency", {})),
    ):
        if (
            frequency.get("active_frequency") != "hybrid_evq_low12"
            or frequency.get("active_sha256_float32") != HYBRID_SHA256
        ):
            raise RuntimeError(f"{label} hybrid frequency identity drift")

    protocol = training.get("protocol", {})
    required_protocol = {
        "model_scope": "OLMo-2 1.485B Instruct only",
        "frequency": "hybrid_evq_low12",
        "frequency_sha256_float32": HYBRID_SHA256,
        "adaptation": "qk_answer",
        "qk_output_mask_sha256": QK_MASK_SHA256,
        "v_and_o_trainable": False,
        "parent_adapter": None,
        "maximum_physical_training_sequence_length": 4_096,
        "hard_maximum_training_length": 4_096,
        "real_8k_or_16k_training_sequences": 0,
        "virtual_target_length": 16_384,
        "supervision_contract": SUPERVISION_CONTRACT,
        "supervision": (
            "causally_shifted_answer_ce_plus_weighted_immediate_eos_ce"
        ),
        "final_eos_supervised": True,
        "expanded_evaluation_before_minimal_exact_gate": False,
    }
    for name, expected in required_protocol.items():
        if protocol.get(name) != expected:
            raise RuntimeError(f"training protocol drift for {name}")

    routing = training.get("routing_data", {})
    required_routing = {
        "format_version": 2,
        "final_eos_supervised": True,
        "labels_only_cover_answer_and_final_eos": True,
        "answer_string_tokenizer_roundtrip_exact": True,
    }
    for name, expected in required_routing.items():
        if routing.get(name) != expected:
            raise RuntimeError(f"routing data drift for {name}")

    expected_metadata = {
        "base_checkpoint_sha256": MODEL_SHA256,
        "frequency": "hybrid_evq_low12",
        "frequency_sha256_float32": HYBRID_SHA256,
        "adaptation": "qk_answer",
        "qk_output_mask_sha256": QK_MASK_SHA256,
        "rank": 32,
        "alpha": 64.0,
        "training_sequence_length": 4_096,
        "maximum_physical_training_sequence_length": 4_096,
        "virtual_target_length": 16_384,
        "final_eos_supervised": True,
        "supervision_contract": SUPERVISION_CONTRACT,
        "parent_adapter_sha256": None,
    }
    training_metadata = training.get("adapter_metadata", {})
    for name, expected in expected_metadata.items():
        if training_metadata.get(name) != expected:
            raise RuntimeError(
                f"training adapter metadata drift for {name}"
            )
    adapter = evaluation.get("adapter")
    if not isinstance(adapter, dict):
        raise RuntimeError("exact screen did not load an adapter")
    if adapter.get("sha256") != training.get("adapter_sha256"):
        raise RuntimeError("training/evaluation adapter SHA mismatch")
    if adapter.get("qk_output_mask_sha256") != QK_MASK_SHA256:
        raise RuntimeError("evaluation Q/K mask identity drift")
    for name, expected in expected_metadata.items():
        if adapter.get("metadata", {}).get(name) != expected:
            raise RuntimeError(
                f"evaluated adapter metadata drift for {name}"
            )

    evaluation_protocol = evaluation.get("protocol", {})
    required_evaluation = {
        "task": "niah_single_1",
        "lengths": list(SCREEN_POLICY),
        "limit_per_length": 8,
        "greedy": True,
        "string_normalization": "none",
        "decode_cleanup": False,
        "terminal_eos_removed_before_string_decode": True,
        "other_special_tokens_removed": False,
        "substring_is_success": False,
        "first_number_is_success": False,
        "training_length_if_adapted": 4_096,
    }
    for name, expected in required_evaluation.items():
        if evaluation_protocol.get(name) != expected:
            raise RuntimeError(f"evaluation protocol drift for {name}")

    cells = evaluation.get("results", {}).get("cells", {})
    if set(cells) != {str(length) for length in SCREEN_POLICY}:
        raise RuntimeError("minimal exact-screen cells drift")
    checks: dict[str, bool] = {}
    counts: dict[str, dict[str, int]] = {}
    for length, policy in SCREEN_POLICY.items():
        cell = cells[str(length)]
        examples = int(cell.get("examples", -1))
        if examples != int(policy["examples"]):
            raise RuntimeError(f"L{length} row count drift")
        exact = _exact_count(
            cell.get("exact_generation_pass"),
            examples,
            f"L{length}.exact_generation_pass",
        )
        full = _exact_count(
            cell.get("full_string_exact"),
            examples,
            f"L{length}.full_string_exact",
        )
        eos = _exact_count(
            cell.get("eos_terminated"),
            examples,
            f"L{length}.eos_terminated",
        )
        token = _exact_count(
            cell.get("answer_eos_token_exact"),
            examples,
            f"L{length}.answer_eos_token_exact",
        )
        if exact > min(full, eos):
            raise RuntimeError(
                f"L{length} exact pass exceeds component metrics"
            )
        minimum = int(policy["minimum_exact_generation_passes"])
        checks[f"L{length}_exact_at_least_{minimum}_of_8"] = (
            exact >= minimum
        )
        counts[str(length)] = {
            "examples": examples,
            "exact_generation_passes": exact,
            "full_string_exact": full,
            "eos_terminated": eos,
            "answer_eos_token_exact": token,
            "minimum_exact_generation_passes": minimum,
        }
    passed = all(checks.values())
    return passed, {
        "purpose": (
            "admission to expanded evaluation only; not final capability "
            "or 4K no-catastrophic-forgetting evidence"
        ),
        "success_metric": (
            "literal whole decoded generated string exact with observed "
            "terminal EOS"
        ),
        "ignored_for_admission": [
            "first_number_exact",
            "official_string_match",
            "NLL",
            "PPL",
        ],
        "checks": checks,
        "counts": counts,
        "expanded_evaluation_authorized": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    training = json.loads(
        args.training_result.resolve().read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        args.evaluation_result.resolve().read_text(encoding="utf-8")
    )
    passed, details = exact_screen_gate(training, evaluation)
    receipt = {
        "status": "PASS" if passed else "STOP",
        "gate": "OLMO2_HYBRID_MINIMAL_FULL_STRING_EXACT_EOS_V1",
        **details,
    }
    atomic_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
