#!/usr/bin/env python3
"""Fail-closed minimal exact-generation gate for OLMo-2 1.485B.

This gate is only an admission check for expanded evaluation. It is not the
final capability or no-forgetting claim. Substring, first-number, NLL, and PPL
fields are deliberately ignored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    row_sha256,
)
from .prepare_data import atomic_json, sha256_file


MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
EVQ_SHA256 = (
    "917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607"
)
TRAINING_STATUS = "OLMO2_4K_QUERY_GAP_EOS_REPAIR_COMPLETE_V1"
EVALUATION_STATUS = "OLMO2_INSTRUCT_RULER_EXACT_SCREEN_COMPLETE_V2"
SUPERVISION_CONTRACT = "numeric_answer_plus_immediate_eos_v1"
OLMO2_EOS_TOKEN_ID = 100_257
EXAMPLE_SCHEMA_VERSION = 2
RUN_MANIFEST_STATUS = "OLMO2_INSTRUCT_RULER_EXACT_SCREEN_RUN_V2"
GENERATION_TOKENS = 128
SCREEN_POLICY = {
    4_096: {"examples": 8, "minimum_exact_generation_passes": 8},
    8_192: {"examples": 8, "minimum_exact_generation_passes": 6},
    16_384: {"examples": 8, "minimum_exact_generation_passes": 2},
}


def bound_code_sha256() -> dict[str, str]:
    gate = Path(__file__).resolve()
    maturity_root = gate.parent
    experiments_root = maturity_root.parent
    paths = {
        "gate": gate,
        "row_identity": (
            experiments_root / "olmo2_1b_evq" / "evaluate_ruler.py"
        ),
        "receipt_utils": maturity_root / "prepare_data.py",
    }
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def _exact_count(rate: Any, examples: int, label: str) -> int:
    value = _finite(rate, label)
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"{label} is outside [0, 1]")
    raw_count = value * int(examples)
    count = int(round(raw_count))
    if not math.isclose(raw_count, count, abs_tol=1e-8):
        raise RuntimeError(f"{label} is not an exact count ratio")
    return count


def _recompute_raw_counts(
    *,
    evaluation: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    source_rows: dict[tuple[int, int], dict[str, Any]],
    decode_generated: Callable[[list[int]], str],
    policy: dict[int, dict[str, int]] = SCREEN_POLICY,
) -> dict[str, dict[str, int]]:
    cells = evaluation.get("results", {}).get("cells", {})
    if set(cells) != {str(length) for length in policy}:
        raise RuntimeError("exact-screen length cells drift")
    expected_keys = {
        (length, local_index)
        for length in policy
        for local_index in range(int(policy[length]["examples"]))
    }
    observed_rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in raw_rows:
        key = (
            int(row.get("nominal_length", -1)),
            int(row.get("local_index", -1)),
        )
        if key in observed_rows:
            raise RuntimeError(f"duplicate exact-screen raw row: {key}")
        observed_rows[key] = row
    if set(observed_rows) != expected_keys:
        raise RuntimeError("exact-screen raw row set drift")
    if set(source_rows) != expected_keys:
        raise RuntimeError("exact-screen frozen source row set drift")

    counts: dict[str, dict[str, int]] = {}
    for length, length_policy in policy.items():
        cell = cells[str(length)]
        examples = int(cell.get("examples", -1))
        if examples != int(length_policy["examples"]):
            raise RuntimeError(f"L{length} screen row count drift")
        receipt_exact_count = _exact_count(
            cell.get("exact_generation_pass"),
            examples,
            f"L{length}.exact_generation_pass",
        )
        receipt_full_count = _exact_count(
            cell.get("full_string_exact"),
            examples,
            f"L{length}.full_string_exact",
        )
        receipt_eos_count = _exact_count(
            cell.get("eos_terminated"),
            examples,
            f"L{length}.eos_terminated",
        )
        token_count = _exact_count(
            cell.get("answer_eos_token_exact"),
            examples,
            f"L{length}.answer_eos_token_exact",
        )
        raw_exact_count = 0
        raw_full_count = 0
        raw_eos_count = 0
        for local_index in range(examples):
            row = observed_rows[(length, local_index)]
            source = source_rows[(length, local_index)]
            prediction = row.get("prediction")
            references = row.get("references")
            token_ids = row.get("generated_token_ids")
            source_references = [
                str(value) for value in source["outputs"]
            ]
            if (
                int(row.get("schema_version", -1))
                != EXAMPLE_SCHEMA_VERSION
                or row.get("task") != "niah_single_1"
                or not isinstance(prediction, str)
                or not isinstance(references, list)
                or not references
                or not all(isinstance(value, str) for value in references)
                or not isinstance(token_ids, list)
                or not all(isinstance(value, int) for value in token_ids)
                or len(token_ids) > GENERATION_TOKENS
                or int(row.get("generated_tokens", -1)) != len(token_ids)
                or OLMO2_EOS_TOKEN_ID in token_ids[:-1]
                or references != source_references
                or int(row.get("source_row_index", -1))
                != int(source["index"])
                or int(row.get("source_token_position_answer", -1))
                != int(source["token_position_answer"])
                or row.get("row_sha256") != row_sha256(source)
            ):
                raise RuntimeError(
                    f"L{length}/row{local_index} raw payload drift"
                )
            decoded_prediction = decode_generated(token_ids)
            if prediction != decoded_prediction:
                raise RuntimeError(
                    f"L{length}/row{local_index} raw decode drift"
                )
            full = decoded_prediction in source_references
            eos = bool(
                token_ids
                and int(token_ids[-1]) == OLMO2_EOS_TOKEN_ID
            )
            exact = full and eos
            for name, expected in (
                ("full_string_exact", full),
                ("eos_terminated", eos),
                ("exact_generation_pass", exact),
            ):
                observed = _finite(
                    row.get(name),
                    f"L{length}/row{local_index}.{name}",
                )
                if observed != float(expected):
                    raise RuntimeError(
                        f"L{length}/row{local_index} {name} drift"
                    )
            raw_full_count += int(full)
            raw_eos_count += int(eos)
            raw_exact_count += int(exact)
        if (
            receipt_exact_count != raw_exact_count
            or receipt_full_count != raw_full_count
            or receipt_eos_count != raw_eos_count
        ):
            raise RuntimeError(
                f"L{length} receipt/raw exact component drift"
            )
        counts[str(length)] = {
            "examples": examples,
            "exact_generation_passes": raw_exact_count,
            "full_string_exact": raw_full_count,
            "eos_terminated": raw_eos_count,
            "answer_eos_token_exact": token_count,
        }
    return counts


def exact_screen_gate(
    training: dict[str, Any],
    evaluation: dict[str, Any],
    parent_evaluation: dict[str, Any],
    *,
    expected_parent_sha256: str,
    raw_rows: list[dict[str, Any]],
    run_manifest: dict[str, Any],
    parent_raw_rows: list[dict[str, Any]],
    parent_run_manifest: dict[str, Any],
    source_rows: dict[tuple[int, int], dict[str, Any]],
    decode_generated: Callable[[list[int]], str],
    experiment_ready: dict[str, Any],
    experiment_ready_sha256: str,
    gate_script_sha256: str,
) -> tuple[bool, dict[str, Any]]:
    if training.get("status") != TRAINING_STATUS:
        raise RuntimeError("training receipt status drift")
    if (
        evaluation.get("status") != EVALUATION_STATUS
        or parent_evaluation.get("status") != EVALUATION_STATUS
    ):
        raise RuntimeError("evaluation receipt predates exact/EOS scoring")
    if (
        training.get("checkpoint_sha256") != MODEL_SHA256
        or evaluation.get("checkpoint_sha256") != MODEL_SHA256
        or parent_evaluation.get("checkpoint_sha256") != MODEL_SHA256
    ):
        raise RuntimeError("gate is not bound to OLMo-2 1.485B")
    if (
        len(expected_parent_sha256) != 64
        or training.get("parent_adapter_sha256")
        != expected_parent_sha256
        or not training.get("script_sha256")
        or not training.get("experiment_ready_receipt_sha256")
        or not evaluation.get("script_sha256")
        or not evaluation.get("run_manifest_sha256")
        or not parent_evaluation.get("run_manifest_sha256")
    ):
        raise RuntimeError(
            "gate lacks immutable parent, READY, or evaluator bindings"
        )
    if (
        experiment_ready.get("status")
        != "OLMO2_4K_QUERY_GAP_EOS_REPAIR_READY_V1"
        or training.get("experiment_ready_receipt_sha256")
        != experiment_ready_sha256
        or evaluation.get("experiment_ready_receipt_sha256")
        != experiment_ready_sha256
        or parent_evaluation.get("experiment_ready_receipt_sha256")
        != experiment_ready_sha256
        or experiment_ready["trainer"]["sha256"]
        != training.get("script_sha256")
        or experiment_ready["evaluator"]["sha256"]
        != evaluation.get("script_sha256")
        or experiment_ready["gate"]["sha256"] != gate_script_sha256
        or experiment_ready.get("gate_bound_code_sha256")
        != bound_code_sha256()
        or experiment_ready.get("bound_code_sha256")
        != training.get("bound_code_sha256")
        or experiment_ready.get("evaluator_bound_code_sha256")
        != evaluation.get("bound_code_sha256")
        or experiment_ready.get("evaluator_bound_code_sha256")
        != parent_evaluation.get("bound_code_sha256")
        or experiment_ready.get("protocol") != training.get("protocol")
        or experiment_ready["inputs"]["parent_adapter"]["sha256"]
        != expected_parent_sha256
    ):
        raise RuntimeError("training/evaluation/READY binding drift")

    training_frequency = training.get("frequency", {})
    evaluation_frequency = evaluation.get("frequency", {})
    parent_evaluation_frequency = parent_evaluation.get("frequency", {})
    for label, frequency in (
        ("training", training_frequency),
        ("evaluation", evaluation_frequency),
        ("parent_evaluation", parent_evaluation_frequency),
    ):
        if (
            frequency.get("active_frequency") != "evq_endpoint_cosh"
            or frequency.get("active_sha256_float32") != EVQ_SHA256
        ):
            raise RuntimeError(f"{label} is not the frozen EVQ table")

    protocol = training.get("protocol", {})
    if (
        int(protocol.get("maximum_physical_training_sequence_length", -1))
        != 4_096
        or int(protocol.get("hard_maximum_training_length", -1)) != 4_096
        or int(protocol.get("virtual_target_length", -1)) != 16_384
        or int(protocol.get("maximum_physical_token_index", -1)) != 4_095
        or int(protocol.get("maximum_allowed_position_id", -1))
        != 16_383
        or protocol.get("position_policy")
        != "semantic_query_block_continuous_gap"
        or protocol.get("supervision_contract")
        != SUPERVISION_CONTRACT
        or protocol.get("supervision")
        != "answer_ce_plus_weighted_immediate_eos_ce"
        or int(protocol.get("eos_token_id", -1))
        != OLMO2_EOS_TOKEN_ID
        or protocol.get("final_eos_supervised") is not True
        or protocol.get("counterfactual_margin_scope")
        != "answer_tokens_where_gold_differs"
    ):
        raise RuntimeError("training violates the 4K/EOS/query-gap contract")
    observed_maximum_position = int(
        training.get("training", {}).get(
            "maximum_observed_position_id", -1
        )
    )
    if not 12_288 < observed_maximum_position <= 16_383:
        raise RuntimeError("training did not realize the registered 16K band")
    routing_data = training.get("routing_data", {})
    if (
        int(routing_data.get("format_version", -1)) != 2
        or routing_data.get("supervision_contract")
        != SUPERVISION_CONTRACT
        or int(routing_data.get("eos_token_id", -1))
        != OLMO2_EOS_TOKEN_ID
        or routing_data.get("final_eos_supervised") is not True
        or routing_data.get(
            "labels_only_cover_answer_and_final_eos"
        )
        is not True
        or routing_data.get(
            "answer_string_tokenizer_roundtrip_exact"
        )
        is not True
    ):
        raise RuntimeError("training used routing data without final EOS")

    adapter = evaluation.get("adapter")
    if not isinstance(adapter, dict):
        raise RuntimeError("exact screen did not load the trained adapter")
    if adapter.get("sha256") != training.get("adapter_sha256"):
        raise RuntimeError("training/evaluation adapter SHA mismatch")
    adapter_metadata = adapter.get("metadata", {})
    if (
        adapter_metadata.get("base_checkpoint_sha256") != MODEL_SHA256
        or adapter_metadata.get("frequency") != "evq"
        or adapter_metadata.get("frequency_sha256_float32") != EVQ_SHA256
        or adapter_metadata.get("adaptation") != "qkvo_answer"
        or int(adapter_metadata.get("training_sequence_length", -1))
        != 4_096
        or adapter_metadata.get("final_eos_supervised") is not True
        or adapter_metadata.get("stage")
        != "counterfactual_routing_semantic_query_gap_16k_eos_v2"
        or adapter_metadata.get("parent_adapter_sha256")
        != expected_parent_sha256
        or adapter_metadata.get("routing_data_sha256")
        != routing_data.get("manifest_sha256")
        or adapter_metadata.get("position_policy")
        != "semantic_query_block_continuous_gap"
        or adapter_metadata.get("supervision_contract")
        != SUPERVISION_CONTRACT
        or int(adapter_metadata.get("eos_token_id", -1))
        != OLMO2_EOS_TOKEN_ID
    ):
        raise RuntimeError("evaluated adapter metadata violates the contract")
    parent_adapter = parent_evaluation.get("adapter")
    if (
        not isinstance(parent_adapter, dict)
        or parent_adapter.get("sha256") != expected_parent_sha256
    ):
        raise RuntimeError("parent exact baseline adapter SHA drift")
    parent_metadata = parent_adapter.get("metadata", {})
    if (
        parent_metadata.get("base_checkpoint_sha256") != MODEL_SHA256
        or parent_metadata.get("frequency") != "evq"
        or parent_metadata.get("frequency_sha256_float32") != EVQ_SHA256
        or parent_metadata.get("adaptation") != "qkvo_answer"
        or int(parent_metadata.get("rank", -1)) != 64
        or float(parent_metadata.get("alpha", -1.0)) != 128.0
        or int(parent_metadata.get("training_sequence_length", -1))
        != 4_096
    ):
        raise RuntimeError("parent exact baseline metadata drift")

    evaluation_protocol = evaluation.get("protocol", {})
    expected_lengths = list(SCREEN_POLICY)
    if (
        evaluation_protocol.get("task") != "niah_single_1"
        or evaluation_protocol.get("lengths") != expected_lengths
        or int(evaluation_protocol.get("limit_per_length", -1)) != 8
        or evaluation_protocol.get("greedy") is not True
        or evaluation_protocol.get("string_normalization") != "none"
        or evaluation_protocol.get("decode_cleanup") is not False
        or evaluation_protocol.get(
            "terminal_eos_removed_before_string_decode"
        )
        is not True
        or evaluation_protocol.get("other_special_tokens_removed") is not False
        or evaluation_protocol.get("substring_is_success") is not False
        or evaluation_protocol.get("first_number_is_success") is not False
        or int(evaluation_protocol.get("maximum_new_tokens", -1))
        != GENERATION_TOKENS
        or int(
            evaluation_protocol.get("training_length_if_adapted", -1)
        )
        != 4_096
    ):
        raise RuntimeError("evaluation violates the minimal exact contract")
    if (
        parent_evaluation.get("protocol") != evaluation_protocol
        or parent_evaluation.get("data") != evaluation.get("data")
    ):
        raise RuntimeError("parent/candidate exact evaluation contract drift")
    if (
        run_manifest.get("status") != RUN_MANIFEST_STATUS
        or int(run_manifest.get("example_schema_version", -1))
        != EXAMPLE_SCHEMA_VERSION
        or run_manifest.get("checkpoint_sha256") != MODEL_SHA256
        or run_manifest.get("experiment_ready_receipt_sha256")
        != experiment_ready_sha256
        or run_manifest.get("experiment_role")
        != "candidate_exact_screen"
        or run_manifest.get("bound_code_sha256")
        != experiment_ready.get("evaluator_bound_code_sha256")
        or run_manifest.get("data_manifest_sha256")
        != evaluation.get("data", {}).get("manifest_sha256")
        or run_manifest.get("frequency") != "evq"
        or run_manifest.get("adapter_sha256")
        != training.get("adapter_sha256")
        or run_manifest.get("adaptation") != "qkvo_answer"
        or int(run_manifest.get("rank", -1)) != 64
        or float(run_manifest.get("alpha", -1.0)) != 128.0
        or run_manifest.get("task") != "niah_single_1"
        or run_manifest.get("lengths") != expected_lengths
        or int(run_manifest.get("limit_per_length", -1)) != 8
        or run_manifest.get("greedy") is not True
        or int(run_manifest.get("maximum_new_tokens", -1))
        != GENERATION_TOKENS
        or run_manifest.get("string_normalization") != "none"
        or run_manifest.get("decode_cleanup") is not False
        or run_manifest.get(
            "terminal_eos_removed_before_string_decode"
        )
        is not True
        or run_manifest.get("other_special_tokens_removed") is not False
    ):
        raise RuntimeError("exact-screen run manifest violates the contract")
    expected_parent_manifest = {
        **run_manifest,
        "adapter_sha256": expected_parent_sha256,
        "experiment_role": "parent_exact_baseline",
    }
    if parent_run_manifest != expected_parent_manifest:
        raise RuntimeError("parent exact run manifest is not candidate-matched")

    parent_counts = _recompute_raw_counts(
        evaluation=parent_evaluation,
        raw_rows=parent_raw_rows,
        source_rows=source_rows,
        decode_generated=decode_generated,
    )
    counts = _recompute_raw_counts(
        evaluation=evaluation,
        raw_rows=raw_rows,
        source_rows=source_rows,
        decode_generated=decode_generated,
    )
    checks: dict[str, bool] = {}
    for length, policy in SCREEN_POLICY.items():
        examples = int(counts[str(length)]["examples"])
        raw_exact_count = int(
            counts[str(length)]["exact_generation_passes"]
        )
        minimum = int(policy["minimum_exact_generation_passes"])
        checks[f"L{length}_exact_generation_at_least_{minimum}_of_8"] = (
            raw_exact_count >= minimum
        )
        counts[str(length)]["minimum_exact_generation_passes"] = minimum

    passed = all(checks.values())
    return passed, {
        "purpose": (
            "admission to expanded evaluation only; not final capability "
            "or no-catastrophic-forgetting evidence"
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
        "parent_baseline_counts": parent_counts,
        "counts": counts,
        "expanded_evaluation_authorized": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument(
        "--experiment-ready-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--parent-evaluation-result", type=Path, required=True
    )
    parser.add_argument(
        "--parent-evaluation-examples", type=Path, required=True
    )
    parser.add_argument(
        "--parent-evaluation-run-manifest", type=Path, required=True
    )
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--evaluation-examples", type=Path, required=True)
    parser.add_argument("--evaluation-run-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
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
    experiment_ready_path = args.experiment_ready_receipt.resolve()
    experiment_ready_sha256 = sha256_file(experiment_ready_path)
    experiment_ready = json.loads(
        experiment_ready_path.read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        args.evaluation_result.resolve().read_text(encoding="utf-8")
    )
    parent_evaluation = json.loads(
        args.parent_evaluation_result.resolve().read_text(encoding="utf-8")
    )
    parent_examples_path = args.parent_evaluation_examples.resolve()
    if (
        sha256_file(parent_examples_path)
        != parent_evaluation.get("results", {}).get("examples_sha256")
    ):
        raise RuntimeError("parent evaluation examples SHA drift")
    parent_raw_rows = [
        json.loads(line)
        for line in parent_examples_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    parent_run_manifest_path = (
        args.parent_evaluation_run_manifest.resolve()
    )
    if (
        sha256_file(parent_run_manifest_path)
        != parent_evaluation.get("run_manifest_sha256")
    ):
        raise RuntimeError("parent evaluation run-manifest SHA drift")
    parent_run_manifest = json.loads(
        parent_run_manifest_path.read_text(encoding="utf-8")
    )
    parent_baseline_binding = training.get("parent_exact_baseline", {})
    if (
        parent_baseline_binding.get("result_sha256")
        != sha256_file(args.parent_evaluation_result.resolve())
        or parent_baseline_binding.get("examples_sha256")
        != sha256_file(parent_examples_path)
        or parent_baseline_binding.get("run_manifest_sha256")
        != sha256_file(parent_run_manifest_path)
        or parent_baseline_binding.get("adapter_sha256")
        != str(args.expected_parent_sha256)
    ):
        raise RuntimeError("training did not consume this parent baseline")
    examples_path = args.evaluation_examples.resolve()
    if (
        sha256_file(examples_path)
        != evaluation.get("results", {}).get("examples_sha256")
    ):
        raise RuntimeError("evaluation examples SHA drift")
    raw_rows = [
        json.loads(line)
        for line in examples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_manifest_path = args.evaluation_run_manifest.resolve()
    if (
        sha256_file(run_manifest_path)
        != evaluation.get("run_manifest_sha256")
    ):
        raise RuntimeError("evaluation run-manifest SHA drift")
    run_manifest = json.loads(
        run_manifest_path.read_text(encoding="utf-8")
    )
    checkpoint = Path(
        experiment_ready["inputs"]["checkpoint"]["path"]
    ).resolve()
    if (
        experiment_ready["inputs"]["checkpoint"]["composite_sha256"]
        != MODEL_SHA256
    ):
        raise RuntimeError("READY checkpoint is not OLMo-2 1.485B")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if int(tokenizer.eos_token_id) != OLMO2_EOS_TOKEN_ID:
        raise RuntimeError("gate tokenizer EOS drift")

    data_root = Path(
        experiment_ready["inputs"]["exact_eval_data"]["path"]
    ).resolve()
    data_manifest_path = data_root / "manifest.json"
    ready_data = experiment_ready["inputs"]["exact_eval_data"]
    if (
        sha256_file(data_manifest_path) != ready_data["manifest_sha256"]
        or ready_data["manifest_sha256"]
        != evaluation.get("data", {}).get("manifest_sha256")
    ):
        raise RuntimeError("gate frozen-data manifest drift")
    data_manifest = json.loads(
        data_manifest_path.read_text(encoding="utf-8")
    )
    source_rows: dict[tuple[int, int], dict[str, Any]] = {}
    for length in SCREEN_POLICY:
        entry = data_manifest["files"][str(length)]
        path = data_root / entry["relative_path"]
        expected_sha = ready_data["files"][str(length)]["sha256"]
        if (
            sha256_file(path) != expected_sha
            or expected_sha
            != evaluation["data"]["files"][str(length)]["sha256"]
        ):
            raise RuntimeError(f"gate frozen-data file drift at L{length}")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        examples = int(SCREEN_POLICY[length]["examples"])
        if len(rows) < examples:
            raise RuntimeError(f"gate lacks frozen rows at L{length}")
        for local_index, row in enumerate(rows[:examples]):
            source_rows[(length, local_index)] = row

    def decode_generated(token_ids: list[int]) -> str:
        content = list(token_ids)
        if content and content[-1] == OLMO2_EOS_TOKEN_ID:
            content = content[:-1]
        return str(
            tokenizer.decode(
                content,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    passed, details = exact_screen_gate(
        training,
        evaluation,
        parent_evaluation,
        expected_parent_sha256=str(args.expected_parent_sha256),
        raw_rows=raw_rows,
        run_manifest=run_manifest,
        parent_raw_rows=parent_raw_rows,
        parent_run_manifest=parent_run_manifest,
        source_rows=source_rows,
        decode_generated=decode_generated,
        experiment_ready=experiment_ready,
        experiment_ready_sha256=experiment_ready_sha256,
        gate_script_sha256=sha256_file(Path(__file__).resolve()),
    )
    receipt = {
        "status": "PASS" if passed else "STOP",
        "gate": "OLMO2_MINIMAL_FULL_STRING_EXACT_EOS_V1",
        **details,
    }
    atomic_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
