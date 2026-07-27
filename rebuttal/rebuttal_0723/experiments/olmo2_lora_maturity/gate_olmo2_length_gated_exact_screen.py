#!/usr/bin/env python3
"""Fail-closed minimal exact gate for length-gated OLMo-2 EVQ+LoRA.

Passing this gate only authorizes expanded evaluation.  It does not itself
establish broad 8K/16K capability.  The long-context admission metric is the
literal complete decoded continuation plus an observed terminal EOS.  The 4K
check combines the training-time same-load canary with a fresh-checkpoint
reload comparison of hidden states, logits, and KV-cache behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    row_sha256,
)
from .evaluate_instruct_ruler_screen import (
    EXAMPLE_SCHEMA_VERSION,
    GENERATION_TOKENS,
    RUN_MANIFEST_STATUS,
    TRAINING_GAP_MAXIMUM_TOKENS,
    decode_generated_string,
    validate_actual_prompt_geometry,
    validate_data,
)
from .prepare_data import atomic_json, sha256_file
from .train_4k_counterfactual_routing import (
    deterministic_query_offset_stream,
)


MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
PARENT_ADAPTER_SHA256 = (
    "a0ccd2cf141300ba4489882dda1324b2f237e65444a9a71d687e8c5fad57ae8b"
)
NATIVE_FREQUENCY_SHA256 = (
    "dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34"
)
EVQ_FREQUENCY_SHA256 = (
    "917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607"
)
FREQUENCY = "length_gated_native_le4k_evq_gt4k"
PARENT_ADAPTATION = "qkvo_answer"
ADAPTATION = "length_gated_eos_vocab_row"
PARENT_RANK = 64
PARENT_ALPHA = 128.0
CHILD_RANK = 1
CHILD_ALPHA = 1.0
TRAINABLE_SCOPE = "long_only_eos_vocab_row_plus_scalar_bias"
PRE_QUERY_GAP_PARENT_SHA256 = (
    "95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a"
)
TRAINING_STATUS = "OLMO2_4K_LENGTH_GATED_EOS_REPAIR_COMPLETE_V1"
EVALUATION_STATUS = "OLMO2_INSTRUCT_RULER_EXACT_SCREEN_COMPLETE_V3"
SUPERVISION_CONTRACT = "numeric_answer_plus_immediate_eos_v1"
OLMO2_EOS_TOKEN_ID = 100_257
SELECTION_CHECKPOINT_STEPS = (4, 8, 16, 32)
SELECTION_QUERY_OFFSETS = (4_096, 12_289)
SELECTION_MIN_TERMINATION_EXACT = {4_096: 12, 12_289: 4}
SELECTION_ANSWER_NLL_TOLERANCE = 1e-4
SCREEN_POLICY = {
    8_192: {
        "examples": 8,
        "minimum_exact_generation_passes": 6,
        "role": "minimal long exact admission",
    },
    16_384: {
        "examples": 8,
        "minimum_exact_generation_passes": 2,
        "role": "minimal long exact admission",
    },
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


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _require_fields(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    for name, value in expected.items():
        if observed.get(name) != value:
            raise RuntimeError(f"{label} drift for {name}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_adapter_payload(
    path: Path,
    *,
    label: str,
) -> tuple[str, dict[str, torch.Tensor], dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"{label} adapter is not a file")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("state"), dict)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise RuntimeError(f"{label} adapter payload drift")
    state = payload["state"]
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise RuntimeError(f"{label} adapter state drift")
    return sha256_file(path), state, dict(payload["metadata"])


def _named_tensor_sha256(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        payload = value.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _validate_checkpoint_selection(
    *,
    training: dict[str, Any],
    experiment_ready: dict[str, Any],
    child_state: dict[str, torch.Tensor],
    child_metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = training.get("checkpoint_selection", {})
    relative_path = str(summary.get("relative_path", ""))
    selection_path = (
        Path(str(experiment_ready.get("run_output", ""))).resolve()
        / relative_path
    )
    if (
        relative_path != "checkpoint_selection.json"
        or not selection_path.is_file()
        or summary.get("sha256") != sha256_file(selection_path)
    ):
        raise RuntimeError("checkpoint-selection artifact binding drift")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if summary != {
        **selection,
        "relative_path": relative_path,
        "sha256": sha256_file(selection_path),
    }:
        raise RuntimeError("checkpoint-selection result/file drift")

    protocol_policy = training.get("protocol", {}).get(
        "checkpoint_selection", {}
    )
    if (
        selection.get("status")
        != "OLMO2_EOS_CHECKPOINT_SELECTION_V1"
        or selection.get("decision") != "SELECTED"
        or selection.get("policy") != protocol_policy
        or protocol_policy.get("rule") != "first_passing_checkpoint"
        or protocol_policy.get("maximum_steps") != 32
        or protocol_policy.get("checkpoint_steps")
        != list(SELECTION_CHECKPOINT_STEPS)
        or protocol_policy.get("rows") != 8
        or protocol_policy.get("row_indices") != list(range(8))
        or protocol_policy.get("query_offsets")
        != list(SELECTION_QUERY_OFFSETS)
        or protocol_policy.get("minimum_termination_exact_counts")
        != {
            str(key): value
            for key, value in SELECTION_MIN_TERMINATION_EXACT.items()
        }
        or float(protocol_policy.get("answer_nll_tolerance", -1.0))
        != SELECTION_ANSWER_NLL_TOLERANCE
        or protocol_policy.get("selection_is_capability_evidence")
        is not False
    ):
        raise RuntimeError("checkpoint-selection policy drift")

    cache = selection.get("cache", {})
    calibration_ready = experiment_ready["inputs"]["routing_data"][
        "splits"
    ]["calibration"]
    if (
        cache.get("split") != "calibration"
        or cache.get("purpose") != "counterfactual_routing_calibration"
        or cache.get("rows") != 8
        or cache.get("row_indices") != list(range(8))
        or cache.get("query_offsets") != list(SELECTION_QUERY_OFFSETS)
        or cache.get("manifest_sha256")
        != calibration_ready["manifest"]["sha256"]
        or cache.get("files") != calibration_ready["files"]
        or not _is_sha256(cache.get("subset_rows_sha256"))
        or set(cache.get("cells", {}))
        != {str(value) for value in SELECTION_QUERY_OFFSETS}
    ):
        raise RuntimeError("checkpoint-selection calibration binding drift")
    for query_offset in SELECTION_QUERY_OFFSETS:
        cell = cache["cells"][str(query_offset)]
        if (
            int(cell.get("query_offset", -1)) != query_offset
            or int(cell.get("pairs", -1)) != 8
            or int(cell.get("variants", -1)) != 16
            or int(cell.get("termination_tokens", -1)) != 16
            or int(cell.get("supervised_tokens", 0))
            != int(cell.get("answer_tokens", -1)) + 16
            or not _is_sha256(cell.get("position_ids_sha256"))
            or not _is_sha256(cell.get("selected_hidden_sha256"))
            or not _is_sha256(cell.get("selected_labels_sha256"))
        ):
            raise RuntimeError("checkpoint-selection cache-cell drift")

    selected_step = int(selection.get("selected_step", -1))
    actual_steps = int(selection.get("actual_steps", -1))
    history = selection.get("history")
    expected_history_steps = [
        step
        for step in SELECTION_CHECKPOINT_STEPS
        if step <= selected_step
    ]
    if (
        selected_step not in SELECTION_CHECKPOINT_STEPS
        or actual_steps != selected_step
        or int(selection.get("maximum_steps", -1)) != 32
        or bool(selection.get("stopped_early"))
        != (selected_step < 32)
        or not isinstance(history, list)
        or [int(row.get("step", -1)) for row in history]
        != expected_history_steps
        or history != training.get("training", {}).get(
            "checkpoint_history"
        )
        or training.get("training", {}).get("selected_step")
        != selected_step
        or training.get("training", {}).get("actual_steps")
        != selected_step
    ):
        raise RuntimeError("checkpoint first-pass history drift")

    parent = selection.get("parent_step0_metrics", {})
    parent_parameter = parent.get("parameter_state", {})
    if (
        set(parent.get("cells", {}))
        != {str(value) for value in SELECTION_QUERY_OFFSETS}
        or float(parent_parameter.get("delta_weight_l2", -1.0)) != 0.0
        or float(parent_parameter.get("eos_bias", float("nan"))) != 0.0
        or parent_parameter.get("finite") is not True
        or not _is_sha256(parent_parameter.get("sha256"))
    ):
        raise RuntimeError("checkpoint-selection step-0 parent drift")

    for index, row in enumerate(history):
        candidate_parameter = row.get("parameter_state", {})
        expected_checks = {
            "finite_parameter_state": (
                candidate_parameter.get("finite") is True
            ),
            "child_state_changed": (
                candidate_parameter.get("sha256")
                != parent_parameter.get("sha256")
            ),
        }
        for query_offset in SELECTION_QUERY_OFFSETS:
            key = str(query_offset)
            parent_cell = parent["cells"][key]
            candidate_cell = row["cells"][key]
            prefix = f"query_offset_{query_offset}"
            cell_checks = {
                "termination_floor": (
                    int(candidate_cell["termination_exact_count"])
                    >= SELECTION_MIN_TERMINATION_EXACT[query_offset]
                ),
                "termination_improves_parent": (
                    int(candidate_cell["termination_exact_count"])
                    > int(parent_cell["termination_exact_count"])
                ),
                "answer_exact_not_worse": (
                    int(candidate_cell["answer_token_exact_count"])
                    >= int(parent_cell["answer_token_exact_count"])
                ),
                "no_new_answer_eos_top1": (
                    int(candidate_cell["new_answer_eos_top1_count"])
                    == 0
                ),
                "answer_nll_not_worse": (
                    float(candidate_cell["mean_answer_nll"])
                    <= float(parent_cell["mean_answer_nll"])
                    + SELECTION_ANSWER_NLL_TOLERANCE
                ),
            }
            if candidate_cell.get("checks") != cell_checks:
                raise RuntimeError("checkpoint-selection cell checks drift")
            expected_checks.update(
                {
                    f"{prefix}_{name}": value
                    for name, value in cell_checks.items()
                }
            )
        expected_pass = all(expected_checks.values())
        if (
            row.get("checks") != expected_checks
            or row.get("passed") is not expected_pass
            or bool(row.get("passed")) != (
                index == len(history) - 1
            )
        ):
            raise RuntimeError("checkpoint-selection decision drift")

    selected_state_sha = str(
        history[-1]["parameter_state"]["sha256"]
    )
    if (
        selection.get("selected_child_state_sha256")
        != selected_state_sha
        or child_metadata.get("selected_child_state_sha256")
        != selected_state_sha
        or child_metadata.get("selected_optimizer_step") != selected_step
        or child_metadata.get("maximum_optimizer_steps") != 32
        or child_metadata.get("checkpoint_selection_sha256")
        != sha256_file(selection_path)
        or _named_tensor_sha256(child_state) != selected_state_sha
        or not all(
            torch.is_floating_point(value)
            and bool(torch.isfinite(value).all())
            for value in child_state.values()
        )
    ):
        raise RuntimeError("selected EOS child state drift")
    return {
        "selection_sha256": sha256_file(selection_path),
        "selected_step": selected_step,
        "selected_child_state_sha256": selected_state_sha,
        "stopped_early": selected_step < 32,
    }


def validate_raw_evaluation(
    *,
    training: dict[str, Any],
    evaluation: dict[str, Any],
    experiment_ready: dict[str, Any],
    experiment_ready_sha256: str,
    examples_path: Path,
    run_manifest_path: Path,
) -> dict[str, Any]:
    if experiment_ready.get("status") != (
        "OLMO2_4K_LENGTH_GATED_EOS_REPAIR_READY_V1"
    ):
        raise RuntimeError("length-gated READY status drift")
    if (
        training.get("ready_receipt_sha256")
        != experiment_ready_sha256
        or evaluation.get("experiment_ready_receipt_sha256")
        != experiment_ready_sha256
        or experiment_ready["code"]["trainer"]["sha256"]
        != training.get("script_sha256")
        or experiment_ready["code"]["exact_evaluator"]["sha256"]
        != evaluation.get("script_sha256")
        or experiment_ready["code"]["exact_gate"]["sha256"]
        != sha256_file(Path(__file__).resolve())
        or experiment_ready.get("protocol")
        != training.get("protocol")
        or training.get("bound_code")
        != {
            name: entry["sha256"]
            for name, entry in experiment_ready["code"].items()
        }
    ):
        raise RuntimeError("training/evaluation/READY binding drift")

    adapter_chain = experiment_ready.get("registered_adapter_chain", {})
    _require_fields(
        adapter_chain,
        {
            "load_order": [
                "frozen_parent_qkvo",
                "eos_vocab_row_child",
            ],
        },
        label="READY adapter chain",
    )
    _require_fields(
        adapter_chain.get("parent", {}),
        {
            "sha256": PARENT_ADAPTER_SHA256,
            "adaptation": PARENT_ADAPTATION,
            "rank": PARENT_RANK,
            "alpha": PARENT_ALPHA,
            "frozen_during_child_training": True,
        },
        label="READY parent adapter",
    )
    _require_fields(
        adapter_chain.get("child", {}),
        {
            "adaptation": ADAPTATION,
            "rank": CHILD_RANK,
            "alpha": CHILD_ALPHA,
            "modified_vocab_rows": [OLMO2_EOS_TOKEN_ID],
            "parameterization": "direct_single_eos_row_delta",
            "rank1_equivalent": True,
            "trainable_parameters": 2_049,
            "trainable_parameter_tensors": 2,
            "trainable_scope": TRAINABLE_SCOPE,
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
        },
        label="READY child adapter",
    )

    ready_parent = experiment_ready.get("inputs", {}).get(
        "parent_adapter", {}
    )
    parent_path = Path(str(ready_parent.get("path", ""))).resolve()
    parent_sha256, parent_state, parent_metadata = _load_adapter_payload(
        parent_path,
        label="parent",
    )
    expected_parent_shapes: dict[str, tuple[int, int]] = {}
    for layer in range(16):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            prefix = (
                f"model.model.layers.{layer}.self_attn.{projection}"
            )
            expected_parent_shapes[f"{prefix}.a"] = (PARENT_RANK, 2_048)
            expected_parent_shapes[f"{prefix}.b"] = (2_048, PARENT_RANK)
    if (
        parent_sha256 != PARENT_ADAPTER_SHA256
        or ready_parent.get("sha256") != parent_sha256
        or ready_parent.get("metadata") != parent_metadata
        or set(parent_state) != set(expected_parent_shapes)
        or any(
            tuple(parent_state[name].shape) != shape
            for name, shape in expected_parent_shapes.items()
        )
        or Path(str(adapter_chain["parent"].get("path", ""))).resolve()
        != parent_path
    ):
        raise RuntimeError("actual parent adapter binding drift")

    training_parent = training.get("parent_adapter", {})
    evaluation_parent = evaluation.get("parent_adapter", {})
    if (
        Path(str(training_parent.get("path", ""))).resolve() != parent_path
        or training_parent.get("sha256") != parent_sha256
        or training_parent.get("metadata") != parent_metadata
        or Path(str(evaluation_parent.get("path", ""))).resolve()
        != parent_path
        or evaluation_parent.get("sha256") != parent_sha256
        or evaluation_parent.get("metadata") != parent_metadata
    ):
        raise RuntimeError("training/evaluation parent adapter drift")

    evaluation_child = evaluation.get("adapter", {})
    child_path = Path(str(evaluation_child.get("path", ""))).resolve()
    child_sha256, child_state, child_metadata = _load_adapter_payload(
        child_path,
        label="child",
    )
    expected_child_shapes = {
        "model.lm_head.delta_weight": (2_048,),
        "model.lm_head.eos_bias": (),
    }
    if (
        Path(str(adapter_chain["child"].get("path", ""))).resolve()
        != child_path
        or child_sha256 != training.get("adapter_sha256")
        or child_sha256 != evaluation_child.get("sha256")
        or child_metadata != training.get("adapter_metadata")
        or child_metadata != evaluation_child.get("metadata")
        or set(child_state) != set(expected_child_shapes)
        or any(
            tuple(child_state[name].shape) != shape
            for name, shape in expected_child_shapes.items()
        )
    ):
        raise RuntimeError("actual EOS child adapter binding drift")
    checkpoint_selection_validation = _validate_checkpoint_selection(
        training=training,
        experiment_ready=experiment_ready,
        child_state=child_state,
        child_metadata=child_metadata,
    )

    parent_baseline = training.get("parent_exact_baseline", {})
    parent_baseline_root = Path(
        str(
            experiment_ready.get("registered_outputs", {}).get(
                "parent_exact_baseline", ""
            )
        )
    ).resolve()
    parent_baseline_paths = {
        "result_sha256": parent_baseline_root / "results.json",
        "examples_sha256": parent_baseline_root / "examples.jsonl",
        "run_manifest_sha256": (
            parent_baseline_root / "run_manifest.json"
        ),
    }
    if (
        parent_baseline.get("adapter_sha256") != parent_sha256
        or parent_baseline.get("decision") != "EOS_ONLY_REPAIR_FEASIBLE"
        or any(
            not path.is_file()
            or parent_baseline.get(field) != sha256_file(path)
            for field, path in parent_baseline_paths.items()
        )
    ):
        raise RuntimeError("parent exact-baseline artifact binding drift")

    run_manifest = json.loads(
        run_manifest_path.read_text(encoding="utf-8")
    )
    training_result_path = (
        Path(str(experiment_ready.get("run_output", ""))).resolve()
        / "results.json"
    )
    expected_artifact_preflight = {
        "status": "PASS",
        "parent_adapter_sha256": parent_sha256,
        "child_adapter_sha256": child_sha256,
        "training_result": str(training_result_path),
        "training_result_sha256": sha256_file(training_result_path),
        "validation_stage": "cpu_before_cuda_configuration",
    }
    if (
        sha256_file(run_manifest_path)
        != evaluation.get("run_manifest_sha256")
        or run_manifest.get("status") != RUN_MANIFEST_STATUS
        or int(run_manifest.get("example_schema_version", -1))
        != EXAMPLE_SCHEMA_VERSION
        or run_manifest.get("checkpoint_sha256") != MODEL_SHA256
        or run_manifest.get("experiment_ready_receipt_sha256")
        != experiment_ready_sha256
        or run_manifest.get("experiment_role")
        != "candidate_exact_screen"
        or run_manifest.get("frequency") != FREQUENCY
        or run_manifest.get("adaptation") != ADAPTATION
        or int(run_manifest.get("rank", -1)) != CHILD_RANK
        or float(run_manifest.get("alpha", -1.0)) != CHILD_ALPHA
        or run_manifest.get("parent_adapter_sha256")
        != PARENT_ADAPTER_SHA256
        or int(run_manifest.get("parent_rank", -1)) != PARENT_RANK
        or float(run_manifest.get("parent_alpha", -1.0))
        != PARENT_ALPHA
        or run_manifest.get("task") != "niah_single_1"
        or run_manifest.get("lengths") != list(SCREEN_POLICY)
        or int(run_manifest.get("limit_per_length", -1)) != 8
        or run_manifest.get("greedy") is not True
        or int(run_manifest.get("maximum_new_tokens", -1))
        != GENERATION_TOKENS
        or run_manifest.get("string_normalization") != "none"
        or run_manifest.get("decode_cleanup") is not False
        or run_manifest.get("terminal_eos_removed_before_string_decode")
        is not True
        or run_manifest.get("other_special_tokens_removed") is not False
        or run_manifest.get("adapter_sha256") != child_sha256
        or run_manifest.get("bound_code_sha256")
        != evaluation.get("bound_code_sha256")
        or run_manifest.get("length_gated_artifact_preflight")
        != expected_artifact_preflight
        or evaluation.get("length_gated_artifact_preflight")
        != expected_artifact_preflight
        or evaluation.get("bound_code_sha256")
        != experiment_ready.get("evaluator_bound_code_sha256")
    ):
        raise RuntimeError("exact-screen run manifest drift")

    if (
        sha256_file(examples_path)
        != evaluation.get("results", {}).get("examples_sha256")
    ):
        raise RuntimeError("exact-screen raw examples hash drift")
    checkpoint = Path(evaluation["checkpoint"]).resolve()
    ready_checkpoint = Path(
        experiment_ready["inputs"]["checkpoint"]["path"]
    ).resolve()
    if checkpoint != ready_checkpoint:
        raise RuntimeError("exact-screen checkpoint path drift")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if int(tokenizer.eos_token_id) != OLMO2_EOS_TOKEN_ID:
        raise RuntimeError("OLMo tokenizer EOS identity drift")

    ready_data = experiment_ready["inputs"]["exact_eval_data"]
    data_root = Path(ready_data["path"]).resolve()
    data_receipt, selected_rows = validate_data(
        data_root,
        checkpoint,
        "niah_single_1",
        tuple(SCREEN_POLICY),
        8,
    )
    if (
        data_receipt["manifest_sha256"]
        != ready_data["manifest_sha256"]
        or {
            key: value["sha256"]
            for key, value in data_receipt["files"].items()
        }
        != {
            key: value["sha256"]
            for key, value in ready_data["files"].items()
        }
        or evaluation.get("data") != data_receipt
    ):
        raise RuntimeError("exact-screen frozen source data drift")
    actual_prompt_geometry = validate_actual_prompt_geometry(
        rows=selected_rows,
        tokenizer=tokenizer,
    )
    if (
        ready_data.get("actual_prompt_geometry")
        != actual_prompt_geometry
        or run_manifest.get("actual_prompt_geometry")
        != actual_prompt_geometry
        or evaluation.get("actual_prompt_geometry")
        != actual_prompt_geometry
    ):
        raise RuntimeError("exact-screen actual prompt geometry drift")
    geometry_rows = {
        (int(row["nominal_length"]), int(row["local_index"])): row
        for row in actual_prompt_geometry["rows"]
    }
    source_rows = {
        (int(row["_nominal_length"]), int(row["_local_index"])): {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        for row in selected_rows
    }
    expected_keys = {
        (length, local_index)
        for length, policy in SCREEN_POLICY.items()
        for local_index in range(int(policy["examples"]))
    }
    raw_rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in _read_jsonl(examples_path):
        key = (
            int(row.get("nominal_length", -1)),
            int(row.get("local_index", -1)),
        )
        if key in raw_rows:
            raise RuntimeError(f"duplicate exact-screen raw row: {key}")
        raw_rows[key] = row
    if set(raw_rows) != expected_keys or set(source_rows) != expected_keys:
        raise RuntimeError("exact-screen raw/source row set drift")

    counts: dict[str, dict[str, int]] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    for length, policy in SCREEN_POLICY.items():
        totals = {
            "examples": int(policy["examples"]),
            "exact_generation_passes": 0,
            "full_string_exact": 0,
            "eos_terminated": 0,
            "answer_eos_token_exact": 0,
            "far_gap_examples": 0,
            "far_gap_exact_generation_passes": 0,
        }
        failure_modes = {
            "full_string_exact_without_eos": 0,
            "eos_with_wrong_full_string": 0,
            "wrong_full_string_without_eos": 0,
            "reference_prefix_with_extra_generation": 0,
        }
        for local_index in range(int(policy["examples"])):
            row = raw_rows[(length, local_index)]
            source = source_rows[(length, local_index)]
            geometry = geometry_rows[(length, local_index)]
            references = [str(value) for value in source["outputs"]]
            reference_token_ids = [
                [
                    int(value)
                    for value in tokenizer(
                        reference,
                        add_special_tokens=False,
                    ).input_ids
                ]
                for reference in references
            ]
            token_ids = row.get("generated_token_ids")
            if (
                int(row.get("schema_version", -1))
                != EXAMPLE_SCHEMA_VERSION
                or row.get("task") != "niah_single_1"
                or not isinstance(token_ids, list)
                or not all(isinstance(value, int) for value in token_ids)
                or not 1 <= len(token_ids) <= GENERATION_TOKENS
                or int(row.get("generated_tokens", -1))
                != len(token_ids)
                or OLMO2_EOS_TOKEN_ID in token_ids[:-1]
                or row.get("references") != references
                or row.get("reference_token_ids")
                != reference_token_ids
                or int(row.get("source_row_index", -1))
                != int(source["index"])
                or int(row.get("source_token_position_answer", -1))
                != int(source["token_position_answer"])
                or row.get("row_sha256") != row_sha256(source)
                or int(row.get("input_tokens", -1))
                + GENERATION_TOKENS
                > length
                or int(row.get("input_tokens", -1))
                != int(geometry["prompt_tokens"])
                or int(
                    row.get("actual_maximum_prompt_position_id", -1)
                )
                != int(
                    geometry["actual_maximum_prompt_position_id"]
                )
                or int(
                    row.get("generation_boundary_gap_tokens", -1)
                )
                != int(geometry["generation_boundary_gap_tokens"])
                or int(
                    row.get("required_minimum_prompt_tokens", -1)
                )
                != length // 2 + 1
                or int(
                    row.get(
                        "required_minimum_actual_maximum_position_id",
                        -1,
                    )
                )
                != length // 2
                or row.get("far_gap_beyond_training_support")
                is not bool(
                    int(geometry["generation_boundary_gap_tokens"])
                    > TRAINING_GAP_MAXIMUM_TOKENS
                )
            ):
                raise RuntimeError(
                    f"L{length}/row{local_index} raw payload drift"
                )
            prediction = decode_generated_string(
                tokenizer=tokenizer,
                generated_token_ids=token_ids,
                eos_token_id=OLMO2_EOS_TOKEN_ID,
            )
            if row.get("prediction") != prediction:
                raise RuntimeError(
                    f"L{length}/row{local_index} raw decode drift"
                )
            full = prediction in references
            eos = int(token_ids[-1]) == OLMO2_EOS_TOKEN_ID
            token_exact = any(
                token_ids == [*reference_ids, OLMO2_EOS_TOKEN_ID]
                for reference_ids in reference_token_ids
            )
            # The registered success contract is literal decoded-string
            # equality plus terminal EOS. Canonically identical token IDs are
            # deliberately diagnostic because alternate tokenizations of the
            # same literal answer remain valid.
            exact = full and eos
            for name, expected in (
                ("full_string_exact", full),
                ("eos_terminated", eos),
                ("answer_eos_token_exact", token_exact),
                ("exact_generation_pass", exact),
            ):
                if float(row.get(name, float("nan"))) != float(expected):
                    raise RuntimeError(
                        f"L{length}/row{local_index} {name} drift"
                    )
            totals["full_string_exact"] += int(full)
            totals["eos_terminated"] += int(eos)
            totals["answer_eos_token_exact"] += int(token_exact)
            totals["exact_generation_passes"] += int(exact)
            far_gap = bool(
                int(geometry["generation_boundary_gap_tokens"])
                > TRAINING_GAP_MAXIMUM_TOKENS
            )
            totals["far_gap_examples"] += int(far_gap)
            totals["far_gap_exact_generation_passes"] += int(
                far_gap and exact
            )
            if full and not eos:
                failure_modes["full_string_exact_without_eos"] += 1
            elif eos and not full:
                failure_modes["eos_with_wrong_full_string"] += 1
            elif not full and not eos:
                failure_modes["wrong_full_string_without_eos"] += 1
            if (
                not full
                and any(
                    prediction.startswith(reference)
                    for reference in references
                )
            ):
                failure_modes[
                    "reference_prefix_with_extra_generation"
                ] += 1
        counts[str(length)] = totals
        if (
            failure_modes["full_string_exact_without_eos"]
            + failure_modes["eos_with_wrong_full_string"]
            + failure_modes["wrong_full_string_without_eos"]
            != int(policy["examples"])
            - totals["exact_generation_passes"]
        ):
            raise RuntimeError(
                f"L{length} raw failure partition drift"
            )
        diagnostics[str(length)] = failure_modes
    return {
        "counts": counts,
        "checkpoint_selection": checkpoint_selection_validation,
        "failure_diagnostics": diagnostics,
        "diagnostics_are_success_metrics": False,
    }


def exact_screen_gate(
    training: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    raw_counts: dict[str, dict[str, int]],
) -> tuple[bool, dict[str, Any]]:
    if training.get("status") != TRAINING_STATUS:
        raise RuntimeError("length-gated training receipt status drift")
    if evaluation.get("status") != EVALUATION_STATUS:
        raise RuntimeError("evaluation predates exact/EOS scoring")
    if (
        training.get("checkpoint_sha256") != MODEL_SHA256
        or evaluation.get("checkpoint_sha256") != MODEL_SHA256
    ):
        raise RuntimeError("gate is not bound to OLMo-2 1.485B")
    if (
        training.get("ready_receipt_sha256")
        != evaluation.get("experiment_ready_receipt_sha256")
    ):
        raise RuntimeError("training/evaluation READY receipt mismatch")

    bound_code = training.get("bound_code", {})
    if (
        evaluation.get("script_sha256")
        != bound_code.get("exact_evaluator")
        or sha256_file(Path(__file__).resolve())
        != bound_code.get("exact_gate")
    ):
        raise RuntimeError("exact evaluator or gate changed after READY")
    evaluator_bound = evaluation.get("bound_code_sha256", {})
    if (
        evaluator_bound.get("length_gated_import_dependency")
        != bound_code.get("method")
    ):
        raise RuntimeError("evaluation method dependency changed after READY")

    method = training.get("method", {})
    _require_fields(
        method,
        {
            "active_frequency": FREQUENCY,
            "active_sha256_float32": EVQ_FREQUENCY_SHA256,
            "adaptation": ADAPTATION,
            "parent_adaptation": PARENT_ADAPTATION,
            "parent_rank": PARENT_RANK,
            "parent_alpha": PARENT_ALPHA,
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "trainable_scope": TRAINABLE_SCOPE,
            "short_lora_dispatch": "base_linear_direct_call",
            "long_lora_dispatch": "qkvo_lora",
        },
        label="training method",
    )
    _require_fields(
        method.get("child_adapter", {}),
        {
            "adaptation": ADAPTATION,
            "scope": "long_mode_only",
            "modified_vocab_rows": [OLMO2_EOS_TOKEN_ID],
            "short_dispatch": "base_lm_head_direct_call",
            "rank": CHILD_RANK,
            "alpha": CHILD_ALPHA,
            "scalar_bias": True,
            "trainable_parameters": 2_049,
            "trainable_parameter_tensors": 2,
            "parameterization": "direct_single_eos_row_delta",
            "rank1_equivalent": True,
        },
        label="training EOS child",
    )
    _require_fields(
        method.get("short_branch", {}),
        {
            "maximum_position_id": 4_095,
            "frequency": "native_endpoint_rope",
            "frequency_sha256_float32": NATIVE_FREQUENCY_SHA256,
            "rotary_dispatch": "original_module_direct_call",
        },
        label="training short branch",
    )
    _require_fields(
        method.get("long_branch", {}),
        {
            "minimum_maximum_position_id": 4_096,
            "frequency": "evq_endpoint_cosh",
            "frequency_sha256_float32": EVQ_FREQUENCY_SHA256,
            "scope": "entire_sequence",
        },
        label="training long branch",
    )

    protocol = training.get("protocol", {})
    _require_fields(
        protocol,
        {
            "model_scope": "OLMo-2 1.485B Instruct only",
            "method": FREQUENCY,
            "frequency": FREQUENCY,
            "short_branch_maximum_position_id": 4_095,
            "short_branch_lora": "frozen_base_linear_direct_call",
            "long_branch_minimum_maximum_position_id": 4_096,
            "long_branch_scope": "entire_sequence",
            "cached_generation_branch_selection": (
                "force_from_total_context_budget_before_prompt"
            ),
            "cached_short_to_long_transition": "fail_closed",
            "adaptation": ADAPTATION,
            "parent_adaptation": PARENT_ADAPTATION,
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "parent_rank": PARENT_RANK,
            "parent_alpha": PARENT_ALPHA,
            "steps": 32,
            "child_rank": CHILD_RANK,
            "child_alpha": CHILD_ALPHA,
            "trainable_scope": TRAINABLE_SCOPE,
            "learning_rate": 1e-3,
            "warmup_steps": 4,
            "training_branch": (
                "forced_long_full_evq_frozen_qkvo_parent_plus_eos_child"
            ),
            "maximum_physical_training_sequence_length": 4_096,
            "hard_maximum_training_length": 4_096,
            "real_8k_or_16k_training_sequences": 0,
            "virtual_target_length": 16_384,
            "routing_optimizer_steps": 32,
            "supervision_contract": SUPERVISION_CONTRACT,
            "supervision": (
                "causally_shifted_answer_ce_plus_weighted_immediate_eos_ce"
            ),
            "final_eos_supervised": True,
            "natural_replay": "none_eos_only_parent_preserving_repair",
            "expanded_evaluation_before_minimal_exact_gate": False,
            "minimum_free_bytes_before_run": 20 * 1024**3,
            "checkpoint_selection": {
                "rule": "first_passing_checkpoint",
                "maximum_steps": 32,
                "checkpoint_steps": list(SELECTION_CHECKPOINT_STEPS),
                "split": "calibration",
                "rows": 8,
                "row_indices": list(range(8)),
                "query_offsets": list(SELECTION_QUERY_OFFSETS),
                "minimum_termination_exact_counts": {
                    str(key): value
                    for key, value in (
                        SELECTION_MIN_TERMINATION_EXACT.items()
                    )
                },
                "answer_nll_tolerance": (
                    SELECTION_ANSWER_NLL_TOLERANCE
                ),
                "selection_is_capability_evidence": False,
            },
        },
        label="training protocol",
    )
    routing_data = training.get("routing_data", {})
    _require_fields(
        routing_data,
        {
            "status": "OLMO2_4K_COUNTERFACTUAL_ROUTING_DATA_PREPARED_V2",
            "format_version": 2,
            "supervision_contract": SUPERVISION_CONTRACT,
            "eos_token_id": OLMO2_EOS_TOKEN_ID,
            "final_eos_supervised": True,
            "labels_only_cover_answer_and_final_eos": True,
            "answer_string_tokenizer_roundtrip_exact": True,
        },
        label="routing data",
    )
    if not _is_sha256(routing_data.get("manifest_sha256")):
        raise RuntimeError("routing-data manifest SHA drift")
    executed = training.get("training", {})
    actual_steps = int(executed.get("actual_steps", -1))
    if actual_steps not in SELECTION_CHECKPOINT_STEPS:
        raise RuntimeError("executed checkpoint selection step drift")
    expected_position_bucket_counts = {
        "contiguous": (actual_steps + 1) // 2,
        "transition": actual_steps // 2,
        "middle": actual_steps,
        "far": 2 * actual_steps,
    }
    planned_offsets = deterministic_query_offset_stream(
        seed=int(protocol["seed"]),
        routing_steps=32,
    )
    if (
        hashlib.sha256(planned_offsets.tobytes(order="C")).hexdigest()
        != protocol.get("query_offset_stream_sha256")
    ):
        raise RuntimeError("planned query-offset stream drift")
    consumed_offsets = planned_offsets[: 4 * actual_steps]
    expected_consumed_prefix_sha256 = hashlib.sha256(
        consumed_offsets.tobytes(order="C")
    ).hexdigest()
    _require_fields(
        executed,
        {
            "steps": actual_steps,
            "maximum_steps": 32,
            "actual_steps": actual_steps,
            "selected_step": actual_steps,
            "stopped_early": actual_steps < 32,
            "checkpoint_steps": list(SELECTION_CHECKPOINT_STEPS),
            "checkpoint_selection_required": True,
            "checkpoint_selection_passed": True,
            "family_pattern": ["routing"],
            "family_steps": {
                "routing": actual_steps,
                "natural": 0,
            },
            "processed_input_tokens": actual_steps * 32_760,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "global_batch_size": 8,
            "learning_rate": 1e-3,
            "warmup_steps": 4,
            "compile_mode": "max-autotune-no-cudagraphs",
            "trainable_parameters": 2_049,
            "position_policy": "semantic_query_block_continuous_gap",
            "virtual_target_length": 16_384,
            "virtual_bucket_weights": [1, 1, 2],
            "position_bucket_counts": expected_position_bucket_counts,
            "query_offset_stream_sha256": protocol[
                "query_offset_stream_sha256"
            ],
            "consumed_query_offset_prefix_sha256": (
                expected_consumed_prefix_sha256
            ),
            "consumed_query_offset_values": 4 * actual_steps,
        },
        label="executed training",
    )
    checkpoint_history = executed.get("checkpoint_history")
    expected_checkpoint_steps = [
        step
        for step in SELECTION_CHECKPOINT_STEPS
        if step <= actual_steps
    ]
    if (
        not isinstance(checkpoint_history, list)
        or [int(row.get("step", -1)) for row in checkpoint_history]
        != expected_checkpoint_steps
        or any(
            row.get("passed") is not False
            for row in checkpoint_history[:-1]
        )
        or checkpoint_history[-1].get("passed") is not True
    ):
        raise RuntimeError("executed checkpoint history drift")
    if (
        not 12_288
        < int(executed.get("maximum_observed_position_id", -1))
        <= 16_383
        or not _is_sha256(
            executed.get("realized_position_stream_sha256")
        )
        or not _is_sha256(
            executed.get("realized_exposure_stream_sha256")
        )
        or not math.isfinite(float(executed.get("tokens_per_second", 0.0)))
        or float(executed.get("tokens_per_second", 0.0)) <= 0.0
    ):
        raise RuntimeError("executed long-branch training receipt drift")

    _require_fields(
        training.get("trainable_scope", {}),
        {
            "scope": TRAINABLE_SCOPE,
            "parameter_names": [
                "model.lm_head.delta_weight",
                "model.lm_head.eos_bias",
            ],
            "parameter_tensors": 2,
            "parameters": 2_049,
        },
        label="EOS-child trainable scope",
    )
    parent_integrity = training.get("parent_qkvo_integrity", {})
    parent_state_before = parent_integrity.get(
        "state_sha256_before_training"
    )
    _require_fields(
        parent_integrity,
        {
            "state_sha256_after_training": parent_state_before,
            "torch_state_equal": True,
            "modules": 64,
            "parameter_tensors": 128,
        },
        label="frozen QKVO parent integrity",
    )
    _require_fields(
        parent_integrity.get("freeze_receipt", {}),
        {
            "qkvo_lora_modules": 64,
            "parameter_tensors": 128,
            "parameters": 16_777_216,
            "trainable_after_freeze": False,
        },
        label="frozen QKVO parent scope",
    )
    if not _is_sha256(parent_state_before):
        raise RuntimeError("frozen QKVO parent state SHA drift")

    parent_baseline = training.get("parent_exact_baseline", {})
    if (
        parent_baseline.get("adapter_sha256")
        != PARENT_ADAPTER_SHA256
        or parent_baseline.get("decision")
        != "EOS_ONLY_REPAIR_FEASIBLE"
        or not all(
            _is_sha256(parent_baseline.get(name))
            for name in (
                "result_sha256",
                "examples_sha256",
                "run_manifest_sha256",
            )
        )
    ):
        raise RuntimeError("parent exact-baseline receipt drift")

    training_log = training.get("training_log", {})
    expected_log_steps = sorted(
        {
            1,
            actual_steps,
            *(
                step
                for step in SELECTION_CHECKPOINT_STEPS
                if step <= actual_steps
            ),
            *(step for step in (25,) if step <= actual_steps),
        }
    )
    if (
        not _is_sha256(training_log.get("sha256"))
        or int(training_log.get("rows", -1))
        != len(expected_log_steps)
    ):
        raise RuntimeError("optimizer-step log receipt drift")
    for label, expected_step in (
        ("first_optimizer_step", 1),
        ("last_optimizer_step", actual_steps),
    ):
        row = training_log.get(label, {})
        if (
            int(row.get("step", -1)) != expected_step
            or int(row.get("processed_input_tokens", 0)) <= 0
            or int(row.get("peak_memory_allocated_bytes", 0)) <= 0
            or not all(
                math.isfinite(float(row.get(name, float("nan"))))
                for name in (
                    "loss",
                    "grad_norm",
                    "interval_tokens_per_second",
                )
            )
        ):
            raise RuntimeError(f"{label} runtime receipt drift")
    runtime = training.get("runtime", {})
    output_filesystem = runtime.get("output_filesystem", {})
    capability = runtime.get("capability")
    if (
        not str(runtime.get("name", "")).strip()
        or not isinstance(capability, list)
        or len(capability) != 2
        or int(capability[0]) < 8
        or runtime.get("flash_sdp_enabled") is not True
        or runtime.get("math_sdp_enabled") is not False
        or runtime.get("mem_efficient_sdp_enabled") is not False
        or not str(runtime.get("compile_cache", "")).strip()
        or "expandable_segments:True"
        not in str(runtime.get("allocator", ""))
        or int(output_filesystem.get("minimum_free_bytes", -1))
        != 20 * 1024**3
        or int(output_filesystem.get("free_bytes_before_run", -1))
        < 20 * 1024**3
        or int(output_filesystem.get("free_bytes_after_training", -1))
        <= 0
    ):
        raise RuntimeError("training runtime environment receipt drift")

    short_parity = training.get("short_branch_parity", {})
    _require_fields(
        short_parity,
        {
            "scope": "same_load_4096_hidden_and_final_logits",
            "input_tokens": 4_096,
            "maximum_position_id": 4_095,
            "mode": "short",
        },
        label="short-branch runtime parity",
    )
    short_same_load_canary = (
        short_parity.get(
            "pristine_native_vs_pretraining_hidden_torch_equal"
        )
        is True
        and short_parity.get(
            "pristine_native_vs_pretraining_final_logits_torch_equal"
        )
        is True
        and short_parity.get(
            "pristine_native_vs_posttraining_hidden_torch_equal"
        )
        is True
        and short_parity.get(
            "pristine_native_vs_posttraining_final_logits_torch_equal"
        )
        is True
        and float(
            short_parity.get(
                "maximum_hidden_absolute_difference", float("nan")
            )
        )
        == 0.0
        and float(
            short_parity.get(
                "maximum_final_logits_absolute_difference", float("nan")
            )
        )
        == 0.0
        and short_parity.get("finite_pristine_native") is True
        and short_parity.get("finite_posttraining") is True
    )

    expected_metadata = {
        "base_checkpoint_sha256": MODEL_SHA256,
        "frequency": FREQUENCY,
        "frequency_sha256_float32": EVQ_FREQUENCY_SHA256,
        "short_branch_frequency": "native_endpoint_rope",
        "short_branch_frequency_sha256_float32": (
            NATIVE_FREQUENCY_SHA256
        ),
        "short_branch_maximum_position_id": 4_095,
        "short_branch_lora": "frozen_base_linear_direct_call",
        "long_branch_frequency": "evq_endpoint_cosh",
        "long_branch_minimum_maximum_position_id": 4_096,
        "long_branch_scope": "entire_sequence",
        "adaptation": ADAPTATION,
        "stage": "length_gated_query_gap_16k_eos_repair_v1",
        "rank": CHILD_RANK,
        "alpha": CHILD_ALPHA,
        "parent_adaptation": PARENT_ADAPTATION,
        "parent_rank": PARENT_RANK,
        "parent_alpha": PARENT_ALPHA,
        "trainable_scope": TRAINABLE_SCOPE,
        "modified_vocab_rows": [OLMO2_EOS_TOKEN_ID],
        "scalar_eos_bias": True,
        "parameterization": "direct_single_eos_row_delta",
        "rank1_equivalent": True,
        "trainable_parameters": 2_049,
        "trainable_parameter_tensors": 2,
        "training_sequence_length": 4_096,
        "maximum_physical_training_sequence_length": 4_096,
        "real_8k_or_16k_training_sequences": 0,
        "virtual_target_length": 16_384,
        "position_policy": "semantic_query_block_continuous_gap",
        "cached_generation_branch_selection": (
            "force_from_total_context_budget_before_prompt"
        ),
        "final_eos_supervised": True,
        "supervision_contract": SUPERVISION_CONTRACT,
        "eos_token_id": OLMO2_EOS_TOKEN_ID,
        "termination_weight": 1.0,
        "routing_data_sha256": routing_data["manifest_sha256"],
        "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
        "parent_qkvo_state_sha256": parent_state_before,
        "maximum_optimizer_steps": 32,
        "selected_optimizer_step": actual_steps,
        "checkpoint_selection_sha256": training[
            "checkpoint_selection"
        ]["sha256"],
        "selected_child_state_sha256": training[
            "checkpoint_selection"
        ]["selected_child_state_sha256"],
    }
    _require_fields(
        training.get("adapter_metadata", {}),
        expected_metadata,
        label="training adapter metadata",
    )
    adapter = evaluation.get("adapter")
    if not isinstance(adapter, dict):
        raise RuntimeError("exact screen did not load an adapter")
    if adapter.get("sha256") != training.get("adapter_sha256"):
        raise RuntimeError("training/evaluation adapter SHA mismatch")
    _require_fields(
        adapter,
        {
            "adaptation": ADAPTATION,
            "rank": CHILD_RANK,
            "alpha": CHILD_ALPHA,
            "trainable_parameter_names": [
                "model.lm_head.delta_weight",
                "model.lm_head.eos_bias",
            ],
        },
        label="evaluated EOS child",
    )
    _require_fields(
        adapter.get("method", {}),
        {
            "adaptation": ADAPTATION,
            "scope": "long_mode_only",
            "modified_vocab_rows": [OLMO2_EOS_TOKEN_ID],
            "short_dispatch": "base_lm_head_direct_call",
            "rank": CHILD_RANK,
            "alpha": CHILD_ALPHA,
            "scalar_bias": True,
            "trainable_parameters": 2_049,
            "trainable_parameter_tensors": 2,
            "parameterization": "direct_single_eos_row_delta",
            "rank1_equivalent": True,
        },
        label="evaluated EOS-child method",
    )
    _require_fields(
        adapter.get("metadata", {}),
        expected_metadata,
        label="evaluated adapter metadata",
    )

    expected_parent_metadata = {
        "base_checkpoint_sha256": MODEL_SHA256,
        "frequency": "evq",
        "adaptation": PARENT_ADAPTATION,
        "rank": PARENT_RANK,
        "alpha": PARENT_ALPHA,
        "training_sequence_length": 4_096,
        "stage": "counterfactual_routing_semantic_query_gap_16k",
        "parent_adapter_sha256": PRE_QUERY_GAP_PARENT_SHA256,
        "position_policy": "semantic_query_block_continuous_gap",
        "virtual_target_length": 16_384,
    }
    _require_fields(
        training.get("parent_adapter", {}).get("metadata", {}),
        expected_parent_metadata,
        label="training parent metadata",
    )
    evaluated_parent = evaluation.get("parent_adapter")
    if not isinstance(evaluated_parent, dict):
        raise RuntimeError("exact screen did not load the QKVO parent")
    _require_fields(
        evaluated_parent,
        {
            "sha256": PARENT_ADAPTER_SHA256,
            "adaptation": PARENT_ADAPTATION,
            "rank": PARENT_RANK,
            "alpha": PARENT_ALPHA,
        },
        label="evaluated parent adapter",
    )
    _require_fields(
        evaluated_parent.get("metadata", {}),
        expected_parent_metadata,
        label="evaluated parent metadata",
    )
    _require_fields(
        evaluated_parent.get("freeze", {}),
        {
            "qkvo_lora_modules": 64,
            "parameter_tensors": 128,
            "parameters": 16_777_216,
            "trainable_after_freeze": False,
        },
        label="evaluated frozen parent scope",
    )

    evaluation_frequency = evaluation.get("frequency", {})
    _require_fields(
        evaluation_frequency,
        {
            "active_frequency": FREQUENCY,
            "active_sha256_float32": EVQ_FREQUENCY_SHA256,
            "adaptation": ADAPTATION,
            "parent_adaptation": PARENT_ADAPTATION,
            "parent_rank": PARENT_RANK,
            "parent_alpha": PARENT_ALPHA,
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "trainable_scope": TRAINABLE_SCOPE,
            "short_lora_dispatch": "base_linear_direct_call",
            "long_lora_dispatch": "qkvo_lora",
        },
        label="evaluation method",
    )
    _require_fields(
        evaluation_frequency.get("child_adapter", {}),
        {
            "adaptation": ADAPTATION,
            "scope": "long_mode_only",
            "modified_vocab_rows": [OLMO2_EOS_TOKEN_ID],
            "short_dispatch": "base_lm_head_direct_call",
            "rank": CHILD_RANK,
            "alpha": CHILD_ALPHA,
            "scalar_bias": True,
            "trainable_parameters": 2_049,
            "trainable_parameter_tensors": 2,
            "parameterization": "direct_single_eos_row_delta",
            "rank1_equivalent": True,
        },
        label="evaluation method EOS child",
    )
    _require_fields(
        evaluation.get("protocol", {}),
        {
            "task": "niah_single_1",
            "lengths": list(SCREEN_POLICY),
            "limit_per_length": 8,
            "greedy": True,
            "maximum_new_tokens": GENERATION_TOKENS,
            "minimum_far_gap_rows_per_cell": 1,
            "far_gap_threshold_tokens": (
                TRAINING_GAP_MAXIMUM_TOKENS
            ),
            "actual_prompt_length_requirement": (
                "actual_maximum_prompt_position_id >= "
                "nominal_length / 2"
            ),
            "primary_metric": (
                "literal exact decoded generated string plus observed "
                "terminal EOS"
            ),
            "string_normalization": "none",
            "decode_cleanup": False,
            "terminal_eos_removed_before_string_decode": True,
            "other_special_tokens_removed": False,
            "substring_is_success": False,
            "first_number_is_success": False,
            "training_length_if_adapted": 4_096,
            "length_gated_cache_branch_selection": (
                "forced_from_nominal_context_budget_before_prompt"
            ),
        },
        label="evaluation protocol",
    )

    reload_parity = evaluation.get("short_structural_parity", {})
    _require_fields(
        reload_parity,
        {
            "status": "PASS",
            "input_tokens": 4_096,
            "maximum_position_id": 4_095,
            "prefill_tokens": 32,
            "decode_steps": 1,
            "runtime_contract": (
                "fresh checkpoint load; same BF16/Flash runtime; parent then "
                "child artifacts reloaded; branch forced short before prefill"
            ),
        },
        label="fresh-reload short structural parity",
    )
    reload_torch_equal = reload_parity.get("torch_equal", {})
    _require_fields(
        reload_torch_equal,
        {
            "hidden": True,
            "final_logits": True,
            "prefill_logits": True,
            "decode_logits": True,
            "prefill_cache": True,
            "decode_cache": True,
        },
        label="fresh-reload short tensor parity",
    )
    reload_maximum_difference = reload_parity.get(
        "maximum_absolute_difference", {}
    )
    _require_fields(
        reload_maximum_difference,
        {
            "hidden": 0.0,
            "final_logits": 0.0,
            "prefill_logits": 0.0,
            "decode_logits": 0.0,
        },
        label="fresh-reload short numerical parity",
    )
    _require_fields(
        reload_parity.get("cache_tensor_counts", {}),
        {
            "prefill_cache": 32,
            "decode_cache": 32,
        },
        label="fresh-reload short cache depth",
    )
    for cache_name in ("prefill_cache", "decode_cache"):
        cache_hashes = reload_parity.get("cache_sha256", {}).get(
            cache_name, {}
        )
        if (
            not _is_sha256(cache_hashes.get("pristine_native"))
            or cache_hashes.get("pristine_native")
            != cache_hashes.get("reloaded_candidate")
        ):
            raise RuntimeError(
                f"fresh-reload {cache_name} hash parity drift"
            )

    cells = evaluation.get("results", {}).get("cells", {})
    if set(cells) != {str(length) for length in SCREEN_POLICY}:
        raise RuntimeError("minimal exact-screen cells drift")
    if set(raw_counts) != {str(length) for length in SCREEN_POLICY}:
        raise RuntimeError("minimal exact-screen raw-count cells drift")
    checks = {
        "short_branch_same_load_native_canary": short_same_load_canary,
        "short_branch_fresh_reload_hidden_logits_kv_parity": True,
    }
    counts: dict[str, dict[str, Any]] = {}
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
        far_examples = int(cell.get("far_gap_examples", -1))
        if not 1 <= far_examples <= examples:
            raise RuntimeError(f"L{length} far-gap row count drift")
        far_exact = _exact_count(
            cell.get("far_gap_exact_generation_pass"),
            far_examples,
            f"L{length}.far_gap_exact_generation_pass",
        )
        if exact > min(full, eos):
            raise RuntimeError(
                f"L{length} exact pass exceeds component metrics"
            )
        observed_raw = raw_counts.get(str(length))
        expected_raw = {
            "examples": examples,
            "exact_generation_passes": exact,
            "full_string_exact": full,
            "eos_terminated": eos,
            "answer_eos_token_exact": token,
            "far_gap_examples": far_examples,
            "far_gap_exact_generation_passes": far_exact,
        }
        if observed_raw != expected_raw:
            raise RuntimeError(
                f"L{length} aggregate/raw exact component drift"
            )
        minimum = policy["minimum_exact_generation_passes"]
        if minimum is not None:
            checks[f"L{length}_exact_at_least_{minimum}_of_8"] = (
                exact >= int(minimum)
            )
        checks[f"L{length}_far_gap_exact_at_least_1"] = far_exact >= 1
        counts[str(length)] = {
            "examples": examples,
            "role": policy["role"],
            "exact_generation_passes": exact,
            "full_string_exact": full,
            "eos_terminated": eos,
            "answer_eos_token_exact": token,
            "far_gap_examples": far_examples,
            "far_gap_exact_generation_passes": far_exact,
            "minimum_far_gap_exact_generation_passes": 1,
            "minimum_exact_generation_passes": minimum,
        }

    passed = all(checks.values())
    return passed, {
        "purpose": (
            "admission to expanded evaluation only; not final broad "
            "long-context capability evidence"
        ),
        "success_metric": (
            "literal whole decoded generated string exact with observed "
            "terminal EOS"
        ),
        "short_context_preservation": (
            "same-load 4096-token canary plus fresh-checkpoint candidate "
            "reload parity for hidden states, full-vocabulary logits, "
            "prefill/decode logits, and every KV-cache tensor"
        ),
        "short_context_independent_reload_parity": (
            "established for the registered 4096-token and 32+1 cache "
            "trajectories under the recorded BF16/Flash runtime"
        ),
        "adapter_chain": {
            "load_order": [
                "frozen_parent_qkvo",
                "eos_vocab_row_child",
            ],
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "parent_adaptation": PARENT_ADAPTATION,
            "parent_rank": PARENT_RANK,
            "parent_alpha": PARENT_ALPHA,
            "child_adapter_sha256": training["adapter_sha256"],
            "child_adaptation": ADAPTATION,
            "child_rank": CHILD_RANK,
            "child_alpha": CHILD_ALPHA,
            "child_trainable_scope": TRAINABLE_SCOPE,
        },
        "parent_exact_baseline": {
            name: parent_baseline[name]
            for name in (
                "result_sha256",
                "examples_sha256",
                "run_manifest_sha256",
                "adapter_sha256",
                "decision",
            )
        },
        "ignored_for_admission": [
            "answer_eos_token_exact",
            "first_number_exact",
            "official_string_match",
            "substring_match",
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
    parser.add_argument(
        "--experiment-ready-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--evaluation-examples", type=Path, required=True
    )
    parser.add_argument(
        "--evaluation-run-manifest", type=Path, required=True
    )
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
    ready_path = args.experiment_ready_receipt.resolve()
    experiment_ready = json.loads(
        ready_path.read_text(encoding="utf-8")
    )
    raw_validation = validate_raw_evaluation(
        training=training,
        evaluation=evaluation,
        experiment_ready=experiment_ready,
        experiment_ready_sha256=sha256_file(ready_path),
        examples_path=args.evaluation_examples.resolve(),
        run_manifest_path=args.evaluation_run_manifest.resolve(),
    )
    passed, details = exact_screen_gate(
        training,
        evaluation,
        raw_counts=raw_validation["counts"],
    )
    details["raw_failure_diagnostics"] = raw_validation[
        "failure_diagnostics"
    ]
    details["checkpoint_selection"] = raw_validation[
        "checkpoint_selection"
    ]
    details["raw_failure_diagnostics_are_success_metrics"] = False
    receipt = {
        "status": "PASS" if passed else "STOP",
        "gate": "OLMO2_LENGTH_GATED_MINIMAL_FULL_STRING_EXACT_EOS_V2",
        **details,
        "evidence": {
            "training_result_sha256": sha256_file(
                args.training_result.resolve()
            ),
            "evaluation_result_sha256": sha256_file(
                args.evaluation_result.resolve()
            ),
            "experiment_ready_receipt_sha256": sha256_file(ready_path),
            "evaluation_examples_sha256": sha256_file(
                args.evaluation_examples.resolve()
            ),
            "evaluation_run_manifest_sha256": sha256_file(
                args.evaluation_run_manifest.resolve()
            ),
        },
    }
    atomic_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
