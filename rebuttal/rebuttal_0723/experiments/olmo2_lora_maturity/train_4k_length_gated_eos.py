#!/usr/bin/env python3
"""Repair OLMo-2 terminal EOS in the length-gated EVQ+LoRA branch.

Every training tensor has at most 4096 physical tokens.  The long branch is
forced during this continuation so that the existing full-EVQ QKVO adapter can
be reused and repaired with answer-plus-immediate-EOS labels.  At inference,
the frozen Native/base branch is selected for sequences ending at position
4095 or earlier; sequences reaching position 4096 select full EVQ+QKVO LoRA
for the entire sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    load_model,
    save_adapter,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
    seed_everything,
    sha256_file,
)

from .olmo2_length_gated_method import (
    EVQ_FREQUENCY_SHA256,
    LENGTH_GATED_FREQUENCY_NAME,
    NATIVE_FREQUENCY_SHA256,
    SHORT_CONTEXT_LIMIT,
    LengthGatedLoRALinear,
    LengthModeState,
    eos_head_trainable_named_parameters,
    freeze_length_gated_qkvo_adapter,
    install_length_gated_eos_vocab_row_head,
    install_length_gated_qkvo,
)
from .evaluate_instruct_ruler_screen import (
    EXAMPLE_SCHEMA_VERSION,
    RUN_MANIFEST_STATUS,
    decode_generated_string,
    first_number_exact,
    validate_actual_prompt_geometry,
    validate_data,
)
from .gate_olmo2_exact_screen import _recompute_raw_counts
from .prepare_4k_routing_pairs import (
    LENGTH,
    OLMO2_EOS_TOKEN_ID,
    ROOT_STATUS,
    SUPERVISION_CONTRACT,
)
from .train_4k_counterfactual_routing import (
    CALIBRATION_QUERY_MARKER,
    TRAIN_QUERY_MARKER,
    RoutingPairView,
    deterministic_query_offset_stream,
    query_gap_position_ids,
    routing_batch,
    train,
)
from .train_4k_stage_a import ready_checkpoint_digest
from .train_screen import load_fixed_view


READY_STATUS = "OLMO2_4K_LENGTH_GATED_EOS_REPAIR_READY_V1"
RESULT_STATUS = "OLMO2_4K_LENGTH_GATED_EOS_REPAIR_COMPLETE_V1"
MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
PARENT_ADAPTER_SHA256 = (
    "a0ccd2cf141300ba4489882dda1324b2f237e65444a9a71d687e8c5fad57ae8b"
)
PRE_QUERY_GAP_PARENT_SHA256 = (
    "95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a"
)
PARENT_ADAPTATION = "qkvo_answer"
ADAPTATION = "length_gated_eos_vocab_row"
STEPS = 32
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
PARENT_RANK = 64
PARENT_ALPHA = 128.0
EOS_HEAD_RANK = 1
EOS_HEAD_ALPHA = 1.0
LEARNING_RATE = 1e-3
WARMUP_STEPS = 4
COUNTERFACTUAL_MARGIN = 1.0
COUNTERFACTUAL_MARGIN_WEIGHT = 0.0
TERMINATION_WEIGHT = 1.0
COMPILE_MODE = "max-autotune-no-cudagraphs"
VIRTUAL_TARGET_LENGTH = 4 * LENGTH
VIRTUAL_BUCKET_WEIGHTS = (1, 1, 2)
SEED = 20_260_728
TRAIN_FAMILY_PATTERN = ("routing",)
SELECTION_CHECKPOINT_STEPS = (4, 8, 16, 32)
SELECTION_ROWS = 8
SELECTION_QUERY_OFFSETS = (LENGTH, 3 * LENGTH + 1)
SELECTION_MIN_TERMINATION_EXACT = {
    LENGTH: 12,
    3 * LENGTH + 1: 4,
}
SELECTION_ANSWER_NLL_TOLERANCE = 1e-4
PARENT_EXACT_POLICY = {
    8_192: {"examples": 8, "minimum_prefixes": 6},
    16_384: {"examples": 8, "minimum_prefixes": 2},
}
MINIMUM_FREE_BYTES = 20 * 1024**3


def protocol() -> dict[str, Any]:
    routing_steps = sum(
        TRAIN_FAMILY_PATTERN[
            (step - 1) % len(TRAIN_FAMILY_PATTERN)
        ]
        == "routing"
        for step in range(1, STEPS + 1)
    )
    offset_stream = deterministic_query_offset_stream(
        seed=SEED,
        routing_steps=routing_steps,
    )
    return {
        "model_scope": "OLMo-2 1.485B Instruct only",
        "method": LENGTH_GATED_FREQUENCY_NAME,
        "frequency": LENGTH_GATED_FREQUENCY_NAME,
        "short_branch_frequency": "native_endpoint_rope",
        "short_branch_frequency_sha256_float32": (
            NATIVE_FREQUENCY_SHA256
        ),
        "short_branch_maximum_position_id": SHORT_CONTEXT_LIMIT - 1,
        "short_branch_lora": "frozen_base_linear_direct_call",
        "long_branch_frequency": "evq_endpoint_cosh",
        "long_branch_frequency_sha256_float32": EVQ_FREQUENCY_SHA256,
        "long_branch_minimum_maximum_position_id": SHORT_CONTEXT_LIMIT,
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
        "steps": STEPS,
        "family_pattern": list(TRAIN_FAMILY_PATTERN),
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "global_batch_size": (
            MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        ),
        "child_rank": EOS_HEAD_RANK,
        "child_alpha": EOS_HEAD_ALPHA,
        "trainable_scope": "long_only_eos_vocab_row_plus_scalar_bias",
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "counterfactual_margin": COUNTERFACTUAL_MARGIN,
        "counterfactual_margin_weight": COUNTERFACTUAL_MARGIN_WEIGHT,
        "termination_weight": TERMINATION_WEIGHT,
        "compile_mode": COMPILE_MODE,
        "training_branch": (
            "forced_long_full_evq_frozen_qkvo_parent_plus_eos_child"
        ),
        "maximum_physical_training_sequence_length": LENGTH,
        "maximum_physical_model_input_tokens": LENGTH - 1,
        "hard_maximum_training_length": LENGTH,
        "real_8k_or_16k_training_sequences": 0,
        "position_policy": "semantic_query_block_continuous_gap",
        "virtual_target_length": VIRTUAL_TARGET_LENGTH,
        "maximum_allowed_position_id": VIRTUAL_TARGET_LENGTH - 1,
        "virtual_bucket_weights": list(VIRTUAL_BUCKET_WEIGHTS),
        "routing_optimizer_steps": routing_steps,
        "routing_pair_exposures": int(len(offset_stream)),
        "query_offset_stream_sha256": hashlib.sha256(
            offset_stream.tobytes(order="C")
        ).hexdigest(),
        "routing_data_format_version": 2,
        "supervision_contract": SUPERVISION_CONTRACT,
        "supervision": (
            "causally_shifted_answer_ce_plus_weighted_immediate_eos_ce"
        ),
        "eos_token_id": OLMO2_EOS_TOKEN_ID,
        "final_eos_supervised": True,
        "checkpoint_selection": {
            "rule": "first_passing_checkpoint",
            "maximum_steps": STEPS,
            "checkpoint_steps": list(SELECTION_CHECKPOINT_STEPS),
            "split": "calibration",
            "rows": SELECTION_ROWS,
            "row_indices": list(range(SELECTION_ROWS)),
            "query_offsets": list(SELECTION_QUERY_OFFSETS),
            "minimum_termination_exact_counts": {
                str(offset): int(count)
                for offset, count in sorted(
                    SELECTION_MIN_TERMINATION_EXACT.items()
                )
            },
            "answer_nll_tolerance": (
                SELECTION_ANSWER_NLL_TOLERANCE
            ),
            "selection_is_capability_evidence": False,
        },
        "natural_replay": "none_eos_only_parent_preserving_repair",
        "short_branch_preservation": (
            "EVQ, parent QKVO, and EOS child are bypassed for "
            "max_position_id_le_4095"
        ),
        "expanded_evaluation_before_minimal_exact_gate": False,
        "minimum_free_bytes_before_run": MINIMUM_FREE_BYTES,
        "seed": SEED,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-ready-receipt", type=Path, required=True
    )
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--routing-data", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument(
        "--parent-exact-baseline-result", type=Path, required=True
    )
    parser.add_argument(
        "--parent-exact-baseline-examples", type=Path, required=True
    )
    parser.add_argument(
        "--parent-exact-baseline-run-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validate_registered_adapter_chain(
    *,
    receipt: dict[str, Any],
    parent_adapter: Path,
    output: Path,
) -> None:
    expected = {
        "load_order": [
            "frozen_parent_qkvo",
            "eos_vocab_row_child",
        ],
        "parent": {
            "path": str(parent_adapter.resolve()),
            "sha256": PARENT_ADAPTER_SHA256,
            "adaptation": PARENT_ADAPTATION,
            "rank": PARENT_RANK,
            "alpha": PARENT_ALPHA,
            "frozen_during_child_training": True,
        },
        "child": {
            "path": str((output / "adapter.pt").resolve()),
            "adaptation": ADAPTATION,
            "rank": EOS_HEAD_RANK,
            "alpha": EOS_HEAD_ALPHA,
            "modified_vocab_rows": [OLMO2_EOS_TOKEN_ID],
            "parameterization": "direct_single_eos_row_delta",
            "rank1_equivalent": True,
            "trainable_parameters": 2_049,
            "trainable_parameter_tensors": 2,
            "trainable_scope": (
                "long_only_eos_vocab_row_plus_scalar_bias"
            ),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
        },
        "code_sha256": {
            "candidate_evaluator": receipt["code"]["exact_evaluator"][
                "sha256"
            ],
            "parent_adapter_loader": receipt["code"]["adapter_loader"][
                "sha256"
            ],
            "parent_child_method": receipt["code"]["method"]["sha256"],
        },
    }
    if receipt.get("registered_adapter_chain") != expected:
        raise RuntimeError("length-gated READY adapter-chain drift")


def _validate_ready(
    *,
    path: Path,
    checkpoint: Path,
    checkpoint_ready: Path,
    parent_adapter: Path,
    prepared_data: Path,
    routing_data: Path,
    output: Path,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != READY_STATUS:
        raise RuntimeError("length-gated READY status drift")
    if receipt.get("protocol") != protocol():
        raise RuntimeError("length-gated READY protocol drift")
    expected_paths = {
        "checkpoint": checkpoint,
        "checkpoint_ready_receipt": checkpoint_ready,
        "parent_adapter": parent_adapter,
        "prepared_data": prepared_data,
        "routing_data": routing_data,
    }
    for name, expected in expected_paths.items():
        observed = Path(receipt["inputs"][name]["path"]).resolve()
        if observed != expected.resolve():
            raise RuntimeError(f"length-gated READY {name} path drift")
    if Path(receipt["run_output"]).resolve() != output.resolve():
        raise RuntimeError("length-gated READY output drift")

    trainer = Path(__file__).resolve()
    expected_code = {
        "preflight": trainer.with_name(
            "preflight_4k_length_gated_eos.py"
        ),
        "preflight_validation_helpers": trainer.with_name(
            "preflight_4k_query_gap_eos_repair.py"
        ),
        "trainer": trainer,
        "shared_trainer": trainer.with_name(
            "train_4k_counterfactual_routing.py"
        ),
        "routing_data_contract": trainer.with_name(
            "prepare_4k_routing_pairs.py"
        ),
        "training_primitives": trainer.with_name("train_screen.py"),
        "checkpoint_contract": trainer.with_name("train_4k_stage_a.py"),
        "method": trainer.with_name("olmo2_length_gated_method.py"),
        "conversion": trainer.parents[1] / "olmo2_lora_conversion.py",
        "model_loader_attention_dependency": (
            trainer.parents[1] / "olmo2_1b_evq" / "train.py"
        ),
        "adapter_loader": (
            trainer.parents[1] / "olmo2_lora_ood_factorial.py"
        ),
        "shared_training_utils": (
            trainer.parents[1] / "small_model_lora_conversion.py"
        ),
        "evq_contract": (
            trainer.parents[1] / "olmo2_1b_evq" / "contract.py"
        ),
        "exact_evaluator": trainer.with_name(
            "evaluate_instruct_ruler_screen.py"
        ),
        "exact_gate": trainer.with_name(
            "gate_olmo2_length_gated_exact_screen.py"
        ),
        "parent_raw_exact_validator": trainer.with_name(
            "gate_olmo2_exact_screen.py"
        ),
    }
    for name, expected in expected_code.items():
        entry = receipt["code"][name]
        if (
            Path(entry["path"]).resolve() != expected
            or entry["sha256"] != sha256_file(expected)
        ):
            raise RuntimeError(
                f"length-gated {name} changed after READY"
            )
    _validate_registered_adapter_chain(
        receipt=receipt,
        parent_adapter=parent_adapter,
        output=output,
    )
    if (
        receipt["inputs"]["checkpoint_ready_receipt"]["sha256"]
        != sha256_file(checkpoint_ready)
    ):
        raise RuntimeError("checkpoint READY receipt changed")
    for name, relative_path in (
        ("config", "config.json"),
        ("model", "model.safetensors"),
        ("tokenizer", "tokenizer.json"),
    ):
        if (
            receipt["inputs"]["checkpoint"][name]["sha256"]
            != sha256_file(checkpoint / relative_path)
        ):
            raise RuntimeError(
                f"checkpoint {relative_path} changed after READY"
            )
    if (
        receipt["inputs"]["parent_adapter"]["sha256"]
        != sha256_file(parent_adapter)
    ):
        raise RuntimeError("parent adapter changed after READY")
    if (
        receipt["inputs"]["routing_data"]["manifest_sha256"]
        != sha256_file(routing_data / "manifest.json")
    ):
        raise RuntimeError("routing data changed after READY")
    for split_name, split in receipt["inputs"]["routing_data"][
        "splits"
    ].items():
        split_root = routing_data / split_name
        if split["manifest"]["sha256"] != sha256_file(
            split_root / "manifest.json"
        ):
            raise RuntimeError(
                f"routing {split_name} manifest changed after READY"
            )
        for filename, entry in split["files"].items():
            if entry["sha256"] != sha256_file(split_root / filename):
                raise RuntimeError(
                    f"routing {split_name}/{filename} changed after READY"
                )
    natural_manifest = (
        prepared_data / "longalign_paired_L4096" / "manifest.json"
    )
    if (
        receipt["inputs"]["prepared_data"]["natural_manifest_sha256"]
        != sha256_file(natural_manifest)
    ):
        raise RuntimeError("natural replay changed after READY")
    for filename, entry in receipt["inputs"]["prepared_data"][
        "natural_replay"
    ]["files"].items():
        if entry["sha256"] != sha256_file(
            natural_manifest.parent / filename
        ):
            raise RuntimeError(
                f"natural replay {filename} changed after READY"
            )
    return receipt


def _load_routing_views(
    *,
    checkpoint: Path,
    routing_root: Path,
) -> tuple[RoutingPairView, RoutingPairView]:
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if int(tokenizer.eos_token_id) != OLMO2_EOS_TOKEN_ID:
        raise RuntimeError("OLMo tokenizer EOS identity drift")
    train_marker = tuple(
        int(value)
        for value in tokenizer(
            TRAIN_QUERY_MARKER,
            add_special_tokens=False,
        ).input_ids
    )
    calibration_marker = tuple(
        int(value)
        for value in tokenizer(
            CALIBRATION_QUERY_MARKER,
            add_special_tokens=False,
        ).input_ids
    )
    return (
        RoutingPairView(
            routing_root / "train",
            require_virtual_geometry=True,
            query_marker_token_ids=train_marker,
        ),
        RoutingPairView(
            routing_root / "calibration",
            require_virtual_geometry=True,
            query_marker_token_ids=calibration_marker,
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_parent_exact_baseline(
    *,
    result_path: Path,
    examples_path: Path,
    run_manifest_path: Path,
    ready: dict[str, Any],
    ready_path: Path,
    checkpoint: Path,
    parent_adapter: Path,
) -> dict[str, Any]:
    registered = Path(
        ready["registered_outputs"]["parent_exact_baseline"]
    ).resolve()
    if (
        result_path.resolve() != registered / "results.json"
        or examples_path.resolve() != registered / "examples.jsonl"
        or run_manifest_path.resolve()
        != registered / "run_manifest.json"
    ):
        raise RuntimeError("parent exact baseline path is not registered")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(
        run_manifest_path.read_text(encoding="utf-8")
    )
    ready_sha = sha256_file(ready_path)
    parent_sha = sha256_file(parent_adapter)
    if (
        result.get("status")
        != "OLMO2_INSTRUCT_RULER_EXACT_SCREEN_COMPLETE_V3"
        or result.get("experiment_ready_receipt_sha256") != ready_sha
        or result.get("adapter", {}).get("sha256") != parent_sha
        or result.get("run_manifest_sha256")
        != sha256_file(run_manifest_path)
        or result.get("results", {}).get("examples_sha256")
        != sha256_file(examples_path)
        or run_manifest.get("status") != RUN_MANIFEST_STATUS
        or int(run_manifest.get("example_schema_version", -1))
        != EXAMPLE_SCHEMA_VERSION
        or run_manifest.get("experiment_role")
        != "parent_exact_baseline"
        or run_manifest.get("experiment_ready_receipt_sha256")
        != ready_sha
        or run_manifest.get("adapter_sha256") != parent_sha
        or run_manifest.get("frequency") != "evq"
        or run_manifest.get("adaptation") != "qkvo_answer"
        or run_manifest.get("task") != "niah_single_1"
        or run_manifest.get("lengths")
        != list(PARENT_EXACT_POLICY)
        or int(run_manifest.get("limit_per_length", -1)) != 8
        or run_manifest.get("greedy") is not True
    ):
        raise RuntimeError("registered parent exact baseline receipt drift")
    protocol_receipt = result.get("protocol", {})
    if (
        protocol_receipt.get("task") != "niah_single_1"
        or protocol_receipt.get("lengths")
        != list(PARENT_EXACT_POLICY)
        or int(protocol_receipt.get("limit_per_length", -1)) != 8
        or protocol_receipt.get("greedy") is not True
        or protocol_receipt.get("string_normalization") != "none"
        or protocol_receipt.get("decode_cleanup") is not False
        or protocol_receipt.get("substring_is_success") is not False
        or protocol_receipt.get("first_number_is_success") is not False
    ):
        raise RuntimeError("parent exact baseline protocol drift")

    ready_data = ready["inputs"]["exact_eval_data"]
    data_root = Path(ready_data["path"]).resolve()
    data_receipt, selected_rows = validate_data(
        data_root,
        checkpoint,
        "niah_single_1",
        tuple(PARENT_EXACT_POLICY),
        8,
    )
    if (
        result.get("data") != data_receipt
        or data_receipt["manifest_sha256"]
        != ready_data["manifest_sha256"]
        or {
            name: entry["sha256"]
            for name, entry in data_receipt["files"].items()
        }
        != {
            name: entry["sha256"]
            for name, entry in ready_data["files"].items()
        }
    ):
        raise RuntimeError("parent exact baseline frozen-data drift")
    source_rows = {
        (int(row["_nominal_length"]), int(row["_local_index"])): {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        for row in selected_rows
    }
    raw_rows = _read_jsonl(examples_path)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    actual_prompt_geometry = validate_actual_prompt_geometry(
        rows=selected_rows,
        tokenizer=tokenizer,
    )
    if (
        ready_data.get("actual_prompt_geometry")
        != actual_prompt_geometry
        or result.get("actual_prompt_geometry")
        != actual_prompt_geometry
        or run_manifest.get("actual_prompt_geometry")
        != actual_prompt_geometry
    ):
        raise RuntimeError(
            "parent exact baseline actual prompt geometry drift"
        )
    geometry_by_key = {
        (
            int(row["nominal_length"]),
            int(row["local_index"]),
        ): row
        for row in actual_prompt_geometry["rows"]
    }

    def decode_generated(token_ids: list[int]) -> str:
        return decode_generated_string(
            tokenizer=tokenizer,
            generated_token_ids=token_ids,
            eos_token_id=OLMO2_EOS_TOKEN_ID,
        )

    counts = _recompute_raw_counts(
        evaluation=result,
        raw_rows=raw_rows,
        source_rows=source_rows,
        decode_generated=decode_generated,
        policy=PARENT_EXACT_POLICY,
    )
    rows_by_key = {
        (
            int(row["nominal_length"]),
            int(row["local_index"]),
        ): row
        for row in raw_rows
    }
    observed_repairable_prefixes: dict[str, dict[str, int]] = {}
    already_passes_all_lengths = True
    for length, policy in PARENT_EXACT_POLICY.items():
        canonical_prefixes = 0
        far_gap_canonical_prefixes = 0
        far_gap_exact_generation_passes = 0
        canonical_prefix_then_eos = 0
        canonical_prefix_then_extra = 0
        first_number_count = 0
        for local_index in range(int(policy["examples"])):
            row = rows_by_key[(length, local_index)]
            generated = [int(value) for value in row["generated_token_ids"]]
            references = [
                [int(value) for value in values]
                for values in row["reference_token_ids"]
            ]
            expected_reference_token_ids = [
                [
                    int(value)
                    for value in tokenizer(
                        reference,
                        add_special_tokens=False,
                    ).input_ids
                ]
                for reference in row["references"]
            ]
            if references != expected_reference_token_ids:
                raise RuntimeError(
                    f"L{length}/row{local_index} reference token IDs drift"
                )
            prefix_length = next(
                (
                    len(reference)
                    for reference in references
                    if generated[: len(reference)] == reference
                ),
                None,
            )
            if prefix_length is not None:
                canonical_prefixes += 1
                if bool(
                    geometry_by_key[(length, local_index)][
                        "far_gap_beyond_training_support"
                    ]
                ):
                    far_gap_canonical_prefixes += 1
                if (
                    len(generated) > prefix_length
                    and generated[prefix_length] == OLMO2_EOS_TOKEN_ID
                ):
                    canonical_prefix_then_eos += 1
                else:
                    canonical_prefix_then_extra += 1
            if (
                bool(
                    geometry_by_key[(length, local_index)][
                        "far_gap_beyond_training_support"
                    ]
                )
                and float(row["exact_generation_pass"]) == 1.0
            ):
                far_gap_exact_generation_passes += 1
            expected_first = first_number_exact(
                str(row["prediction"]),
                [str(value) for value in row["references"]],
            )
            if float(row.get("first_number_exact", float("nan"))) != float(
                expected_first
            ):
                raise RuntimeError(
                    f"L{length}/row{local_index} first-number diagnostic drift"
                )
            first_number_count += int(expected_first)
        observed_repairable_prefixes[str(length)] = {
            "examples": int(policy["examples"]),
            "observed_complete_gold_token_prefixes": canonical_prefixes,
            "observed_far_gap_complete_gold_token_prefixes": (
                far_gap_canonical_prefixes
            ),
            "observed_far_gap_exact_generation_passes": (
                far_gap_exact_generation_passes
            ),
            "canonical_prefix_then_eos": canonical_prefix_then_eos,
            "canonical_prefix_then_extra": canonical_prefix_then_extra,
            "first_number_exact": first_number_count,
            "minimum_observed_repairable_prefixes": int(
                policy["minimum_prefixes"]
            ),
            "interpretation": (
                "conservative lower bound from free generation; not a "
                "teacher-forced reachability ceiling"
            ),
        }
        if canonical_prefixes < int(policy["minimum_prefixes"]):
            raise RuntimeError(
                f"L{length} parent has only {canonical_prefixes}/8 complete "
                "gold-token prefixes; EOS-only repair cannot reach the gate"
            )
        if far_gap_canonical_prefixes < 1:
            raise RuntimeError(
                f"L{length} parent has no complete gold-token prefix on a "
                "source-to-generation gap beyond 3933 tokens; EOS-only "
                "repair is not established at the extrapolative gap"
            )
        already_passes_all_lengths = (
            already_passes_all_lengths
            and int(counts[str(length)]["exact_generation_passes"])
            >= int(policy["minimum_prefixes"])
            and far_gap_exact_generation_passes >= 1
        )
    if already_passes_all_lengths:
        raise RuntimeError(
            "parent already passes every minimal strict gate; do not spend "
            "GPU time on EOS repair"
        )
    return {
        "result_sha256": sha256_file(result_path),
        "examples_sha256": sha256_file(examples_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "adapter_sha256": parent_sha,
        "strict_counts": counts,
        "actual_prompt_geometry": actual_prompt_geometry,
        "observed_repairable_prefix_lower_bound": (
            observed_repairable_prefixes
        ),
        "decision": "EOS_ONLY_REPAIR_FEASIBLE",
    }


def _validate_parent_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "base_checkpoint_sha256": MODEL_SHA256,
        "frequency": "evq",
        "adaptation": PARENT_ADAPTATION,
        "rank": PARENT_RANK,
        "alpha": PARENT_ALPHA,
        "training_sequence_length": LENGTH,
        "stage": "counterfactual_routing_semantic_query_gap_16k",
        "parent_adapter_sha256": PRE_QUERY_GAP_PARENT_SHA256,
        "position_policy": "semantic_query_block_continuous_gap",
        "virtual_target_length": VIRTUAL_TARGET_LENGTH,
    }
    for name, expected_value in expected.items():
        if metadata.get(name) != expected_value:
            raise RuntimeError(
                f"query-gap parent metadata drift for {name}"
            )


def _assert_trainable_scope(model: Any) -> dict[str, Any]:
    named = eos_head_trainable_named_parameters(model)
    names = [name for name, _ in named]
    expected = {
        "model.lm_head.delta_weight",
        "model.lm_head.eos_bias",
    }
    if set(names) != expected:
        raise RuntimeError("length-gated trainable scope escaped EOS child")
    return {
        "scope": "long_only_eos_vocab_row_plus_scalar_bias",
        "parameter_names": names,
        "parameter_tensors": len(names),
        "parameters": int(
            sum(parameter.numel() for _, parameter in named)
        ),
    }


def _named_tensor_sha256(
    values: list[tuple[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(values, key=lambda item: item[0]):
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


@torch.inference_mode()
def _build_eos_selection_cache(
    *,
    model: Any,
    state: LengthModeState,
    view: RoutingPairView,
) -> dict[str, Any]:
    if (
        view.manifest.get("purpose")
        != "counterfactual_routing_calibration"
        or len(view.input_ids) < SELECTION_ROWS
    ):
        raise RuntimeError("EOS selection requires the frozen calibration set")
    row_indices = np.arange(SELECTION_ROWS, dtype=np.int64)
    subset_payload = json.dumps(
        [view.rows[int(index)] for index in row_indices],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    runtime_cells: dict[str, dict[str, torch.Tensor]] = {}
    receipt_cells: dict[str, dict[str, Any]] = {}
    state.force("long")
    model.eval()
    for query_offset in SELECTION_QUERY_OFFSETS:
        hidden_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        position_digest = hashlib.sha256()
        for start in range(0, SELECTION_ROWS, 2):
            indices = row_indices[start : start + 2]
            contexts, labels, alternate_labels, _ = routing_batch(
                view=view,
                row_indices=indices,
            )
            position_ids, _, payload = query_gap_position_ids(
                view=view,
                row_indices=indices,
                query_offsets=np.full(
                    len(indices),
                    int(query_offset),
                    dtype=np.int64,
                ),
            )
            position_digest.update(payload)
            mask = labels != -100
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(
                    input_ids=contexts,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=False,
                )[0]
            if state.mode != "long":
                raise RuntimeError("EOS selection cache used the wrong branch")
            hidden_parts.append(hidden[mask].detach())
            label_parts.append(labels[mask].detach())
            del (
                contexts,
                labels,
                alternate_labels,
                position_ids,
                hidden,
            )
        selected_hidden = torch.cat(hidden_parts, dim=0)
        selected_labels = torch.cat(label_parts, dim=0)
        termination_tokens = int(
            selected_labels.eq(OLMO2_EOS_TOKEN_ID).sum()
        )
        if termination_tokens != 2 * SELECTION_ROWS:
            raise RuntimeError("EOS selection termination count drift")
        key = str(int(query_offset))
        runtime_cells[key] = {
            "hidden": selected_hidden,
            "labels": selected_labels,
        }
        receipt_cells[key] = {
            "query_offset": int(query_offset),
            "pairs": SELECTION_ROWS,
            "variants": 2 * SELECTION_ROWS,
            "supervised_tokens": int(selected_labels.numel()),
            "answer_tokens": int(
                selected_labels.ne(OLMO2_EOS_TOKEN_ID).sum()
            ),
            "termination_tokens": termination_tokens,
            "position_ids_sha256": position_digest.hexdigest(),
            "selected_hidden_sha256": _named_tensor_sha256(
                [(f"query_offset_{query_offset}", selected_hidden)]
            ),
            "selected_labels_sha256": _named_tensor_sha256(
                [(f"query_offset_{query_offset}", selected_labels)]
            ),
        }
    return {
        "runtime_cells": runtime_cells,
        "receipt": {
            "split": "calibration",
            "purpose": view.manifest["purpose"],
            "rows": SELECTION_ROWS,
            "row_indices": row_indices.tolist(),
            "query_offsets": list(SELECTION_QUERY_OFFSETS),
            "manifest_sha256": sha256_file(
                view.path / "manifest.json"
            ),
            "files": dict(view.manifest["files"]),
            "subset_rows_sha256": hashlib.sha256(
                subset_payload
            ).hexdigest(),
            "cells": receipt_cells,
        },
    }


@torch.inference_mode()
def _score_eos_selection_cache(
    *,
    model: Any,
    cache: dict[str, Any],
) -> dict[str, Any]:
    head = model.lm_head
    for name in (
        "base",
        "delta_weight",
        "eos_bias",
        "eos_token_id",
        "length_mode_state",
    ):
        if not hasattr(head, name):
            raise RuntimeError("EOS selection head identity drift")
    if int(head.eos_token_id) != OLMO2_EOS_TOKEN_ID:
        raise RuntimeError("EOS selection token identity drift")
    head.length_mode_state.force("long")
    model.eval()
    cells: dict[str, dict[str, Any]] = {}
    for key, cell in cache["runtime_cells"].items():
        hidden = cell["hidden"]
        labels = cell["labels"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(hidden).float()
            base_eos = F.linear(
                hidden,
                head.base.weight[
                    OLMO2_EOS_TOKEN_ID : OLMO2_EOS_TOKEN_ID + 1
                ],
                (
                    None
                    if head.base.bias is None
                    else head.base.bias[
                        OLMO2_EOS_TOKEN_ID : OLMO2_EOS_TOKEN_ID + 1
                    ]
                ),
            ).squeeze(-1).float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite EOS selection logits")
        nll = F.cross_entropy(logits, labels, reduction="none")
        termination_mask = labels.eq(OLMO2_EOS_TOKEN_ID)
        answer_mask = ~termination_mask
        prediction = logits.argmax(dim=-1)
        eos_logits = logits[:, OLMO2_EOS_TOKEN_ID]
        max_non_eos = torch.maximum(
            logits[:, :OLMO2_EOS_TOKEN_ID].amax(dim=-1),
            logits[:, OLMO2_EOS_TOKEN_ID + 1 :].amax(dim=-1),
        )
        eos_delta = eos_logits - base_eos
        answer_eos_bits = prediction[answer_mask].eq(
            OLMO2_EOS_TOKEN_ID
        )
        cells[key] = {
            "query_offset": int(key),
            "supervised_tokens": int(labels.numel()),
            "answer_tokens": int(answer_mask.sum()),
            "termination_tokens": int(termination_mask.sum()),
            "answer_token_exact_count": int(
                prediction[answer_mask]
                .eq(labels[answer_mask])
                .sum()
            ),
            "answer_eos_top1_count": int(answer_eos_bits.sum()),
            "answer_eos_top1_bits": [
                bool(value)
                for value in answer_eos_bits.detach().cpu().tolist()
            ],
            "mean_answer_nll": float(nll[answer_mask].mean()),
            "termination_exact_count": int(
                prediction[termination_mask]
                .eq(OLMO2_EOS_TOKEN_ID)
                .sum()
            ),
            "mean_termination_nll": float(
                nll[termination_mask].mean()
            ),
            "minimum_termination_eos_margin": float(
                (
                    eos_logits[termination_mask]
                    - max_non_eos[termination_mask]
                ).min()
            ),
            "mean_answer_eos_logit_delta": float(
                eos_delta[answer_mask].mean()
            ),
            "maximum_answer_eos_logit_delta": float(
                eos_delta[answer_mask].max()
            ),
            "mean_termination_eos_logit_delta": float(
                eos_delta[termination_mask].mean()
            ),
            "minimum_termination_eos_logit_delta": float(
                eos_delta[termination_mask].min()
            ),
        }
    parameters = [
        ("model.lm_head.delta_weight", head.delta_weight),
        ("model.lm_head.eos_bias", head.eos_bias),
    ]
    return {
        "cells": cells,
        "parameter_state": {
            "sha256": _named_tensor_sha256(parameters),
            "delta_weight_l2": float(
                head.delta_weight.float().norm()
            ),
            "delta_weight_maximum_absolute": float(
                head.delta_weight.float().abs().max()
            ),
            "eos_bias": float(head.eos_bias),
            "finite": bool(
                torch.isfinite(head.delta_weight).all()
                and torch.isfinite(head.eos_bias)
            ),
        },
    }


def _eos_selection_gate(
    *,
    parent: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if set(parent["cells"]) != {
        str(value) for value in SELECTION_QUERY_OFFSETS
    } or set(candidate["cells"]) != set(parent["cells"]):
        raise RuntimeError("EOS selection cell identity drift")
    checks: dict[str, bool] = {
        "finite_parameter_state": bool(
            candidate["parameter_state"]["finite"]
        ),
        "child_state_changed": (
            candidate["parameter_state"]["sha256"]
            != parent["parameter_state"]["sha256"]
        ),
    }
    cells: dict[str, dict[str, Any]] = {}
    for query_offset in SELECTION_QUERY_OFFSETS:
        key = str(query_offset)
        parent_cell = parent["cells"][key]
        candidate_cell = candidate["cells"][key]
        identity_fields = (
            "supervised_tokens",
            "answer_tokens",
            "termination_tokens",
        )
        if any(
            int(candidate_cell[name]) != int(parent_cell[name])
            for name in identity_fields
        ):
            raise RuntimeError("EOS selection token geometry drift")
        parent_bits = list(parent_cell["answer_eos_top1_bits"])
        candidate_bits = list(candidate_cell["answer_eos_top1_bits"])
        if len(parent_bits) != len(candidate_bits):
            raise RuntimeError("EOS selection answer-bit geometry drift")
        new_answer_eos = sum(
            bool(observed) and not bool(baseline)
            for baseline, observed in zip(parent_bits, candidate_bits)
        )
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
            "no_new_answer_eos_top1": new_answer_eos == 0,
            "answer_nll_not_worse": (
                float(candidate_cell["mean_answer_nll"])
                <= float(parent_cell["mean_answer_nll"])
                + SELECTION_ANSWER_NLL_TOLERANCE
            ),
        }
        checks.update(
            {
                f"{prefix}_{name}": value
                for name, value in cell_checks.items()
            }
        )
        cells[key] = {
            **candidate_cell,
            "parent_termination_exact_count": int(
                parent_cell["termination_exact_count"]
            ),
            "parent_answer_token_exact_count": int(
                parent_cell["answer_token_exact_count"]
            ),
            "parent_mean_answer_nll": float(
                parent_cell["mean_answer_nll"]
            ),
            "new_answer_eos_top1_count": int(new_answer_eos),
            "checks": cell_checks,
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "cells": cells,
        "parameter_state": dict(candidate["parameter_state"]),
    }


def _require_selected_checkpoint(training: dict[str, Any]) -> int:
    selected = training.get("selected_step")
    if (
        training.get("checkpoint_selection_required") is not True
        or training.get("checkpoint_selection_passed") is not True
        or selected not in SELECTION_CHECKPOINT_STEPS
    ):
        raise RuntimeError(
            "no eligible EOS checkpoint; stop before adapter promotion"
        )
    history = list(training.get("checkpoint_history", []))
    passing_steps = [
        int(row["step"]) for row in history if row.get("passed") is True
    ]
    if passing_steps != [int(selected)]:
        raise RuntimeError("EOS checkpoint first-pass selection drift")
    return int(selected)


def _qkvo_parent_state(model: Any) -> tuple[str, int, int]:
    values: list[tuple[str, torch.Tensor]] = []
    modules = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, LengthGatedLoRALinear):
            continue
        modules += 1
        values.extend(
            (
                (f"model.{module_name}.a", module.a),
                (f"model.{module_name}.b", module.b),
            )
        )
    if modules != 64 or len(values) != 128:
        raise RuntimeError("length-gated QKVO parent tensor set drift")
    return _named_tensor_sha256(values), modules, len(values)


def _training_log_receipt(
    path: Path,
    *,
    training: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual_steps = int(training["actual_steps"])
    expected_steps = sorted(
        {
            1,
            actual_steps,
            *(
                value
                for value in SELECTION_CHECKPOINT_STEPS
                if value <= actual_steps
            ),
            *(value for value in (25,) if value <= actual_steps),
        }
    )
    if [int(row.get("step", -1)) for row in rows] != expected_steps:
        raise RuntimeError("length-gated optimizer-step log cadence drift")
    for label, row in (("first", rows[0]), ("last", rows[-1])):
        for name in (
            "loss",
            "grad_norm",
            "interval_tokens_per_second",
        ):
            if not math.isfinite(float(row.get(name, float("nan")))):
                raise RuntimeError(
                    f"{label} optimizer step has non-finite {name}"
                )
        if (
            int(row.get("processed_input_tokens", 0)) <= 0
            or int(row.get("peak_memory_allocated_bytes", 0)) <= 0
        ):
            raise RuntimeError(
                f"{label} optimizer step lacks token or memory receipt"
            )

    def selected(row: dict[str, Any]) -> dict[str, Any]:
        return {
            name: row[name]
            for name in (
                "step",
                "family",
                "loss",
                "lr",
                "grad_norm",
                "processed_input_tokens",
                "elapsed_seconds",
                "interval_tokens_per_second",
                "peak_memory_allocated_bytes",
                "peak_memory_reserved_bytes",
            )
        }

    return {
        "relative_path": path.name,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "first_optimizer_step": selected(rows[0]),
        "last_optimizer_step": selected(rows[-1]),
    }


@torch.no_grad()
def _short_outputs(
    *,
    model: Any,
    input_ids: torch.Tensor,
    state: LengthModeState | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state is not None:
        state.force(None)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
        )[0]
        final_logits = model.lm_head(hidden[:, -1:, :])
    if state is not None and state.mode != "short":
        raise RuntimeError("4096-token parity input did not use short mode")
    if not torch.isfinite(hidden).all() or not torch.isfinite(
        final_logits
    ).all():
        raise RuntimeError("non-finite short-branch parity output")
    return (
        hidden.detach().cpu().contiguous(),
        final_logits.detach().cpu().contiguous(),
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output)
    disk_before = shutil.disk_usage(output.parent)
    if int(disk_before.free) < MINIMUM_FREE_BYTES:
        raise RuntimeError("training output filesystem has less than 20 GiB")
    checkpoint = args.checkpoint.resolve()
    checkpoint_ready = args.checkpoint_ready_receipt.resolve()
    parent_adapter = args.parent_adapter.resolve()
    prepared_data = args.prepared_data.resolve()
    routing_root = args.routing_data.resolve()
    ready_path = args.ready_receipt.resolve()
    parent_exact_result = (
        args.parent_exact_baseline_result.resolve()
    )
    parent_exact_examples = (
        args.parent_exact_baseline_examples.resolve()
    )
    parent_exact_run_manifest = (
        args.parent_exact_baseline_run_manifest.resolve()
    )

    checkpoint_digest = ready_checkpoint_digest(
        checkpoint,
        checkpoint_ready,
    )
    if checkpoint_digest != MODEL_SHA256:
        raise RuntimeError("OLMo-2 1.485B checkpoint SHA drift")
    if sha256_file(parent_adapter) != PARENT_ADAPTER_SHA256:
        raise RuntimeError("query-gap parent adapter SHA drift")
    ready = _validate_ready(
        path=ready_path,
        checkpoint=checkpoint,
        checkpoint_ready=checkpoint_ready,
        parent_adapter=parent_adapter,
        prepared_data=prepared_data,
        routing_data=routing_root,
        output=output,
    )
    parent_exact_baseline = _validate_parent_exact_baseline(
        result_path=parent_exact_result,
        examples_path=parent_exact_examples,
        run_manifest_path=parent_exact_run_manifest,
        ready=ready,
        ready_path=ready_path,
        checkpoint=checkpoint,
        parent_adapter=parent_adapter,
    )

    routing_manifest = json.loads(
        (routing_root / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        routing_manifest.get("status") != ROOT_STATUS
        or int(routing_manifest.get("format_version", -1)) != 2
        or routing_manifest.get("supervision_contract")
        != SUPERVISION_CONTRACT
        or routing_manifest.get("final_eos_supervised") is not True
        or routing_manifest.get(
            "labels_only_cover_answer_and_final_eos"
        )
        is not True
        or routing_manifest.get(
            "answer_string_tokenizer_roundtrip_exact"
        )
        is not True
        or int(
            routing_manifest.get(
                "hard_maximum_training_length", -1
            )
        )
        != LENGTH
    ):
        raise RuntimeError("length-gated routing collection contract drift")

    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    allocator = (
        os.environ.get("PYTORCH_ALLOC_CONF")
        or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        or ""
    )
    if not cache:
        raise RuntimeError("persistent TORCHINDUCTOR_CACHE_DIR is required")
    if "expandable_segments:True" not in allocator:
        raise RuntimeError("expandable_segments allocator is required")

    routing_view, calibration_view = _load_routing_views(
        checkpoint=checkpoint,
        routing_root=routing_root,
    )
    natural_path = prepared_data / "longalign_paired_L4096"
    natural_view = load_fixed_view(natural_path)
    first_row = int(natural_view.training_rows[0])
    parity_ids = torch.from_numpy(
        np.asarray(
            natural_view.input_ids[first_row : first_row + 1],
            dtype=np.int64,
        ).copy()
    )
    if tuple(parity_ids.shape) != (1, LENGTH):
        raise RuntimeError("short-branch parity row is not exactly 4096 tokens")

    incomplete.mkdir(parents=True)
    seed_everything(SEED)
    runtime = configure_cuda()
    model = load_model(checkpoint)
    model.to("cuda")
    parity_ids = parity_ids.to("cuda", non_blocking=True)
    native_hidden, native_final_logits = _short_outputs(
        model=model,
        input_ids=parity_ids,
    )

    state, parent_method = install_length_gated_qkvo(
        model,
        rank=PARENT_RANK,
        alpha=PARENT_ALPHA,
    )
    parent_metadata = load_adapter(parent_adapter, model, None)
    _validate_parent_metadata(parent_metadata)
    parent_state_before, parent_modules, parent_tensors = (
        _qkvo_parent_state(model)
    )
    parent_freeze = freeze_length_gated_qkvo_adapter(model)
    _, child_method = install_length_gated_eos_vocab_row_head(
        model,
        state,
        rank=EOS_HEAD_RANK,
        alpha=EOS_HEAD_ALPHA,
        eos_token_id=OLMO2_EOS_TOKEN_ID,
    )
    model.to("cuda")
    trainable = _assert_trainable_scope(model)
    short_before_hidden, short_before_final_logits = _short_outputs(
        model=model,
        state=state,
        input_ids=parity_ids,
    )
    if (
        not torch.equal(native_hidden, short_before_hidden)
        or not torch.equal(
            native_final_logits, short_before_final_logits
        )
    ):
        raise RuntimeError(
            "length-gated short branch differs from pristine Native "
            "before EOS-child training"
        )
    parent_rank = int(parent_method.pop("rank"))
    parent_alpha = float(parent_method.pop("alpha"))
    parent_runtime_adaptation = str(parent_method["adaptation"])
    method = {
        **parent_method,
        "adaptation": ADAPTATION,
        "parent_adaptation": PARENT_ADAPTATION,
        "parent_runtime_adaptation": parent_runtime_adaptation,
        "parent_rank": parent_rank,
        "parent_alpha": parent_alpha,
        "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
        "child_adapter": child_method,
        "trainable_scope": trainable["scope"],
    }
    state.force("long")
    selection_cache = _build_eos_selection_cache(
        model=model,
        state=state,
        view=calibration_view,
    )
    parent_selection_metrics = _score_eos_selection_cache(
        model=model,
        cache=selection_cache,
    )
    if (
        float(
            parent_selection_metrics["parameter_state"][
                "delta_weight_l2"
            ]
        )
        != 0.0
        or float(
            parent_selection_metrics["parameter_state"]["eos_bias"]
        )
        != 0.0
    ):
        raise RuntimeError("EOS selection parent is not the zero child")

    def checkpoint_callback(
        step: int,
        observed_model: Any,
    ) -> dict[str, Any]:
        candidate = _score_eos_selection_cache(
            model=observed_model,
            cache=selection_cache,
        )
        decision = _eos_selection_gate(
            parent=parent_selection_metrics,
            candidate=candidate,
        )
        return {"step": int(step), **decision}

    training = train(
        model=model,
        routing_view=routing_view,
        calibration_view=calibration_view,
        natural_view_path=natural_path,
        steps=STEPS,
        micro_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        margin=COUNTERFACTUAL_MARGIN,
        margin_weight=COUNTERFACTUAL_MARGIN_WEIGHT,
        termination_weight=TERMINATION_WEIGHT,
        compile_mode=COMPILE_MODE,
        seed=SEED,
        log_path=incomplete / "train_log.jsonl",
        virtual_target_length=VIRTUAL_TARGET_LENGTH,
        virtual_bucket_weights=VIRTUAL_BUCKET_WEIGHTS,
        family_pattern=TRAIN_FAMILY_PATTERN,
        calibration_rows_during_training=0,
        checkpoint_steps=SELECTION_CHECKPOINT_STEPS,
        checkpoint_callback=checkpoint_callback,
    )
    training_log = _training_log_receipt(
        incomplete / "train_log.jsonl",
        training=training,
    )
    selection_receipt = {
        "status": "OLMO2_EOS_CHECKPOINT_SELECTION_V1",
        "policy": protocol()["checkpoint_selection"],
        "cache": selection_cache["receipt"],
        "parent_step0_metrics": parent_selection_metrics,
        "history": training["checkpoint_history"],
        "selected_step": training["selected_step"],
        "actual_steps": training["actual_steps"],
        "maximum_steps": training["maximum_steps"],
        "stopped_early": training["stopped_early"],
        "planned_query_offset_stream_sha256": training[
            "query_offset_stream_sha256"
        ],
        "consumed_query_offset_prefix_sha256": training[
            "consumed_query_offset_prefix_sha256"
        ],
        "consumed_pair_exposures": training[
            "consumed_query_offset_values"
        ],
        "selection_is_capability_evidence": False,
    }
    try:
        selected_step = _require_selected_checkpoint(training)
    except RuntimeError:
        atomic_json(
            incomplete / "checkpoint_selection_stop.json",
            {
                **selection_receipt,
                "decision": "STOP_NO_ELIGIBLE_CHECKPOINT",
            },
        )
        raise
    selected_history = [
        row
        for row in training["checkpoint_history"]
        if int(row["step"]) == selected_step
    ]
    if len(selected_history) != 1:
        raise RuntimeError("selected EOS checkpoint receipt drift")
    selected_child_state_sha256 = selected_history[0][
        "parameter_state"
    ]["sha256"]
    selection_receipt.update(
        {
            "decision": "SELECTED",
            "selected_child_state_sha256": (
                selected_child_state_sha256
            ),
        }
    )
    selection_path = incomplete / "checkpoint_selection.json"
    atomic_json(selection_path, selection_receipt)
    selection_sha256 = sha256_file(selection_path)
    parent_state_after, _, _ = _qkvo_parent_state(model)
    if parent_state_after != parent_state_before:
        raise RuntimeError("frozen QKVO parent changed during EOS repair")
    adapter_metadata = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": LENGTH_GATED_FREQUENCY_NAME,
        "frequency_sha256_float32": EVQ_FREQUENCY_SHA256,
        "short_branch_frequency": "native_endpoint_rope",
        "short_branch_frequency_sha256_float32": (
            NATIVE_FREQUENCY_SHA256
        ),
        "short_branch_maximum_position_id": SHORT_CONTEXT_LIMIT - 1,
        "short_branch_lora": "frozen_base_linear_direct_call",
        "long_branch_frequency": "evq_endpoint_cosh",
        "long_branch_minimum_maximum_position_id": SHORT_CONTEXT_LIMIT,
        "long_branch_scope": "entire_sequence",
        "adaptation": ADAPTATION,
        "adaptation_description": (
            "frozen_length_gated_qkvo_parent_plus_long_only_eos_vocab_row"
        ),
        "stage": "length_gated_query_gap_16k_eos_repair_v1",
        "rank": EOS_HEAD_RANK,
        "alpha": EOS_HEAD_ALPHA,
        "parent_adaptation": PARENT_ADAPTATION,
        "parent_rank": PARENT_RANK,
        "parent_alpha": PARENT_ALPHA,
        "trainable_scope": trainable["scope"],
        "modified_vocab_rows": [OLMO2_EOS_TOKEN_ID],
        "scalar_eos_bias": True,
        "parameterization": "direct_single_eos_row_delta",
        "rank1_equivalent": True,
        "trainable_parameters": 2_049,
        "trainable_parameter_tensors": 2,
        "training_sequence_length": LENGTH,
        "maximum_physical_training_sequence_length": LENGTH,
        "real_8k_or_16k_training_sequences": 0,
        "virtual_target_length": VIRTUAL_TARGET_LENGTH,
        "position_policy": "semantic_query_block_continuous_gap",
        "cached_generation_branch_selection": (
            "force_from_total_context_budget_before_prompt"
        ),
        "final_eos_supervised": True,
        "supervision_contract": SUPERVISION_CONTRACT,
        "eos_token_id": OLMO2_EOS_TOKEN_ID,
        "termination_weight": TERMINATION_WEIGHT,
        "routing_data_sha256": sha256_file(
            routing_root / "manifest.json"
        ),
        "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
        "parent_qkvo_state_sha256": parent_state_before,
        "maximum_optimizer_steps": STEPS,
        "selected_optimizer_step": selected_step,
        "checkpoint_selection_sha256": selection_sha256,
        "selected_child_state_sha256": (
            selected_child_state_sha256
        ),
        "seed": SEED,
    }
    adapter_sha = save_adapter(
        incomplete / "adapter.pt",
        model,
        None,
        adapter_metadata,
    )
    if adapter_sha is None:
        raise RuntimeError("length-gated adapter was not written")

    short_after_hidden, short_after_final_logits = _short_outputs(
        model=model,
        state=state,
        input_ids=parity_ids,
    )
    hidden_equal = torch.equal(native_hidden, short_after_hidden)
    final_logits_equal = torch.equal(
        native_final_logits, short_after_final_logits
    )
    short_equal = hidden_equal and final_logits_equal
    hidden_difference = (
        native_hidden.float() - short_after_hidden.float()
    ).abs()
    final_logits_difference = (
        native_final_logits.float()
        - short_after_final_logits.float()
    ).abs()
    short_parity = {
        "scope": "same_load_4096_hidden_and_final_logits",
        "input_tokens": int(parity_ids.numel()),
        "maximum_position_id": LENGTH - 1,
        "mode": state.mode,
        "pristine_native_vs_pretraining_hidden_torch_equal": True,
        "pristine_native_vs_pretraining_final_logits_torch_equal": True,
        "pristine_native_vs_posttraining_hidden_torch_equal": hidden_equal,
        "pristine_native_vs_posttraining_final_logits_torch_equal": (
            final_logits_equal
        ),
        "maximum_hidden_absolute_difference": float(
            hidden_difference.max()
        ),
        "maximum_final_logits_absolute_difference": float(
            final_logits_difference.max()
        ),
        "finite_pristine_native": bool(
            torch.isfinite(native_hidden).all()
            and torch.isfinite(native_final_logits).all()
        ),
        "finite_posttraining": bool(
            torch.isfinite(short_after_hidden).all()
            and torch.isfinite(short_after_final_logits).all()
        ),
        "method_boundary": (
            "Same-load runtime canary over the complete 4096-token hidden "
            "tensor and final-position full-vocabulary logits. Final "
            "independent-reload hidden/logits/KV parity remains a separate "
            "structural gate."
        ),
    }
    if not short_equal:
        raise RuntimeError(
            "length-gated short branch changed after adapter training"
        )

    receipt = {
        "status": RESULT_STATUS,
        "metric_boundary": (
            "Training, teacher-forced routing calibration, and exact "
            "short-path preservation only. Long capability still requires "
            "the bound greedy full-string+terminal-EOS minimal screen."
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "ready_receipt_sha256": sha256_file(ready_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "bound_code": {
            name: entry["sha256"]
            for name, entry in ready["code"].items()
        },
        "method": method,
        "trainable_scope": trainable,
        "parent_adapter": {
            "path": str(parent_adapter),
            "sha256": PARENT_ADAPTER_SHA256,
            "metadata": parent_metadata,
        },
        "parent_qkvo_integrity": {
            "state_sha256_before_training": parent_state_before,
            "state_sha256_after_training": parent_state_after,
            "torch_state_equal": parent_state_before == parent_state_after,
            "modules": parent_modules,
            "parameter_tensors": parent_tensors,
            "freeze_receipt": parent_freeze,
        },
        "parent_exact_baseline": parent_exact_baseline,
        "adapter_sha256": adapter_sha,
        "adapter_metadata": adapter_metadata,
        "routing_data": {
            "path": str(routing_root),
            "manifest_sha256": sha256_file(
                routing_root / "manifest.json"
            ),
            "status": ROOT_STATUS,
            "format_version": 2,
            "supervision_contract": SUPERVISION_CONTRACT,
            "eos_token_id": OLMO2_EOS_TOKEN_ID,
            "final_eos_supervised": True,
            "labels_only_cover_answer_and_final_eos": True,
            "answer_string_tokenizer_roundtrip_exact": True,
        },
        "protocol": protocol(),
        "runtime": {
            **runtime,
            "compile_cache": cache,
            "allocator": allocator,
            "output_filesystem": {
                "path": str(output.parent),
                "free_bytes_before_run": int(disk_before.free),
                "free_bytes_after_training": int(
                    shutil.disk_usage(output.parent).free
                ),
                "minimum_free_bytes": MINIMUM_FREE_BYTES,
            },
        },
        "short_branch_parity": short_parity,
        "training": training,
        "training_log": training_log,
        "checkpoint_selection": {
            **selection_receipt,
            "relative_path": selection_path.name,
            "sha256": selection_sha256,
        },
        "expanded_evaluation_authorized": False,
    }
    atomic_json(incomplete / "results.json", receipt)
    incomplete.replace(output)
    print(
        json.dumps(
            {
                "status": RESULT_STATUS,
                "output": str(output / "results.json"),
                "adapter_sha256": adapter_sha,
                "short_branch_exact": short_equal,
                "expanded_evaluation_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
