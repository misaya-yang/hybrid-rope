from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    evaluate_instruct_ruler_screen as evaluator,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    gate_olmo2_length_gated_exact_screen as gate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    preflight_4k_length_gated_eos as preflight,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    train_4k_length_gated_eos as trainer,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_screen import (
    validate_adapter_metadata,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_data import (
    sha256_file,
)


def _metadata() -> dict[str, object]:
    return {
        "base_checkpoint_sha256": gate.MODEL_SHA256,
        "frequency": gate.FREQUENCY,
        "frequency_sha256_float32": gate.EVQ_FREQUENCY_SHA256,
        "short_branch_frequency": "native_endpoint_rope",
        "short_branch_frequency_sha256_float32": (
            gate.NATIVE_FREQUENCY_SHA256
        ),
        "short_branch_maximum_position_id": 4_095,
        "short_branch_lora": "frozen_base_linear_direct_call",
        "long_branch_frequency": "evq_endpoint_cosh",
        "long_branch_minimum_maximum_position_id": 4_096,
        "long_branch_scope": "entire_sequence",
        "adaptation": gate.ADAPTATION,
        "stage": "length_gated_query_gap_16k_eos_repair_v1",
        "rank": 1,
        "alpha": 1.0,
        "parent_adaptation": "qkvo_answer",
        "parent_rank": 64,
        "parent_alpha": 128.0,
        "modified_vocab_rows": [gate.OLMO2_EOS_TOKEN_ID],
        "scalar_eos_bias": True,
        "parameterization": "direct_single_eos_row_delta",
        "rank1_equivalent": True,
        "trainable_parameters": 2_049,
        "trainable_parameter_tensors": 2,
        "trainable_scope": (
            "long_only_eos_vocab_row_plus_scalar_bias"
        ),
        "training_sequence_length": 4_096,
        "maximum_physical_training_sequence_length": 4_096,
        "real_8k_or_16k_training_sequences": 0,
        "virtual_target_length": 16_384,
        "position_policy": "semantic_query_block_continuous_gap",
        "cached_generation_branch_selection": (
            "force_from_total_context_budget_before_prompt"
        ),
        "final_eos_supervised": True,
        "supervision_contract": gate.SUPERVISION_CONTRACT,
        "eos_token_id": gate.OLMO2_EOS_TOKEN_ID,
        "termination_weight": 1.0,
        "routing_data_sha256": "f" * 64,
        "parent_adapter_sha256": gate.PARENT_ADAPTER_SHA256,
        "parent_qkvo_state_sha256": "4" * 64,
        "maximum_optimizer_steps": 32,
        "selected_optimizer_step": 32,
        "checkpoint_selection_sha256": "5" * 64,
        "selected_child_state_sha256": "7" * 64,
    }


def _receipts() -> tuple[dict[str, object], dict[str, object]]:
    ready_sha = "r" * 64
    evaluator_sha = "e" * 64
    method_sha = "m" * 64
    adapter_sha = "a" * 64
    method = {
        "active_frequency": gate.FREQUENCY,
        "active_sha256_float32": gate.EVQ_FREQUENCY_SHA256,
        "adaptation": gate.ADAPTATION,
        "parent_adaptation": gate.PARENT_ADAPTATION,
        "parent_rank": gate.PARENT_RANK,
        "parent_alpha": gate.PARENT_ALPHA,
        "parent_adapter_sha256": gate.PARENT_ADAPTER_SHA256,
        "trainable_scope": gate.TRAINABLE_SCOPE,
        "short_lora_dispatch": "base_linear_direct_call",
        "long_lora_dispatch": "qkvo_lora",
        "child_adapter": {
            "adaptation": gate.ADAPTATION,
            "scope": "long_mode_only",
            "modified_vocab_rows": [gate.OLMO2_EOS_TOKEN_ID],
            "short_dispatch": "base_lm_head_direct_call",
            "rank": gate.CHILD_RANK,
            "alpha": gate.CHILD_ALPHA,
            "scalar_bias": True,
            "trainable_parameters": 2_049,
            "trainable_parameter_tensors": 2,
            "parameterization": "direct_single_eos_row_delta",
            "rank1_equivalent": True,
        },
        "short_branch": {
            "maximum_position_id": 4_095,
            "frequency": "native_endpoint_rope",
            "frequency_sha256_float32": (
                gate.NATIVE_FREQUENCY_SHA256
            ),
            "rotary_dispatch": "original_module_direct_call",
        },
        "long_branch": {
            "minimum_maximum_position_id": 4_096,
            "frequency": "evq_endpoint_cosh",
            "frequency_sha256_float32": gate.EVQ_FREQUENCY_SHA256,
            "scope": "entire_sequence",
        },
    }
    training = {
        "status": gate.TRAINING_STATUS,
        "checkpoint_sha256": gate.MODEL_SHA256,
        "ready_receipt_sha256": ready_sha,
        "bound_code": {
            "exact_evaluator": evaluator_sha,
            "exact_gate": sha256_file(Path(gate.__file__).resolve()),
            "method": method_sha,
        },
        "method": method,
        "trainable_scope": {
            "scope": gate.TRAINABLE_SCOPE,
            "parameter_names": [
                "model.lm_head.delta_weight",
                "model.lm_head.eos_bias",
            ],
            "parameter_tensors": 2,
            "parameters": 2_049,
        },
        "parent_adapter": {
            "sha256": gate.PARENT_ADAPTER_SHA256,
            "metadata": {
                "base_checkpoint_sha256": gate.MODEL_SHA256,
                "frequency": "evq",
                "adaptation": gate.PARENT_ADAPTATION,
                "rank": gate.PARENT_RANK,
                "alpha": gate.PARENT_ALPHA,
                "training_sequence_length": 4_096,
                "stage": (
                    "counterfactual_routing_semantic_query_gap_16k"
                ),
                "parent_adapter_sha256": (
                    gate.PRE_QUERY_GAP_PARENT_SHA256
                ),
                "position_policy": (
                    "semantic_query_block_continuous_gap"
                ),
                "virtual_target_length": 16_384,
            },
        },
        "parent_qkvo_integrity": {
            "state_sha256_before_training": "4" * 64,
            "state_sha256_after_training": "4" * 64,
            "torch_state_equal": True,
            "modules": 64,
            "parameter_tensors": 128,
            "freeze_receipt": {
                "qkvo_lora_modules": 64,
                "parameter_tensors": 128,
                "parameters": 16_777_216,
                "trainable_after_freeze": False,
            },
        },
        "parent_exact_baseline": {
            "adapter_sha256": gate.PARENT_ADAPTER_SHA256,
            "decision": "EOS_ONLY_REPAIR_FEASIBLE",
            "result_sha256": "1" * 64,
            "examples_sha256": "2" * 64,
            "run_manifest_sha256": "3" * 64,
        },
        "protocol": trainer.protocol(),
        "routing_data": {
            "manifest_sha256": "f" * 64,
            "status": (
                "OLMO2_4K_COUNTERFACTUAL_ROUTING_DATA_PREPARED_V2"
            ),
            "format_version": 2,
            "supervision_contract": gate.SUPERVISION_CONTRACT,
            "eos_token_id": gate.OLMO2_EOS_TOKEN_ID,
            "final_eos_supervised": True,
            "labels_only_cover_answer_and_final_eos": True,
            "answer_string_tokenizer_roundtrip_exact": True,
        },
        "training": {
            "steps": 32,
            "maximum_steps": 32,
            "actual_steps": 32,
            "selected_step": 32,
            "stopped_early": False,
            "checkpoint_steps": [4, 8, 16, 32],
            "checkpoint_selection_required": True,
            "checkpoint_selection_passed": True,
            "checkpoint_history": [
                {"step": 4, "passed": False},
                {"step": 8, "passed": False},
                {"step": 16, "passed": False},
                {"step": 32, "passed": True},
            ],
            "family_pattern": ["routing"],
            "family_steps": {"routing": 32, "natural": 0},
            "processed_input_tokens": 1_048_320,
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
            "position_bucket_counts": {
                "contiguous": 16,
                "transition": 16,
                "middle": 32,
                "far": 64,
            },
            "query_offset_stream_sha256": trainer.protocol()[
                "query_offset_stream_sha256"
            ],
            "consumed_query_offset_prefix_sha256": trainer.protocol()[
                "query_offset_stream_sha256"
            ],
            "consumed_query_offset_values": 128,
            "maximum_observed_position_id": 16_000,
            "realized_position_stream_sha256": "b" * 64,
            "realized_exposure_stream_sha256": "c" * 64,
            "tokens_per_second": 100.0,
        },
        "training_log": {
            "sha256": "d" * 64,
            "rows": 6,
            "first_optimizer_step": {
                "step": 1,
                "loss": 1.0,
                "grad_norm": 1.0,
                "processed_input_tokens": 32_760,
                "interval_tokens_per_second": 100.0,
                "peak_memory_allocated_bytes": 1,
            },
            "last_optimizer_step": {
                "step": 32,
                "loss": 0.5,
                "grad_norm": 0.5,
                "processed_input_tokens": 1_048_320,
                "interval_tokens_per_second": 100.0,
                "peak_memory_allocated_bytes": 1,
            },
        },
        "checkpoint_selection": {
            "path": "checkpoint_selection.json",
            "sha256": "5" * 64,
            "selected_child_state_sha256": "7" * 64,
        },
        "runtime": {
            "name": "test-gpu",
            "capability": [12, 0],
            "flash_sdp_enabled": True,
            "math_sdp_enabled": False,
            "mem_efficient_sdp_enabled": False,
            "compile_cache": "/tmp/inductor",
            "allocator": "expandable_segments:True",
            "output_filesystem": {
                "minimum_free_bytes": 20 * 1024**3,
                "free_bytes_before_run": 21 * 1024**3,
                "free_bytes_after_training": 20 * 1024**3,
            },
        },
        "short_branch_parity": {
            "scope": "same_load_4096_hidden_and_final_logits",
            "input_tokens": 4_096,
            "maximum_position_id": 4_095,
            "mode": "short",
            "pristine_native_vs_pretraining_hidden_torch_equal": True,
            "pristine_native_vs_pretraining_final_logits_torch_equal": True,
            "pristine_native_vs_posttraining_hidden_torch_equal": True,
            "pristine_native_vs_posttraining_final_logits_torch_equal": True,
            "maximum_hidden_absolute_difference": 0.0,
            "maximum_final_logits_absolute_difference": 0.0,
            "finite_pristine_native": True,
            "finite_posttraining": True,
        },
        "adapter_sha256": adapter_sha,
        "adapter_metadata": _metadata(),
    }
    evaluation = {
        "status": gate.EVALUATION_STATUS,
        "checkpoint_sha256": gate.MODEL_SHA256,
        "experiment_ready_receipt_sha256": ready_sha,
        "script_sha256": evaluator_sha,
        "bound_code_sha256": {
            "length_gated_import_dependency": method_sha,
        },
        "frequency": method,
        "adapter": {
            "sha256": adapter_sha,
            "metadata": _metadata(),
            "adaptation": gate.ADAPTATION,
            "rank": gate.CHILD_RANK,
            "alpha": gate.CHILD_ALPHA,
            "trainable_parameter_names": [
                "model.lm_head.delta_weight",
                "model.lm_head.eos_bias",
            ],
            "method": method["child_adapter"],
        },
        "parent_adapter": {
            **training["parent_adapter"],
            "adaptation": gate.PARENT_ADAPTATION,
            "rank": gate.PARENT_RANK,
            "alpha": gate.PARENT_ALPHA,
            "freeze": training["parent_qkvo_integrity"][
                "freeze_receipt"
            ],
        },
        "protocol": {
            "task": "niah_single_1",
            "lengths": [8_192, 16_384],
            "limit_per_length": 8,
            "greedy": True,
            "maximum_new_tokens": gate.GENERATION_TOKENS,
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
            "actual_prompt_length_requirement": (
                "actual_maximum_prompt_position_id >= nominal_length / 2"
            ),
            "minimum_far_gap_rows_per_cell": 1,
            "far_gap_threshold_tokens": 3_933,
        },
        "short_structural_parity": {
            "status": "PASS",
            "input_tokens": 4_096,
            "maximum_position_id": 4_095,
            "prefill_tokens": 32,
            "decode_steps": 1,
            "torch_equal": {
                "hidden": True,
                "final_logits": True,
                "prefill_logits": True,
                "decode_logits": True,
                "prefill_cache": True,
                "decode_cache": True,
            },
            "maximum_absolute_difference": {
                "hidden": 0.0,
                "final_logits": 0.0,
                "prefill_logits": 0.0,
                "decode_logits": 0.0,
            },
            "cache_tensor_counts": {
                "prefill_cache": 32,
                "decode_cache": 32,
            },
            "cache_sha256": {
                "prefill_cache": {
                    "pristine_native": "5" * 64,
                    "reloaded_candidate": "5" * 64,
                },
                "decode_cache": {
                    "pristine_native": "6" * 64,
                    "reloaded_candidate": "6" * 64,
                },
            },
            "runtime_contract": (
                "fresh checkpoint load; same BF16/Flash runtime; parent then "
                "child artifacts reloaded; branch forced short before prefill"
            ),
        },
        "results": {
            "cells": {
                "8192": {
                    "examples": 8,
                    "exact_generation_pass": 0.75,
                    "full_string_exact": 0.75,
                    "eos_terminated": 1.0,
                    "answer_eos_token_exact": 0.75,
                    "far_gap_examples": 1,
                    "far_gap_exact_generation_pass": 1.0,
                },
                "16384": {
                    "examples": 8,
                    "exact_generation_pass": 0.25,
                    "full_string_exact": 0.25,
                    "eos_terminated": 1.0,
                    "answer_eos_token_exact": 0.25,
                    "far_gap_examples": 1,
                    "far_gap_exact_generation_pass": 1.0,
                },
            }
        },
    }
    return training, evaluation


def _raw_counts(
    evaluation: dict[str, object],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for length, cell in evaluation["results"]["cells"].items():
        examples = int(cell["examples"])
        result[length] = {
            "examples": examples,
            "exact_generation_passes": round(
                float(cell["exact_generation_pass"]) * examples
            ),
            "full_string_exact": round(
                float(cell["full_string_exact"]) * examples
            ),
            "eos_terminated": round(
                float(cell["eos_terminated"]) * examples
            ),
            "answer_eos_token_exact": round(
                float(cell["answer_eos_token_exact"]) * examples
            ),
            "far_gap_examples": int(cell["far_gap_examples"]),
            "far_gap_exact_generation_passes": round(
                float(cell["far_gap_exact_generation_pass"])
                * int(cell["far_gap_examples"])
            ),
        }
    return result


def test_length_gated_protocol_never_uses_real_long_training() -> None:
    protocol = trainer.protocol()
    assert protocol["model_scope"] == "OLMo-2 1.485B Instruct only"
    assert protocol["maximum_physical_training_sequence_length"] == 4_096
    assert protocol["maximum_physical_model_input_tokens"] == 4_095
    assert protocol["real_8k_or_16k_training_sequences"] == 0
    assert protocol["maximum_allowed_position_id"] == 16_383
    assert "maximum_realized_position_id" not in protocol
    assert protocol["adaptation"] == "length_gated_eos_vocab_row"
    assert protocol["trainable_scope"] == (
        "long_only_eos_vocab_row_plus_scalar_bias"
    )
    assert protocol["final_eos_supervised"] is True
    assert protocol["cached_short_to_long_transition"] == "fail_closed"


def test_evaluator_accepts_only_complete_length_gated_metadata() -> None:
    metadata = _metadata()
    validate_adapter_metadata(
        metadata,
        checkpoint_digest=gate.MODEL_SHA256,
        frequency={
            "active_sha256_float32": gate.EVQ_FREQUENCY_SHA256,
        },
        frequency_name=gate.FREQUENCY,
        rank=1,
        alpha=1.0,
        adaptation=gate.ADAPTATION,
        parent_adapter_sha256=gate.PARENT_ADAPTER_SHA256,
    )

    invalid = copy.deepcopy(metadata)
    invalid["trainable_scope"] = "all_qkvo_plus_eos"
    with pytest.raises(RuntimeError, match="trainable_scope"):
        validate_adapter_metadata(
            invalid,
            checkpoint_digest=gate.MODEL_SHA256,
            frequency={
                "active_sha256_float32": gate.EVQ_FREQUENCY_SHA256,
            },
            frequency_name=gate.FREQUENCY,
            rank=1,
            alpha=1.0,
            adaptation=gate.ADAPTATION,
            parent_adapter_sha256=gate.PARENT_ADAPTER_SHA256,
        )


def test_eos_candidate_loads_parent_freezes_then_loads_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent.pt"
    child = tmp_path / "child.pt"
    parent.write_bytes(b"frozen-a0ccd-parent")
    child.write_bytes(b"eos-child")
    parent_sha = evaluator.sha256_file(parent)
    child_sha = evaluator.sha256_file(child)
    monkeypatch.setattr(
        evaluator,
        "LENGTH_GATED_PARENT_ADAPTER_SHA256",
        parent_sha,
    )
    checkpoint_sha = "c" * 64
    frequency = {
        "active_frequency": evaluator.LENGTH_GATED_FREQUENCY_NAME,
        "active_sha256_float32": evaluator.EVQ_FREQUENCY_SHA256,
        "adaptation": "length_gated_qkvo_answer",
        "rank": evaluator.LENGTH_GATED_PARENT_RANK,
        "alpha": evaluator.LENGTH_GATED_PARENT_ALPHA,
    }
    parent_metadata = {
        "base_checkpoint_sha256": checkpoint_sha,
        "frequency": "evq",
        "frequency_sha256_float32": evaluator.EVQ_FREQUENCY_SHA256,
        "adaptation": evaluator.LENGTH_GATED_PARENT_ADAPTATION,
        "rank": evaluator.LENGTH_GATED_PARENT_RANK,
        "alpha": evaluator.LENGTH_GATED_PARENT_ALPHA,
        "training_sequence_length": 4_096,
    }
    child_metadata = {
        "base_checkpoint_sha256": checkpoint_sha,
        "frequency": evaluator.LENGTH_GATED_FREQUENCY_NAME,
        "frequency_sha256_float32": evaluator.EVQ_FREQUENCY_SHA256,
        "adaptation": evaluator.LENGTH_GATED_EOS_ADAPTATION,
        "rank": evaluator.LENGTH_GATED_EOS_HEAD_RANK,
        "alpha": evaluator.LENGTH_GATED_EOS_HEAD_ALPHA,
        "training_sequence_length": 4_096,
        "parent_adapter_sha256": parent_sha,
        "parent_adaptation": evaluator.LENGTH_GATED_PARENT_ADAPTATION,
        "parent_rank": evaluator.LENGTH_GATED_PARENT_RANK,
        "parent_alpha": evaluator.LENGTH_GATED_PARENT_ALPHA,
        "modified_vocab_rows": [evaluator.OLMO2_EOS_TOKEN_ID],
        "trainable_scope": evaluator.LENGTH_GATED_EOS_TRAINABLE_SCOPE,
        "eos_token_id": evaluator.OLMO2_EOS_TOKEN_ID,
        "parameterization": "direct_single_eos_row_delta",
        "rank1_equivalent": True,
        "trainable_parameters": 2_049,
        "trainable_parameter_tensors": 2,
    }
    events: list[str] = []
    state = object()
    model = object()

    def fake_install_qkvo(
        observed_model: object,
        *,
        rank: int,
        alpha: float,
    ) -> tuple[object, dict[str, object]]:
        assert observed_model is model
        assert rank == evaluator.LENGTH_GATED_PARENT_RANK
        assert alpha == evaluator.LENGTH_GATED_PARENT_ALPHA
        events.append("install_qkvo")
        return state, frequency

    def fake_load_adapter(
        path: Path,
        observed_model: object,
        readout: object,
    ) -> dict[str, object]:
        assert observed_model is model
        assert readout is None
        if path == parent:
            events.append("load_parent")
            return parent_metadata
        assert path == child
        events.append("load_child")
        return child_metadata

    def fake_freeze(observed_model: object) -> dict[str, object]:
        assert observed_model is model
        events.append("freeze_parent")
        return {"trainable_after_freeze": False}

    def fake_install_head(
        observed_model: object,
        observed_state: object,
        *,
        rank: int,
        alpha: float,
        eos_token_id: int,
    ) -> tuple[object, dict[str, object]]:
        assert observed_model is model
        assert observed_state is state
        assert rank == evaluator.LENGTH_GATED_EOS_HEAD_RANK
        assert alpha == evaluator.LENGTH_GATED_EOS_HEAD_ALPHA
        assert eos_token_id == evaluator.OLMO2_EOS_TOKEN_ID
        events.append("install_child")
        return object(), {"modified_vocab_rows": [eos_token_id]}

    def fake_child_scope(
        observed_model: object,
    ) -> list[tuple[str, object]]:
        assert observed_model is model
        events.append("enumerate_child")
        return [
            ("model.lm_head.delta_weight", object()),
            ("model.lm_head.eos_bias", object()),
        ]

    monkeypatch.setattr(
        evaluator,
        "install_length_gated_qkvo",
        fake_install_qkvo,
    )
    monkeypatch.setattr(evaluator, "load_adapter", fake_load_adapter)
    monkeypatch.setattr(
        evaluator,
        "freeze_length_gated_qkvo_adapter",
        fake_freeze,
    )
    monkeypatch.setattr(
        evaluator,
        "install_length_gated_eos_vocab_row_head",
        fake_install_head,
    )
    monkeypatch.setattr(
        evaluator,
        "eos_head_trainable_named_parameters",
        fake_child_scope,
    )

    observed_state, observed_frequency, parent_receipt, child_receipt = (
        evaluator.load_length_gated_parent_child(
            model,
            checkpoint_digest=checkpoint_sha,
            parent_adapter_path=parent,
            child_adapter_path=child,
            parent_rank=evaluator.LENGTH_GATED_PARENT_RANK,
            parent_alpha=evaluator.LENGTH_GATED_PARENT_ALPHA,
            child_rank=evaluator.LENGTH_GATED_EOS_HEAD_RANK,
            child_alpha=evaluator.LENGTH_GATED_EOS_HEAD_ALPHA,
            eos_token_id=evaluator.OLMO2_EOS_TOKEN_ID,
        )
    )

    assert events == [
        "install_qkvo",
        "load_parent",
        "freeze_parent",
        "install_child",
        "enumerate_child",
        "load_child",
    ]
    assert observed_state is state
    assert observed_frequency["adaptation"] == (
        evaluator.LENGTH_GATED_EOS_ADAPTATION
    )
    assert observed_frequency["parent_adaptation"] == (
        evaluator.LENGTH_GATED_PARENT_ADAPTATION
    )
    assert observed_frequency["parent_rank"] == (
        evaluator.LENGTH_GATED_PARENT_RANK
    )
    assert observed_frequency["parent_alpha"] == (
        evaluator.LENGTH_GATED_PARENT_ALPHA
    )
    assert observed_frequency["parent_adapter_sha256"] == parent_sha
    assert observed_frequency["child_adapter"] == {
        "modified_vocab_rows": [evaluator.OLMO2_EOS_TOKEN_ID]
    }
    assert observed_frequency["trainable_scope"] == (
        evaluator.LENGTH_GATED_EOS_TRAINABLE_SCOPE
    )
    assert observed_frequency["adaptation"] != (
        "length_gated_qkvo_answer"
    )
    assert parent_receipt["sha256"] == parent_sha
    assert child_receipt["sha256"] == child_sha
    assert child_receipt["metadata"]["parent_adapter_sha256"] == parent_sha
    assert child_receipt["trainable_parameter_names"] == [
        "model.lm_head.delta_weight",
        "model.lm_head.eos_bias",
    ]


def test_eos_child_metadata_must_bind_exact_parent_sha() -> None:
    metadata = {
        "base_checkpoint_sha256": "c" * 64,
        "frequency": evaluator.LENGTH_GATED_FREQUENCY_NAME,
        "frequency_sha256_float32": evaluator.EVQ_FREQUENCY_SHA256,
        "adaptation": evaluator.LENGTH_GATED_EOS_ADAPTATION,
        "rank": evaluator.LENGTH_GATED_EOS_HEAD_RANK,
        "alpha": evaluator.LENGTH_GATED_EOS_HEAD_ALPHA,
        "training_sequence_length": 4_096,
        "parent_adapter_sha256": "f" * 64,
        "parent_adaptation": evaluator.LENGTH_GATED_PARENT_ADAPTATION,
        "parent_rank": evaluator.LENGTH_GATED_PARENT_RANK,
        "parent_alpha": evaluator.LENGTH_GATED_PARENT_ALPHA,
        "modified_vocab_rows": [evaluator.OLMO2_EOS_TOKEN_ID],
        "trainable_scope": evaluator.LENGTH_GATED_EOS_TRAINABLE_SCOPE,
        "eos_token_id": evaluator.OLMO2_EOS_TOKEN_ID,
        "parameterization": "direct_single_eos_row_delta",
        "rank1_equivalent": True,
        "trainable_parameters": 2_049,
        "trainable_parameter_tensors": 2,
    }

    with pytest.raises(
        RuntimeError,
        match="parent_adapter_sha256",
    ):
        evaluator.validate_adapter_metadata(
            metadata,
            checkpoint_digest="c" * 64,
            frequency={
                "active_sha256_float32": (
                    evaluator.EVQ_FREQUENCY_SHA256
                ),
            },
            frequency_name=evaluator.LENGTH_GATED_FREQUENCY_NAME,
            rank=evaluator.LENGTH_GATED_EOS_HEAD_RANK,
            alpha=evaluator.LENGTH_GATED_EOS_HEAD_ALPHA,
            adaptation=evaluator.LENGTH_GATED_EOS_ADAPTATION,
            parent_adapter_sha256="f" * 64,
        )


def _short_trace() -> dict[str, object]:
    return {
        "hidden": torch.zeros(1, 4, 3),
        "final_logits": torch.zeros(1, 1, 5),
        "prefill_logits": torch.zeros(1, 1, 5),
        "decode_logits": torch.zeros(1, 1, 5),
        "prefill_cache": ((torch.zeros(1, 2), torch.zeros(1, 2)),),
        "decode_cache": ((torch.zeros(1, 3), torch.zeros(1, 3)),),
    }


def test_short_structural_parity_requires_logits_and_every_cache() -> None:
    pristine = _short_trace()
    candidate = copy.deepcopy(pristine)
    receipt = evaluator.validate_short_structural_parity(
        pristine=pristine,
        candidate=candidate,
    )
    assert receipt["status"] == "PASS"
    assert all(receipt["torch_equal"].values())

    changed = copy.deepcopy(candidate)
    changed["decode_cache"][0][1][0, 0] = 1.0
    with pytest.raises(RuntimeError, match="changed the Native short path"):
        evaluator.validate_short_structural_parity(
            pristine=pristine,
            candidate=changed,
        )


def test_minimal_gate_uses_structural_4k_and_strict_long_exact() -> None:
    training, evaluation = _receipts()
    passed, details = gate.exact_screen_gate(
        training,
        evaluation,
        raw_counts=_raw_counts(evaluation),
    )

    assert passed is True
    assert details["expanded_evaluation_authorized"] is True
    assert details["checks"]["short_branch_same_load_native_canary"] is True
    assert (
        details["checks"][
            "short_branch_fresh_reload_hidden_logits_kv_parity"
        ]
        is True
    )
    assert set(details["counts"]) == {"8192", "16384"}
    assert "first_number_exact" in details["ignored_for_admission"]


def test_minimal_gate_stops_on_16k_full_string_exact_failure() -> None:
    training, evaluation = _receipts()
    evaluation["results"]["cells"]["16384"][
        "exact_generation_pass"
    ] = 0.125
    evaluation["results"]["cells"]["16384"]["full_string_exact"] = 0.125
    evaluation["results"]["cells"]["16384"][
        "answer_eos_token_exact"
    ] = 0.125

    passed, details = gate.exact_screen_gate(
        training,
        evaluation,
        raw_counts=_raw_counts(evaluation),
    )

    assert passed is False
    assert details["expanded_evaluation_authorized"] is False
    assert details["checks"]["L16384_exact_at_least_2_of_8"] is False


def test_minimal_gate_rejects_aggregate_raw_mismatch() -> None:
    training, evaluation = _receipts()
    raw = _raw_counts(evaluation)
    raw["8192"]["exact_generation_passes"] -= 1

    with pytest.raises(RuntimeError, match="aggregate/raw"):
        gate.exact_screen_gate(
            training,
            evaluation,
            raw_counts=raw,
        )


def test_raw_gate_recomputes_token_level_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, evaluation = _receipts()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    data_root = tmp_path / "exact_data"
    data_root.mkdir()
    ready_sha = str(training["ready_receipt_sha256"])
    parent_path = tmp_path / "parent.pt"
    child_path = tmp_path / "child.pt"
    parent_path.write_bytes(b"parent")
    child_path.write_bytes(b"child")
    parent_metadata = training["parent_adapter"]["metadata"]
    parent_state = {}
    for layer in range(16):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            prefix = (
                f"model.model.layers.{layer}.self_attn.{projection}"
            )
            parent_state[f"{prefix}.a"] = torch.empty(
                (gate.PARENT_RANK, 2_048), device="meta"
            )
            parent_state[f"{prefix}.b"] = torch.empty(
                (2_048, gate.PARENT_RANK), device="meta"
            )
    child_state = {
        "model.lm_head.delta_weight": torch.empty(
            (2_048,), device="meta"
        ),
        "model.lm_head.eos_bias": torch.empty((), device="meta"),
    }

    def fake_load_adapter_payload(
        path: Path,
        *,
        label: str,
    ) -> tuple[str, dict[str, torch.Tensor], dict[str, object]]:
        if label == "parent":
            assert path == parent_path
            return gate.PARENT_ADAPTER_SHA256, parent_state, parent_metadata
        assert label == "child"
        assert path == child_path
        return str(training["adapter_sha256"]), child_state, _metadata()

    monkeypatch.setattr(
        gate,
        "_load_adapter_payload",
        fake_load_adapter_payload,
    )
    monkeypatch.setattr(
        gate,
        "_validate_checkpoint_selection",
        lambda **kwargs: {
            "status": "PASS",
            "selected_step": 32,
            "selection_sha256": "5" * 64,
            "selected_child_state_sha256": "7" * 64,
        },
    )
    training["parent_adapter"]["path"] = str(parent_path)
    evaluation["parent_adapter"]["path"] = str(parent_path)
    evaluation["adapter"]["path"] = str(child_path)

    class FakeTokenizer:
        eos_token_id = gate.OLMO2_EOS_TOKEN_ID

        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool,
        ) -> SimpleNamespace:
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=[int(text)])

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return "".join(str(value) for value in token_ids)

    source_rows = []
    raw_rows = []
    geometry_rows = []
    exact_by_length = {8_192: 6, 16_384: 2}
    for length, exact_count in exact_by_length.items():
        for local_index in range(8):
            answer = 1_000 + local_index
            source = {
                "index": length + local_index,
                "token_position_answer": 32 + local_index,
                "outputs": [str(answer)],
            }
            source_rows.append(
                {
                    **source,
                    "_nominal_length": length,
                    "_local_index": local_index,
                }
            )
            predicted = answer if local_index < exact_count else 9_999
            token_ids = [predicted, gate.OLMO2_EOS_TOKEN_ID]
            prompt_tokens = length - gate.GENERATION_TOKENS
            source_position = int(source["token_position_answer"])
            generation_gap = prompt_tokens - source_position
            raw_rows.append(
                {
                    "schema_version": gate.EXAMPLE_SCHEMA_VERSION,
                    "task": "niah_single_1",
                    "nominal_length": length,
                    "local_index": local_index,
                    "source_row_index": source["index"],
                    "source_token_position_answer": source[
                        "token_position_answer"
                    ],
                    "row_sha256": gate.row_sha256(source),
                    "input_tokens": prompt_tokens,
                    "actual_maximum_prompt_position_id": (
                        prompt_tokens - 1
                    ),
                    "generation_boundary_gap_tokens": generation_gap,
                    "required_minimum_prompt_tokens": length // 2 + 1,
                    "required_minimum_actual_maximum_position_id": (
                        length // 2
                    ),
                    "far_gap_beyond_training_support": True,
                    "generated_tokens": len(token_ids),
                    "generated_token_ids": token_ids,
                    "prediction": str(predicted),
                    "references": [str(answer)],
                    "reference_token_ids": [[answer]],
                    "full_string_exact": float(predicted == answer),
                    "eos_terminated": 1.0,
                    "answer_eos_token_exact": float(
                        predicted == answer
                    ),
                    "exact_generation_pass": float(
                        predicted == answer
                    ),
                }
            )
            geometry_rows.append(
                {
                    "nominal_length": length,
                    "local_index": local_index,
                    "source_index_field": source["index"],
                    "row_sha256": gate.row_sha256(source),
                    "prompt_tokens": prompt_tokens,
                    "actual_maximum_prompt_position_id": (
                        prompt_tokens - 1
                    ),
                    "source_token_position_answer": source_position,
                    "generation_boundary_gap_tokens": generation_gap,
                    "far_gap_beyond_training_support": True,
                }
            )

    geometry_cells = {}
    for length in exact_by_length:
        selected = [
            row
            for row in geometry_rows
            if int(row["nominal_length"]) == length
        ]
        prompt_values = [int(row["prompt_tokens"]) for row in selected]
        position_values = [
            int(row["actual_maximum_prompt_position_id"])
            for row in selected
        ]
        gap_values = [
            int(row["generation_boundary_gap_tokens"])
            for row in selected
        ]
        geometry_cells[str(length)] = {
            "rows": len(selected),
            "required_minimum_prompt_tokens": length // 2 + 1,
            "required_minimum_actual_maximum_position_id": length // 2,
            "minimum_prompt_tokens": min(prompt_values),
            "maximum_prompt_tokens": max(prompt_values),
            "minimum_actual_maximum_position_id": min(position_values),
            "maximum_actual_maximum_position_id": max(position_values),
            "minimum_generation_boundary_gap_tokens": min(gap_values),
            "maximum_generation_boundary_gap_tokens": max(gap_values),
            "far_gap_rows": len(selected),
        }
    geometry_payload = json.dumps(
        geometry_rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual_geometry = {
        "status": "PASS",
        "policy": (
            "actual_maximum_prompt_position_id reaches the nominal "
            "dyadic band and actual_prompt_tokens + 128 <= nominal_length"
        ),
        "minimum_far_gap_rows_per_cell": 1,
        "far_gap_threshold_tokens": 3_933,
        "generation_tokens_reserved": gate.GENERATION_TOKENS,
        "cells": geometry_cells,
        "rows": geometry_rows,
        "rows_sha256": gate.hashlib.sha256(geometry_payload).hexdigest(),
    }

    data_receipt = {
        "preparation_status": "TEST_PREPARED",
        "manifest_sha256": "1" * 64,
        "ruler_commit": "test",
        "tokenizer_sha256": "2" * 64,
        "files": {
            str(length): {
                "path": str(data_root / f"{length}.jsonl"),
                "sha256": str(index) * 64,
                "rows": 8,
            }
            for index, length in enumerate(exact_by_length, start=3)
        },
    }

    def fake_validate_data(*args: object) -> tuple[
        dict[str, object],
        list[dict[str, object]],
    ]:
        return data_receipt, source_rows

    monkeypatch.setattr(
        gate.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(gate, "validate_data", fake_validate_data)
    monkeypatch.setattr(
        gate,
        "validate_actual_prompt_geometry",
        lambda **kwargs: actual_geometry,
    )

    examples_path = tmp_path / "examples.jsonl"
    examples_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in raw_rows
        ),
        encoding="utf-8",
    )
    code = {
        "trainer": {"sha256": "6" * 64},
        "exact_evaluator": {
            "sha256": str(evaluation["script_sha256"])
        },
        "exact_gate": {
            "sha256": gate.sha256_file(
                Path(gate.__file__).resolve()
            )
        },
        "method": {"sha256": "m" * 64},
    }
    training["script_sha256"] = code["trainer"]["sha256"]
    training["bound_code"] = {
        name: entry["sha256"] for name, entry in code.items()
    }
    run_output = tmp_path / "run"
    run_output.mkdir()
    experiment_ready_run_output = str(run_output)
    training_result_path = run_output / "results.json"
    training_result_path.write_text(
        json.dumps(training, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_preflight = {
        "status": "PASS",
        "parent_adapter_sha256": gate.PARENT_ADAPTER_SHA256,
        "child_adapter_sha256": training["adapter_sha256"],
        "training_result": str(training_result_path.resolve()),
        "training_result_sha256": gate.sha256_file(
            training_result_path
        ),
        "validation_stage": "cpu_before_cuda_configuration",
    }
    evaluation["checkpoint"] = str(checkpoint)
    evaluation["data"] = data_receipt
    evaluation["actual_prompt_geometry"] = actual_geometry
    evaluation["length_gated_artifact_preflight"] = artifact_preflight
    evaluation["bound_code_sha256"] = {
        "length_gated_import_dependency": code["method"]["sha256"],
    }
    evaluation["results"]["examples_sha256"] = gate.sha256_file(
        examples_path
    )

    run_manifest = {
        "status": gate.RUN_MANIFEST_STATUS,
        "example_schema_version": gate.EXAMPLE_SCHEMA_VERSION,
        "checkpoint_sha256": gate.MODEL_SHA256,
        "experiment_ready_receipt_sha256": ready_sha,
        "experiment_role": "candidate_exact_screen",
        "frequency": gate.FREQUENCY,
        "adaptation": gate.ADAPTATION,
        "rank": gate.CHILD_RANK,
        "alpha": gate.CHILD_ALPHA,
        "parent_adapter_sha256": gate.PARENT_ADAPTER_SHA256,
        "parent_rank": gate.PARENT_RANK,
        "parent_alpha": gate.PARENT_ALPHA,
        "task": "niah_single_1",
        "lengths": [8_192, 16_384],
        "limit_per_length": 8,
        "greedy": True,
        "maximum_new_tokens": gate.GENERATION_TOKENS,
        "string_normalization": "none",
        "decode_cleanup": False,
        "terminal_eos_removed_before_string_decode": True,
        "other_special_tokens_removed": False,
        "adapter_sha256": training["adapter_sha256"],
        "bound_code_sha256": evaluation["bound_code_sha256"],
        "length_gated_artifact_preflight": artifact_preflight,
        "actual_prompt_geometry": actual_geometry,
    }
    run_manifest_path = tmp_path / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluation["run_manifest_sha256"] = gate.sha256_file(
        run_manifest_path
    )
    parent_baseline_root = tmp_path / "parent_baseline"
    parent_baseline_root.mkdir()
    parent_baseline_paths = {
        "result_sha256": parent_baseline_root / "results.json",
        "examples_sha256": parent_baseline_root / "examples.jsonl",
        "run_manifest_sha256": parent_baseline_root
        / "run_manifest.json",
    }
    for field, path in parent_baseline_paths.items():
        path.write_text(field + "\n", encoding="utf-8")
        training["parent_exact_baseline"][field] = gate.sha256_file(
            path
        )
    experiment_ready = {
        "status": "OLMO2_4K_LENGTH_GATED_EOS_REPAIR_READY_V1",
        "protocol": training["protocol"],
        "code": code,
        "run_output": experiment_ready_run_output,
        "evaluator_bound_code_sha256": (
            evaluation["bound_code_sha256"]
        ),
        "registered_adapter_chain": {
            "load_order": [
                "frozen_parent_qkvo",
                "eos_vocab_row_child",
            ],
            "parent": {
                "path": str(parent_path),
                "sha256": gate.PARENT_ADAPTER_SHA256,
                "adaptation": gate.PARENT_ADAPTATION,
                "rank": gate.PARENT_RANK,
                "alpha": gate.PARENT_ALPHA,
                "frozen_during_child_training": True,
            },
            "child": {
                "path": str(child_path),
                "adaptation": gate.ADAPTATION,
                "rank": gate.CHILD_RANK,
                "alpha": gate.CHILD_ALPHA,
                "modified_vocab_rows": [gate.OLMO2_EOS_TOKEN_ID],
                "trainable_scope": gate.TRAINABLE_SCOPE,
                "parent_adapter_sha256": gate.PARENT_ADAPTER_SHA256,
                "parameterization": "direct_single_eos_row_delta",
                "rank1_equivalent": True,
                "trainable_parameters": 2_049,
                "trainable_parameter_tensors": 2,
            },
        },
        "registered_outputs": {
            "parent_exact_baseline": str(parent_baseline_root),
        },
        "inputs": {
            "checkpoint": {"path": str(checkpoint)},
            "parent_adapter": {
                "path": str(parent_path),
                "sha256": gate.PARENT_ADAPTER_SHA256,
                "metadata": parent_metadata,
            },
            "exact_eval_data": {
                "path": str(data_root),
                **data_receipt,
                "actual_prompt_geometry": actual_geometry,
            },
        },
    }

    raw = gate.validate_raw_evaluation(
        training=training,
        evaluation=evaluation,
        experiment_ready=experiment_ready,
        experiment_ready_sha256=ready_sha,
        examples_path=examples_path,
        run_manifest_path=run_manifest_path,
    )

    assert raw["counts"]["8192"]["exact_generation_passes"] == 6
    assert raw["counts"]["16384"]["exact_generation_passes"] == 2
    assert raw["failure_diagnostics"]["8192"][
        "eos_with_wrong_full_string"
    ] == 2
    assert raw["diagnostics_are_success_metrics"] is False


def test_preflight_registers_frozen_parent_plus_eos_child_chain(
    tmp_path: Path,
) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "checkpoint",
            "checkpoint_ready_receipt",
            "parent_adapter",
            "routing_data_v2",
            "prepared_data",
            "exact_eval_data",
            "run_output",
            "parent_eval_output",
            "candidate_eval_output",
            "gate_output",
            "receipt_output",
        )
    }
    commands = preflight._registered_commands(paths)
    parent = commands["parent_exact_baseline"]
    candidate = commands["candidate_exact_screen"]
    training = commands["training"]

    def value(argv: list[str], flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert value(parent, "--adapter") == str(paths["parent_adapter"])
    assert value(parent, "--adaptation") == "qkvo_answer"
    assert value(parent, "--rank") == "64"
    assert value(parent, "--alpha") == "128.0"
    assert value(candidate, "--parent-adapter") == str(
        paths["parent_adapter"]
    )
    assert value(candidate, "--parent-rank") == "64"
    assert value(candidate, "--parent-alpha") == "128.0"
    assert value(candidate, "--adapter") == str(
        paths["run_output"] / "adapter.pt"
    )
    assert value(candidate, "--adaptation") == (
        "length_gated_eos_vocab_row"
    )
    assert value(candidate, "--rank") == "1"
    assert value(candidate, "--alpha") == "1.0"
    assert value(training, "--parent-exact-baseline-result") == str(
        paths["parent_eval_output"] / "results.json"
    )
    assert value(training, "--parent-exact-baseline-examples") == str(
        paths["parent_eval_output"] / "examples.jsonl"
    )
    assert value(
        training,
        "--parent-exact-baseline-run-manifest",
    ) == str(paths["parent_eval_output"] / "run_manifest.json")

    chain = preflight._registered_adapter_chain(paths)
    assert chain["load_order"] == [
        "frozen_parent_qkvo",
        "eos_vocab_row_child",
    ]
    assert chain["parent"] == {
        "path": str(paths["parent_adapter"]),
        "sha256": preflight.PARENT_ADAPTER_SHA256,
        "adaptation": "qkvo_answer",
        "rank": 64,
        "alpha": 128.0,
        "frozen_during_child_training": True,
    }
    assert chain["child"] == {
        "path": str(paths["run_output"] / "adapter.pt"),
        "adaptation": "length_gated_eos_vocab_row",
        "rank": 1,
        "alpha": 1.0,
        "modified_vocab_rows": [preflight.OLMO2_EOS_TOKEN_ID],
        "trainable_scope": (
            "long_only_eos_vocab_row_plus_scalar_bias"
        ),
        "parent_adapter_sha256": preflight.PARENT_ADAPTER_SHA256,
        "parameterization": "direct_single_eos_row_delta",
        "rank1_equivalent": True,
        "trainable_parameters": 2_049,
        "trainable_parameter_tensors": 2,
    }


def test_ready_receipt_binds_every_training_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_files = {
        "config.json": b"config",
        "model.safetensors": b"model",
        "tokenizer.json": b"tokenizer",
    }
    for filename, payload in checkpoint_files.items():
        (checkpoint / filename).write_bytes(payload)
    checkpoint_ready = tmp_path / "checkpoint_ready.json"
    checkpoint_ready.write_text("{}\n", encoding="utf-8")
    parent_adapter = tmp_path / "parent.pt"
    parent_adapter.write_bytes(b"parent")
    parent_sha = trainer.sha256_file(parent_adapter)
    monkeypatch.setattr(trainer, "PARENT_ADAPTER_SHA256", parent_sha)
    monkeypatch.setattr(preflight, "PARENT_ADAPTER_SHA256", parent_sha)

    prepared_data = tmp_path / "prepared"
    natural = prepared_data / "longalign_paired_L4096"
    natural.mkdir(parents=True)
    natural_files = {
        "manifest.json": b"{}",
        "input_ids.npy": b"input",
        "assistant_mask.npy": b"mask",
        "lengths.npy": b"lengths",
        "split.npy": b"split",
    }
    for filename, payload in natural_files.items():
        (natural / filename).write_bytes(payload)

    routing_data = tmp_path / "routing"
    routing_data.mkdir()
    routing_manifest = routing_data / "manifest.json"
    routing_manifest.write_text("{}\n", encoding="utf-8")
    routing_splits = {}
    for split_name in ("train", "calibration"):
        split_root = routing_data / split_name
        split_root.mkdir()
        split_files = {
            "manifest.json": f"{split_name}-manifest".encode(),
            "input_ids.npy": f"{split_name}-input".encode(),
            "labels.npy": f"{split_name}-labels".encode(),
            "rows.jsonl": f"{split_name}-rows".encode(),
        }
        for filename, payload in split_files.items():
            (split_root / filename).write_bytes(payload)
        routing_splits[split_name] = {
            "manifest": {
                "sha256": trainer.sha256_file(
                    split_root / "manifest.json"
                )
            },
            "files": {
                filename: {
                    "sha256": trainer.sha256_file(
                        split_root / filename
                    )
                }
                for filename in (
                    "input_ids.npy",
                    "labels.npy",
                    "rows.jsonl",
                )
            },
        }

    output = tmp_path / "run"
    parent_eval_output = tmp_path / "parent_eval"
    candidate_eval_output = tmp_path / "candidate_eval"
    ready_path = tmp_path / "ready.json"
    command_paths = {
        "checkpoint": checkpoint,
        "checkpoint_ready_receipt": checkpoint_ready,
        "parent_adapter": parent_adapter,
        "routing_data_v2": routing_data,
        "prepared_data": prepared_data,
        "exact_eval_data": tmp_path / "exact_eval_data",
        "run_output": output,
        "parent_eval_output": parent_eval_output,
        "candidate_eval_output": candidate_eval_output,
        "gate_output": tmp_path / "gate.json",
        "receipt_output": ready_path,
    }

    trainer_path = Path(trainer.__file__).resolve()
    code_paths = {
        "preflight": trainer_path.with_name(
            "preflight_4k_length_gated_eos.py"
        ),
        "preflight_validation_helpers": trainer_path.with_name(
            "preflight_4k_query_gap_eos_repair.py"
        ),
        "trainer": trainer_path,
        "shared_trainer": trainer_path.with_name(
            "train_4k_counterfactual_routing.py"
        ),
        "routing_data_contract": trainer_path.with_name(
            "prepare_4k_routing_pairs.py"
        ),
        "training_primitives": trainer_path.with_name(
            "train_screen.py"
        ),
        "checkpoint_contract": trainer_path.with_name(
            "train_4k_stage_a.py"
        ),
        "method": trainer_path.with_name(
            "olmo2_length_gated_method.py"
        ),
        "conversion": (
            trainer_path.parents[1] / "olmo2_lora_conversion.py"
        ),
        "model_loader_attention_dependency": (
            trainer_path.parents[1] / "olmo2_1b_evq" / "train.py"
        ),
        "adapter_loader": (
            trainer_path.parents[1] / "olmo2_lora_ood_factorial.py"
        ),
        "shared_training_utils": (
            trainer_path.parents[1] / "small_model_lora_conversion.py"
        ),
        "evq_contract": (
            trainer_path.parents[1] / "olmo2_1b_evq" / "contract.py"
        ),
        "exact_evaluator": trainer_path.with_name(
            "evaluate_instruct_ruler_screen.py"
        ),
        "exact_gate": trainer_path.with_name(
            "gate_olmo2_length_gated_exact_screen.py"
        ),
        "parent_raw_exact_validator": trainer_path.with_name(
            "gate_olmo2_exact_screen.py"
        ),
    }
    receipt = {
        "status": trainer.READY_STATUS,
        "protocol": trainer.protocol(),
        "inputs": {
            "checkpoint": {
                "path": str(checkpoint),
                **{
                    name: {
                        "sha256": trainer.sha256_file(
                            checkpoint / filename
                        )
                    }
                    for name, filename in (
                        ("config", "config.json"),
                        ("model", "model.safetensors"),
                        ("tokenizer", "tokenizer.json"),
                    )
                },
            },
            "checkpoint_ready_receipt": {
                "path": str(checkpoint_ready),
                "sha256": trainer.sha256_file(checkpoint_ready),
            },
            "parent_adapter": {
                "path": str(parent_adapter),
                "sha256": trainer.sha256_file(parent_adapter),
            },
            "prepared_data": {
                "path": str(prepared_data),
                "natural_manifest_sha256": trainer.sha256_file(
                    natural / "manifest.json"
                ),
                "natural_replay": {
                    "files": {
                        filename: {
                            "sha256": trainer.sha256_file(
                                natural / filename
                            )
                        }
                        for filename in natural_files
                    }
                },
            },
            "routing_data": {
                "path": str(routing_data),
                "manifest_sha256": trainer.sha256_file(
                    routing_manifest
                ),
                "splits": routing_splits,
            },
        },
        "run_output": str(output),
        "code": {
            name: {
                "path": str(path),
                "sha256": trainer.sha256_file(path),
            }
            for name, path in code_paths.items()
        },
        "registered_outputs": {
            "parent_exact_baseline": str(parent_eval_output),
            "candidate_exact_screen": str(candidate_eval_output),
        },
        "registered_adapter_chain": {
            "load_order": [
                "frozen_parent_qkvo",
                "eos_vocab_row_child",
            ],
            "parent": {
                "path": str(parent_adapter),
                "sha256": parent_sha,
                "adaptation": "qkvo_answer",
                "rank": 64,
                "alpha": 128.0,
                "frozen_during_child_training": True,
            },
            "child": {
                "path": str(output / "adapter.pt"),
                "adaptation": "length_gated_eos_vocab_row",
                "rank": 1,
                "alpha": 1.0,
                "modified_vocab_rows": [gate.OLMO2_EOS_TOKEN_ID],
                "trainable_scope": (
                    "long_only_eos_vocab_row_plus_scalar_bias"
                ),
                "parent_adapter_sha256": parent_sha,
                "parameterization": "direct_single_eos_row_delta",
                "rank1_equivalent": True,
                "trainable_parameters": 2_049,
                "trainable_parameter_tensors": 2,
            },
            "code_sha256": {
                "candidate_evaluator": trainer.sha256_file(
                    code_paths["exact_evaluator"]
                ),
                "parent_adapter_loader": trainer.sha256_file(
                    code_paths["adapter_loader"]
                ),
                "parent_child_method": trainer.sha256_file(
                    code_paths["method"]
                ),
            },
        },
        "registered_parent_exact_baseline": {
            "output": str(parent_eval_output),
            "result": str(parent_eval_output / "results.json"),
            "examples": str(parent_eval_output / "examples.jsonl"),
            "run_manifest": str(parent_eval_output / "run_manifest.json"),
            "adapter_sha256": parent_sha,
            "adaptation": "qkvo_answer",
            "rank": 64,
            "alpha": 128.0,
            "evaluator_sha256": trainer.sha256_file(
                code_paths["exact_evaluator"]
            ),
            "raw_validator_sha256": trainer.sha256_file(
                code_paths["parent_raw_exact_validator"]
            ),
            "required_before_training": True,
        },
        "registered_commands": preflight._registered_commands(command_paths),
    }
    ready_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    observed = trainer._validate_ready(
        path=ready_path,
        checkpoint=checkpoint,
        checkpoint_ready=checkpoint_ready,
        parent_adapter=parent_adapter,
        prepared_data=prepared_data,
        routing_data=routing_data,
        output=output,
    )
    assert observed == receipt

    receipt["code"]["training_primitives"]["sha256"] = "0" * 64
    ready_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="training_primitives changed"):
        trainer._validate_ready(
            path=ready_path,
            checkpoint=checkpoint,
            checkpoint_ready=checkpoint_ready,
            parent_adapter=parent_adapter,
            prepared_data=prepared_data,
            routing_data=routing_data,
            output=output,
        )
