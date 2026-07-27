from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    evaluate_instruct_ruler_screen as evaluate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    gate_olmo2_exact_screen as exact_gate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    gate_olmo2_hybrid_exact_screen as hybrid_gate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    olmo2_exact_method as exact_method,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    prepare_4k_routing_pairs as prepare,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    preflight_4k_query_gap_eos_repair as eos_preflight,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    train_4k_counterfactual_routing as train,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    train_4k_hybrid_exact as hybrid_train,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    LoRALinear,
)


class _FakeTokenizer:
    pad_token_id = 100_277
    eos_token_id = prepare.OLMO2_EOS_TOKEN_ID

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert messages and add_generation_prompt
        return [1, 50]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> SimpleNamespace:
        assert not add_special_tokens
        if text.isdecimal():
            return SimpleNamespace(input_ids=[100 + int(text)])
        return SimpleNamespace(input_ids=[60])

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not clean_up_tokenization_spaces
        decoded = []
        for value in token_ids:
            token = int(value)
            if token == 1:
                if not skip_special_tokens:
                    decoded.append("<SPECIAL>")
            elif token == self.eos_token_id:
                if not skip_special_tokens:
                    decoded.append("<EOS>")
            else:
                decoded.append(str(token - 100))
        return "".join(decoded)


def test_routing_pair_supervises_answer_then_final_eos(tmp_path) -> None:
    output = tmp_path / "routing"
    row = {
        "input": "The code for amber is 1234567.",
        "outputs": ["1234567"],
        "answer_prefix": "Answer: ",
        "token_position_answer": 100,
        "length": prepare.LENGTH,
    }

    manifest = prepare.build_pair_set(
        output=output,
        tokenizer=_FakeTokenizer(),
        rows=[row],
        seed=7,
        purpose="test",
    )

    metadata = json.loads(
        (output / "rows.jsonl").read_text(encoding="utf-8")
    )
    labels = np.load(output / "labels.npy", allow_pickle=False)
    inputs = np.load(output / "input_ids.npy", allow_pickle=False)
    answer_start = int(metadata["answer_start"])
    eos_position = int(metadata["eos_position"])

    assert manifest["format_version"] == 2
    assert manifest["status"] == prepare.SET_STATUS
    assert manifest["supervision_contract"] == (
        prepare.SUPERVISION_CONTRACT
    )
    assert manifest["eos_token_id"] == prepare.OLMO2_EOS_TOKEN_ID
    assert manifest["final_eos_supervised"] is True
    assert manifest["answer_string_tokenizer_roundtrip_exact"] is True
    assert eos_position == answer_start + metadata["answer_tokens"]
    assert labels[0, 0, answer_start] == metadata["gold_token_ids"][0]
    assert labels[0, 1, answer_start] == metadata["alternate_token_ids"][0]
    assert labels[0, :, eos_position].tolist() == [
        prepare.OLMO2_EOS_TOKEN_ID,
        prepare.OLMO2_EOS_TOKEN_ID,
    ]
    assert inputs[0, :, eos_position].tolist() == [
        prepare.OLMO2_EOS_TOKEN_ID,
        prepare.OLMO2_EOS_TOKEN_ID,
    ]
    assert np.all(labels[0, :, eos_position + 1 :] == -100)
    assert manifest["supervised_answer_tokens"] == 2
    assert manifest["supervised_eos_tokens"] == 2
    assert manifest["supervised_answer_and_eos_tokens"] == 4

    train.RoutingPairView(output)


def test_routing_pair_view_rejects_missing_eos_contract(tmp_path) -> None:
    output = tmp_path / "routing"
    row = {
        "input": "The code for amber is 1234567.",
        "outputs": ["1234567"],
        "answer_prefix": "Answer: ",
        "token_position_answer": 100,
        "length": prepare.LENGTH,
    }
    prepare.build_pair_set(
        output=output,
        tokenizer=_FakeTokenizer(),
        rows=[row],
        seed=7,
        purpose="test",
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_eos_supervised"] = False
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="lacks the answer-plus-immediate-EOS contract",
    ):
        train.RoutingPairView(output)


def test_nonvirtual_view_rejects_tampered_eos_label(tmp_path) -> None:
    output = tmp_path / "routing"
    row = {
        "input": "The code for amber is 1234567.",
        "outputs": ["1234567"],
        "answer_prefix": "Answer: ",
        "token_position_answer": 100,
        "length": prepare.LENGTH,
    }
    prepare.build_pair_set(
        output=output,
        tokenizer=_FakeTokenizer(),
        rows=[row],
        seed=7,
        purpose="test",
    )
    metadata = json.loads(
        (output / "rows.jsonl").read_text(encoding="utf-8")
    )
    labels_path = output / "labels.npy"
    labels = np.load(labels_path, mmap_mode="r+", allow_pickle=False)
    labels[0, 0, int(metadata["eos_position"])] = 17
    labels.flush()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["labels.npy"] = prepare.sha256_file(labels_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="final-EOS label drift"):
        train.RoutingPairView(
            output,
            require_virtual_geometry=False,
        )


def test_routing_objective_separates_answer_eos_and_margin() -> None:
    model = SimpleNamespace(lm_head=nn.Linear(5, 5, bias=False))
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.eye(5))
    labels = torch.tensor([[1, 2, -100]])
    alternate = torch.tensor([[3, 2, -100]])
    hidden = torch.zeros((1, 3, 5))
    hidden[0, 0, 1] = 4.0
    hidden[0, 1, 2] = 4.0

    _, metrics = train.routing_objective(
        model=model,
        hidden=hidden,
        labels=labels,
        alternate_labels=alternate,
        eos_token_id=2,
        margin=1.0,
        margin_weight=0.5,
        termination_weight=1.0,
    )
    changed_eos_hidden = hidden.clone()
    changed_eos_hidden[0, 1].zero_()
    changed_eos_hidden[0, 1, 4] = 4.0
    _, changed_metrics = train.routing_objective(
        model=model,
        hidden=changed_eos_hidden,
        labels=labels,
        alternate_labels=alternate,
        eos_token_id=2,
        margin=1.0,
        margin_weight=0.5,
        termination_weight=1.0,
    )

    assert metrics["answer_tokens"] == 1.0
    assert metrics["termination_eos_tokens"] == 1.0
    assert metrics["source_tokens"] == 1.0
    assert changed_metrics["preference_mean"] == metrics["preference_mean"]
    assert changed_metrics["counterfactual_loss"] == (
        metrics["counterfactual_loss"]
    )
    assert changed_metrics["termination_eos_ce"] > (
        metrics["termination_eos_ce"]
    )


def test_realized_gap_curriculum_hits_registered_bands() -> None:
    targets = train.deterministic_realized_gap_target_stream(
        seed=20260728,
        routing_steps=22,
    )
    assert len(targets) == 88
    assert int((targets == -1).sum()) == 11
    assert int(
        ((targets >= 4096) & (targets < 8192)).sum()
    ) == 11
    assert int(
        ((targets >= 8192) & (targets < 12288)).sum()
    ) == 22
    assert int(
        ((targets >= 12288) & (targets < 16384)).sum()
    ) == 44

    view = SimpleNamespace(
        answer_starts=np.asarray([3960, 3900, 3800, 3700]),
        source_stops=np.asarray([3000, 2000, 1000, 500]),
        active_lengths=np.asarray([3964, 3904, 3804, 3704]),
    )
    row_indices = np.arange(4, dtype=np.int64)
    selected_targets = np.asarray([-1, 5000, 9000, 14000])
    offsets = train.realized_gap_query_offsets(
        view=view,
        row_indices=row_indices,
        gap_targets=selected_targets,
    )
    physical = (
        view.answer_starts[row_indices] - view.source_stops[row_indices]
    )
    realized = physical + offsets
    assert [train.query_gap_band(int(value)) for value in realized] == [
        "contiguous",
        "transition",
        "middle",
        "far",
    ]
    assert realized[1:].tolist() == selected_targets[1:].tolist()


def test_exact_generation_requires_whole_string_and_terminal_eos() -> None:
    exact = evaluate.exact_generation_metrics(
        prediction="1234567",
        references=["1234567"],
        generated_token_ids=[101, 2],
        reference_token_ids=[[101]],
        eos_token_id=2,
    )
    extra_text = evaluate.exact_generation_metrics(
        prediction="1234567 extra",
        references=["1234567"],
        generated_token_ids=[101, 102, 2],
        reference_token_ids=[[101]],
        eos_token_id=2,
    )
    missing_eos = evaluate.exact_generation_metrics(
        prediction="1234567",
        references=["1234567"],
        generated_token_ids=[101],
        reference_token_ids=[[101]],
        eos_token_id=2,
    )
    hidden_extra_special = evaluate.exact_generation_metrics(
        prediction="1234567",
        references=["1234567"],
        generated_token_ids=[1, 101, 2],
        reference_token_ids=[[101]],
        eos_token_id=2,
    )

    assert exact == {
        "full_string_exact": 1.0,
        "eos_terminated": 1.0,
        "answer_eos_token_exact": 1.0,
        "exact_generation_pass": 1.0,
    }
    assert extra_text["eos_terminated"] == 1.0
    assert extra_text["full_string_exact"] == 0.0
    assert extra_text["exact_generation_pass"] == 0.0
    assert missing_eos["full_string_exact"] == 1.0
    assert missing_eos["eos_terminated"] == 0.0
    assert missing_eos["exact_generation_pass"] == 0.0
    assert hidden_extra_special["full_string_exact"] == 1.0
    assert hidden_extra_special["answer_eos_token_exact"] == 0.0
    assert hidden_extra_special["exact_generation_pass"] == 1.0


def test_evaluator_removes_only_terminal_eos_before_literal_decode() -> None:
    tokenizer = _FakeTokenizer()
    exact_prediction = evaluate.decode_generated_string(
        tokenizer=tokenizer,
        generated_token_ids=[101, tokenizer.eos_token_id],
        eos_token_id=tokenizer.eos_token_id,
    )
    extra_special_prediction = evaluate.decode_generated_string(
        tokenizer=tokenizer,
        generated_token_ids=[1, 101, tokenizer.eos_token_id],
        eos_token_id=tokenizer.eos_token_id,
    )

    assert exact_prediction == "1"
    assert extra_special_prediction == "<SPECIAL>1"
    metrics = evaluate.exact_generation_metrics(
        prediction=extra_special_prediction,
        references=["1"],
        generated_token_ids=[1, 101, tokenizer.eos_token_id],
        reference_token_ids=[[101]],
        eos_token_id=tokenizer.eos_token_id,
    )
    assert metrics["eos_terminated"] == 1.0
    assert metrics["full_string_exact"] == 0.0
    assert metrics["exact_generation_pass"] == 0.0


def _gate_receipts() -> tuple[dict[str, object], dict[str, object]]:
    frequency = {
        "active_frequency": "evq_endpoint_cosh",
        "active_sha256_float32": exact_gate.EVQ_SHA256,
    }
    adapter_sha = "a" * 64
    parent_sha = "b" * 64
    routing_sha = "c" * 64
    training: dict[str, object] = {
        "status": exact_gate.TRAINING_STATUS,
        "checkpoint_sha256": exact_gate.MODEL_SHA256,
        "script_sha256": "d" * 64,
        "experiment_ready_receipt_sha256": "e" * 64,
        "bound_code_sha256": {"trainer": "3" * 64},
        "parent_adapter_sha256": parent_sha,
        "frequency": frequency,
        "adapter_sha256": adapter_sha,
        "routing_data": {
            "format_version": 2,
            "manifest_sha256": routing_sha,
            "supervision_contract": exact_gate.SUPERVISION_CONTRACT,
            "eos_token_id": exact_gate.OLMO2_EOS_TOKEN_ID,
            "final_eos_supervised": True,
            "labels_only_cover_answer_and_final_eos": True,
            "answer_string_tokenizer_roundtrip_exact": True,
        },
        "protocol": {
            "maximum_physical_training_sequence_length": 4096,
            "maximum_physical_token_index": 4095,
            "maximum_allowed_position_id": 16383,
            "hard_maximum_training_length": 4096,
            "virtual_target_length": 16384,
            "position_policy": (
                "semantic_query_block_realized_gap_curriculum"
            ),
            "supervision_contract": exact_gate.SUPERVISION_CONTRACT,
            "supervision": (
                "answer_ce_plus_weighted_immediate_eos_ce"
            ),
            "eos_token_id": exact_gate.OLMO2_EOS_TOKEN_ID,
            "final_eos_supervised": True,
            "counterfactual_margin_scope": (
                "answer_tokens_where_gold_differs"
            ),
        },
        "training": {"maximum_observed_position_id": 16000},
    }
    evaluation: dict[str, object] = {
        "status": exact_gate.EVALUATION_STATUS,
        "checkpoint_sha256": exact_gate.MODEL_SHA256,
        "script_sha256": "f" * 64,
        "run_manifest_sha256": "1" * 64,
        "experiment_ready_receipt_sha256": "e" * 64,
        "bound_code_sha256": {"evaluator": "4" * 64},
        "frequency": frequency,
        "adapter": {
            "sha256": adapter_sha,
            "metadata": {
                "base_checkpoint_sha256": exact_gate.MODEL_SHA256,
                "frequency": "evq",
                "frequency_sha256_float32": exact_gate.EVQ_SHA256,
                "adaptation": "qkvo_answer",
                "rank": 64,
                "alpha": 128.0,
                "training_sequence_length": 4096,
                "final_eos_supervised": True,
                "stage": (
                    "counterfactual_routing_realized_gap_16k_eos_v2"
                ),
                "parent_adapter_sha256": parent_sha,
                "routing_data_sha256": routing_sha,
                "position_policy": (
                    "semantic_query_block_realized_gap_curriculum"
                ),
                "supervision_contract": (
                    exact_gate.SUPERVISION_CONTRACT
                ),
                "eos_token_id": exact_gate.OLMO2_EOS_TOKEN_ID,
            },
        },
        "protocol": {
            "task": "niah_single_1",
            "lengths": [4096, 8192, 16384],
            "limit_per_length": 8,
            "greedy": True,
            "maximum_new_tokens": exact_gate.GENERATION_TOKENS,
            "string_normalization": "none",
            "decode_cleanup": False,
            "terminal_eos_removed_before_string_decode": True,
            "other_special_tokens_removed": False,
            "substring_is_success": False,
            "first_number_is_success": False,
            "training_length_if_adapted": 4096,
        },
        "data": {"manifest_sha256": "2" * 64},
        "results": {
            "cells": {
                "4096": {
                    "examples": 8,
                    "full_string_exact": 1.0,
                    "eos_terminated": 1.0,
                    "answer_eos_token_exact": 1.0,
                    "exact_generation_pass": 1.0,
                },
                "8192": {
                    "examples": 8,
                    "full_string_exact": 0.75,
                    "eos_terminated": 1.0,
                    "answer_eos_token_exact": 0.75,
                    "exact_generation_pass": 0.75,
                },
                "16384": {
                    "examples": 8,
                    "full_string_exact": 0.25,
                    "eos_terminated": 1.0,
                    "answer_eos_token_exact": 0.25,
                    "exact_generation_pass": 0.25,
                },
            }
        },
    }
    return training, evaluation


def _exact_gate_raw_rows(
    evaluation: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cells = evaluation["results"]["cells"]
    for length in exact_gate.SCREEN_POLICY:
        cell = cells[str(length)]
        examples = int(cell["examples"])
        full_count = round(float(cell["full_string_exact"]) * examples)
        eos_count = round(float(cell["eos_terminated"]) * examples)
        exact_count = round(
            float(cell["exact_generation_pass"]) * examples
        )
        for local_index in range(examples):
            full = local_index < full_count
            eos = local_index < eos_count
            exact = local_index < exact_count
            assert exact == (full and eos)
            token_ids = [101]
            if not full:
                token_ids = [999]
            if eos:
                token_ids.append(exact_gate.OLMO2_EOS_TOKEN_ID)
            source = {
                "index": local_index,
                "token_position_answer": 100 + local_index,
                "outputs": ["1"],
            }
            rows.append(
                {
                    "schema_version": exact_gate.EXAMPLE_SCHEMA_VERSION,
                    "task": "niah_single_1",
                    "nominal_length": length,
                    "local_index": local_index,
                    "prediction": "1" if full else "wrong",
                    "references": ["1"],
                    "generated_token_ids": token_ids,
                    "generated_tokens": len(token_ids),
                    "source_row_index": source["index"],
                    "source_token_position_answer": source[
                        "token_position_answer"
                    ],
                    "row_sha256": exact_gate.row_sha256(source),
                    "full_string_exact": float(full),
                    "eos_terminated": float(eos),
                    "exact_generation_pass": float(exact),
                }
            )
    return rows


def _exact_gate_run_manifest(
    training: dict[str, object],
    evaluation: dict[str, object],
) -> dict[str, object]:
    return {
        "status": exact_gate.RUN_MANIFEST_STATUS,
        "example_schema_version": exact_gate.EXAMPLE_SCHEMA_VERSION,
        "checkpoint_sha256": exact_gate.MODEL_SHA256,
        "experiment_ready_receipt_sha256": "e" * 64,
        "experiment_role": "candidate_exact_screen",
        "bound_code_sha256": {"evaluator": "4" * 64},
        "data_manifest_sha256": evaluation["data"]["manifest_sha256"],
        "frequency": "evq",
        "adapter_sha256": training["adapter_sha256"],
        "adaptation": "qkvo_answer",
        "rank": 64,
        "alpha": 128.0,
        "task": "niah_single_1",
        "lengths": list(exact_gate.SCREEN_POLICY),
        "limit_per_length": 8,
        "greedy": True,
        "maximum_new_tokens": exact_gate.GENERATION_TOKENS,
        "string_normalization": "none",
        "decode_cleanup": False,
        "terminal_eos_removed_before_string_decode": True,
        "other_special_tokens_removed": False,
    }


def _exact_gate_sources() -> dict[tuple[int, int], dict[str, object]]:
    return {
        (length, local_index): {
            "index": local_index,
            "token_position_answer": 100 + local_index,
            "outputs": ["1"],
        }
        for length in exact_gate.SCREEN_POLICY
        for local_index in range(8)
    }


def _exact_gate_decode(token_ids: list[int]) -> str:
    content = list(token_ids)
    if content and content[-1] == exact_gate.OLMO2_EOS_TOKEN_ID:
        content.pop()
    return "1" if content == [101] else "wrong"


def _exact_gate_ready(
    training: dict[str, object],
    evaluation: dict[str, object],
) -> dict[str, object]:
    return {
        "status": "OLMO2_4K_QUERY_GAP_EOS_REPAIR_READY_V1",
        "trainer": {"sha256": training["script_sha256"]},
        "evaluator": {"sha256": evaluation["script_sha256"]},
        "gate": {"sha256": "5" * 64},
        "bound_code_sha256": training["bound_code_sha256"],
        "evaluator_bound_code_sha256": evaluation["bound_code_sha256"],
        "gate_bound_code_sha256": exact_gate.bound_code_sha256(),
        "protocol": training["protocol"],
        "inputs": {
            "parent_adapter": {"sha256": "b" * 64},
        },
    }


def _parent_exact_evaluation(
    evaluation: dict[str, object],
) -> dict[str, object]:
    parent = copy.deepcopy(evaluation)
    parent["adapter"]["sha256"] = "b" * 64
    return parent


def test_minimal_exact_gate_passes_only_exact_eos_counts() -> None:
    training, evaluation = _gate_receipts()
    evaluation["results"]["cells"]["8192"][
        "answer_eos_token_exact"
    ] = 0.0

    run_manifest = _exact_gate_run_manifest(training, evaluation)
    parent_evaluation = _parent_exact_evaluation(evaluation)
    parent_run_manifest = {
        **run_manifest,
        "adapter_sha256": "b" * 64,
        "experiment_role": "parent_exact_baseline",
    }
    passed, details = exact_gate.exact_screen_gate(
        training,
        evaluation,
        parent_evaluation,
        expected_parent_sha256="b" * 64,
        raw_rows=_exact_gate_raw_rows(evaluation),
        run_manifest=run_manifest,
        parent_raw_rows=_exact_gate_raw_rows(parent_evaluation),
        parent_run_manifest=parent_run_manifest,
        source_rows=_exact_gate_sources(),
        decode_generated=_exact_gate_decode,
        experiment_ready=_exact_gate_ready(training, evaluation),
        experiment_ready_sha256="e" * 64,
        gate_script_sha256="5" * 64,
    )

    assert passed is True
    assert details["expanded_evaluation_authorized"] is True
    assert details["counts"]["16384"]["exact_generation_passes"] == 2


def test_minimal_exact_gate_ignores_first_number_success() -> None:
    training, evaluation = _gate_receipts()
    cell = evaluation["results"]["cells"]["16384"]
    cell["full_string_exact"] = 0.125
    cell["answer_eos_token_exact"] = 0.125
    cell["exact_generation_pass"] = 0.125
    cell["first_number_exact"] = 1.0
    cell["official_string_match"] = 1.0

    run_manifest = _exact_gate_run_manifest(training, evaluation)
    parent_evaluation = _parent_exact_evaluation(evaluation)
    parent_run_manifest = {
        **run_manifest,
        "adapter_sha256": "b" * 64,
        "experiment_role": "parent_exact_baseline",
    }
    passed, details = exact_gate.exact_screen_gate(
        training,
        evaluation,
        parent_evaluation,
        expected_parent_sha256="b" * 64,
        raw_rows=_exact_gate_raw_rows(evaluation),
        run_manifest=run_manifest,
        parent_raw_rows=_exact_gate_raw_rows(parent_evaluation),
        parent_run_manifest=parent_run_manifest,
        source_rows=_exact_gate_sources(),
        decode_generated=_exact_gate_decode,
        experiment_ready=_exact_gate_ready(training, evaluation),
        experiment_ready_sha256="e" * 64,
        gate_script_sha256="5" * 64,
    )

    assert passed is False
    assert details["expanded_evaluation_authorized"] is False


def test_eos_ready_protocol_is_the_trainer_protocol() -> None:
    expected = train.registered_protocol(
        SimpleNamespace(
            frequency="evq",
            steps=eos_preflight.STEPS,
            micro_batch_size=eos_preflight.MICRO_BATCH_SIZE,
            gradient_accumulation_steps=(
                eos_preflight.GRADIENT_ACCUMULATION_STEPS
            ),
            rank=eos_preflight.RANK,
            alpha=eos_preflight.ALPHA,
            learning_rate=eos_preflight.LEARNING_RATE,
            warmup_steps=eos_preflight.WARMUP_STEPS,
            counterfactual_margin=eos_preflight.COUNTERFACTUAL_MARGIN,
            counterfactual_margin_weight=(
                eos_preflight.COUNTERFACTUAL_MARGIN_WEIGHT
            ),
            termination_weight=eos_preflight.TERMINATION_WEIGHT,
            compile_mode=eos_preflight.COMPILE_MODE,
            natural_eval_rows=eos_preflight.NATURAL_EVAL_ROWS,
            virtual_target_length=eos_preflight.VIRTUAL_TARGET_LENGTH,
            virtual_bucket_weights=eos_preflight.VIRTUAL_BUCKET_WEIGHTS,
            seed=eos_preflight.SEED,
        ),
        virtual_query_gap=True,
    )

    assert eos_preflight.protocol() == expected
    assert expected["final_eos_supervised"] is True
    assert expected["maximum_physical_training_sequence_length"] == 4096
    assert expected["maximum_allowed_position_id"] == 16383


def _hybrid_gate_receipts() -> tuple[
    dict[str, object],
    dict[str, object],
]:
    frequency = {
        "active_frequency": "hybrid_evq_low12",
        "active_sha256_float32": hybrid_gate.HYBRID_SHA256,
    }
    metadata = {
        "base_checkpoint_sha256": hybrid_gate.MODEL_SHA256,
        "frequency": "hybrid_evq_low12",
        "frequency_sha256_float32": hybrid_gate.HYBRID_SHA256,
        "adaptation": "qk_answer",
        "qk_output_mask_sha256": hybrid_gate.QK_MASK_SHA256,
        "rank": 32,
        "alpha": 64.0,
        "training_sequence_length": 4096,
        "maximum_physical_training_sequence_length": 4096,
        "virtual_target_length": 16384,
        "final_eos_supervised": True,
        "supervision_contract": hybrid_gate.SUPERVISION_CONTRACT,
        "parent_adapter_sha256": None,
    }
    adapter_sha = "a" * 64
    training: dict[str, object] = {
        "status": hybrid_gate.TRAINING_STATUS,
        "checkpoint_sha256": hybrid_gate.MODEL_SHA256,
        "frequency": frequency,
        "adapter_sha256": adapter_sha,
        "adapter_metadata": metadata,
        "bound_code": {
            "exact_evaluator": hybrid_gate.sha256_file(
                Path(evaluate.__file__).resolve()
            ),
            "exact_gate": hybrid_gate.sha256_file(
                Path(hybrid_gate.__file__).resolve()
            ),
        },
        "routing_data": {
            "format_version": 2,
            "final_eos_supervised": True,
            "labels_only_cover_answer_and_final_eos": True,
            "answer_string_tokenizer_roundtrip_exact": True,
        },
        "protocol": hybrid_train.protocol(),
    }
    evaluation: dict[str, object] = {
        "status": hybrid_gate.EVALUATION_STATUS,
        "checkpoint_sha256": hybrid_gate.MODEL_SHA256,
        "script_sha256": hybrid_gate.sha256_file(
            Path(evaluate.__file__).resolve()
        ),
        "frequency": frequency,
        "adapter": {
            "sha256": adapter_sha,
            "qk_output_mask_sha256": hybrid_gate.QK_MASK_SHA256,
            "metadata": metadata,
        },
        "protocol": {
            "task": "niah_single_1",
            "lengths": [4096, 8192, 16384],
            "limit_per_length": 8,
            "greedy": True,
            "string_normalization": "none",
            "decode_cleanup": False,
            "terminal_eos_removed_before_string_decode": True,
            "other_special_tokens_removed": False,
            "substring_is_success": False,
            "first_number_is_success": False,
            "training_length_if_adapted": 4096,
        },
        "results": {
            "cells": {
                "4096": {
                    "examples": 8,
                    "full_string_exact": 1.0,
                    "eos_terminated": 1.0,
                    "answer_eos_token_exact": 1.0,
                    "exact_generation_pass": 1.0,
                },
                "8192": {
                    "examples": 8,
                    "full_string_exact": 0.75,
                    "eos_terminated": 1.0,
                    "answer_eos_token_exact": 0.75,
                    "exact_generation_pass": 0.75,
                },
                "16384": {
                    "examples": 8,
                    "full_string_exact": 0.25,
                    "eos_terminated": 1.0,
                    "answer_eos_token_exact": 0.25,
                    "exact_generation_pass": 0.25,
                },
            }
        },
    }
    return training, evaluation


def test_hybrid_gate_enforces_exact_eos_and_no_long_training() -> None:
    training, evaluation = _hybrid_gate_receipts()

    passed, details = hybrid_gate.exact_screen_gate(
        training,
        evaluation,
    )

    assert passed is True
    assert details["expanded_evaluation_authorized"] is True
    assert hybrid_train.protocol()["real_8k_or_16k_training_sequences"] == 0
    assert hybrid_train.protocol()["v_and_o_trainable"] is False


def test_hybrid_gate_rejects_first_number_only_success() -> None:
    training, evaluation = _hybrid_gate_receipts()
    cell = evaluation["results"]["cells"]["16384"]
    cell["full_string_exact"] = 0.125
    cell["answer_eos_token_exact"] = 0.125
    cell["exact_generation_pass"] = 0.125
    cell["first_number_exact"] = 1.0
    cell["official_string_match"] = 1.0

    passed, details = hybrid_gate.exact_screen_gate(
        training,
        evaluation,
    )

    assert passed is False
    assert details["expanded_evaluation_authorized"] is False


class _Rotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
            endpoint_geo_inv_freq,
        )

        self.register_buffer(
            "inv_freq",
            endpoint_geo_inv_freq().clone(),
            persistent=False,
        )


def test_olmo_hybrid_low12_frequency_and_qk_mask_contract() -> None:
    config = SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
    )
    model = SimpleNamespace(
        model=SimpleNamespace(rotary_emb=_Rotary())
    )

    receipt = exact_method.apply_hybrid_evq_low12(model)
    mask = exact_method.qk_evq_tail_output_mask(config)
    geometry = exact_method.phase_geometry_receipt()

    assert receipt["active_sha256_float32"] == (
        exact_method.HYBRID_FREQUENCY_SHA256
    )
    assert receipt["native_pair_count"] == 52
    assert receipt["evq_pair_count"] == 12
    assert int(mask.sum()) == 16 * 24
    assert mask[:52].sum() == 0
    assert mask[52:64].sum() == 12
    assert mask[64:116].sum() == 0
    assert mask[116:128].sum() == 12
    assert (
        geometry["lengths"]["4096"]["maximum_absolute_phase_shift"]
        < 0.5
    )
    assert (
        geometry["lengths"]["16384"]["pairs_above_0p5_radians"]
        == 4
    )


def test_qk_masked_lora_leaves_value_and_output_projections_frozen() -> None:
    attention = nn.Module()
    attention.q_proj = nn.Linear(8, 8, bias=False)
    attention.k_proj = nn.Linear(8, 8, bias=False)
    attention.v_proj = nn.Linear(8, 8, bias=False)
    attention.o_proj = nn.Linear(8, 8, bias=False)
    original_v = attention.v_proj
    original_o = attention.o_proj
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].self_attn = attention
    mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)

    install_adaptation(
        model,
        "qk_answer",
        rank=2,
        alpha=4.0,
        qk_output_mask=mask,
    )

    adapted = model.model.layers[0].self_attn
    assert isinstance(adapted.q_proj, LoRALinear)
    assert isinstance(adapted.k_proj, LoRALinear)
    assert adapted.v_proj is original_v
    assert adapted.o_proj is original_o
    assert torch.equal(adapted.q_proj.output_mask, mask)
    assert torch.equal(adapted.k_proj.output_mask, mask)
