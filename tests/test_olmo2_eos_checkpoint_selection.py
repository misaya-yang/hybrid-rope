from __future__ import annotations

import json
from pathlib import Path

import pytest

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    train_4k_counterfactual_routing as shared_trainer,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    train_4k_length_gated_eos as trainer,
)


def _selection_metrics(
    *,
    state_sha256: str,
    termination_counts: tuple[int, int],
    answer_nll_delta: float = 0.0,
    new_answer_eos: bool = False,
) -> dict[str, object]:
    cells: dict[str, dict[str, object]] = {}
    for index, query_offset in enumerate(
        trainer.SELECTION_QUERY_OFFSETS
    ):
        answer_bits = [False, False, False, False]
        if new_answer_eos and index == 1:
            answer_bits[0] = True
        cells[str(query_offset)] = {
            "query_offset": query_offset,
            "supervised_tokens": 20,
            "answer_tokens": 4,
            "termination_tokens": 16,
            "answer_token_exact_count": 3,
            "answer_eos_top1_count": sum(answer_bits),
            "answer_eos_top1_bits": answer_bits,
            "mean_answer_nll": 0.5 + answer_nll_delta,
            "termination_exact_count": termination_counts[index],
            "mean_termination_nll": 0.25,
            "minimum_termination_eos_margin": 0.1,
            "mean_answer_eos_logit_delta": 0.0,
            "maximum_answer_eos_logit_delta": 0.0,
            "mean_termination_eos_logit_delta": 1.0,
            "minimum_termination_eos_logit_delta": 1.0,
        }
    return {
        "cells": cells,
        "parameter_state": {
            "sha256": state_sha256,
            "delta_weight_l2": 1.0,
            "delta_weight_maximum_absolute": 1.0,
            "eos_bias": 0.0,
            "finite": True,
        },
    }


def test_checkpoint_selection_protocol_is_deterministic() -> None:
    selection = trainer.protocol()["checkpoint_selection"]
    assert selection == {
        "rule": "first_passing_checkpoint",
        "maximum_steps": 32,
        "checkpoint_steps": [4, 8, 16, 32],
        "split": "calibration",
        "rows": 8,
        "row_indices": list(range(8)),
        "query_offsets": [4096, 12289],
        "minimum_termination_exact_counts": {
            "4096": 12,
            "12289": 4,
        },
        "answer_nll_tolerance": 1e-4,
        "selection_is_capability_evidence": False,
    }


def test_checkpoint_gate_accepts_first_safe_termination_improvement() -> None:
    parent = _selection_metrics(
        state_sha256="0" * 64,
        termination_counts=(2, 0),
    )
    candidate = _selection_metrics(
        state_sha256="1" * 64,
        termination_counts=(12, 4),
    )
    outcome = trainer._eos_selection_gate(
        parent=parent,
        candidate=candidate,
    )
    assert outcome["passed"] is True
    assert all(outcome["checks"].values())


@pytest.mark.parametrize(
    ("candidate", "failed_check"),
    [
        (
            _selection_metrics(
                state_sha256="1" * 64,
                termination_counts=(12, 3),
            ),
            "query_offset_12289_termination_floor",
        ),
        (
            _selection_metrics(
                state_sha256="1" * 64,
                termination_counts=(12, 4),
                new_answer_eos=True,
            ),
            "query_offset_12289_no_new_answer_eos_top1",
        ),
        (
            _selection_metrics(
                state_sha256="1" * 64,
                termination_counts=(12, 4),
                answer_nll_delta=2e-4,
            ),
            "query_offset_4096_answer_nll_not_worse",
        ),
    ],
)
def test_checkpoint_gate_rejects_unsafe_candidate(
    candidate: dict[str, object],
    failed_check: str,
) -> None:
    parent = _selection_metrics(
        state_sha256="0" * 64,
        termination_counts=(2, 0),
    )
    outcome = trainer._eos_selection_gate(
        parent=parent,
        candidate=candidate,
    )
    assert outcome["passed"] is False
    assert outcome["checks"][failed_check] is False


def test_no_eligible_checkpoint_stops_promotion() -> None:
    with pytest.raises(
        RuntimeError,
        match="stop before adapter promotion",
    ):
        trainer._require_selected_checkpoint(
            {
                "selected_step": None,
                "checkpoint_selection_required": True,
                "checkpoint_selection_passed": False,
                "checkpoint_history": [
                    {"step": step, "passed": False}
                    for step in trainer.SELECTION_CHECKPOINT_STEPS
                ],
            }
        )

    assert (
        trainer._require_selected_checkpoint(
            {
                "selected_step": 8,
                "checkpoint_selection_required": True,
                "checkpoint_selection_passed": True,
                "checkpoint_history": [
                    {"step": 4, "passed": False},
                    {"step": 8, "passed": True},
                ],
            }
        )
        == 8
    )


def test_training_log_accepts_early_stop_cadence(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "train.jsonl"
    rows = []
    for step in (1, 4, 8):
        rows.append(
            {
                "step": step,
                "family": "routing",
                "loss": 1.0,
                "lr": 1e-4,
                "grad_norm": 0.5,
                "processed_input_tokens": step * 4096,
                "elapsed_seconds": float(step),
                "interval_tokens_per_second": 1000.0,
                "peak_memory_allocated_bytes": 1,
                "peak_memory_reserved_bytes": 2,
            }
        )
    log_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = trainer._training_log_receipt(
        log_path,
        training={"actual_steps": 8},
    )
    assert receipt["rows"] == 3
    assert receipt["last_optimizer_step"]["step"] == 8


def test_ready_adapter_chain_is_exact(
    tmp_path: Path,
) -> None:
    parent = (tmp_path / "parent.pt").resolve()
    output = (tmp_path / "output").resolve()
    code = {
        "exact_evaluator": {"sha256": "1" * 64},
        "adapter_loader": {"sha256": "2" * 64},
        "method": {"sha256": "3" * 64},
    }
    chain = {
        "load_order": [
            "frozen_parent_qkvo",
            "eos_vocab_row_child",
        ],
        "parent": {
            "path": str(parent),
            "sha256": trainer.PARENT_ADAPTER_SHA256,
            "adaptation": trainer.PARENT_ADAPTATION,
            "rank": trainer.PARENT_RANK,
            "alpha": trainer.PARENT_ALPHA,
            "frozen_during_child_training": True,
        },
        "child": {
            "path": str(output / "adapter.pt"),
            "adaptation": trainer.ADAPTATION,
            "rank": trainer.EOS_HEAD_RANK,
            "alpha": trainer.EOS_HEAD_ALPHA,
            "modified_vocab_rows": [trainer.OLMO2_EOS_TOKEN_ID],
            "parameterization": "direct_single_eos_row_delta",
            "rank1_equivalent": True,
            "trainable_parameters": 2_049,
            "trainable_parameter_tensors": 2,
            "trainable_scope": (
                "long_only_eos_vocab_row_plus_scalar_bias"
            ),
            "parent_adapter_sha256": (
                trainer.PARENT_ADAPTER_SHA256
            ),
        },
        "code_sha256": {
            "candidate_evaluator": "1" * 64,
            "parent_adapter_loader": "2" * 64,
            "parent_child_method": "3" * 64,
        },
    }
    receipt = {"code": code, "registered_adapter_chain": chain}
    trainer._validate_registered_adapter_chain(
        receipt=receipt,
        parent_adapter=parent,
        output=output,
    )
    receipt["registered_adapter_chain"] = {
        **chain,
        "unregistered": True,
    }
    with pytest.raises(RuntimeError, match="adapter-chain drift"):
        trainer._validate_registered_adapter_chain(
            receipt=receipt,
            parent_adapter=parent,
            output=output,
        )


def test_shared_trainer_rejects_partial_checkpoint_contract() -> None:
    with pytest.raises(
        ValueError,
        match="invalid checkpoint-selection contract",
    ):
        shared_trainer.train(
            model=None,
            routing_view=None,
            calibration_view=None,
            natural_view_path=Path("unused"),
            steps=4,
            micro_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            warmup_steps=0,
            margin=1.0,
            margin_weight=1.0,
            termination_weight=1.0,
            compile_mode="default",
            seed=1,
            log_path=Path("unused"),
            checkpoint_steps=(4,),
            checkpoint_callback=None,
        )
