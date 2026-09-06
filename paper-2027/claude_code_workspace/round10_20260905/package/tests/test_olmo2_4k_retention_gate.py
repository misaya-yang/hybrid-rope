from __future__ import annotations

from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity import (
    gate_olmo2_4k_retention as gate,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_instruct_ruler_transfer import (
    TASK_CONFIGS,
)


def _decode(token_ids: list[int]) -> str:
    return "answer" if token_ids == [1] else "wrong"


def _rows(
    *,
    failed_task: str | None = None,
) -> dict[tuple[str, int], dict[str, object]]:
    result: dict[tuple[str, int], dict[str, object]] = {}
    for task in gate.TASKS:
        metric = str(TASK_CONFIGS[task]["official_metric"])
        for local_index in range(gate.ROWS_PER_TASK):
            failed = task == failed_task
            result[(task, local_index)] = {
                "task": task,
                "nominal_length": gate.LENGTH,
                "local_index": local_index,
                "source_row_index": local_index,
                "row_sha256": f"{task}:{local_index}",
                "references": ["answer"],
                "official_metric": metric,
                "input_tokens": 4000,
                "maximum_generation_tokens": 128,
                "generated_tokens": 1,
                "generated_token_ids": [2] if failed else [1],
                "prediction": "wrong" if failed else "answer",
                "official_task_score": 0.0 if failed else 1.0,
            }
    return result


def _receipt(
    *,
    role: str,
    adapter_sha256: str,
    parent_sha256: str | None,
    rows: dict[tuple[str, int], dict[str, object]],
) -> dict[str, object]:
    cells = {
        task: {
            str(gate.LENGTH): {
                "examples": gate.ROWS_PER_TASK,
                "official_task_score": sum(
                    float(rows[(task, index)]["official_task_score"])
                    for index in range(gate.ROWS_PER_TASK)
                )
                / gate.ROWS_PER_TASK,
            }
        }
        for task in gate.TASKS
    }
    macro = sum(
        float(cells[task][str(gate.LENGTH)]["official_task_score"])
        for task in gate.TASKS
    ) / len(gate.TASKS)
    return {
        "status": gate.TRANSFER_STATUS,
        "checkpoint_sha256": gate.MODEL_SHA256,
        "script_sha256": "1" * 64,
        "bound_code_sha256": {"evaluator": "2" * 64},
        "run_manifest_sha256": "3" * 64,
        "experiment_ready_receipt_sha256": "4" * 64,
        "experiment_role": role,
        "frequency": {
            "active_frequency": "evq_endpoint_cosh",
            "active_sha256_float32": gate.EVQ_SHA256,
        },
        "adapter": {
            "sha256": adapter_sha256,
            "metadata": {
                "base_checkpoint_sha256": gate.MODEL_SHA256,
                "frequency": "evq",
                "frequency_sha256_float32": gate.EVQ_SHA256,
                "adaptation": "qkvo_answer",
                "rank": 64,
                "alpha": 128.0,
                "training_sequence_length": gate.LENGTH,
                "parent_adapter_sha256": parent_sha256,
            },
        },
        "data": {"manifest_sha256": "5" * 64},
        "protocol": {
            "tasks": list(gate.TASKS),
            "lengths": [gate.LENGTH],
            "limit_per_cell": gate.ROWS_PER_TASK,
            "greedy": True,
            "training_length": gate.LENGTH,
        },
        "results": {
            "cells": cells,
            "macro_official_task_score": macro,
        },
    }


def _inputs(
    *,
    failed_candidate_task: str | None = None,
) -> dict[str, object]:
    candidate_rows = _rows(failed_task=failed_candidate_task)
    immediate_rows = _rows()
    pre_rows = _rows()
    candidate_sha = "a" * 64
    candidate_result = _receipt(
        role="candidate",
        adapter_sha256=candidate_sha,
        parent_sha256=gate.IMMEDIATE_PARENT_SHA256,
        rows=candidate_rows,
    )
    immediate_result = _receipt(
        role="immediate_parent",
        adapter_sha256=gate.IMMEDIATE_PARENT_SHA256,
        parent_sha256=gate.PRE_QUERY_GAP_PARENT_SHA256,
        rows=immediate_rows,
    )
    pre_result = _receipt(
        role="pre_query_gap_parent",
        adapter_sha256=gate.PRE_QUERY_GAP_PARENT_SHA256,
        parent_sha256=None,
        rows=pre_rows,
    )
    training = {
        "status": gate.TRAINING_STATUS,
        "adapter_sha256": candidate_sha,
        "parent_adapter_sha256": gate.IMMEDIATE_PARENT_SHA256,
        "natural_nll": {
            "baseline": {
                "adapter_sha256": gate.IMMEDIATE_PARENT_SHA256,
                "cells": {
                    "L4096": {
                        "rows": 16,
                        "tokens": 65_520,
                        "tail_tokens_per_row": 1024,
                        "mean_nll": 2.5,
                        "tail_mean_nll": 2.6,
                    }
                },
            },
            "candidate": {
                "cells": {
                    "L4096": {
                        "rows": 16,
                        "tokens": 65_520,
                        "tail_tokens_per_row": 1024,
                        "mean_nll": 2.52,
                        "tail_mean_nll": 2.63,
                    }
                }
            },
        },
    }
    exact = {
        "status": "PASS",
        "gate": gate.EXACT_GATE_NAME,
        "expanded_evaluation_authorized": True,
    }
    ready = {
        "status": "OLMO2_4K_RETENTION_READY_V1",
        "evaluator": {"sha256": "1" * 64},
        "evaluator_bound_code_sha256": {"evaluator": "2" * 64},
    }
    return {
        "training": training,
        "exact_gate": exact,
        "candidate_result": candidate_result,
        "candidate_rows": candidate_rows,
        "immediate_result": immediate_result,
        "immediate_rows": immediate_rows,
        "pre_result": pre_result,
        "pre_rows": pre_rows,
        "experiment_ready": ready,
        "experiment_ready_sha256": "4" * 64,
        "decode_generated": _decode,
    }


def test_retention_gate_passes_identical_candidate() -> None:
    passed, details = gate.retention_gate(**_inputs())

    assert passed is True
    assert details["overall_4k_retention_pass"] is True


def test_retention_gate_rejects_single_task_collapse() -> None:
    inputs = _inputs(failed_candidate_task="niah_single_3")
    passed, details = gate.retention_gate(**inputs)

    assert passed is False
    assert (
        details["no_additional_forgetting"]["task_checks"][
            "niah_single_3"
        ]["passed"]
        is False
    )
