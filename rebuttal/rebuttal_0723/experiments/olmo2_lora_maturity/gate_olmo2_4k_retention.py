#!/usr/bin/env python3
"""Fail-closed 4K retention gate for the OLMo-2 EOS repair.

This gate is evaluated only after the minimal full-string-exact-plus-EOS gate
passes. It independently compares the repaired adapter with its immediate
query-gap parent and with the pre-query-gap parent. It does not claim global
retention relative to the untouched OLMo-2 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

from transformers import AutoTokenizer

from .prepare_data import atomic_json, sha256_file
from .prepare_instruct_ruler_transfer import TASK_CONFIGS


MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
EVQ_SHA256 = (
    "917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607"
)
IMMEDIATE_PARENT_SHA256 = (
    "a0ccd2cf141300ba4489882dda1324b2f237e65444a9a71d687e8c5fad57ae8b"
)
PRE_QUERY_GAP_PARENT_SHA256 = (
    "95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a"
)
TRAINING_STATUS = "OLMO2_4K_QUERY_GAP_EOS_REPAIR_COMPLETE_V1"
TRANSFER_STATUS = "OLMO2_INSTRUCT_RULER_TRANSFER_COMPLETE"
EXACT_GATE_NAME = "OLMO2_MINIMAL_FULL_STRING_EXACT_EOS_V1"
TASKS = tuple(TASK_CONFIGS)
LENGTH = 4_096
ROWS_PER_TASK = 20
FAMILIES = {
    "niah": tuple(task for task in TASKS if task.startswith("niah_")),
    "operations": ("vt", "cwe", "fwe"),
    "qa": ("qa_1", "qa_2"),
}
TASK_DELTA_FLOOR = -0.10
FAMILY_DELTA_FLOOR = -0.05
GLOBAL_DELTA_FLOOR = -0.05
MAX_TASK_LOST_MASS = 2.0
MAX_EXCESS_LOSS_ROWS = 2
MIN_RETAINED_PARENT_MASS = 0.80
NATURAL_NLL_DELTA_CEILING = 0.10


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def _official_score(
    prediction: str,
    references: list[str],
    metric: str,
) -> float:
    if not references:
        raise RuntimeError("retention row has no references")
    lowered = prediction.lower()
    matches = [
        float(str(reference).lower() in lowered)
        for reference in references
    ]
    if metric == "string_match_all":
        return sum(matches) / len(matches)
    if metric == "string_match_part":
        return max(matches)
    raise RuntimeError(f"unsupported retention metric: {metric}")


def _load_examples(
    result: dict[str, Any],
    path: Path,
    *,
    label: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    if (
        sha256_file(path)
        != result.get("results", {}).get("examples_sha256")
    ):
        raise RuntimeError(f"{label} examples SHA drift")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["task"]), int(row["local_index"]))
        if key in rows:
            raise RuntimeError(f"{label} duplicate row: {key}")
        rows[key] = row
    expected = {
        (task, local_index)
        for task in TASKS
        for local_index in range(ROWS_PER_TASK)
    }
    if set(rows) != expected:
        raise RuntimeError(f"{label} 13-task 4K row set drift")
    return rows


def _validate_result(
    result: dict[str, Any],
    rows: dict[tuple[str, int], dict[str, Any]],
    *,
    adapter_sha256: str,
    label: str,
    decode_generated: Callable[[list[int]], str],
) -> dict[str, list[float]]:
    if (
        result.get("status") != TRANSFER_STATUS
        or result.get("checkpoint_sha256") != MODEL_SHA256
        or result.get("adapter", {}).get("sha256") != adapter_sha256
    ):
        raise RuntimeError(f"{label} result identity drift")
    frequency = result.get("frequency", {})
    if (
        frequency.get("active_frequency") != "evq_endpoint_cosh"
        or frequency.get("active_sha256_float32") != EVQ_SHA256
    ):
        raise RuntimeError(f"{label} did not evaluate frozen EVQ")
    metadata = result["adapter"].get("metadata", {})
    if (
        metadata.get("base_checkpoint_sha256") != MODEL_SHA256
        or metadata.get("frequency") != "evq"
        or metadata.get("frequency_sha256_float32") != EVQ_SHA256
        or metadata.get("adaptation") != "qkvo_answer"
        or int(metadata.get("rank", -1)) != 64
        or float(metadata.get("alpha", -1.0)) != 128.0
        or int(metadata.get("training_sequence_length", -1)) != LENGTH
    ):
        raise RuntimeError(f"{label} adapter metadata drift")
    protocol = result.get("protocol", {})
    if (
        protocol.get("tasks") != list(TASKS)
        or protocol.get("lengths") != [LENGTH]
        or int(protocol.get("limit_per_cell", -1)) != ROWS_PER_TASK
        or protocol.get("greedy") is not True
        or int(protocol.get("training_length", -1)) != LENGTH
    ):
        raise RuntimeError(f"{label} retention protocol drift")

    scores: dict[str, list[float]] = {}
    cells = result.get("results", {}).get("cells", {})
    if set(cells) != set(TASKS):
        raise RuntimeError(f"{label} retention task cells drift")
    for task in TASKS:
        task_scores: list[float] = []
        for local_index in range(ROWS_PER_TASK):
            row = rows[(task, local_index)]
            references = [str(value) for value in row["references"]]
            metric = str(TASK_CONFIGS[task]["official_metric"])
            generated_token_ids = row.get("generated_token_ids")
            if (
                str(row.get("official_metric")) != metric
                or int(row.get("nominal_length", -1)) != LENGTH
                or not isinstance(row.get("prediction"), str)
                or not isinstance(generated_token_ids, list)
                or not all(
                    isinstance(value, int)
                    for value in generated_token_ids
                )
                or int(row.get("generated_tokens", -1))
                != len(generated_token_ids)
                or str(row["prediction"])
                != decode_generated(generated_token_ids)
            ):
                raise RuntimeError(
                    f"{label} row contract drift: {task}/{local_index}"
                )
            score = _official_score(
                str(row["prediction"]),
                references,
                metric,
            )
            if not math.isclose(
                _finite(
                    row.get("official_task_score"),
                    f"{label}.{task}.{local_index}",
                ),
                score,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"{label} stored score drift: {task}/{local_index}"
                )
            task_scores.append(score)
        observed_mean = sum(task_scores) / len(task_scores)
        cell = cells[task].get(str(LENGTH), {})
        if (
            int(cell.get("examples", -1)) != ROWS_PER_TASK
            or not math.isclose(
                _finite(
                    cell.get("official_task_score"),
                    f"{label}.{task}.cell",
                ),
                observed_mean,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(f"{label} task summary drift: {task}")
        scores[task] = task_scores
    observed_macro = sum(
        sum(task_scores) / len(task_scores)
        for task_scores in scores.values()
    ) / len(scores)
    if not math.isclose(
        _finite(
            result.get("results", {}).get(
                "macro_official_task_score"
            ),
            f"{label}.macro",
        ),
        observed_macro,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{label} macro summary drift")
    return scores


def _validate_pairing(
    candidate: dict[tuple[str, int], dict[str, Any]],
    parent: dict[tuple[str, int], dict[str, Any]],
    *,
    label: str,
) -> None:
    for key in candidate:
        left = candidate[key]
        right = parent[key]
        for field in (
            "source_row_index",
            "row_sha256",
            "references",
            "official_metric",
            "input_tokens",
            "maximum_generation_tokens",
        ):
            if left.get(field) != right.get(field):
                raise RuntimeError(
                    f"{label} unmatched row field {field}: {key}"
                )


def _comparison_gate(
    candidate: dict[str, list[float]],
    parent: dict[str, list[float]],
    *,
    label: str,
) -> tuple[bool, dict[str, Any]]:
    task_checks: dict[str, Any] = {}
    passed = True
    for task in TASKS:
        candidate_rows = candidate[task]
        parent_rows = parent[task]
        candidate_mean = sum(candidate_rows) / ROWS_PER_TASK
        parent_mean = sum(parent_rows) / ROWS_PER_TASK
        delta = candidate_mean - parent_mean
        lost_mass = sum(
            max(parent_value - candidate_value, 0.0)
            for candidate_value, parent_value in zip(
                candidate_rows, parent_rows
            )
        )
        loss_rows = sum(
            candidate_value < parent_value
            for candidate_value, parent_value in zip(
                candidate_rows, parent_rows
            )
        )
        win_rows = sum(
            candidate_value > parent_value
            for candidate_value, parent_value in zip(
                candidate_rows, parent_rows
            )
        )
        parent_mass = sum(parent_rows)
        retained_mass = sum(
            min(candidate_value, parent_value)
            for candidate_value, parent_value in zip(
                candidate_rows, parent_rows
            )
        )
        retained_fraction = (
            1.0 if parent_mass == 0.0 else retained_mass / parent_mass
        )
        checks = {
            "mean_delta_at_least_minus_0p10": (
                delta >= TASK_DELTA_FLOOR - 1e-12
            ),
            "lost_mass_at_most_2": (
                lost_mass <= MAX_TASK_LOST_MASS + 1e-12
            ),
            "loss_rows_at_most_wins_plus_2": (
                loss_rows <= win_rows + MAX_EXCESS_LOSS_ROWS
            ),
            "retains_80pct_parent_mass_when_parent_mass_ge_5": (
                parent_mass < 5.0
                or retained_fraction >= MIN_RETAINED_PARENT_MASS - 1e-12
            ),
            "nonzero_when_parent_mean_ge_0p10": (
                parent_mean < 0.10 or candidate_mean > 0.0
            ),
        }
        task_passed = all(checks.values())
        passed = passed and task_passed
        task_checks[task] = {
            "candidate": candidate_mean,
            "parent": parent_mean,
            "delta": delta,
            "lost_mass": lost_mass,
            "loss_rows": loss_rows,
            "win_rows": win_rows,
            "parent_mass": parent_mass,
            "retained_parent_mass_fraction": retained_fraction,
            "checks": checks,
            "passed": task_passed,
        }

    family_checks: dict[str, Any] = {}
    for family, tasks in FAMILIES.items():
        candidate_mean = sum(
            sum(candidate[task]) / ROWS_PER_TASK for task in tasks
        ) / len(tasks)
        parent_mean = sum(
            sum(parent[task]) / ROWS_PER_TASK for task in tasks
        ) / len(tasks)
        delta = candidate_mean - parent_mean
        family_passed = delta >= FAMILY_DELTA_FLOOR - 1e-12
        passed = passed and family_passed
        family_checks[family] = {
            "candidate": candidate_mean,
            "parent": parent_mean,
            "delta": delta,
            "passed": family_passed,
        }
    candidate_macro = sum(
        sum(candidate[task]) / ROWS_PER_TASK for task in TASKS
    ) / len(TASKS)
    parent_macro = sum(
        sum(parent[task]) / ROWS_PER_TASK for task in TASKS
    ) / len(TASKS)
    global_delta = candidate_macro - parent_macro
    global_passed = global_delta >= GLOBAL_DELTA_FLOOR - 1e-12
    passed = passed and global_passed
    return passed, {
        "comparison": label,
        "task_checks": task_checks,
        "family_checks": family_checks,
        "global": {
            "candidate": candidate_macro,
            "parent": parent_macro,
            "delta": global_delta,
            "passed": global_passed,
        },
        "passed": passed,
    }


def retention_gate(
    *,
    training: dict[str, Any],
    exact_gate: dict[str, Any],
    candidate_result: dict[str, Any],
    candidate_rows: dict[tuple[str, int], dict[str, Any]],
    immediate_result: dict[str, Any],
    immediate_rows: dict[tuple[str, int], dict[str, Any]],
    pre_result: dict[str, Any],
    pre_rows: dict[tuple[str, int], dict[str, Any]],
    experiment_ready: dict[str, Any],
    experiment_ready_sha256: str,
    decode_generated: Callable[[list[int]], str],
) -> tuple[bool, dict[str, Any]]:
    if (
        training.get("status") != TRAINING_STATUS
        or exact_gate.get("status") != "PASS"
        or exact_gate.get("gate") != EXACT_GATE_NAME
        or exact_gate.get("expanded_evaluation_authorized") is not True
    ):
        raise RuntimeError("retention evaluation lacks a passing exact gate")
    if experiment_ready.get("status") != "OLMO2_4K_RETENTION_READY_V1":
        raise RuntimeError("retention READY status drift")
    expected_roles = {
        "candidate": candidate_result,
        "immediate_parent": immediate_result,
        "pre_query_gap_parent": pre_result,
    }
    for role, result in expected_roles.items():
        if (
            result.get("experiment_ready_receipt_sha256")
            != experiment_ready_sha256
            or result.get("experiment_role") != role
            or result.get("script_sha256")
            != experiment_ready["evaluator"]["sha256"]
            or result.get("bound_code_sha256")
            != experiment_ready.get("evaluator_bound_code_sha256")
            or not result.get("run_manifest_sha256")
        ):
            raise RuntimeError(f"{role} retention READY binding drift")
    candidate_sha = str(training.get("adapter_sha256"))
    if len(candidate_sha) != 64:
        raise RuntimeError("candidate adapter SHA drift")
    candidate_scores = _validate_result(
        candidate_result,
        candidate_rows,
        adapter_sha256=candidate_sha,
        label="candidate",
        decode_generated=decode_generated,
    )
    immediate_scores = _validate_result(
        immediate_result,
        immediate_rows,
        adapter_sha256=IMMEDIATE_PARENT_SHA256,
        label="immediate_parent",
        decode_generated=decode_generated,
    )
    pre_scores = _validate_result(
        pre_result,
        pre_rows,
        adapter_sha256=PRE_QUERY_GAP_PARENT_SHA256,
        label="pre_query_gap_parent",
        decode_generated=decode_generated,
    )
    if (
        training.get("parent_adapter_sha256")
        != IMMEDIATE_PARENT_SHA256
        or candidate_result["adapter"]["metadata"].get(
            "parent_adapter_sha256"
        )
        != IMMEDIATE_PARENT_SHA256
        or immediate_result["adapter"]["metadata"].get(
            "parent_adapter_sha256"
        )
        != PRE_QUERY_GAP_PARENT_SHA256
    ):
        raise RuntimeError("candidate/parent lineage drift")
    for result in (immediate_result, pre_result):
        if (
            result.get("data") != candidate_result.get("data")
            or result.get("protocol") != candidate_result.get("protocol")
        ):
            raise RuntimeError("retention arms are not protocol matched")
    _validate_pairing(
        candidate_rows,
        immediate_rows,
        label="candidate_vs_immediate_parent",
    )
    _validate_pairing(
        candidate_rows,
        pre_rows,
        label="candidate_vs_pre_query_gap_parent",
    )

    immediate_passed, immediate_details = _comparison_gate(
        candidate_scores,
        immediate_scores,
        label="no_additional_forgetting_vs_a0ccd",
    )
    pipeline_passed, pipeline_details = _comparison_gate(
        candidate_scores,
        pre_scores,
        label="full_pipeline_retention_vs_95ce",
    )
    natural = training.get("natural_nll", {})
    baseline = natural.get("baseline", {}).get("cells", {}).get("L4096", {})
    candidate_nll = (
        natural.get("candidate", {}).get("cells", {}).get("L4096", {})
    )
    if (
        natural.get("baseline", {}).get("adapter_sha256")
        != IMMEDIATE_PARENT_SHA256
        or int(baseline.get("rows", -1))
        != int(candidate_nll.get("rows", -2))
        or int(baseline.get("tokens", -1))
        != int(candidate_nll.get("tokens", -2))
        or int(baseline.get("tail_tokens_per_row", -1))
        != int(candidate_nll.get("tail_tokens_per_row", -2))
    ):
        raise RuntimeError("matched 4K natural-NLL baseline drift")
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
    passed = immediate_passed and pipeline_passed and natural_passed
    return passed, {
        "scope": (
            "fixed 13-task 4K RULER matrix and matched 4K natural NLL; "
            "not global retention relative to untouched OLMo-2"
        ),
        "no_additional_forgetting": immediate_details,
        "full_pipeline_retention": pipeline_details,
        "natural_nll_vs_immediate_parent": {
            "mean_nll_delta": mean_delta,
            "tail_mean_nll_delta": tail_delta,
            "ceiling": NATURAL_NLL_DELTA_CEILING,
            "passed": natural_passed,
        },
        "overall_4k_retention_pass": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--exact-gate", type=Path, required=True)
    parser.add_argument(
        "--experiment-ready-receipt", type=Path, required=True
    )
    for label in ("candidate", "immediate-parent", "pre-parent"):
        parser.add_argument(f"--{label}-result", type=Path, required=True)
        parser.add_argument(f"--{label}-examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    training = _read_json(args.training_result)
    exact_gate = _read_json(args.exact_gate)
    experiment_ready_path = args.experiment_ready_receipt.resolve()
    experiment_ready_sha256 = sha256_file(experiment_ready_path)
    experiment_ready = _read_json(experiment_ready_path)
    if (
        experiment_ready.get("gate", {}).get("sha256")
        != sha256_file(Path(__file__).resolve())
    ):
        raise RuntimeError("retention gate code drift")
    checkpoint = Path(
        experiment_ready["inputs"]["checkpoint"]["path"]
    ).resolve()
    if (
        experiment_ready["inputs"]["checkpoint"]["composite_sha256"]
        != MODEL_SHA256
    ):
        raise RuntimeError("retention READY checkpoint drift")
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
    immediate_result = _read_json(args.immediate_parent_result)
    pre_result = _read_json(args.pre_parent_result)
    candidate_rows = _load_examples(
        candidate_result,
        args.candidate_examples.resolve(),
        label="candidate",
    )
    immediate_rows = _load_examples(
        immediate_result,
        args.immediate_parent_examples.resolve(),
        label="immediate_parent",
    )
    pre_rows = _load_examples(
        pre_result,
        args.pre_parent_examples.resolve(),
        label="pre_query_gap_parent",
    )
    passed, details = retention_gate(
        training=training,
        exact_gate=exact_gate,
        candidate_result=candidate_result,
        candidate_rows=candidate_rows,
        immediate_result=immediate_result,
        immediate_rows=immediate_rows,
        pre_result=pre_result,
        pre_rows=pre_rows,
        experiment_ready=experiment_ready,
        experiment_ready_sha256=experiment_ready_sha256,
        decode_generated=decode_generated,
    )
    receipt = {
        "status": "PASS" if passed else "STOP",
        "gate": "OLMO2_4K_RETENTION_V1",
        **details,
    }
    atomic_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(21)


if __name__ == "__main__":
    main()
