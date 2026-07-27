#!/usr/bin/env python3
"""Recompute the final OLMo-2 query-gap capability diagnostic from raw rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


EOS_TOKEN_ID = 100_257
NUMBER_PATTERN = re.compile(r"\b[0-9]+\b")
MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
PARENT_ADAPTER_SHA256 = (
    "95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a"
)
QUERY_GAP_ADAPTER_SHA256 = (
    "a0ccd2cf141300ba4489882dda1324b2f237e65444a9a71d687e8c5fad57ae8b"
)
DATA_MANIFEST_SHA256 = (
    "274f5df85420aabde38191a3bc7da7d466eb13bff69f0fde5ad618aea3f07511"
)
OLD_EVALUATION_SHA256 = (
    "6c87b7eb5b07880c33225ef05bf4d1e76830297c2e2db145e3ec113b7e9c09b2"
)
QUERY_TRAINING_RESULT_SHA256 = (
    "ac66dca2aaa9e35cbf10c117eabe1ce0e293249e1f80bd0e05b0be54f9548e3e"
)
PARENT_TRAINING_RESULT_SHA256 = (
    "78843a65df9c5d40883f48b78941e376fd332e4c10500d9b67d42156975cfbde"
)
QUERY_RAW_SHA256 = (
    "09e579558b632326ce86a7acc027c431603a69debc34ded9628eafc031c11774"
)
QUERY_RUN_MANIFEST_SHA256 = (
    "7b703b3b27b7ab84139d6f99917ac32922b7391eebed77945a6e5202ea48c20e"
)
PARENT_RAW_SHA256 = (
    "e4be3047c0865aa99c062c502c1bd7d5c340e81de4ca8ba79a95b2c4eb0c9a45"
)
PARENT_RESULT_SHA256 = (
    "e5bc2ddf40e739be497049b169f07f4f7408a481e2a39fad29662206826bede6"
)
PARENT_RUN_MANIFEST_SHA256 = (
    "977acb19cfe916c867f5b73bcf5bd8a2da1e408ebd43d205f4a23f3061f6c37d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-raw", type=Path, required=True)
    parser.add_argument("--query-run-manifest", type=Path, required=True)
    parser.add_argument("--parent-raw", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--parent-run-manifest", type=Path, required=True)
    parser.add_argument("--query-training-result", type=Path, required=True)
    parser.add_argument("--parent-training-result", type=Path, required=True)
    parser.add_argument("--old-evaluation-result", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--lineage-output", type=Path, required=True)
    parser.add_argument("--classification-output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path.name}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"JSONL object rows required: {path.name}")
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def official_string_match(prediction: str, references: list[str]) -> float:
    normalized = "".join(
        char if char.isprintable() else "\n" for char in prediction
    ).strip().lower()
    return sum(
        reference.lower() in normalized for reference in references
    ) / len(references)


def first_number_exact(prediction: str, references: list[str]) -> float:
    match = NUMBER_PATTERN.search(prediction)
    if match is None:
        return 0.0
    return float(
        match.group(0) in {str(value).strip() for value in references}
    )


def analyze_row(row: dict[str, Any]) -> dict[str, Any]:
    if (
        int(row.get("schema_version", -1)) != 3
        or row.get("task") != "niah_single_1"
        or not isinstance(row.get("prediction"), str)
        or not isinstance(row.get("references"), list)
        or len(row["references"]) != 1
        or not isinstance(row.get("generated_token_ids"), list)
        or not isinstance(row.get("reference_token_ids"), list)
        or len(row["reference_token_ids"]) != 1
    ):
        raise RuntimeError("raw generation schema drift")
    prediction = str(row["prediction"])
    references = [str(value) for value in row["references"]]
    generated_ids = [int(value) for value in row["generated_token_ids"]]
    reference_ids = [
        [int(value) for value in values]
        for values in row["reference_token_ids"]
    ]
    if (
        len(generated_ids) != int(row.get("generated_tokens", -1))
        or len(generated_ids) > 128
        or EOS_TOKEN_ID in generated_ids[:-1]
    ):
        raise RuntimeError("generated-token payload drift")
    eos_terminated = bool(
        generated_ids and generated_ids[-1] == EOS_TOKEN_ID
    )
    content_ids = (
        generated_ids[:-1] if eos_terminated else generated_ids
    )
    full_string_exact = prediction in references
    answer_eos_token_exact = any(
        generated_ids == [*tokens, EOS_TOKEN_ID]
        for tokens in reference_ids
    )
    exact_generation_pass = full_string_exact and eos_terminated
    official = official_string_match(prediction, references)
    first_number = first_number_exact(prediction, references)
    for name, recomputed in (
        ("full_string_exact", full_string_exact),
        ("eos_terminated", eos_terminated),
        ("answer_eos_token_exact", answer_eos_token_exact),
        ("exact_generation_pass", exact_generation_pass),
        ("official_string_match", official),
        ("first_number_exact", first_number),
    ):
        observed = float(row.get(name, math.nan))
        if not math.isfinite(observed) or observed != float(recomputed):
            raise RuntimeError(
                f"{name} does not recompute for row "
                f"{row.get('nominal_length')}/{row.get('local_index')}"
            )

    answer_continues = first_number == 1.0 and any(
        prediction.startswith(reference) and prediction != reference
        for reference in references
    )
    incomplete_token_sequence = any(
        0 < len(content_ids) < len(tokens)
        and content_ids == tokens[: len(content_ids)]
        for tokens in reference_ids
    )
    if exact_generation_pass:
        category = "pass"
    elif full_string_exact:
        category = "eos_readout_failure"
    elif incomplete_token_sequence:
        category = "incomplete_answer_token_sequence"
    elif first_number == 1.0 and answer_continues:
        category = "correct_answer_then_continued"
    elif official == 1.0 or first_number == 1.0:
        category = "format_error"
    else:
        category = "correct_answer_not_retrieved"

    return {
        "nominal_length": int(row["nominal_length"]),
        "local_index": int(row["local_index"]),
        "source_row_index": int(row["source_row_index"]),
        "row_sha256": str(row["row_sha256"]),
        "reference": references[0],
        "prediction": prediction,
        "generated_token_ids": generated_ids,
        "reference_token_ids": reference_ids[0],
        "generated_tokens": len(generated_ids),
        "first_number_exact": int(first_number),
        "official_string_match": int(official),
        "full_string_exact": int(full_string_exact),
        "terminal_eos": int(eos_terminated),
        "answer_eos_token_exact": int(answer_eos_token_exact),
        "exact_generation_pass": int(exact_generation_pass),
        "answer_continues": int(answer_continues),
        "failure_category": category,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for length in sorted({int(row["nominal_length"]) for row in rows}):
        selected = [
            row for row in rows if int(row["nominal_length"]) == length
        ]
        examples = len(selected)
        counts = {
            name: sum(int(row[name]) for row in selected)
            for name in (
                "first_number_exact",
                "official_string_match",
                "full_string_exact",
                "terminal_eos",
                "answer_eos_token_exact",
                "exact_generation_pass",
                "answer_continues",
            )
        }
        output[str(length)] = {
            "examples": examples,
            "counts": counts,
            "rates": {
                name: value / examples
                for name, value in counts.items()
            },
            "failure_categories": dict(
                sorted(
                    Counter(
                        str(row["failure_category"])
                        for row in selected
                    ).items()
                )
            ),
            "generated_token_count_distribution": {
                str(tokens): count
                for tokens, count in sorted(
                    Counter(
                        int(row["generated_tokens"])
                        for row in selected
                    ).items()
                )
            },
        }
    return output


def validate_manifest(
    manifest: dict[str, Any],
    *,
    adapter_sha256: str,
    lengths: list[int],
) -> None:
    expected = {
        "status": "OLMO2_INSTRUCT_RULER_EXACT_SCREEN_RUN_V3",
        "example_schema_version": 3,
        "checkpoint_sha256": MODEL_SHA256,
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "adapter_sha256": adapter_sha256,
        "frequency": "evq",
        "adaptation": "qkvo_answer",
        "rank": 64,
        "alpha": 128.0,
        "task": "niah_single_1",
        "lengths": lengths,
        "limit_per_length": 100,
        "greedy": True,
        "maximum_new_tokens": 128,
        "string_normalization": "none",
        "decode_cleanup": False,
        "terminal_eos_removed_before_string_decode": True,
        "other_special_tokens_removed": False,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise RuntimeError(f"run-manifest drift: {name}")


def natural_metrics(result: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for length in (4_096, 8_192, 16_384):
        cell = result["natural_nll"][f"L{length}"]
        output[str(length)] = {
            "mean_nll": float(cell["mean_nll"]),
            "perplexity": float(cell["perplexity"]),
            "rows": int(cell["rows"]),
            "tokens": int(cell["tokens"]),
        }
    return output


def main() -> None:
    args = parse_args()
    expected_hashes = {
        args.query_raw: QUERY_RAW_SHA256,
        args.query_run_manifest: QUERY_RUN_MANIFEST_SHA256,
        args.parent_raw: PARENT_RAW_SHA256,
        args.parent_result: PARENT_RESULT_SHA256,
        args.parent_run_manifest: PARENT_RUN_MANIFEST_SHA256,
        args.query_training_result: QUERY_TRAINING_RESULT_SHA256,
        args.parent_training_result: PARENT_TRAINING_RESULT_SHA256,
        args.old_evaluation_result: OLD_EVALUATION_SHA256,
    }
    for path, expected in expected_hashes.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"input hash drift for {path.name}: {observed}"
            )

    query_manifest = load_json(args.query_run_manifest)
    parent_manifest = load_json(args.parent_run_manifest)
    validate_manifest(
        query_manifest,
        adapter_sha256=QUERY_GAP_ADAPTER_SHA256,
        lengths=[4_096, 8_192, 16_384],
    )
    validate_manifest(
        parent_manifest,
        adapter_sha256=PARENT_ADAPTER_SHA256,
        lengths=[4_096],
    )
    query_rows = [
        analyze_row(row) for row in load_jsonl(args.query_raw)
    ]
    parent_rows = [
        analyze_row(row) for row in load_jsonl(args.parent_raw)
    ]
    if (
        len(query_rows) != 300
        or Counter(row["nominal_length"] for row in query_rows)
        != Counter({4_096: 100, 8_192: 100, 16_384: 100})
        or len(parent_rows) != 100
        or {row["nominal_length"] for row in parent_rows} != {4_096}
    ):
        raise RuntimeError("final diagnostic row-set drift")

    query_summary = summarize(query_rows)
    parent_summary = summarize(parent_rows)
    old_evaluation = load_json(args.old_evaluation_result)
    old_cells = old_evaluation["results"]["cells"]
    for length, expected in ((8_192, 95), (16_384, 51)):
        recomputed = query_summary[str(length)]["counts"][
            "first_number_exact"
        ]
        old_count = round(
            float(old_cells[str(length)]["first_number_exact"]) * 100
        )
        if recomputed != expected or old_count != expected:
            raise RuntimeError(f"old L{length} result did not reproduce")

    query_training = load_json(args.query_training_result)
    parent_training = load_json(args.parent_training_result)
    if (
        query_training.get("adapter_sha256")
        != QUERY_GAP_ADAPTER_SHA256
        or query_training.get("parent_adapter_sha256")
        != PARENT_ADAPTER_SHA256
        or parent_training.get("adapter_sha256")
        != PARENT_ADAPTER_SHA256
    ):
        raise RuntimeError("training-result adapter lineage drift")

    query_4k = query_summary["4096"]["counts"]
    parent_4k = parent_summary["4096"]["counts"]
    query_nll = natural_metrics(query_training)
    parent_nll = natural_metrics(parent_training)
    metrics = {
        "status": "EVQ_QUERY_GAP_FINAL_DIAGNOSTIC_COMPLETE",
        "primary_decision": {
            "selection": 3,
            "label": (
                "stop_new_experiments_and_retain_first_number_"
                "length_transfer_result"
            ),
            "minimal_readout_repair_authorized": False,
            "reason": (
                "No tested length has any full-string-plus-terminal-EOS "
                "success, and the 16K cell contains 43/100 retrieval "
                "failures plus 6/100 format failures. The registered "
                "content-success prerequisite for repair is not met."
            ),
        },
        "query_gap_adapter": {
            "by_length": query_summary,
            "natural_text_nll_ppl": query_nll,
        },
        "pre_query_gap_parent_4k": {
            "by_length": parent_summary,
            "natural_text_nll_ppl": parent_nll,
        },
        "four_k_retention": {
            "scope": (
                "same-family niah_single_1 plus the frozen natural-text "
                "NLL set; not broad instruction or downstream retention"
            ),
            "parent_to_query_gap_counts": {
                name: {
                    "parent": parent_4k[name],
                    "query_gap": query_4k[name],
                    "delta": query_4k[name] - parent_4k[name],
                }
                for name in (
                    "first_number_exact",
                    "official_string_match",
                    "full_string_exact",
                    "terminal_eos",
                    "exact_generation_pass",
                )
            },
            "natural_text": {
                "parent_nll": parent_nll["4096"]["mean_nll"],
                "query_gap_nll": query_nll["4096"]["mean_nll"],
                "nll_delta": (
                    query_nll["4096"]["mean_nll"]
                    - parent_nll["4096"]["mean_nll"]
                ),
                "parent_ppl": parent_nll["4096"]["perplexity"],
                "query_gap_ppl": query_nll["4096"]["perplexity"],
                "ppl_delta": (
                    query_nll["4096"]["perplexity"]
                    - parent_nll["4096"]["perplexity"]
                ),
            },
        },
        "old_result_reproduction": {
            "same_adapter": True,
            "same_data_manifest": True,
            "same_greedy_max_new_tokens_128_protocol": True,
            "first_number_exact_counts": {
                "8192": 95,
                "16384": 51,
            },
            "official_string_match_counts": {
                "8192": 98,
                "16384": 57,
            },
        },
        "failure_category_definitions": {
            "correct_answer_not_retrieved": (
                "neither first-number exact nor official reference substring"
            ),
            "incomplete_answer_token_sequence": (
                "generated content is a strict prefix of the canonical "
                "reference token sequence"
            ),
            "format_error": (
                "the reference appears under the official substring metric "
                "or first-number extraction, but not as a clean complete "
                "answer followed by extra output"
            ),
            "correct_answer_then_continued": (
                "the clean first generated number is correct and the decoded "
                "continuation contains additional content"
            ),
            "eos_readout_failure": (
                "the complete decoded string equals the reference but "
                "terminal EOS is absent"
            ),
        },
    }

    classification_rows = [
        {"arm": "query_gap_plus_100", **row}
        for row in query_rows
    ] + [
        {"arm": "pre_query_gap_parent_4k", **row}
        for row in parent_rows
    ]
    args.classification_output.parent.mkdir(
        parents=True, exist_ok=True
    )
    args.classification_output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in classification_rows
        ),
        encoding="utf-8",
    )
    atomic_json(args.metrics_output, metrics)

    lineage = {
        "status": "EVQ_QUERY_GAP_FINAL_LINEAGE_VERIFIED",
        "checkpoint": {
            "model": "OLMo-2-0425-1B-Instruct",
            "actual_parameters": "1.485B",
            "composite_sha256": MODEL_SHA256,
        },
        "adapters": {
            "pre_query_gap_parent": {
                "sha256": PARENT_ADAPTER_SHA256,
                "training_result_sha256": (
                    PARENT_TRAINING_RESULT_SHA256
                ),
            },
            "query_gap_plus_100": {
                "sha256": QUERY_GAP_ADAPTER_SHA256,
                "parent_sha256": PARENT_ADAPTER_SHA256,
                "training_result_sha256": (
                    QUERY_TRAINING_RESULT_SHA256
                ),
                "frequency": "evq_endpoint_cosh",
                "frequency_sha256_float32": (
                    query_training["frequency"][
                        "active_sha256_float32"
                    ]
                ),
                "adaptation": "qkvo_answer",
                "rank": 64,
                "alpha": 128.0,
            },
        },
        "training_position_contract": {
            "physical_sequence_length_maximum": 4_096,
            "physical_input_tokens": int(
                query_training["training"]["processed_input_tokens"]
            ),
            "position_policy": query_training["protocol"][
                "position_policy"
            ],
            "virtual_target_length": int(
                query_training["protocol"]["virtual_target_length"]
            ),
            "hard_maximum_training_position_id": int(
                query_training["protocol"][
                    "hard_maximum_training_position_id"
                ]
            ),
            "query_offsets": (
                "context/source remain contiguous; final semantic query "
                "block, answer prefix, and teacher-forced answer shift "
                "together under a deterministic 0..12289 offset stream"
            ),
            "query_offset_stream_sha256": query_training["training"][
                "query_offset_stream_sha256"
            ],
            "realized_position_stream_sha256": query_training["training"][
                "realized_position_stream_sha256"
            ],
            "realized_exposure_stream_sha256": query_training["training"][
                "realized_exposure_stream_sha256"
            ],
        },
        "evaluation_data": {
            "task": "niah_single_1",
            "manifest_sha256": DATA_MANIFEST_SHA256,
            "files": query_manifest["data_files"],
            "rows_per_length": 100,
            "lengths": [4_096, 8_192, 16_384],
            "row_identity": (
                "the 8K/16K row set is identical to the old 95/51 receipt"
            ),
        },
        "evaluation_protocol": {
            "physical_position_ids": (
                "contiguous 0..input_tokens-1; no virtual query gap at "
                "evaluation"
            ),
            "greedy": True,
            "maximum_new_tokens": 128,
            "raw_generated_token_ids_saved": True,
            "complete_decoded_continuation_saved": True,
            "decode_cleanup": False,
            "terminal_eos_removed_only_for_visible_string_decode": True,
            "other_special_tokens_removed": False,
            "query_generation_code_sha256": query_manifest[
                "bound_code_sha256"
            ]["greedy_generation_and_row_identity"],
            "query_evaluator_sha256": query_manifest[
                "bound_code_sha256"
            ]["evaluator"],
            "parent_evaluator_sha256": parent_manifest[
                "bound_code_sha256"
            ]["evaluator"],
            "analysis_script_sha256": sha256_file(
                Path(__file__).resolve()
            ),
        },
        "retained_artifacts": {
            "old_8k16k_result_sha256": OLD_EVALUATION_SHA256,
            "query_raw_generations_sha256": QUERY_RAW_SHA256,
            "query_run_manifest_sha256": (
                QUERY_RUN_MANIFEST_SHA256
            ),
            "parent_4k_raw_generations_sha256": PARENT_RAW_SHA256,
            "parent_4k_result_sha256": PARENT_RESULT_SHA256,
            "parent_4k_run_manifest_sha256": (
                PARENT_RUN_MANIFEST_SHA256
            ),
            "classification_sha256": sha256_file(
                args.classification_output
            ),
            "metrics_sha256": sha256_file(args.metrics_output),
        },
        "runtime": load_json(args.parent_result)["runtime"],
    }
    atomic_json(args.lineage_output, lineage)
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "decision": metrics["primary_decision"],
                "query_gap_by_length": query_summary,
                "outputs": {
                    "metrics": str(args.metrics_output),
                    "lineage": str(args.lineage_output),
                    "classification": str(
                        args.classification_output
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
