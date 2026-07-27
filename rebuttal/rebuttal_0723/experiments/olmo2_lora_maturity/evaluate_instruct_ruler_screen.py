#!/usr/bin/env python3
"""Evaluate native or EVQ-adapted OLMo-2 Instruct on a RULER screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.models.olmo2 import modeling_olmo2

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    configure_cuda,
    configure_ruler_flash_attention,
    greedy_generate,
    row_sha256,
    score_prediction,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
    load_model,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_data import (
    atomic_json,
    sha256_file,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_stage_a import (
    ready_checkpoint_digest,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_screen import (
    apply_frequency,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.olmo2_exact_method import (
    HYBRID_FREQUENCY_NAME,
    qk_evq_tail_output_mask,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.olmo2_length_gated_method import (
    EVQ_FREQUENCY_SHA256,
    LENGTH_GATED_FREQUENCY_NAME,
    NATIVE_FREQUENCY_SHA256,
    SHORT_CONTEXT_LIMIT,
    eos_head_trainable_named_parameters,
    freeze_length_gated_qkvo_adapter,
    install_length_gated_eos_vocab_row_head,
    install_length_gated_qkvo,
)


SUPPORTED_DATA_STATUSES = (
    "OLMO2_INSTRUCT_RULER_SCREEN_PREPARED",
    "OLMO2_INSTRUCT_RULER_LONG_GAP_SCREEN_PREPARED",
)
SUPPORTED_TASKS = ("niah_single_1", "niah_single_2")
GENERATION_TOKENS = 128
TRAINING_GAP_MAXIMUM_TOKENS = 3_933
NUMBER_PATTERN = re.compile(r"\b[0-9]+\b")
EXAMPLE_SCHEMA_VERSION = 3
RUN_MANIFEST_STATUS = "OLMO2_INSTRUCT_RULER_EXACT_SCREEN_RUN_V3"
EOS_REPAIR_READY_STATUS = "OLMO2_4K_QUERY_GAP_EOS_REPAIR_READY_V1"
LENGTH_GATED_READY_STATUS = (
    "OLMO2_4K_LENGTH_GATED_EOS_REPAIR_READY_V1"
)
LENGTH_GATED_EOS_ADAPTATION = "length_gated_eos_vocab_row"
LENGTH_GATED_PARENT_ADAPTATION = "qkvo_answer"
LENGTH_GATED_PARENT_RANK = 64
LENGTH_GATED_PARENT_ALPHA = 128.0
LENGTH_GATED_EOS_HEAD_RANK = 1
LENGTH_GATED_EOS_HEAD_ALPHA = 1.0
LENGTH_GATED_PARENT_ADAPTER_SHA256 = (
    "a0ccd2cf141300ba4489882dda1324b2f237e65444a9a71d687e8c5fad57ae8b"
)
LENGTH_GATED_EOS_TRAINABLE_SCOPE = (
    "long_only_eos_vocab_row_plus_scalar_bias"
)
OLMO2_EOS_TOKEN_ID = 100_257
FREQUENCIES = (
    "native",
    "evq",
    LENGTH_GATED_FREQUENCY_NAME,
    "hybrid_native_low8",
    "hybrid_native_low16",
    "hybrid_native_low32",
    "hybrid_native_high16",
    "hybrid_native_ends16",
    "hybrid_evq_low4",
    "hybrid_evq_low8",
    "hybrid_evq_low12",
    "hybrid_evq_low16",
    "hybrid_evq_low24",
    "hybrid_evq_low32",
    "hybrid_evq_low40",
    "hybrid_evq_high8",
    "hybrid_evq_mid8",
    "hybrid_blend10",
    "hybrid_blend25",
    "hybrid_blend_evq_0p1pct",
    "hybrid_blend_evq_0p5pct",
    "hybrid_blend_evq_1pct",
    "hybrid_blend_evq_2pct",
    "hybrid_blend_evq_5pct",
    "hybrid_heads_evq1",
    "hybrid_heads_evq2",
    "hybrid_heads_evq4",
    "hybrid_heads_evq8",
    "hybrid_heads_custom",
)

LOG_FREQUENCY_BLEND_WEIGHTS = {
    "hybrid_blend10": 0.10,
    "hybrid_blend25": 0.25,
    "hybrid_blend_evq_0p1pct": 0.001,
    "hybrid_blend_evq_0p5pct": 0.005,
    "hybrid_blend_evq_1pct": 0.01,
    "hybrid_blend_evq_2pct": 0.02,
    "hybrid_blend_evq_5pct": 0.05,
}


def bound_code_sha256() -> dict[str, str]:
    evaluator = Path(__file__).resolve()
    maturity_root = evaluator.parent
    experiments_root = maturity_root.parent
    paths = {
        "evaluator": evaluator,
        "greedy_generation_and_row_identity": (
            experiments_root / "olmo2_1b_evq" / "evaluate_ruler.py"
        ),
        "evq_contract": experiments_root / "olmo2_1b_evq" / "contract.py",
        "lora_conversion": experiments_root / "olmo2_lora_conversion.py",
        "lora_primitives": (
            experiments_root / "small_model_lora_conversion.py"
        ),
        "model_loader_attention_dependency": (
            experiments_root / "olmo2_1b_evq" / "train.py"
        ),
        "adapter_loader": (
            experiments_root / "olmo2_lora_ood_factorial.py"
        ),
        "checkpoint_contract": maturity_root / "train_4k_stage_a.py",
        "frequency_application": maturity_root / "train_screen.py",
        "hybrid_import_dependency": maturity_root / "olmo2_exact_method.py",
        "length_gated_import_dependency": (
            maturity_root / "olmo2_length_gated_method.py"
        ),
    }
    return {
        name: sha256_file(path)
        for name, path in sorted(paths.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--experiment-ready-receipt", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frequency", choices=FREQUENCIES, required=True
    )
    parser.add_argument(
        "--task",
        choices=SUPPORTED_TASKS,
        default="niah_single_1",
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--parent-adapter", type=Path)
    parser.add_argument(
        "--adaptation",
        choices=(
            "qkvo_answer",
            "qk_answer",
            "length_gated_qkvo_answer",
            LENGTH_GATED_EOS_ADAPTATION,
        ),
        default="qkvo_answer",
    )
    parser.add_argument("--evq-head-indices", type=int, nargs="*")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument(
        "--parent-rank",
        type=int,
        default=LENGTH_GATED_PARENT_RANK,
    )
    parser.add_argument(
        "--parent-alpha",
        type=float,
        default=LENGTH_GATED_PARENT_ALPHA,
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[4_096, 8_192, 16_384],
    )
    parser.add_argument("--limit-per-length", type=int, default=20)
    return parser.parse_args()


def validate_data(
    root: Path,
    checkpoint: Path,
    task: str,
    lengths: tuple[int, ...],
    limit_per_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in SUPPORTED_DATA_STATUSES:
        raise RuntimeError("RULER screen data is not prepared")
    if manifest.get("task") != task:
        raise RuntimeError("RULER screen task drift")
    if Path(manifest["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError("RULER screen tokenizer checkpoint drift")
    tokenizer_digest = sha256_file(checkpoint / "tokenizer.json")
    if manifest.get("tokenizer_sha256") != tokenizer_digest:
        raise RuntimeError("RULER screen tokenizer hash drift")
    manifest_lengths = tuple(
        int(value) for value in manifest["lengths"]
    )
    if (
        not manifest_lengths
        or tuple(sorted(set(manifest_lengths))) != manifest_lengths
        or any(
            length not in {4_096, 8_192, 16_384}
            for length in manifest_lengths
        )
    ):
        raise RuntimeError("RULER screen manifest length drift")
    if any(length not in set(manifest_lengths) for length in lengths):
        raise RuntimeError("requested length is absent from data manifest")
    if not 1 <= limit_per_length <= int(manifest["samples_per_length"]):
        raise RuntimeError("limit-per-length exceeds prepared rows")

    selected: list[dict[str, Any]] = []
    file_receipts: dict[str, Any] = {}
    for length in lengths:
        entry = manifest["files"][str(length)]
        path = root / entry["relative_path"]
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise RuntimeError(f"RULER screen hash drift at L={length}")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != int(entry["rows"]):
            raise RuntimeError(f"RULER screen row-count drift at L={length}")
        for local_index, row in enumerate(rows[:limit_per_length]):
            row["_nominal_length"] = int(length)
            row["_local_index"] = int(local_index)
            selected.append(row)
        file_receipts[str(length)] = {
            "path": str(path),
            "sha256": digest,
            "rows": len(rows),
        }
    return (
        {
            "preparation_status": manifest["status"],
            "manifest_sha256": sha256_file(manifest_path),
            "ruler_commit": manifest["ruler_commit"],
            "tokenizer_sha256": tokenizer_digest,
            "files": file_receipts,
        },
        selected,
    )


def validate_actual_prompt_geometry(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Require each nominal long row to physically occupy that length band."""

    cells: dict[str, dict[str, Any]] = {}
    geometry_rows: list[dict[str, Any]] = []
    for row in rows:
        nominal_length = int(row["_nominal_length"])
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError(
                "physical NIAH geometry requires one target value"
            )
        target = str(outputs[0])
        source_text = str(row["input"])
        source_character_index = source_text.find(target)
        if (
            not target
            or source_character_index < 0
            or source_character_index != source_text.rfind(target)
        ):
            raise RuntimeError(
                "NIAH target value is not unique in the source prompt"
            )
        recomputed_source_position = len(
            tokenizer(
                source_text[:source_character_index],
                add_special_tokens=False,
            ).input_ids
        )
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": source_text}],
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer(
            row.get("answer_prefix", ""),
            add_special_tokens=False,
        ).input_ids
        prompt_tokens = len(chat_ids) + len(prefix_ids)
        minimum_actual_maximum_position_id = nominal_length // 2
        minimum_prompt_tokens = (
            minimum_actual_maximum_position_id + 1
        )
        source_position = int(row["token_position_answer"])
        if source_position != recomputed_source_position:
            raise RuntimeError(
                "NIAH source token position does not match its unique "
                "target-value span"
            )
        generation_boundary_gap = prompt_tokens - source_position
        if (
            prompt_tokens < minimum_prompt_tokens
            or prompt_tokens + GENERATION_TOKENS > nominal_length
            or source_position < 0
            or source_position >= prompt_tokens
            or generation_boundary_gap <= 0
        ):
            raise RuntimeError(
                f"nominal L={nominal_length} row is not a physical "
                "near-boundary long-context example"
            )
        cell = cells.setdefault(
            str(nominal_length),
            {
                "rows": 0,
                "required_minimum_prompt_tokens": minimum_prompt_tokens,
                "required_minimum_actual_maximum_position_id": (
                    minimum_actual_maximum_position_id
                ),
                "minimum_prompt_tokens": prompt_tokens,
                "maximum_prompt_tokens": prompt_tokens,
                "minimum_actual_maximum_position_id": prompt_tokens - 1,
                "maximum_actual_maximum_position_id": prompt_tokens - 1,
                "minimum_generation_boundary_gap_tokens": (
                    generation_boundary_gap
                ),
                "maximum_generation_boundary_gap_tokens": (
                    generation_boundary_gap
                ),
                "far_gap_rows": 0,
            },
        )
        if (
            int(cell["required_minimum_prompt_tokens"])
            != minimum_prompt_tokens
            or int(
                cell[
                    "required_minimum_actual_maximum_position_id"
                ]
            )
            != minimum_actual_maximum_position_id
        ):
            raise RuntimeError("prompt-geometry length cell drift")
        cell["rows"] = int(cell["rows"]) + 1
        cell["minimum_prompt_tokens"] = min(
            int(cell["minimum_prompt_tokens"]), prompt_tokens
        )
        cell["maximum_prompt_tokens"] = max(
            int(cell["maximum_prompt_tokens"]), prompt_tokens
        )
        cell["minimum_actual_maximum_position_id"] = min(
            int(cell["minimum_actual_maximum_position_id"]),
            prompt_tokens - 1,
        )
        cell["maximum_actual_maximum_position_id"] = max(
            int(cell["maximum_actual_maximum_position_id"]),
            prompt_tokens - 1,
        )
        cell["minimum_generation_boundary_gap_tokens"] = min(
            int(cell["minimum_generation_boundary_gap_tokens"]),
            generation_boundary_gap,
        )
        cell["maximum_generation_boundary_gap_tokens"] = max(
            int(cell["maximum_generation_boundary_gap_tokens"]),
            generation_boundary_gap,
        )
        if generation_boundary_gap > TRAINING_GAP_MAXIMUM_TOKENS:
            cell["far_gap_rows"] = int(cell["far_gap_rows"]) + 1
        geometry_rows.append(
            {
                "nominal_length": nominal_length,
                "local_index": int(row["_local_index"]),
                "source_index_field": int(row["index"]),
                "row_sha256": row_sha256(
                    {
                        name: value
                        for name, value in row.items()
                        if not name.startswith("_")
                    }
                ),
                "prompt_tokens": prompt_tokens,
                "actual_maximum_prompt_position_id": prompt_tokens - 1,
                "source_token_position_answer": source_position,
                "generation_boundary_gap_tokens": (
                    generation_boundary_gap
                ),
                "far_gap_beyond_training_support": (
                    generation_boundary_gap
                    > TRAINING_GAP_MAXIMUM_TOKENS
                ),
            }
        )
    if not cells:
        raise RuntimeError("prompt-geometry validation received no rows")
    if any(
        int(length) > 4_096 and int(cell["far_gap_rows"]) < 1
        for length, cell in cells.items()
    ):
        raise RuntimeError(
            "each extrapolation cell requires at least one "
            "source-to-generation gap beyond the 3933-token training support"
        )
    geometry_payload = json.dumps(
        geometry_rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "status": "PASS",
        "policy": (
            "actual_maximum_prompt_position_id reaches the nominal "
            "dyadic band and actual_prompt_tokens + 128 <= nominal_length"
        ),
        "minimum_far_gap_rows_per_cell": 1,
        "far_gap_threshold_tokens": TRAINING_GAP_MAXIMUM_TOKENS,
        "generation_tokens_reserved": GENERATION_TOKENS,
        "cells": cells,
        "rows": geometry_rows,
        "rows_sha256": hashlib.sha256(geometry_payload).hexdigest(),
    }


def first_number_exact(prediction: str, references: list[str]) -> float:
    match = NUMBER_PATTERN.search(prediction)
    if match is None:
        return 0.0
    return float(match.group(0) in {str(value).strip() for value in references})


def decode_generated_string(
    *,
    tokenizer: Any,
    generated_token_ids: list[int],
    eos_token_id: int | None,
) -> str:
    """Decode all generated content except one required terminal EOS.

    Other special tokens are deliberately not hidden. This preserves literal
    whole-string scoring while treating EOS as the generation terminator
    rather than visible answer text.
    """

    content_ids = list(generated_token_ids)
    if (
        content_ids
        and eos_token_id is not None
        and int(content_ids[-1]) == int(eos_token_id)
    ):
        content_ids = content_ids[:-1]
    return str(
        tokenizer.decode(
            content_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def exact_generation_metrics(
    *,
    prediction: str,
    references: list[str],
    generated_token_ids: list[int],
    reference_token_ids: list[list[int]],
    eos_token_id: int | None,
) -> dict[str, float]:
    """Score the entire decoded continuation and its terminal EOS.

    No whitespace, case, punctuation, or substring normalization is applied.
    The primary pass requires exact decoded-string equality and an observed
    terminal EOS. Canonical reference-token equality is reported separately
    because different token sequences can decode to the same literal string.
    """

    if len(references) != len(reference_token_ids):
        raise ValueError("reference strings and token IDs are misaligned")
    canonical_references = [str(value) for value in references]
    full_string_exact = prediction in canonical_references
    eos_terminated = bool(
        generated_token_ids
        and eos_token_id is not None
        and int(generated_token_ids[-1]) == int(eos_token_id)
    )
    answer_eos_token_exact = bool(
        eos_token_id is not None
        and any(
            generated_token_ids
            == [*tokens, int(eos_token_id)]
            for tokens in reference_token_ids
        )
    )
    return {
        "full_string_exact": float(full_string_exact),
        "eos_terminated": float(eos_terminated),
        "answer_eos_token_exact": float(answer_eos_token_exact),
        "exact_generation_pass": float(
            full_string_exact and eos_terminated
        ),
    }


def load_completed(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("schema_version", 0)) != EXAMPLE_SCHEMA_VERSION:
            raise RuntimeError(
                "completed RULER rows predate the full-string/EOS contract; "
                "use a fresh output directory"
            )
        key = (int(row["nominal_length"]), int(row["local_index"]))
        if key in completed:
            raise RuntimeError(f"duplicate completed RULER row: {key}")
        completed[key] = row
    return completed


def validate_completed_exact_metrics(
    *,
    completed: dict[tuple[int, int], dict[str, Any]],
    tokenizer: Any,
) -> None:
    eos_token_id = tokenizer.eos_token_id
    for key, row in completed.items():
        token_ids = row.get("generated_token_ids")
        references = row.get("references")
        reference_token_ids = row.get("reference_token_ids")
        if (
            not isinstance(token_ids, list)
            or not all(isinstance(value, int) for value in token_ids)
            or not isinstance(references, list)
            or not isinstance(reference_token_ids, list)
            or int(row.get("generated_tokens", -1)) != len(token_ids)
            or len(token_ids) > GENERATION_TOKENS
            or (
                eos_token_id is not None
                and int(eos_token_id) in token_ids[:-1]
            )
        ):
            raise RuntimeError(f"completed exact row payload drift: {key}")
        prediction = decode_generated_string(
            tokenizer=tokenizer,
            generated_token_ids=token_ids,
            eos_token_id=eos_token_id,
        )
        metrics = exact_generation_metrics(
            prediction=prediction,
            references=[str(value) for value in references],
            generated_token_ids=token_ids,
            reference_token_ids=[
                [int(value) for value in tokens]
                for tokens in reference_token_ids
            ],
            eos_token_id=eos_token_id,
        )
        if row.get("prediction") != prediction or any(
            float(row.get(name, -1.0)) != float(expected)
            for name, expected in metrics.items()
        ):
            raise RuntimeError(
                f"completed exact row metric/decode drift: {key}"
            )


def validate_or_create_run_manifest(
    path: Path,
    expected: dict[str, Any],
) -> None:
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(
                "exact-screen output belongs to a different adapter, "
                "dataset, or decoding contract"
            )
        return
    atomic_json(path, expected)


def validate_adapter_metadata(
    metadata: dict[str, Any],
    *,
    checkpoint_digest: str,
    frequency: dict[str, Any],
    frequency_name: str,
    rank: int,
    alpha: float,
    adaptation: str = "qkvo_answer",
    qk_output_mask_sha256: str | None = None,
    parent_adapter_sha256: str | None = None,
) -> None:
    expected = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": frequency_name,
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": adaptation,
        "rank": int(rank),
        "alpha": float(alpha),
        "training_sequence_length": 4_096,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(
                f"adapter metadata drift for {name}: "
                f"{metadata.get(name)!r} != {value!r}"
            )
    if (
        qk_output_mask_sha256 is not None
        and metadata.get("qk_output_mask_sha256")
        != qk_output_mask_sha256
    ):
        raise RuntimeError("adapter Q/K output-mask identity drift")
    if adaptation == LENGTH_GATED_EOS_ADAPTATION:
        if parent_adapter_sha256 is None:
            raise RuntimeError(
                "EOS child metadata validation requires its parent SHA"
            )
        if (
            parent_adapter_sha256
            != LENGTH_GATED_PARENT_ADAPTER_SHA256
        ):
            raise RuntimeError(
                "EOS child parent_adapter_sha256 is not the registered "
                "a0ccd parent artifact"
            )
        eos_expected = {
            "parent_adapter_sha256": (
                LENGTH_GATED_PARENT_ADAPTER_SHA256
            ),
            "parent_adaptation": LENGTH_GATED_PARENT_ADAPTATION,
            "parent_rank": LENGTH_GATED_PARENT_RANK,
            "parent_alpha": LENGTH_GATED_PARENT_ALPHA,
            "modified_vocab_rows": [OLMO2_EOS_TOKEN_ID],
            "trainable_scope": LENGTH_GATED_EOS_TRAINABLE_SCOPE,
            "eos_token_id": OLMO2_EOS_TOKEN_ID,
            "parameterization": "direct_single_eos_row_delta",
            "rank1_equivalent": True,
            "trainable_parameters": 2_049,
            "trainable_parameter_tensors": 2,
        }
        for name, value in eos_expected.items():
            if metadata.get(name) != value:
                raise RuntimeError(
                    f"EOS child adapter metadata drift for {name}"
                )
    if adaptation == "length_gated_qkvo_answer":
        gated_expected = {
            "short_branch_frequency": "native_endpoint_rope",
            "short_branch_frequency_sha256_float32": (
                NATIVE_FREQUENCY_SHA256
            ),
            "short_branch_maximum_position_id": (
                SHORT_CONTEXT_LIMIT - 1
            ),
            "short_branch_lora": "frozen_base_linear_direct_call",
            "long_branch_frequency": "evq_endpoint_cosh",
            "long_branch_minimum_maximum_position_id": (
                SHORT_CONTEXT_LIMIT
            ),
            "long_branch_scope": "entire_sequence",
            "stage": "length_gated_query_gap_16k_eos_repair_v1",
            "maximum_physical_training_sequence_length": 4_096,
            "real_8k_or_16k_training_sequences": 0,
            "cached_generation_branch_selection": (
                "force_from_total_context_budget_before_prompt"
            ),
            "final_eos_supervised": True,
            "supervision_contract": (
                "numeric_answer_plus_immediate_eos_v1"
            ),
            "eos_token_id": 100_257,
            "termination_weight": 1.0,
        }
        for name, value in gated_expected.items():
            if metadata.get(name) != value:
                raise RuntimeError(
                    f"length-gated adapter metadata drift for {name}"
                )
        routing_sha = str(metadata.get("routing_data_sha256", ""))
        if len(routing_sha) != 64 or any(
            character not in "0123456789abcdef"
            for character in routing_sha
        ):
            raise RuntimeError(
                "length-gated adapter routing-data SHA drift"
            )


def load_length_gated_parent_child(
    model: Any,
    *,
    checkpoint_digest: str,
    parent_adapter_path: Path,
    child_adapter_path: Path,
    parent_rank: int,
    parent_alpha: float,
    child_rank: int,
    child_alpha: float,
    eos_token_id: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the frozen a0ccd QKVO parent, then the EOS-only child."""

    if (
        int(parent_rank) != LENGTH_GATED_PARENT_RANK
        or float(parent_alpha) != LENGTH_GATED_PARENT_ALPHA
        or int(child_rank) != LENGTH_GATED_EOS_HEAD_RANK
        or float(child_alpha) != LENGTH_GATED_EOS_HEAD_ALPHA
        or int(eos_token_id) != OLMO2_EOS_TOKEN_ID
    ):
        raise RuntimeError("length-gated parent/child hyperparameter drift")

    state, parent_method = install_length_gated_qkvo(
        model,
        rank=int(parent_rank),
        alpha=float(parent_alpha),
    )
    parent_sha256 = sha256_file(parent_adapter_path)
    if parent_sha256 != LENGTH_GATED_PARENT_ADAPTER_SHA256:
        raise RuntimeError("length-gated QKVO parent SHA drift")
    parent_metadata = load_adapter(parent_adapter_path, model, None)
    validate_adapter_metadata(
        parent_metadata,
        checkpoint_digest=checkpoint_digest,
        frequency={"active_sha256_float32": EVQ_FREQUENCY_SHA256},
        frequency_name="evq",
        rank=int(parent_rank),
        alpha=float(parent_alpha),
        adaptation=LENGTH_GATED_PARENT_ADAPTATION,
    )
    freeze_receipt = freeze_length_gated_qkvo_adapter(model)

    _, head_method = install_length_gated_eos_vocab_row_head(
        model,
        state,
        rank=int(child_rank),
        alpha=float(child_alpha),
        eos_token_id=int(eos_token_id),
    )
    child_scope = eos_head_trainable_named_parameters(model)
    child_metadata = load_adapter(child_adapter_path, model, None)
    validate_adapter_metadata(
        child_metadata,
        checkpoint_digest=checkpoint_digest,
        frequency=parent_method,
        frequency_name=LENGTH_GATED_FREQUENCY_NAME,
        rank=int(child_rank),
        alpha=float(child_alpha),
        adaptation=LENGTH_GATED_EOS_ADAPTATION,
        parent_adapter_sha256=parent_sha256,
    )
    method = dict(parent_method)
    observed_parent_rank = int(method.pop("rank"))
    observed_parent_alpha = float(method.pop("alpha"))
    observed_parent_adaptation = str(method["adaptation"])
    if (
        observed_parent_rank != int(parent_rank)
        or observed_parent_alpha != float(parent_alpha)
        or observed_parent_adaptation
        != "length_gated_qkvo_answer"
    ):
        raise RuntimeError("length-gated parent method receipt drift")
    method.update(
        {
            "adaptation": LENGTH_GATED_EOS_ADAPTATION,
            "parent_adaptation": LENGTH_GATED_PARENT_ADAPTATION,
            "parent_rank": int(parent_rank),
            "parent_alpha": float(parent_alpha),
            "parent_adapter_sha256": parent_sha256,
            "child_adapter": head_method,
            "trainable_scope": LENGTH_GATED_EOS_TRAINABLE_SCOPE,
        }
    )
    return (
        state,
        method,
        {
            "path": str(parent_adapter_path),
            "sha256": parent_sha256,
            "metadata": parent_metadata,
            "adaptation": LENGTH_GATED_PARENT_ADAPTATION,
            "rank": int(parent_rank),
            "alpha": float(parent_alpha),
            "freeze": freeze_receipt,
        },
        {
            "path": str(child_adapter_path),
            "sha256": sha256_file(child_adapter_path),
            "metadata": child_metadata,
            "adaptation": LENGTH_GATED_EOS_ADAPTATION,
            "rank": int(child_rank),
            "alpha": float(child_alpha),
            "method": head_method,
            "trainable_parameter_names": [
                name for name, _ in child_scope
            ],
        },
    )


def validate_length_gated_artifacts_before_cuda(
    *,
    checkpoint_digest: str,
    ready_path: Path,
    ready: dict[str, Any],
    parent_adapter_path: Path,
    child_adapter_path: Path,
) -> dict[str, Any]:
    """Bind and validate both registered artifacts without loading the model."""

    chain = ready.get("registered_adapter_chain", {})
    registered_parent = chain.get("parent", {})
    registered_child = chain.get("child", {})
    if (
        chain.get("load_order")
        != ["frozen_parent_qkvo", "eos_vocab_row_child"]
        or Path(str(registered_parent.get("path", ""))).resolve()
        != parent_adapter_path
        or Path(str(registered_child.get("path", ""))).resolve()
        != child_adapter_path
        or registered_parent.get("sha256")
        != LENGTH_GATED_PARENT_ADAPTER_SHA256
        or registered_parent.get("adaptation")
        != LENGTH_GATED_PARENT_ADAPTATION
        or int(registered_parent.get("rank", -1))
        != LENGTH_GATED_PARENT_RANK
        or float(registered_parent.get("alpha", float("nan")))
        != LENGTH_GATED_PARENT_ALPHA
        or registered_parent.get("frozen_during_child_training") is not True
        or registered_child.get("adaptation")
        != LENGTH_GATED_EOS_ADAPTATION
        or int(registered_child.get("rank", -1))
        != LENGTH_GATED_EOS_HEAD_RANK
        or float(registered_child.get("alpha", float("nan")))
        != LENGTH_GATED_EOS_HEAD_ALPHA
        or registered_child.get("modified_vocab_rows")
        != [OLMO2_EOS_TOKEN_ID]
        or registered_child.get("parameterization")
        != "direct_single_eos_row_delta"
        or registered_child.get("rank1_equivalent") is not True
        or int(registered_child.get("trainable_parameters", -1))
        != 2_049
        or int(
            registered_child.get("trainable_parameter_tensors", -1)
        )
        != 2
        or registered_child.get("trainable_scope")
        != LENGTH_GATED_EOS_TRAINABLE_SCOPE
        or registered_child.get("parent_adapter_sha256")
        != LENGTH_GATED_PARENT_ADAPTER_SHA256
        or chain.get("code_sha256")
        != {
            "candidate_evaluator": ready["code"]["exact_evaluator"][
                "sha256"
            ],
            "parent_adapter_loader": ready["code"]["adapter_loader"][
                "sha256"
            ],
            "parent_child_method": ready["code"]["method"]["sha256"],
        }
    ):
        raise RuntimeError("length-gated READY adapter-chain drift")

    def load_payload(
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
            or not all(
                isinstance(name, str)
                and isinstance(value, torch.Tensor)
                for name, value in payload["state"].items()
            )
        ):
            raise RuntimeError(f"{label} adapter payload drift")
        return (
            sha256_file(path),
            dict(payload["state"]),
            dict(payload["metadata"]),
        )

    parent_sha, parent_state, parent_metadata = load_payload(
        parent_adapter_path,
        label="parent",
    )
    if parent_sha != LENGTH_GATED_PARENT_ADAPTER_SHA256:
        raise RuntimeError("length-gated parent SHA drift before CUDA")
    expected_parent_shapes: dict[str, tuple[int, ...]] = {}
    for layer in range(16):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            prefix = (
                f"model.model.layers.{layer}.self_attn.{projection}"
            )
            expected_parent_shapes[f"{prefix}.a"] = (
                LENGTH_GATED_PARENT_RANK,
                2_048,
            )
            expected_parent_shapes[f"{prefix}.b"] = (
                2_048,
                LENGTH_GATED_PARENT_RANK,
            )
    if (
        set(parent_state) != set(expected_parent_shapes)
        or any(
            tuple(parent_state[name].shape) != shape
            for name, shape in expected_parent_shapes.items()
        )
        or any(
            not torch.is_floating_point(value)
            or not bool(torch.isfinite(value).all())
            for value in parent_state.values()
        )
    ):
        raise RuntimeError("length-gated parent tensor contract drift")
    validate_adapter_metadata(
        parent_metadata,
        checkpoint_digest=checkpoint_digest,
        frequency={"active_sha256_float32": EVQ_FREQUENCY_SHA256},
        frequency_name="evq",
        rank=LENGTH_GATED_PARENT_RANK,
        alpha=LENGTH_GATED_PARENT_ALPHA,
        adaptation=LENGTH_GATED_PARENT_ADAPTATION,
    )

    child_sha, child_state, child_metadata = load_payload(
        child_adapter_path,
        label="child",
    )
    expected_child_shapes = {
        "model.lm_head.delta_weight": (2_048,),
        "model.lm_head.eos_bias": (),
    }
    if (
        set(child_state) != set(expected_child_shapes)
        or any(
            tuple(child_state[name].shape) != shape
            for name, shape in expected_child_shapes.items()
        )
        or any(
            not torch.is_floating_point(value)
            or not bool(torch.isfinite(value).all())
            for value in child_state.values()
        )
    ):
        raise RuntimeError("length-gated child tensor contract drift")
    validate_adapter_metadata(
        child_metadata,
        checkpoint_digest=checkpoint_digest,
        frequency={"active_sha256_float32": EVQ_FREQUENCY_SHA256},
        frequency_name=LENGTH_GATED_FREQUENCY_NAME,
        rank=LENGTH_GATED_EOS_HEAD_RANK,
        alpha=LENGTH_GATED_EOS_HEAD_ALPHA,
        adaptation=LENGTH_GATED_EOS_ADAPTATION,
        parent_adapter_sha256=parent_sha,
    )

    training_result_path = (
        Path(str(ready.get("run_output", ""))).resolve()
        / "results.json"
    )
    if not training_result_path.is_file():
        raise RuntimeError(
            "length-gated training result is absent before candidate CUDA"
        )
    training_result = json.loads(
        training_result_path.read_text(encoding="utf-8")
    )
    if (
        training_result.get("status")
        != "OLMO2_4K_LENGTH_GATED_EOS_REPAIR_COMPLETE_V1"
        or training_result.get("checkpoint_sha256") != checkpoint_digest
        or training_result.get("ready_receipt_sha256")
        != sha256_file(ready_path)
        or training_result.get("script_sha256")
        != ready["code"]["trainer"]["sha256"]
        or training_result.get("bound_code")
        != {
            name: entry["sha256"]
            for name, entry in ready["code"].items()
        }
        or training_result.get("protocol") != ready.get("protocol")
        or training_result.get("adapter_sha256") != child_sha
        or training_result.get("adapter_metadata") != child_metadata
        or training_result.get("parent_adapter", {}).get("sha256")
        != parent_sha
    ):
        raise RuntimeError(
            "length-gated child is not bound to its completed training result"
        )
    return {
        "status": "PASS",
        "parent_adapter_sha256": parent_sha,
        "child_adapter_sha256": child_sha,
        "training_result": str(training_result_path),
        "training_result_sha256": sha256_file(training_result_path),
        "validation_stage": "cpu_before_cuda_configuration",
    }


def _legacy_cache_cpu(
    past_key_values: Any,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    legacy = (
        past_key_values.to_legacy_cache()
        if hasattr(past_key_values, "to_legacy_cache")
        else tuple(past_key_values)
    )
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in legacy:
        if len(layer) < 2:
            raise RuntimeError("short-parity cache layer lacks K/V tensors")
        result.append(
            (
                layer[0].detach().cpu().contiguous(),
                layer[1].detach().cpu().contiguous(),
            )
        )
    if not result:
        raise RuntimeError("short-parity trace produced an empty KV cache")
    return tuple(result)


@torch.inference_mode()
def capture_short_structural_trace(
    *,
    model: Any,
    input_ids: torch.Tensor,
    length_mode_state: Any | None,
) -> dict[str, Any]:
    if tuple(input_ids.shape) != (1, SHORT_CONTEXT_LIMIT):
        raise RuntimeError("short structural parity requires exactly 4096 tokens")
    if length_mode_state is not None:
        length_mode_state.force("short")
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
        )[0]
        final_logits = model.lm_head(hidden[:, -1:, :])
        prefill = model(
            input_ids=input_ids[:, :32],
            use_cache=True,
            return_dict=True,
        )
        prefill_logits = prefill.logits[:, -1:, :]
        prefill_cache = _legacy_cache_cpu(prefill.past_key_values)
        decode = model(
            input_ids=input_ids[:, 32:33],
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        decode_logits = decode.logits[:, -1:, :]
        decode_cache = _legacy_cache_cpu(decode.past_key_values)
    if (
        length_mode_state is not None
        and length_mode_state.mode != "short"
    ):
        raise RuntimeError("short structural parity used the long branch")
    values = {
        "hidden": hidden.detach().cpu().contiguous(),
        "final_logits": final_logits.detach().cpu().contiguous(),
        "prefill_logits": prefill_logits.detach().cpu().contiguous(),
        "decode_logits": decode_logits.detach().cpu().contiguous(),
        "prefill_cache": prefill_cache,
        "decode_cache": decode_cache,
    }
    tensors = [
        values["hidden"],
        values["final_logits"],
        values["prefill_logits"],
        values["decode_logits"],
        *[
            tensor
            for cache_name in ("prefill_cache", "decode_cache")
            for layer in values[cache_name]
            for tensor in layer
        ],
    ]
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise RuntimeError("short structural parity produced non-finite values")
    return values


def _nested_tensor_sha256(values: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for index, tensor in enumerate(values):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {
                "index": index,
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


def validate_short_structural_parity(
    *,
    pristine: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    scalar_names = (
        "hidden",
        "final_logits",
        "prefill_logits",
        "decode_logits",
    )
    equality = {
        name: torch.equal(pristine[name], candidate[name])
        for name in scalar_names
    }
    maximum_absolute_difference = {
        name: float(
            (
                pristine[name].float()
                - candidate[name].float()
            ).abs().max()
        )
        for name in scalar_names
    }
    cache_equality: dict[str, bool] = {}
    cache_hashes: dict[str, dict[str, str]] = {}
    cache_tensor_counts: dict[str, int] = {}
    for cache_name in ("prefill_cache", "decode_cache"):
        pristine_cache = pristine[cache_name]
        candidate_cache = candidate[cache_name]
        if len(pristine_cache) != len(candidate_cache):
            raise RuntimeError("short structural parity cache depth drift")
        pristine_tensors = [
            tensor for layer in pristine_cache for tensor in layer
        ]
        candidate_tensors = [
            tensor for layer in candidate_cache for tensor in layer
        ]
        cache_equality[cache_name] = (
            len(pristine_tensors) == len(candidate_tensors)
            and all(
                torch.equal(left, right)
                for left, right in zip(
                    pristine_tensors, candidate_tensors
                )
            )
        )
        cache_tensor_counts[cache_name] = len(pristine_tensors)
        cache_hashes[cache_name] = {
            "pristine_native": _nested_tensor_sha256(
                pristine_tensors
            ),
            "reloaded_candidate": _nested_tensor_sha256(
                candidate_tensors
            ),
        }
    passed = all(equality.values()) and all(cache_equality.values())
    if not passed:
        raise RuntimeError(
            "reloaded length-gated candidate changed the Native short path"
        )
    return {
        "status": "PASS",
        "input_tokens": SHORT_CONTEXT_LIMIT,
        "maximum_position_id": SHORT_CONTEXT_LIMIT - 1,
        "prefill_tokens": 32,
        "decode_steps": 1,
        "torch_equal": {
            **equality,
            **cache_equality,
        },
        "maximum_absolute_difference": maximum_absolute_difference,
        "cache_tensor_counts": cache_tensor_counts,
        "cache_sha256": cache_hashes,
        "runtime_contract": (
            "fresh checkpoint load; same BF16/Flash runtime; parent then "
            "child artifacts reloaded; branch forced short before prefill"
        ),
        "claim_boundary": (
            "exact for trajectories whose maximum actual position ID stays "
            "at most 4095; it does not cover a short prompt decoded across "
            "the 4096 boundary"
        ),
    }


class HeadHybridRotaryEmbedding(nn.Module):
    """Return native/EVQ rotary phases independently for each head."""

    def __init__(
        self,
        native: torch.Tensor,
        evq: torch.Tensor,
        *,
        head_count: int,
        evq_head_indices: tuple[int, ...],
    ) -> None:
        super().__init__()
        inv_freq = native.repeat(int(head_count), 1)
        inv_freq[list(evq_head_indices)] = evq
        self.register_buffer(
            "inv_freq_by_head",
            inv_freq.to(torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = position_ids[:, None, :, None].float()
        frequencies = (
            self.inv_freq_by_head[None, :, None, :].float()
            * positions
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            return embedding.cos(), embedding.sin()


_ORIGINAL_APPLY_ROTARY_POS_EMB = modeling_olmo2.apply_rotary_pos_emb


def apply_head_aware_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.ndim != 4:
        return _ORIGINAL_APPLY_ROTARY_POS_EMB(
            q,
            k,
            cos,
            sin,
            position_ids=position_ids,
            unsqueeze_dim=unsqueeze_dim,
        )
    if (
        cos.shape != sin.shape
        or cos.shape[1] != q.shape[1]
        or cos.shape[-1] != q.shape[-1]
    ):
        raise RuntimeError("head-hybrid rotary shape drift")
    q_type, k_type = q.dtype, k.dtype
    q_embed = (q * cos) + (modeling_olmo2.rotate_half(q) * sin)
    k_embed = (k * cos) + (modeling_olmo2.rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)


def apply_screen_frequency(
    model: Any,
    frequency_name: str,
    *,
    custom_evq_head_indices: tuple[int, ...] = (),
) -> dict[str, Any]:
    if frequency_name in {"native", "evq"}:
        return apply_frequency(model, frequency_name)
    native = (
        model.model.rotary_emb.inv_freq.detach()
        .cpu()
        .to(torch.float32)
        .clone()
    )
    receipt = apply_frequency(model, "evq")
    evq = (
        model.model.rotary_emb.inv_freq.detach()
        .cpu()
        .to(torch.float32)
        .clone()
    )
    if torch.equal(native, evq):
        raise RuntimeError(
            "Native and EVQ frequency snapshots unexpectedly alias or match"
        )
    pair_count = int(native.numel())
    if pair_count != 64:
        raise RuntimeError(f"expected 64 rotary pairs, got {pair_count}")
    if frequency_name.startswith("hybrid_heads_"):
        head_count = int(model.config.num_attention_heads)
        if frequency_name == "hybrid_heads_custom":
            evq_head_indices = custom_evq_head_indices
        else:
            evq_head_count = int(
                frequency_name.removeprefix("hybrid_heads_evq")
            )
            evq_head_indices = tuple(
                range(head_count - evq_head_count, head_count)
            )
        if (
            not evq_head_indices
            or len(set(evq_head_indices)) != len(evq_head_indices)
            or any(
                index < 0 or index >= head_count
                for index in evq_head_indices
            )
            or len(evq_head_indices) >= head_count
        ):
            raise RuntimeError("invalid head-hybrid EVQ head indices")
        evq_head_indices = tuple(sorted(evq_head_indices))
        evq_head_count = len(evq_head_indices)
        rotary = HeadHybridRotaryEmbedding(
            native,
            evq,
            head_count=head_count,
            evq_head_indices=evq_head_indices,
        )
        model.model.rotary_emb = rotary
        modeling_olmo2.apply_rotary_pos_emb = (
            apply_head_aware_rotary_pos_emb
        )
        return {
            **receipt,
            "active_frequency": frequency_name,
            "active_sha256_float32": tensor_sha256(
                rotary.inv_freq_by_head
            ),
            "hybrid_axis": "attention_head",
            "hybrid_native_head_indices": [
                index
                for index in range(head_count)
                if index not in set(evq_head_indices)
            ],
            "hybrid_evq_head_indices": list(evq_head_indices),
            "hybrid_native_head_count": (
                head_count - evq_head_count
            ),
            "hybrid_evq_head_count": evq_head_count,
            "hybrid_flash_attention_compatible": True,
        }
    blend_weight = None
    if frequency_name == "hybrid_native_low8":
        native_indices = tuple(range(56, 64))
    elif frequency_name == "hybrid_native_low16":
        native_indices = tuple(range(48, 64))
    elif frequency_name == "hybrid_native_low32":
        native_indices = tuple(range(32, 64))
    elif frequency_name == "hybrid_native_high16":
        native_indices = tuple(range(0, 16))
    elif frequency_name == "hybrid_native_ends16":
        native_indices = tuple(range(0, 8)) + tuple(range(56, 64))
    elif frequency_name == "hybrid_evq_low4":
        native_indices = tuple(range(0, 60))
    elif frequency_name == "hybrid_evq_low8":
        native_indices = tuple(range(0, 56))
    elif frequency_name == "hybrid_evq_low12":
        native_indices = tuple(range(0, 52))
    elif frequency_name == "hybrid_evq_low16":
        native_indices = tuple(range(0, 48))
    elif frequency_name == "hybrid_evq_low24":
        native_indices = tuple(range(0, 40))
    elif frequency_name == "hybrid_evq_low32":
        native_indices = tuple(range(0, 32))
    elif frequency_name == "hybrid_evq_low40":
        native_indices = tuple(range(0, 24))
    elif frequency_name == "hybrid_evq_high8":
        native_indices = tuple(range(8, 64))
    elif frequency_name == "hybrid_evq_mid8":
        native_indices = tuple(range(0, 28)) + tuple(range(36, 64))
    elif frequency_name in LOG_FREQUENCY_BLEND_WEIGHTS:
        native_indices = ()
        blend_weight = LOG_FREQUENCY_BLEND_WEIGHTS[frequency_name]
    else:
        raise ValueError(f"unknown frequency {frequency_name!r}")
    if blend_weight is None:
        hybrid = evq.clone()
        hybrid[list(native_indices)] = native[list(native_indices)]
    else:
        hybrid = torch.exp(
            (1.0 - blend_weight) * torch.log(native)
            + blend_weight * torch.log(evq)
        )
    with torch.no_grad():
        model.model.rotary_emb.inv_freq.copy_(
            hybrid.to(
                device=model.model.rotary_emb.inv_freq.device,
                dtype=model.model.rotary_emb.inv_freq.dtype,
            )
        )
    model.model.rotary_emb.original_inv_freq = (
        model.model.rotary_emb.inv_freq
    )
    receipt.update(
        {
            "active_frequency": frequency_name,
            "active_sha256_float32": tensor_sha256(hybrid),
            "hybrid_native_pair_indices": list(native_indices),
            "hybrid_evq_pair_indices": (
                [
                    index
                    for index in range(pair_count)
                    if index not in set(native_indices)
                ]
                if blend_weight is None
                else []
            ),
            "hybrid_native_pair_count": (
                len(native_indices) if blend_weight is None else 0
            ),
            "hybrid_evq_pair_count": (
                pair_count - len(native_indices)
                if blend_weight is None
                else 0
            ),
            "hybrid_blended_pair_count": (
                pair_count if blend_weight is not None else 0
            ),
            "hybrid_log_frequency_evq_weight": blend_weight,
            "hybrid_pair_order": (
                "index 0 is highest frequency; index 63 is lowest"
            ),
        }
    )
    return receipt


def main() -> None:
    args = parse_args()
    lengths = tuple(int(value) for value in args.lengths)
    if not lengths or tuple(sorted(set(lengths))) != lengths:
        raise RuntimeError("lengths must be unique and ascending")
    length_gated_adaptations = {
        "length_gated_qkvo_answer",
        LENGTH_GATED_EOS_ADAPTATION,
    }
    if args.adaptation in length_gated_adaptations:
        if (
            args.adapter is None
            or args.frequency != LENGTH_GATED_FREQUENCY_NAME
        ):
            raise RuntimeError(
                "length-gated evaluation requires its registered adapter "
                "and frequency"
            )
        if args.adaptation == LENGTH_GATED_EOS_ADAPTATION:
            if (
                args.parent_adapter is None
                or args.experiment_ready_receipt is None
                or int(args.rank) != LENGTH_GATED_EOS_HEAD_RANK
                or float(args.alpha) != LENGTH_GATED_EOS_HEAD_ALPHA
                or int(args.parent_rank) != LENGTH_GATED_PARENT_RANK
                or float(args.parent_alpha) != LENGTH_GATED_PARENT_ALPHA
            ):
                raise RuntimeError(
                    "EOS candidate requires the registered a0ccd parent and "
                    "length-gated READY plus fixed parent/child ranks and "
                    "alphas"
                )
        elif args.parent_adapter is not None:
            raise RuntimeError(
                "legacy length-gated adapter does not admit a parent artifact"
            )
    elif args.frequency == LENGTH_GATED_FREQUENCY_NAME:
        raise RuntimeError(
            "length-gated frequency requires length-gated adaptation"
        )
    elif args.parent_adapter is not None:
        raise RuntimeError(
            "--parent-adapter is only valid for the EOS child candidate"
        )
    elif args.adapter is not None and args.adaptation == "qk_answer":
        if args.frequency != HYBRID_FREQUENCY_NAME:
            raise RuntimeError(
                "Q/K-tail exact adapter requires hybrid_evq_low12"
            )
    elif (
        args.adapter is not None
        and args.frequency not in {"native", "evq"}
    ):
        raise RuntimeError(
            "only the registered Q/K-tail adapter admits a hybrid frequency"
        )
    output = args.output.resolve()
    examples_path = output / "examples.jsonl"
    run_manifest_path = output / "run_manifest.json"
    checkpoint = args.checkpoint.resolve()
    ready_receipt = args.ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, ready_receipt
    )
    data_receipt, rows = validate_data(
        args.data_root.resolve(),
        checkpoint,
        str(args.task),
        lengths,
        int(args.limit_per_length),
    )
    adapter_path = (
        None if args.adapter is None else args.adapter.resolve()
    )
    parent_adapter_path = (
        None
        if args.parent_adapter is None
        else args.parent_adapter.resolve()
    )
    experiment_ready_path = (
        None
        if args.experiment_ready_receipt is None
        else args.experiment_ready_receipt.resolve()
    )
    experiment_ready = None
    length_gated_artifact_preflight = None
    if experiment_ready_path is not None:
        experiment_ready = json.loads(
            experiment_ready_path.read_text(encoding="utf-8")
        )
        experiment_status = experiment_ready.get("status")
        if experiment_status not in {
            EOS_REPAIR_READY_STATUS,
            LENGTH_GATED_READY_STATUS,
        }:
            raise RuntimeError("exact evaluation READY status drift")
        if (
            args.adaptation == LENGTH_GATED_EOS_ADAPTATION
            and experiment_status != LENGTH_GATED_READY_STATUS
        ):
            raise RuntimeError(
                "EOS child requires the dedicated length-gated READY"
            )
        if (
            experiment_ready["evaluator"]["sha256"]
            != sha256_file(Path(__file__).resolve())
            or experiment_ready.get("evaluator_bound_code_sha256")
            != bound_code_sha256()
        ):
            raise RuntimeError("exact evaluation bound-code drift")
        if (
            experiment_ready["inputs"]["checkpoint"][
                "composite_sha256"
            ]
            != checkpoint_digest
            or experiment_ready["inputs"]["checkpoint"][
                "ready_receipt"
            ]["sha256"]
            != sha256_file(ready_receipt)
        ):
            raise RuntimeError("exact evaluation checkpoint input drift")
        for name, relative_path in (
            ("config", "config.json"),
            ("model", "model.safetensors"),
            ("tokenizer", "tokenizer.json"),
        ):
            if (
                experiment_ready["inputs"]["checkpoint"][name]["sha256"]
                != sha256_file(checkpoint / relative_path)
            ):
                raise RuntimeError(
                    f"exact evaluation checkpoint {relative_path} drift"
                )
        ready_data = experiment_ready["inputs"]["exact_eval_data"]
        if (
            Path(ready_data["path"]).resolve()
            != args.data_root.resolve()
            or ready_data["manifest_sha256"]
            != data_receipt["manifest_sha256"]
            or {
                key: value["sha256"]
                for key, value in ready_data["files"].items()
            }
            != {
                key: value["sha256"]
                for key, value in data_receipt["files"].items()
            }
        ):
            raise RuntimeError("exact evaluation frozen-data drift")
        role = None
        registered_outputs = experiment_ready["registered_outputs"]
        if (
            "parent_exact_baseline" in registered_outputs
            and output
            == Path(
                registered_outputs["parent_exact_baseline"]
            ).resolve()
        ):
            role = "parent_exact_baseline"
            if (
                adapter_path is None
                or sha256_file(adapter_path)
                != experiment_ready["inputs"]["parent_adapter"]["sha256"]
                or parent_adapter_path is not None
            ):
                raise RuntimeError("parent exact-baseline adapter drift")
        elif output == Path(
            registered_outputs["candidate_exact_screen"]
        ).resolve():
            role = "candidate_exact_screen"
            if adapter_path is None:
                raise RuntimeError("candidate exact screen lacks adapter")
            if (
                args.adaptation == LENGTH_GATED_EOS_ADAPTATION
                and (
                    parent_adapter_path is None
                    or sha256_file(parent_adapter_path)
                    != experiment_ready["inputs"]["parent_adapter"]["sha256"]
                )
            ):
                raise RuntimeError("candidate QKVO parent adapter drift")
        else:
            raise RuntimeError("exact evaluation output is not registered")
        if experiment_status == LENGTH_GATED_READY_STATUS:
            expected_method = (
                ("evq", "qkvo_answer")
                if role == "parent_exact_baseline"
                else (
                    LENGTH_GATED_FREQUENCY_NAME,
                    LENGTH_GATED_EOS_ADAPTATION,
                )
            )
            expected_hyperparameters = (
                (
                    LENGTH_GATED_PARENT_RANK,
                    LENGTH_GATED_PARENT_ALPHA,
                )
                if role == "parent_exact_baseline"
                else (
                    LENGTH_GATED_EOS_HEAD_RANK,
                    LENGTH_GATED_EOS_HEAD_ALPHA,
                )
            )
            if (
                (args.frequency, args.adaptation) != expected_method
                or (int(args.rank), float(args.alpha))
                != expected_hyperparameters
                or lengths != (8_192, 16_384)
                or int(args.limit_per_length) != 8
            ):
                raise RuntimeError(
                    "length-gated minimal exact-screen protocol drift"
                )
        if args.adaptation == LENGTH_GATED_EOS_ADAPTATION:
            if (
                adapter_path is None
                or parent_adapter_path is None
                or role != "candidate_exact_screen"
            ):
                raise RuntimeError(
                    "length-gated artifact preflight lacks candidate inputs"
                )
            length_gated_artifact_preflight = (
                validate_length_gated_artifacts_before_cuda(
                    checkpoint_digest=checkpoint_digest,
                    ready_path=experiment_ready_path,
                    ready=experiment_ready,
                    parent_adapter_path=parent_adapter_path,
                    child_adapter_path=adapter_path,
                )
            )
    else:
        role = None
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    actual_prompt_geometry = validate_actual_prompt_geometry(
        rows=rows,
        tokenizer=tokenizer,
    )
    actual_prompt_geometry_rows = {
        (int(row["nominal_length"]), int(row["local_index"])): row
        for row in actual_prompt_geometry["rows"]
    }
    if (
        experiment_ready is not None
        and experiment_ready.get("status") == LENGTH_GATED_READY_STATUS
        and experiment_ready["inputs"]["exact_eval_data"].get(
            "actual_prompt_geometry"
        )
        != actual_prompt_geometry
    ):
        raise RuntimeError(
            "actual prompt geometry changed after length-gated READY"
        )
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "status": RUN_MANIFEST_STATUS,
        "example_schema_version": EXAMPLE_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_digest,
        "ready_receipt_sha256": sha256_file(ready_receipt),
        "experiment_ready_receipt_sha256": (
            None
            if experiment_ready_path is None
            else sha256_file(experiment_ready_path)
        ),
        "experiment_role": role,
        "bound_code_sha256": bound_code_sha256(),
        "data_manifest_sha256": data_receipt["manifest_sha256"],
        "data_files": {
            length: entry["sha256"]
            for length, entry in data_receipt["files"].items()
        },
        "frequency": str(args.frequency),
        "evq_head_indices": [
            int(value) for value in (args.evq_head_indices or ())
        ],
        "adapter_sha256": (
            None if adapter_path is None else sha256_file(adapter_path)
        ),
        "parent_adapter_sha256": (
            None
            if parent_adapter_path is None
            else sha256_file(parent_adapter_path)
        ),
        "length_gated_artifact_preflight": (
            length_gated_artifact_preflight
        ),
        "adaptation": str(args.adaptation),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "parent_rank": (
            int(args.parent_rank)
            if parent_adapter_path is not None
            else None
        ),
        "parent_alpha": (
            float(args.parent_alpha)
            if parent_adapter_path is not None
            else None
        ),
        "task": str(args.task),
        "lengths": list(lengths),
        "limit_per_length": int(args.limit_per_length),
        "greedy": True,
        "maximum_new_tokens": GENERATION_TOKENS,
        "string_normalization": "none",
        "decode_cleanup": False,
        "terminal_eos_removed_before_string_decode": True,
        "other_special_tokens_removed": False,
        "actual_prompt_geometry": actual_prompt_geometry,
    }
    if examples_path.exists() != run_manifest_path.exists():
        if run_manifest_path.is_file() and not examples_path.exists():
            validate_or_create_run_manifest(run_manifest_path, run_manifest)
            examples_path.touch()
        elif (
            examples_path.is_file()
            and examples_path.stat().st_size == 0
            and not run_manifest_path.exists()
        ):
            validate_or_create_run_manifest(run_manifest_path, run_manifest)
        else:
            raise RuntimeError(
                "non-empty exact-screen examples and run manifest are not "
                "a resumable bound pair"
            )
    validate_or_create_run_manifest(run_manifest_path, run_manifest)
    examples_path.touch(exist_ok=True)
    completed = load_completed(examples_path)
    expected_rows = {
        (int(row["_nominal_length"]), int(row["_local_index"])): row
        for row in rows
    }
    for key, result in completed.items():
        if key not in expected_rows:
            raise RuntimeError(f"completed row is outside this run: {key}")
        source = expected_rows[key]
        geometry = actual_prompt_geometry_rows[key]
        expected_row_sha = row_sha256(
            {
                name: value
                for name, value in source.items()
                if not name.startswith("_")
            }
        )
        if (
            result.get("task") != str(args.task)
            or result.get("row_sha256") != expected_row_sha
            or result.get("references")
            != [str(value) for value in source["outputs"]]
            or int(result.get("input_tokens", -1))
            != int(geometry["prompt_tokens"])
            or int(
                result.get("actual_maximum_prompt_position_id", -1)
            )
            != int(geometry["actual_maximum_prompt_position_id"])
            or int(
                result.get("generation_boundary_gap_tokens", -1)
            )
            != int(geometry["generation_boundary_gap_tokens"])
            or result.get("far_gap_beyond_training_support")
            is not bool(
                geometry["far_gap_beyond_training_support"]
            )
        ):
            raise RuntimeError(
                f"completed exact-screen row identity drift: {key}"
            )

    validate_completed_exact_metrics(
        completed=completed,
        tokenizer=tokenizer,
    )
    existing_result_path = output / "results.json"
    if existing_result_path.exists() and len(completed) != len(rows):
        raise RuntimeError(
            "results.json exists with an incomplete raw-row set; refuse "
            "GPU resume or overwrite"
        )
    if len(completed) == len(rows):
        if not existing_result_path.is_file():
            raise RuntimeError(
                "all exact rows are complete but results.json is absent; "
                "refuse automatic GPU reload"
            )
        existing_result = json.loads(
            existing_result_path.read_text(encoding="utf-8")
        )
        if (
            existing_result.get("status")
            != "OLMO2_INSTRUCT_RULER_EXACT_SCREEN_COMPLETE_V3"
            or existing_result.get("checkpoint_sha256")
            != checkpoint_digest
            or existing_result.get("run_manifest_sha256")
            != sha256_file(run_manifest_path)
            or existing_result.get("results", {}).get("examples_sha256")
            != sha256_file(examples_path)
            or existing_result.get("experiment_ready_receipt_sha256")
            != (
                None
                if experiment_ready_path is None
                else sha256_file(experiment_ready_path)
            )
            or (
                None
                if existing_result.get("adapter") is None
                else existing_result["adapter"].get("sha256")
            )
            != (
                None
                if adapter_path is None
                else sha256_file(adapter_path)
            )
        ):
            raise RuntimeError("completed exact result binding drift")
        print(
            json.dumps(
                {
                    "status": existing_result["status"],
                    "output": str(existing_result_path),
                    "resume": "already_complete_no_gpu_reload",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    configure_cuda()
    runtime = {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "mem_efficient_sdp_enabled": (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
    }
    model = load_model(checkpoint)
    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    model.eval()
    length_mode_state = None
    parent_adapter_receipt = None
    adapter_receipt = None
    short_structural_parity = None
    pristine_short_trace = None
    short_parity_ids = None
    if args.adaptation == LENGTH_GATED_EOS_ADAPTATION:
        if adapter_path is None or parent_adapter_path is None:
            raise RuntimeError("EOS candidate lacks parent or child adapter")
        parity_row = next(
            (
                row
                for row in rows
                if int(row["_nominal_length"])
                > SHORT_CONTEXT_LIMIT
            ),
            None,
        )
        if parity_row is None:
            raise RuntimeError("EOS candidate lacks a long row for 4K parity")
        parity_chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": parity_row["input"]}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if int(parity_chat_ids.shape[1]) < SHORT_CONTEXT_LIMIT:
            raise RuntimeError(
                "registered long row is too short for 4096-token parity"
            )
        short_parity_ids = parity_chat_ids[
            :, :SHORT_CONTEXT_LIMIT
        ].to("cuda")
        model.to("cuda")
        pristine_short_trace = capture_short_structural_trace(
            model=model,
            input_ids=short_parity_ids,
            length_mode_state=None,
        )
        (
            length_mode_state,
            frequency,
            parent_adapter_receipt,
            adapter_receipt,
        ) = load_length_gated_parent_child(
            model,
            checkpoint_digest=checkpoint_digest,
            parent_adapter_path=parent_adapter_path,
            child_adapter_path=adapter_path,
            parent_rank=int(args.parent_rank),
            parent_alpha=float(args.parent_alpha),
            child_rank=int(args.rank),
            child_alpha=float(args.alpha),
            eos_token_id=int(tokenizer.eos_token_id),
        )
    else:
        if args.adaptation == "length_gated_qkvo_answer":
            length_mode_state, frequency = install_length_gated_qkvo(
                model,
                rank=int(args.rank),
                alpha=float(args.alpha),
            )
        else:
            frequency = apply_screen_frequency(
                model,
                args.frequency,
                custom_evq_head_indices=tuple(args.evq_head_indices or ()),
            )
        if adapter_path is not None:
            qk_output_mask = (
                qk_evq_tail_output_mask(model.config)
                if args.adaptation == "qk_answer"
                else None
            )
            if args.adaptation == "length_gated_qkvo_answer":
                if length_mode_state is None:
                    raise RuntimeError(
                        "length-gated method was not installed"
                    )
            else:
                readout = install_adaptation(
                    model,
                    args.adaptation,
                    rank=int(args.rank),
                    alpha=float(args.alpha),
                    qk_output_mask=qk_output_mask,
                )
                if readout is not None:
                    raise RuntimeError("RULER screen does not admit a readout")
            adapter_metadata = load_adapter(adapter_path, model, None)
            validate_adapter_metadata(
                adapter_metadata,
                checkpoint_digest=checkpoint_digest,
                frequency=frequency,
                frequency_name=args.frequency,
                rank=int(args.rank),
                alpha=float(args.alpha),
                adaptation=str(args.adaptation),
                qk_output_mask_sha256=(
                    None
                    if qk_output_mask is None
                    else tensor_sha256(qk_output_mask)
                ),
            )
            adapter_receipt = {
                "path": str(adapter_path),
                "sha256": sha256_file(adapter_path),
                "metadata": adapter_metadata,
                "adaptation": str(args.adaptation),
                "qk_output_mask_sha256": (
                    None
                    if qk_output_mask is None
                    else tensor_sha256(qk_output_mask)
                ),
            }

    model.to("cuda")
    if args.adaptation == LENGTH_GATED_EOS_ADAPTATION:
        if (
            pristine_short_trace is None
            or short_parity_ids is None
            or length_mode_state is None
        ):
            raise RuntimeError("short structural parity trace is incomplete")
        candidate_short_trace = capture_short_structural_trace(
            model=model,
            input_ids=short_parity_ids,
            length_mode_state=length_mode_state,
        )
        short_structural_parity = validate_short_structural_parity(
            pristine=pristine_short_trace,
            candidate=candidate_short_trace,
        )
        del pristine_short_trace, candidate_short_trace, short_parity_ids
    expected = len(rows)
    torch.cuda.reset_peak_memory_stats()

    with examples_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            length = int(row["_nominal_length"])
            local_index = int(row["_local_index"])
            key = (length, local_index)
            if key in completed:
                continue
            if length_mode_state is not None:
                length_mode_state.force(
                    "short" if length <= SHORT_CONTEXT_LIMIT else "long"
                )
            chat_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["input"]}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            prefix_ids = tokenizer(
                row.get("answer_prefix", ""),
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids
            input_ids = torch.cat((chat_ids, prefix_ids), dim=1).to(
                "cuda"
            )
            input_tokens = int(input_ids.shape[1])
            source_token_position_answer = int(
                row["token_position_answer"]
            )
            generation_boundary_gap_tokens = (
                input_tokens - source_token_position_answer
            )
            if (
                input_tokens - 1 < length // 2
                or input_tokens + GENERATION_TOKENS > length
                or source_token_position_answer < 0
                or source_token_position_answer >= input_tokens
                or generation_boundary_gap_tokens <= 0
            ):
                raise RuntimeError(
                    f"RULER prompt geometry drift at L={length}"
                )
            started = time.perf_counter()
            output_ids = greedy_generate(
                model,
                input_ids,
                max_new_tokens=GENERATION_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            generated_token_ids = [
                int(value)
                for value in output_ids[0].detach().cpu().tolist()
            ]
            prediction = decode_generated_string(
                tokenizer=tokenizer,
                generated_token_ids=generated_token_ids,
                eos_token_id=tokenizer.eos_token_id,
            )
            references = [str(value) for value in row["outputs"]]
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
            for reference, token_ids in zip(
                references,
                reference_token_ids,
            ):
                decoded_reference = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if decoded_reference != reference:
                    raise RuntimeError(
                        "reference does not round-trip to its literal string"
                    )
            exact_metrics = exact_generation_metrics(
                prediction=prediction,
                references=references,
                generated_token_ids=generated_token_ids,
                reference_token_ids=reference_token_ids,
                eos_token_id=tokenizer.eos_token_id,
            )
            result = {
                "schema_version": EXAMPLE_SCHEMA_VERSION,
                "task": str(args.task),
                "nominal_length": length,
                "local_index": local_index,
                "source_row_index": int(row["index"]),
                "source_token_position_answer": int(
                    row["token_position_answer"]
                ),
                "row_sha256": row_sha256(
                    {
                        key: value
                        for key, value in row.items()
                        if not key.startswith("_")
                    }
                ),
                "input_tokens": input_tokens,
                "actual_maximum_prompt_position_id": input_tokens - 1,
                "generation_boundary_gap_tokens": (
                    generation_boundary_gap_tokens
                ),
                "required_minimum_prompt_tokens": (
                    length // 2 + 1
                ),
                "required_minimum_actual_maximum_position_id": (
                    length // 2
                ),
                "far_gap_beyond_training_support": (
                    generation_boundary_gap_tokens
                    > TRAINING_GAP_MAXIMUM_TOKENS
                ),
                "generated_tokens": int(output_ids.numel()),
                "generated_token_ids": generated_token_ids,
                "prediction": prediction,
                "references": references,
                "reference_token_ids": reference_token_ids,
                **exact_metrics,
                "official_string_match": score_prediction(
                    prediction, references
                ),
                "first_number_exact": first_number_exact(
                    prediction, references
                ),
                "elapsed_seconds": elapsed,
            }
            handle.write(
                json.dumps(result, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
            handle.flush()
            completed[key] = result
            print(
                f"{len(completed)}/{expected} L={length} "
                f"exact_eos={result['exact_generation_pass']:.3f} "
                f"full={result['full_string_exact']:.3f} "
                f"seconds={elapsed:.2f}",
                flush=True,
            )

    relevant = [
        completed[(length, local_index)]
        for length in lengths
        for local_index in range(int(args.limit_per_length))
    ]
    if len(relevant) != expected:
        raise RuntimeError("RULER screen result-count drift")
    cells = {}
    for length in lengths:
        selected = [
            row
            for row in relevant
            if int(row["nominal_length"]) == length
        ]
        far_gap_selected = [
            row
            for row in selected
            if row["far_gap_beyond_training_support"] is True
        ]
        if length > 4_096 and not far_gap_selected:
            raise RuntimeError(
                f"L{length} exact cell lacks a far-gap example"
            )
        cells[str(length)] = {
            "examples": len(selected),
            "full_string_exact": sum(
                float(row["full_string_exact"]) for row in selected
            )
            / len(selected),
            "eos_terminated": sum(
                float(row["eos_terminated"]) for row in selected
            )
            / len(selected),
            "answer_eos_token_exact": sum(
                float(row["answer_eos_token_exact"]) for row in selected
            )
            / len(selected),
            "far_gap_examples": len(far_gap_selected),
            "far_gap_exact_generation_pass": (
                None
                if not far_gap_selected
                else sum(
                    float(row["exact_generation_pass"])
                    for row in far_gap_selected
                )
                / len(far_gap_selected)
            ),
            "exact_generation_pass": sum(
                float(row["exact_generation_pass"]) for row in selected
            )
            / len(selected),
            "official_string_match": sum(
                float(row["official_string_match"]) for row in selected
            )
            / len(selected),
            "first_number_exact": sum(
                float(row["first_number_exact"]) for row in selected
            )
            / len(selected),
            "mean_elapsed_seconds": sum(
                float(row["elapsed_seconds"]) for row in selected
            )
            / len(selected),
        }
    macro_exact = sum(
        float(cell["exact_generation_pass"]) for cell in cells.values()
    ) / len(cells)
    macro_official = sum(
        float(cell["official_string_match"]) for cell in cells.values()
    ) / len(cells)
    if not math.isfinite(macro_exact) or not math.isfinite(macro_official):
        raise RuntimeError("non-finite RULER screen result")
    receipt = {
        "status": "OLMO2_INSTRUCT_RULER_EXACT_SCREEN_COMPLETE_V3",
        "metric_boundary": (
            "greedy autoregressive whole-continuation decoded-string exact "
            "with terminal EOS; official substring and first-number scores "
            "are secondary diagnostics; not the full 13-task RULER suite"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "ready_receipt_sha256": sha256_file(ready_receipt),
        "experiment_ready_receipt_sha256": (
            None
            if experiment_ready_path is None
            else sha256_file(experiment_ready_path)
        ),
        "bound_code_sha256": bound_code_sha256(),
        "frequency": frequency,
        "adapter": adapter_receipt,
        "parent_adapter": parent_adapter_receipt,
        "length_gated_artifact_preflight": (
            length_gated_artifact_preflight
        ),
        "short_structural_parity": short_structural_parity,
        "data": data_receipt,
        "actual_prompt_geometry": actual_prompt_geometry,
        "protocol": {
            "task": str(args.task),
            "lengths": list(lengths),
            "limit_per_length": int(args.limit_per_length),
            "greedy": True,
            "maximum_new_tokens": GENERATION_TOKENS,
            "minimum_far_gap_rows_per_cell": 1,
            "far_gap_threshold_tokens": TRAINING_GAP_MAXIMUM_TOKENS,
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
            "training_length_if_adapted": (
                4_096 if adapter_receipt is not None else None
            ),
            "length_gated_cache_branch_selection": (
                "forced_from_nominal_context_budget_before_prompt"
                if length_mode_state is not None
                else None
            ),
            "attention": "flash_only_custom_kv_cache",
            "precision": "bf16_weights_and_autocast",
        },
        "runtime": {
            **runtime,
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "peak_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved()
            ),
        },
        "results": {
            "cells": cells,
            "macro_exact_generation_pass": macro_exact,
            "macro_official_string_match": macro_official,
            "examples": len(relevant),
            "examples_sha256": sha256_file(examples_path),
        },
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output / "results.json"),
                "cells": cells,
                "macro_exact_generation_pass": macro_exact,
                "macro_official_string_match": macro_official,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
