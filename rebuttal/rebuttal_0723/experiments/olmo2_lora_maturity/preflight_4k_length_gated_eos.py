#!/usr/bin/env python3
"""Create the no-GPU READY receipt for length-gated OLMo-2 EOS repair."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    endpoint_evq_inv_freq,
    endpoint_geo_inv_freq,
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    row_sha256,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    sha256_file,
)

from .evaluate_instruct_ruler_screen import (
    EXAMPLE_SCHEMA_VERSION,
    GENERATION_TOKENS,
    bound_code_sha256 as evaluator_bound_code_sha256,
    validate_actual_prompt_geometry,
    validate_data,
)
from .olmo2_length_gated_method import (
    EVQ_FREQUENCY_SHA256,
    LENGTH_GATED_FREQUENCY_NAME,
    NATIVE_FREQUENCY_SHA256,
    SHORT_CONTEXT_LIMIT,
)
from .preflight_4k_query_gap_eos_repair import (
    LEGACY_ROUTING_MANIFEST_SHA256,
    file_entry,
    validate_natural_view,
    validate_routing_upgrade,
)
from .prepare_4k_routing_pairs import (
    LENGTH,
    OLMO2_EOS_TOKEN_ID,
)
from .train_4k_length_gated_eos import (
    MODEL_SHA256,
    PARENT_ADAPTER_SHA256,
    PRE_QUERY_GAP_PARENT_SHA256,
    READY_STATUS,
    protocol,
)
from .train_4k_stage_a import ready_checkpoint_digest


EXACT_LENGTHS = (8_192, 16_384)
EXACT_ROWS_PER_LENGTH = 8
MINIMUM_FREE_BYTES = 20 * 1024**3
ADAPTATION = "length_gated_eos_vocab_row"
PARENT_ADAPTATION = "qkvo_answer"
PARENT_RANK = 64
PARENT_ALPHA = 128.0
EOS_HEAD_RANK = 1
EOS_HEAD_ALPHA = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-ready-receipt", type=Path, required=True
    )
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--legacy-routing-data", type=Path, required=True)
    parser.add_argument("--routing-data-v2", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--exact-eval-data", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--parent-eval-output", type=Path, required=True)
    parser.add_argument("--candidate-eval-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args()


def _parent_adapter_entry(path: Path) -> dict[str, Any]:
    entry = file_entry(path)
    if entry["sha256"] != PARENT_ADAPTER_SHA256:
        raise RuntimeError("query-gap parent adapter SHA drift")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("state"), dict)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise RuntimeError("query-gap parent adapter payload drift")
    metadata = payload["metadata"]
    expected_metadata = {
        "base_checkpoint_sha256": MODEL_SHA256,
        "frequency": "evq",
        "adaptation": PARENT_ADAPTATION,
        "rank": PARENT_RANK,
        "alpha": PARENT_ALPHA,
        "training_sequence_length": LENGTH,
        "stage": "counterfactual_routing_semantic_query_gap_16k",
        "parent_adapter_sha256": PRE_QUERY_GAP_PARENT_SHA256,
        "position_policy": "semantic_query_block_continuous_gap",
        "virtual_target_length": 16_384,
    }
    for name, value in expected_metadata.items():
        if metadata.get(name) != value:
            raise RuntimeError(
                f"query-gap parent metadata drift for {name}"
            )

    expected_shapes: dict[str, tuple[int, int]] = {}
    for layer in range(16):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            prefix = (
                f"model.model.layers.{layer}.self_attn."
                f"{projection}"
            )
            expected_shapes[f"{prefix}.a"] = (PARENT_RANK, 2_048)
            expected_shapes[f"{prefix}.b"] = (2_048, PARENT_RANK)
    state = payload["state"]
    if set(state) != set(expected_shapes):
        raise RuntimeError(
            "query-gap parent is not a complete QKVO LoRA state"
        )
    for name, expected_shape in expected_shapes.items():
        if (
            tuple(state[name].shape) != expected_shape
            or not torch.is_floating_point(state[name])
            or not bool(torch.isfinite(state[name]).all())
        ):
            raise RuntimeError(
                f"query-gap parent tensor contract drift for {name}"
            )
    entry["metadata"] = metadata
    entry["state_tensors"] = len(state)
    entry["length_gated_state_compatible"] = True
    return entry


def _method_contract() -> dict[str, Any]:
    native = endpoint_geo_inv_freq()
    evq = endpoint_evq_inv_freq()
    if (
        tensor_sha256(native) != NATIVE_FREQUENCY_SHA256
        or tensor_sha256(evq) != EVQ_FREQUENCY_SHA256
        or torch.equal(native, evq)
        or SHORT_CONTEXT_LIMIT != LENGTH
    ):
        raise RuntimeError("length-gated frequency contract drift")
    return {
        "method": LENGTH_GATED_FREQUENCY_NAME,
        "short_branch_maximum_position_id": LENGTH - 1,
        "short_frequency_sha256_float32": NATIVE_FREQUENCY_SHA256,
        "long_branch_minimum_maximum_position_id": LENGTH,
        "long_frequency_sha256_float32": EVQ_FREQUENCY_SHA256,
        "branch_scope": "entire_sequence",
        "short_rotary_dispatch": "original_module_direct_call",
        "short_lora_dispatch": "base_linear_direct_call",
        "cached_generation_branch_selection": (
            "forced from total context budget before prompt"
        ),
        "cached_short_to_long_transition": "fail_closed",
        "parent_state_compatibility": (
            "same QKVO LoRA parameter names and shapes"
        ),
        "parent_adaptation": PARENT_ADAPTATION,
        "parent_rank": PARENT_RANK,
        "parent_alpha": PARENT_ALPHA,
        "parent_trainable_after_load": False,
        "child_adaptation": ADAPTATION,
        "child_rank": EOS_HEAD_RANK,
        "child_alpha": EOS_HEAD_ALPHA,
        "child_modified_vocab_rows": [OLMO2_EOS_TOKEN_ID],
        "child_parameterization": "direct_single_eos_row_delta",
        "child_rank1_equivalent": True,
        "child_trainable_parameters": 2_049,
        "child_trainable_parameter_tensors": 2,
        "child_trainable_scope": (
            "long_only_eos_vocab_row_plus_scalar_bias"
        ),
        "adapter_load_order": [
            "frozen_parent_qkvo",
            "eos_vocab_row_child",
        ],
    }


def _registered_commands(paths: dict[str, Path]) -> dict[str, list[str]]:
    training_command = [
        "python",
        "-m",
        (
            "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
            "train_4k_length_gated_eos"
        ),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--checkpoint-ready-receipt",
        str(paths["checkpoint_ready_receipt"]),
        "--parent-adapter",
        str(paths["parent_adapter"]),
        "--prepared-data",
        str(paths["prepared_data"]),
        "--routing-data",
        str(paths["routing_data_v2"]),
        "--ready-receipt",
        str(paths["receipt_output"]),
        "--parent-exact-baseline-result",
        str(paths["parent_eval_output"] / "results.json"),
        "--parent-exact-baseline-examples",
        str(paths["parent_eval_output"] / "examples.jsonl"),
        "--parent-exact-baseline-run-manifest",
        str(paths["parent_eval_output"] / "run_manifest.json"),
        "--output",
        str(paths["run_output"]),
    ]
    parent_evaluation_command = [
        "python",
        "-m",
        (
            "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
            "evaluate_instruct_ruler_screen"
        ),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--ready-receipt",
        str(paths["checkpoint_ready_receipt"]),
        "--experiment-ready-receipt",
        str(paths["receipt_output"]),
        "--data-root",
        str(paths["exact_eval_data"]),
        "--output",
        str(paths["parent_eval_output"]),
        "--frequency",
        "evq",
        "--task",
        "niah_single_1",
        "--adapter",
        str(paths["parent_adapter"]),
        "--adaptation",
        PARENT_ADAPTATION,
        "--rank",
        str(PARENT_RANK),
        "--alpha",
        str(PARENT_ALPHA),
        "--lengths",
        *(str(value) for value in EXACT_LENGTHS),
        "--limit-per-length",
        str(EXACT_ROWS_PER_LENGTH),
    ]
    candidate_evaluation_command = [
        "python",
        "-m",
        (
            "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
            "evaluate_instruct_ruler_screen"
        ),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--ready-receipt",
        str(paths["checkpoint_ready_receipt"]),
        "--experiment-ready-receipt",
        str(paths["receipt_output"]),
        "--data-root",
        str(paths["exact_eval_data"]),
        "--output",
        str(paths["candidate_eval_output"]),
        "--frequency",
        LENGTH_GATED_FREQUENCY_NAME,
        "--task",
        "niah_single_1",
        "--parent-adapter",
        str(paths["parent_adapter"]),
        "--parent-rank",
        str(PARENT_RANK),
        "--parent-alpha",
        str(PARENT_ALPHA),
        "--adapter",
        str(paths["run_output"] / "adapter.pt"),
        "--adaptation",
        ADAPTATION,
        "--rank",
        str(EOS_HEAD_RANK),
        "--alpha",
        str(EOS_HEAD_ALPHA),
        "--lengths",
        *(str(value) for value in EXACT_LENGTHS),
        "--limit-per-length",
        str(EXACT_ROWS_PER_LENGTH),
    ]
    gate_command = [
        "python",
        "-m",
        (
            "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
            "gate_olmo2_length_gated_exact_screen"
        ),
        "--training-result",
        str(paths["run_output"] / "results.json"),
        "--evaluation-result",
        str(paths["candidate_eval_output"] / "results.json"),
        "--experiment-ready-receipt",
        str(paths["receipt_output"]),
        "--evaluation-examples",
        str(paths["candidate_eval_output"] / "examples.jsonl"),
        "--evaluation-run-manifest",
        str(paths["candidate_eval_output"] / "run_manifest.json"),
        "--output",
        str(paths["gate_output"]),
    ]
    return {
        "parent_exact_baseline": parent_evaluation_command,
        "training": training_command,
        "candidate_exact_screen": candidate_evaluation_command,
        "exact_gate": gate_command,
    }


def _registered_adapter_chain(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "load_order": [
            "frozen_parent_qkvo",
            "eos_vocab_row_child",
        ],
        "parent": {
            "path": str(paths["parent_adapter"]),
            "sha256": PARENT_ADAPTER_SHA256,
            "adaptation": PARENT_ADAPTATION,
            "rank": PARENT_RANK,
            "alpha": PARENT_ALPHA,
            "frozen_during_child_training": True,
        },
        "child": {
            "path": str(paths["run_output"] / "adapter.pt"),
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
    }


def main() -> None:
    args = parse_args()
    paths = {
        name: getattr(args, name).resolve()
        for name in (
            "checkpoint",
            "checkpoint_ready_receipt",
            "parent_adapter",
            "legacy_routing_data",
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
    for name in (
        "run_output",
        "parent_eval_output",
        "candidate_eval_output",
        "gate_output",
        "receipt_output",
    ):
        if paths[name].exists():
            raise FileExistsError(paths[name])
        if not paths[name].parent.is_dir():
            raise FileNotFoundError(paths[name].parent)
    output_targets = [
        paths[name]
        for name in (
            "run_output",
            "parent_eval_output",
            "candidate_eval_output",
            "gate_output",
            "receipt_output",
        )
    ]
    if len(set(output_targets)) != len(output_targets):
        raise RuntimeError("registered output paths must be pairwise distinct")

    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    allocator = (
        os.environ.get("PYTORCH_ALLOC_CONF")
        or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        or ""
    )
    if not cache or not Path(cache).resolve().is_dir():
        raise RuntimeError(
            "persistent TORCHINDUCTOR_CACHE_DIR must already exist"
        )
    if "expandable_segments:True" not in allocator:
        raise RuntimeError("expandable_segments allocator is required")
    output_filesystems: dict[str, dict[str, int]] = {}
    for parent in sorted(
        {
            path.parent
            for name, path in paths.items()
            if name
            in {
                "run_output",
                "parent_eval_output",
                "candidate_eval_output",
                "gate_output",
                "receipt_output",
            }
        },
        key=str,
    ):
        usage = shutil.disk_usage(parent)
        if int(usage.free) < MINIMUM_FREE_BYTES:
            raise RuntimeError(
                f"output filesystem has less than 20 GiB free: {parent}"
            )
        output_filesystems[str(parent)] = {
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
        }

    checkpoint_digest = ready_checkpoint_digest(
        paths["checkpoint"],
        paths["checkpoint_ready_receipt"],
    )
    if checkpoint_digest != MODEL_SHA256:
        raise RuntimeError("OLMo-2 1.485B checkpoint SHA drift")
    tokenizer = AutoTokenizer.from_pretrained(
        paths["checkpoint"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if (
        int(tokenizer.eos_token_id) != OLMO2_EOS_TOKEN_ID
        or int(tokenizer.bos_token_id) != OLMO2_EOS_TOKEN_ID
        or int(tokenizer.pad_token_id) != 100_277
    ):
        raise RuntimeError("OLMo-2 tokenizer special-token contract drift")

    module_root = Path(__file__).resolve().parent
    trainer = module_root / "train_4k_length_gated_eos.py"
    shared_trainer = module_root / "train_4k_counterfactual_routing.py"
    method = module_root / "olmo2_length_gated_method.py"
    evaluator = module_root / "evaluate_instruct_ruler_screen.py"
    gate = module_root / "gate_olmo2_length_gated_exact_screen.py"
    conversion = module_root.parent / "olmo2_lora_conversion.py"
    adapter_loader = module_root.parent / "olmo2_lora_ood_factorial.py"

    data_receipt, selected_rows = validate_data(
        paths["exact_eval_data"],
        paths["checkpoint"],
        "niah_single_1",
        EXACT_LENGTHS,
        EXACT_ROWS_PER_LENGTH,
    )
    if len(selected_rows) != len(EXACT_LENGTHS) * EXACT_ROWS_PER_LENGTH:
        raise RuntimeError("minimal exact screen row-count drift")
    actual_prompt_geometry = validate_actual_prompt_geometry(
        rows=selected_rows,
        tokenizer=tokenizer,
    )
    routing_receipt = validate_routing_upgrade(
        paths["legacy_routing_data"],
        paths["routing_data_v2"],
        tokenizer,
    )
    if (
        routing_receipt["legacy_manifest"]["sha256"]
        != LEGACY_ROUTING_MANIFEST_SHA256
    ):
        raise RuntimeError("legacy routing manifest identity drift")
    exact_source_hashes = {
        row_sha256(
            {
                key: value
                for key, value in row.items()
                if not key.startswith("_")
            }
        )
        for row in selected_rows
    }
    exact_answer_values = {
        str(value)
        for row in selected_rows
        for value in row["outputs"]
    }
    if (
        exact_source_hashes
        & set(routing_receipt["routing_source_row_sha256"])
        or exact_answer_values
        & set(routing_receipt["routing_gold_and_alternate_values"])
    ):
        raise RuntimeError(
            "minimal exact screen overlaps routing rows or answer values"
        )
    routing_receipt["exact_screen_disjointness"] = {
        "selected_exact_rows": len(selected_rows),
        "source_row_sha256_overlap": 0,
        "gold_or_alternate_value_overlap": 0,
    }
    routing_receipt["manifest_sha256"] = routing_receipt[
        "manifest"
    ]["sha256"]

    natural_path = (
        paths["prepared_data"] / "longalign_paired_L4096"
    )
    natural_receipt = validate_natural_view(natural_path)
    if int(natural_receipt["maximum_training_length"]) > LENGTH:
        raise RuntimeError("natural replay exceeds physical 4K")

    code_paths = {
        "preflight": Path(__file__).resolve(),
        "preflight_validation_helpers": (
            module_root / "preflight_4k_query_gap_eos_repair.py"
        ),
        "trainer": trainer,
        "shared_trainer": shared_trainer,
        "routing_data_contract": (
            module_root / "prepare_4k_routing_pairs.py"
        ),
        "training_primitives": module_root / "train_screen.py",
        "checkpoint_contract": module_root / "train_4k_stage_a.py",
        "method": method,
        "conversion": conversion,
        "model_loader_attention_dependency": (
            module_root.parent / "olmo2_1b_evq" / "train.py"
        ),
        "adapter_loader": adapter_loader,
        "shared_training_utils": (
            module_root.parent / "small_model_lora_conversion.py"
        ),
        "evq_contract": (
            module_root.parent / "olmo2_1b_evq" / "contract.py"
        ),
        "exact_evaluator": evaluator,
        "exact_gate": gate,
        "parent_raw_exact_validator": (
            module_root / "gate_olmo2_exact_screen.py"
        ),
    }
    code = {
        name: file_entry(path)
        for name, path in sorted(code_paths.items())
    }

    frozen_protocol = protocol()
    expected_protocol_identity = {
        "adaptation": ADAPTATION,
        "parent_adaptation": PARENT_ADAPTATION,
        "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
        "parent_rank": PARENT_RANK,
        "parent_alpha": PARENT_ALPHA,
        "child_rank": EOS_HEAD_RANK,
        "child_alpha": EOS_HEAD_ALPHA,
        "trainable_scope": (
            "long_only_eos_vocab_row_plus_scalar_bias"
        ),
    }
    for name, expected in expected_protocol_identity.items():
        if frozen_protocol.get(name) != expected:
            raise RuntimeError(
                f"EOS parent/child protocol drift for {name}"
            )
    registered_commands = _registered_commands(paths)
    registered_adapter_chain = _registered_adapter_chain(paths)
    registered_adapter_chain["code_sha256"] = {
        "candidate_evaluator": code["exact_evaluator"]["sha256"],
        "parent_adapter_loader": code["adapter_loader"]["sha256"],
        "parent_child_method": code["method"]["sha256"],
    }

    receipt = {
        "status": READY_STATUS,
        "objective": (
            "first measure the frozen parent under literal full-string+EOS; "
            "only if its complete answer prefix is already recoverable, freeze "
            "that QKVO parent and train only a long-branch EOS-vocabulary-row "
            "child while keeping <=4K on the untouched Native/base computation"
        ),
        "existing_evidence": (
            "The frozen full-EVQ query-gap parent recovered the first target "
            "number strongly, but that proxy is not strict generation success."
        ),
        "smallest_missing_evidence": (
            "Whether answer-plus-immediate-EOS continuation yields literal "
            "whole-string exact with terminal EOS at both 8K and 16K."
        ),
        "smallest_executable_plan": (
            "One frozen-parent n=8 raw prefix/EOS diagnostic; only after that "
            "gate, one physical-4K-only EOS-child continuation and one "
            "parent-plus-child n=8 exact screen."
        ),
        "stop_condition": (
            "Do not train if the parent lacks enough complete gold prefixes "
            "for the registered 8K/16K gate. If the candidate exact+EOS gate "
            "fails, do not run broad evaluation."
        ),
        "protocol": frozen_protocol,
        "method_contract": _method_contract(),
        "inputs": {
            "checkpoint": {
                "path": str(paths["checkpoint"]),
                "composite_sha256": checkpoint_digest,
                "ready_receipt": file_entry(
                    paths["checkpoint_ready_receipt"]
                ),
                "config": file_entry(
                    paths["checkpoint"] / "config.json"
                ),
                "model": file_entry(
                    paths["checkpoint"] / "model.safetensors"
                ),
                "tokenizer": file_entry(
                    paths["checkpoint"] / "tokenizer.json"
                ),
            },
            "checkpoint_ready_receipt": file_entry(
                paths["checkpoint_ready_receipt"]
            ),
            "parent_adapter": _parent_adapter_entry(
                paths["parent_adapter"]
            ),
            "legacy_routing_data": {
                "path": str(paths["legacy_routing_data"]),
                "manifest_sha256": routing_receipt[
                    "legacy_manifest"
                ]["sha256"],
            },
            "routing_data": routing_receipt,
            "prepared_data": {
                "path": str(paths["prepared_data"]),
                "natural_manifest_sha256": natural_receipt[
                    "manifest_sha256"
                ],
                "natural_replay": natural_receipt,
            },
            "exact_eval_data": {
                "path": str(paths["exact_eval_data"]),
                **data_receipt,
                "selected_rows": len(selected_rows),
                "actual_prompt_geometry": actual_prompt_geometry,
            },
        },
        "code": code,
        "evaluator": file_entry(evaluator),
        "evaluator_bound_code_sha256": evaluator_bound_code_sha256(),
        "run_output": str(paths["run_output"]),
        "registered_adapter_chain": registered_adapter_chain,
        "registered_parent_exact_baseline": {
            "output": str(paths["parent_eval_output"]),
            "result": str(
                paths["parent_eval_output"] / "results.json"
            ),
            "examples": str(
                paths["parent_eval_output"] / "examples.jsonl"
            ),
            "run_manifest": str(
                paths["parent_eval_output"] / "run_manifest.json"
            ),
            "adapter_sha256": PARENT_ADAPTER_SHA256,
            "adaptation": PARENT_ADAPTATION,
            "rank": PARENT_RANK,
            "alpha": PARENT_ALPHA,
            "evaluator_sha256": code["exact_evaluator"]["sha256"],
            "raw_validator_sha256": code[
                "parent_raw_exact_validator"
            ]["sha256"],
            "required_before_training": True,
        },
        "registered_outputs": {
            "parent_exact_baseline": str(paths["parent_eval_output"]),
            "candidate_exact_screen": str(
                paths["candidate_eval_output"]
            ),
        },
        "registered_commands": registered_commands,
        "required_execution_order": [
            "parent_exact_baseline",
            "training",
            "candidate_exact_screen",
            "exact_gate",
        ],
        "evaluation_contract": {
            "example_schema_version": EXAMPLE_SCHEMA_VERSION,
            "lengths": list(EXACT_LENGTHS),
            "rows_per_length": EXACT_ROWS_PER_LENGTH,
            "maximum_new_tokens": GENERATION_TOKENS,
            "actual_prompt_geometry": actual_prompt_geometry,
            "full_string_exact": (
                "literal decoded continuation equality; no normalization"
            ),
            "terminal_eos": "last generated token equals tokenizer EOS",
            "joint_success": "full_string_exact AND terminal_eos",
            "first_number_and_substring": "diagnostic_only",
            "candidate_adapter_load_order": [
                "frozen_parent_qkvo",
                "eos_vocab_row_child",
            ],
            "4k_preservation": (
                "exact Native/base computational branch; 4K task cell is "
                "diagnostic and is not forced to solve a new task"
            ),
        },
        "next_gate_if_exact_passes": (
            "Only then run the frozen broader 8K/16K capability matrix. "
            "The <=4K branch remains structurally identical by construction."
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "liger_kernel": importlib.metadata.version("liger-kernel"),
            "torchinductor_cache_dir": str(Path(cache).resolve()),
            "allocator": allocator,
            "output_filesystems": output_filesystems,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
        },
        "output_and_cleanup_contract": {
            "training_outputs": [
                "adapter.pt",
                "results.json",
                "train_log.jsonl",
            ],
            "optimizer_checkpoints": 0,
            "failed_run_path_suffix": ".incomplete",
            "failed_run_preserved_for_diagnosis": True,
            "remote_shutdown_authorized": False,
        },
        "authorization": (
            "READY is offline evidence only and does not authorize GPU use "
            "or remote access."
        ),
    }
    atomic_json(paths["receipt_output"], receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(paths["receipt_output"]),
                "receipt_sha256": sha256_file(
                    paths["receipt_output"]
                ),
                "protocol": receipt["protocol"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
