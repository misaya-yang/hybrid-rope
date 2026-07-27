#!/usr/bin/env python3
"""Create the no-GPU READY receipt for OLMo-2 query-gap EOS repair."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

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
    validate_data,
)
from .prepare_4k_routing_pairs import (
    LENGTH,
    OLMO2_EOS_TOKEN_ID,
    ROOT_STATUS,
    SUPERVISION_CONTRACT,
)
from .gate_olmo2_exact_screen import (
    bound_code_sha256 as gate_bound_code_sha256,
)
from .train_4k_counterfactual_routing import (
    CALIBRATION_QUERY_MARKER,
    TRAIN_QUERY_MARKER,
    VIRTUAL_QUERY_GAP_READY_STATUS,
    RoutingPairView,
    bound_code_sha256,
    registered_protocol,
)
from .train_4k_stage_a import ready_checkpoint_digest


MODEL_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
PARENT_QUERY_GAP_ADAPTER_SHA256 = (
    "a0ccd2cf141300ba4489882dda1324b2f237e65444a9a71d687e8c5fad57ae8b"
)
PRE_QUERY_GAP_PARENT_SHA256 = (
    "95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a"
)
LEGACY_ROUTING_MANIFEST_SHA256 = (
    "83a745b25893a73749dd85ff66dea5b305be6e4e6dcf16a5d0e13cae935ec63b"
)
STEPS = 32
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
RANK = 64
ALPHA = 128.0
LEARNING_RATE = 1e-5
WARMUP_STEPS = 4
COUNTERFACTUAL_MARGIN = 1.0
COUNTERFACTUAL_MARGIN_WEIGHT = 0.5
TERMINATION_WEIGHT = 1.0
COMPILE_MODE = "max-autotune-no-cudagraphs"
NATURAL_EVAL_ROWS = 16
VIRTUAL_TARGET_LENGTH = 16_384
VIRTUAL_BUCKET_WEIGHTS = (1, 1, 2)
SEED = 20_260_728
EXACT_LENGTHS = (4_096, 8_192, 16_384)
EXACT_ROWS_PER_LENGTH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--legacy-routing-data", type=Path, required=True)
    parser.add_argument("--routing-data-v2", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--exact-eval-data", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--parent-eval-output", type=Path, required=True)
    parser.add_argument("--candidate-eval-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args()


def file_entry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def adapter_entry(path: Path) -> dict[str, Any]:
    entry = file_entry(path)
    if entry["sha256"] != PARENT_QUERY_GAP_ADAPTER_SHA256:
        raise RuntimeError("query-gap repair parent adapter SHA drift")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("state"), dict)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise RuntimeError("query-gap parent adapter payload drift")
    metadata = payload["metadata"]
    expected = {
        "base_checkpoint_sha256": MODEL_SHA256,
        "frequency": "evq",
        "adaptation": "qkvo_answer",
        "rank": RANK,
        "alpha": ALPHA,
        "training_sequence_length": LENGTH,
        "stage": "counterfactual_routing_semantic_query_gap_16k",
        "parent_adapter_sha256": PRE_QUERY_GAP_PARENT_SHA256,
        "position_policy": "semantic_query_block_continuous_gap",
        "virtual_target_length": VIRTUAL_TARGET_LENGTH,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(
                f"query-gap parent metadata drift for {name}"
            )
    entry["metadata"] = metadata
    entry["state_tensors"] = len(payload["state"])
    return entry


def validate_routing_upgrade(
    legacy_root: Path,
    v2_root: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    legacy_manifest_path = legacy_root / "manifest.json"
    v2_manifest_path = v2_root / "manifest.json"
    if sha256_file(legacy_manifest_path) != LEGACY_ROUTING_MANIFEST_SHA256:
        raise RuntimeError("legacy routing manifest SHA drift")
    legacy = json.loads(
        legacy_manifest_path.read_text(encoding="utf-8")
    )
    v2 = json.loads(v2_manifest_path.read_text(encoding="utf-8"))
    if (
        int(legacy.get("format_version", -1)) != 1
        or legacy.get("status")
        != "OLMO2_4K_COUNTERFACTUAL_ROUTING_DATA_PREPARED"
    ):
        raise RuntimeError("legacy routing-data contract drift")
    if (
        int(v2.get("format_version", -1)) != 2
        or v2.get("status") != ROOT_STATUS
        or v2.get("supervision_contract") != SUPERVISION_CONTRACT
        or int(v2.get("eos_token_id", -1)) != OLMO2_EOS_TOKEN_ID
        or v2.get("final_eos_supervised") is not True
        or int(v2.get("hard_maximum_training_length", -1)) != LENGTH
        or int(v2.get("hard_maximum_training_position_id", -1))
        != LENGTH - 1
    ):
        raise RuntimeError("routing v2 contract drift")

    splits: dict[str, Any] = {}
    routing_source_hashes: set[str] = set()
    routing_answer_values: set[str] = set()
    for name, marker_text in (
        ("train", TRAIN_QUERY_MARKER),
        ("calibration", CALIBRATION_QUERY_MARKER),
    ):
        legacy_set = legacy_root / name
        v2_set = v2_root / name
        legacy_set_manifest_path = legacy_set / "manifest.json"
        v2_set_manifest_path = v2_set / "manifest.json"
        legacy_set_manifest = json.loads(
            legacy_set_manifest_path.read_text(encoding="utf-8")
        )
        v2_set_manifest = json.loads(
            v2_set_manifest_path.read_text(encoding="utf-8")
        )
        if (
            sha256_file(legacy_set_manifest_path)
            != legacy["sets"][name]["manifest_sha256"]
            or sha256_file(v2_set_manifest_path)
            != v2["sets"][name]["manifest_sha256"]
        ):
            raise RuntimeError(
                f"{name} routing root/set manifest chain drift"
            )
        legacy_files = {
            filename: file_entry(legacy_set / filename)
            for filename in (
                "input_ids.npy",
                "labels.npy",
                "rows.jsonl",
            )
        }
        v2_files = {
            filename: file_entry(v2_set / filename)
            for filename in (
                "input_ids.npy",
                "labels.npy",
                "rows.jsonl",
            )
        }
        for filename in legacy_files:
            if (
                legacy_files[filename]["sha256"]
                != legacy_set_manifest["files"][filename]
                or v2_files[filename]["sha256"]
                != v2_set_manifest["files"][filename]
            ):
                raise RuntimeError(
                    f"{name} routing file/manifest hash drift: {filename}"
                )
        legacy_input_sha = legacy_set_manifest["files"][
            "input_ids.npy"
        ]
        v2_input_sha = v2_set_manifest["files"]["input_ids.npy"]
        if legacy_input_sha != v2_input_sha:
            raise RuntimeError(
                f"{name} routing input IDs changed during EOS upgrade"
            )
        if (
            legacy_set_manifest["files"]["labels.npy"]
            == v2_set_manifest["files"]["labels.npy"]
        ):
            raise RuntimeError(
                f"{name} routing labels did not change for EOS supervision"
            )
        legacy_labels = np.load(
            legacy_set / "labels.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        v2_labels = np.load(
            v2_set / "labels.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        v2_inputs = np.load(
            v2_set / "input_ids.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            legacy_labels.shape != v2_labels.shape
            or v2_inputs.shape != v2_labels.shape
        ):
            raise RuntimeError(f"{name} EOS-upgrade tensor shape drift")
        legacy_supervised = legacy_labels != -100
        if not np.array_equal(
            v2_labels[legacy_supervised],
            legacy_labels[legacy_supervised],
        ):
            raise RuntimeError(
                f"{name} legacy answer supervision changed"
            )
        added = (~legacy_supervised) & (v2_labels != -100)
        expected_added = int(v2_labels.shape[0] * v2_labels.shape[1])
        if (
            int(added.sum()) != expected_added
            or not np.all(v2_labels[added] == OLMO2_EOS_TOKEN_ID)
            or not np.all(v2_inputs[added] == OLMO2_EOS_TOKEN_ID)
            or np.any((legacy_labels != v2_labels) & ~added)
        ):
            raise RuntimeError(
                f"{name} EOS is not the unique added supervision token"
            )
        marker = tuple(
            int(value)
            for value in tokenizer(
                marker_text, add_special_tokens=False
            ).input_ids
        )
        view = RoutingPairView(
            v2_set,
            require_virtual_geometry=True,
            query_marker_token_ids=marker,
        )
        if not view.virtual_geometry_ready:
            raise RuntimeError(f"{name} virtual geometry is unavailable")
        metadata_rows = [
            json.loads(line)
            for line in (v2_set / "rows.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if len(metadata_rows) != len(view.rows):
            raise RuntimeError(f"{name} routing metadata row-count drift")
        routing_source_hashes.update(
            str(row["source_row_sha256"]) for row in metadata_rows
        )
        for row in metadata_rows:
            routing_answer_values.add(str(row["gold_value"]))
            routing_answer_values.add(str(row["alternate_value"]))
        splits[name] = {
            "rows": len(view.rows),
            "input_ids_sha256": v2_input_sha,
            "legacy_labels_sha256": legacy_set_manifest["files"][
                "labels.npy"
            ],
            "v2_labels_sha256": v2_set_manifest["files"][
                "labels.npy"
            ],
            "legacy_supervised_tokens_preserved": int(
                legacy_supervised.sum()
            ),
            "unique_added_supervision": "one_immediate_eos_per_variant",
            "added_eos_tokens": int(added.sum()),
            "manifest": file_entry(v2_set / "manifest.json"),
            "legacy_manifest": file_entry(
                legacy_set / "manifest.json"
            ),
            "legacy_files": legacy_files,
            "files": v2_files,
        }
    return {
        "path": str(v2_root),
        "manifest": file_entry(v2_manifest_path),
        "legacy_path": str(legacy_root),
        "legacy_manifest": file_entry(legacy_manifest_path),
        "only_supervision_contract_changed": True,
        "routing_source_row_sha256": sorted(routing_source_hashes),
        "routing_gold_and_alternate_values": sorted(
            routing_answer_values
        ),
        "splits": splits,
    }


def validate_natural_view(path: Path) -> dict[str, Any]:
    required = (
        "manifest.json",
        "input_ids.npy",
        "assistant_mask.npy",
        "lengths.npy",
        "split.npy",
    )
    files = {
        name: file_entry(path / name)
        for name in required
    }
    input_ids = np.load(
        path / "input_ids.npy", mmap_mode="r", allow_pickle=False
    )
    assistant_mask = np.load(
        path / "assistant_mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    lengths = np.load(
        path / "lengths.npy", mmap_mode="r", allow_pickle=False
    )
    split = np.load(
        path / "split.npy", mmap_mode="r", allow_pickle=False
    )
    if (
        input_ids.shape != assistant_mask.shape
        or input_ids.shape[1] != LENGTH
        or lengths.shape != (input_ids.shape[0],)
        or split.shape != (input_ids.shape[0],)
        or np.any(lengths <= 1)
        or np.any(lengths > LENGTH)
        or not np.any(split == 0)
    ):
        raise RuntimeError("natural replay violates the physical 4K contract")
    return {
        "path": str(path),
        "manifest_sha256": files["manifest.json"]["sha256"],
        "files": files,
        "shape": [int(value) for value in input_ids.shape],
        "training_rows": int((split == 0).sum()),
        "maximum_training_length": int(lengths[split == 0].max()),
    }


def protocol() -> dict[str, Any]:
    return registered_protocol(
        argparse.Namespace(
            frequency="evq",
            steps=STEPS,
            micro_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            rank=RANK,
            alpha=ALPHA,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            counterfactual_margin=COUNTERFACTUAL_MARGIN,
            counterfactual_margin_weight=COUNTERFACTUAL_MARGIN_WEIGHT,
            termination_weight=TERMINATION_WEIGHT,
            compile_mode=COMPILE_MODE,
            natural_eval_rows=NATURAL_EVAL_ROWS,
            virtual_target_length=VIRTUAL_TARGET_LENGTH,
            virtual_bucket_weights=VIRTUAL_BUCKET_WEIGHTS,
            seed=SEED,
        ),
        virtual_query_gap=True,
    )


def main() -> None:
    args = parse_args()
    paths = {
        name: getattr(args, name).resolve()
        for name in (
            "checkpoint",
            "ready_receipt",
            "parent_adapter",
            "legacy_routing_data",
            "routing_data_v2",
            "prepared_data",
            "background_dir",
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

    checkpoint_digest = ready_checkpoint_digest(
        paths["checkpoint"], paths["ready_receipt"]
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
    trainer = module_root / "train_4k_counterfactual_routing.py"
    evaluator = module_root / "evaluate_instruct_ruler_screen.py"
    gate = module_root / "gate_olmo2_exact_screen.py"
    conversion = module_root.parent / "olmo2_lora_conversion.py"
    data_receipt, selected_rows = validate_data(
        paths["exact_eval_data"],
        paths["checkpoint"],
        "niah_single_1",
        EXACT_LENGTHS,
        EXACT_ROWS_PER_LENGTH,
    )
    if len(selected_rows) != len(EXACT_LENGTHS) * EXACT_ROWS_PER_LENGTH:
        raise RuntimeError("minimal exact screen row-count drift")
    routing_receipt = validate_routing_upgrade(
        paths["legacy_routing_data"],
        paths["routing_data_v2"],
        tokenizer,
    )
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
    source_overlap = exact_source_hashes & set(
        routing_receipt["routing_source_row_sha256"]
    )
    value_overlap = exact_answer_values & set(
        routing_receipt["routing_gold_and_alternate_values"]
    )
    if source_overlap or value_overlap:
        raise RuntimeError(
            "minimal exact screen overlaps routing rows or answer values"
        )
    routing_receipt["exact_screen_disjointness"] = {
        "selected_exact_rows": len(selected_rows),
        "source_row_sha256_overlap": 0,
        "gold_or_alternate_value_overlap": 0,
    }
    natural_view = (
        paths["prepared_data"] / "longalign_paired_L4096"
    )
    background_manifest = paths["background_dir"] / "manifest.json"
    if not background_manifest.is_file():
        raise FileNotFoundError(background_manifest)
    background_files = {
        filename: file_entry(paths["background_dir"] / filename)
        for filename in (
            "manifest.json",
            "documents_L16384.npy",
            "documents_L16384.metadata.json",
        )
    }

    frozen_protocol = protocol()
    training_argv = [
        sys.executable,
        "-m",
        (
            "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
            "train_4k_counterfactual_routing"
        ),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--parent-adapter",
        str(paths["parent_adapter"]),
        "--prepared-data",
        str(paths["prepared_data"]),
        "--routing-data",
        str(paths["routing_data_v2"]),
        "--background-dir",
        str(paths["background_dir"]),
        "--ready-receipt",
        str(paths["ready_receipt"]),
        "--experiment-ready-receipt",
        str(paths["receipt_output"]),
        "--parent-exact-baseline-result",
        str(paths["parent_eval_output"] / "results.json"),
        "--parent-exact-baseline-examples",
        str(paths["parent_eval_output"] / "examples.jsonl"),
        "--parent-exact-baseline-run-manifest",
        str(paths["parent_eval_output"] / "run_manifest.json"),
        "--output",
        str(paths["run_output"]),
        "--frequency",
        "evq",
        "--steps",
        str(STEPS),
        "--micro-batch-size",
        str(MICRO_BATCH_SIZE),
        "--gradient-accumulation-steps",
        str(GRADIENT_ACCUMULATION_STEPS),
        "--rank",
        str(RANK),
        "--alpha",
        str(ALPHA),
        "--learning-rate",
        str(LEARNING_RATE),
        "--warmup-steps",
        str(WARMUP_STEPS),
        "--counterfactual-margin",
        str(COUNTERFACTUAL_MARGIN),
        "--counterfactual-margin-weight",
        str(COUNTERFACTUAL_MARGIN_WEIGHT),
        "--termination-weight",
        str(TERMINATION_WEIGHT),
        "--compile-mode",
        COMPILE_MODE,
        "--natural-eval-rows",
        str(NATURAL_EVAL_ROWS),
        "--virtual-target-length",
        str(VIRTUAL_TARGET_LENGTH),
        "--virtual-bucket-weights",
        *(str(value) for value in VIRTUAL_BUCKET_WEIGHTS),
        "--seed",
        str(SEED),
    ]

    def evaluation_argv(adapter: Path, output: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            (
                "rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity."
                "evaluate_instruct_ruler_screen"
            ),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--ready-receipt",
            str(paths["ready_receipt"]),
            "--experiment-ready-receipt",
            str(paths["receipt_output"]),
            "--data-root",
            str(paths["exact_eval_data"]),
            "--output",
            str(output),
            "--frequency",
            "evq",
            "--task",
            "niah_single_1",
            "--adapter",
            str(adapter),
            "--rank",
            str(RANK),
            "--alpha",
            str(ALPHA),
            "--lengths",
            *(str(value) for value in EXACT_LENGTHS),
            "--limit-per-length",
            str(EXACT_ROWS_PER_LENGTH),
        ]

    receipt = {
        "status": VIRTUAL_QUERY_GAP_READY_STATUS,
        "objective": (
            "repair immediate terminal EOS with a low-disturbance continuation "
            "while preserving and independently remeasuring the existing "
            "query-gap retrieval behavior"
        ),
        "existing_evidence": (
            "The frozen parent adapter has strong first-number extraction "
            "but zero recovered literal full-string exact because generation "
            "usually continues after the correct number."
        ),
        "smallest_missing_evidence": (
            "Whether explicit answer-plus-immediate-EOS supervision converts "
            "that retrieval into literal full-string exact plus terminal EOS."
        ),
        "smallest_executable_plan": (
            "One 32-step low-LR continuation from the frozen successful "
            "query-gap adapter, followed by one disjoint n=8 exact gate."
        ),
        "stop_condition": (
            "If the n=8 literal full-string-plus-EOS gate fails, do not run "
            "the full n=100 or broad retention matrix; diagnose the failed "
            "component first."
        ),
        "protocol": frozen_protocol,
        "bound_code_sha256": bound_code_sha256(),
        "evaluator_bound_code_sha256": evaluator_bound_code_sha256(),
        "gate_bound_code_sha256": gate_bound_code_sha256(),
        "inputs": {
            "checkpoint": {
                "path": str(paths["checkpoint"]),
                "composite_sha256": checkpoint_digest,
                "ready_receipt": file_entry(paths["ready_receipt"]),
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
            "parent_adapter": adapter_entry(paths["parent_adapter"]),
            "routing_data": routing_receipt,
            "natural_replay": validate_natural_view(natural_view),
            "background": {
                "path": str(paths["background_dir"]),
                "manifest": background_files["manifest.json"],
                "files": background_files,
            },
            "exact_eval_data": {
                "path": str(paths["exact_eval_data"]),
                **data_receipt,
                "selected_rows": len(selected_rows),
            },
            "conversion": file_entry(conversion),
        },
        "trainer": file_entry(trainer),
        "evaluator": file_entry(evaluator),
        "gate": file_entry(gate),
        "run_output": str(paths["run_output"]),
        "registered_working_directory": str(
            Path(__file__).resolve().parents[4]
        ),
        "registered_outputs": {
            "parent_exact_baseline": str(paths["parent_eval_output"]),
            "candidate_exact_screen": str(paths["candidate_eval_output"]),
        },
        "registered_commands": {
            "parent_exact_baseline": evaluation_argv(
                paths["parent_adapter"],
                paths["parent_eval_output"],
            ),
            "training": training_argv,
            "candidate_exact_screen": evaluation_argv(
                paths["run_output"] / "adapter.pt",
                paths["candidate_eval_output"],
            ),
            "exact_gate": [
                sys.executable,
                "-m",
                (
                    "rebuttal.rebuttal_0723.experiments."
                    "olmo2_lora_maturity.gate_olmo2_exact_screen"
                ),
                "--training-result",
                str(paths["run_output"] / "results.json"),
                "--experiment-ready-receipt",
                str(paths["receipt_output"]),
                "--parent-evaluation-result",
                str(paths["parent_eval_output"] / "results.json"),
                "--parent-evaluation-examples",
                str(paths["parent_eval_output"] / "examples.jsonl"),
                "--parent-evaluation-run-manifest",
                str(paths["parent_eval_output"] / "run_manifest.json"),
                "--evaluation-result",
                str(paths["candidate_eval_output"] / "results.json"),
                "--evaluation-examples",
                str(paths["candidate_eval_output"] / "examples.jsonl"),
                "--evaluation-run-manifest",
                str(paths["candidate_eval_output"] / "run_manifest.json"),
                "--expected-parent-sha256",
                PARENT_QUERY_GAP_ADAPTER_SHA256,
                "--output",
                str(paths["gate_output"]),
            ],
        },
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
            "full_string_exact": (
                "literal decoded continuation equality; no normalization"
            ),
            "terminal_eos": "last generated token equals tokenizer EOS",
            "joint_success": "full_string_exact AND terminal_eos",
            "first_number_and_substring": "diagnostic_only",
        },
        "next_gate_if_exact_passes": (
            "Compare the repaired adapter against both a0ccd2... and "
            "95ceeb... on the frozen 13-task 4K matrix and natural NLL; "
            "do not call this READY receipt a no-forgetting result."
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "liger_kernel": importlib.metadata.version("liger-kernel"),
        },
        "authorization": (
            "READY is offline evidence only and does not authorize GPU use."
        ),
    }
    atomic_json(paths["receipt_output"], receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(paths["receipt_output"]),
                "receipt_sha256": sha256_file(paths["receipt_output"]),
                "protocol": frozen_protocol,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
