#!/usr/bin/env python3
"""Train the fixed OLMo-2 hybrid-EVQ exact-generation candidate.

This is a physical-4K-only training entry point. It starts from the untouched
OLMo-2 1.485B Instruct checkpoint, keeps Native RoPE pairs 0..51, replaces
only pairs 52..63 with EVQ-Cosh, and trains Q/K LoRA updates restricted to the
same rotary coordinates. V/O and all base parameters remain frozen.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
    load_model,
    save_adapter,
    trainable_named_parameters,
)
from rebuttal.rebuttal_0723.experiments.small_model_lora_conversion import (
    atomic_json,
    configure_cuda,
    seed_everything,
    sha256_file,
)

from .olmo2_exact_method import (
    HYBRID_FREQUENCY_NAME,
    HYBRID_FREQUENCY_SHA256,
    QK_EVQ_TAIL_OUTPUT_MASK_SHA256,
    apply_hybrid_evq_low12,
    phase_geometry_receipt,
    qk_evq_tail_output_mask,
)
from .prepare_4k_routing_pairs import (
    LENGTH,
    OLMO2_EOS_TOKEN_ID,
    ROOT_STATUS,
    SUPERVISION_CONTRACT,
)
from .train_4k_counterfactual_routing import (
    CALIBRATION_QUERY_MARKER,
    FAMILY_PATTERN,
    TRAIN_QUERY_MARKER,
    RoutingPairView,
    deterministic_query_offset_stream,
    evaluate_routing,
    train,
)
from .train_4k_stage_a import ready_checkpoint_digest


READY_STATUS = "OLMO2_4K_HYBRID_EXACT_READY_V1"
RESULT_STATUS = "OLMO2_4K_HYBRID_EXACT_TRAINING_COMPLETE_V1"
ADAPTATION = "qk_answer"
STEPS = 400
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
RANK = 32
ALPHA = 64.0
LEARNING_RATE = 5e-5
WARMUP_STEPS = 20
COUNTERFACTUAL_MARGIN = 1.0
COUNTERFACTUAL_MARGIN_WEIGHT = 0.5
TERMINATION_WEIGHT = 1.0
COMPILE_MODE = "max-autotune-no-cudagraphs"
VIRTUAL_TARGET_LENGTH = 4 * LENGTH
VIRTUAL_BUCKET_WEIGHTS = (1, 1, 2)
SEED = 20_260_726


def protocol() -> dict[str, Any]:
    routing_steps = sum(
        FAMILY_PATTERN[(step - 1) % len(FAMILY_PATTERN)] == "routing"
        for step in range(1, STEPS + 1)
    )
    offset_stream = deterministic_query_offset_stream(
        seed=SEED,
        routing_steps=routing_steps,
    )
    import hashlib

    return {
        "model_scope": "OLMo-2 1.485B Instruct only",
        "frequency": HYBRID_FREQUENCY_NAME,
        "frequency_sha256_float32": HYBRID_FREQUENCY_SHA256,
        "native_frequency_pairs": [0, 51],
        "evq_frequency_pairs": [52, 63],
        "adaptation": ADAPTATION,
        "qk_output_mask_sha256": QK_EVQ_TAIL_OUTPUT_MASK_SHA256,
        "qk_active_output_coordinates_per_head": 24,
        "v_and_o_trainable": False,
        "parent_adapter": None,
        "steps": STEPS,
        "family_pattern": list(FAMILY_PATTERN),
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "global_batch_size": (
            MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        ),
        "rank": RANK,
        "alpha": ALPHA,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "counterfactual_margin": COUNTERFACTUAL_MARGIN,
        "counterfactual_margin_weight": COUNTERFACTUAL_MARGIN_WEIGHT,
        "termination_weight": TERMINATION_WEIGHT,
        "compile_mode": COMPILE_MODE,
        "maximum_physical_training_sequence_length": LENGTH,
        "maximum_physical_model_input_tokens": LENGTH - 1,
        "hard_maximum_training_length": LENGTH,
        "real_8k_or_16k_training_sequences": 0,
        "position_policy": "semantic_query_block_continuous_gap",
        "virtual_target_length": VIRTUAL_TARGET_LENGTH,
        "maximum_realized_position_id": VIRTUAL_TARGET_LENGTH - 1,
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
        "natural_replay": "contiguous_physical_4k_full_token_ce",
        "expanded_evaluation_before_minimal_exact_gate": False,
        "seed": SEED,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--routing-data", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validate_ready(
    *,
    path: Path,
    checkpoint: Path,
    checkpoint_ready: Path,
    prepared_data: Path,
    routing_data: Path,
    output: Path,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != READY_STATUS:
        raise RuntimeError("hybrid-exact READY status drift")
    if receipt.get("protocol") != protocol():
        raise RuntimeError("hybrid-exact READY protocol drift")
    expected_paths = {
        "checkpoint": checkpoint,
        "checkpoint_ready_receipt": checkpoint_ready,
        "prepared_data": prepared_data,
        "routing_data": routing_data,
    }
    for name, expected in expected_paths.items():
        observed = Path(receipt["inputs"][name]["path"]).resolve()
        if observed != expected.resolve():
            raise RuntimeError(f"hybrid-exact READY {name} path drift")
    if Path(receipt["run_output"]).resolve() != output.resolve():
        raise RuntimeError("hybrid-exact READY output drift")
    code = receipt["code"]
    trainer = Path(__file__).resolve()
    expected_code = {
        "preflight": trainer.with_name("preflight_4k_hybrid_exact.py"),
        "trainer": trainer,
        "shared_trainer": trainer.with_name(
            "train_4k_counterfactual_routing.py"
        ),
        "method": trainer.with_name("olmo2_exact_method.py"),
        "conversion": trainer.parents[1] / "olmo2_lora_conversion.py",
        "exact_evaluator": trainer.with_name(
            "evaluate_instruct_ruler_screen.py"
        ),
        "exact_gate": trainer.with_name(
            "gate_olmo2_hybrid_exact_screen.py"
        ),
    }
    for name, expected in expected_code.items():
        if (
            Path(code[name]["path"]).resolve() != expected
            or code[name]["sha256"] != sha256_file(expected)
        ):
            raise RuntimeError(
                f"hybrid-exact {name} changed after READY"
            )
    if (
        receipt["inputs"]["checkpoint_ready_receipt"]["sha256"]
        != sha256_file(checkpoint_ready)
    ):
        raise RuntimeError("checkpoint READY receipt changed")
    if (
        receipt["inputs"]["routing_data"]["manifest_sha256"]
        != sha256_file(routing_data / "manifest.json")
    ):
        raise RuntimeError("routing data changed after READY")
    natural_manifest = (
        prepared_data / "longalign_paired_L4096" / "manifest.json"
    )
    if (
        receipt["inputs"]["prepared_data"]["natural_manifest_sha256"]
        != sha256_file(natural_manifest)
    ):
        raise RuntimeError("natural replay changed after READY")
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


def _assert_trainable_scope(model: Any) -> dict[str, Any]:
    named = trainable_named_parameters(model, None)
    names = [name for name, _ in named]
    if not names:
        raise RuntimeError("hybrid-exact adapter has no trainable parameters")
    if any(
        ".q_proj." not in name and ".k_proj." not in name
        for name in names
    ):
        raise RuntimeError("hybrid-exact trainable scope escaped Q/K")
    if any(".v_proj." in name or ".o_proj." in name for name in names):
        raise RuntimeError("hybrid-exact V/O projection became trainable")
    return {
        "parameter_names": names,
        "parameter_tensors": len(names),
        "parameters": int(
            sum(parameter.numel() for _, parameter in named)
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output)
    checkpoint = args.checkpoint.resolve()
    checkpoint_ready = args.checkpoint_ready_receipt.resolve()
    prepared_data = args.prepared_data.resolve()
    routing_root = args.routing_data.resolve()
    ready_path = args.ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint,
        checkpoint_ready,
    )
    ready = _validate_ready(
        path=ready_path,
        checkpoint=checkpoint,
        checkpoint_ready=checkpoint_ready,
        prepared_data=prepared_data,
        routing_data=routing_root,
        output=output,
    )
    if ready["inputs"]["checkpoint"]["digest"] != checkpoint_digest:
        raise RuntimeError("hybrid-exact checkpoint digest drift")

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
    ):
        raise RuntimeError("hybrid-exact routing collection contract drift")

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
    incomplete.mkdir(parents=True)
    seed_everything(SEED)
    runtime = configure_cuda()
    model = load_model(checkpoint)
    frequency = apply_hybrid_evq_low12(model)
    qk_mask = qk_evq_tail_output_mask(model.config)
    if tensor_sha256(qk_mask) != QK_EVQ_TAIL_OUTPUT_MASK_SHA256:
        raise RuntimeError("hybrid-exact Q/K mask drift")
    readout = install_adaptation(
        model,
        ADAPTATION,
        rank=RANK,
        alpha=ALPHA,
        qk_output_mask=qk_mask,
    )
    if readout is not None:
        raise RuntimeError("hybrid-exact method forbids a readout")
    trainable = _assert_trainable_scope(model)
    model.to("cuda")

    initial_contiguous = evaluate_routing(
        model=model,
        view=calibration_view,
        rows=16,
        pair_batch_size=2,
        margin=COUNTERFACTUAL_MARGIN,
    )
    initial_virtual = {
        f"query_offset_{offset}": evaluate_routing(
            model=model,
            view=calibration_view,
            rows=16,
            pair_batch_size=2,
            margin=COUNTERFACTUAL_MARGIN,
            query_offset=offset,
        )
        for offset in (LENGTH, 2 * LENGTH, 3 * LENGTH + 1)
    }
    training = train(
        model=model,
        routing_view=routing_view,
        calibration_view=calibration_view,
        natural_view_path=(
            prepared_data / "longalign_paired_L4096"
        ),
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
    )
    final_contiguous = evaluate_routing(
        model=model,
        view=calibration_view,
        rows=len(calibration_view.input_ids),
        pair_batch_size=2,
        margin=COUNTERFACTUAL_MARGIN,
    )
    final_virtual = {
        f"query_offset_{offset}": evaluate_routing(
            model=model,
            view=calibration_view,
            rows=len(calibration_view.input_ids),
            pair_batch_size=2,
            margin=COUNTERFACTUAL_MARGIN,
            query_offset=offset,
        )
        for offset in (LENGTH, 2 * LENGTH, 3 * LENGTH + 1)
    }

    adapter_metadata = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": HYBRID_FREQUENCY_NAME,
        "frequency_sha256_float32": HYBRID_FREQUENCY_SHA256,
        "adaptation": ADAPTATION,
        "adaptation_description": (
            f"qk_evq_tail_r{RANK}_alpha{ALPHA:g}"
        ),
        "qk_output_mask_sha256": QK_EVQ_TAIL_OUTPUT_MASK_SHA256,
        "rank": RANK,
        "alpha": ALPHA,
        "training_sequence_length": LENGTH,
        "maximum_physical_training_sequence_length": LENGTH,
        "virtual_target_length": VIRTUAL_TARGET_LENGTH,
        "position_policy": "semantic_query_block_continuous_gap",
        "final_eos_supervised": True,
        "supervision_contract": SUPERVISION_CONTRACT,
        "parent_adapter_sha256": None,
        "seed": SEED,
    }
    adapter_sha = save_adapter(
        incomplete / "adapter.pt",
        model,
        None,
        adapter_metadata,
    )
    if adapter_sha is None:
        raise RuntimeError("hybrid-exact adapter was not written")
    receipt = {
        "status": RESULT_STATUS,
        "metric_boundary": (
            "Training and teacher-forced routing calibration only. This is "
            "not strict autoregressive capability and not 4K retention "
            "evidence; the bound minimal exact+EOS screen must run next."
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "ready_receipt_sha256": sha256_file(ready_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "bound_code": {
            name: entry["sha256"]
            for name, entry in ready["code"].items()
        },
        "frequency": frequency,
        "frequency_geometry": phase_geometry_receipt(),
        "qk_output_mask_sha256": QK_EVQ_TAIL_OUTPUT_MASK_SHA256,
        "trainable_scope": trainable,
        "adapter_sha256": adapter_sha,
        "adapter_metadata": adapter_metadata,
        "routing_data": {
            "path": str(routing_root),
            "manifest_sha256": sha256_file(
                routing_root / "manifest.json"
            ),
            "format_version": 2,
            "final_eos_supervised": True,
            "labels_only_cover_answer_and_final_eos": True,
            "answer_string_tokenizer_roundtrip_exact": True,
        },
        "protocol": protocol(),
        "runtime": {
            **runtime,
            "compile_cache": cache,
            "allocator": allocator,
        },
        "initial_routing_calibration": initial_contiguous,
        "initial_virtual_routing_calibration": initial_virtual,
        "training": training,
        "final_routing_calibration": final_contiguous,
        "final_virtual_routing_calibration": final_virtual,
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
                "expanded_evaluation_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
