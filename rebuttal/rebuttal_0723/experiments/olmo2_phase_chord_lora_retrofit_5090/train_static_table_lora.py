#!/usr/bin/env python3
"""Matched three-arm static-table QKVO-LoRA retrofit for OLMo-2.

All arms use stock Hugging Face OLMo attention/rotary/cache and stock PEFT
rank-64 QKVO LoRA.  Only the global static inverse-frequency table differs.
EVQ and phase-chord move from Native to their frozen target during the first
60 optimizer steps using log-space smoothstep; inference uses the exact target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .receipts import (
    ARMS,
    COMPLETE_STATUS,
    GPU_READY_STATUS,
    assert_peft_state_roundtrip,
    atomic_json,
    checkpoint_receipt,
    load_standard_peft_bundle,
    require_gpu_authorization,
    save_standard_peft_bundle,
    sha256_file,
    smoothstep_log_morph,
    tensor_bundle_sha256,
    validate_ready_receipt,
    write_ready_receipt,
)


DATA_STATUS = "OLMO2_PHASE_CHORD_CONTEXT_DATA_PREPARED_V1"
TEACHER_STATUS = "OLMO2_PHASE_CHORD_NATIVE_TEACHER_PRECOMPUTED_V1"
LOCKED_STEPS = 300
LOCKED_MORPH_STEPS = 60
LOCKED_RANK = 64
LOCKED_ALPHA = 128.0
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        path.name: sha256_file(path)
        for path in sorted(root.glob("*.py"))
    }


def _manifest_receipt(root: Path, expected_status: str) -> dict[str, Any]:
    root = root.resolve()
    path = root / "manifest.json"
    manifest = _json(path)
    if manifest.get("status") != expected_status:
        raise RuntimeError(f"manifest status drift: {path}")
    hashes = manifest.get("output_hashes", manifest.get("files"))
    if not isinstance(hashes, dict):
        raise RuntimeError(f"manifest has no output hashes: {path}")
    verified: dict[str, dict[str, Any]] = {}
    for name, record in hashes.items():
        candidate = root / str(name)
        expected = record.get("sha256") if isinstance(record, dict) else record
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise RuntimeError(f"manifest-bound file drift: {candidate}")
        verified[str(name)] = {
            "sha256": str(expected),
            "bytes": int(candidate.stat().st_size),
        }
    return {
        "root": str(root),
        "manifest_sha256": sha256_file(path),
        "manifest": manifest,
        "files": verified,
    }


def _target_contract(
    correct: np.ndarray,
    swapped: np.ndarray,
    correct_mask: np.ndarray,
    swapped_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        correct.ndim != 2
        or swapped.shape != correct.shape
        or correct_mask.shape != correct.shape
        or swapped_mask.shape != correct.shape
        or not np.array_equal(correct_mask, swapped_mask)
    ):
        raise RuntimeError("correct/swapped target contract drift")
    counts = correct_mask.sum(axis=1)
    if len(counts) == 0 or np.any(counts != counts[0]) or int(counts[0]) <= 0:
        raise RuntimeError("target masks must have one fixed positive width")
    positions = np.stack(
        [np.flatnonzero(row) for row in correct_mask], axis=0
    ).astype(np.int64)
    if np.any(positions < 1):
        raise RuntimeError("target position zero violates p-1 prediction")
    if positions.shape[1] > 1 and not np.array_equal(
        positions[:, 1:], positions[:, :-1] + 1
    ):
        raise RuntimeError("target positions must be contiguous")
    tokens = np.take_along_axis(correct, positions, axis=1).astype(np.int64)
    if not np.array_equal(tokens, np.take_along_axis(swapped, positions, axis=1)):
        raise RuntimeError("correct/swapped target token drift")
    return positions, tokens


class PhysicalPairView:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.receipt = _manifest_receipt(self.root, DATA_STATUS)
        manifest = self.receipt["manifest"]
        self.correct = np.load(
            self.root / "correct_input_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.swapped = np.load(
            self.root / "swapped_input_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.correct_mask = np.load(
            self.root / "correct_target_mask.npy", mmap_mode="r", allow_pickle=False
        )
        self.swapped_mask = np.load(
            self.root / "swapped_target_mask.npy", mmap_mode="r", allow_pickle=False
        )
        self.short_correct = np.load(
            self.root / "short_correct_input_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.short_swapped = np.load(
            self.root / "short_swapped_input_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.short_correct_mask = np.load(
            self.root / "short_correct_target_mask.npy", mmap_mode="r", allow_pickle=False
        )
        self.short_swapped_mask = np.load(
            self.root / "short_swapped_target_mask.npy", mmap_mode="r", allow_pickle=False
        )
        self.positions, self.tokens = _target_contract(
            self.correct, self.swapped, self.correct_mask, self.swapped_mask
        )
        self.short_positions, short_tokens = _target_contract(
            self.short_correct,
            self.short_swapped,
            self.short_correct_mask,
            self.short_swapped_mask,
        )
        if not np.array_equal(short_tokens, self.tokens):
            raise RuntimeError("short/physical target token identity drift")
        expected = (int(manifest["examples"]), int(manifest["length"]))
        if tuple(self.correct.shape) != expected:
            raise RuntimeError("physical data shape drift")


class TeacherAssets:
    def __init__(self, root: Path, data_manifest_sha256: str) -> None:
        self.root = root.resolve()
        self.receipt = _manifest_receipt(self.root, TEACHER_STATUS)
        manifest = self.receipt["manifest"]
        if manifest.get("data_manifest_sha256") != data_manifest_sha256:
            raise RuntimeError("teacher/data manifest binding drift")
        self.example_indices = np.load(
            self.root / "example_indices.npy", allow_pickle=False
        ).astype(np.int64)
        self.target_tokens = np.load(
            self.root / "target_token_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.teacher_delta = np.load(
            self.root / "teacher_delta_logprob.npy", mmap_mode="r", allow_pickle=False
        )
        self.short_pool_source_rows = np.load(
            self.root / "short_pool_source_rows.npy", allow_pickle=False
        ).astype(np.int64)
        self.eligible_indices = np.load(
            self.root / "eligible_indices.npy", allow_pickle=False
        ).astype(np.int64)
        self.short_pool_split = np.load(
            self.root / "short_pool_split.npy", allow_pickle=False
        ).astype(np.int8)
        self.short_target_positions = np.load(
            self.root / "short_target_positions.npy", allow_pickle=False
        ).astype(np.int64)
        self.short_gold_logprob = np.load(
            self.root / "short_gold_logprob.npy", mmap_mode="r", allow_pickle=False
        )
        self.short_hidden_positions = np.load(
            self.root / "short_hidden_positions.npy", allow_pickle=False
        ).astype(np.int64)
        self.short_final_hidden = np.load(
            self.root / "short_final_hidden.npy", mmap_mode="r", allow_pickle=False
        )
        if self.target_tokens.shape != self.teacher_delta.shape:
            raise RuntimeError("teacher source-effect shape drift")
        if len(self.example_indices) != len(self.target_tokens):
            raise RuntimeError("teacher example-index shape drift")
        if len(self.eligible_indices) == 0 or not set(self.eligible_indices.tolist()).issubset(set(self.example_indices.tolist())):
            raise RuntimeError("teacher eligible-index contract drift")
        if len(self.short_pool_source_rows) != len(self.short_gold_logprob) or len(self.short_pool_source_rows) != len(self.short_final_hidden) or len(self.short_pool_source_rows) != len(self.short_pool_split):
            raise RuntimeError("teacher short replay shape drift")
        self.lookup = {
            int(example): index
            for index, example in enumerate(self.example_indices.tolist())
        }


def _frequency_tables(path: Path) -> dict[str, np.ndarray]:
    from .frequency_assets import load_frequency_assets

    assets = load_frequency_assets(path.resolve())
    return {
        "native": assets["Native"],
        "anchored_evq_cosh_tau_2": assets["anchored_evq_cosh_tau_2"],
        "phase_chord_olmo_r0_lambda_0p1": assets[
            "phase_chord_olmo_r0_lambda_0p1"
        ],
    }


def protocol_from_args(args: argparse.Namespace) -> dict[str, Any]:
    from .source_effect_loss import SourceEffectConfig

    if int(args.steps) != LOCKED_STEPS or int(args.morph_steps) != LOCKED_MORPH_STEPS:
        raise ValueError("registered protocol requires 300 steps and 60 morph steps")
    if int(args.rank) != LOCKED_RANK or float(args.alpha) != LOCKED_ALPHA:
        raise ValueError("registered protocol requires PEFT rank64 alpha128")
    if int(args.micro_pairs) <= 0 or int(args.gradient_accumulation_steps) <= 0:
        raise ValueError("micro-pairs and accumulation must be positive")
    if not 1 <= int(args.short_replay_every) <= int(
        args.gradient_accumulation_steps
    ):
        raise ValueError("short replay cadence must fit one optimizer step")
    config = SourceEffectConfig(
        primary_tokens=int(args.primary_tokens),
        primary_weight=float(args.primary_weight),
        continuation_weight=float(args.continuation_weight),
        effect_weight=float(args.effect_weight),
        margin_weight=float(args.margin_weight),
        correct_ce_weight=float(args.correct_ce_weight),
        source_margin=float(args.source_margin),
        min_teacher_effect=float(args.min_teacher_effect),
    )
    config.validate()
    if config.effect_weight <= 0 and config.margin_weight <= 0:
        raise ValueError("pure answer CE is forbidden")
    if config.correct_ce_weight > 0.25 * (
        config.effect_weight + config.margin_weight
    ):
        raise ValueError("correct CE must remain a low-weight auxiliary")
    return {
        "arm": str(args.arm),
        "checkpoint": "OLMo-2-0425-1B-Instruct",
        "steps": LOCKED_STEPS,
        "morph_steps": LOCKED_MORPH_STEPS,
        "morph": (
            "none_exact_native"
            if args.arm == "native"
            else "training_only_log_inv_freq_smoothstep_native_to_target"
        ),
        "deployment_table": str(args.arm),
        "training_length": 8_192,
        "pair_batch": "physical8k_correct_plus_swapped",
        "training_pool": "all_teacher_positive_eligible_pairs",
        "micro_pairs": int(args.micro_pairs),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "global_sequences": 2
        * int(args.micro_pairs)
        * int(args.gradient_accumulation_steps),
        "lora": {
            "implementation": "standard_peft",
            "rank": LOCKED_RANK,
            "alpha": LOCKED_ALPHA,
            "dropout": 0.0,
            "bias": "none",
            "target_modules": list(TARGET_MODULES),
        },
        "optimizer": "fused_adamw",
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "weight_decay": 0.0,
        "precision": "bf16_autocast",
        "compile_mode": str(args.compile_mode),
        "source_effect": asdict(config),
        "short_replay": {
            "length": 4_096,
            "every_accumulations": int(args.short_replay_every),
            "nll_delta_budget": 0.10,
            "selected_final_hidden_mse_budget": 1e-3,
            "constraint": "augmented_lagrangian",
        },
        "seed": int(args.seed),
        "metric_boundary": "training diagnostics are not capability evidence",
    }


def _input_receipts(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = checkpoint_receipt(
        args.checkpoint.resolve(), args.checkpoint_ready_receipt.resolve()
    )
    data = _manifest_receipt(args.train_data.resolve(), DATA_STATUS)
    teacher = _manifest_receipt(args.teacher_assets.resolve(), TEACHER_STATUS)
    if teacher["manifest"].get("data_manifest_sha256") != data["manifest_sha256"]:
        raise RuntimeError("teacher assets bind another training dataset")
    tables = _frequency_tables(args.frequency_manifest.resolve())
    target = tables[str(args.arm)]
    from .prepare_phase_chord_data import load_source

    _, _, observed_source = load_source(
        args.source_tensor.resolve(), args.source_token_receipt.resolve()
    )
    expected_source = teacher["manifest"].get("source_hashes", {})
    if expected_source != observed_source:
        raise RuntimeError("teacher/raw-source binding drift")
    return {
        "checkpoint": checkpoint,
        "train_data": {
            "manifest_sha256": data["manifest_sha256"],
            "files": data["files"],
        },
        "teacher_assets": {
            "manifest_sha256": teacher["manifest_sha256"],
            "files": teacher["files"],
        },
        "raw_source": {
            "tensor_path": str(args.source_tensor.resolve()),
            "receipt_path": str(args.source_token_receipt.resolve()),
            **observed_source,
        },
        "frequency_manifest": {
            "path": str(args.frequency_manifest.resolve()),
            "sha256": sha256_file(args.frequency_manifest.resolve()),
            "native_float32_sha256": _array_sha256(tables["native"]),
            "target_float32_sha256": _array_sha256(target),
        },
        "runtime_dependencies": _dependency_versions(),
    }


def _peft_version() -> str:
    if importlib.util.find_spec("peft") is None:
        raise RuntimeError(
            "PEFT is missing; run the runner's explicit noGPU install-peft command"
        )
    import peft

    return str(peft.__version__)


def _dependency_versions() -> dict[str, str]:
    import accelerate
    import peft
    import torch
    import transformers

    versions = {
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "peft": str(peft.__version__),
        "accelerate": str(accelerate.__version__),
    }
    if not versions["torch"].startswith("2.8."):
        raise RuntimeError("registered runtime requires PyTorch 2.8.x")
    if not versions["transformers"].startswith("5.15."):
        raise RuntimeError("registered runtime requires Transformers 5.15.x")
    if versions["peft"] != "0.20.0" or versions["accelerate"] != "1.14.0":
        raise RuntimeError("registered PEFT/Accelerate version drift")
    return versions


def _configure_cuda(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    if architecture not in torch.cuda.get_arch_list():
        raise RuntimeError(f"active architecture {architecture} is unavailable")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        raise RuntimeError("persistent TORCHINDUCTOR_CACHE_DIR is required")
    if "expandable_segments:True" not in os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", ""
    ):
        raise RuntimeError("expandable_segments allocator is required")
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "architecture": architecture,
        "attention": "standard_hf_flash_sdpa_only",
        "dependencies": _dependency_versions(),
        "compile_cache": os.environ["TORCHINDUCTOR_CACHE_DIR"],
    }


def _load_base(checkpoint: Path) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def _install_peft(base: Any, args: argparse.Namespace) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=LOCKED_RANK,
        lora_alpha=LOCKED_ALPHA,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(TARGET_MODULES),
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        init_lora_weights=True,
    )
    model = get_peft_model(base, config)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or any(
        not ("lora_A" in name or "lora_B" in name) for name in names
    ):
        raise RuntimeError("PEFT trainable scope escaped QKVO LoRA")
    if any(
        not any(f".{module}." in name for module in TARGET_MODULES)
        for name in names
    ):
        raise RuntimeError("PEFT target-module scope drift")
    count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if count != 16_777_216:
        raise RuntimeError(f"unexpected OLMo QKVO rank64 parameter count: {count}")
    return model


def _base_model(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _backbone(model: Any) -> Any:
    return _base_model(model).model


def _lm_head(model: Any) -> Any:
    return _base_model(model).lm_head


def _gold_logprobs(
    *, model: Any, hidden: Any, positions: Any, tokens: Any
) -> Any:
    import torch

    selected = hidden.gather(
        1, (positions - 1).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    )
    logits = _lm_head(model)(selected).float()
    values = torch.log_softmax(logits, dim=-1).gather(
        -1, tokens.unsqueeze(-1)
    ).squeeze(-1)
    del selected, logits
    return values


def _target_logits(*, model: Any, hidden: Any, positions: Any) -> Any:
    selected = hidden.gather(
        1, (positions - 1).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    )
    return _lm_head(model)(selected)


def _set_inv_freq(model: Any, value: Any) -> None:
    import torch

    rotary = _base_model(model).model.rotary_emb
    tensor = torch.as_tensor(value, dtype=torch.float32, device=rotary.inv_freq.device)
    if rotary.inv_freq.shape != tensor.shape:
        raise RuntimeError("frequency shape does not match HF OLMo rotary")
    with torch.no_grad():
        rotary.inv_freq.copy_(tensor)
        rotary.original_inv_freq = rotary.inv_freq


def _teacher_target_positions(view: PhysicalPairView, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return view.positions[indices], view.tokens[indices]


def run_teacher(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    runtime = _configure_cuda(torch)
    checkpoint = checkpoint_receipt(
        args.checkpoint.resolve(), args.checkpoint_ready_receipt.resolve()
    )
    view = PhysicalPairView(args.train_data.resolve())
    from .prepare_phase_chord_data import load_source

    source_tokens, source_rows, source_hashes = load_source(
        args.source_tensor.resolve(), args.source_token_receipt.resolve()
    )
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output if output.exists() else incomplete)
    examples = len(view.correct)
    if examples <= 0:
        raise RuntimeError("teacher precompute selects no example")
    example_indices = np.arange(examples, dtype=np.int64)
    model = _load_base(args.checkpoint.resolve()).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    positions = view.short_positions[example_indices]
    tokens = view.tokens[example_indices]
    delta = np.empty(tokens.shape, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, examples, int(args.teacher_batch_pairs)):
            stop = min(start + int(args.teacher_batch_pairs), examples)
            selected = example_indices[start:stop]
            correct = torch.from_numpy(
                np.asarray(view.short_correct[selected], dtype=np.int64)
            ).to("cuda")
            swapped = torch.from_numpy(
                np.asarray(view.short_swapped[selected], dtype=np.int64)
            ).to("cuda")
            target_positions = torch.from_numpy(positions[start:stop]).to("cuda")
            target_tokens = torch.from_numpy(tokens[start:stop]).to("cuda")
            hidden = model.model(
                input_ids=torch.cat((correct, swapped), dim=0),
                use_cache=False,
                return_dict=False,
            )[0]
            correct_lp = _gold_logprobs(
                model=model,
                hidden=hidden[: len(selected)],
                positions=target_positions,
                tokens=target_tokens,
            )
            swapped_lp = _gold_logprobs(
                model=model,
                hidden=hidden[len(selected) :],
                positions=target_positions,
                tokens=target_tokens,
            )
            value = correct_lp - swapped_lp
            if not torch.isfinite(value).all():
                raise RuntimeError("non-finite Native teacher effect")
            delta[start:stop] = value.cpu().numpy().astype(np.float32)

    eligible_indices = example_indices[
        np.any(delta > float(args.min_teacher_effect), axis=1)
    ]
    if len(eligible_indices) == 0:
        raise RuntimeError("Native teacher identifies no positive source-effect pair")

    train_source_rows = [
        index for index, row in enumerate(source_rows) if row["split"] == "train"
    ][: int(args.short_pool_rows)]
    validation_source_rows = [
        index
        for index, row in enumerate(source_rows)
        if row["split"] == "validation"
    ][: int(args.short_pool_rows)]
    if len(train_source_rows) != int(args.short_pool_rows) or len(validation_source_rows) != int(args.short_pool_rows):
        raise RuntimeError("teacher replay requires fixed train+validation document pools")
    pool_source_rows = np.asarray(
        [*train_source_rows, *validation_source_rows], dtype=np.int64
    )
    pool_split = np.asarray(
        [0] * len(train_source_rows) + [1] * len(validation_source_rows),
        dtype=np.int8,
    )
    pool_size = len(pool_source_rows)
    short_positions = np.arange(
        4_096 - int(args.short_nll_tokens), 4_096, dtype=np.int64
    )
    hidden_positions = np.unique(
        np.linspace(0, 4_095, int(args.short_hidden_positions), dtype=np.int64)
    )
    short_logprob = np.empty((pool_size, len(short_positions)), dtype=np.float32)
    short_hidden = np.empty(
        (pool_size, len(hidden_positions), 2_048), dtype=np.float32
    )
    with torch.inference_mode():
        for slot, source_row in enumerate(pool_source_rows):
            ids = torch.from_numpy(
                np.asarray(source_tokens[int(source_row)], dtype=np.int64)[None, :]
            ).to("cuda")
            hidden = model.model(
                input_ids=ids, use_cache=False, return_dict=False
            )[0]
            target_positions = torch.from_numpy(short_positions[None, :]).to("cuda")
            target_tokens = ids.gather(1, target_positions)
            value = _gold_logprobs(
                model=model,
                hidden=hidden,
                positions=target_positions,
                tokens=target_tokens,
            )
            short_logprob[slot] = value.cpu().numpy().astype(np.float32)
            short_hidden[slot] = hidden[0, hidden_positions].float().cpu().numpy()

    incomplete.mkdir(parents=True)
    outputs = {
        "example_indices.npy": example_indices,
        "target_token_ids.npy": tokens.astype(np.uint32),
        "teacher_delta_logprob.npy": delta,
        "eligible_indices.npy": eligible_indices,
        "short_pool_source_rows.npy": pool_source_rows,
        "short_pool_split.npy": pool_split,
        "short_target_positions.npy": short_positions,
        "short_gold_logprob.npy": short_logprob,
        "short_hidden_positions.npy": hidden_positions,
        "short_final_hidden.npy": short_hidden,
    }
    for name, value in outputs.items():
        np.save(incomplete / name, value, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "status": TEACHER_STATUS,
        "checkpoint": checkpoint,
        "data_manifest_sha256": view.receipt["manifest_sha256"],
        "source_hashes": source_hashes,
        "contract": {
            "teacher": "released_Native",
            "source_effect": "gold_logprob_correct_minus_swapped",
            "prediction_shift": "target_position_p_minus_1",
            "full_logits_persisted": False,
            "component_source_logprobs_persisted": False,
            "short_replay": "4K_gold_logprob_and_selected_final_hidden",
            "training_examples": examples,
            "eligible_training_examples": int(len(eligible_indices)),
            "eligible_indices_sha256": _array_sha256(eligible_indices),
            "short_pool_rows_per_split": int(args.short_pool_rows),
            "short_pool_rows_total": pool_size,
        },
        "output_hashes": {
            name: sha256_file(incomplete / name) for name in outputs
        },
        "runtime": runtime,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "metric_boundary": "teacher assets are supervision, not capability evidence",
    }
    atomic_json(incomplete / "manifest.json", manifest)
    incomplete.replace(output)
    return manifest


def _source_loss(
    *, model: Any, backbone: Any, view: PhysicalPairView, teacher: TeacherAssets, example_indices: np.ndarray, config: Any
) -> tuple[Any, dict[str, float]]:
    import torch
    from .source_effect_loss import source_effect_loss

    teacher_rows = np.asarray(
        [teacher.lookup[int(index)] for index in example_indices], dtype=np.int64
    )
    correct = torch.from_numpy(
        np.asarray(view.correct[example_indices], dtype=np.int64)
    ).to("cuda")
    swapped = torch.from_numpy(
        np.asarray(view.swapped[example_indices], dtype=np.int64)
    ).to("cuda")
    positions = torch.from_numpy(view.positions[example_indices]).to("cuda")
    teacher_positions = torch.from_numpy(
        view.short_positions[example_indices]
    ).to("cuda")
    tokens = torch.from_numpy(view.tokens[example_indices]).to("cuda")
    hidden = backbone(
        input_ids=torch.cat((correct, swapped), dim=0),
        use_cache=False,
        return_dict=False,
    )[0]
    correct_lp = _gold_logprobs(
        model=model,
        hidden=hidden[: len(example_indices)],
        positions=positions,
        tokens=tokens,
    )
    swapped_lp = _gold_logprobs(
        model=model,
        hidden=hidden[len(example_indices) :],
        positions=positions,
        tokens=tokens,
    )
    teacher_delta = torch.from_numpy(
        np.asarray(teacher.teacher_delta[teacher_rows], dtype=np.float32)
    ).to("cuda")
    teacher_correct_ids = torch.from_numpy(
        np.asarray(view.short_correct[example_indices], dtype=np.int64)
    ).to("cuda")
    teacher_swapped_ids = torch.from_numpy(
        np.asarray(view.short_swapped[example_indices], dtype=np.int64)
    ).to("cuda")
    loss, metrics = source_effect_loss(
        teacher_correct=teacher_delta,
        teacher_swapped=torch.zeros_like(teacher_delta),
        student_correct=correct_lp,
        student_swapped=swapped_lp,
        target_tokens=tokens,
        values_are_logits=False,
        shift_contract={
            "teacher_correct": (teacher_correct_ids, teacher_positions),
            "teacher_swapped": (teacher_swapped_ids, teacher_positions),
            "student_correct": (correct, positions),
            "student_swapped": (swapped, positions),
        },
        config=config,
    )
    metrics = {
        **metrics,
        "correct_gold_logprob_mean": float(correct_lp.detach().float().mean()),
        "swapped_gold_logprob_mean": float(swapped_lp.detach().float().mean()),
    }
    return loss, metrics


def _short_replay_loss(
    *, model: Any, backbone: Any, source_tokens: np.ndarray, teacher: TeacherAssets, pool_slot: int, args: argparse.Namespace
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    source_row = int(teacher.short_pool_source_rows[pool_slot])
    ids = torch.from_numpy(
        np.asarray(source_tokens[source_row], dtype=np.int64)[None, :]
    ).to("cuda")
    positions = torch.from_numpy(
        teacher.short_target_positions[None, :]
    ).to("cuda")
    tokens = ids.gather(1, positions)
    hidden = backbone(input_ids=ids, use_cache=False, return_dict=False)[0]
    student_lp = _gold_logprobs(
        model=model, hidden=hidden, positions=positions, tokens=tokens
    )
    native_lp = torch.from_numpy(
        np.asarray(teacher.short_gold_logprob[pool_slot], dtype=np.float32)[None, :]
    ).to("cuda")
    student_nll = -student_lp.mean()
    native_nll = -native_lp.mean()
    nll_violation = student_nll - native_nll - 0.10
    selected_student = hidden[:, teacher.short_hidden_positions].float()
    selected_native = torch.from_numpy(
        np.asarray(teacher.short_final_hidden[pool_slot], dtype=np.float32)[None, :]
    ).to("cuda")
    denominator = selected_native.square().mean().clamp_min(1e-8)
    hidden_trust = (
        selected_student - selected_native
    ).square().mean() / denominator
    return (nll_violation, hidden_trust), {
        "short_nll_delta": float((student_nll - native_nll).detach()),
        "short_nll_violation": float(nll_violation.detach()),
        "short_selected_hidden_trust": float(hidden_trust.detach()),
        "short_student_nll": float(student_nll.detach()),
        "short_native_nll": float(native_nll),
    }


def _cosine_lr(step: int, steps: int, warmup: int, maximum: float) -> float:
    if warmup > 0 and step <= warmup:
        return maximum * step / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return maximum * (
        0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def _adapter_state(model: Any) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    return {
        name: value.detach().cpu().contiguous()
        for name, value in get_peft_model_state_dict(model).items()
    }


def _train(
    *, model: Any, backbone: Any, view: PhysicalPairView, source_tokens: np.ndarray, teacher: TeacherAssets, native: Any, target: Any, args: argparse.Namespace, output: Path, smoke: bool
) -> dict[str, Any]:
    import torch
    from .source_effect_loss import SourceEffectConfig

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    config = SourceEffectConfig(
        primary_tokens=int(args.primary_tokens),
        primary_weight=float(args.primary_weight),
        continuation_weight=float(args.continuation_weight),
        effect_weight=float(args.effect_weight),
        margin_weight=float(args.margin_weight),
        correct_ce_weight=float(args.correct_ce_weight),
        source_margin=float(args.source_margin),
        min_teacher_effect=float(args.min_teacher_effect),
    )
    steps = 1 if smoke else LOCKED_STEPS
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 53_001)
    pool = torch.from_numpy(teacher.eligible_indices.copy())
    eligible_stream_sha256 = _array_sha256(teacher.eligible_indices)
    started = time.perf_counter()
    last_log_time = started
    last_log_tokens = 0
    processed_tokens = 0
    selection_hash = hashlib.sha256()
    recent: list[float] = []
    dual_nll = torch.zeros((), device="cuda")
    dual_hidden = torch.zeros((), device="cuda")
    augmented_rho = 10.0
    train_short_slots = np.flatnonzero(teacher.short_pool_split == 0)
    if len(train_short_slots) == 0:
        raise RuntimeError("teacher assets have no train short replay pool")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    log_path = output / "train_log.jsonl"
    for step in range(1, steps + 1):
        table, morph = smoothstep_log_morph(
            native,
            target,
            step=(LOCKED_MORPH_STEPS if args.arm == "native" else step),
            morph_steps=LOCKED_MORPH_STEPS,
        )
        if args.arm == "native":
            table = native
            morph = {
                **morph,
                "smoothstep_amount": 0.0,
                "is_exact_target": True,
                "native_control": True,
            }
        _set_inv_freq(model, table)
        lr = _cosine_lr(step, steps, int(args.warmup_steps), float(args.learning_rate))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        last_source: dict[str, float] = {}
        last_short: dict[str, float] = {}
        for accumulation in range(int(args.gradient_accumulation_steps)):
            selected = torch.randint(
                len(pool),
                (int(args.micro_pairs),),
                generator=generator,
            )
            indices = pool[selected].numpy()
            selection_hash.update(np.asarray(indices, dtype="<i8").tobytes())
            source_loss, last_source = _source_loss(
                model=model,
                backbone=backbone,
                view=view,
                teacher=teacher,
                example_indices=indices,
                config=config,
            )
            scaled = source_loss / float(args.gradient_accumulation_steps)
            if not torch.isfinite(scaled):
                raise RuntimeError(f"non-finite source-effect loss at step {step}")
            scaled.backward()
            value = float(source_loss.detach())
            del source_loss, scaled
            if accumulation % int(args.short_replay_every) == 0:
                local_slot = (
                    (step - 1) * int(args.gradient_accumulation_steps)
                    + accumulation
                ) % len(train_short_slots)
                slot = int(train_short_slots[local_slot])
                violations, last_short = _short_replay_loss(
                    model=model,
                    backbone=backbone,
                    source_tokens=source_tokens,
                    teacher=teacher,
                    pool_slot=slot,
                    args=args,
                )
                nll_violation, hidden_mse = violations
                hidden_violation = hidden_mse - 1e-3
                positive_nll = torch.relu(nll_violation)
                positive_hidden = torch.relu(hidden_violation)
                short_loss = (
                    dual_nll.detach() * positive_nll
                    + 0.5 * augmented_rho * positive_nll.square()
                    + dual_hidden.detach() * positive_hidden
                    + 0.5 * augmented_rho * positive_hidden.square()
                )
                short_scaled = short_loss / float(args.gradient_accumulation_steps)
                if not torch.isfinite(short_scaled):
                    raise RuntimeError(f"non-finite short replay loss at step {step}")
                short_scaled.backward()
                value += float(short_loss.detach())
                processed_tokens += 4_096
                del short_loss, short_scaled
            step_losses.append(value)
            processed_tokens += int(len(indices) * 2 * 8_192)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        if last_short:
            dual_nll = torch.clamp(
                dual_nll
                + augmented_rho
                * torch.tensor(last_short["short_nll_violation"], device="cuda"),
                min=0.0,
            )
            dual_hidden = torch.clamp(
                dual_hidden
                + augmented_rho
                * torch.tensor(
                    last_short["short_selected_hidden_trust"] - 1e-3,
                    device="cuda",
                ),
                min=0.0,
            )
        mean_loss = float(np.mean(step_losses))
        recent.append(mean_loss)
        if step == 1 or step % 10 == 0 or step == steps:
            torch.cuda.synchronize()
            now = time.perf_counter()
            row = {
                "step": step,
                "loss": mean_loss,
                "mean_loss_last_10": float(np.mean(recent[-10:])),
                "gradient_norm": float(grad_norm),
                "learning_rate": lr,
                "morph": morph,
                "processed_input_tokens": processed_tokens,
                "interval_tokens_per_second": (
                    (processed_tokens - last_log_tokens)
                    / max(now - last_log_time, 1e-9)
                ),
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "elapsed_seconds": now - started,
                **last_source,
                **last_short,
                "dual_nll": float(dual_nll),
                "dual_hidden": float(dual_hidden),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            last_log_time, last_log_tokens = now, processed_tokens
    _set_inv_freq(model, target)
    realized = _base_model(model).model.rotary_emb.inv_freq.detach().cpu().float()
    expected = __import__("torch").as_tensor(target).float()
    if not __import__("torch").equal(realized, expected):
        raise RuntimeError("final training table is not exact frozen target")
    return {
        "steps": steps,
        "processed_input_tokens": processed_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "eligible_indices_sha256": eligible_stream_sha256,
        "selection_stream_sha256": selection_hash.hexdigest(),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "final_table_float32_sha256": _array_sha256(realized.numpy()),
        "final_table_exact_target": True,
        "short_trust": {
            "nll_delta_budget": 0.10,
            "normalized_hidden_mse_budget": 1e-3,
            "augmented_rho": augmented_rho,
            "dual_nll": float(dual_nll),
            "dual_hidden": float(dual_hidden),
        },
    }


def _roundtrip(
    *, expected_state: Mapping[str, Any], bundle: Path, checkpoint: Path
) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import get_peft_model_state_dict

    base = _load_base(checkpoint).to("cuda")
    loaded, load_receipt = load_standard_peft_bundle(
        base_model=base,
        bundle=bundle,
    )
    observed = {
        name: value.detach().cpu().contiguous()
        for name, value in get_peft_model_state_dict(loaded).items()
    }
    state_receipt = assert_peft_state_roundtrip(expected_state, observed)
    return loaded, {"load": load_receipt, "state": state_receipt}


def _compiled_final_table_smoke(
    *,
    model: Any,
    backbone: Any,
    view: PhysicalPairView,
    teacher: TeacherAssets,
    config: Any,
    target: Any,
) -> dict[str, Any]:
    """Prove the grad-enabled compiled graph reads the exact final table."""
    import torch

    index = np.asarray([int(teacher.eligible_indices[0])], dtype=np.int64)
    active = (
        _base_model(model)
        .model.rotary_emb.inv_freq.detach()
        .cpu()
        .float()
        .numpy()
    )
    target_hash = _array_sha256(np.asarray(target, dtype=np.float32))
    active_hash = _array_sha256(active)
    if active_hash != target_hash:
        raise RuntimeError("compiled final-table smoke did not start at target")

    model.zero_grad(set_to_none=True)
    compiled_loss, compiled_metrics = _source_loss(
        model=model,
        backbone=backbone,
        view=view,
        teacher=teacher,
        example_indices=index,
        config=config,
    )
    if not torch.isfinite(compiled_loss):
        raise RuntimeError("non-finite exact-target compiled smoke loss")
    compiled_loss.backward()
    gradient_norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"missing/non-finite final-table gradient: {name}")
        norm = float(parameter.grad.detach().float().norm())
        if norm <= 0.0:
            raise RuntimeError(f"zero final-table gradient: {name}")
        gradient_norms[name] = norm
    if not gradient_norms:
        raise RuntimeError("final-table compiled smoke found no trainable gradients")
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        eager_loss, eager_metrics = _source_loss(
            model=model,
            backbone=_backbone(model),
            view=view,
            teacher=teacher,
            example_indices=index,
            config=config,
        )
    comparisons = {
        "loss": (float(compiled_loss.detach()), float(eager_loss.detach())),
        "correct_gold_logprob_mean": (
            float(compiled_metrics["correct_gold_logprob_mean"]),
            float(eager_metrics["correct_gold_logprob_mean"]),
        ),
        "swapped_gold_logprob_mean": (
            float(compiled_metrics["swapped_gold_logprob_mean"]),
            float(eager_metrics["swapped_gold_logprob_mean"]),
        ),
    }
    for name, (compiled_value, eager_value) in comparisons.items():
        if not math.isclose(
            compiled_value, eager_value, rel_tol=5e-3, abs_tol=5e-3
        ):
            raise RuntimeError(
                "compiled/eager exact-target parity failed for "
                f"{name}: {compiled_value} != {eager_value}"
            )
    return {
        "physical_shape": [2, 8_192],
        "active_target_float32_sha256": active_hash,
        "compiled": {name: values[0] for name, values in comparisons.items()},
        "eager": {name: values[1] for name, values in comparisons.items()},
        "parity_rtol": 5e-3,
        "parity_atol": 5e-3,
        "trainable_gradient_tensors": len(gradient_norms),
        "minimum_trainable_gradient_norm": min(gradient_norms.values()),
        "maximum_trainable_gradient_norm": max(gradient_norms.values()),
        "source_follow_rate": float(compiled_metrics["source_follow_rate"]),
        "optimizer_step": False,
        "finite": True,
    }


def _fresh_load_cache_parity(model: Any, view: PhysicalPairView) -> dict[str, Any]:
    """Compare full forward with ordinary DynamicCache prefill then decode."""
    import torch

    model.eval()
    ids = torch.from_numpy(
        np.asarray(view.correct[0, :256], dtype=np.int64)[None, :]
    ).to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model(
            input_ids=ids,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        ).logits[:, -1].float()
        prefill = model(
            input_ids=ids[:, :-1],
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        decoded = model(
            input_ids=ids[:, -1:],
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        ).logits[:, -1].float()
    maximum = float((full - decoded).abs().max())
    top1_equal = bool(torch.equal(full.argmax(-1), decoded.argmax(-1)))
    close = bool(torch.allclose(full, decoded, rtol=0.02, atol=0.10))
    if not close or not top1_equal:
        raise RuntimeError(
            "fresh PEFT target-table full/decode cache parity failed: "
            f"max_abs={maximum}, top1_equal={top1_equal}"
        )
    return {
        "sequence_length": 256,
        "cache": type(prefill.past_key_values).__name__,
        "maximum_logit_absolute_difference": maximum,
        "top1_equal": top1_equal,
        "allclose_rtol": 0.02,
        "allclose_atol": 0.10,
        "pass": True,
    }


def run_arm(args: argparse.Namespace, *, smoke: bool) -> dict[str, Any]:
    import torch

    ready = validate_ready_receipt(
        path=args.ready_receipt.resolve(),
        protocol=protocol_from_args(args),
        inputs=_input_receipts(args),
        code_sha256=_code_hashes(),
        run_output=args.output.resolve(),
    )
    runtime = _configure_cuda(torch)
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output if output.exists() else incomplete)
    incomplete.mkdir(parents=True)
    view = PhysicalPairView(args.train_data.resolve())
    from .prepare_phase_chord_data import load_source

    source_tokens, _, _ = load_source(
        args.source_tensor.resolve(), args.source_token_receipt.resolve()
    )
    teacher = TeacherAssets(
        args.teacher_assets.resolve(), view.receipt["manifest_sha256"]
    )
    tables = _frequency_tables(args.frequency_manifest.resolve())
    native = tables["native"]
    target = tables[str(args.arm)]
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))
    base = _load_base(args.checkpoint.resolve())
    runtime_native = base.model.rotary_emb.inv_freq.detach().cpu().float().numpy()
    if not np.array_equal(runtime_native, native):
        raise RuntimeError("released checkpoint Native table differs from frozen asset")
    model = _install_peft(base, args).to("cuda")
    initial_state = _adapter_state(model)
    initial_sha = tensor_bundle_sha256(initial_state)
    expected_initial = getattr(args, "gpu_ready_initial_sha256", None)
    if expected_initial is not None and expected_initial != initial_sha:
        raise RuntimeError("LoRA initialization differs from GPU READY smoke")
    backbone: Any = _backbone(model)
    if str(args.compile_mode) != "none":
        backbone = torch.compile(
            backbone,
            fullgraph=False,
            dynamic=False,
            mode=str(args.compile_mode),
        )
    training = _train(
        model=model,
        backbone=backbone,
        view=view,
        source_tokens=source_tokens,
        teacher=teacher,
        native=native,
        target=target,
        args=args,
        output=incomplete,
        smoke=smoke,
    )
    from .source_effect_loss import SourceEffectConfig

    exact_target_compiled = None
    if smoke:
        exact_target_compiled = _compiled_final_table_smoke(
            model=model,
            backbone=backbone,
            view=view,
            teacher=teacher,
            config=SourceEffectConfig(**protocol_from_args(args)["source_effect"]),
            target=target,
        )
    final_state = _adapter_state(model)
    bundle = incomplete / "artifacts"
    artifact_receipt = save_standard_peft_bundle(
        model=model,
        output=bundle,
        inv_freq=target,
        metadata={
            "arm": str(args.arm),
            "checkpoint_sha256": _input_receipts(args)["checkpoint"]["weight_sha256"],
            "frequency_manifest_sha256": sha256_file(args.frequency_manifest.resolve()),
            "target_float32_sha256": _array_sha256(target),
            "protocol": protocol_from_args(args),
        },
    )
    # Drop references to the compiled graph before the fresh standard-PEFT
    # reload.  The roundtrip is a storage/load contract, not a capability test.
    import gc

    del backbone, model, base
    gc.collect()
    torch.cuda.empty_cache()
    loaded, roundtrip = _roundtrip(
        expected_state=final_state,
        bundle=bundle,
        checkpoint=args.checkpoint.resolve(),
    )
    loaded_table = _base_model(loaded).model.rotary_emb.inv_freq.detach().cpu().float().numpy()
    if not np.array_equal(loaded_table, target):
        raise RuntimeError("PEFT roundtrip lost custom static frequency table")
    cache_parity = _fresh_load_cache_parity(loaded, view)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    training_headroom = total_memory - int(
        training["peak_memory_allocated_bytes"]
    )
    if smoke and training_headroom < (1 << 30):
        raise RuntimeError(
            "exact-shape smoke leaves less than 1 GiB allocated-memory headroom"
        )
    status = GPU_READY_STATUS if smoke else COMPLETE_STATUS
    result = {
        "schema_version": 1,
        "status": status,
        "arm": str(args.arm),
        "protocol": protocol_from_args(args),
        "inputs": _input_receipts(args),
        "code_sha256": _code_hashes(),
        "ready_receipt": {
            "path": str(args.ready_receipt.resolve()),
            "sha256": sha256_file(args.ready_receipt.resolve()),
            "status": ready["status"],
        },
        "runtime": runtime,
        "training": training,
        "gpu_contract": {
            "compiled_exact_target_forward": exact_target_compiled,
            "fresh_load_dynamic_cache_parity": cache_parity,
            "device_total_memory_bytes": total_memory,
            "training_allocated_headroom_bytes": training_headroom,
            "minimum_required_headroom_bytes": 1 << 30,
        },
        "lora": {
            "implementation": "standard_peft",
            "trainable_parameters": 16_777_216,
            "initial_state_sha256": initial_sha,
            "final_state_sha256": tensor_bundle_sha256(final_state),
            "save_load_roundtrip": roundtrip,
        },
        "artifacts": artifact_receipt,
        "results_file_hash_excluded_until_atomic_finalize": True,
        "metric_boundary": (
            "Source-effect, 4K NLL, and selected-hidden trust are training "
            "diagnostics. This run does not establish autoregressive capability."
        ),
    }
    atomic_json(incomplete / "results.json", result)
    result["output_hashes"] = {
        "train_log.jsonl": sha256_file(incomplete / "train_log.jsonl"),
        "artifacts": artifact_receipt["files"],
    }
    atomic_json(incomplete / "results.json", result)
    incomplete.replace(output)
    return result


def _natural_cell(
    *, model: Any, view: PhysicalPairView, rows: int
) -> dict[str, Any]:
    import torch

    count = min(int(rows), len(view.correct))
    per_row: list[dict[str, Any]] = []
    model.eval()
    for index in range(count):
        correct = torch.from_numpy(
            np.asarray(view.correct[index], dtype=np.int64)[None, :]
        ).to("cuda")
        swapped = torch.from_numpy(
            np.asarray(view.swapped[index], dtype=np.int64)[None, :]
        ).to("cuda")
        positions = torch.from_numpy(view.positions[index][None, :]).to("cuda")
        tokens = torch.from_numpy(view.tokens[index][None, :]).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = _backbone(model)(
                input_ids=torch.cat((correct, swapped), dim=0),
                use_cache=False,
                return_dict=False,
            )[0]
            correct_logits = _target_logits(
                model=model, hidden=hidden[:1], positions=positions
            ).float()
            swapped_logits = _target_logits(
                model=model, hidden=hidden[1:], positions=positions
            ).float()
            correct_lp = torch.log_softmax(correct_logits, -1).gather(
                -1, tokens.unsqueeze(-1)
            ).squeeze(-1)
            swapped_lp = torch.log_softmax(swapped_logits, -1).gather(
                -1, tokens.unsqueeze(-1)
            ).squeeze(-1)
            first = correct_logits[0, 0]
            gold = int(tokens[0, 0])
            first_rank = int((first > first[gold]).sum()) + 1

            target_start = int(view.positions[index, 0])
            generated = correct[:, :target_start]
            past = None
            next_input = generated
            values: list[int] = []
            for _ in range(int(view.tokens.shape[1])):
                outputs = model(
                    input_ids=next_input,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
                token = outputs.logits[:, -1].argmax(dim=-1)
                values.append(int(token.item()))
                past = outputs.past_key_values
                next_input = token[:, None]
        exact = values == [int(value) for value in view.tokens[index].tolist()]
        per_row.append(
            {
                "row": index,
                "mean_correct_gold_logprob": float(correct_lp.mean()),
                "mean_swapped_gold_logprob": float(swapped_lp.mean()),
                "mean_source_effect": float((correct_lp - swapped_lp).mean()),
                "source_follow_rate": float((correct_lp > swapped_lp).float().mean()),
                "first_token_top1": first_rank == 1,
                "first_token_rank": first_rank,
                "natural_16_token_greedy_exact": bool(exact),
                "generated_token_ids": values,
                "target_token_ids": [int(value) for value in view.tokens[index]],
            }
        )
    return {
        "rows": count,
        "mean_source_effect": float(np.mean([row["mean_source_effect"] for row in per_row])),
        "source_follow_rate": float(np.mean([row["source_follow_rate"] for row in per_row])),
        "first_token_top1": float(np.mean([row["first_token_top1"] for row in per_row])),
        "median_first_token_rank": float(np.median([row["first_token_rank"] for row in per_row])),
        "natural_16_token_greedy_exact": float(np.mean([row["natural_16_token_greedy_exact"] for row in per_row])),
        "per_row": per_row,
    }


def _retention_4k(
    *, model: Any, source_tokens: np.ndarray, teacher: TeacherAssets
) -> dict[str, Any]:
    import torch

    slots = np.flatnonzero(teacher.short_pool_split == 1)
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for slot in slots:
            source_row = int(teacher.short_pool_source_rows[slot])
            ids = torch.from_numpy(
                np.asarray(source_tokens[source_row], dtype=np.int64)[None, :]
            ).to("cuda")
            hidden = _backbone(model)(
                input_ids=ids, use_cache=False, return_dict=False
            )[0]
            positions = torch.from_numpy(
                teacher.short_target_positions[None, :]
            ).to("cuda")
            tokens = ids.gather(1, positions)
            student_lp = _gold_logprobs(
                model=model, hidden=hidden, positions=positions, tokens=tokens
            )
            native_lp = torch.from_numpy(
                np.asarray(teacher.short_gold_logprob[slot], dtype=np.float32)[None, :]
            ).to("cuda")
            selected_student = hidden[:, teacher.short_hidden_positions].float()
            selected_native = torch.from_numpy(
                np.asarray(teacher.short_final_hidden[slot], dtype=np.float32)[None, :]
            ).to("cuda")
            hidden_mse = (
                (selected_student - selected_native).square().mean()
                / selected_native.square().mean().clamp_min(1e-8)
            )
            rows.append(
                {
                    "source_row": source_row,
                    "gold_nll_delta": float((-student_lp.mean()) - (-native_lp.mean())),
                    "selected_final_hidden_normalized_mse": float(hidden_mse),
                }
            )
    return {
        "rows": len(rows),
        "mean_gold_nll_delta": float(np.mean([row["gold_nll_delta"] for row in rows])),
        "mean_selected_final_hidden_normalized_mse": float(
            np.mean([row["selected_final_hidden_normalized_mse"] for row in rows])
        ),
        "per_row": rows,
    }


def run_natural_eval(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    runtime = _configure_cuda(torch)
    checkpoint = checkpoint_receipt(
        args.checkpoint.resolve(), args.checkpoint_ready_receipt.resolve()
    )
    if args.adapter_bundle is None or args.validation_data_8k is None or args.validation_data_16k is None:
        raise ValueError("eval-natural requires adapter bundle and 8K/16K validation data")
    bundle = args.adapter_bundle.resolve()
    metadata = _json(bundle / "metadata.json")
    tables = _frequency_tables(args.frequency_manifest.resolve())
    target = tables[str(args.arm)]
    if (
        metadata.get("arm") != str(args.arm)
        or metadata.get("checkpoint_sha256") != checkpoint["weight_sha256"]
        or metadata.get("target_float32_sha256") != _array_sha256(target)
    ):
        raise RuntimeError("adapter bundle metadata drift")
    base = _load_base(args.checkpoint.resolve())
    model, load_receipt = load_standard_peft_bundle(
        base_model=base, bundle=bundle
    )
    if load_receipt["active_inv_freq_float32_sha256"] != _array_sha256(target):
        raise RuntimeError("loaded adapter frequency sidecar differs from frozen target")
    model = model.to("cuda").eval()
    view8 = PhysicalPairView(args.validation_data_8k.resolve())
    view16 = PhysicalPairView(args.validation_data_16k.resolve())
    teacher = TeacherAssets(
        args.teacher_assets.resolve(),
        _manifest_receipt(args.train_data.resolve(), DATA_STATUS)["manifest_sha256"],
    )
    from .prepare_phase_chord_data import load_source

    source_tokens, _, source_hashes = load_source(
        args.source_tensor.resolve(), args.source_token_receipt.resolve()
    )
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(output if output.exists() else incomplete)
    incomplete.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "status": "OLMO2_PHASE_CHORD_NATURAL_EVAL_COMPLETE_V1",
        "arm": str(args.arm),
        "checkpoint": checkpoint,
        "adapter_bundle": {
            "path": str(bundle),
            "files": load_receipt["adapter_files"],
            "custom_inv_freq_sha256": load_receipt["custom_inv_freq_sha256"],
            "active_inv_freq_float32_sha256": load_receipt[
                "active_inv_freq_float32_sha256"
            ],
        },
        "validation_data": {
            "8k_manifest_sha256": view8.receipt["manifest_sha256"],
            "16k_manifest_sha256": view16.receipt["manifest_sha256"],
            "raw_source": source_hashes,
        },
        "natural": {
            "8k": _natural_cell(model=model, view=view8, rows=int(args.eval_rows)),
            "16k": _natural_cell(model=model, view=view16, rows=int(args.eval_rows)),
        },
        "retention_4k": _retention_4k(
            model=model, source_tokens=source_tokens, teacher=teacher
        ),
        "runtime": runtime,
        "metric_contract": {
            "source_effect": "correct_minus_swapped_gold_logprob",
            "generation": "exactly_16_natural_continuation_tokens",
            "generation_is_answer_plus_eos": False,
            "first_token": "full_vocab_top1_and_rank",
            "teacher_forced_lm_head_scope": "target_positions_only",
        },
        "metric_boundary": (
            "Natural 16-token continuation exact and absolute first-token "
            "top-1 are descriptive because natural continuation is not a "
            "unique-answer task. The matched primary natural endpoints are "
            "source effect, source-follow, and relative first-token rank. "
            "These do not replace held-out RULER/2Wiki capability evaluation."
        ),
    }
    atomic_json(incomplete / "results.json", result)
    result["results_payload_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    atomic_json(incomplete / "results.json", result)
    incomplete.replace(output)
    return result


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("preflight must not initialize CUDA")
    return write_ready_receipt(
        path=args.ready_receipt.resolve(),
        protocol=protocol_from_args(args),
        inputs=_input_receipts(args),
        code_sha256=_code_hashes(),
        run_output=args.output.resolve(),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_initialized=bool(torch.cuda.is_initialized()),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("teacher", "preflight", "smoke", "run", "eval-natural"),
        required=True,
    )
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--arm", choices=ARMS, default="native")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ready-receipt", type=Path, required=True)
    parser.add_argument("--frequency-manifest", type=Path)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--teacher-assets", type=Path)
    parser.add_argument("--validation-data-8k", type=Path)
    parser.add_argument("--validation-data-16k", type=Path)
    parser.add_argument("--adapter-bundle", type=Path)
    parser.add_argument("--eval-rows", type=int, default=64)
    parser.add_argument("--ready-receipt", type=Path)
    parser.add_argument("--gpu-ready-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=LOCKED_STEPS)
    parser.add_argument("--morph-steps", type=int, default=LOCKED_MORPH_STEPS)
    parser.add_argument("--rank", type=int, default=LOCKED_RANK)
    parser.add_argument("--alpha", type=float, default=LOCKED_ALPHA)
    parser.add_argument("--source-tensor", type=Path, required=True)
    parser.add_argument("--source-token-receipt", type=Path, required=True)
    parser.add_argument("--micro-pairs", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "max-autotune-no-cudagraphs"),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--effect-weight", type=float, default=1.0)
    parser.add_argument("--primary-tokens", type=int, default=8)
    parser.add_argument("--primary-weight", type=float, default=1.0)
    parser.add_argument("--continuation-weight", type=float, default=0.25)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--correct-ce-weight", type=float, default=0.1)
    parser.add_argument("--source-margin", type=float, default=1.0)
    parser.add_argument("--min-teacher-effect", type=float, default=0.0)
    parser.add_argument("--short-replay-every", type=int, default=1)
    parser.add_argument("--short-pool-rows", type=int, default=16)
    parser.add_argument("--short-nll-tokens", type=int, default=64)
    parser.add_argument("--short-hidden-positions", type=int, default=8)
    parser.add_argument("--teacher-batch-pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_822)
    return parser.parse_args(argv)


def _require_arm_inputs(args: argparse.Namespace) -> None:
    if args.frequency_manifest is None or args.teacher_assets is None or args.ready_receipt is None:
        raise ValueError("preflight/smoke/run requires frequency, teacher, and READY paths")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "teacher":
        require_gpu_authorization(cli_authorize=bool(args.authorize), environment=os.environ)
        result = run_teacher(args)
    elif args.mode == "eval-natural":
        if args.frequency_manifest is None or args.teacher_assets is None:
            raise ValueError("eval-natural requires frequency and teacher assets")
        require_gpu_authorization(
            cli_authorize=bool(args.authorize), environment=os.environ
        )
        result = run_natural_eval(args)
    else:
        _require_arm_inputs(args)
        protocol_from_args(args)
        if args.mode == "preflight":
            result = _preflight(args)
        else:
            require_gpu_authorization(cli_authorize=bool(args.authorize), environment=os.environ)
            if args.mode == "run":
                if args.gpu_ready_receipt is None:
                    raise ValueError("run requires --gpu-ready-receipt")
                gpu_ready = _json(args.gpu_ready_receipt.resolve())
                if (
                    gpu_ready.get("status") != GPU_READY_STATUS
                    or gpu_ready.get("arm") != str(args.arm)
                    or gpu_ready.get("protocol") != protocol_from_args(args)
                    or gpu_ready.get("inputs") != _input_receipts(args)
                    or gpu_ready.get("code_sha256") != _code_hashes()
                ):
                    raise RuntimeError("GPU READY receipt drift")
                args.gpu_ready_initial_sha256 = gpu_ready.get("lora", {}).get(
                    "initial_state_sha256"
                )
                if not args.gpu_ready_initial_sha256:
                    raise RuntimeError("GPU READY receipt lacks LoRA initialization hash")
            result = run_arm(args, smoke=args.mode == "smoke")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
