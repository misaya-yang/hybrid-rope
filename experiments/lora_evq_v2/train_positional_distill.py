#!/usr/bin/env python3
"""Train q/k-only LoRA to preserve native-Geo hidden states after RoPE change."""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_positional_distill_data import sha256_file, tokenizer_source_fingerprint
from train_evq_lora import (
    build_training_inv_freq,
    evaluation_strategy_kwargs,
    find_rotary_modules,
    inject_inv_freq,
    public_model_identifier,
    resolve_model_rope_geometry,
    verify_model_inv_freq,
)


def position_bucket_ranges(seq_len: int) -> Tuple[Tuple[int, int], ...]:
    if seq_len != 8192:
        raise ValueError("the frozen pilot protocol requires seq_len=8192")
    return ((0, 2048), (2048, 4096), (4096, 8192))


@functools.lru_cache(maxsize=8)
def fingerprint_model_source(model_name: str) -> dict:
    """Return a path-safe identity that binds local weight bytes exactly."""
    fingerprint = {"identifier": public_model_identifier(model_name)}
    model_path = Path(model_name).expanduser()
    if not model_path.is_dir():
        return fingerprint

    config_path = model_path / "config.json"
    if config_path.is_file():
        fingerprint["config_sha256"] = sha256_file(config_path)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        fingerprint["index_sha256"] = sha256_file(index_path)
    fingerprint["shards"] = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(model_path.glob("model*.safetensors"))
    ]
    return fingerprint


def checkpoint_starting_global_step(checkpoint: Optional[Path]) -> int:
    if checkpoint is None:
        return 0
    state_path = Path(checkpoint) / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("resume checkpoint is missing trainer_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    step = int(state.get("global_step", -1))
    if step < 0:
        raise ValueError("resume checkpoint has an invalid global_step")
    return step


def invocation_ledger_ending_step(output_dir: Path) -> int:
    ledger_path = Path(output_dir) / "invocation_ledger.jsonl"
    if not ledger_path.exists():
        return 0
    last_entry = None
    with ledger_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_entry = json.loads(line)
    if last_entry is None:
        return 0
    return int(last_entry.get("ending_global_step", -1))


def ensure_immutable_run_protocol(output_dir: Path, protocol: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_protocol.json"
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != protocol:
            raise RuntimeError(
                "run protocol mismatch; A/B probes and changed resumes require "
                "a fresh output directory"
            )
        return path
    if (output_dir / "adapter_model.safetensors").exists() or any(
        output_dir.glob("checkpoint-*")
    ):
        raise RuntimeError(
            "existing training artifacts lack immutable run_protocol.json"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        return ensure_immutable_run_protocol(output_dir, protocol)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def normalized_bucket_hidden_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    buckets: Sequence[Tuple[int, int]],
    epsilon: float = 1e-8,
    assume_all_tokens_valid: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if student.shape != teacher.shape:
        raise ValueError(
            f"hidden-state shape mismatch: student={tuple(student.shape)} "
            f"teacher={tuple(teacher.shape)}"
        )
    if not assume_all_tokens_valid and attention_mask is None:
        raise ValueError("attention_mask is required unless all tokens are valid")
    if attention_mask is not None and attention_mask.shape != student.shape[:2]:
        raise ValueError("attention_mask must match hidden-state batch and sequence axes")

    bucket_tensors = []
    hidden_size = student.shape[-1]
    for start, end in buckets:
        end = min(end, student.shape[1])
        if start >= end:
            continue
        student_slice = student[:, start:end].float()
        teacher_slice = teacher[:, start:end].float()
        if assume_all_tokens_valid:
            squared_error = (student_slice - teacher_slice).square().mean()
            teacher_energy = teacher_slice.square().mean().clamp_min(epsilon)
        else:
            valid = attention_mask[:, start:end].to(torch.bool).unsqueeze(-1)
            valid_count = valid.sum()
            if valid_count.item() == 0:
                continue
            valid_float = valid.to(student_slice.dtype)
            denominator_count = valid_count.to(student_slice.dtype) * hidden_size
            squared_error = (
                ((student_slice - teacher_slice).square() * valid_float).sum()
                / denominator_count
            )
            teacher_energy = (
                (teacher_slice.square() * valid_float).sum() / denominator_count
            ).clamp_min(epsilon)
        bucket_loss = squared_error / teacher_energy
        bucket_tensors.append(bucket_loss)

    if not bucket_tensors:
        raise ValueError("no valid tokens were available in any position bucket")
    stacked = torch.stack(bucket_tensors)
    return stacked.mean(), stacked.detach()


def validate_distill_manifest(manifest: dict, data_dir: Path) -> dict:
    if int(manifest.get("format_version", -1)) != 1:
        raise ValueError("distillation manifest must use format_version=1")
    if manifest.get("purpose") != "llama8b_positional_hidden_distillation":
        raise ValueError("distillation manifest has the wrong purpose")
    if int(manifest.get("seed", -1)) != 42:
        raise ValueError("distillation manifest must record seed=42")
    if int(manifest.get("seq_len", -1)) != 8192:
        raise ValueError("distillation manifest must record seq_len=8192")
    if int(manifest.get("train_sequences", 0)) != 2400:
        raise ValueError("distillation manifest must contain exactly 2400 train sequences")
    if int(manifest.get("validation_sequences", 0)) != 128:
        raise ValueError("distillation manifest must contain exactly 128 validation sequences")
    if manifest.get("split_policy") != "document_disjoint_validation_then_train":
        raise ValueError("distillation manifest must use document-disjoint splits")
    tokenizer_record = manifest.get("tokenizer")
    if not isinstance(tokenizer_record, dict) or not tokenizer_record.get("identifier"):
        raise ValueError("distillation manifest must fingerprint the tokenizer")
    for split in ("train", "validation"):
        record = manifest.get("files", {}).get(split, {})
        name = record.get("name")
        expected_hash = record.get("sha256")
        if not isinstance(name, str) or not isinstance(expected_hash, str):
            raise ValueError(f"manifest is missing {split} file identity")
        path = Path(data_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"frozen {split} tensor is missing")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"frozen {split} tensor hash mismatch")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        expected_rows = 2400 if split == "train" else 128
        if tensor.dtype != torch.int32 or tuple(tensor.shape) != (expected_rows, 8192):
            raise ValueError(
                f"frozen {split} tensor must be int32 with shape "
                f"[{expected_rows}, 8192]"
            )
    return manifest


def validate_claim_protocol(
    *,
    student_method: str,
    tau: float,
    seed: int,
    max_steps: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
    max_grad_norm: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_targets: Sequence[str],
    bf16: bool,
) -> None:
    expected = {
        "seed": (seed, 42),
        "max_steps": (max_steps, 1 if student_method == "native_geo" else 300),
        "effective_batch_size": (
            per_device_batch_size * gradient_accumulation_steps,
            8,
        ),
        "learning_rate": (learning_rate, 2e-5),
        "warmup_steps": (warmup_steps, 0 if student_method == "native_geo" else 30),
        "weight_decay": (weight_decay, 0.01),
        "max_grad_norm": (max_grad_norm, 1.0),
        "lora_r": (lora_r, 64),
        "lora_alpha": (lora_alpha, 128),
        "lora_dropout": (lora_dropout, 0.0),
        "lora_targets": (tuple(lora_targets), ("q_proj", "k_proj")),
        "bf16": (bf16, True),
    }
    if student_method == "evq_cosh":
        expected["tau"] = (tau, 1.414)
    for name, (actual, required) in expected.items():
        if actual != required:
            raise ValueError(
                f"approved claim protocol requires {name}={required!r}; found {actual!r}"
            )


def validate_single_gpu_runtime() -> dict:
    if sys.version_info < (3, 11):
        raise RuntimeError("the locked experiment stack requires Python 3.11+")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("positional distillation currently supports WORLD_SIZE=1 only")
    cuda_selector = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_selector or "," in cuda_selector:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one GPU index or UUID")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one CUDA GPU for the claim run")
    capability = tuple(torch.cuda.get_device_capability(0))
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the visible CUDA GPU must support BF16")
    if capability >= (12, 0):
        cuda_version = tuple(int(item) for item in str(torch.version.cuda).split(".")[:2])
        if cuda_version < (12, 8):
            raise RuntimeError("Blackwell requires a PyTorch CUDA 12.8+ build")
        if "sm_120" not in torch.cuda.get_arch_list():
            raise RuntimeError("PyTorch build lacks native sm_120 kernels")
    q = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    torch.cuda.synchronize()
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    package_versions = {
        package: importlib.metadata.version(package)
        for package in ("transformers", "peft", "accelerate", "datasets")
    }
    driver_record = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            cuda_selector,
            "--query-gpu=driver_version,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if not driver_record or "\n" in driver_record:
        raise RuntimeError("exactly one NVIDIA driver/GPU record is required")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_capability": list(capability),
        "cuda_arch_list": torch.cuda.get_arch_list(),
        "device_name": torch.cuda.get_device_name(0),
        "device_total_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 3
        ),
        "bf16_supported": True,
        "flash_sdpa_smoke": True,
        "sdpa_policy": "flash_only",
        "packages": package_versions,
        "nvidia_driver_and_gpu_uuid": driver_record,
        "cuda_visible_devices": cuda_selector,
    }


def build_distill_metadata(
    *,
    model_name: str,
    student_method: str,
    tau: float,
    seed: int,
    max_steps: int,
    seq_len: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
    max_grad_norm: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_targets: Iterable[str],
    data_manifest_sha256: str,
    train_time_hours: float,
    final_loss: float,
    peak_cuda_allocated_gb: float,
    peak_cuda_reserved_gb: float,
    base_model_fingerprint: dict,
    adapter_sha256: str,
) -> dict:
    targets = [str(target) for target in lora_targets]
    return {
        "format_version": 1,
        "model": public_model_identifier(model_name),
        "objective": "positional_hidden_distillation",
        "teacher_method": "native_geo",
        "student_method": student_method,
        "rope_method": student_method,
        "tau": tau if student_method == "evq_cosh" else None,
        "seed": seed,
        "max_steps": max_steps,
        "max_seq_len": seq_len,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": per_device_batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_targets": targets,
        "position_buckets": [list(item) for item in position_bucket_ranges(seq_len)],
        "data_manifest_sha256": data_manifest_sha256,
        "train_time_hours": round(train_time_hours, 4),
        "train_loss_final": float(final_loss),
        "peak_cuda_allocated_gb": round(float(peak_cuda_allocated_gb), 3),
        "peak_cuda_reserved_gb": round(float(peak_cuda_reserved_gb), 3),
        "base_model_fingerprint": base_model_fingerprint,
        "adapter_sha256": adapter_sha256,
    }


class FrozenSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, tensor: torch.Tensor):
        if tensor.ndim != 2 or tensor.shape[1] != 8192:
            raise ValueError("frozen tensor must have shape [N, 8192]")
        self.tensor = tensor

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        input_ids = self.tensor[index].to(torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def collate_frozen_sequences(features: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in features]),
    }


def causal_backbone(model):
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(base_model, "model", None)
    if backbone is None:
        raise RuntimeError("expected a causal LM with a .model transformer backbone")
    return backbone


def build_compiled_student_backbone(
    model,
    *,
    enabled: bool,
    backend: str,
    mode: str,
):
    """Compile only the gradient-bearing student callable.

    The eager backbone remains registered on the PEFT model, so checkpoint keys
    and the teacher path are unchanged. This also avoids relying on Trainer's
    outer-model compile wrapper, which this custom loss does not call.
    """
    backbone = causal_backbone(model)
    if not enabled:
        return backbone
    return torch.compile(
        backbone,
        backend=backend,
        mode=mode,
        dynamic=False,
    )


class FrequencySwitcher:
    """Cache rotary module references and device-local frequency tensors."""

    def __init__(
        self,
        model,
        teacher_inv_freq: torch.Tensor,
        student_inv_freq: torch.Tensor,
    ) -> None:
        modules = [module for _, module in find_rotary_modules(model)]
        if not modules:
            raise RuntimeError("no rotary modules were found for frequency switching")
        self.entries = [
            {
                "module": module,
                "teacher": teacher_inv_freq.detach(),
                "student": student_inv_freq.detach(),
            }
            for module in modules
        ]

    def install(self, which: str) -> None:
        if which not in {"teacher", "student"}:
            raise ValueError("frequency switch must be teacher or student")
        for entry in self.entries:
            module = entry["module"]
            old = module.inv_freq
            value = entry[which]
            if value.device != old.device or value.dtype != old.dtype:
                value = value.to(device=old.device, dtype=old.dtype)
                entry[which] = value
            with torch.no_grad():
                old.copy_(value)
            for attr in (
                "_cos_cached",
                "_sin_cached",
                "cos_cached",
                "sin_cached",
                "_cos_cache",
                "_sin_cache",
                "max_seq_len_cached",
            ):
                if hasattr(module, attr):
                    cached = getattr(module, attr)
                    setattr(module, attr, 0 if isinstance(cached, (int, float)) else None)


def snapshot_trainable_parameters(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def trainable_update_stats(model, initial: dict[str, torch.Tensor]) -> dict:
    parameter_sq = 0.0
    update_sq = 0.0
    update_max = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        current = parameter.detach().cpu().float()
        before = initial[name]
        delta = current - before
        parameter_sq += float(current.square().sum())
        update_sq += float(delta.square().sum())
        update_max = max(update_max, float(delta.abs().max()))
    return {
        "parameter_l2": math.sqrt(parameter_sq),
        "update_l2": math.sqrt(update_sq),
        "update_max_abs": update_max,
    }


class PositionalDistillationTrainer:
    """Factory namespace to avoid importing Transformers during unit tests."""

    @staticmethod
    def build(trainer_cls):
        class _Trainer(trainer_cls):
            def __init__(
                self,
                *args,
                teacher_inv_freq,
                student_inv_freq,
                compile_student,
                compile_backend,
                compile_mode,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)
                self.teacher_inv_freq = teacher_inv_freq
                self.student_inv_freq = student_inv_freq
                self.frequency_switcher = FrequencySwitcher(
                    self.model,
                    teacher_inv_freq,
                    student_inv_freq,
                )
                self.student_backbone = build_compiled_student_backbone(
                    self.model,
                    enabled=compile_student,
                    backend=compile_backend,
                    mode=compile_mode,
                )
                self.last_bucket_losses: Optional[torch.Tensor] = None
                self.last_loss: Optional[torch.Tensor] = None

            def compute_loss(
                self,
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=None,
            ):
                del num_items_in_batch
                input_ids = inputs["input_ids"]
                backbone = causal_backbone(model)

                self.frequency_switcher.install("teacher")
                model.eval()
                disable_adapter = getattr(model, "disable_adapter", None)
                if disable_adapter is None:
                    raise RuntimeError("installed PEFT version lacks disable_adapter()")
                with disable_adapter(), torch.no_grad():
                    teacher_hidden = backbone(
                        input_ids=input_ids,
                        attention_mask=None,
                        use_cache=False,
                        return_dict=True,
                    ).last_hidden_state.detach()

                self.frequency_switcher.install("student")
                model.train()
                student_outputs = self.student_backbone(
                    input_ids=input_ids,
                    attention_mask=None,
                    use_cache=False,
                    return_dict=True,
                )
                loss, bucket_losses = normalized_bucket_hidden_mse(
                    student_outputs.last_hidden_state,
                    teacher_hidden,
                    None,
                    position_bucket_ranges(input_ids.shape[1]),
                    assume_all_tokens_valid=True,
                )
                self.last_bucket_losses = bucket_losses
                self.last_loss = loss.detach()
                if return_outputs:
                    return loss, {"last_hidden_state": student_outputs.last_hidden_state}
                return loss

        return _Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--student_method",
        choices=("native_geo", "evq_cosh"),
        default="evq_cosh",
    )
    parser.add_argument("--tau", type=float, default=1.414)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", default="q_proj,k_proj")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile_backend", default="inductor")
    parser.add_argument(
        "--compile_mode",
        choices=("default", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument(
        "--performance_probe_steps",
        type=int,
        default=0,
        help="run a non-claim 12-step throughput probe instead of training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.performance_probe_steps not in (0, 12):
        raise ValueError("performance probes use exactly 12 optimizer steps")
    is_performance_probe = args.performance_probe_steps > 0
    objective = (
        "positional_hidden_distillation_performance_probe"
        if is_performance_probe
        else "positional_hidden_distillation"
    )
    targets = tuple(item.strip() for item in args.lora_targets.split(",") if item.strip())
    validate_claim_protocol(
        student_method=args.student_method,
        tau=args.tau,
        seed=args.seed,
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=targets,
        bf16=args.bf16,
    )
    runtime = validate_single_gpu_runtime()

    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_distill_manifest(manifest, args.data_dir)
    manifest_hash = sha256_file(manifest_path)
    base_model_fingerprint = fingerprint_model_source(args.model_name)
    run_protocol = {
        "format_version": 1,
        "objective": objective,
        "student_method": args.student_method,
        "base_model_fingerprint": base_model_fingerprint,
        "data_manifest_sha256": manifest_hash,
        "scientific": {
            "tau": args.tau if args.student_method == "evq_cosh" else None,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": list(targets),
            "bf16": args.bf16,
        },
        "performance": {
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "compile": args.compile,
            "compile_backend": args.compile_backend if args.compile else None,
            "compile_mode": args.compile_mode if args.compile else None,
            "compile_dynamic": False if args.compile else None,
            "performance_probe_steps": args.performance_probe_steps,
        },
        "runtime": runtime,
        "code_sha256": {
            "train_positional_distill.py": sha256_file(Path(__file__)),
            "train_evq_lora.py": sha256_file(SCRIPT_DIR / "train_evq_lora.py"),
            "prepare_positional_distill_data.py": sha256_file(
                SCRIPT_DIR / "prepare_positional_distill_data.py"
            ),
            "validate_checkpoint_artifact.py": sha256_file(
                SCRIPT_DIR / "validate_checkpoint_artifact.py"
            ),
            "01_lora_positional_distill_seed42.sh": sha256_file(
                SCRIPT_DIR.parents[1]
                / "scripts"
                / "2026-07"
                / "01_lora_positional_distill_seed42.sh"
            ),
        },
    }
    run_protocol_path = ensure_immutable_run_protocol(args.output_dir, run_protocol)
    performance_probe_path = args.output_dir / "performance_probe.json"
    if is_performance_probe and performance_probe_path.exists():
        raise FileExistsError("performance probe result already exists")
    train_path = args.data_dir / manifest["files"]["train"]["name"]
    train_tensor = torch.load(train_path, map_location="cpu", weights_only=True)
    train_dataset = FrozenSequenceDataset(train_tensor)

    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    geometry = resolve_model_rope_geometry(config)
    if getattr(config, "model_type", None) != "llama":
        raise ValueError("the approved pilot requires a LLaMA checkpoint")
    if getattr(config, "rope_scaling", None) not in (None, {}):
        raise ValueError("the approved pilot requires default, non-dynamic RoPE")
    if (
        geometry.head_dim != 128
        or not math.isclose(geometry.rope_base, 500_000.0)
        or int(getattr(config, "max_position_embeddings", -1)) != 8192
    ):
        raise ValueError(
            "the approved pilot requires head_dim=128, rope_theta=500000, "
            "and max_position_embeddings=8192"
        )
    teacher_inv_freq, teacher_meta = build_training_inv_freq(
        rope_method="native_geo",
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=0.0,
    )
    student_inv_freq, student_meta = build_training_inv_freq(
        rope_method=args.student_method,
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=args.tau,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if manifest["tokenizer"] != tokenizer_source_fingerprint(args.model_name):
        raise RuntimeError("frozen data tokenizer fingerprint does not match the model")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    inject_inv_freq(model, student_inv_freq)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(targets),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or any(
        not (".q_proj." in name or ".k_proj." in name) for name in trainable_names
    ):
        raise RuntimeError("trainable parameter audit found a non-q/k parameter")
    verify_model_inv_freq(model, student_inv_freq)
    initial_trainable = snapshot_trainable_parameters(model)

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "max_steps": (
            args.performance_probe_steps if is_performance_probe else args.max_steps
        ),
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "optim": "adamw_torch_fused",
        "lr_scheduler_type": "cosine",
        "bf16": args.bf16,
        "fp16": not args.bf16,
        "logging_steps": args.logging_steps,
        "save_strategy": "no" if is_performance_probe else "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }
    training_kwargs.update(evaluation_strategy_kwargs(TrainingArguments))
    training_args = TrainingArguments(**training_kwargs)

    class ProbeTimingCallback(TrainerCallback):
        def __init__(self, warmup_optimizer_steps: int, total_optimizer_steps: int):
            self.warmup_optimizer_steps = warmup_optimizer_steps
            self.total_optimizer_steps = total_optimizer_steps
            self.started: Optional[float] = None
            self.elapsed: Optional[float] = None

        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if state.global_step == self.warmup_optimizer_steps:
                torch.cuda.synchronize()
                self.started = time.perf_counter()
            elif state.global_step == self.total_optimizer_steps:
                torch.cuda.synchronize()
                if self.started is None:
                    raise RuntimeError("performance probe timing never started")
                self.elapsed = time.perf_counter() - self.started
            return control

    probe_timing = (
        ProbeTimingCallback(
            warmup_optimizer_steps=2,
            total_optimizer_steps=args.performance_probe_steps,
        )
        if is_performance_probe
        else None
    )
    trainer_type = PositionalDistillationTrainer.build(Trainer)
    trainer = trainer_type(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_frozen_sequences,
        teacher_inv_freq=teacher_inv_freq,
        student_inv_freq=student_inv_freq,
        compile_student=args.compile,
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
        callbacks=[probe_timing] if probe_timing is not None else None,
    )
    if not next(model.parameters()).is_cuda:
        raise RuntimeError("Trainer did not place the complete model on the visible CUDA GPU")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    last_checkpoint = get_last_checkpoint(str(args.output_dir))
    starting_global_step = checkpoint_starting_global_step(
        Path(last_checkpoint) if last_checkpoint is not None else None
    )
    ledger_ending_step = invocation_ledger_ending_step(args.output_dir)
    if ledger_ending_step != starting_global_step:
        raise RuntimeError(
            "resume checkpoint lacks continuous invocation evidence; "
            f"ledger ends at {ledger_ending_step}, checkpoint starts at "
            f"{starting_global_step}"
        )
    torch.cuda.synchronize()
    started = time.time()
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    torch.cuda.synchronize()
    train_time_hours = (time.time() - started) / 3600.0
    ending_global_step = int(trainer.state.global_step)
    if ending_global_step < starting_global_step:
        raise RuntimeError("Trainer global_step moved backwards during resume")
    invocation_steps = ending_global_step - starting_global_step
    peak_cuda_allocated_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    peak_cuda_reserved_gb = (
        torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    if trainer.last_loss is not None:
        final_loss = float(trainer.last_loss.cpu())
    else:
        logged_losses = [
            float(row["loss"])
            for row in trainer.state.log_history
            if isinstance(row, dict) and row.get("loss") is not None
        ]
        if not logged_losses:
            raise RuntimeError("completed training has no recoverable final loss")
        final_loss = logged_losses[-1]
    if not math.isfinite(final_loss):
        raise RuntimeError("completed training produced a non-finite final loss")

    if is_performance_probe:
        if probe_timing is None or probe_timing.elapsed is None:
            raise RuntimeError("performance probe did not produce steady-state timing")
        timed_steps = args.performance_probe_steps - probe_timing.warmup_optimizer_steps
        timed_tokens = (
            timed_steps
            * args.per_device_batch_size
            * args.gradient_accumulation_steps
            * manifest["seq_len"]
        )
        probe = {
            "format_version": 1,
            "objective": objective,
            "not_claim_artifact": True,
            "student_method": args.student_method,
            "run_protocol_sha256": sha256_file(run_protocol_path),
            "optimizer_steps": args.performance_probe_steps,
            "timing_warmup_steps": probe_timing.warmup_optimizer_steps,
            "timed_optimizer_steps": timed_steps,
            "steady_state_seconds": probe_timing.elapsed,
            "steady_state_nominal_tokens_per_second": timed_tokens
            / probe_timing.elapsed,
            "end_to_end_seconds": train_time_hours * 3600.0,
            "peak_cuda_allocated_gb": round(peak_cuda_allocated_gb, 3),
            "peak_cuda_reserved_gb": round(peak_cuda_reserved_gb, 3),
            "final_loss": final_loss,
            "final_bucket_losses": (
                trainer.last_bucket_losses.cpu().tolist()
                if trainer.last_bucket_losses is not None
                else None
            ),
            "runtime": runtime,
            "performance": run_protocol["performance"],
        }
        performance_probe_path.write_text(
            json.dumps(probe, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            "performance probe: "
            f"{probe['steady_state_nominal_tokens_per_second']:.0f} nominal tokens/s, "
            f"peak reserved {peak_cuda_reserved_gb:.1f} GiB"
        )
        return

    inject_inv_freq(model, student_inv_freq)
    verify_model_inv_freq(model, student_inv_freq)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    adapter_path = args.output_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError("PEFT did not write adapter_model.safetensors")
    adapter_sha256 = sha256_file(adapter_path)
    torch.save(
        {
            "inv_freq": student_inv_freq.detach().cpu(),
            "method": student_meta["method"],
            "tau": student_meta["tau"],
            "midpoint": student_meta["midpoint"],
            "head_dim": geometry.head_dim,
            "base": geometry.rope_base,
        },
        args.output_dir / "custom_inv_freq.pt",
    )
    metadata = build_distill_metadata(
        model_name=args.model_name,
        student_method=args.student_method,
        tau=args.tau,
        seed=args.seed,
        max_steps=args.max_steps,
        seq_len=manifest["seq_len"],
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=targets,
        data_manifest_sha256=manifest_hash,
        train_time_hours=train_time_hours,
        final_loss=final_loss,
        peak_cuda_allocated_gb=peak_cuda_allocated_gb,
        peak_cuda_reserved_gb=peak_cuda_reserved_gb,
        base_model_fingerprint=base_model_fingerprint,
        adapter_sha256=adapter_sha256,
    )
    metadata["teacher_frequency"] = teacher_meta
    metadata["student_frequency"] = student_meta
    metadata["runtime"] = runtime
    metadata["run_protocol_sha256"] = sha256_file(run_protocol_path)
    metadata["compile"] = {
        "enabled": args.compile,
        "scope": "student_backbone",
        "backend": args.compile_backend if args.compile else None,
        "mode": args.compile_mode if args.compile else None,
        "dynamic": False if args.compile else None,
    }
    metadata["gradient_checkpointing"] = args.gradient_checkpointing
    metadata["optimizer"] = "adamw_torch_fused"
    metadata["resumed_from_checkpoint"] = (
        Path(last_checkpoint).name if last_checkpoint is not None else None
    )
    metadata["train_time_hours"] = (
        metadata["train_time_hours"] if starting_global_step == 0 else None
    )
    metadata["train_loss_mean"] = (
        float(train_result.metrics["train_loss"])
        if starting_global_step == 0
        else None
    )
    metadata["train_samples_per_second"] = (
        train_result.metrics.get("train_samples_per_second")
        if starting_global_step == 0
        else None
    )
    metadata["train_steps_per_second"] = (
        train_result.metrics.get("train_steps_per_second")
        if starting_global_step == 0
        else None
    )
    invocation_tokens = (
        invocation_steps
        * args.per_device_batch_size
        * args.gradient_accumulation_steps
        * manifest["seq_len"]
    )
    metadata["invocation"] = {
        "starting_global_step": starting_global_step,
        "ending_global_step": ending_global_step,
        "optimizer_steps": invocation_steps,
        "nominal_tokens": invocation_tokens,
        "time_hours": round(train_time_hours, 4),
        "nominal_tokens_per_second": (
            invocation_tokens / (train_time_hours * 3600.0)
            if invocation_tokens
            else 0.0
        ),
    }
    metadata["nominal_train_tokens"] = (
        args.max_steps
        * args.per_device_batch_size
        * args.gradient_accumulation_steps
        * manifest["seq_len"]
    )
    metadata["nominal_train_tokens_per_second"] = (
        metadata["nominal_train_tokens"] / (train_time_hours * 3600.0)
        if starting_global_step == 0
        else None
    )
    metadata["trainable_parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    metadata["final_bucket_losses"] = (
        trainer.last_bucket_losses.cpu().tolist()
        if trainer.last_bucket_losses is not None
        else None
    )
    metadata["trainable_update"] = trainable_update_stats(model, initial_trainable)
    trainer.save_state()
    (args.output_dir / "experiment_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"completed {args.student_method} positional distillation in "
        f"{train_time_hours:.2f} hours"
    )


if __name__ == "__main__":
    main()
