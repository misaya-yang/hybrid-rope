#!/usr/bin/env python3
"""Train q/k-only LoRA to preserve native-Geo hidden states after RoPE change."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_positional_distill_data import sha256_file
from train_evq_lora import (
    build_training_inv_freq,
    evaluation_strategy_kwargs,
    inject_inv_freq,
    public_model_identifier,
    resolve_model_rope_geometry,
    verify_model_inv_freq,
)


def position_bucket_ranges(seq_len: int) -> Tuple[Tuple[int, int], ...]:
    if seq_len != 8192:
        raise ValueError("the frozen pilot protocol requires seq_len=8192")
    return ((0, 2048), (2048, 4096), (4096, 8192))


def fingerprint_model_source(model_name: str) -> dict:
    """Return a path-safe identity without hashing multi-gigabyte weight shards."""
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
        {"name": path.name, "size_bytes": path.stat().st_size}
        for path in sorted(model_path.glob("model*.safetensors"))
    ]
    return fingerprint


def normalized_bucket_hidden_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
    attention_mask: torch.Tensor,
    buckets: Sequence[Tuple[int, int]],
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, list[float]]:
    if student.shape != teacher.shape:
        raise ValueError(
            f"hidden-state shape mismatch: student={tuple(student.shape)} "
            f"teacher={tuple(teacher.shape)}"
        )
    if attention_mask.shape != student.shape[:2]:
        raise ValueError("attention_mask must match hidden-state batch and sequence axes")

    bucket_tensors = []
    bucket_values: list[float] = []
    hidden_size = student.shape[-1]
    for start, end in buckets:
        end = min(end, student.shape[1])
        if start >= end:
            continue
        valid = attention_mask[:, start:end].to(torch.bool).unsqueeze(-1)
        valid_count = valid.sum()
        if valid_count.item() == 0:
            continue
        student_slice = student[:, start:end].float()
        teacher_slice = teacher[:, start:end].float()
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
        bucket_values.append(float(bucket_loss.detach().cpu()))

    if not bucket_tensors:
        raise ValueError("no valid tokens were available in any position bucket")
    return torch.stack(bucket_tensors).mean(), bucket_values


def validate_distill_manifest(manifest: dict, data_dir: Path) -> dict:
    if int(manifest.get("seq_len", -1)) != 8192:
        raise ValueError("distillation manifest must record seq_len=8192")
    if int(manifest.get("train_sequences", 0)) < 2400:
        raise ValueError("distillation manifest must contain at least 2400 train sequences")
    if int(manifest.get("validation_sequences", 0)) < 1:
        raise ValueError("distillation manifest must contain validation sequences")
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
    return manifest


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
        "attention_mask": torch.stack([item["attention_mask"] for item in features]),
    }


def causal_backbone(model):
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(base_model, "model", None)
    if backbone is None:
        raise RuntimeError("expected a causal LM with a .model transformer backbone")
    return backbone


class PositionalDistillationTrainer:
    """Factory namespace to avoid importing Transformers during unit tests."""

    @staticmethod
    def build(trainer_cls):
        class _Trainer(trainer_cls):
            def __init__(self, *args, teacher_inv_freq, student_inv_freq, **kwargs):
                super().__init__(*args, **kwargs)
                self.teacher_inv_freq = teacher_inv_freq
                self.student_inv_freq = student_inv_freq
                self.last_bucket_losses: list[float] = []

            def compute_loss(
                self,
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=None,
            ):
                del num_items_in_batch
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                backbone = causal_backbone(model)

                inject_inv_freq(model, self.teacher_inv_freq)
                model.eval()
                disable_adapter = getattr(model, "disable_adapter", None)
                if disable_adapter is None:
                    raise RuntimeError("installed PEFT version lacks disable_adapter()")
                with disable_adapter(), torch.no_grad():
                    teacher_hidden = backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    ).last_hidden_state.detach()

                inject_inv_freq(model, self.student_inv_freq)
                model.train()
                student_outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                loss, bucket_losses = normalized_bucket_hidden_mse(
                    student_outputs.last_hidden_state,
                    teacher_hidden,
                    attention_mask,
                    position_bucket_ranges(input_ids.shape[1]),
                )
                self.last_bucket_losses = bucket_losses
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
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 42 or args.max_steps != 300:
        raise ValueError("the approved pilot requires seed=42 and max_steps=300")
    targets = tuple(item.strip() for item in args.lora_targets.split(",") if item.strip())
    if targets != ("q_proj", "k_proj"):
        raise ValueError("the clean pilot permits only q_proj,k_proj LoRA targets")

    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_distill_manifest(manifest, args.data_dir)
    manifest_hash = sha256_file(manifest_path)
    train_path = args.data_dir / manifest["files"]["train"]["name"]
    train_tensor = torch.load(train_path, map_location="cpu", weights_only=True)
    train_dataset = FrozenSequenceDataset(train_tensor)

    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    geometry = resolve_model_rope_geometry(config)
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
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

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "bf16": args.bf16,
        "fp16": not args.bf16,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "seed": args.seed,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }
    training_kwargs.update(evaluation_strategy_kwargs(TrainingArguments))
    training_args = TrainingArguments(**training_kwargs)
    trainer_type = PositionalDistillationTrainer.build(Trainer)
    trainer = trainer_type(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_frozen_sequences,
        teacher_inv_freq=teacher_inv_freq,
        student_inv_freq=student_inv_freq,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    train_result = trainer.train()
    train_time_hours = (time.time() - started) / 3600.0
    peak_cuda_allocated_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    peak_cuda_reserved_gb = (
        torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
    )
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=targets,
        data_manifest_sha256=manifest_hash,
        train_time_hours=train_time_hours,
        final_loss=float(train_result.metrics["train_loss"]),
        peak_cuda_allocated_gb=peak_cuda_allocated_gb,
        peak_cuda_reserved_gb=peak_cuda_reserved_gb,
        base_model_fingerprint=fingerprint_model_source(args.model_name),
        adapter_sha256=adapter_sha256,
    )
    metadata["teacher_frequency"] = teacher_meta
    metadata["student_frequency"] = student_meta
    metadata["trainable_parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    metadata["final_bucket_losses"] = trainer.last_bucket_losses
    (args.output_dir / "experiment_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trainer.save_state()
    print(
        f"completed {args.student_method} positional distillation in "
        f"{train_time_hours:.2f} hours"
    )


if __name__ == "__main__":
    main()
