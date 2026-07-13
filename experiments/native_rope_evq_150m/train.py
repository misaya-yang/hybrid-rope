#!/usr/bin/env python3
"""Matched, compiled training for one registered 151.9M frequency arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.native_rope_evq_150m.prepare_data import (
    sha256_file,
    validate_data_manifest,
)
from experiments.native_rope_evq_150m.model import GPT
from experiments.native_rope_evq_150m.protocol import (
    ARMS,
    SPEC,
    ExperimentSpec,
    estimate_parameter_count,
    get_arm_inv_freq,
)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def deterministic_row_order(n_rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(int(n_rows), generator=generator, dtype=torch.int64)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def registered_inv_freq_sha256(arm: str) -> str:
    """Hash the canonical float64 schedule, not a rounded runtime copy."""
    return tensor_sha256(get_arm_inv_freq(arm))


def trainable_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        value = parameter.detach().cpu().float().contiguous()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_model(
    arm: str,
    *,
    spec: ExperimentSpec = SPEC,
    seed: int = SPEC.seed,
) -> GPT:
    seed_everything(seed)
    inv_freq = get_arm_inv_freq(arm, spec=spec).float()
    return GPT(spec.model_config(), inv_freq)


def meta_parameter_count(
    arm: str,
    *,
    spec: ExperimentSpec = SPEC,
) -> int:
    """Count the registered model parameters without allocating model storage."""
    inv_freq = get_arm_inv_freq(arm, spec=spec).float()
    with torch.device("meta"):
        model = GPT(spec.model_config(), inv_freq.to("meta"))
    return sum(parameter.numel() for parameter in model.parameters())


def learning_rate_for_step(step: int, spec: ExperimentSpec = SPEC) -> float:
    step_i = int(step)
    if step_i < 0 or step_i >= spec.optimizer_steps:
        raise ValueError(
            f"step must be in [0, {spec.optimizer_steps}), got {step_i}"
        )
    if step_i < spec.warmup_steps:
        return spec.learning_rate * step_i / max(spec.warmup_steps, 1)
    decay_steps = max(spec.optimizer_steps - spec.warmup_steps - 1, 1)
    progress = min(max((step_i - spec.warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return spec.min_learning_rate + (
        spec.learning_rate - spec.min_learning_rate
    ) * cosine


class FrozenMixedDataset(Dataset[torch.Tensor]):
    """Memmap natural rows with frozen Passkey substitutions."""

    def __init__(
        self,
        *,
        train_path: str | Path,
        passkey_path: str | Path,
        indices_path: str | Path,
        expected_rows: int,
        seq_len: int,
    ) -> None:
        self.train_path = str(Path(train_path).resolve())
        self.passkey_path = str(Path(passkey_path).resolve())
        self.indices_path = str(Path(indices_path).resolve())
        self.expected_rows = int(expected_rows)
        self.seq_len = int(seq_len)
        indices = np.load(self.indices_path, allow_pickle=False)
        if indices.ndim != 1 or indices.dtype != np.int64:
            raise ValueError("passkey indices must be a one-dimensional int64 tensor")
        if len(indices) and (
            int(indices.min()) < 0 or int(indices.max()) >= self.expected_rows
        ):
            raise ValueError("passkey index is outside the registered training rows")
        if len(set(indices.tolist())) != len(indices):
            raise ValueError("passkey indices contain duplicates")
        self.lookup = np.full(self.expected_rows, -1, dtype=np.int32)
        self.lookup[indices] = np.arange(len(indices), dtype=np.int32)
        self._train: np.ndarray | None = None
        self._passkeys: np.ndarray | None = None
        self._open_and_validate()
        self._train = None
        self._passkeys = None

    def _open_and_validate(self) -> None:
        if self._train is None:
            self._train = np.load(self.train_path, mmap_mode="r", allow_pickle=False)
        if self._passkeys is None:
            self._passkeys = np.load(
                self.passkey_path, mmap_mode="r", allow_pickle=False
            )
        expected_train = (self.expected_rows, self.seq_len)
        expected_passkeys = (int((self.lookup >= 0).sum()), self.seq_len)
        if self._train.shape != expected_train or self._train.dtype != np.int64:
            raise ValueError(
                f"train tensor must be int64 {expected_train}, got "
                f"{self._train.dtype} {self._train.shape}"
            )
        if self._passkeys.shape != expected_passkeys or self._passkeys.dtype != np.int64:
            raise ValueError(
                f"passkey tensor must be int64 {expected_passkeys}, got "
                f"{self._passkeys.dtype} {self._passkeys.shape}"
            )

    def __len__(self) -> int:
        return self.expected_rows

    def __getitem__(self, index: int) -> torch.Tensor:
        self._open_and_validate()
        idx = int(index)
        passkey_index = int(self.lookup[idx])
        source = (
            self._train[idx]
            if passkey_index < 0
            else self._passkeys[passkey_index]
        )
        return torch.from_numpy(np.array(source, dtype=np.int64, copy=True))


class CausalLanguageModelLoss(nn.Module):
    def __init__(self, model: GPT) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = self.model(batch[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
        )


def validate_cuda_runtime() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered training command")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the registered training command requires CUDA BF16 support")
    props = torch.cuda.get_device_properties(0)
    total_memory = getattr(props, "total_memory", getattr(props, "total_mem", 0))
    return {
        "name": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": int(total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _load_manifest(path: Path, *, full_hash_check: bool) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    validate_data_manifest(manifest, check_files=full_hash_check)
    if not full_hash_check:
        for record, key in (
            (manifest["train"], "path"),
            (manifest["passkey"], "indices_path"),
            (manifest["passkey"], "passkey_path"),
            (manifest["validation"], "path"),
        ):
            if not Path(record[key]).is_file():
                raise FileNotFoundError(record[key])
    return manifest


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def _save_checkpoint(
    path: Path,
    model: GPT,
    metadata: dict[str, Any],
) -> str:
    temporary = path.with_name(path.name + ".incomplete")
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
    }
    torch.save({"model": state, "metadata": metadata}, temporary)
    temporary.replace(path)
    del state
    return sha256_file(path)


def run_dry(
    *, arm: str, manifest_path: Path, full_hash_check: bool
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path, full_hash_check=full_hash_check)
    train = np.load(manifest["train"]["path"], mmap_mode="r", allow_pickle=False)
    if train.shape != (SPEC.train_rows, SPEC.seq_len):
        raise ValueError(f"unexpected training tensor shape: {train.shape}")
    inv = get_arm_inv_freq(arm)
    parameter_count = meta_parameter_count(arm)
    if parameter_count != estimate_parameter_count():
        raise RuntimeError(
            f"parameter count mismatch: {parameter_count} != "
            f"{estimate_parameter_count()}"
        )
    report = {
        "arm": arm,
        "dry_run": True,
        "parameter_count": parameter_count,
        "train_shape": list(train.shape),
        "batch_size": SPEC.batch_size,
        "optimizer_steps": SPEC.optimizer_steps,
        "passkey_rows": int(manifest["passkey"]["rows"]),
        "inv_freq_sha256": tensor_sha256(inv),
        "grid": "endpoint u=k/K",
        "tau": None if arm == "native_rope" else SPEC.evq_tau,
        "data_manifest_sha256": sha256_file(manifest_path),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def train_arm(args: argparse.Namespace) -> dict[str, Any]:
    runtime = validate_cuda_runtime()
    manifest_path = Path(args.data_manifest).resolve()
    manifest = _load_manifest(
        manifest_path, full_hash_check=bool(args.full_hash_check)
    )
    output = Path(args.work_dir).resolve() / args.arm
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_everything(SPEC.seed)
    model = build_model(args.arm, spec=SPEC, seed=SPEC.seed)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != estimate_parameter_count():
        raise RuntimeError(
            f"parameter count mismatch: {parameter_count} != {estimate_parameter_count()}"
        )
    initial_hash = trainable_state_sha256(model)
    inv_freq = get_arm_inv_freq(args.arm).float()
    inv_hash = registered_inv_freq_sha256(args.arm)
    np.save(output / "inv_freq.npy", inv_freq.numpy())

    order = deterministic_row_order(SPEC.train_rows, SPEC.seed)
    order_hash = tensor_sha256(order)
    dataset = FrozenMixedDataset(
        train_path=manifest["train"]["path"],
        passkey_path=manifest["passkey"]["passkey_path"],
        indices_path=manifest["passkey"]["indices_path"],
        expected_rows=SPEC.train_rows,
        seq_len=SPEC.seq_len,
    )
    workers = max(0, int(args.num_workers))
    loader = DataLoader(
        dataset,
        batch_size=SPEC.batch_size,
        sampler=order.tolist(),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=False,
    )
    if len(loader) != SPEC.optimizer_steps:
        raise RuntimeError(f"loader has {len(loader)} steps, expected {SPEC.optimizer_steps}")

    model = model.to("cuda")
    loss_module = CausalLanguageModelLoss(model)
    compiled = torch.compile(
        loss_module,
        mode=str(args.compile_mode),
        dynamic=False,
        fullgraph=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SPEC.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=SPEC.weight_decay,
        fused=True,
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "arm": args.arm,
        "public_label": (
            "native endpoint RoPE"
            if args.arm == "native_rope"
            else "endpoint EVQ-Cosh tau=1.5"
        ),
        "frequency_grid": "endpoint u=k/K",
        "tau": None if args.arm == "native_rope" else SPEC.evq_tau,
        "base": SPEC.rope_base,
        "seed": SPEC.seed,
        "parameter_count": parameter_count,
        "model_config": SPEC.model_config(),
        "optimizer": {
            "name": "fused AdamW",
            "lr": SPEC.learning_rate,
            "min_lr": SPEC.min_learning_rate,
            "betas": [0.9, 0.95],
            "weight_decay": SPEC.weight_decay,
            "warmup_steps": SPEC.warmup_steps,
            "max_grad_norm": 1.0,
        },
        "precision": "FP32 master weights + BF16 autocast",
        "compile": {"enabled": True, "mode": args.compile_mode},
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
        "gradient_checkpointing": False,
        "initial_trainable_sha256": initial_hash,
        "inv_freq_sha256": inv_hash,
        "row_order_sha256": order_hash,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": sha256_file(manifest_path),
        "passkey_rows": int(manifest["passkey"]["rows"]),
        "passkey_tokens": int(manifest["passkey"]["tokens"]),
        "passkey_target_ratio": float(manifest["passkey"]["ratio"]),
        "passkey_budget_rationale": (
            "approximately 10M synthetic tokens, matching historical "
            "100M-token x 10% absolute exposure"
        ),
        "runtime": runtime,
        "python": platform.python_version(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(output / "train_meta.json", metadata)
    log_path = output / "train_log.jsonl"

    model.train()
    started = time.time()
    previous = started
    tokens_since_log = 0
    for step, batch in enumerate(loader):
        if tuple(batch.shape) != (SPEC.batch_size, SPEC.seq_len):
            raise RuntimeError(f"unexpected batch shape at step {step}: {batch.shape}")
        lr = learning_rate_for_step(step, SPEC)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = batch.to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = compiled(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite grad norm at step {step}: {grad_norm.item()}")
        optimizer.step()
        tokens_since_log += SPEC.batch_size * (SPEC.seq_len - 1)

        should_log = step == 0 or (step + 1) % int(args.log_every) == 0
        if should_log or step + 1 == SPEC.optimizer_steps:
            torch.cuda.synchronize()
            now = time.time()
            interval = max(now - previous, 1e-6)
            elapsed = now - started
            completed = step + 1
            eta = elapsed / completed * (SPEC.optimizer_steps - completed)
            record = {
                "step": step,
                "completed_steps": completed,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr": lr,
                "tokens_per_second": tokens_since_log / interval,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
            _append_jsonl(log_path, record)
            print(
                f"[{args.arm}] {completed}/{SPEC.optimizer_steps} "
                f"loss={record['loss']:.4f} lr={lr:.2e} "
                f"tok/s={record['tokens_per_second']:.0f} "
                f"mem={record['memory_allocated_gib']:.1f}GiB "
                f"ETA={eta / 60:.1f}m",
                flush=True,
            )
            previous = now
            tokens_since_log = 0

    torch.cuda.synchronize()
    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metadata["train_seconds"] = time.time() - started
    metadata["optimizer_steps"] = SPEC.optimizer_steps
    checkpoint_path = output / "model.pt"
    checkpoint_sha = _save_checkpoint(checkpoint_path, model, metadata)
    metadata["checkpoint_sha256"] = checkpoint_sha
    _write_json(output / "train_meta.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--data_manifest", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument(
        "--compile_mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--full_hash_check", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        run_dry(
            arm=args.arm,
            manifest_path=Path(args.data_manifest).resolve(),
            full_hash_check=bool(args.full_hash_check),
        )
        return
    train_arm(args)


if __name__ == "__main__":
    main()
