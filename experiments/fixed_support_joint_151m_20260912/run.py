#!/usr/bin/env python3
"""Qualify and run the paired 151.9M fixed-support S1 training trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from learnable_rope import install_joint_full_z
from protocol import ARMS, SEEDS, SPEC, SUPPORTS, learning_rate, table_for
from runtime import validate as validate_runtime


EXPECTED_WEIGHT_PARAMETERS = 151_898_880
PROBE_UPDATES = 10


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def weight_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.endswith("gap_delta_logits"):
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TwoEpochDataset(Dataset[torch.Tensor]):
    """Expose the locked 499,974,144-token prefix twice without copying it."""

    def __init__(self, base: Dataset[torch.Tensor], rows_per_epoch: int) -> None:
        self.base = base
        self.rows_per_epoch = int(rows_per_epoch)

    def __len__(self) -> int:
        return self.rows_per_epoch * SPEC.epochs

    def __getitem__(self, index: int) -> torch.Tensor:
        idx = int(index)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        return self.base[idx % self.rows_per_epoch]


def paired_order(old: Any, seed: int) -> torch.Tensor:
    parts = []
    for epoch in range(SPEC.epochs):
        order = old.deterministic_row_order(
            SPEC.rows_per_epoch, seed=int(seed) + epoch * 1_000_003
        )
        parts.append(order + epoch * SPEC.rows_per_epoch)
    return torch.cat(parts)


def build_model(old: Any, arm: str, support: int, seed: int):
    old.seed_everything(int(seed))
    initial_table = table_for(arm, support)
    model = old.GPT(SPEC.model_config(), initial_table)
    full_z = install_joint_full_z(model) if arm == "full_z" else None
    weight_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.endswith("gap_delta_logits")
    )
    if weight_count != EXPECTED_WEIGHT_PARAMETERS:
        raise RuntimeError(
            f"weight parameter count changed: {weight_count} != {EXPECTED_WEIGHT_PARAMETERS}"
        )
    if full_z is not None and not torch.equal(
        full_z.realized_inv_freq(), table_for("geo", support)
    ):
        raise RuntimeError("full-z initialization is not exactly fixed Geo")
    return model, full_z


def optimizer_for(model, full_z):
    allocation = [] if full_z is None else [full_z.gap_delta_logits]
    allocation_ids = {id(value) for value in allocation}
    weights = [value for value in model.parameters() if id(value) not in allocation_ids]
    groups = [
        {
            "name": "model_weights",
            "params": weights,
            "lr": SPEC.learning_rate,
            "weight_decay": SPEC.weight_decay,
        }
    ]
    if allocation:
        groups.append(
            {
                "name": "allocation",
                "params": allocation,
                "lr": SPEC.learning_rate * SPEC.allocation_lr_multiplier,
                "weight_decay": SPEC.allocation_weight_decay,
            }
        )
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--support", type=int, choices=SUPPORTS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--micro-batch", type=int, default=SPEC.micro_batch_size)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if SPEC.global_batch_size % int(args.micro_batch):
        raise ValueError("micro-batch must divide the locked global batch")

    sys.path.insert(0, str(args.original_root.resolve()))
    from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m import (  # noqa: PLC0415
        run_experiment as old,
    )
    runtime = validate_runtime(old)
    manifest_path = args.data_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "READY":
        raise ValueError("data manifest is not READY")
    if int(manifest["train"]["tokens"]) != SPEC.source_tokens_per_epoch:
        raise ValueError("training prefix does not match the locked source epoch")

    model, full_z = build_model(old, args.arm, args.support, args.seed)
    initial_weight_sha = weight_state_sha256(model)
    initial_table = (
        full_z.realized_inv_freq().detach().clone()
        if full_z is not None
        else model.blocks[0].attention.rope.inv_freq.detach().clone()
    )
    order = paired_order(old, args.seed)
    order_sha = tensor_sha256(order)

    class UIntDataset(old.FlatPrefixDataset):
        def _open(self):
            if self._array is None:
                self._array = np.load(self.path, mmap_mode="r", allow_pickle=False)
                if self._array.dtype not in (np.dtype("uint16"), np.dtype("int64")):
                    raise ValueError("unsupported lossless token storage dtype")
                self._flat = self._array.reshape(-1)

    base = UIntDataset(
        manifest["train"]["path"],
        rows=SPEC.rows_per_epoch,
        seq_len=SPEC.train_length,
    )
    dataset = TwoEpochDataset(base, SPEC.rows_per_epoch)

    scientific = {
        "schema_version": 1,
        "question": "fixed-support analytic allocation versus direct full-z joint learning",
        "arm": args.arm,
        "support": int(args.support),
        "seed": int(args.seed),
        "protocol_sha256": SPEC.fingerprint(),
        "model_config": SPEC.model_config(),
        "weight_parameter_count": EXPECTED_WEIGHT_PARAMETERS,
        "allocation_parameter_count": 0 if full_z is None else 31,
        "initial_weight_sha256": initial_weight_sha,
        "initial_table_sha256": tensor_sha256(initial_table),
        "row_order_sha256": order_sha,
        "source_tokens_per_epoch": SPEC.source_tokens_per_epoch,
        "source_epochs": SPEC.epochs,
        "source_epoch_policy": "same frozen prefix; independent deterministic permutation per epoch",
        "train_tokens": SPEC.train_tokens,
        "optimizer_steps": SPEC.optimizer_steps,
        "midpoint_step": SPEC.midpoint_step,
        "optimizer": {
            "weights": {"lr": SPEC.learning_rate, "weight_decay": SPEC.weight_decay},
            "allocation": {
                "lr_multiplier": SPEC.allocation_lr_multiplier,
                "weight_decay": SPEC.allocation_weight_decay,
                "projection": SPEC.allocation_projection,
            },
            "betas": [0.9, 0.95],
            "warmup_steps": SPEC.warmup_steps,
            "minimum_lr": SPEC.min_learning_rate,
            "max_grad_norm": SPEC.max_grad_norm,
        },
        "precision": "FP32 master weights, FP64 gap normalization, BF16 autocast",
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": sha256_file(manifest_path),
        "train_storage_sha256": manifest["train"]["sha256"],
        "validation_sha256": manifest["validation"]["sha256"],
    }
    signature = hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    start = 0
    loaded = None
    if args.resume:
        loaded = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        if loaded["signature"] != signature:
            raise ValueError("resume scientific contract mismatch")
        model.load_state_dict(loaded["model"])
        start = int(loaded["completed_updates"])

    model = model.cuda()
    if full_z is not None:
        full_z = model.blocks[0].attention.rope
    optimizer = optimizer_for(model, full_z)
    if loaded is not None:
        optimizer.load_state_dict(loaded["optimizer"])
        torch.set_rng_state(loaded["torch_rng"])
        torch.cuda.set_rng_state_all(loaded["cuda_rng"])
        np.random.set_state(loaded["numpy_rng"])
        random.setstate(loaded["python_rng"])
        del loaded

    loss_module = old.CausalLanguageModelLoss(model)
    if not args.no_compile:
        loss_module = torch.compile(
            loss_module,
            mode=str(args.compile_mode),
            dynamic=False,
            fullgraph=False,
        )
    loader = iter(
        DataLoader(
            dataset,
            batch_size=int(args.micro_batch),
            sampler=old.TensorOrderSampler(order[start * SPEC.global_batch_size :]),
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True,
        )
    )
    total = start + PROBE_UPDATES if args.probe else SPEC.optimizer_steps
    if args.probe and start:
        raise ValueError("discarded probe cannot resume")
    status = {
        **scientific,
        "signature": signature,
        "status": "PROBE_RUNNING" if args.probe else "RUNNING",
        "discarded": bool(args.probe),
        "pid": os.getpid(),
        "runtime": runtime,
        "compile_mode": None if args.no_compile else args.compile_mode,
        "micro_batch": int(args.micro_batch),
        "completed_updates": start,
    }
    atomic_json(output / "status.json", status)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    durations: list[float] = []
    first_allocation_update = None

    for step in range(start, total):
        torch.cuda.synchronize()
        tick = time.monotonic()
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            multiplier = SPEC.allocation_lr_multiplier if group["name"] == "allocation" else 1.0
            group["lr"] = lr * multiplier
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        accumulation = SPEC.global_batch_size // int(args.micro_batch)
        before_allocation = (
            None if full_z is None else full_z.gap_delta_logits.detach().clone()
        )
        for _ in range(accumulation):
            batch = next(loader).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_module(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at update {step}")
            (loss / accumulation).backward()
            loss_sum += float(loss.detach()) / accumulation
        allocation_grad_norm = None
        if full_z is not None:
            gradient = full_z.gap_delta_logits.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("full-z allocation gradient is absent or non-finite")
            allocation_grad_norm = float(torch.linalg.vector_norm(gradient.detach()))
            if step == 0 and allocation_grad_norm <= 0.0:
                raise RuntimeError("full-z first allocation gradient is zero")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), SPEC.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite global gradient norm")
        optimizer.step()
        if full_z is not None:
            full_z.fix_gauge_()
            update = float(
                (full_z.gap_delta_logits.detach() - before_allocation).abs().max()
            )
            if step == 0:
                if update <= 0.0:
                    raise RuntimeError("full-z optimizer did not update allocation")
                first_allocation_update = update
        torch.cuda.synchronize()
        seconds = time.monotonic() - tick
        durations.append(seconds)
        record = {
            "completed_updates": step + 1,
            "input_tokens": (step + 1) * SPEC.global_batch_size * SPEC.train_length,
            "loss": loss_sum,
            "lr": lr,
            "grad_norm": float(grad_norm),
            "allocation_grad_norm": allocation_grad_norm,
            "seconds": seconds,
            "tokens_per_second": SPEC.global_batch_size * SPEC.train_length / seconds,
        }
        with (output / "train.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step == start or (step + 1) % 25 == 0 or args.probe:
            print(json.dumps(record, sort_keys=True), flush=True)
            atomic_json(output / "live.json", record)
        if not args.probe and (step + 1) == SPEC.midpoint_step:
            atomic_torch_save(
                output / "model_500m.pt",
                {"model": model.state_dict(), "metadata": status, "completed_updates": step + 1},
            )
        if not args.probe and ((step + 1) % 512 == 0 or (step + 1) == total):
            atomic_torch_save(
                output / "resume.pt",
                {
                    "signature": signature,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "completed_updates": step + 1,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                },
            )

    status.update(
        {
            "status": "PROBE_PASS" if args.probe else "COMPLETE",
            "completed_updates": total,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "first_allocation_update_max_abs": first_allocation_update,
            "allocation": None if full_z is None else full_z.receipt(),
        }
    )
    if args.probe:
        steady = durations[2:]
        status["steady_tokens_per_second"] = (
            SPEC.global_batch_size * SPEC.train_length / float(np.mean(steady))
        )
    else:
        atomic_torch_save(
            output / "model_1b.pt",
            {"model": model.state_dict(), "metadata": status},
        )
    atomic_json(output / "status.json", status)
    print(json.dumps(status, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
