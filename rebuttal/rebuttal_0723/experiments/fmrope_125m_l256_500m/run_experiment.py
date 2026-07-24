#!/usr/bin/env python3
"""Train, evaluate, and summarize the matched FMRoPE / EVQ experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from experiments.native_rope_evq_150m.model import GPT
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.prepare import (
    sha256_file,
    validate_experiment_manifest,
)
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.protocol import (
    ARMS,
    ARM_CONDITIONS,
    SPEC,
    ExperimentSpec,
    estimate_parameter_count,
    frequency_contract,
    learning_rate_for_step,
    runtime_frequency,
    training_inv_freq,
)


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (
            (candidate / "AGENTS.md").is_file()
            and (candidate / "scripts/lib/rope/schedules.py").is_file()
        ):
            return candidate
    raise RuntimeError(f"could not locate repository root from {start}")


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _find_repo_root(PACKAGE_DIR)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def trainable_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        value = parameter.detach().cpu().float().contiguous()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def code_fingerprint() -> str:
    paths = (
        PACKAGE_DIR / "protocol.py",
        PACKAGE_DIR / "prepare.py",
        PACKAGE_DIR / "run_experiment.py",
        PACKAGE_DIR / "run_5090.sh",
        REPO_ROOT / "experiments/native_rope_evq_150m/model.py",
        REPO_ROOT / "rebuttal/rebuttal_0723/experiments/geo_rope_contract.py",
        REPO_ROOT / "rebuttal/rebuttal_0723/theory_results/FREQUENCY_DEFINITION_MANIFEST.json",
        REPO_ROOT / "scripts/lib/rope/schedules.py",
        REPO_ROOT / "scripts/lib/rope/official_yarn.py",
        REPO_ROOT / "tests/test_fmrope_125m_l256_500m.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"code fingerprint input is missing: {path}")
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def _save_checkpoint(
    path: Path, model: GPT, metadata: dict[str, Any]
) -> str:
    temporary = path.with_name(path.name + ".incomplete")
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    checkpoint_inv = _state_inv_freq(state, label="checkpoint save")
    if tensor_sha256(checkpoint_inv) != metadata.get(
        "training_inv_freq_sha256"
    ):
        raise ValueError("model inv_freq changed before checkpoint save")
    torch.save({"model": state, "metadata": metadata}, temporary)
    temporary.replace(path)
    del state
    return sha256_file(path)


def _state_inv_freq(
    state: dict[str, torch.Tensor], *, label: str
) -> torch.Tensor:
    values = [
        value.detach().cpu().float().contiguous()
        for name, value in state.items()
        if name.endswith(".rope.inv_freq")
    ]
    if not values:
        raise ValueError(f"{label} has no persistent inv_freq buffer")
    reference = values[0]
    if any(not torch.equal(value, reference) for value in values[1:]):
        raise ValueError(f"{label} contains inconsistent inv_freq buffers")
    return reference.clone()


def _model_inv_freq(model: GPT) -> torch.Tensor:
    rope = model.blocks[0].attention.rope
    if any(block.attention.rope is not rope for block in model.blocks):
        raise RuntimeError("model blocks do not share one rotary module")
    return rope.inv_freq.detach().cpu().float().contiguous().clone()


def _load_manifest(
    path: Path, *, full_hash_check: bool
) -> dict[str, Any]:
    manifest = json.loads(path.resolve().read_text())
    validate_experiment_manifest(
        manifest,
        check_files=True,
        check_content_hashes=bool(full_hash_check),
    )
    return manifest


def deterministic_row_order(
    n_rows: int, seed: int = SPEC.seed
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(int(n_rows), generator=generator, dtype=torch.int64)


class TensorOrderSampler(Sampler[int]):
    def __init__(self, order: torch.Tensor) -> None:
        if order.dtype != torch.int64 or order.ndim != 1:
            raise ValueError("row order must be a one-dimensional int64 tensor")
        self.order = order

    def __iter__(self) -> Iterator[int]:
        return (int(value) for value in self.order.tolist())

    def __len__(self) -> int:
        return int(self.order.numel())


class FlatPrefixDataset(Dataset[torch.Tensor]):
    """Read fixed L=256 rows from the registered flat token prefix."""

    def __init__(
        self,
        path: str | Path,
        *,
        rows: int = SPEC.train_rows,
        seq_len: int = SPEC.train_length,
    ) -> None:
        self.path = str(Path(path).resolve())
        self.rows = int(rows)
        self.seq_len = int(seq_len)
        self._array: np.ndarray | None = None
        self._flat: np.ndarray | None = None
        self._open()
        assert self._flat is not None
        if len(self._flat) < self.rows * self.seq_len:
            raise ValueError("training tensor is shorter than the registered prefix")
        self._array = None
        self._flat = None

    def _open(self) -> None:
        if self._array is None:
            self._array = np.load(
                self.path, mmap_mode="r", allow_pickle=False
            )
            if self._array.dtype != np.int64 or self._array.ndim != 2:
                raise ValueError(
                    "training tensor must be a two-dimensional int64 NPY"
                )
            self._flat = self._array.reshape(-1)

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> torch.Tensor:
        self._open()
        assert self._flat is not None
        idx = int(index)
        if idx < 0 or idx >= self.rows:
            raise IndexError(idx)
        start = idx * self.seq_len
        value = np.array(
            self._flat[start : start + self.seq_len],
            dtype=np.int64,
            copy=True,
        )
        return torch.from_numpy(value)


class CausalLanguageModelLoss(nn.Module):
    def __init__(self, model: GPT) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = self.model(batch[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
        )


def build_model(
    arm: str,
    *,
    spec: ExperimentSpec = SPEC,
    seed: int = SPEC.seed,
) -> GPT:
    seed_everything(seed)
    inv = training_inv_freq(arm, spec=spec).float()
    return GPT(spec.model_config(), inv)


def meta_parameter_count(
    arm: str, *, spec: ExperimentSpec = SPEC
) -> int:
    inv = training_inv_freq(arm, spec=spec).float()
    with torch.device("meta"):
        model = GPT(spec.model_config(), inv.to("meta"))
    return sum(parameter.numel() for parameter in model.parameters())


def configure_cuda_kernels() -> dict[str, Any]:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    reduced_bf16 = getattr(
        torch.backends.cuda.matmul,
        "allow_bf16_reduced_precision_reduction",
        None,
    )
    if reduced_bf16 is not None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    flash_available_fn = getattr(
        torch.backends.cuda, "is_flash_attention_available", None
    )
    if flash_available_fn is not None and not flash_available_fn():
        raise RuntimeError("PyTorch reports Flash SDPA unavailable")
    result = {
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "cudnn_sdp_enabled": (
            bool(torch.backends.cuda.cudnn_sdp_enabled())
            if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
            else None
        ),
    }
    if result["flash_sdp_enabled"] is not True or any(
        result[key] is True
        for key in (
            "mem_efficient_sdp_enabled",
            "math_sdp_enabled",
            "cudnn_sdp_enabled",
        )
    ):
        raise RuntimeError(f"Flash-only SDPA gate failed: {result}")
    return result


def validate_cuda_runtime() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training/evaluation")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the registered run requires BF16-capable CUDA")
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    compiled_architectures = list(torch.cuda.get_arch_list())
    if compiled_architectures and not any(
        item == architecture or item.startswith(architecture + "a")
        for item in compiled_architectures
    ):
        raise RuntimeError(
            f"PyTorch build lacks native {architecture}; compiled for "
            f"{compiled_architectures}"
        )
    total_memory = getattr(
        props, "total_memory", getattr(props, "total_mem", 0)
    )
    if int(total_memory) < 30 * 2**30:
        raise RuntimeError(
            f"registered micro-batch requires >=30 GiB, found "
            f"{int(total_memory) / 2**30:.1f} GiB"
        )
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable")
    return {
        "name": props.name,
        "capability": list(capability),
        "architecture": architecture,
        "compiled_architectures": compiled_architectures,
        "total_memory_bytes": int(total_memory),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "kernels": configure_cuda_kernels(),
        "torch_compile_available": True,
    }


def run_preflight(
    manifest_path: Path,
    *,
    full_hash_check: bool,
    verify_full_initialization: bool,
    arms: tuple[str, ...] = ARMS,
) -> dict[str, Any]:
    manifest = _load_manifest(
        manifest_path, full_hash_check=bool(full_hash_check)
    )
    parameter_counts = {
        arm: meta_parameter_count(arm) for arm in arms
    }
    expected = estimate_parameter_count()
    if set(parameter_counts.values()) != {expected}:
        raise RuntimeError(
            f"parameter count mismatch: {parameter_counts}, expected {expected}"
        )

    initial_hashes: dict[str, str] = {}
    if verify_full_initialization:
        for arm in arms:
            model = build_model(arm)
            initial_hashes[arm] = trainable_state_sha256(model)
            del model
            gc.collect()
        if len(set(initial_hashes.values())) != 1:
            raise RuntimeError(
                f"trainable initialization differs across arms: {initial_hashes}"
            )

    schedule_report: dict[str, Any] = {}
    for arm in arms:
        train_inv = training_inv_freq(arm)
        schedule_report[arm] = {
            "train_sha256": tensor_sha256(train_inv),
            "dtype": str(train_inv.numpy().dtype),
            "first_channels": [
                float(value) for value in train_inv[:4]
            ],
            "last_channels": [
                float(value) for value in train_inv[-4:]
            ],
            "min": float(train_inv.min()),
            "max": float(train_inv.max()),
            "conditions": {},
        }
        for condition in ARM_CONDITIONS[arm]:
            condition_rows = {}
            for length in SPEC.eval_lengths:
                inv, mscale, meta = runtime_frequency(
                    arm, condition, length
                )
                condition_rows[str(length)] = {
                    "inv_freq_sha256": tensor_sha256(inv),
                    "mscale": float(mscale),
                    "meta": meta,
                }
            schedule_report[arm]["conditions"][condition] = condition_rows

    report = {
        "status": "PASS",
        "model_tier": SPEC.model_tier,
        "seed": SPEC.seed,
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "data_manifest_sha256": sha256_file(manifest_path),
        "parameter_count": expected,
        "parameter_counts": parameter_counts,
        "initial_trainable_sha256": initial_hashes,
        "train_rows": SPEC.train_rows,
        "train_tokens": SPEC.train_tokens,
        "prediction_tokens": SPEC.prediction_tokens,
        "optimizer_steps": SPEC.optimizer_steps,
        "micro_steps": SPEC.micro_steps,
        "data": {
            "train_used_prefix_sha256": manifest["train"][
                "used_prefix_sha256"
            ],
            "validation_sha256": manifest["validation"]["sha256"],
            "anchors_sha256": manifest["anchors"]["sha256"],
        },
        "frequency_contract": frequency_contract(),
        "schedules": schedule_report,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def probe_gpu(args: argparse.Namespace) -> dict[str, Any]:
    """Run one compile step plus discarded steady-state steps."""
    runtime = validate_cuda_runtime()
    manifest = _load_manifest(
        Path(args.data_manifest).resolve(), full_hash_check=False
    )
    output = Path(args.work_dir).resolve() / f"gpu_probe_{args.arm}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GPU probe: {output}")
    source = np.load(
        manifest["train"]["path"], mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    token_count = SPEC.micro_batch_size * SPEC.train_length
    batch = torch.from_numpy(
        np.array(source[:token_count], dtype=np.int64, copy=True).reshape(
            SPEC.micro_batch_size, SPEC.train_length
        )
    ).to("cuda")
    model = build_model(args.arm).to("cuda")
    loss_module = torch.compile(
        CausalLanguageModelLoss(model),
        mode=args.compile_mode,
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
    torch.cuda.reset_peak_memory_stats()

    def step() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_module(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite probe loss: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss

    started = time.time()
    loss = step()
    torch.cuda.synchronize()
    compile_seconds = time.time() - started
    timed_steps = int(args.timed_steps)
    if timed_steps < 5:
        raise ValueError("the registered probe requires at least five timed steps")
    started = time.time()
    for _ in range(timed_steps):
        loss = step()
    torch.cuda.synchronize()
    timed_seconds = time.time() - started
    timed_tokens = timed_steps * token_count
    tokens_per_second = timed_tokens / timed_seconds
    report = {
        "status": "PASS",
        "discarded_probe": True,
        "arm": args.arm,
        "compile_mode": args.compile_mode,
        "timed_steps": timed_steps,
        "compile_step_seconds": compile_seconds,
        "timed_seconds": timed_seconds,
        "timed_tokens": timed_tokens,
        "tokens_per_second": tokens_per_second,
        "estimated_train_seconds": SPEC.train_tokens / tokens_per_second,
        "loss": float(loss.detach()),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "runtime": runtime,
    }
    _write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def train_arm(args: argparse.Namespace) -> dict[str, Any]:
    runtime = validate_cuda_runtime()
    manifest_path = Path(args.data_manifest).resolve()
    manifest = _load_manifest(
        manifest_path, full_hash_check=bool(args.full_hash_check)
    )
    output = Path(args.work_dir).resolve() / "runs" / args.arm
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(args.arm)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != estimate_parameter_count():
        raise RuntimeError(
            f"parameter count {parameter_count} != "
            f"{estimate_parameter_count()}"
        )
    initial_hash = trainable_state_sha256(model)
    inv = training_inv_freq(args.arm)
    inv_hash = tensor_sha256(inv)
    np.save(output / "inv_freq.npy", inv.numpy())

    order = deterministic_row_order(SPEC.train_rows)
    order_hash = tensor_sha256(order)
    dataset = FlatPrefixDataset(
        manifest["train"]["path"],
        rows=SPEC.train_rows,
        seq_len=SPEC.train_length,
    )
    workers = max(0, int(args.num_workers))
    loader = DataLoader(
        dataset,
        batch_size=SPEC.micro_batch_size,
        sampler=TensorOrderSampler(order),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=True,
    )
    if len(loader) != SPEC.micro_steps:
        raise RuntimeError(
            f"loader has {len(loader)} micro-steps, expected {SPEC.micro_steps}"
        )

    model = model.to("cuda")
    loss_module: nn.Module = CausalLanguageModelLoss(model)
    if not args.no_compile:
        loss_module = torch.compile(
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
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": sha256_file(manifest_path),
        "train_used_prefix_sha256": manifest["train"][
            "used_prefix_sha256"
        ],
        "validation_sha256": manifest["validation"]["sha256"],
        "anchors_sha256": manifest["anchors"]["sha256"],
        "seed": SPEC.seed,
        "model_tier": SPEC.model_tier,
        "parameter_count": parameter_count,
        "model_config": SPEC.model_config(),
        "training_inv_freq_sha256": inv_hash,
        "training_inv_freq_dtype": str(inv.numpy().dtype),
        "training_inv_freq_first": [
            float(value) for value in inv[:4]
        ],
        "training_inv_freq_last": [
            float(value) for value in inv[-4:]
        ],
        "training_inv_freq_min": float(inv.min()),
        "training_inv_freq_max": float(inv.max()),
        "initial_trainable_sha256": initial_hash,
        "row_order_sha256": order_hash,
        "train_rows": SPEC.train_rows,
        "train_tokens": SPEC.train_tokens,
        "prediction_tokens": SPEC.prediction_tokens,
        "optimizer_steps": SPEC.optimizer_steps,
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
        "compile": {
            "enabled": not args.no_compile,
            "mode": None if args.no_compile else args.compile_mode,
            "cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
        },
        "runtime": runtime,
        "python": platform.python_version(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(output / "train_meta.json", metadata)
    log_path = output / "train_log.jsonl"

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    previous = started
    tokens_since_log = 0
    accumulated_loss = 0.0
    optimizer_step = 0

    for micro_step, batch in enumerate(loader):
        if tuple(batch.shape) != (
            SPEC.micro_batch_size,
            SPEC.train_length,
        ):
            raise RuntimeError(f"unexpected training batch shape: {batch.shape}")
        if micro_step % SPEC.grad_accum_steps == 0:
            lr = learning_rate_for_step(optimizer_step)
            for group in optimizer.param_groups:
                group["lr"] = lr

        batch = batch.to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_module(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at micro-step {micro_step}: {loss.item()}"
            )
        (loss / SPEC.grad_accum_steps).backward()
        accumulated_loss += float(loss.detach().cpu())
        tokens_since_log += SPEC.micro_batch_size * SPEC.train_length

        if (micro_step + 1) % SPEC.grad_accum_steps:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"non-finite grad norm at step {optimizer_step}: "
                f"{grad_norm.item()}"
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed = optimizer_step + 1
        mean_loss = accumulated_loss / SPEC.grad_accum_steps
        accumulated_loss = 0.0

        if (
            optimizer_step == 0
            or completed % int(args.log_every) == 0
            or completed == SPEC.optimizer_steps
        ):
            torch.cuda.synchronize()
            now = time.time()
            interval = max(now - previous, 1e-6)
            elapsed = now - started
            eta = elapsed / completed * (SPEC.optimizer_steps - completed)
            record = {
                "step": optimizer_step,
                "completed_steps": completed,
                "loss": mean_loss,
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr": lr,
                "tokens_per_second": tokens_since_log / interval,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "memory_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 2**30
                ),
                "memory_reserved_gib": (
                    torch.cuda.max_memory_reserved() / 2**30
                ),
            }
            _append_jsonl(log_path, record)
            print(
                f"[{args.arm}] {completed}/{SPEC.optimizer_steps} "
                f"loss={mean_loss:.4f} lr={lr:.2e} "
                f"tok/s={record['tokens_per_second']:.0f} "
                f"mem={record['memory_allocated_gib']:.1f}GiB "
                f"ETA={eta / 60:.1f}m",
                flush=True,
            )
            previous = now
            tokens_since_log = 0
        optimizer_step += 1

    if optimizer_step != SPEC.optimizer_steps:
        raise RuntimeError(
            f"completed {optimizer_step} optimizer steps, expected "
            f"{SPEC.optimizer_steps}"
        )

    torch.cuda.synchronize()
    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metadata["train_seconds"] = time.time() - started
    checkpoint_path = output / "model.pt"
    checkpoint_sha = _save_checkpoint(checkpoint_path, model, metadata)
    metadata["checkpoint_sha256"] = checkpoint_sha
    _write_json(output / "train_meta.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def set_runtime_rope(
    model: GPT,
    inv_freq: torch.Tensor,
    *,
    length: int,
    mscale: float,
) -> None:
    rope = model.blocks[0].attn.rope
    if any(block.attn.rope is not rope for block in model.blocks):
        raise RuntimeError("model blocks do not share one rotary module")
    value = inv_freq.to(
        device=rope.inv_freq.device, dtype=rope.inv_freq.dtype
    )
    if value.shape != rope.inv_freq.shape:
        raise ValueError(
            f"runtime inv_freq shape {value.shape} != {rope.inv_freq.shape}"
        )
    rope.inv_freq.copy_(value)
    rope.attention_scaling = float(mscale)
    rope._build(int(length))


def _load_checkpoint(
    work_dir: Path,
    arm: str,
    *,
    spec: ExperimentSpec = SPEC,
) -> tuple[GPT, dict[str, Any]]:
    arm_dir = work_dir / "runs" / arm
    checkpoint_path = arm_dir / "model.pt"
    metadata_path = arm_dir / "train_meta.json"
    inv_path = arm_dir / "inv_freq.npy"
    for path in (checkpoint_path, metadata_path, inv_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("arm") != arm:
        raise ValueError(f"checkpoint arm mismatch in {metadata_path}")
    if metadata.get("protocol_sha256") != spec.fingerprint():
        raise ValueError(f"checkpoint protocol mismatch for {arm}")
    if metadata.get("code_sha256") != code_fingerprint():
        raise ValueError(f"code changed after checkpoint creation for {arm}")
    if sha256_file(checkpoint_path) != metadata.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint SHA-256 mismatch for {arm}")
    expected_inv = training_inv_freq(arm, spec=spec)
    saved_inv = torch.from_numpy(
        np.load(inv_path, allow_pickle=False)
    ).contiguous()
    if saved_inv.dtype != torch.float32:
        raise ValueError(f"saved training frequency dtype mismatch for {arm}")
    if not torch.equal(saved_inv, expected_inv):
        raise ValueError(f"saved training frequency mismatch for {arm}")
    if metadata.get("training_inv_freq_sha256") != tensor_sha256(
        expected_inv
    ):
        raise ValueError(f"frequency metadata mismatch for {arm}")

    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    payload_metadata = (
        payload.get("metadata") if isinstance(payload, dict) else None
    )
    if not isinstance(payload_metadata, dict):
        raise ValueError(
            f"checkpoint has no metadata: {checkpoint_path}"
        )
    for key, value in payload_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"checkpoint metadata differs from sidecar for {key}"
            )
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state: {checkpoint_path}")
    checkpoint_inv = _state_inv_freq(state, label=str(checkpoint_path))
    if tensor_sha256(checkpoint_inv) != metadata.get(
        "training_inv_freq_sha256"
    ):
        raise ValueError(f"checkpoint inv_freq hash mismatch for {arm}")
    if not torch.equal(checkpoint_inv, saved_inv):
        raise ValueError(f"checkpoint/sidecar inv_freq mismatch for {arm}")
    model = build_model(arm, spec=spec)
    model.load_state_dict(state, strict=True)
    if not torch.equal(_model_inv_freq(model), checkpoint_inv):
        raise ValueError(f"checkpoint inv_freq was overwritten for {arm}")
    return model, metadata


def _target_sha256(targets: np.ndarray) -> str:
    value = np.ascontiguousarray(targets, dtype=np.int64)
    return hashlib.sha256(value.tobytes()).hexdigest()


def build_eval_windows(
    validation: np.ndarray,
    anchors: np.ndarray,
    *,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build conventional L-token LM windows ending at exclusive anchors."""
    length_i = int(length)
    if length_i <= SPEC.eval_tail_tokens:
        raise ValueError(
            f"evaluation length {length_i} must exceed the "
            f"{SPEC.eval_tail_tokens}-token tail"
        )
    windows = []
    for end in anchors.tolist():
        end_i = int(end)
        start_i = end_i - length_i
        if start_i < 0 or end_i > len(validation):
            raise ValueError(
                f"invalid evaluation window [{start_i}, {end_i}) for "
                f"validation length {len(validation)}"
            )
        window = np.asarray(validation[start_i:end_i], dtype=np.int64)
        if tuple(window.shape) != (length_i,):
            raise RuntimeError(
                f"evaluation window has shape {window.shape}, expected "
                f"{(length_i,)}"
            )
        windows.append(window)
    stacked = np.stack(windows).astype(np.int64, copy=False)
    # Match the historical evaluator: an L-token window yields L-1 next-token
    # predictions. The FMRoPE target base remains the registered window L.
    return stacked[:, :-1], stacked[:, 1:]


@torch.no_grad()
def evaluate_condition(
    model: GPT,
    validation: np.ndarray,
    anchors: np.ndarray,
    *,
    arm: str,
    condition: str,
    length: int,
    batch_size: int,
    checkpoint_inv_freq: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inv, mscale, operator_meta = runtime_frequency(
        arm,
        condition,
        length,
        checkpoint_inv_freq=checkpoint_inv_freq,
    )
    set_runtime_rope(model, inv, length=length, mscale=mscale)
    model.eval()
    records: list[dict[str, Any]] = []
    for start in range(0, len(anchors), int(batch_size)):
        batch_anchors = anchors[start : start + int(batch_size)]
        inputs_np, targets_np = build_eval_windows(
            validation,
            batch_anchors,
            length=length,
        )
        inputs = torch.from_numpy(inputs_np).to("cuda", non_blocking=True)
        targets = torch.from_numpy(targets_np).to(
            "cuda", non_blocking=True
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs)
        token_nll = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        full = token_nll.mean(dim=1)
        tail = token_nll[:, -SPEC.eval_tail_tokens :].mean(dim=1)
        if not torch.isfinite(full).all() or not torch.isfinite(tail).all():
            raise RuntimeError(
                f"non-finite evaluation NLL for {arm}/{condition}/L={length}"
            )
        for index, end in enumerate(batch_anchors.tolist()):
            records.append(
                {
                    "arm": arm,
                    "condition": condition,
                    "length": int(length),
                    "anchor": int(end),
                    "full_nll": float(full[index].cpu()),
                    "tail_nll": float(tail[index].cpu()),
                    "tail_target_sha256": _target_sha256(
                        targets_np[index, -SPEC.eval_tail_tokens :]
                    ),
                }
            )
        del inputs, targets, logits, token_nll, full, tail
    return records, {
        **operator_meta,
        "inv_freq_sha256": tensor_sha256(inv),
        "mscale": float(mscale),
        "sequence_window_tokens": int(length),
        "model_input_positions": int(length) - 1,
        "prediction_positions": int(length) - 1,
    }


def _aggregate(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid aggregation values: {values}")
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean_nll": mean,
        "ppl": math.exp(mean) if mean < 700 else float("inf"),
        "stdev_nll": stdev,
        "se_nll": stdev / math.sqrt(len(values)),
    }


def summarize_records(
    records: list[dict[str, Any]],
    *,
    arms: tuple[str, ...] = ARMS,
) -> tuple[dict[str, Any], str]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in records:
        key = (row["arm"], row["condition"], int(row["length"]))
        grouped.setdefault(key, []).append(row)

    expected_groups = {
        (arm, condition, int(length))
        for arm in arms
        for condition in ARM_CONDITIONS[arm]
        for length in SPEC.eval_lengths
    }
    if set(grouped) != expected_groups:
        missing = sorted(expected_groups - set(grouped))
        extra = sorted(set(grouped) - expected_groups)
        raise RuntimeError(
            f"evaluation record matrix mismatch: missing={missing}, extra={extra}"
        )
    reference_anchors: set[int] | None = None
    target_hashes: dict[int, set[str]] = {}
    for key, rows in grouped.items():
        anchors = [int(row["anchor"]) for row in rows]
        if len(anchors) != len(set(anchors)):
            raise RuntimeError(f"duplicate evaluation anchor in group {key}")
        anchor_set = set(anchors)
        if reference_anchors is None:
            reference_anchors = anchor_set
        elif anchor_set != reference_anchors:
            raise RuntimeError(f"evaluation anchor mismatch in group {key}")
        for row in rows:
            anchor = int(row["anchor"])
            digest = str(row.get("tail_target_sha256", ""))
            if len(digest) != 64:
                raise RuntimeError(
                    f"invalid target hash at anchor {anchor} in group {key}"
                )
            target_hashes.setdefault(anchor, set()).add(digest)
    for anchor, digests in target_hashes.items():
        if len(digests) != 1:
            raise RuntimeError(
                f"paired tail targets differ at anchor {anchor}: "
                f"{sorted(digests)}"
            )

    identity_checks = (
        (
            ("paper_geo_base500k", "raw", SPEC.train_length),
            (
                "paper_geo_base500k",
                "yarn_derived_inference_only",
                SPEC.train_length,
            ),
        ),
        (
            ("fmrope_base256", "fixed_train_base", SPEC.train_length),
            ("fmrope_base256", "target_matched_base", SPEC.train_length),
        ),
        (
            (
                "anchored_cosh_tau4_fmrope_range",
                "fixed_train_range",
                SPEC.train_length,
            ),
            (
                "anchored_cosh_tau4_fmrope_range",
                "target_matched_range",
                SPEC.train_length,
            ),
        ),
    )
    for left_key, right_key in identity_checks:
        if left_key not in grouped and right_key not in grouped:
            continue
        left_rows = {
            int(row["anchor"]): row for row in grouped[left_key]
        }
        right_rows = {
            int(row["anchor"]): row for row in grouped[right_key]
        }
        if left_rows.keys() != right_rows.keys():
            raise RuntimeError(
                f"identity-control anchor mismatch: {left_key} vs {right_key}"
            )
        for anchor in left_rows:
            for metric in ("full_nll", "tail_nll"):
                difference = abs(
                    float(left_rows[anchor][metric])
                    - float(right_rows[anchor][metric])
                )
                if difference > 1e-6:
                    raise RuntimeError(
                        f"identity control failed for {metric} at anchor "
                        f"{anchor}: {left_key} vs {right_key}, "
                        f"|delta|={difference}"
                    )

    aggregate: dict[str, Any] = {}
    for (arm, condition, length), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda value: int(value["anchor"]))
        aggregate.setdefault(arm, {}).setdefault(condition, {})[
            str(length)
        ] = {
            "full": _aggregate([float(row["full_nll"]) for row in rows]),
            "tail": _aggregate([float(row["tail_nll"]) for row in rows]),
        }

    comparisons = (
        (
            "fmrope_target_minus_fmrope_fixed",
            ("fmrope_base256", "target_matched_base"),
            ("fmrope_base256", "fixed_train_base"),
        ),
        (
            "paper_geo_yarn_derived_minus_paper_geo_raw",
            ("paper_geo_base500k", "yarn_derived_inference_only"),
            ("paper_geo_base500k", "raw"),
        ),
        (
            "evq_raw_minus_fmrope_target",
            ("evq_cosh_tau4_paper_grid_base500k", "raw"),
            ("fmrope_base256", "target_matched_base"),
        ),
        (
            "fmrope_target_minus_paper_geo_raw",
            ("fmrope_base256", "target_matched_base"),
            ("paper_geo_base500k", "raw"),
        ),
        (
            "fmrope_target_minus_paper_geo_yarn_derived",
            ("fmrope_base256", "target_matched_base"),
            ("paper_geo_base500k", "yarn_derived_inference_only"),
        ),
        (
            "evq_raw_minus_paper_geo_raw",
            ("evq_cosh_tau4_paper_grid_base500k", "raw"),
            ("paper_geo_base500k", "raw"),
        ),
        (
            "anchored_cosh_target_minus_fmrope_target",
            (
                "anchored_cosh_tau4_fmrope_range",
                "target_matched_range",
            ),
            ("fmrope_base256", "target_matched_base"),
        ),
        (
            "anchored_cosh_fixed_minus_fmrope_fixed",
            (
                "anchored_cosh_tau4_fmrope_range",
                "fixed_train_range",
            ),
            ("fmrope_base256", "fixed_train_base"),
        ),
        (
            "anchored_cosh_target_minus_evq_raw",
            (
                "anchored_cosh_tau4_fmrope_range",
                "target_matched_range",
            ),
            ("evq_cosh_tau4_paper_grid_base500k", "raw"),
        ),
    )
    paired: dict[str, Any] = {}
    for name, left, right in comparisons:
        if left[0] not in arms or right[0] not in arms:
            continue
        paired[name] = {}
        for length in SPEC.eval_lengths:
            left_rows = {
                int(row["anchor"]): row
                for row in grouped[(left[0], left[1], length)]
            }
            right_rows = {
                int(row["anchor"]): row
                for row in grouped[(right[0], right[1], length)]
            }
            if left_rows.keys() != right_rows.keys():
                raise RuntimeError(
                    f"paired anchor mismatch for {name} at L={length}"
                )
            diffs = [
                float(left_rows[anchor]["tail_nll"])
                - float(right_rows[anchor]["tail_nll"])
                for anchor in sorted(left_rows)
            ]
            stats = _aggregate(diffs)
            paired[name][str(length)] = {
                "left": {"arm": left[0], "condition": left[1]},
                "right": {"arm": right[0], "condition": right[1]},
                "mean_left_minus_right_tail_nll": stats["mean_nll"],
                "stdev_paired_difference": stats["stdev_nll"],
                "se_paired_difference": stats["se_nll"],
                "left_win_rate": sum(value < 0 for value in diffs)
                / len(diffs),
                "n": len(diffs),
                "interpretation": "negative means left has lower tail NLL",
            }

    summary = {
        "metric_priority": "paired final-128-token NLL",
        "in_domain_identity_controls": "PASS",
        "aggregate": aggregate,
        "paired": paired,
    }

    display_rows = (
        ("Paper-Geo raw", "paper_geo_base500k", "raw"),
        (
            "Paper-Geo + YaRN-derived virtual ramp (inference-only)",
            "paper_geo_base500k",
            "yarn_derived_inference_only",
        ),
        (
            "FMRoPE fixed base=256",
            "fmrope_base256",
            "fixed_train_base",
        ),
        (
            "FMRoPE target base=L",
            "fmrope_base256",
            "target_matched_base",
        ),
        (
            "EVQ tau=4 raw",
            "evq_cosh_tau4_paper_grid_base500k",
            "raw",
        ),
        (
            "Range-anchored Cosh tau=4, fixed train range",
            "anchored_cosh_tau4_fmrope_range",
            "fixed_train_range",
        ),
        (
            "Range-anchored Cosh tau=4, target-matched range",
            "anchored_cosh_tau4_fmrope_range",
            "target_matched_range",
        ),
    )
    lines = [
        "# FMRoPE vs EVQ L=256 result",
        "",
        "Primary metric: paired final-128-token NLL; table displays "
        "`exp(mean NLL)` for readability.",
        "",
        "Each `L` is an L-token window evaluated with L-1 model inputs and "
        "next-token predictions, matching the historical evaluator.",
        "",
        "In-domain identity controls: PASS (Paper-Geo raw = YaRN-derived "
        "operator at scale 1; "
        "FMRoPE fixed = target-matched at L=256).",
        "",
        "| condition | "
        + " | ".join(f"L={length}" for length in SPEC.eval_lengths)
        + " |",
        "| --- | " + " | ".join("---:" for _ in SPEC.eval_lengths) + " |",
    ]
    for label, arm, condition in display_rows:
        if arm not in arms:
            continue
        values = [
            aggregate[arm][condition][str(length)]["tail"]["ppl"]
            for length in SPEC.eval_lengths
        ]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{float(value):.3f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired tail-NLL differences",
            "",
            "Negative means the left-hand method is better.",
            "",
            "| comparison | "
            + " | ".join(f"L={length}" for length in SPEC.eval_lengths)
            + " |",
            "| --- | " + " | ".join("---:" for _ in SPEC.eval_lengths) + " |",
        ]
    )
    for name in paired:
        values = [
            paired[name][str(length)][
                "mean_left_minus_right_tail_nll"
            ]
            for length in SPEC.eval_lengths
        ]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{float(value):+.5f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            f"Single seed ({SPEC.seed}). This is a matched rebuttal "
            "diagnostic, not a general performance claim.",
            "",
        ]
    )
    return summary, "\n".join(lines)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    runtime = validate_cuda_runtime()
    work_dir = Path(args.work_dir).resolve()
    manifest_path = Path(args.data_manifest).resolve()
    manifest = _load_manifest(
        manifest_path, full_hash_check=bool(args.full_hash_check)
    )
    output_dir = work_dir / "evaluation"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty evaluation: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_arms = tuple(args.arms or ARMS)
    if len(selected_arms) != len(set(selected_arms)):
        raise ValueError(f"duplicate evaluation arm: {selected_arms}")
    metadata_by_arm = {
        arm: json.loads(
            (work_dir / "runs" / arm / "train_meta.json").read_text()
        )
        for arm in selected_arms
    }
    current_code_sha = code_fingerprint()
    current_manifest_sha = sha256_file(manifest_path)
    shared_keys = (
        "protocol_sha256",
        "code_sha256",
        "data_manifest_sha256",
        "train_used_prefix_sha256",
        "validation_sha256",
        "anchors_sha256",
        "seed",
        "parameter_count",
        "initial_trainable_sha256",
        "row_order_sha256",
        "train_rows",
        "train_tokens",
        "prediction_tokens",
        "optimizer_steps",
    )
    reference = metadata_by_arm[selected_arms[0]]
    for arm, metadata in metadata_by_arm.items():
        if metadata.get("code_sha256") != current_code_sha:
            raise ValueError(
                f"code changed after training checkpoint {arm} was produced"
            )
        if metadata.get("data_manifest_sha256") != current_manifest_sha:
            raise ValueError(
                f"data manifest changed after training checkpoint {arm} was produced"
            )
        for key in shared_keys:
            if metadata.get(key) != reference.get(key):
                raise ValueError(
                    f"paired checkpoint mismatch: {arm} differs on {key}"
                )

    validation = np.load(
        manifest["validation"]["path"], mmap_mode="r", allow_pickle=False
    )
    anchors = np.load(
        manifest["anchors"]["path"], allow_pickle=False
    )
    if int(validation.max()) >= SPEC.vocab_size or int(validation.min()) < 0:
        raise ValueError("validation token id is outside model vocabulary")

    records: list[dict[str, Any]] = []
    condition_metadata: dict[str, Any] = {}
    for arm in selected_arms:
        model, checkpoint_metadata = _load_checkpoint(work_dir, arm)
        model = model.to("cuda")
        checkpoint_inv_freq = _model_inv_freq(model)
        if tensor_sha256(checkpoint_inv_freq) != checkpoint_metadata.get(
            "training_inv_freq_sha256"
        ):
            raise ValueError(
                f"loaded checkpoint frequency receipt mismatch for {arm}"
            )
        condition_metadata[arm] = {}
        for condition in ARM_CONDITIONS[arm]:
            condition_metadata[arm][condition] = {}
            for length in SPEC.eval_lengths:
                print(
                    f"[eval] {arm}/{condition}/L={length}",
                    flush=True,
                )
                rows, operator_meta = evaluate_condition(
                    model,
                    validation,
                    anchors,
                    arm=arm,
                    condition=condition,
                    length=length,
                    batch_size=int(args.eval_batch_size),
                    checkpoint_inv_freq=checkpoint_inv_freq,
                )
                records.extend(rows)
                condition_metadata[arm][condition][str(length)] = (
                    operator_meta
                )
        del model, checkpoint_metadata
        gc.collect()
        torch.cuda.empty_cache()

    summary, markdown = summarize_records(records, arms=selected_arms)
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": current_code_sha,
        "data_manifest_sha256": current_manifest_sha,
        "runtime": runtime,
        "seed": SPEC.seed,
        "model_tier": SPEC.model_tier,
        "arms": list(selected_arms),
        "eval_lengths": list(SPEC.eval_lengths),
        "eval_length_definition": (
            "L-token window; L-1 model inputs and next-token predictions"
        ),
        "eval_tail_tokens": SPEC.eval_tail_tokens,
        "eval_anchor_count": SPEC.eval_anchor_count,
        "condition_metadata": condition_metadata,
        "records": records,
        "summary": summary,
        "claim_boundary": (
            f"single-seed ({SPEC.seed}) matched diagnostic; FMRoPE is "
            "paper-faithful local implementation; YaRN cell is inference-only"
        ),
    }
    _write_json(output_dir / "results.json", payload)
    (output_dir / "summary.md").write_text(markdown)
    print(markdown)
    return payload


def compare_results(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the new arm with immutable prior results, never checkpoints."""
    baseline_path = Path(args.baseline_results).resolve()
    new_path = Path(args.new_results).resolve()
    baseline = json.loads(baseline_path.read_text())
    new = json.loads(new_path.read_text())
    for key in (
        "protocol_sha256",
        "data_manifest_sha256",
        "seed",
        "eval_lengths",
        "eval_tail_tokens",
        "eval_anchor_count",
    ):
        if baseline.get(key) != new.get(key):
            raise ValueError(f"baseline/new result mismatch on {key}")

    baseline_records = baseline.get("records")
    new_records = new.get("records")
    if not isinstance(baseline_records, list) or not isinstance(new_records, list):
        raise ValueError("both result files must contain raw evaluation records")
    baseline_arms = {str(row.get("arm")) for row in baseline_records}
    new_arms = {str(row.get("arm")) for row in new_records}
    expected_baseline = set(ARMS) - {"anchored_cosh_tau4_fmrope_range"}
    if baseline_arms != expected_baseline:
        raise ValueError(
            f"baseline arms are {sorted(baseline_arms)}, expected "
            f"{sorted(expected_baseline)}"
        )
    if new_arms != {"anchored_cosh_tau4_fmrope_range"}:
        raise ValueError(f"new result arms are invalid: {sorted(new_arms)}")

    records = [*baseline_records, *new_records]
    summary, markdown = summarize_records(records, arms=ARMS)
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "PASS",
        "single_seed_supporting": True,
        "paper_claim": False,
        "protocol_sha256": baseline["protocol_sha256"],
        "data_manifest_sha256": baseline["data_manifest_sha256"],
        "seed": baseline["seed"],
        "eval_lengths": baseline["eval_lengths"],
        "sources": {
            "baseline_results": str(baseline_path),
            "baseline_results_sha256": sha256_file(baseline_path),
            "new_results": str(new_path),
            "new_results_sha256": sha256_file(new_path),
        },
        "summary": summary,
        "claim_boundary": (
            "single-seed post-submission matched-range diagnostic; existing "
            "500M arms were reused without retraining"
        ),
    }
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", payload)
    (output_dir / "REPORT.md").write_text(markdown)
    print(markdown)
    return payload


def aggregate_exact_range(args: argparse.Namespace) -> dict[str, Any]:
    """Aggregate the registered exact-range contrast across training seeds."""
    payloads = []
    for path_arg in args.inputs:
        path = Path(path_arg).resolve()
        payload = json.loads(path.read_text())
        payloads.append((path, payload))

    expected_seeds = (42, 137, 256)
    seeds = tuple(sorted(int(payload.get("seed", -1)) for _, payload in payloads))
    if seeds != expected_seeds:
        raise ValueError(
            f"exact-range aggregate requires seeds {expected_seeds}, got {seeds}"
        )
    if len({payload["data_manifest_sha256"] for _, payload in payloads}) != 1:
        raise ValueError("multi-seed inputs use different data manifests")
    if len({tuple(payload["eval_lengths"]) for _, payload in payloads}) != 1:
        raise ValueError("multi-seed inputs use different evaluation lengths")

    comparison_names = (
        "anchored_cosh_fixed_minus_fmrope_fixed",
        "anchored_cosh_target_minus_fmrope_target",
    )
    by_seed: dict[int, dict[str, Any]] = {}
    for _, payload in payloads:
        seed = int(payload["seed"])
        paired = payload.get("summary", {}).get("paired", {})
        missing = [name for name in comparison_names if name not in paired]
        if missing:
            raise ValueError(
                f"seed {seed} is missing exact-range comparisons: {missing}"
            )
        by_seed[seed] = {name: paired[name] for name in comparison_names}

    # 95% two-sided Student-t critical value for three independent seeds.
    t95_df2 = 4.302652729911275
    aggregate: dict[str, Any] = {}
    lengths = tuple(int(value) for value in payloads[0][1]["eval_lengths"])
    for name in comparison_names:
        aggregate[name] = {}
        for length in lengths:
            values = [
                float(
                    by_seed[seed][name][str(length)][
                        "mean_left_minus_right_tail_nll"
                    ]
                )
                for seed in expected_seeds
            ]
            mean = statistics.fmean(values)
            stdev = statistics.stdev(values)
            half_width = t95_df2 * stdev / math.sqrt(len(values))
            aggregate[name][str(length)] = {
                "seed_values": {
                    str(seed): value
                    for seed, value in zip(expected_seeds, values)
                },
                "mean_nll_difference": mean,
                "stdev_across_seeds": stdev,
                "paired_t_95_ci": [mean - half_width, mean + half_width],
                "cosh_wins": sum(value < 0 for value in values),
                "n_training_seeds": len(values),
            }

    fixed = aggregate["anchored_cosh_fixed_minus_fmrope_fixed"]
    target = aggregate["anchored_cosh_target_minus_fmrope_target"]
    ood_lengths = tuple(length for length in lengths if length > SPEC.train_length)
    decision = (
        "THREE_SEED_SHAPE_EFFECT_WITHOUT_TARGET_SYNERGY"
        if all(fixed[str(length)]["cosh_wins"] == 3 for length in ood_lengths)
        and all(target[str(length)]["cosh_wins"] == 0 for length in ood_lengths)
        else "THREE_SEED_RESULT_MIXED"
    )
    output = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "PASS",
        "decision": decision,
        "three_seed_supporting": True,
        "paper_claim": False,
        "seeds": list(expected_seeds),
        "eval_lengths": list(lengths),
        "data_manifest_sha256": payloads[0][1]["data_manifest_sha256"],
        "sources": [
            {
                "seed": int(payload["seed"]),
                "sha256": sha256_file(path),
            }
            for path, payload in payloads
        ],
        "aggregate": aggregate,
        "claim_boundary": (
            "post-submission three-seed matched-range diagnostic; training "
            "seeds are the independent units"
        ),
    }

    labels = (
        ("Fixed training range", comparison_names[0]),
        ("Target-matched range", comparison_names[1]),
    )
    lines = [
        "# Three-seed exact-range allocation result",
        "",
        "Cosh minus uniform FMRoPE tail NLL; negative favors Cosh.",
        "",
        "| condition | "
        + " | ".join(f"L={length}" for length in lengths)
        + " |",
        "| --- | " + " | ".join("---:" for _ in lengths) + " |",
    ]
    for label, name in labels:
        values = [
            aggregate[name][str(length)]["mean_nll_difference"]
            for length in lengths
        ]
        lines.append(
            f"| {label}, 3-seed mean | "
            + " | ".join(f"{value:+.4f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            f"Decision: `{decision}`.",
            "",
            "Intervals in the JSON are paired t intervals across the three "
            "independent training seeds; evaluation anchors are not treated "
            "as independent seeds.",
            "",
        ]
    )

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", output)
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--data_manifest", type=Path, required=True)
    preflight_parser.add_argument("--full_hash_check", action="store_true")
    preflight_parser.add_argument(
        "--verify_full_initialization", action="store_true"
    )
    preflight_parser.add_argument("--arms", nargs="+", choices=ARMS)

    probe_parser = subparsers.add_parser("probe-gpu")
    probe_parser.add_argument("--arm", choices=ARMS, required=True)
    probe_parser.add_argument("--data_manifest", type=Path, required=True)
    probe_parser.add_argument("--work_dir", type=Path, required=True)
    probe_parser.add_argument("--timed_steps", type=int, default=5)
    probe_parser.add_argument(
        "--compile_mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--arm", choices=ARMS, required=True)
    train_parser.add_argument("--data_manifest", type=Path, required=True)
    train_parser.add_argument("--work_dir", type=Path, required=True)
    train_parser.add_argument("--num_workers", type=int, default=8)
    train_parser.add_argument("--log_every", type=int, default=25)
    train_parser.add_argument(
        "--compile_mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
    )
    train_parser.add_argument("--no_compile", action="store_true")
    train_parser.add_argument("--full_hash_check", action="store_true")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--data_manifest", type=Path, required=True)
    eval_parser.add_argument("--work_dir", type=Path, required=True)
    eval_parser.add_argument("--eval_batch_size", type=int, default=2)
    eval_parser.add_argument("--full_hash_check", action="store_true")
    eval_parser.add_argument("--arms", nargs="+", choices=ARMS)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline_results", type=Path, required=True)
    compare_parser.add_argument("--new_results", type=Path, required=True)
    compare_parser.add_argument("--output_dir", type=Path, required=True)

    aggregate_parser = subparsers.add_parser("aggregate-exact-range")
    aggregate_parser.add_argument(
        "--inputs", type=Path, nargs=3, required=True
    )
    aggregate_parser.add_argument("--output_dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "preflight":
        run_preflight(
            args.data_manifest,
            full_hash_check=bool(args.full_hash_check),
            verify_full_initialization=bool(
                args.verify_full_initialization
            ),
            arms=tuple(args.arms or ARMS),
        )
    elif args.command == "probe-gpu":
        probe_gpu(args)
    elif args.command == "train":
        train_arm(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "compare":
        compare_results(args)
    elif args.command == "aggregate-exact-range":
        aggregate_exact_range(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
