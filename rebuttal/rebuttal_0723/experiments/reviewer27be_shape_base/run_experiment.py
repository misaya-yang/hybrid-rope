#!/usr/bin/env python3
"""Train, evaluate, select tau, and summarize Reviewer 27bE experiments."""

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
from rebuttal.rebuttal_0723.experiments.reviewer27be_shape_base.prepare import (
    sha256_file,
    validate_manifest,
)
from rebuttal.rebuttal_0723.experiments.reviewer27be_shape_base.protocol import (
    SHAPE_CORE_ARMS,
    SHAPE_TAU_ARMS,
    SPECS,
    arms_for_suite,
    estimate_parameter_count,
    frequency_contract,
    learning_rate_for_step,
    seeds_for_arm,
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
        digest.update(name.encode())
        digest.update(
            parameter.detach().cpu().float().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def code_fingerprint() -> str:
    paths = (
        PACKAGE_DIR / "protocol.py",
        PACKAGE_DIR / "derive_real_rope_shapes.py",
        PACKAGE_DIR / "real_rope_schedules.py",
        PACKAGE_DIR / "prepare.py",
        PACKAGE_DIR / "run_experiment.py",
        PACKAGE_DIR / "run_5090.sh",
        REPO_ROOT / "experiments/native_rope_evq_150m/model.py",
        REPO_ROOT / "rebuttal/rebuttal_0723/experiments/geo_rope_contract.py",
        REPO_ROOT / "rebuttal/rebuttal_0723/theory_results/FREQUENCY_DEFINITION_MANIFEST.json",
        REPO_ROOT / "scripts/lib/rope/schedules.py",
        REPO_ROOT / "tests/test_reviewer27be_shape_base.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"code fingerprint input missing: {path}")
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def _load_manifest(path: Path, *, full_hash_check: bool) -> dict[str, Any]:
    manifest = json.loads(path.resolve().read_text())
    validate_manifest(manifest, check_hashes=bool(full_hash_check))
    return manifest


def deterministic_row_order(n_rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(int(n_rows), generator=generator, dtype=torch.int64)


class TensorOrderSampler(Sampler[int]):
    def __init__(self, order: torch.Tensor) -> None:
        if order.ndim != 1 or order.dtype != torch.int64:
            raise ValueError("order must be a one-dimensional int64 tensor")
        self.order = order

    def __iter__(self) -> Iterator[int]:
        return (int(value) for value in self.order.tolist())

    def __len__(self) -> int:
        return int(self.order.numel())


class FlatPrefixDataset(Dataset[torch.Tensor]):
    """View one registered flat prefix as fixed-length training rows."""

    def __init__(self, path: str | Path, *, rows: int, seq_len: int) -> None:
        self.path = str(Path(path).resolve())
        self.rows = int(rows)
        self.seq_len = int(seq_len)
        self._array: np.ndarray | None = None
        self._flat: np.ndarray | None = None
        self._open()
        assert self._flat is not None
        if len(self._flat) < self.rows * self.seq_len:
            raise ValueError("training tensor is shorter than registered prefix")
        self._array = None
        self._flat = None

    def _open(self) -> None:
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r", allow_pickle=False)
            if self._array.dtype != np.int64:
                raise ValueError("training tensor must contain int64 token ids")
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
        return torch.from_numpy(
            np.array(
                self._flat[start : start + self.seq_len],
                dtype=np.int64,
                copy=True,
            )
        )


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


def build_model(suite: str, arm: str, seed: int) -> GPT:
    spec = SPECS[suite]
    seed_everything(seed)
    inv, _ = training_inv_freq(suite, arm)
    return GPT(spec.model_config(), inv.float())


def meta_parameter_count(suite: str, arm: str) -> int:
    spec = SPECS[suite]
    inv, _ = training_inv_freq(suite, arm)
    with torch.device("meta"):
        model = GPT(spec.model_config(), inv.float().to("meta"))
    return sum(parameter.numel() for parameter in model.parameters())


def validate_cuda_runtime() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for train/evaluate")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA is required")
    props = torch.cuda.get_device_properties(0)
    total_memory = getattr(
        props, "total_memory", getattr(props, "total_mem", 0)
    )
    if int(total_memory) < 30 * 2**30:
        raise RuntimeError(
            f"registered 5090 run requires >=30 GiB, found "
            f"{int(total_memory) / 2**30:.1f} GiB"
        )
    return {
        "name": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": int(total_memory),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }


def run_preflight(
    manifest_path: Path,
    *,
    full_hash_check: bool,
    verify_full_initialization: bool,
) -> dict[str, Any]:
    manifest = _load_manifest(
        manifest_path, full_hash_check=bool(full_hash_check)
    )
    suites: dict[str, Any] = {}
    for suite, spec in SPECS.items():
        parameter_counts = {
            arm: meta_parameter_count(suite, arm)
            for arm in arms_for_suite(suite)
        }
        expected = estimate_parameter_count(spec)
        if set(parameter_counts.values()) != {expected}:
            raise RuntimeError(f"{suite} parameter count mismatch")
        initial_hashes: dict[str, str] = {}
        if verify_full_initialization:
            for arm in arms_for_suite(suite):
                model = build_model(suite, arm, 42)
                initial_hashes[arm] = trainable_state_sha256(model)
                del model
                gc.collect()
            if len(set(initial_hashes.values())) != 1:
                raise RuntimeError(
                    f"{suite} trainable initialization differs across arms"
                )
        schedules: dict[str, Any] = {}
        for arm in arms_for_suite(suite):
            inv, metadata = training_inv_freq(suite, arm)
            if not torch.all(torch.diff(inv) < 0):
                raise RuntimeError(f"{suite}/{arm} is not strictly decreasing")
            schedules[arm] = {
                "sha256": tensor_sha256(inv),
                "dtype": str(inv.numpy().dtype),
                "first_channels": [float(value) for value in inv[:4]],
                "last_channels": [float(value) for value in inv[-4:]],
                "minimum": float(inv.min()),
                "maximum": float(inv.max()),
                "metadata": metadata,
                "seeds": list(seeds_for_arm(suite, arm)),
            }
        suites[suite] = {
            "protocol_sha256": spec.fingerprint(),
            "parameter_count": expected,
            "parameter_counts": parameter_counts,
            "initial_trainable_sha256": initial_hashes,
            "initialization_check": (
                "full CPU check completed"
                if verify_full_initialization
                else "deferred to actual per-seed run receipts"
            ),
            "train_tokens": spec.train_tokens,
            "optimizer_steps": spec.optimizer_steps,
            "micro_steps": spec.micro_steps,
            "global_batch_size": spec.global_batch_size,
            "micro_batch_size": spec.micro_batch_size,
            "grad_accum_steps": spec.grad_accum_steps,
            "schedules": schedules,
            "frequency_contract": frequency_contract(suite),
        }
    report = {
        "status": "PASS",
        "cpu_only_preflight": not torch.cuda.is_available(),
        "code_sha256": code_fingerprint(),
        "data_manifest_sha256": sha256_file(manifest_path),
        "source_manifest": manifest["source_manifest"],
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "suites": suites,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _run_dir(work_dir: Path, suite: str, arm: str, seed: int) -> Path:
    return work_dir.resolve() / "runs" / suite / arm / f"seed{seed}"


def _save_checkpoint(
    path: Path, model: GPT, metadata: dict[str, Any]
) -> str:
    temporary = path.with_name(path.name + ".incomplete")
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    checkpoint_inv = _state_inv_freq(state, label="checkpoint save")
    if tensor_sha256(checkpoint_inv) != metadata.get("inv_freq_sha256"):
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


def probe_gpu(args: argparse.Namespace) -> dict[str, Any]:
    """Compile and execute one discarded optimizer step before paid runs."""
    runtime = validate_cuda_runtime()
    spec = SPECS[args.suite]
    manifest = _load_manifest(
        args.data_manifest.resolve(), full_hash_check=False
    )
    path = args.work_dir.resolve() / f"gpu_probe_{args.suite}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite GPU probe: {path}")
    source = np.load(
        manifest["train"]["path"], mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    token_count = spec.micro_batch_size * spec.train_length
    batch = torch.from_numpy(
        np.array(source[:token_count], dtype=np.int64, copy=True).reshape(
            spec.micro_batch_size, spec.train_length
        )
    ).to("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(args.suite, arms_for_suite(args.suite)[0], 42).to("cuda")
    loss_module: nn.Module = CausalLanguageModelLoss(model)
    loss_module = torch.compile(
        loss_module,
        mode="reduce-overhead",
        dynamic=False,
        fullgraph=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=spec.weight_decay,
        fused=True,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = loss_module(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - started
    report = {
        "status": "PASS",
        "discarded_step": True,
        "suite": args.suite,
        "protocol_sha256": spec.fingerprint(),
        "code_sha256": code_fingerprint(),
        "runtime": runtime,
        "micro_batch_size": spec.micro_batch_size,
        "sequence_length": spec.train_length,
        "elapsed_seconds_including_compile": elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "loss": float(loss.detach()),
    }
    _atomic_json(path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    runtime = validate_cuda_runtime()
    spec = SPECS[args.suite]
    if args.arm not in arms_for_suite(args.suite):
        raise ValueError("arm is not registered for suite")
    if args.seed not in seeds_for_arm(args.suite, args.arm):
        raise ValueError("seed is not registered for this suite/arm")
    manifest_path = args.data_manifest.resolve()
    manifest = _load_manifest(
        manifest_path, full_hash_check=bool(args.full_hash_check)
    )
    output = _run_dir(args.work_dir, args.suite, args.arm, args.seed)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(args.suite, args.arm, args.seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != estimate_parameter_count(spec):
        raise RuntimeError("model parameter count mismatch")
    initial_hash = trainable_state_sha256(model)
    inv, schedule_metadata = training_inv_freq(args.suite, args.arm)
    inv_hash = tensor_sha256(inv)
    np.save(output / "inv_freq.npy", inv.numpy())

    order = deterministic_row_order(spec.train_rows, args.seed)
    order_hash = tensor_sha256(order)
    dataset = FlatPrefixDataset(
        manifest["train"]["path"],
        rows=spec.train_rows,
        seq_len=spec.train_length,
    )
    workers = max(0, int(args.num_workers))
    loader = DataLoader(
        dataset,
        batch_size=spec.micro_batch_size,
        sampler=TensorOrderSampler(order),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=False,
    )
    if len(loader) != spec.micro_steps:
        raise RuntimeError(
            f"loader has {len(loader)} micro steps, expected {spec.micro_steps}"
        )

    model = model.to("cuda")
    loss_module: nn.Module = CausalLanguageModelLoss(model)
    if not args.no_compile:
        loss_module = torch.compile(
            loss_module,
            mode=args.compile_mode,
            dynamic=False,
            fullgraph=False,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=spec.weight_decay,
        fused=True,
    )
    metadata = {
        "schema_version": 1,
        "suite": args.suite,
        "arm": args.arm,
        "seed": args.seed,
        "protocol_sha256": spec.fingerprint(),
        "code_sha256": code_fingerprint(),
        "data_manifest_sha256": sha256_file(manifest_path),
        "train_prefix_sha256": manifest["suites"][args.suite][
            "train_prefix_sha256"
        ],
        "validation_sha256": manifest["validation"]["sha256"],
        "selection_anchor_sha256": manifest["selection_anchors"]["sha256"],
        "test_anchor_sha256": manifest["test_anchors"]["sha256"],
        "initial_trainable_sha256": initial_hash,
        "row_order_sha256": order_hash,
        "inv_freq_sha256": inv_hash,
        "inv_freq_dtype": str(inv.numpy().dtype),
        "inv_freq_first": [float(value) for value in inv[:4]],
        "inv_freq_last": [float(value) for value in inv[-4:]],
        "schedule": schedule_metadata,
        "parameter_count": parameter_count,
        "runtime": runtime,
        "compile": {
            "enabled": not args.no_compile,
            "mode": None if args.no_compile else args.compile_mode,
            "torchinductor_cache_dir": os.environ.get(
                "TORCHINDUCTOR_CACHE_DIR"
            ),
        },
        "train": {
            "length": spec.train_length,
            "tokens": spec.train_tokens,
            "optimizer_steps": spec.optimizer_steps,
            "global_batch_size": spec.global_batch_size,
            "micro_batch_size": spec.micro_batch_size,
            "grad_accum_steps": spec.grad_accum_steps,
        },
    }
    _atomic_json(output / "metadata.json", metadata)

    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    peak_memory = 0
    for micro_step, batch in enumerate(loader):
        step = micro_step // spec.grad_accum_steps
        lr = learning_rate_for_step(step, spec)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = batch.to("cuda", non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_module(batch)
            scaled = loss / spec.grad_accum_steps
        scaled.backward()
        losses.append(float(loss.detach()))
        if (micro_step + 1) % spec.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed = step + 1
            if completed == 1 or completed % args.log_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                tokens_done = completed * spec.global_batch_size * spec.train_length
                row = {
                    "step": completed,
                    "optimizer_steps": spec.optimizer_steps,
                    "loss": statistics.fmean(
                        losses[-spec.grad_accum_steps :]
                    ),
                    "lr": lr,
                    "tokens_per_second": tokens_done / elapsed,
                    "elapsed_seconds": elapsed,
                }
                _append_jsonl(output / "train_log.jsonl", row)
                print(json.dumps(row, sort_keys=True), flush=True)
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

    elapsed = time.time() - started
    final_metadata = {
        **metadata,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": int(peak_memory),
        "final_loss": statistics.fmean(losses[-spec.grad_accum_steps :]),
    }
    checkpoint = output / "checkpoint.pt"
    checkpoint_sha = _save_checkpoint(checkpoint, model, final_metadata)
    result = {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        **final_metadata,
    }
    _atomic_json(output / "train_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_checkpoint(
    checkpoint: Path, suite: str, arm: str, seed: int
) -> tuple[GPT, dict[str, Any]]:
    receipt_path = checkpoint.resolve().parent / "train_result.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("checkpoint SHA-256 differs from train receipt")
    payload = torch.load(
        checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    metadata = payload["metadata"]
    expected = {
        "suite": suite,
        "arm": arm,
        "seed": seed,
        "protocol_sha256": SPECS[suite].fingerprint(),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"checkpoint {key} mismatch: {metadata.get(key)!r} != {value!r}"
            )
        if receipt.get(key) != value:
            raise ValueError(
                f"train receipt {key} mismatch: "
                f"{receipt.get(key)!r} != {value!r}"
            )
    if metadata.get("code_sha256") != code_fingerprint():
        raise ValueError("code changed after checkpoint creation")
    if receipt.get("code_sha256") != metadata.get("code_sha256"):
        raise ValueError("checkpoint code hash differs from train receipt")
    if receipt.get("inv_freq_sha256") != metadata.get(
        "inv_freq_sha256"
    ):
        raise ValueError("checkpoint inv_freq hash differs from train receipt")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state: {checkpoint}")
    checkpoint_inv = _state_inv_freq(state, label=str(checkpoint))
    if tensor_sha256(checkpoint_inv) != metadata.get("inv_freq_sha256"):
        raise ValueError("checkpoint inv_freq hash differs from metadata")
    sidecar_path = checkpoint.resolve().parent / "inv_freq.npy"
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = torch.from_numpy(
        np.load(sidecar_path, allow_pickle=False)
    ).contiguous()
    if sidecar.dtype != torch.float32:
        raise ValueError("checkpoint inv_freq sidecar must be float32")
    if not torch.equal(sidecar, checkpoint_inv):
        raise ValueError("checkpoint inv_freq differs from sidecar")
    model = build_model(suite, arm, seed)
    model.load_state_dict(state, strict=True)
    if not torch.equal(_model_inv_freq(model), checkpoint_inv):
        raise ValueError("checkpoint inv_freq was overwritten after load")
    return model, metadata


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    runtime = validate_cuda_runtime()
    spec = SPECS[args.suite]
    manifest_path = args.data_manifest.resolve()
    manifest = _load_manifest(
        manifest_path, full_hash_check=bool(args.full_hash_check)
    )
    output = _run_dir(args.work_dir, args.suite, args.arm, args.seed)
    checkpoint = output / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result_path = output / f"eval_{args.split}.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {result_path}")
    model, train_metadata = _load_checkpoint(
        checkpoint, args.suite, args.arm, args.seed
    )
    loaded_inv_freq_sha256 = tensor_sha256(_model_inv_freq(model))
    current_suite = manifest["suites"][args.suite]
    if (
        train_metadata["train_prefix_sha256"]
        != current_suite["train_prefix_sha256"]
    ):
        raise ValueError("training token prefix differs from checkpoint")
    if (
        train_metadata["validation_sha256"]
        != manifest["validation"]["sha256"]
    ):
        raise ValueError("validation tensor differs from checkpoint")
    if sha256_file(manifest["validation"]["path"]) != train_metadata[
        "validation_sha256"
    ]:
        raise ValueError("validation tensor content hash mismatch")

    validation = np.load(
        manifest["validation"]["path"], mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    anchor_record = manifest[f"{args.split}_anchors"]
    expected_anchor = train_metadata[f"{args.split}_anchor_sha256"]
    if anchor_record["sha256"] != expected_anchor:
        raise ValueError("evaluation anchors differ from checkpoint")
    if sha256_file(anchor_record["path"]) != expected_anchor:
        raise ValueError("evaluation anchor content hash mismatch")
    anchors = np.load(anchor_record["path"], allow_pickle=False)
    model.extend_rope(max(spec.eval_lengths))
    model = model.to("cuda").eval()
    rows: list[dict[str, Any]] = []
    started = time.time()
    with torch.inference_mode():
        for length in spec.eval_lengths:
            for anchor_index, endpoint in enumerate(anchors.tolist()):
                start = int(endpoint) - int(length)
                tokens = np.array(
                    validation[start : int(endpoint)],
                    dtype=np.int64,
                    copy=True,
                )
                batch = torch.from_numpy(tokens).unsqueeze(0).to("cuda")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch[:, :-1])
                    targets = batch[:, 1:]
                    full_nll = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                    )
                    tail = min(spec.eval_tail_tokens, targets.size(1))
                    tail_nll = F.cross_entropy(
                        logits[:, -tail:].reshape(-1, logits.size(-1)),
                        targets[:, -tail:].reshape(-1),
                    )
                rows.append(
                    {
                        "length": length,
                        "anchor_index": anchor_index,
                        "anchor_endpoint": int(endpoint),
                        "full_nll": float(full_nll),
                        "tail_nll": float(tail_nll),
                        "tail_tokens": tail,
                    }
                )
                del batch, logits

    summary: dict[str, Any] = {}
    for length in spec.eval_lengths:
        selected = [row for row in rows if row["length"] == length]
        full = [row["full_nll"] for row in selected]
        tail = [row["tail_nll"] for row in selected]
        summary[str(length)] = {
            "full_nll_mean": statistics.fmean(full),
            "full_nll_sample_std": statistics.stdev(full) if len(full) > 1 else 0.0,
            "tail_nll_mean": statistics.fmean(tail),
            "tail_nll_sample_std": statistics.stdev(tail) if len(tail) > 1 else 0.0,
            "tail_ppl": math.exp(statistics.fmean(tail)),
            "anchors": len(selected),
        }
    result = {
        "status": "PASS",
        "suite": args.suite,
        "arm": args.arm,
        "seed": args.seed,
        "split": args.split,
        "protocol_sha256": spec.fingerprint(),
        "code_sha256": code_fingerprint(),
        "training_code_sha256": train_metadata["code_sha256"],
        "data_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "loaded_inv_freq_sha256": loaded_inv_freq_sha256,
        "anchor_sha256": anchor_record["sha256"],
        "runtime": runtime,
        "elapsed_seconds": time.time() - started,
        "metric_definition": (
            "teacher-forced causal NLL; tail NLL covers the last registered "
            "target tokens of each fixed held-out window"
        ),
        "summary": summary,
        "rows": rows,
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def select_tau(work_dir: Path) -> dict[str, Any]:
    suite = "shape_l128"
    primary_lengths = (1_024, 2_048, 4_096, 8_192)
    candidates = []
    for arm in SHAPE_TAU_ARMS:
        path = _run_dir(work_dir, suite, arm, 42) / "eval_selection.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text())
        score = statistics.fmean(
            result["summary"][str(length)]["tail_nll_mean"]
            for length in primary_lengths
        )
        _, schedule = training_inv_freq(suite, arm)
        candidates.append(
            {
                "arm": arm,
                "tau": float(schedule.get("tau", 0.0)),
                "selection_score": score,
                "source": str(path),
                "source_sha256": sha256_file(path),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            row["selection_score"],
            row["tau"],
            row["arm"],
        ),
    )
    receipt = {
        "status": "PASS",
        "suite": suite,
        "selection_split": "selection_anchors",
        "metric": "mean tail NLL across L=1024,2048,4096,8192",
        "selected_arm": selected["arm"],
        "selected_tau": selected["tau"],
        "candidates": candidates,
        "test_split_not_read": True,
    }
    path = work_dir.resolve() / "tau_selection.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite selection receipt: {path}")
    _atomic_json(path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def summarize(work_dir: Path, suite: str) -> dict[str, Any]:
    spec = SPECS[suite]
    rows = []
    initialization_by_seed: dict[int, dict[str, str]] = {}
    for arm in arms_for_suite(suite):
        for seed in seeds_for_arm(suite, arm):
            run_dir = _run_dir(work_dir, suite, arm, seed)
            train_path = run_dir / "train_result.json"
            path = run_dir / "eval_test.json"
            if not path.is_file():
                continue
            if not train_path.is_file():
                raise FileNotFoundError(train_path)
            train_result = json.loads(train_path.read_text())
            initialization_by_seed.setdefault(seed, {})[arm] = train_result[
                "initial_trainable_sha256"
            ]
            result = json.loads(path.read_text())
            for length, metrics in result["summary"].items():
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "length": int(length),
                        "tail_nll": metrics["tail_nll_mean"],
                        "full_nll": metrics["full_nll_mean"],
                        "source": str(path),
                        "source_sha256": sha256_file(path),
                    }
                )
    for seed, hashes in initialization_by_seed.items():
        if len(set(hashes.values())) != 1:
            raise RuntimeError(
                f"trainable initialization differs within seed {seed}: {hashes}"
            )
    aggregates = []
    for arm in arms_for_suite(suite):
        for length in spec.eval_lengths:
            selected = [
                row
                for row in rows
                if row["arm"] == arm and row["length"] == length
            ]
            if not selected:
                continue
            tail = [row["tail_nll"] for row in selected]
            aggregates.append(
                {
                    "arm": arm,
                    "length": length,
                    "seeds": [row["seed"] for row in selected],
                    "tail_nll_mean": statistics.fmean(tail),
                    "tail_nll_sample_std": (
                        statistics.stdev(tail) if len(tail) > 1 else None
                    ),
                }
            )
    report = {
        "status": "PASS" if rows else "INCOMPLETE",
        "suite": suite,
        "protocol_sha256": spec.fingerprint(),
        "rows": rows,
        "aggregates": aggregates,
        "initial_trainable_sha256_by_seed": initialization_by_seed,
        "claim_boundary": (
            "Matched fresh FineWeb-Edu ablation. Single-seed tau-grid rows and "
            "three-seed core rows retain their distinct evidence levels."
        ),
    }
    path = work_dir.resolve() / f"summary_{suite}.json"
    _atomic_json(path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--data_manifest", type=Path, required=True)
    preflight.add_argument("--full_hash_check", action="store_true")
    preflight.add_argument("--verify_full_initialization", action="store_true")

    probe_parser = sub.add_parser("probe-gpu")
    probe_parser.add_argument("--suite", choices=tuple(SPECS), required=True)
    probe_parser.add_argument("--data_manifest", type=Path, required=True)
    probe_parser.add_argument("--work_dir", type=Path, required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--suite", choices=tuple(SPECS), required=True)
    train_parser.add_argument("--arm", required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--data_manifest", type=Path, required=True)
    train_parser.add_argument("--work_dir", type=Path, required=True)
    train_parser.add_argument("--num_workers", type=int, default=8)
    train_parser.add_argument("--log_every", type=int, default=25)
    train_parser.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    train_parser.add_argument("--no_compile", action="store_true")
    train_parser.add_argument("--full_hash_check", action="store_true")

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--suite", choices=tuple(SPECS), required=True)
    eval_parser.add_argument("--arm", required=True)
    eval_parser.add_argument("--seed", type=int, required=True)
    eval_parser.add_argument(
        "--split", choices=("selection", "test"), required=True
    )
    eval_parser.add_argument("--data_manifest", type=Path, required=True)
    eval_parser.add_argument("--work_dir", type=Path, required=True)
    eval_parser.add_argument("--full_hash_check", action="store_true")

    select_parser = sub.add_parser("select-tau")
    select_parser.add_argument("--work_dir", type=Path, required=True)

    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--suite", choices=tuple(SPECS), required=True)
    summary_parser.add_argument("--work_dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "preflight":
        run_preflight(
            args.data_manifest,
            full_hash_check=args.full_hash_check,
            verify_full_initialization=args.verify_full_initialization,
        )
    elif args.command == "probe-gpu":
        probe_gpu(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "select-tau":
        select_tau(args.work_dir)
    elif args.command == "summarize":
        summarize(args.work_dir, args.suite)


if __name__ == "__main__":
    main()
