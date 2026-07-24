#!/usr/bin/env python3
"""Train, evaluate, gate, and summarize the MLA scarcity experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from rebuttal.rebuttal_0723.mla_scarcity_5090.analyze_schedules import (
    build_diagnostics,
)
from rebuttal.rebuttal_0723.mla_scarcity_5090.prepare import (
    sha256_file,
    validate_manifest,
)
from rebuttal.rebuttal_0723.mla_scarcity_5090.protocol import (
    ARMS,
    BASE,
    CHECKPOINT_LABELS,
    CONFIRMATORY_SEEDS,
    FREQUENCY_PAIRS,
    GATE_SEED,
    SEEDS,
    SPEC,
    TAU,
    learning_rate_for_step,
    schedule_phi,
    training_inv_freq,
)
from scripts.core_text_phases.run_gqa_evq_experiment import GPT
from scripts.lib.rope.official_yarn import (
    official_yarn_on_inv_freq,
    official_yarn_on_native_grid,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
GIB = 2**30
MINIMUM_TRAIN_FREE_BYTES = 8 * GIB


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def trainable_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(
            parameter.detach().cpu().float().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    paths = (
        PACKAGE_DIR / "protocol.py",
        PACKAGE_DIR / "prepare.py",
        PACKAGE_DIR / "analyze_schedules.py",
        PACKAGE_DIR / "run_experiment.py",
        PACKAGE_DIR / "run_5090.sh",
        REPO_ROOT / "scripts/core_text_phases/run_gqa_evq_experiment.py",
        REPO_ROOT / "scripts/core_text_phases/run_evq_sweep.py",
        REPO_ROOT / "scripts/lib/rope/schedules.py",
        REPO_ROOT / "scripts/lib/rope/official_yarn.py",
        REPO_ROOT / "tests/test_mla_scarcity_5090.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"fingerprint input missing: {path}")
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(path)
        current = current.parent
    return current


def disk_status(work_dir: Path) -> dict[str, Any]:
    existing = _existing_parent(work_dir)
    usage = shutil.disk_usage(existing)
    return {
        "path": str(existing),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "minimum_free_before_train_bytes": MINIMUM_TRAIN_FREE_BYTES,
        "enough_for_one_train_run": (
            int(usage.free) >= MINIMUM_TRAIN_FREE_BYTES
        ),
    }


def checkpoint_storage_budget(parameter_count: int) -> dict[str, Any]:
    fp32_parameter_bytes = int(parameter_count) * 4
    # State dict metadata and persistent buffers are small here. Use a 25%
    # conservative envelope so the receipt remains useful across serializers.
    checkpoint_upper_bound = math.ceil(fp32_parameter_bytes * 1.25)
    seed42_retained = 6 * 2
    # At confirmation start, twelve seed-42 proof checkpoints remain while one
    # new run temporarily holds 100M/200M/300M.
    maximum_simultaneous = seed42_retained + 3
    checkpoint_peak = maximum_simultaneous * checkpoint_upper_bound
    return {
        "fp32_parameter_bytes": fp32_parameter_bytes,
        "checkpoint_upper_bound_bytes": checkpoint_upper_bound,
        "seed42_gate_retained_checkpoints": seed42_retained,
        "maximum_simultaneous_checkpoints": maximum_simultaneous,
        "checkpoint_peak_upper_bound_bytes": checkpoint_peak,
        "minimum_free_before_each_train_bytes": MINIMUM_TRAIN_FREE_BYTES,
        "reserve_after_checkpoint_peak_bytes": (
            MINIMUM_TRAIN_FREE_BYTES - checkpoint_peak
        ),
        "confirmatory_terminal_checkpoint_count": 0,
    }


def require_training_disk_budget(work_dir: Path) -> dict[str, Any]:
    record = disk_status(work_dir)
    if not record["enough_for_one_train_run"]:
        raise RuntimeError(
            "refusing to start training with only "
            f"{record['free_bytes'] / GIB:.2f} GiB free; "
            f"at least {MINIMUM_TRAIN_FREE_BYTES / GIB:.0f} GiB is required"
        )
    return record


def validate_compile_cache(work_dir: Path) -> dict[str, Any]:
    expected = work_dir.resolve() / "torchinductor_cache"
    configured = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not configured:
        raise RuntimeError("TORCHINDUCTOR_CACHE_DIR must be set")
    actual = Path(configured).resolve()
    if actual != expected:
        raise RuntimeError(
            f"compile cache must be the registered shared path {expected}, "
            f"found {actual}"
        )
    if not actual.is_dir():
        raise FileNotFoundError(actual)
    return {
        "path": str(actual),
        "shared_across_runs": True,
        "cleanup_policy": (
            "retain until STOP gate or confirmatory summary, then explicit "
            "cleanup-compile-cache"
        ),
    }


def configure_cuda_kernels() -> dict[str, Any]:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(
        torch.backends.cuda.matmul,
        "allow_bf16_reduced_precision_reduction",
    ):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            True
        )
    # The experiment has an eligible BF16/head-dim-64 causal SDPA shape.
    # Fail early instead of silently falling back to quadratic math attention.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    flash_available_fn = getattr(
        torch.backends.cuda, "is_flash_attention_available", None
    )
    flash_available = (
        bool(flash_available_fn())
        if flash_available_fn is not None
        else None
    )
    if flash_available is False:
        raise RuntimeError("PyTorch reports Flash SDPA unavailable")
    result = {
        "flash_sdp_enabled": bool(
            torch.backends.cuda.flash_sdp_enabled()
        ),
        "mem_efficient_sdp_enabled": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "math_sdp_enabled": bool(
            torch.backends.cuda.math_sdp_enabled()
        ),
        "flash_attention_available": flash_available,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    cudnn_enabled_fn = getattr(
        torch.backends.cuda, "cudnn_sdp_enabled", None
    )
    result["cudnn_sdp_enabled"] = (
        bool(cudnn_enabled_fn()) if cudnn_enabled_fn else None
    )
    if (
        not result["flash_sdp_enabled"]
        or result["mem_efficient_sdp_enabled"]
        or result["math_sdp_enabled"]
        or result["cudnn_sdp_enabled"] is True
    ):
        raise RuntimeError("Flash-only SDPA kernel contract is not active")
    return result


def deterministic_row_order(rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(int(rows), generator=generator, dtype=torch.int64)


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
    def __init__(self, path: Path, *, rows: int, length: int) -> None:
        self.path = str(path.resolve())
        self.rows = int(rows)
        self.length = int(length)
        self._array: np.ndarray | None = None
        self._flat: np.ndarray | None = None
        self._open()
        assert self._flat is not None
        if len(self._flat) < self.rows * self.length:
            raise ValueError("training tensor is shorter than frozen prefix")
        self._array = None
        self._flat = None

    def _open(self) -> None:
        if self._array is None:
            self._array = np.load(
                self.path, mmap_mode="r", allow_pickle=False
            )
            if self._array.dtype != np.int64:
                raise ValueError("training tensor must contain int64 ids")
            self._flat = self._array.reshape(-1)

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> torch.Tensor:
        self._open()
        assert self._flat is not None
        current = int(index)
        if current < 0 or current >= self.rows:
            raise IndexError(current)
        start = current * self.length
        return torch.from_numpy(
            np.array(
                self._flat[start : start + self.length],
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


def per_sequence_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    tail_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if (
        logits.ndim != 3
        or targets.ndim != 2
        or logits.shape[:2] != targets.shape
    ):
        raise ValueError("logits/targets have incompatible sequence shapes")
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(targets.size(0), targets.size(1))
    tail = min(int(tail_tokens), int(targets.size(1)))
    if tail <= 0:
        raise ValueError("tail token count must be positive")
    return (
        per_token.mean(dim=1),
        per_token[:, -tail:].mean(dim=1),
        tail,
    )


def build_model(frequency_pairs: int, arm: str, seed: int) -> GPT:
    seed_everything(seed)
    inv, _ = training_inv_freq(arm, frequency_pairs)
    return GPT(SPEC.model_config(frequency_pairs), inv)


def _rope(model: GPT):
    reference = model.blocks[0].attn.rope
    if any(block.attn.rope is not reference for block in model.blocks):
        raise RuntimeError("MLA blocks do not share one rotary module")
    return reference


def model_inv_freq(model: GPT) -> torch.Tensor:
    return (
        _rope(model)
        .inv_freq.detach()
        .cpu()
        .float()
        .contiguous()
        .clone()
    )


def _state_inv_freq(
    state: dict[str, torch.Tensor], *, label: str
) -> torch.Tensor:
    values = [
        value.detach().cpu().float().contiguous()
        for name, value in state.items()
        if name.endswith(".rope.inv_freq")
    ]
    if not values:
        raise ValueError(f"{label} has no persistent inv_freq")
    reference = values[0]
    if any(not torch.equal(reference, value) for value in values[1:]):
        raise ValueError(f"{label} contains inconsistent inv_freq buffers")
    return reference.clone()


def _run_dir(
    work_dir: Path, frequency_pairs: int, arm: str, seed: int
) -> Path:
    return (
        work_dir.resolve()
        / "runs"
        / f"k{int(frequency_pairs)}"
        / arm
        / f"seed{int(seed)}"
    )


def _load_manifest(
    path: Path, *, full_hash_check: bool, prefix_hash_check: bool
) -> dict[str, Any]:
    manifest = json.loads(path.resolve().read_text())
    validate_manifest(
        manifest,
        check_hashes=bool(full_hash_check),
        check_prefix=bool(prefix_hash_check),
    )
    return manifest


def enforce_nested_operator_parity_phase(
    manifest: dict[str, Any],
    work_dir: Path,
    *,
    seed: int,
    split: str | None = None,
) -> None:
    """Fail closed for the follow-up protocol that reuses this trainer."""
    nested = manifest.get("operator_parity")
    if not isinstance(nested, dict):
        return
    nested_protocol = nested.get("protocol_sha256")
    if (
        not isinstance(nested_protocol, str)
        or len(nested_protocol) != 64
    ):
        raise RuntimeError(
            "operator-parity manifest lacks a valid protocol identity"
        )
    current_seed = int(seed)
    if split == "selection" and current_seed != GATE_SEED:
        raise RuntimeError(
            "operator-parity selection is restricted to seed 42"
        )
    requires_pass = current_seed != GATE_SEED or split == "test"
    if not requires_pass:
        return
    gate_path = (
        Path(work_dir).resolve() / "operator_parity_gate.json"
    )
    if not gate_path.is_file():
        raise RuntimeError(
            "operator-parity confirmatory work requires a PASS seed-42 gate"
        )
    gate_record = json.loads(gate_path.read_text())
    ready_path = (
        Path(work_dir).resolve() / "operator_parity_ready.json"
    )
    if not ready_path.is_file():
        raise RuntimeError(
            "operator-parity confirmatory work requires its READY receipt"
        )
    parity_ready = json.loads(ready_path.read_text())
    expected = {
        "status": "PASS",
        "protocol_sha256": nested_protocol,
        "selection_split_only": True,
        "test_split_read": False,
        "evaluation_code_sha256": parity_ready.get(
            "evaluation_code_sha256"
        ),
        "ready_receipt_sha256": sha256_file(ready_path),
    }
    if (
        parity_ready.get("status") != "READY"
        or parity_ready.get("protocol_sha256") != nested_protocol
    ):
        raise RuntimeError("operator-parity READY identity mismatch")
    for key, value in expected.items():
        if gate_record.get(key) != value:
            raise RuntimeError(
                f"operator-parity confirmatory gate mismatch: {key}"
            )


def validate_cuda_runtime() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA is required")
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
    memory = getattr(
        props, "total_memory", getattr(props, "total_mem", 0)
    )
    if int(memory) < 30 * 2**30:
        raise RuntimeError(
            f"registered 5090 run needs >=30 GiB, found "
            f"{int(memory) / 2**30:.1f} GiB"
        )
    kernels = configure_cuda_kernels()
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable")
    return {
        "name": props.name,
        "capability": list(capability),
        "architecture": architecture,
        "compiled_architectures": compiled_architectures,
        "total_memory_bytes": int(memory),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "kernels": kernels,
        "torch_compile_available": True,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.data_manifest.resolve()
    compile_cache = validate_compile_cache(args.work_dir)
    diagnostic_path = (
        args.work_dir.resolve() / "schedule_diagnostics.json"
    )
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    diagnostic = json.loads(diagnostic_path.read_text())
    if (
        diagnostic.get("status") != "MODEL_FREE_DIAGNOSTIC_ONLY"
        or diagnostic.get("protocol_sha256") != SPEC.fingerprint()
        or diagnostic != build_diagnostics()
    ):
        raise ValueError("schedule diagnostic identity mismatch")
    manifest = _load_manifest(
        manifest_path,
        full_hash_check=bool(args.full_hash_check),
        prefix_hash_check=bool(args.prefix_hash_check),
    )
    records: dict[str, Any] = {}
    parameter_counts: dict[int, set[int]] = {}
    all_initial_hashes: set[str] = set()
    for pairs in FREQUENCY_PAIRS:
        arm_records: dict[str, Any] = {}
        initial_hashes: dict[str, str] = {}
        counts: set[int] = set()
        evq_phi, _ = schedule_phi("evq_cosh", pairs)
        range_phi, _ = schedule_phi("range_matched_uniform", pairs)
        if evq_phi[0] != range_phi[0] or evq_phi[-1] != range_phi[-1]:
            raise RuntimeError("range control does not match EVQ endpoints")
        for arm in ARMS:
            inv, metadata = training_inv_freq(arm, pairs)
            active = inv[:pairs]
            inactive = inv[pairs:]
            if (
                inv.shape != (SPEC.rotary_pair_capacity,)
                or not torch.isfinite(inv).all()
                or not torch.all(torch.diff(active) < 0)
                or not torch.all(inactive == 0)
            ):
                raise RuntimeError(f"invalid schedule for K={pairs}/{arm}")
            model = build_model(pairs, arm, GATE_SEED)
            count = sum(parameter.numel() for parameter in model.parameters())
            counts.add(count)
            if args.verify_full_initialization:
                initial_hashes[arm] = trainable_state_sha256(model)
                all_initial_hashes.add(initial_hashes[arm])
            arm_records[arm] = {
                "parameter_count": count,
                "inv_freq_sha256": tensor_sha256(inv),
                "inv_freq": [float(value) for value in inv],
                "schedule": metadata,
            }
            del model
            gc.collect()
        if len(counts) != 1:
            raise RuntimeError(f"parameter count differs across K={pairs} arms")
        if (
            args.verify_full_initialization
            and len(set(initial_hashes.values())) != 1
        ):
            raise RuntimeError(
                f"trainable initialization differs within K={pairs}"
            )
        parameter_counts[pairs] = counts
        records[f"k{pairs}"] = {
            "arms": arm_records,
            "initial_trainable_sha256": initial_hashes,
        }
    if len({next(iter(value)) for value in parameter_counts.values()}) != 1:
        raise RuntimeError("parameter count differs across K budgets")
    if args.verify_full_initialization and len(all_initial_hashes) != 1:
        raise RuntimeError("trainable initialization differs across K budgets")
    common_parameter_count = next(
        iter(parameter_counts[FREQUENCY_PAIRS[0]])
    )
    storage = disk_status(args.work_dir)
    storage["budget"] = checkpoint_storage_budget(common_parameter_count)
    result = {
        "schema_version": 1,
        "status": "READY",
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "data_manifest_sha256": sha256_file(manifest_path),
        "data": {
            "train_sha256": manifest["train"]["sha256"],
            "train_prefix_sha256": manifest["train"][
                "token_prefix_sha256"
            ],
            "validation_sha256": manifest["validation"]["sha256"],
            "selection_anchor_sha256": manifest["selection_anchors"][
                "sha256"
            ],
            "test_anchor_sha256": manifest["test_anchors"]["sha256"],
        },
        "protocol": {
            "frequency_pairs": FREQUENCY_PAIRS,
            "arms": ARMS,
            "seeds": SEEDS,
            "train_tokens": SPEC.train_tokens,
            "checkpoint_tokens": SPEC.checkpoint_tokens,
            "tau": TAU,
            "base": BASE,
        },
        "configurations": records,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "storage": storage,
        "compile_cache": compile_cache,
        "schedule_diagnostic": {
            "path": str(diagnostic_path),
            "sha256": sha256_file(diagnostic_path),
            "status": diagnostic["status"],
        },
    }
    output = args.work_dir.resolve() / "ready_receipt.json"
    if output.exists():
        raise FileExistsError(output)
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def validate_ready(
    data_manifest: Path, work_dir: Path, *, require_disk: bool
) -> dict[str, Any]:
    manifest_path = data_manifest.resolve()
    work_dir = work_dir.resolve()
    manifest = _load_manifest(
        manifest_path, full_hash_check=False, prefix_hash_check=False
    )
    receipt_path = work_dir / "ready_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "status": "READY",
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "data_manifest_sha256": sha256_file(manifest_path),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"READY receipt mismatch: {key}")
    data_expected = {
        "train_sha256": manifest["train"]["sha256"],
        "train_prefix_sha256": manifest["train"]["token_prefix_sha256"],
        "validation_sha256": manifest["validation"]["sha256"],
        "selection_anchor_sha256": manifest["selection_anchors"]["sha256"],
        "test_anchor_sha256": manifest["test_anchors"]["sha256"],
    }
    if receipt.get("data") != data_expected:
        raise ValueError("READY receipt data identity mismatch")
    compile_cache = validate_compile_cache(work_dir)
    if receipt.get("compile_cache") != compile_cache:
        raise ValueError("READY receipt compile-cache contract mismatch")
    diagnostic_record = receipt.get("schedule_diagnostic", {})
    diagnostic_path = Path(str(diagnostic_record.get("path", "")))
    if (
        not diagnostic_path.is_file()
        or sha256_file(diagnostic_path) != diagnostic_record.get("sha256")
    ):
        raise ValueError("READY receipt schedule diagnostic mismatch")
    diagnostic = json.loads(diagnostic_path.read_text())
    if (
        diagnostic.get("status") != "MODEL_FREE_DIAGNOSTIC_ONLY"
        or diagnostic.get("protocol_sha256") != SPEC.fingerprint()
        or diagnostic != build_diagnostics()
    ):
        raise ValueError("schedule diagnostic protocol mismatch")
    storage = (
        require_training_disk_budget(work_dir)
        if require_disk
        else disk_status(work_dir)
    )
    result = {
        "status": "PASS",
        "ready_receipt": str(receipt_path),
        **expected,
        "storage": storage,
        "compile_cache": compile_cache,
        "schedule_diagnostic": diagnostic_record,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def probe_gpu(args: argparse.Namespace) -> dict[str, Any]:
    ready = validate_ready(
        args.data_manifest, args.work_dir, require_disk=True
    )
    compile_cache = ready["compile_cache"]
    runtime = validate_cuda_runtime()
    pairs = int(args.frequency_pairs)
    manifest = _load_manifest(
        args.data_manifest.resolve(),
        full_hash_check=False,
        prefix_hash_check=False,
    )
    path = args.work_dir.resolve() / f"gpu_probe_k{pairs}.json"
    if path.exists():
        raise FileExistsError(path)
    source = np.load(
        manifest["train"]["path"], mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    count = SPEC.micro_batch_size * SPEC.train_length
    batch = torch.from_numpy(
        np.array(source[:count], dtype=np.int64, copy=True).reshape(
            SPEC.micro_batch_size, SPEC.train_length
        )
    ).to("cuda")
    model = build_model(pairs, "native_geo", GATE_SEED).to("cuda")
    loss_module: nn.Module = CausalLanguageModelLoss(model)
    loss_module = torch.compile(
        loss_module,
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
    compile_started = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = loss_module(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    compile_elapsed = time.time() - compile_started
    timed_steps = int(args.timed_steps)
    if timed_steps < 1:
        raise ValueError("timed probe steps must be positive")
    timed_started = time.time()
    for _ in range(timed_steps):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_module(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    timed_elapsed = time.time() - timed_started
    calibrated_tokens = (
        timed_steps * SPEC.micro_batch_size * SPEC.train_length
    )
    tokens_per_second = calibrated_tokens / timed_elapsed
    result = {
        "status": "PASS",
        "discarded_probe": True,
        "frequency_pairs": pairs,
        "loss": float(loss.detach()),
        "compile_step_seconds": compile_elapsed,
        "timed_steps": timed_steps,
        "timed_tokens": calibrated_tokens,
        "timed_seconds": timed_elapsed,
        "tokens_per_second": tokens_per_second,
        "estimated_seconds_per_300m_run": (
            SPEC.train_tokens / tokens_per_second
        ),
        "estimated_seconds_seed42_gate_training_only": (
            6 * SPEC.train_tokens / tokens_per_second
        ),
        "estimated_seconds_all_18_training_only": (
            18 * SPEC.train_tokens / tokens_per_second
        ),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime": runtime,
        "compile_cache": compile_cache,
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
    }
    _atomic_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _save_checkpoint(
    path: Path, model: GPT, metadata: dict[str, Any]
) -> str:
    temporary = path.with_name(path.name + ".incomplete")
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    checkpoint_inv = _state_inv_freq(state, label=str(path))
    if tensor_sha256(checkpoint_inv) != metadata["inv_freq_sha256"]:
        raise ValueError("model inv_freq changed before checkpoint save")
    torch.save({"model": state, "metadata": metadata}, temporary)
    temporary.replace(path)
    del state
    return sha256_file(path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    ready = validate_ready(
        args.data_manifest, args.work_dir, require_disk=True
    )
    compile_cache = ready["compile_cache"]
    runtime = validate_cuda_runtime()
    pairs = int(args.frequency_pairs)
    arm = str(args.arm)
    seed = int(args.seed)
    if pairs not in FREQUENCY_PAIRS or arm not in ARMS or seed not in SEEDS:
        raise ValueError("unregistered K/arm/seed")
    manifest_path = args.data_manifest.resolve()
    manifest = _load_manifest(
        manifest_path,
        full_hash_check=bool(args.full_hash_check),
        prefix_hash_check=False,
    )
    enforce_nested_operator_parity_phase(
        manifest,
        args.work_dir,
        seed=seed,
    )
    output = _run_dir(args.work_dir, pairs, arm, seed)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(pairs, arm, seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_hash = trainable_state_sha256(model)
    inv, schedule_metadata = training_inv_freq(arm, pairs)
    inv_hash = tensor_sha256(inv)
    np.save(output / "inv_freq.npy", inv.numpy(), allow_pickle=False)
    order = deterministic_row_order(SPEC.train_rows, seed)
    order_hash = tensor_sha256(order)
    dataset = FlatPrefixDataset(
        Path(manifest["train"]["path"]),
        rows=SPEC.train_rows,
        length=SPEC.train_length,
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
        drop_last=False,
    )
    if len(loader) != SPEC.optimizer_steps * SPEC.grad_accum_steps:
        raise RuntimeError("data-loader step count differs from protocol")
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
        lr=SPEC.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=SPEC.weight_decay,
        fused=True,
    )
    metadata = {
        "schema_version": 1,
        "status": "RUNNING",
        "frequency_pairs": pairs,
        "d_rope": 2 * SPEC.rotary_pair_capacity,
        "active_frequency_pairs": pairs,
        "active_rotary_dimensions": 2 * pairs,
        "arm": arm,
        "seed": seed,
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "data_manifest_sha256": sha256_file(manifest_path),
        "train_tensor_sha256": manifest["train"]["sha256"],
        "train_prefix_sha256": manifest["train"]["token_prefix_sha256"],
        "validation_sha256": manifest["validation"]["sha256"],
        "selection_anchor_sha256": manifest["selection_anchors"]["sha256"],
        "test_anchor_sha256": manifest["test_anchors"]["sha256"],
        "initial_trainable_sha256": initial_hash,
        "row_order_sha256": order_hash,
        "inv_freq_sha256": inv_hash,
        "schedule": schedule_metadata,
        "model_config": SPEC.model_config(pairs),
        "parameter_count": parameter_count,
        "runtime": runtime,
        "compile": {
            "enabled": not args.no_compile,
            "mode": None if args.no_compile else args.compile_mode,
            "torchinductor_cache_dir": os.environ.get(
                "TORCHINDUCTOR_CACHE_DIR"
            ),
            "cache_contract": compile_cache,
        },
        "storage_at_start": ready["storage"],
        "train": {
            "length": SPEC.train_length,
            "tokens": SPEC.train_tokens,
            "optimizer_steps": SPEC.optimizer_steps,
            "global_batch_size": SPEC.global_batch_size,
            "micro_batch_size": SPEC.micro_batch_size,
            "checkpoint_steps": SPEC.checkpoint_steps,
            "checkpoint_tokens": SPEC.checkpoint_tokens,
        },
    }
    _atomic_json(output / "metadata.json", metadata)
    checkpoints_by_step = {
        step: label for label, step in SPEC.checkpoint_steps.items()
    }
    checkpoint_records: dict[str, Any] = {}
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    recent_losses: list[float] = []
    peak_memory = 0
    for micro_step, batch in enumerate(loader):
        step = micro_step // SPEC.grad_accum_steps
        lr = learning_rate_for_step(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = batch.to("cuda", non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_module(batch)
            scaled = loss / SPEC.grad_accum_steps
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        scaled.backward()
        recent_losses.append(float(loss.detach()))
        if (micro_step + 1) % SPEC.grad_accum_steps:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed = step + 1
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
        if completed == 1 or completed % int(args.log_every) == 0:
            elapsed = max(time.time() - started, 1e-6)
            tokens_done = completed * SPEC.tokens_per_optimizer_step
            row = {
                "step": completed,
                "optimizer_steps": SPEC.optimizer_steps,
                "tokens_seen": tokens_done,
                "loss": statistics.fmean(
                    recent_losses[-SPEC.grad_accum_steps :]
                ),
                "lr": lr,
                "tokens_per_second": tokens_done / elapsed,
                "elapsed_seconds": elapsed,
            }
            _append_jsonl(output / "train_log.jsonl", row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed in checkpoints_by_step:
            label = checkpoints_by_step[completed]
            stage_metadata = {
                **metadata,
                "status": "PASS",
                "checkpoint_label": label,
                "optimizer_step": completed,
                "tokens_seen": completed * SPEC.tokens_per_optimizer_step,
                "elapsed_seconds": time.time() - started,
                "peak_cuda_memory_bytes": int(peak_memory),
                "loss_at_checkpoint": statistics.fmean(
                    recent_losses[-SPEC.grad_accum_steps :]
                ),
            }
            checkpoint = output / f"checkpoint_{label}.pt"
            checkpoint_sha = _save_checkpoint(
                checkpoint, model, stage_metadata
            )
            record = {
                "status": "PASS",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                **stage_metadata,
            }
            _atomic_json(output / f"checkpoint_{label}.json", record)
            checkpoint_records[label] = {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "optimizer_step": completed,
                "tokens_seen": stage_metadata["tokens_seen"],
            }
    if set(checkpoint_records) != set(CHECKPOINT_LABELS):
        raise RuntimeError("one or more registered checkpoints were not saved")
    result = {
        **metadata,
        "status": "PASS",
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_bytes": int(peak_memory),
        "final_loss": recent_losses[-1],
        "checkpoints": checkpoint_records,
    }
    _atomic_json(output / "train_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_checkpoint(
    checkpoint: Path,
    *,
    frequency_pairs: int,
    arm: str,
    seed: int,
    stage: str,
) -> tuple[GPT, dict[str, Any], str]:
    receipt_path = checkpoint.parent / f"checkpoint_{stage}.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    checkpoint_sha = sha256_file(checkpoint)
    if receipt.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("checkpoint hash differs from stage receipt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model")
    metadata = payload.get("metadata")
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        raise ValueError("checkpoint payload is incomplete")
    expected = {
        "frequency_pairs": int(frequency_pairs),
        "arm": arm,
        "seed": int(seed),
        "checkpoint_label": stage,
        "protocol_sha256": SPEC.fingerprint(),
    }
    for key, value in expected.items():
        if metadata.get(key) != value or receipt.get(key) != value:
            raise ValueError(f"checkpoint identity mismatch: {key}")
    expected_inv, _ = training_inv_freq(arm, frequency_pairs)
    checkpoint_inv = _state_inv_freq(state, label=str(checkpoint))
    if (
        tensor_sha256(expected_inv) != metadata.get("inv_freq_sha256")
        or not torch.equal(expected_inv, checkpoint_inv)
    ):
        raise ValueError("checkpoint frequency schedule mismatch")
    sidecar = torch.from_numpy(
        np.load(checkpoint.parent / "inv_freq.npy", allow_pickle=False)
    ).contiguous()
    if sidecar.dtype != torch.float32 or not torch.equal(
        sidecar, checkpoint_inv
    ):
        raise ValueError("checkpoint frequency sidecar mismatch")
    model = build_model(frequency_pairs, arm, seed)
    model.load_state_dict(state, strict=True)
    if not torch.equal(model_inv_freq(model), checkpoint_inv):
        raise ValueError("checkpoint inv_freq changed after strict load")
    return model, metadata, checkpoint_sha


def _runtime_operator(
    base_inv: torch.Tensor,
    *,
    frequency_pairs: int,
    arm: str,
    length: int,
    operator: str,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    if operator == "raw":
        return base_inv.clone(), 1.0, {
            "operator": "raw",
            "public_label": "raw trained frequency substrate",
            "scale": 1.0,
            "mscale": 1.0,
        }
    scale = float(length) / float(SPEC.train_length)
    d_rope = 2 * int(frequency_pairs)
    active_inv = base_inv[: int(frequency_pairs)]
    if arm == "native_geo":
        active_out, mscale, metadata = official_yarn_on_native_grid(
            head_dim=d_rope,
            base=BASE,
            scale=scale,
            original_max_position_embeddings=SPEC.train_length,
            beta_fast=32.0,
            beta_slow=1.0,
        )
        label = (
            "official YaRN on native endpoint RoPE"
            if frequency_pairs == SPEC.rotary_pair_capacity
            else (
                "official YaRN equations on active native endpoint grid, "
                "identity-padded to fixed rotary capacity"
            )
        )
    else:
        active_out, mscale, metadata = official_yarn_on_inv_freq(
            active_inv.to(torch.float64),
            head_dim=d_rope,
            base=BASE,
            scale=scale,
            original_max_position_embeddings=SPEC.train_length,
            beta_fast=32.0,
            beta_slow=1.0,
        )
        label = f"YaRN-derived virtual-coordinate transform on {arm}"
    inv = torch.zeros_like(base_inv, dtype=torch.float64)
    inv[: int(frequency_pairs)] = active_out
    return inv, float(mscale), {
        **metadata,
        "operator": "yarn_full",
        "public_label": label,
        "scale": scale,
        "mscale": float(mscale),
        "full_standard_official_yarn": (
            arm == "native_geo"
            and frequency_pairs == SPEC.rotary_pair_capacity
        ),
    }


def _set_runtime_rope(
    model: GPT,
    inv_freq: torch.Tensor,
    *,
    max_position: int,
    mscale: float,
) -> None:
    rope = _rope(model)
    value = inv_freq.to(
        device=rope.inv_freq.device, dtype=rope.inv_freq.dtype
    )
    if value.shape != rope.inv_freq.shape:
        raise ValueError("runtime frequency shape mismatch")
    with torch.no_grad():
        rope.inv_freq.copy_(value)
        rope._build(int(max_position))
        if float(mscale) != 1.0:
            rope.cos_c.mul_(float(mscale))
            rope.sin_c.mul_(float(mscale))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    validate_ready(
        args.data_manifest, args.work_dir, require_disk=False
    )
    runtime = validate_cuda_runtime()
    pairs = int(args.frequency_pairs)
    arm = str(args.arm)
    seed = int(args.seed)
    stage = str(args.stage)
    split = str(args.split)
    operator = str(args.operator)
    if (
        pairs not in FREQUENCY_PAIRS
        or arm not in ARMS
        or seed not in SEEDS
        or stage not in CHECKPOINT_LABELS
    ):
        raise ValueError("unregistered evaluation condition")
    manifest_path = args.data_manifest.resolve()
    manifest = _load_manifest(
        manifest_path,
        full_hash_check=bool(args.full_hash_check),
        prefix_hash_check=False,
    )
    enforce_nested_operator_parity_phase(
        manifest,
        args.work_dir,
        seed=seed,
        split=split,
    )
    output = _run_dir(args.work_dir, pairs, arm, seed)
    checkpoint = output / f"checkpoint_{stage}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result_path = output / f"eval_{split}_{stage}_{operator}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    model, train_metadata, checkpoint_sha = _load_checkpoint(
        checkpoint,
        frequency_pairs=pairs,
        arm=arm,
        seed=seed,
        stage=stage,
    )
    if train_metadata["train_prefix_sha256"] != manifest["train"][
        "token_prefix_sha256"
    ]:
        raise ValueError("checkpoint training prefix differs from manifest")
    if train_metadata["validation_sha256"] != manifest["validation"][
        "sha256"
    ]:
        raise ValueError("checkpoint validation tensor differs from manifest")
    anchor_record = manifest[f"{split}_anchors"]
    if train_metadata[f"{split}_anchor_sha256"] != anchor_record["sha256"]:
        raise ValueError("checkpoint evaluation anchors differ from manifest")
    if sha256_file(anchor_record["path"]) != anchor_record["sha256"]:
        raise ValueError("evaluation anchor hash mismatch")
    anchors = np.load(anchor_record["path"], allow_pickle=False)
    validation = np.load(
        manifest["validation"]["path"], mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    nested_parity = manifest.get("operator_parity")
    evaluation_batch_sizes = {
        str(length): 1 for length in SPEC.eval_lengths
    }
    if isinstance(nested_parity, dict):
        registered_batches = nested_parity.get("evaluation_batch_sizes")
        if not isinstance(registered_batches, dict):
            raise ValueError(
                "operator-parity manifest lacks evaluation batch sizes"
            )
        evaluation_batch_sizes = {
            str(length): int(registered_batches.get(str(length), 0))
            for length in SPEC.eval_lengths
        }
        if any(value <= 0 for value in evaluation_batch_sizes.values()):
            raise ValueError("invalid operator-parity evaluation batch size")
    base_inv = model_inv_freq(model).to(torch.float64)
    model = model.to("cuda").eval()
    rows: list[dict[str, Any]] = []
    operators: dict[str, Any] = {}
    started = time.time()
    with torch.inference_mode():
        for length in SPEC.eval_lengths:
            runtime_inv, mscale, operator_metadata = _runtime_operator(
                base_inv,
                frequency_pairs=pairs,
                arm=arm,
                length=length,
                operator=operator,
            )
            _set_runtime_rope(
                model,
                runtime_inv,
                max_position=length,
                mscale=mscale,
            )
            operators[str(length)] = {
                **operator_metadata,
                "evaluation_batch_size": evaluation_batch_sizes[
                    str(length)
                ],
                "runtime_inv_freq_sha256": tensor_sha256(
                    runtime_inv.float()
                ),
            }
            endpoint_values = anchors.tolist()
            evaluation_batch = evaluation_batch_sizes[str(length)]
            for batch_start in range(
                0, len(endpoint_values), evaluation_batch
            ):
                current_endpoints = endpoint_values[
                    batch_start : batch_start + evaluation_batch
                ]
                tokens = np.stack(
                    [
                        np.array(
                            validation[
                                int(endpoint) - int(length) : int(endpoint)
                            ],
                            dtype=np.int64,
                            copy=True,
                        )
                        for endpoint in current_endpoints
                    ]
                )
                batch = torch.from_numpy(tokens).to("cuda")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch[:, :-1])
                    targets = batch[:, 1:]
                    full_nll, tail_nll, tail = per_sequence_nll(
                        logits,
                        targets,
                        tail_tokens=SPEC.eval_tail_tokens,
                    )
                full_values = full_nll.detach().cpu().tolist()
                tail_values = tail_nll.detach().cpu().tolist()
                for offset, endpoint in enumerate(current_endpoints):
                    rows.append(
                        {
                            "length": length,
                            "anchor_index": batch_start + offset,
                            "anchor_endpoint": int(endpoint),
                            "full_nll": float(full_values[offset]),
                            "tail_nll": float(tail_values[offset]),
                            "tail_tokens": tail,
                        }
                    )
                del batch, logits, targets, full_nll, tail_nll
            torch.cuda.empty_cache()
    summary: dict[str, Any] = {}
    for length in SPEC.eval_lengths:
        selected = [row for row in rows if row["length"] == length]
        full = [row["full_nll"] for row in selected]
        tail = [row["tail_nll"] for row in selected]
        summary[str(length)] = {
            "anchors": len(selected),
            "full_nll_mean": statistics.fmean(full),
            "full_nll_sample_std": statistics.stdev(full),
            "tail_nll_mean": statistics.fmean(tail),
            "tail_nll_sample_std": statistics.stdev(tail),
            "tail_ppl": math.exp(statistics.fmean(tail)),
        }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "frequency_pairs": pairs,
        "d_rope": 2 * SPEC.rotary_pair_capacity,
        "active_frequency_pairs": pairs,
        "active_rotary_dimensions": 2 * pairs,
        "arm": arm,
        "seed": seed,
        "stage": stage,
        "tokens_seen": train_metadata["tokens_seen"],
        "split": split,
        "operator": operator,
        "protocol_sha256": SPEC.fingerprint(),
        "evaluation_code_sha256": code_fingerprint(),
        "training_code_sha256": train_metadata["code_sha256"],
        "data_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": checkpoint_sha,
        "trained_inv_freq_sha256": train_metadata["inv_freq_sha256"],
        "anchor_sha256": anchor_record["sha256"],
        "runtime": runtime,
        "elapsed_seconds": time.time() - started,
        "metric_definition": (
            "teacher-forced causal NLL; primary tail NLL covers the final "
            "4096 targets of each fixed held-out window"
        ),
        "evaluation_batch_sizes": evaluation_batch_sizes,
        "operators": operators,
        "summary": summary,
        "rows": rows,
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _eval_path(
    work_dir: Path,
    pairs: int,
    arm: str,
    seed: int,
    split: str,
    stage: str,
    operator: str = "raw",
) -> Path:
    return _run_dir(work_dir, pairs, arm, seed) / (
        f"eval_{split}_{stage}_{operator}.json"
    )


def _effect_rows(
    work_dir: Path,
    *,
    split: str,
    stage: str,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        by_k: dict[int, dict[str, dict[str, Any]]] = {}
        for pairs in FREQUENCY_PAIRS:
            by_k[pairs] = {}
            for arm in ARMS:
                path = _eval_path(
                    work_dir, pairs, arm, seed, split, stage
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                result = json.loads(path.read_text())
                expected = {
                    "status": "PASS",
                    "frequency_pairs": pairs,
                    "arm": arm,
                    "seed": seed,
                    "stage": stage,
                    "split": split,
                    "operator": "raw",
                    "protocol_sha256": SPEC.fingerprint(),
                    "evaluation_code_sha256": code_fingerprint(),
                    "training_code_sha256": code_fingerprint(),
                }
                for key, value in expected.items():
                    if result.get(key) != value:
                        raise ValueError(
                            f"evaluation identity mismatch: {path}/{key}"
                        )
                by_k[pairs][arm] = result["summary"]
        for length in SPEC.eval_lengths:
            effects: dict[int, dict[str, float]] = {}
            for pairs in FREQUENCY_PAIRS:
                values = {
                    arm: by_k[pairs][arm][str(length)]["tail_nll_mean"]
                    for arm in ARMS
                }
                effects[pairs] = {
                    "native_nll": values["native_geo"],
                    "range_nll": values["range_matched_uniform"],
                    "evq_nll": values["evq_cosh"],
                    "range_gain": (
                        values["native_geo"]
                        - values["range_matched_uniform"]
                    ),
                    "shape_gain": (
                        values["range_matched_uniform"]
                        - values["evq_cosh"]
                    ),
                    "evq_minus_native": (
                        values["evq_cosh"] - values["native_geo"]
                    ),
                }
            rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "length": length,
                    "k8": effects[8],
                    "k32": effects[32],
                    "scarcity_interaction": (
                        effects[8]["shape_gain"]
                        - effects[32]["shape_gain"]
                    ),
                }
            )
    return rows


def gate(work_dir: Path) -> dict[str, Any]:
    path = work_dir.resolve() / "gate_receipt.json"
    if path.exists():
        raise FileExistsError(path)
    rows = []
    for stage in ("200m", "300m"):
        rows.extend(
            _effect_rows(
                work_dir,
                split="selection",
                stage=stage,
                seeds=(GATE_SEED,),
            )
        )
    by_stage_length = {
        (row["stage"], row["length"]): row for row in rows
    }
    primary_300 = by_stage_length[("300m", 8_192)]
    primary_200 = by_stage_length[("200m", 8_192)]
    in_domain = by_stage_length[("300m", 4_096)]["k8"][
        "evq_minus_native"
    ]
    criteria = {
        "k8_shape_gain_gt_0p05": (
            primary_300["k8"]["shape_gain"] > 0.05
        ),
        "positive_scarcity_interaction": (
            primary_300["scarcity_interaction"] > 0.0
        ),
        "k8_shape_gain_exceeds_range_gain": (
            primary_300["k8"]["shape_gain"]
            > primary_300["k8"]["range_gain"]
        ),
        "k8_in_domain_cost_lte_0p02": in_domain <= 0.02,
        "k8_shape_gain_does_not_reverse_200m_to_300m": (
            primary_200["k8"]["shape_gain"] > 0.0
            and primary_300["k8"]["shape_gain"] > 0.0
        ),
    }
    passed = all(criteria.values())
    result = {
        "schema_version": 1,
        "status": "PASS" if passed else "STOP",
        "decision": (
            "expand_to_confirmatory_seeds"
            if passed
            else "do_not_expand_without_new_author_decision"
        ),
        "selection_split_only": True,
        "test_split_read": False,
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "primary_length": 8_192,
        "criteria": criteria,
        "primary_200m": primary_200,
        "primary_300m": primary_300,
        "k8_in_domain_evq_minus_native": in_domain,
        "rows": rows,
    }
    _atomic_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _mean_ci(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "ci95": None, "values": values}
    critical = 4.302652729911275 if len(values) == 3 else 1.96
    margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "ci95": [mean - margin, mean + margin],
        "values": values,
    }


def summarize(work_dir: Path) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    gate_path = work_dir / "gate_receipt.json"
    if not gate_path.is_file():
        raise FileNotFoundError(gate_path)
    gate_record = json.loads(gate_path.read_text())
    if (
        gate_record.get("status") != "PASS"
        or gate_record.get("protocol_sha256") != SPEC.fingerprint()
        or gate_record.get("code_sha256") != code_fingerprint()
    ):
        raise RuntimeError("confirmatory summary requires a current PASS gate")
    rows = []
    for stage in ("200m", "300m"):
        rows.extend(
            _effect_rows(
                work_dir,
                split="test",
                stage=stage,
                seeds=SEEDS,
            )
        )
    aggregates: list[dict[str, Any]] = []
    for stage in ("200m", "300m"):
        for length in SPEC.eval_lengths:
            selected = [
                row
                for row in rows
                if row["stage"] == stage and row["length"] == length
            ]
            aggregates.append(
                {
                    "stage": stage,
                    "length": length,
                    "k8_range_gain": _mean_ci(
                        [row["k8"]["range_gain"] for row in selected]
                    ),
                    "k8_shape_gain": _mean_ci(
                        [row["k8"]["shape_gain"] for row in selected]
                    ),
                    "k32_range_gain": _mean_ci(
                        [row["k32"]["range_gain"] for row in selected]
                    ),
                    "k32_shape_gain": _mean_ci(
                        [row["k32"]["shape_gain"] for row in selected]
                    ),
                    "scarcity_interaction": _mean_ci(
                        [row["scarcity_interaction"] for row in selected]
                    ),
                    "k8_evq_minus_native": _mean_ci(
                        [row["k8"]["evq_minus_native"] for row in selected]
                    ),
                }
            )
    primary = next(
        row
        for row in aggregates
        if row["stage"] == "300m" and row["length"] == 8_192
    )
    in_domain = next(
        row
        for row in aggregates
        if row["stage"] == "300m" and row["length"] == 4_096
    )
    previous = next(
        row
        for row in aggregates
        if row["stage"] == "200m" and row["length"] == 8_192
    )
    criteria = {
        "mean_k8_shape_gain_gt_0p05": (
            primary["k8_shape_gain"]["mean"] > 0.05
        ),
        "all_seeds_k8_shape_gain_positive": all(
            value > 0.0 for value in primary["k8_shape_gain"]["values"]
        ),
        "mean_scarcity_interaction_positive": (
            primary["scarcity_interaction"]["mean"] > 0.0
        ),
        "all_seeds_scarcity_interaction_positive": all(
            value > 0.0
            for value in primary["scarcity_interaction"]["values"]
        ),
        "mean_k8_shape_gain_exceeds_range_gain": (
            primary["k8_shape_gain"]["mean"]
            > primary["k8_range_gain"]["mean"]
        ),
        "mean_k8_in_domain_cost_lte_0p02": (
            in_domain["k8_evq_minus_native"]["mean"] <= 0.02
        ),
        "mean_k8_shape_gain_positive_at_200m_and_300m": (
            previous["k8_shape_gain"]["mean"] > 0.0
            and primary["k8_shape_gain"]["mean"] > 0.0
        ),
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "claim_gate": (
            "SUPPORTS_SCARCITY_CLAIM"
            if all(criteria.values())
            else "DOES_NOT_SUPPORT_FULL_SCARCITY_CLAIM"
        ),
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "criteria": criteria,
        "primary_300m_2x": primary,
        "in_domain_300m": in_domain,
        "aggregates": aggregates,
        "per_seed_effects": rows,
        "statistical_boundary": (
            "Seed-level paired contrasts with t-based 95% intervals at n=3; "
            "windows are repeated measurements, not independent seeds."
        ),
    }
    path = work_dir / "summary_mla_scarcity.json"
    if path.exists():
        raise FileExistsError(path)
    _atomic_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def summarize_yarn(work_dir: Path) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    primary_path = work_dir / "summary_mla_scarcity.json"
    if not primary_path.is_file():
        raise FileNotFoundError(primary_path)
    primary = json.loads(primary_path.read_text())
    if (
        primary.get("status") != "PASS"
        or primary.get("protocol_sha256") != SPEC.fingerprint()
        or primary.get("code_sha256") != code_fingerprint()
    ):
        raise ValueError("YaRN diagnostic requires the current primary summary")
    raw_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for pairs in FREQUENCY_PAIRS:
            results: dict[tuple[str, str], dict[str, Any]] = {}
            for arm in ("native_geo", "evq_cosh"):
                for operator in ("raw", "yarn_full"):
                    path = _eval_path(
                        work_dir,
                        pairs,
                        arm,
                        seed,
                        "test",
                        "300m",
                        operator,
                    )
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    result = json.loads(path.read_text())
                    expected = {
                        "status": "PASS",
                        "frequency_pairs": pairs,
                        "arm": arm,
                        "seed": seed,
                        "stage": "300m",
                        "split": "test",
                        "operator": operator,
                        "protocol_sha256": SPEC.fingerprint(),
                        "evaluation_code_sha256": code_fingerprint(),
                        "training_code_sha256": code_fingerprint(),
                    }
                    for key, value in expected.items():
                        if result.get(key) != value:
                            raise ValueError(
                                f"secondary evaluation mismatch: {path}/{key}"
                            )
                    results[(arm, operator)] = result["summary"]
            for length in SPEC.eval_lengths:
                native_raw = results[("native_geo", "raw")][str(length)][
                    "tail_nll_mean"
                ]
                evq_raw = results[("evq_cosh", "raw")][str(length)][
                    "tail_nll_mean"
                ]
                native_yarn = results[("native_geo", "yarn_full")][str(length)][
                    "tail_nll_mean"
                ]
                evq_yarn = results[("evq_cosh", "yarn_full")][str(length)][
                    "tail_nll_mean"
                ]
                raw_gain = native_raw - evq_raw
                yarn_gain = native_yarn - evq_yarn
                raw_rows.append(
                    {
                        "seed": seed,
                        "frequency_pairs": pairs,
                        "length": length,
                        "native_raw_nll": native_raw,
                        "evq_raw_nll": evq_raw,
                        "native_official_yarn_nll": native_yarn,
                        "evq_yarn_derived_nll": evq_yarn,
                        "evq_gain_raw": raw_gain,
                        "evq_gain_under_respective_yarn_operators": yarn_gain,
                        "diagnostic_increment_over_raw": yarn_gain - raw_gain,
                    }
                )
    aggregates = []
    for pairs in FREQUENCY_PAIRS:
        for length in SPEC.eval_lengths:
            selected = [
                row
                for row in raw_rows
                if row["frequency_pairs"] == pairs
                and row["length"] == length
            ]
            aggregates.append(
                {
                    "frequency_pairs": pairs,
                    "length": length,
                    "evq_gain_raw": _mean_ci(
                        [row["evq_gain_raw"] for row in selected]
                    ),
                    "evq_gain_under_respective_yarn_operators": _mean_ci(
                        [
                            row[
                                "evq_gain_under_respective_yarn_operators"
                            ]
                            for row in selected
                        ]
                    ),
                    "diagnostic_increment_over_raw": _mean_ci(
                        [
                            row["diagnostic_increment_over_raw"]
                            for row in selected
                        ]
                    ),
                }
            )
    result = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "aggregates": aggregates,
        "per_seed_rows": raw_rows,
        "interpretation_boundary": (
            "Native Geo uses official YaRN equations on its active native "
            "endpoint grid. EVQ uses the explicitly labeled virtual-coordinate "
            "YaRN-derived generalization. Their contrast is a deployment "
            "diagnostic, not official-YaRN parity or a pure interaction theorem."
        ),
    }
    path = work_dir / "summary_mla_yarn_secondary.json"
    if path.exists():
        raise FileExistsError(path)
    _atomic_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def audit_disk(work_dir: Path) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    checkpoints = list(work_dir.glob("runs/k*/*/seed*/checkpoint_*.pt"))
    incomplete = list(work_dir.rglob("*.incomplete")) if work_dir.exists() else []
    compile_cache = work_dir / "torchinductor_cache"

    def total_bytes(paths: list[Path]) -> int:
        return sum(path.stat().st_size for path in paths if path.is_file())

    compile_files = (
        [path for path in compile_cache.rglob("*") if path.is_file()]
        if compile_cache.is_dir()
        else []
    )
    result = {
        "status": "PASS",
        "filesystem": disk_status(work_dir),
        "work_dir": str(work_dir),
        "checkpoint_count": len(checkpoints),
        "checkpoint_bytes": total_bytes(checkpoints),
        "incomplete_count": len(incomplete),
        "incomplete_bytes": total_bytes(incomplete),
        "compile_cache_files": len(compile_files),
        "compile_cache_bytes": total_bytes(compile_files),
        "policy": (
            "Delete checkpoints only through cleanup-checkpoints after matching "
            "evaluation receipts pass. Preserve compile cache until the suite "
            "is terminal or the author explicitly chooses cleanup."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _verified_checkpoint(
    run_dir: Path, stage: str
) -> tuple[Path, dict[str, Any], str]:
    checkpoint = run_dir / f"checkpoint_{stage}.pt"
    receipt_path = run_dir / f"checkpoint_{stage}.json"
    if not checkpoint.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"checkpoint or receipt missing for {stage}")
    receipt = json.loads(receipt_path.read_text())
    checkpoint_sha = sha256_file(checkpoint)
    if (
        receipt.get("status") != "PASS"
        or receipt.get("checkpoint_label") != stage
        or receipt.get("checkpoint_sha256") != checkpoint_sha
        or receipt.get("protocol_sha256") != SPEC.fingerprint()
        or receipt.get("code_sha256") != code_fingerprint()
    ):
        raise ValueError(f"checkpoint receipt validation failed for {stage}")
    return checkpoint, receipt, checkpoint_sha


def cleanup_checkpoints(args: argparse.Namespace) -> dict[str, Any]:
    pairs = int(args.frequency_pairs)
    arm = str(args.arm)
    seed = int(args.seed)
    if pairs not in FREQUENCY_PAIRS or arm not in ARMS or seed not in SEEDS:
        raise ValueError("unregistered cleanup condition")
    proof_stages = tuple(args.proof_stages)
    if not proof_stages and not args.drop_unclaimed_100m:
        raise ValueError("cleanup has no registered target")
    run_dir = _run_dir(args.work_dir, pairs, arm, seed)
    suffix = (
        f"{args.proof_split}_{args.operator}"
        if proof_stages
        else "prune100"
    )
    output = run_dir / f"cleanup_{suffix}.json"
    if output.is_file():
        existing = json.loads(output.read_text())
        if existing.get("status") == "PASS":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return existing
        if existing.get("status") != "DELETING":
            raise ValueError(f"existing cleanup receipt is invalid: {output}")
        before = disk_status(args.work_dir)
        for target in existing["checkpoint_targets"]:
            checkpoint = Path(target["path"])
            if checkpoint.is_file():
                if sha256_file(checkpoint) != target["sha256"]:
                    raise ValueError(
                        f"cleanup-resume hash mismatch: {checkpoint}"
                    )
                checkpoint.unlink()
        for target in existing.get("incomplete_targets", []):
            path = Path(target["path"])
            if path.is_file():
                path.unlink()
        after = disk_status(args.work_dir)
        existing.update(
            {
                "status": "PASS",
                "cleanup_resumed": True,
                "deleted_checkpoints": existing.pop(
                    "checkpoint_targets"
                ),
                "deleted_incomplete": existing.pop(
                    "incomplete_targets", []
                ),
                "free_bytes_before_resume": before["free_bytes"],
                "free_bytes_after": after["free_bytes"],
            }
        )
        _atomic_json(output, existing)
        print(json.dumps(existing, indent=2, sort_keys=True))
        return existing
    train_result_path = run_dir / "train_result.json"
    if not train_result_path.is_file():
        raise FileNotFoundError(train_result_path)
    train_result = json.loads(train_result_path.read_text())
    if (
        train_result.get("status") != "PASS"
        or train_result.get("protocol_sha256") != SPEC.fingerprint()
    ):
        raise ValueError("training result is not a valid PASS receipt")
    targets: list[dict[str, Any]] = []
    if args.drop_unclaimed_100m:
        checkpoint, _, checkpoint_sha = _verified_checkpoint(run_dir, "100m")
        targets.append(
            {
                "stage": "100m",
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "bytes": checkpoint.stat().st_size,
                "proof": "trained snapshot not used by registered claim",
            }
        )
    for stage in proof_stages:
        if stage == "100m":
            raise ValueError("100m is not a proof-backed evaluation stage")
        checkpoint, _, checkpoint_sha = _verified_checkpoint(run_dir, stage)
        evaluation_path = _eval_path(
            args.work_dir,
            pairs,
            arm,
            seed,
            args.proof_split,
            stage,
            args.operator,
        )
        if not evaluation_path.is_file():
            raise FileNotFoundError(evaluation_path)
        evaluation = json.loads(evaluation_path.read_text())
        expected = {
            "status": "PASS",
            "frequency_pairs": pairs,
            "arm": arm,
            "seed": seed,
            "stage": stage,
            "split": args.proof_split,
            "operator": args.operator,
            "protocol_sha256": SPEC.fingerprint(),
            "checkpoint_sha256": checkpoint_sha,
            "evaluation_code_sha256": code_fingerprint(),
            "training_code_sha256": code_fingerprint(),
        }
        for key, value in expected.items():
            if evaluation.get(key) != value:
                raise ValueError(
                    f"evaluation proof mismatch for {stage}: {key}"
                )
        targets.append(
            {
                "stage": stage,
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "bytes": checkpoint.stat().st_size,
                "proof": str(evaluation_path),
                "proof_sha256": sha256_file(evaluation_path),
            }
        )
    if len({target["path"] for target in targets}) != len(targets):
        raise RuntimeError("cleanup target list contains duplicates")
    before = disk_status(args.work_dir)
    incomplete_targets = [
        {"path": str(path), "bytes": path.stat().st_size}
        for path in run_dir.glob("*.incomplete")
    ]
    plan = {
        "schema_version": 1,
        "status": "DELETING",
        "frequency_pairs": pairs,
        "arm": arm,
        "seed": seed,
        "protocol_sha256": SPEC.fingerprint(),
        "checkpoint_targets": targets,
        "incomplete_targets": incomplete_targets,
        "bytes_deleted": sum(target["bytes"] for target in targets)
        + sum(item["bytes"] for item in incomplete_targets),
        "free_bytes_before": before["free_bytes"],
        "retained": (
            "all JSON/JSONL receipts, raw evaluation rows, schedule sidecar, "
            "and shared compile cache"
        ),
    }
    _atomic_json(output, plan)
    for target in targets:
        Path(target["path"]).unlink()
    for target in incomplete_targets:
        Path(target["path"]).unlink(missing_ok=True)
    after = disk_status(args.work_dir)
    result = {
        **plan,
        "status": "PASS",
        "deleted_checkpoints": targets,
        "deleted_incomplete": incomplete_targets,
        "free_bytes_after": after["free_bytes"],
    }
    result.pop("checkpoint_targets")
    result.pop("incomplete_targets")
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def cleanup_compile_cache(work_dir: Path) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    summary = work_dir / "summary_mla_scarcity.json"
    gate_path = work_dir / "gate_receipt.json"
    terminal = summary.is_file()
    if gate_path.is_file():
        terminal = terminal or json.loads(gate_path.read_text()).get(
            "status"
        ) == "STOP"
    if not terminal:
        raise RuntimeError(
            "compile cache is retained until confirmatory summary or STOP gate"
        )
    cache = work_dir / "torchinductor_cache"
    files = (
        [path for path in cache.rglob("*") if path.is_file()]
        if cache.is_dir()
        else []
    )
    size = sum(path.stat().st_size for path in files)
    if cache.is_dir():
        shutil.rmtree(cache)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "cache": str(cache),
        "files_deleted": len(files),
        "bytes_deleted": size,
        "terminal_evidence": (
            str(summary) if summary.is_file() else str(gate_path)
        ),
    }
    output = work_dir / "cleanup_compile_cache.json"
    if output.exists():
        raise FileExistsError(output)
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--data-manifest", type=Path, required=True)
    preflight.add_argument("--work-dir", type=Path, required=True)
    preflight.add_argument("--full-hash-check", action="store_true")
    preflight.add_argument("--prefix-hash-check", action="store_true")
    preflight.add_argument(
        "--verify-full-initialization", action="store_true"
    )

    ready = sub.add_parser("validate-ready")
    ready.add_argument("--data-manifest", type=Path, required=True)
    ready.add_argument("--work-dir", type=Path, required=True)
    ready.add_argument("--require-disk", action="store_true")

    probe = sub.add_parser("probe-gpu")
    probe.add_argument("--frequency-pairs", type=int, required=True)
    probe.add_argument("--data-manifest", type=Path, required=True)
    probe.add_argument("--work-dir", type=Path, required=True)
    probe.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    probe.add_argument("--timed-steps", type=int, default=5)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--frequency-pairs", type=int, required=True)
    train_parser.add_argument("--arm", choices=ARMS, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--data-manifest", type=Path, required=True)
    train_parser.add_argument("--work-dir", type=Path, required=True)
    train_parser.add_argument("--num-workers", type=int, default=8)
    train_parser.add_argument("--log-every", type=int, default=25)
    train_parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    train_parser.add_argument("--no-compile", action="store_true")
    train_parser.add_argument("--full-hash-check", action="store_true")

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--frequency-pairs", type=int, required=True)
    eval_parser.add_argument("--arm", choices=ARMS, required=True)
    eval_parser.add_argument("--seed", type=int, required=True)
    eval_parser.add_argument("--stage", choices=CHECKPOINT_LABELS, required=True)
    eval_parser.add_argument(
        "--split", choices=("selection", "test"), required=True
    )
    eval_parser.add_argument(
        "--operator", choices=("raw", "yarn_full"), default="raw"
    )
    eval_parser.add_argument("--data-manifest", type=Path, required=True)
    eval_parser.add_argument("--work-dir", type=Path, required=True)
    eval_parser.add_argument("--full-hash-check", action="store_true")

    gate_parser = sub.add_parser("gate")
    gate_parser.add_argument("--work-dir", type=Path, required=True)

    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--work-dir", type=Path, required=True)

    yarn_summary_parser = sub.add_parser("summarize-yarn")
    yarn_summary_parser.add_argument("--work-dir", type=Path, required=True)

    disk_parser = sub.add_parser("audit-disk")
    disk_parser.add_argument("--work-dir", type=Path, required=True)

    cleanup_parser = sub.add_parser("cleanup-checkpoints")
    cleanup_parser.add_argument("--frequency-pairs", type=int, required=True)
    cleanup_parser.add_argument("--arm", choices=ARMS, required=True)
    cleanup_parser.add_argument("--seed", type=int, required=True)
    cleanup_parser.add_argument("--work-dir", type=Path, required=True)
    cleanup_parser.add_argument(
        "--proof-split", choices=("selection", "test"), default="test"
    )
    cleanup_parser.add_argument(
        "--operator", choices=("raw", "yarn_full"), default="raw"
    )
    cleanup_parser.add_argument(
        "--proof-stages",
        nargs="*",
        choices=("200m", "300m"),
        default=(),
    )
    cleanup_parser.add_argument(
        "--drop-unclaimed-100m", action="store_true"
    )

    cache_parser = sub.add_parser("cleanup-compile-cache")
    cache_parser.add_argument("--work-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "preflight":
        run_preflight(args)
    elif args.command == "validate-ready":
        validate_ready(
            args.data_manifest,
            args.work_dir,
            require_disk=bool(args.require_disk),
        )
    elif args.command == "probe-gpu":
        probe_gpu(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "gate":
        gate(args.work_dir)
    elif args.command == "summarize":
        summarize(args.work_dir)
    elif args.command == "summarize-yarn":
        summarize_yarn(args.work_dir)
    elif args.command == "audit-disk":
        audit_disk(args.work_dir)
    elif args.command == "cleanup-checkpoints":
        cleanup_checkpoints(args)
    elif args.command == "cleanup-compile-cache":
        cleanup_compile_cache(args.work_dir)


if __name__ == "__main__":
    main()
