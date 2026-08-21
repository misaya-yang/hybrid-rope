#!/usr/bin/env python3
"""Authorized R3-prime per-layer RoPE probe and training launcher.

This launcher is intentionally narrow.  It imports the registered 151M
FMRoPE runner for the data reader, deterministic sampler, loss, optimizer
schedule, Flash-only CUDA configuration, and runtime checks.  The only model
change is replacing the single shared ``RotaryEmbedding`` with one cloned
module per transformer block and the per-layer inverse-frequency buffers
provided by R0.

The GPU path has two independent gates.  A caller must pass ``--authorize``
and set ``R3_HETERO_ALLOW_GPU_TRAINING=1``.  Missing either gate is a hard
stop before CUDA validation, data loading, model construction, or a probe.
The first GPU operation for ``train`` is always the one-step micro-batch
probe.  The probe is discarded and records finite loss, peak memory, and
throughput for candidates around micro-batch 128 while preserving global
batch 256.

No per-head mode is provided here.  The existing attention contract remains
the explicit per-head feasibility gate in :mod:`protocol`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.native_rope_evq_150m.model import GPT
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m import (
    run_experiment as canonical_runner,
)
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.protocol import (
    SPEC,
    ExperimentSpec,
    estimate_parameter_count,
    learning_rate_for_step,
)

from .protocol import (
    HeterogeneousRopePlan,
    build_layer_inv_freqs,
    hash_tensor_raw,
    install_layerwise_rope,
    load_r0_json,
    model_parameter_contract,
    per_head_feasibility_gate,
    realized_frequency_receipt,
)


AUTH_ENV = "R3_HETERO_ALLOW_GPU_TRAINING"
GLOBAL_BATCH_SIZE = 256
START_MICRO_BATCH = 128
_STATE_ROPE_RE = re.compile(
    r"^blocks\.(?P<layer>\d+)\.(?:attention|attn)\.rope\.inv_freq$"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def authorize_gpu_or_fail(args: argparse.Namespace | None = None) -> None:
    """Enforce the explicit CLI plus environment GPU-training double gate."""

    cli_authorized = bool(getattr(args, "authorize", False))
    env_authorized = os.environ.get(AUTH_ENV) == "1"
    if not cli_authorized or not env_authorized:
        missing: list[str] = []
        if not cli_authorized:
            missing.append("--authorize")
        if not env_authorized:
            missing.append(f"{AUTH_ENV}=1")
        raise PermissionError(
            "GPU probe/training is refused; provide both "
            + " and ".join(missing)
        )


def _validate_151m_plan(plan: HeterogeneousRopePlan) -> None:
    """Check that R0 targets the registered canonical 151M MHA contract."""

    checks = {
        "attention_type": (plan.attention_type, "mha"),
        "num_layers": (plan.num_layers, SPEC.num_layers),
        "rope_dim": (plan.rope_dim, SPEC.head_dim),
        "head_dim": (plan.head_dim, SPEC.head_dim),
        "num_heads": (plan.num_heads, SPEC.num_heads),
        "train_length": (plan.train_length, SPEC.train_length),
    }
    failures = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if failures:
        raise ValueError(
            "R3-prime launcher only supports the canonical 151M MHA "
            "contract: " + ", ".join(failures)
        )
    if plan.d_rope is not None and plan.d_rope != SPEC.head_dim:
        raise ValueError("canonical 151M MHA does not accept an R0 d_rope override")
    if plan.d_nope is not None or plan.n_kv_heads is not None:
        raise ValueError("canonical 151M MHA does not accept MLA/GQA fields")


def _canonical_fingerprint() -> str:
    """Hash the imported canonical path and this launcher for receipts."""

    paths = (
        Path(canonical_runner.__file__).resolve(),
        Path(canonical_runner.__file__).resolve().parent / "protocol.py",
        Path(canonical_runner.__file__).resolve().parent / "prepare.py",
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "protocol.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _build_model(
    plan: HeterogeneousRopePlan,
    *,
    seed: int = SPEC.seed,
    spec: ExperimentSpec = SPEC,
) -> tuple[GPT, tuple[torch.Tensor, ...], dict[str, Any]]:
    """Construct canonical GPT then install only layer-wise RoPE buffers."""

    _validate_151m_plan(plan)
    canonical_runner.seed_everything(seed)
    layer_inv_freqs = build_layer_inv_freqs(plan, dtype=torch.float32)
    model = GPT(spec.model_config(), layer_inv_freqs[0])
    before = model_parameter_contract(model)
    initial_hash = canonical_runner.trainable_state_sha256(model)
    install = install_layerwise_rope(model, layer_inv_freqs)
    after = model_parameter_contract(model)
    if before["parameter_count"] != after["parameter_count"]:
        raise AssertionError("per-layer RoPE installation changed parameter count")
    if before["parameter_tensor_count"] != after["parameter_tensor_count"]:
        raise AssertionError("per-layer RoPE installation changed parameter tensors")
    if before["parameter_schema_sha256"] != after["parameter_schema_sha256"]:
        raise AssertionError("per-layer RoPE installation changed parameter shapes")
    if before["parameter_count"] != estimate_parameter_count(spec):
        raise AssertionError(
            f"parameter count {before['parameter_count']} != "
            f"canonical {estimate_parameter_count(spec)}"
        )
    return model, layer_inv_freqs, {
        "parameter_contract_before": before,
        "parameter_contract_after": after,
        "initial_trainable_sha256": initial_hash,
        "layerwise_installation": install,
        "realized_frequency": realized_frequency_receipt(model),
    }


def microbatch_candidates(
    *,
    global_batch_size: int = GLOBAL_BATCH_SIZE,
    start: int = START_MICRO_BATCH,
) -> tuple[int, ...]:
    """Return divisors of global batch, probing 128, then upward/downward."""

    global_batch_size = int(global_batch_size)
    start = int(start)
    if global_batch_size <= 0 or start <= 0:
        raise ValueError("global_batch_size and start must be positive")
    divisors = [
        value
        for value in range(1, global_batch_size + 1)
        if global_batch_size % value == 0
    ]
    if start not in divisors:
        raise ValueError("start must divide global_batch_size")
    # The first two probes answer the practical question directly: can 128
    # fit, and can the next larger 256 fit?  The remaining probes descend in
    # powers of two so every candidate preserves global batch exactly.
    ordered = [start]
    ordered.extend(value for value in divisors if value > start)
    ordered.extend(value for value in reversed(divisors) if value < start)
    return tuple(dict.fromkeys(ordered))


def _load_manifest(path: Path, *, full_hash_check: bool) -> dict[str, Any]:
    return canonical_runner._load_manifest(path, full_hash_check=full_hash_check)


def _manifest_train_path(manifest: Mapping[str, Any]) -> Path:
    train = manifest.get("train")
    if not isinstance(train, Mapping) or not train.get("path"):
        raise ValueError("canonical data manifest has no train.path")
    path = Path(str(train["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _probe_batch(path: Path, micro_batch: int) -> torch.Tensor:
    source = np.load(path, mmap_mode="r", allow_pickle=False).reshape(-1)
    token_count = int(micro_batch) * SPEC.train_length
    if len(source) < token_count:
        raise ValueError(
            f"training source has {len(source)} tokens; need {token_count}"
        )
    return torch.from_numpy(
        np.array(source[:token_count], dtype=np.int64, copy=True).reshape(
            int(micro_batch), SPEC.train_length
        )
    ).to("cuda", non_blocking=True)


def _release_cuda(*objects: Any) -> None:
    for value in objects:
        del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_one_probe(
    plan: HeterogeneousRopePlan,
    train_path: Path,
    *,
    micro_batch: int,
    compile_mode: str,
    no_compile: bool,
) -> dict[str, Any]:
    """Run exactly one discarded forward/backward/optimizer step."""

    batch = _probe_batch(train_path, micro_batch)
    model, layer_inv_freqs, build_receipt = _build_model(plan)
    model = model.to("cuda")
    loss_module: nn.Module = canonical_runner.CausalLanguageModelLoss(model)
    if not no_compile:
        loss_module = torch.compile(
            loss_module,
            mode=str(compile_mode),
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
    lr = learning_rate_for_step(0, spec=SPEC)
    for group in optimizer.param_groups:
        group["lr"] = lr
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = loss_module(batch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite first-step loss: {loss.item()}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f"non-finite first-step grad norm: {grad_norm.item()}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = int(micro_batch) * SPEC.train_length
    report = {
        "status": "PASS",
        "discarded_probe": True,
        "one_step": True,
        "micro_batch_size": int(micro_batch),
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "grad_accum_steps": GLOBAL_BATCH_SIZE // int(micro_batch),
        "first_loss": float(loss.detach().cpu()),
        "first_loss_finite": bool(torch.isfinite(loss).item()),
        "first_grad_norm": float(grad_norm.detach().cpu()),
        "first_grad_norm_finite": bool(torch.isfinite(grad_norm).item()),
        "step_seconds": elapsed,
        "tokens": tokens,
        "tokens_per_second": tokens / elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "learning_rate": float(lr),
        "layer_inv_freq_sha256_raw": [hash_tensor_raw(value) for value in layer_inv_freqs],
        "parameter_count": build_receipt["parameter_contract_after"]["parameter_count"],
    }
    _release_cuda(batch, optimizer, loss_module, model)
    return report


def run_gpu_probe(
    plan: HeterogeneousRopePlan,
    *,
    manifest_path: Path,
    work_dir: Path,
    compile_mode: str = "default",
    no_compile: bool = False,
    full_hash_check: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Probe candidates and choose the largest one that preserves batch 256."""

    runtime = canonical_runner.validate_cuda_runtime()
    manifest_path = manifest_path.resolve()
    manifest = _load_manifest(manifest_path, full_hash_check=full_hash_check)
    train_path = _manifest_train_path(manifest)
    candidates = microbatch_candidates()
    candidate_rows: list[dict[str, Any]] = []
    for micro_batch in candidates:
        try:
            candidate_rows.append(
                _run_one_probe(
                    plan,
                    train_path,
                    micro_batch=micro_batch,
                    compile_mode=compile_mode,
                    no_compile=no_compile,
                )
            )
        except RuntimeError as exc:
            message = str(exc)
            is_oom = "out of memory" in message.lower()
            if not is_oom:
                raise
            candidate_rows.append(
                {
                    "status": "OOM",
                    "discarded_probe": True,
                    "one_step": False,
                    "micro_batch_size": int(micro_batch),
                    "global_batch_size": GLOBAL_BATCH_SIZE,
                    "grad_accum_steps": GLOBAL_BATCH_SIZE // int(micro_batch),
                    "error": message,
                }
            )
            _release_cuda()
        except FloatingPointError as exc:
            candidate_rows.append(
                {
                    "status": "NONFINITE",
                    "discarded_probe": True,
                    "one_step": False,
                    "micro_batch_size": int(micro_batch),
                    "global_batch_size": GLOBAL_BATCH_SIZE,
                    "grad_accum_steps": GLOBAL_BATCH_SIZE // int(micro_batch),
                    "error": str(exc),
                }
            )
            _release_cuda()
    passing = [
        row
        for row in candidate_rows
        if row.get("status") == "PASS"
        and row.get("first_loss_finite") is True
        and row.get("first_grad_norm_finite") is True
    ]
    if not passing:
        raise RuntimeError("no micro-batch candidate passed the one-step probe")
    selected = max(passing, key=lambda row: int(row["micro_batch_size"]))
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "mode": "AUTHORIZED_ONE_STEP_GPU_PROBE",
        "training_started": False,
        "discarded_probe": True,
        "candidate_order": list(candidates),
        "candidates": candidate_rows,
        "selected_micro_batch_size": int(selected["micro_batch_size"]),
        "selected_grad_accum_steps": GLOBAL_BATCH_SIZE
        // int(selected["micro_batch_size"]),
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "global_batch_preserved": all(
            GLOBAL_BATCH_SIZE % int(row["micro_batch_size"]) == 0
            and int(row["micro_batch_size"]) * int(row["grad_accum_steps"])
            == GLOBAL_BATCH_SIZE
            for row in candidate_rows
        ),
        "r0_plan": plan.as_receipt(),
        "r0_json_sha256": None,
        "data_manifest_sha256": _sha256_file(manifest_path),
        "train_source_sha256": str(manifest["train"]["used_prefix_sha256"]),
        "protocol_sha256": SPEC.fingerprint(),
        "canonical_code_sha256": _canonical_fingerprint(),
        "runtime": runtime,
        "compile": {
            "enabled": not no_compile,
            "mode": None if no_compile else compile_mode,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output = work_dir.resolve() / "gpu_probe_heterogeneous.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite GPU probe: {output}")
    _write_json(output, receipt)
    receipt["output"] = str(output)
    return receipt


def _state_layer_inv_freqs(
    state: Mapping[str, torch.Tensor], *, label: str
) -> tuple[torch.Tensor, ...]:
    values: dict[int, torch.Tensor] = {}
    for name, value in state.items():
        match = _STATE_ROPE_RE.match(name)
        if match:
            values[int(match.group("layer"))] = value.detach().cpu().float().contiguous()
    if not values:
        raise ValueError(f"{label} has no per-layer inv_freq buffers")
    expected = list(range(len(values)))
    if sorted(values) != expected:
        raise ValueError(f"{label} layer buffers are not contiguous: {sorted(values)}")
    return tuple(values[index].clone() for index in expected)


def _save_checkpoint(
    path: Path,
    model: GPT,
    metadata: Mapping[str, Any],
    expected_hashes: Sequence[str],
) -> str:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    realized = _state_layer_inv_freqs(state, label="checkpoint")
    hashes = [hash_tensor_raw(value) for value in realized]
    if list(hashes) != list(expected_hashes):
        raise ValueError("per-layer inv_freq changed before checkpoint save")
    temporary = path.with_name(path.name + ".incomplete")
    torch.save({"model": state, "metadata": dict(metadata)}, temporary)
    temporary.replace(path)
    return _sha256_file(path)


def _load_probe_receipt(
    path: Path,
    *,
    plan: HeterogeneousRopePlan,
    manifest_path: Path,
) -> dict[str, Any]:
    receipt = json.loads(path.resolve().read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ValueError("GPU probe receipt is not PASS")
    if receipt.get("protocol_sha256") != SPEC.fingerprint():
        raise ValueError("GPU probe protocol changed")
    if receipt.get("data_manifest_sha256") != _sha256_file(manifest_path.resolve()):
        raise ValueError("data manifest changed after GPU probe")
    if receipt.get("selected_micro_batch_size") not in microbatch_candidates():
        raise ValueError("GPU probe selected an unregistered micro-batch")
    expected = plan.as_receipt()
    if receipt.get("r0_plan") != expected:
        raise ValueError("R0 plan changed after GPU probe")
    return receipt


def train(
    plan: HeterogeneousRopePlan,
    *,
    manifest_path: Path,
    work_dir: Path,
    probe_receipt: Mapping[str, Any],
    compile_mode: str = "default",
    no_compile: bool = False,
    num_workers: int = 8,
    log_every: int = 25,
    full_hash_check: bool = False,
) -> dict[str, Any]:
    """Run the registered 500M-token training protocol with layer-wise tables."""

    runtime = canonical_runner.validate_cuda_runtime()
    manifest_path = manifest_path.resolve()
    manifest = _load_manifest(manifest_path, full_hash_check=full_hash_check)
    output = work_dir.resolve() / "runs" / "heterogeneous_r3prime"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    micro_batch = int(probe_receipt["selected_micro_batch_size"])
    grad_accum_steps = GLOBAL_BATCH_SIZE // micro_batch
    if micro_batch * grad_accum_steps != GLOBAL_BATCH_SIZE:
        raise AssertionError("selected micro-batch does not preserve global batch")
    model, layer_inv_freqs, build_receipt = _build_model(plan)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_hash = build_receipt["initial_trainable_sha256"]
    expected_hashes = [hash_tensor_raw(value) for value in layer_inv_freqs]
    np.save(output / "layer_inv_freq.npy", np.stack([value.numpy() for value in layer_inv_freqs]))

    order = canonical_runner.deterministic_row_order(SPEC.train_rows, seed=SPEC.seed)
    order_hash = canonical_runner.tensor_sha256(order)
    dataset = canonical_runner.FlatPrefixDataset(
        manifest["train"]["path"],
        rows=SPEC.train_rows,
        seq_len=SPEC.train_length,
    )
    workers = max(0, int(num_workers))
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": micro_batch,
        "sampler": canonical_runner.TensorOrderSampler(order),
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
        "drop_last": True,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(**loader_kwargs)
    expected_micro_steps = SPEC.optimizer_steps * grad_accum_steps
    if len(loader) != expected_micro_steps:
        raise RuntimeError(
            f"loader has {len(loader)} micro-steps, expected {expected_micro_steps}"
        )

    model = model.to("cuda")
    loss_module: nn.Module = canonical_runner.CausalLanguageModelLoss(model)
    if not no_compile:
        loss_module = torch.compile(
            loss_module,
            mode=str(compile_mode),
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
        "arm": "heterogeneous_r3prime",
        "mode": "AUTHORIZED_TRAINING",
        "model_tier": SPEC.model_tier,
        "model_config": SPEC.model_config(),
        "protocol_sha256": SPEC.fingerprint(),
        "canonical_code_sha256": _canonical_fingerprint(),
        "launcher_code_sha256": _sha256_file(Path(__file__).resolve()),
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": _sha256_file(manifest_path),
        "train_used_prefix_sha256": manifest["train"]["used_prefix_sha256"],
        "validation_sha256": manifest["validation"]["sha256"],
        "anchors_sha256": manifest["anchors"]["sha256"],
        "seed": SPEC.seed,
        "parameter_count": parameter_count,
        "parameter_contract": build_receipt["parameter_contract_after"],
        "initial_trainable_sha256": initial_hash,
        "row_order_sha256": order_hash,
        "train_rows": SPEC.train_rows,
        "train_tokens": SPEC.train_tokens,
        "prediction_tokens": SPEC.prediction_tokens,
        "optimizer_steps": SPEC.optimizer_steps,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "micro_batch_size": micro_batch,
        "grad_accum_steps": grad_accum_steps,
        "layer_inv_freq_sha256_raw": expected_hashes,
        "layer_inv_freq_dtype": str(layer_inv_freqs[0].numpy().dtype),
        "layerwise_installation": build_receipt["layerwise_installation"],
        "r0_plan": plan.as_receipt(),
        "probe_receipt": str(probe_receipt.get("output", "")),
        "probe_selected_micro_batch_size": int(probe_receipt["selected_micro_batch_size"]),
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
            "enabled": not no_compile,
            "mode": None if no_compile else compile_mode,
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
        if tuple(batch.shape) != (micro_batch, SPEC.train_length):
            raise RuntimeError(f"unexpected training batch shape: {batch.shape}")
        if micro_step % grad_accum_steps == 0:
            lr = learning_rate_for_step(optimizer_step, spec=SPEC)
            for group in optimizer.param_groups:
                group["lr"] = lr
        batch = batch.to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_module(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at micro-step {micro_step}: {loss.item()}")
        (loss / grad_accum_steps).backward()
        accumulated_loss += float(loss.detach().cpu())
        tokens_since_log += micro_batch * SPEC.train_length
        if (micro_step + 1) % grad_accum_steps:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite grad norm at step {optimizer_step}: {grad_norm.item()}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed = optimizer_step + 1
        mean_loss = accumulated_loss / grad_accum_steps
        accumulated_loss = 0.0
        if optimizer_step == 0 or completed % int(log_every) == 0 or completed == SPEC.optimizer_steps:
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
                "memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
            _append_jsonl(log_path, record)
            print(
                f"[heterogeneous_r3prime] {completed}/{SPEC.optimizer_steps} "
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
        raise RuntimeError(f"completed {optimizer_step} optimizer steps, expected {SPEC.optimizer_steps}")
    torch.cuda.synchronize()
    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metadata["train_seconds"] = time.time() - started
    checkpoint_path = output / "model.pt"
    checkpoint_sha = _save_checkpoint(checkpoint_path, model, metadata, expected_hashes)
    metadata["checkpoint_sha256"] = checkpoint_sha
    _write_json(output / "train_meta.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def _plan_receipt(plan: HeterogeneousRopePlan) -> dict[str, Any]:
    _validate_151m_plan(plan)
    inv = build_layer_inv_freqs(plan)
    return {
        "status": "CPU_PLAN_ONLY",
        "training_started": False,
        "training_authorized": False,
        "model_tier": SPEC.model_tier,
        "protocol_sha256": SPEC.fingerprint(),
        "plan": plan.as_receipt(),
        "layer_inv_freq_sha256_raw": [hash_tensor_raw(value) for value in inv],
        "parameter_count_expected": estimate_parameter_count(SPEC),
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "microbatch_candidates": list(microbatch_candidates()),
        "per_head_feasibility_gate": {
            "status": "GATED_NOT_IMPLEMENTED",
            "training_allowed": False,
        },
        "next_step": "authorized probe requires --authorize and " + f"{AUTH_ENV}=1",
    }


def _require_path(args: argparse.Namespace, name: str) -> Path:
    value = getattr(args, name)
    if value is None:
        raise ValueError(f"--{name.replace('_', '-')} is required for {args.mode}")
    return Path(value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "probe", "train"))
    parser.add_argument("--r0-json", required=True, type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--probe-receipt", type=Path)
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--compile-mode", default="default", choices=(
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    ))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--full-hash-check", action="store_true")
    parser.add_argument("--overwrite-probe", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    plan = load_r0_json(args.r0_json.resolve())
    if args.mode == "plan":
        receipt = _plan_receipt(plan)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return receipt

    # The authorization check is intentionally before manifest/work-dir
    # access and before the first CUDA API call.
    authorize_gpu_or_fail(args)
    manifest_path = _require_path(args, "data_manifest")
    work_dir = _require_path(args, "work_dir")
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "probe":
        receipt = run_gpu_probe(
            plan,
            manifest_path=manifest_path,
            work_dir=work_dir,
            compile_mode=args.compile_mode,
            no_compile=args.no_compile,
            full_hash_check=args.full_hash_check,
            overwrite=args.overwrite_probe,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return receipt

    if args.probe_receipt is None:
        probe = run_gpu_probe(
            plan,
            manifest_path=manifest_path,
            work_dir=work_dir,
            compile_mode=args.compile_mode,
            no_compile=args.no_compile,
            full_hash_check=args.full_hash_check,
            overwrite=args.overwrite_probe,
        )
    else:
        probe = _load_probe_receipt(
            args.probe_receipt,
            plan=plan,
            manifest_path=manifest_path,
        )
    result = train(
        plan,
        manifest_path=manifest_path,
        work_dir=work_dir,
        probe_receipt=probe,
        compile_mode=args.compile_mode,
        no_compile=args.no_compile,
        num_workers=args.num_workers,
        log_every=args.log_every,
        full_hash_check=args.full_hash_check,
    )
    return result


if __name__ == "__main__":
    main()
