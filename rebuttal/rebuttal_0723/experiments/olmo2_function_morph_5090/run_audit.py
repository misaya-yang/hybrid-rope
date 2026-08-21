#!/usr/bin/env python3
"""Run the explicitly authorized, inference-only finite morph audit on GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .authorization import require_gpu_authorization
from .protocol import (
    CANDIDATES,
    LENGTHS,
    METHOD_ID,
    MORPH_GRID,
    ROWS_PER_LENGTH,
    TAIL_TOKENS,
    ContractError,
    load_json,
    sha256_file,
    validate_model_config,
)


BATCH_BY_LENGTH = {4_096: 4, 8_192: 4, 16_384: 2}
PACKAGE_ROOT = Path(__file__).resolve().parent


def _check_bound_file(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise ContractError(f"bound asset is missing: {path}")
    observed = sha256_file(path)
    if observed != record.get("sha256"):
        raise ContractError(f"bound asset hash drift: {path}")
    return path


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _nll_per_example(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as functional

    batch, tokens, vocabulary = logits.shape
    loss = functional.cross_entropy(
        logits.reshape(batch * tokens, vocabulary),
        labels.reshape(batch * tokens),
        reduction="none",
    )
    return loss.view(batch, tokens).mean(dim=1)


def _teacher_kl_per_example(teacher_logits: Any, candidate_logits: Any) -> Any:
    import torch

    teacher_log_probability = torch.log_softmax(teacher_logits, dim=-1)
    candidate_log_probability = torch.log_softmax(candidate_logits, dim=-1)
    probability = teacher_log_probability.exp()
    return (
        probability * (teacher_log_probability - candidate_log_probability)
    ).sum(dim=-1).mean(dim=-1)


def _tail_logits(model: Any, token_ids: Any) -> Any:
    output = model(
        input_ids=token_ids,
        use_cache=False,
        logits_to_keep=TAIL_TOKENS + 1,
    )
    logits = output.logits[:, :-1, :].float()
    if logits.shape[1] != TAIL_TOKENS:
        raise ContractError(f"unexpected tail logit shape: {tuple(logits.shape)}")
    return logits


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["candidate"]), float(row["t"]), int(row["length"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (candidate, t, length), values in sorted(grouped.items()):
        output.append(
            {
                "candidate": candidate,
                "t": t,
                "length": length,
                "examples": len(values),
                "native_nll_mean": sum(float(v["native_nll"]) for v in values)
                / len(values),
                "candidate_nll_mean": sum(
                    float(v["candidate_nll"]) for v in values
                )
                / len(values),
                "delta_nll_mean": sum(float(v["delta_nll"]) for v in values)
                / len(values),
                "teacher_kl_mean": sum(float(v["teacher_kl"]) for v in values)
                / len(values),
            }
        )
    return output


def run_gpu_audit(
    *, preflight_path: Path, output_path: Path, cli_authorize: bool
) -> dict[str, Any]:
    # This check intentionally precedes torch, transformers, token, or model imports.
    require_gpu_authorization(
        cli_authorize=cli_authorize,
        environment=os.environ,
    )
    preflight = load_json(preflight_path)
    if preflight.get("method_id") != METHOD_ID:
        raise ContractError("preflight method identity mismatch")
    if preflight.get("status") != "READY_FOR_EXPLICIT_GPU_AUDIT":
        raise ContractError("preflight is not ready for a GPU audit")

    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM

    from .targets import array_sha256, log_morph

    source_hashes = preflight.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ContractError("preflight has no source hash bindings")
    for name, expected in source_hashes.items():
        source_path = PACKAGE_ROOT / str(name)
        if not source_path.is_file() or sha256_file(source_path) != expected:
            raise ContractError(f"experiment source drift: {name}")

    if not torch.cuda.is_available():
        raise ContractError("authorized audit requires a visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise ContractError("authorized audit requires BF16 support")
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    if architecture not in torch.cuda.get_arch_list():
        raise ContractError(
            f"active architecture {architecture} is absent from torch arch list"
        )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    if not torch.backends.cuda.flash_sdp_enabled():
        raise ContractError("Flash SDPA is not enabled")
    if (
        torch.backends.cuda.math_sdp_enabled()
        or torch.backends.cuda.mem_efficient_sdp_enabled()
        or torch.backends.cuda.cudnn_sdp_enabled()
    ):
        raise ContractError("a forbidden SDPA fallback remains enabled")

    assets = preflight["assets"]
    config_path = _check_bound_file(assets["config"])
    _check_bound_file(assets["weights"])
    _check_bound_file(assets["r0_collection"])
    target_path = _check_bound_file(assets["target_manifest"])
    config = load_json(config_path)
    validate_model_config(config)
    target_manifest = load_json(target_path)
    if target_manifest.get("status") != "FINITE_TARGETS_FROZEN":
        raise ContractError("target manifest is not frozen")
    if target_manifest.get("protocol", {}).get("method_id") != METHOD_ID:
        raise ContractError("target manifest method identity mismatch")

    token_views: dict[int, Any] = {}
    for length in LENGTHS:
        record = assets["token_views"][str(length)]
        path = _check_bound_file(record)
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(tensor.shape) != tuple(record["shape"]):
            raise ContractError(f"token tensor shape drift: {path}")
        token_views[length] = tensor[:ROWS_PER_LENGTH].contiguous()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_dir = Path(str(assets["model_dir"]))
    load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ContractError("model did not enter frozen evaluation mode")
    rotary = model.model.rotary_emb
    native_runtime = rotary.inv_freq.detach().float().cpu().numpy()
    expected_native = np.asarray(
        target_manifest["native"]["inv_freq"], dtype=np.float32
    )
    if not np.array_equal(native_runtime.astype(np.float32), expected_native):
        raise ContractError("runtime Native inv_freq differs from frozen R0 table")
    original = rotary.inv_freq.detach().clone()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "method_id": METHOD_ID,
        "status": "RUNNING",
        "preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": sha256_file(preflight_path),
        },
        "target_manifest": {
            "path": str(target_path.resolve()),
            "sha256": sha256_file(target_path),
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers_attention": "sdpa_flash_only",
            "device": torch.cuda.get_device_name(0),
            "architecture": architecture,
            "dtype": "bfloat16",
            "load_seconds": time.monotonic() - load_started,
            "batch_by_length": {str(k): v for k, v in BATCH_BY_LENGTH.items()},
        },
        "rows": [],
        "execution_proof": {
            "training_attempted": False,
            "optimizer_created": False,
            "gradients_enabled": False,
            "backward_called": False,
            "downloads_allowed": False,
            "math_attention_fallback": False,
        },
    }
    _write_progress(output_path, payload)
    rows: list[dict[str, Any]] = payload["rows"]
    started = time.monotonic()
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for length in LENGTHS:
                tokens = token_views[length]
                batch_size = BATCH_BY_LENGTH[length]
                for start in range(0, ROWS_PER_LENGTH, batch_size):
                    stop = min(start + batch_size, ROWS_PER_LENGTH)
                    batch = tokens[start:stop].to("cuda", non_blocking=True)
                    rotary.inv_freq.copy_(original)
                    teacher_logits = _tail_logits(model, batch)
                    labels = batch[:, -TAIL_TOKENS:]
                    native_nll = _nll_per_example(teacher_logits, labels)
                    for candidate in CANDIDATES:
                        for index in range(start, stop):
                            offset = index - start
                            rows.append(
                                {
                                    "candidate": candidate,
                                    "t": 0.0,
                                    "length": length,
                                    "sample_index": index,
                                    "native_nll": float(native_nll[offset].item()),
                                    "candidate_nll": float(native_nll[offset].item()),
                                    "delta_nll": 0.0,
                                    "teacher_kl": 0.0,
                                }
                            )
                    for candidate in CANDIDATES:
                        target = np.asarray(
                            target_manifest["candidates"][candidate]["inv_freq"],
                            dtype=np.float64,
                        )
                        for t in MORPH_GRID[1:]:
                            table = log_morph(
                                expected_native.astype(np.float64), target, t
                            ).astype(np.float32)
                            frozen_row = next(
                                row
                                for row in target_manifest["morph_tables"][candidate]
                                if float(row["t"]) == t
                            )
                            observed_hash = array_sha256(table)
                            if observed_hash != frozen_row["inv_freq_float32_sha256"]:
                                raise ContractError(
                                    f"morph table hash drift for {candidate} t={t}"
                                )
                            rotary.inv_freq.copy_(
                                torch.from_numpy(table).to(
                                    device=rotary.inv_freq.device,
                                    dtype=rotary.inv_freq.dtype,
                                )
                            )
                            candidate_logits = _tail_logits(model, batch)
                            candidate_nll = _nll_per_example(
                                candidate_logits, labels
                            )
                            teacher_kl = _teacher_kl_per_example(
                                teacher_logits, candidate_logits
                            )
                            for index in range(start, stop):
                                offset = index - start
                                nll = float(candidate_nll[offset].item())
                                base_nll = float(native_nll[offset].item())
                                kl = float(teacher_kl[offset].item())
                                if not all(math.isfinite(x) for x in (nll, base_nll, kl)):
                                    raise ContractError("non-finite audit metric")
                                rows.append(
                                    {
                                        "candidate": candidate,
                                        "t": t,
                                        "length": length,
                                        "sample_index": index,
                                        "native_nll": base_nll,
                                        "candidate_nll": nll,
                                        "delta_nll": nll - base_nll,
                                        "teacher_kl": kl,
                                    }
                                )
                            del candidate_logits, candidate_nll, teacher_kl
                    del batch, teacher_logits, native_nll
                    payload["elapsed_seconds"] = time.monotonic() - started
                    _write_progress(output_path, payload)
        payload["aggregates"] = _aggregate(rows)
        payload["status"] = "FINITE_FUNCTION_MORPH_AUDIT_COMPLETE"
        payload["elapsed_seconds"] = time.monotonic() - started
        payload["peak_cuda_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated()
        )
        _write_progress(output_path, payload)
        return payload
    finally:
        rotary.inv_freq.copy_(original)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args(argv)
    result = run_gpu_audit(
        preflight_path=args.preflight,
        output_path=args.output,
        cli_authorize=args.authorize,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": len(result["rows"]),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
