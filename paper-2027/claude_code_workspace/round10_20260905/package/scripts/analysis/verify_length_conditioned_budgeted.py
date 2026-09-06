#!/usr/bin/env python3
"""Verify exact Native short dispatch and long-branch parity on OLMo-2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.rope.length_conditioned_budgeted import (
    install_length_conditioned_rope,
    matched_attention_scaling,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.train import (
    configure_flash_only_attention,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def endpoint(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, -1]
        logits = model.lm_head(hidden)
    return hidden.detach().clone(), logits.detach().clone()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-name", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    configure_flash_only_attention(model)
    model.eval().to("cuda")

    rows = np.load(args.input_ids, mmap_mode="r", allow_pickle=False)
    if rows.ndim != 2 or rows.shape[1] < 4096:
        raise RuntimeError("verification input must contain a physical 4K row")
    short_ids = torch.as_tensor(
        np.asarray(rows[0, :4096], dtype=np.int64), device="cuda"
    ).unsqueeze(0)
    native_parameters = sum(parameter.numel() for parameter in model.parameters())
    native_hidden, native_logits = endpoint(model, short_ids)

    table = np.load(args.table, allow_pickle=False)
    if table.dtype != np.float32 or table.shape != (64,):
        raise RuntimeError("frozen table must be float32 [64]")
    state, method = install_length_conditioned_rope(
        model,
        long_inv_freq=torch.from_numpy(np.ascontiguousarray(table)),
        long_attention_scaling=matched_attention_scaling(float(args.factor)),
        long_name=str(args.long_name),
        reference_length=4096,
        long_context_budget=int(round(4096 * float(args.factor))),
    )
    wrapped_parameters = sum(parameter.numel() for parameter in model.parameters())
    state.force_for_budget(4096)
    short_hidden, short_logits = endpoint(model, short_ids)
    short_hidden_max_abs = float((short_hidden.float() - native_hidden.float()).abs().max())
    short_logits_max_abs = float((short_logits.float() - native_logits.float()).abs().max())
    short_exact = bool(
        torch.equal(short_hidden, native_hidden)
        and torch.equal(short_logits, native_logits)
    )
    if not short_exact or native_parameters != wrapped_parameters:
        raise RuntimeError("Native short branch is not exactly parameter/function preserving")

    probe_ids = short_ids[:, :512]
    state.force_for_budget(int(round(4096 * float(args.factor))))
    wrapped_long_hidden, wrapped_long_logits = endpoint(model, probe_ids)
    wrapper = model.model.rotary_emb
    model.model.rotary_emb = wrapper.long
    direct_long_hidden, direct_long_logits = endpoint(model, probe_ids)
    model.model.rotary_emb = wrapper
    long_hidden_max_abs = float(
        (wrapped_long_hidden.float() - direct_long_hidden.float()).abs().max()
    )
    long_logits_max_abs = float(
        (wrapped_long_logits.float() - direct_long_logits.float()).abs().max()
    )
    long_exact = bool(
        torch.equal(wrapped_long_hidden, direct_long_hidden)
        and torch.equal(wrapped_long_logits, direct_long_logits)
    )
    if not long_exact:
        raise RuntimeError("length-conditioned long branch differs from direct operator")

    payload = {
        "status": "LENGTH_CONDITIONED_BUDGETED_PARITY_COMPLETE",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_weight_sha256": sha256_file(
            args.checkpoint.resolve() / "model.safetensors"
        ),
        "table": str(args.table.resolve()),
        "table_file_sha256": sha256_file(args.table.resolve()),
        "input_ids": str(args.input_ids.resolve()),
        "input_ids_sha256": sha256_file(args.input_ids.resolve()),
        "method": method,
        "parameter_count_before": int(native_parameters),
        "parameter_count_after": int(wrapped_parameters),
        "short_branch": {
            "tokens": 4096,
            "hidden_bitwise_equal": bool(torch.equal(short_hidden, native_hidden)),
            "logits_bitwise_equal": bool(torch.equal(short_logits, native_logits)),
            "hidden_max_abs_delta": short_hidden_max_abs,
            "logits_max_abs_delta": short_logits_max_abs,
        },
        "long_branch": {
            "probe_tokens": 512,
            "hidden_bitwise_equal_to_direct_operator": bool(
                torch.equal(wrapped_long_hidden, direct_long_hidden)
            ),
            "logits_bitwise_equal_to_direct_operator": bool(
                torch.equal(wrapped_long_logits, direct_long_logits)
            ),
            "hidden_max_abs_delta": long_hidden_max_abs,
            "logits_max_abs_delta": long_logits_max_abs,
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
