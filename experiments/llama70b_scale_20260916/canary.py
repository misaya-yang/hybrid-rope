#!/usr/bin/env python3
"""Bounded NF4 load and Native/TailSpline execution canary for Llama-3-70B."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def first_row(path: Path, task: str) -> dict:
    with path.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("task") == task:
                    return row
    raise ValueError(f"panel lacks {task}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--lm-array", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer
    from experiments.fixed_rope_three_interfaces_20260913.tables import find_table
    from experiments.olmo_recovery_20260912.recovery_v2_eval import greedy_tokens, lm_loss_rows
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from scripts.experiments.cross_audit.tables import install_static, verify_static

    environment = validate_cuda()
    config = json.loads((args.model / "config.json").read_text())
    quantization = config.get("quantization_config") or {}
    if not (
        quantization.get("load_in_4bit") is True
        and quantization.get("bnb_4bit_quant_type") == "nf4"
        and quantization.get("bnb_4bit_compute_dtype") == "bfloat16"
        and quantization.get("bnb_4bit_use_double_quant") is True
    ):
        raise RuntimeError("checkpoint is not the frozen BF16-compute NF4 double-quant model")

    started = time.perf_counter()
    model, wrapper, _ = load_model(args.model, "Native", checkpoint=None, training=False)
    if wrapper is not None or model.training:
        raise RuntimeError("canary unexpectedly created a training wrapper")
    linear4bit = sum(module.__class__.__name__ == "Linear4bit" for module in model.modules())
    if linear4bit <= 0 or next(model.parameters()).device.type != "cuda":
        raise RuntimeError("bitsandbytes Linear4bit CUDA materialization failed")
    device_map = getattr(model, "hf_device_map", {}) or {}
    if any(str(device).startswith(("cpu", "disk")) for device in device_map.values()):
        raise RuntimeError(f"checkpoint offload is forbidden: {device_map}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    eos_value = model.generation_config.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    eos.discard(None)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = model.generation_config.pad_token_id
    if pad is None:
        pad = min(eos)
    lm = np.load(args.lm_array, mmap_mode="r", allow_pickle=False)
    if lm.ndim != 2 or lm.shape[0] < 1 or lm.shape[1] < 32769:
        raise ValueError("LM canary source is shorter than 32K")

    native = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for length in (1024, 8192):
            ids = torch.tensor(lm[0, :length + 1].copy(), dtype=torch.long, device="cuda").unsqueeze(0)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); step = time.perf_counter()
            losses = lm_loss_rows(model, ids, prefill_chunk_size=0)
            torch.cuda.synchronize()
            nll = losses["whole_loss_sum"] / losses["whole_target_count"]
            if not math.isfinite(nll):
                raise RuntimeError(f"nonfinite Native NLL at {length}")
            native[str(length)] = {
                "nll": nll, "seconds": time.perf_counter() - step,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }

    table = find_table(json.loads(args.table.read_text()))
    values = np.asarray(table["values_float32"], dtype=np.float32)
    gain = float(table["gain"])
    install_static(model, values, gain); verify_static(model, values, gain)
    row = first_row(args.panel, "niah_single_1")
    prompt = torch.tensor([row["prompt_ids"]], dtype=torch.long, device="cuda")

    def generation(chunk: int) -> dict:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); step = time.perf_counter()
        tokens = greedy_tokens(
            model, prompt, max_new_tokens=min(4, int(row["max_new_tokens"])),
            eos_ids=eos, pad_token_id=int(pad), prefill_chunk_size=chunk,
        )
        torch.cuda.synchronize()
        return {"chunk": chunk, "generated_ids": tokens, "seconds": time.perf_counter() - step,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}

    def ppl(chunk: int) -> dict:
        ids = torch.tensor(lm[0, :32769].copy(), dtype=torch.long, device="cuda").unsqueeze(0)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); step = time.perf_counter()
        losses = lm_loss_rows(model, ids, prefill_chunk_size=chunk); torch.cuda.synchronize()
        nll = losses["whole_loss_sum"] / losses["whole_target_count"]
        if not math.isfinite(nll):
            raise RuntimeError("nonfinite TailSpline 32K NLL")
        return {"chunk": chunk, "nll": nll, "seconds": time.perf_counter() - step,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}

    def first_working(function):
        failures = []
        for chunk in (0, 8192):
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    return function(chunk), failures
            except (torch.OutOfMemoryError, RuntimeError) as error:
                failures.append({"chunk": chunk, "error": str(error)})
                torch.cuda.empty_cache()
        raise RuntimeError(f"no trustworthy 32K path: {failures}")

    generated, generation_failures = first_working(generation)
    scored, ppl_failures = first_working(ppl)
    report = {
        "status": "LLAMA3_70B_NF4_CANARY_COMPLETE_V1",
        "environment": environment, "model": str(args.model),
        "load_seconds": time.perf_counter() - started,
        "quantization_config": quantization, "linear4bit_modules": linear4bit,
        "hf_device_map": device_map, "native": native,
        "tailspline_32k_generation": generated, "tailspline_32k_ppl": scored,
        "generation_failures": generation_failures, "ppl_failures": ppl_failures,
        "selected_generation_prefill_chunk": generated["chunk"],
        "selected_lm_prefill_chunk": scored["chunk"],
        "scope": "engineering compatibility canary only; not model-quality evidence",
    }
    atomic_json(args.out, report)
    print(json.dumps({"status": report["status"], "generation_chunk": generated["chunk"],
                      "lm_chunk": scored["chunk"]}, sort_keys=True))


if __name__ == "__main__":
    main()
