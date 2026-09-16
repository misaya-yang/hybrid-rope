#!/usr/bin/env python3
"""Test exact-length batch=2 against sequential batch=1 on one frozen pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_pair(rows: list[dict], *, length: int) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        if int(row.get("length_cap", -1)) != length:
            continue
        prompt = row.get("prompt_ids")
        budget = int(row.get("max_new_tokens", 0))
        if not isinstance(prompt, list) or not prompt or budget <= 0:
            continue
        groups.setdefault((len(prompt), budget), []).append(row)
    candidates = [values[:2] for values in groups.values() if len(values) >= 2]
    if not candidates:
        raise ValueError("panel has no exact-length pair with a shared generation budget")
    # Prefer the longest pair because it best represents the expensive endpoint.
    return max(candidates, key=lambda values: len(values[0]["prompt_ids"]))


def token_digest(values: list[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--minimum-free-fraction", type=float, default=0.06)
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.length <= 0 or not 0 <= args.minimum_free_fraction < 1 or args.minimum_speedup <= 0:
        raise ValueError("invalid length, free-memory fraction or speedup threshold")

    import torch
    from transformers import AutoTokenizer
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model
    from experiments.olmo_recovery_20260912.recovery_v2_eval import (
        batched_greedy_tokens, greedy_tokens,
    )
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from scripts.experiments.cross_audit.tables import install_static

    environment = validate_cuda()
    pair = select_pair(read_jsonl(args.panel), length=args.length)
    payload = json.loads(args.table.read_text())
    table = payload.get("table", payload)
    values = np.asarray(table["values_float32"], dtype=np.float32)
    gain = float(table["gain"])
    model, _, _ = load_model(args.model, "Native", training=False)
    install_static(model, values, gain)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = model.generation_config.pad_token_id
    if pad is None:
        pad = min(eos)
    budget = int(pair[0]["max_new_tokens"])
    prompts = [row["prompt_ids"] for row in pair]

    # Short warmup removes first-use kernel setup without consuming the frozen
    # pair's full generation budget or creating a scientific output.
    warm = torch.tensor([prompts[0][-64:]], dtype=torch.long, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        greedy_tokens(model, warm, max_new_tokens=1, eos_ids=eos, pad_token_id=pad)
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    total_bytes = int(torch.cuda.mem_get_info()[1])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        started = time.perf_counter()
        references = []
        for prompt in prompts:
            ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
            references.append(greedy_tokens(
                model, ids, max_new_tokens=budget, eos_ids=eos, pad_token_id=pad,
            ))
        torch.cuda.synchronize()
        sequential_seconds = time.perf_counter() - started
        sequential_peak = int(torch.cuda.max_memory_reserved())
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            candidate = batched_greedy_tokens(
                model, prompts, max_new_tokens=budget, eos_ids=eos,
                pad_token_id=pad, left_pad=False,
            )
            torch.cuda.synchronize()
            batch_seconds = time.perf_counter() - started
            batch_peak = int(torch.cuda.max_memory_reserved())
            status = "ok"
            error = None
        except (torch.OutOfMemoryError, RuntimeError) as failure:
            candidate = []
            batch_seconds = None
            batch_peak = int(torch.cuda.max_memory_reserved())
            status = "failed"
            error = str(failure)
    equal = status == "ok" and candidate == references
    speedup = sequential_seconds / batch_seconds if batch_seconds else 0.0
    free_fraction = (total_bytes - batch_peak) / total_bytes
    stable = bool(
        equal and speedup >= args.minimum_speedup
        and free_fraction >= args.minimum_free_fraction
    )
    report = {
        "status": "GENERATION_BATCH_BENCHMARK_COMPLETE_V1",
        "environment": environment,
        "model": str(args.model.resolve()),
        "table": str(args.table.resolve()),
        "length": args.length,
        "row_ids": [row.get("row_id") for row in pair],
        "prompt_tokens": len(prompts[0]),
        "max_new_tokens": budget,
        "sequential_batch1": {
            "seconds": sequential_seconds,
            "peak_reserved_bytes": sequential_peak,
            "generated_ids_sha256": [token_digest(values) for values in references],
        },
        "candidate_batch2": {
            "status": status, "error": error, "seconds": batch_seconds,
            "peak_reserved_bytes": batch_peak,
            "generated_ids_sha256": [token_digest(values) for values in candidate],
            "exact_ids_equal_batch1": equal,
            "speedup": speedup, "free_fraction_at_peak": free_fraction,
            "stable": stable,
        },
        "minimum_speedup": args.minimum_speedup,
        "minimum_free_fraction": args.minimum_free_fraction,
        "recommended_batch_size": 2 if stable else 1,
        "scope": "runtime engineering canary on one exact-length frozen pair; not model-quality evidence",
    }
    atomic_json(args.out, report)
    print(json.dumps({
        "status": report["status"], "recommended_batch_size": report["recommended_batch_size"],
        "speedup": speedup, "batch2_peak_reserved_bytes": batch_peak,
    }))


if __name__ == "__main__":
    main()
