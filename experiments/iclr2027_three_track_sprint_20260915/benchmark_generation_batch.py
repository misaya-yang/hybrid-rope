#!/usr/bin/env python3
"""Select the fastest exact-length batch against sequential batch=1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_batch_sizes(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or any(item <= 1 for item in values):
        raise ValueError("batch sizes must be unique integers greater than one")
    return values


def select_candidate_groups(
    rows: list[dict], *, length: int, batch_sizes: list[int],
) -> dict[int, dict[str, list[dict]]]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        if int(row.get("length_cap", -1)) != length:
            continue
        prompt = row.get("prompt_ids")
        budget = int(row.get("max_new_tokens", 0))
        if not isinstance(prompt, list) or not prompt or budget <= 0:
            continue
        groups.setdefault((len(prompt), budget), []).append(row)
    selected = {}
    for batch_size in batch_sizes:
        full = [values for values in groups.values() if len(values) >= batch_size]
        if not full:
            continue
        throughput = max(full, key=lambda values: len(values[0]["prompt_ids"]))[:batch_size]
        stress_source = max(
            (values for values in groups.values() if len(values) >= 2),
            key=lambda values: min(batch_size, len(values)) * len(values[0]["prompt_ids"]),
        )
        effective = min(batch_size, len(stress_source))
        selected[batch_size] = {
            "throughput": throughput,
            "stress": stress_source[:effective],
        }
    if not selected:
        raise ValueError("panel has no exact-length batch group with a shared generation budget")
    return selected


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
    parser.add_argument("--batch-sizes", default="2,4",
                        help="comma-separated exact-length batch candidates")
    parser.add_argument("--minimum-free-fraction", type=float, default=0.06)
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    requested_batch_sizes = parse_batch_sizes(args.batch_sizes)
    if args.length <= 0 or not 0 <= args.minimum_free_fraction < 1 or args.minimum_speedup <= 0:
        raise ValueError("invalid length, free-memory fraction or speedup threshold")

    try:
        groups = select_candidate_groups(
            read_jsonl(args.panel), length=args.length, batch_sizes=requested_batch_sizes,
        )
    except ValueError as error:
        report = {
            "status": "GENERATION_BATCH_BENCHMARK_COMPLETE_V2",
            "model": str(args.model.resolve()), "table": str(args.table.resolve()),
            "length": args.length, "row_ids": [], "candidates": {},
            "minimum_speedup": args.minimum_speedup,
            "minimum_free_fraction": args.minimum_free_fraction,
            "recommended_batch_size": 1,
            "not_applicable_reason": str(error),
            "scope": "no exact-length group; safe batch-1 fallback; not model-quality evidence",
        }
        atomic_json(args.out, report)
        print(json.dumps({"status": report["status"], "recommended_batch_size": 1,
                          "not_applicable_reason": str(error)}))
        return

    import torch
    from transformers import AutoTokenizer
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import load_model
    from experiments.olmo_recovery_20260912.recovery_v2_eval import (
        batched_greedy_tokens, greedy_tokens,
    )
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from scripts.experiments.cross_audit.tables import install_static

    environment = validate_cuda()
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
    longest_group = max(
        (group for value in groups.values() for group in value.values()),
        key=lambda values: len(values[0]["prompt_ids"]),
    )

    # Short warmup removes first-use kernel setup without consuming the frozen
    # pair's full generation budget or creating a scientific output.
    warm = torch.tensor([longest_group[0]["prompt_ids"][-64:]], dtype=torch.long, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        greedy_tokens(model, warm, max_new_tokens=1, eos_ids=eos, pad_token_id=pad)
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    total_bytes = int(torch.cuda.mem_get_info()[1])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

        def measure(group):
            prompts = [row["prompt_ids"] for row in group]
            budget = int(group[0]["max_new_tokens"])
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            references = []
            reference_seconds = []
            for prompt in prompts:
                ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
                started = time.perf_counter()
                references.append(greedy_tokens(
                    model, ids, max_new_tokens=budget, eos_ids=eos, pad_token_id=pad,
                ))
                torch.cuda.synchronize()
                reference_seconds.append(time.perf_counter() - started)
            sequential_peak = int(torch.cuda.max_memory_reserved())
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                generated = batched_greedy_tokens(
                    model, prompts, max_new_tokens=budget, eos_ids=eos,
                    pad_token_id=pad, left_pad=False,
                )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                peak = int(torch.cuda.max_memory_reserved())
                status = "ok"
                error = None
            except (torch.OutOfMemoryError, RuntimeError) as failure:
                generated = []
                seconds = None
                peak = int(torch.cuda.max_memory_reserved())
                status = "failed"
                error = str(failure)
            equal = status == "ok" and generated == references
            free_fraction = (total_bytes - peak) / total_bytes
            sequential_seconds = sum(reference_seconds)
            return {
                "status": status, "error": error, "seconds": seconds,
                "effective_batch_size": len(group),
                "row_ids": [row.get("row_id") for row in group],
                "prompt_tokens": len(prompts[0]), "max_new_tokens": budget,
                "batch1_reference": {
                    "per_row_seconds": reference_seconds,
                    "peak_reserved_bytes": sequential_peak,
                    "generated_ids_sha256": [token_digest(values) for values in references],
                },
                "rows_per_second": len(group) / seconds if seconds else 0.0,
                "peak_reserved_bytes": peak,
                "generated_ids_sha256": [token_digest(values) for values in generated],
                "exact_ids_equal_batch1": equal,
                "speedup": sequential_seconds / seconds if seconds else 0.0,
                "free_fraction_at_peak": free_fraction,
            }

        candidates = {}
        for batch_size in requested_batch_sizes:
            selected = groups.get(batch_size)
            if selected is None:
                candidates[str(batch_size)] = {
                    "status": "not_applicable", "stable": False,
                    "reason": "no exact-length group reaches this batch size",
                }
                continue
            throughput = measure(selected["throughput"])
            same_shape = [row.get("row_id") for row in selected["throughput"]] == [
                row.get("row_id") for row in selected["stress"]
            ]
            stress = throughput if same_shape else measure(selected["stress"])
            stable = bool(
                throughput["exact_ids_equal_batch1"]
                and throughput["speedup"] >= args.minimum_speedup
                and throughput["free_fraction_at_peak"] >= args.minimum_free_fraction
                and stress["exact_ids_equal_batch1"]
                and stress["free_fraction_at_peak"] >= args.minimum_free_fraction
            )
            candidates[str(batch_size)] = {
                "status": "ok" if throughput["status"] == stress["status"] == "ok" else "failed",
                "throughput": throughput, "worst_shape_stress": stress,
                "rows_per_second": throughput["rows_per_second"], "stable": stable,
            }
    stable_candidates = [
        (int(batch_size), value) for batch_size, value in candidates.items() if value["stable"]
    ]
    recommended = max(
        stable_candidates, key=lambda item: item[1]["rows_per_second"], default=(1, {}),
    )[0]
    report = {
        "status": "GENERATION_BATCH_BENCHMARK_COMPLETE_V2",
        "environment": environment,
        "model": str(args.model.resolve()),
        "table": str(args.table.resolve()),
        "length": args.length,
        "candidates": candidates,
        "minimum_speedup": args.minimum_speedup,
        "minimum_free_fraction": args.minimum_free_fraction,
        "recommended_batch_size": recommended,
        "scope": "runtime engineering canary with longest full-batch throughput and worst token-volume shape stress per candidate; not model-quality evidence",
    }
    atomic_json(args.out, report)
    print(json.dumps({
        "status": report["status"], "recommended_batch_size": recommended,
        "candidates": {key: {
            "speedup": value.get("throughput", {}).get("speedup"),
            "stable": value["stable"],
            "throughput_peak_reserved_bytes": value.get("throughput", {}).get("peak_reserved_bytes"),
            "stress_peak_reserved_bytes": value.get("worst_shape_stress", {}).get("peak_reserved_bytes"),
        }
                       for key, value in candidates.items()},
    }))


if __name__ == "__main__":
    main()
