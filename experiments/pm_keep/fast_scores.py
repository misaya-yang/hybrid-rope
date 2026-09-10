"""Execution-only FP32 GQA scorer with batched, full-prefix softmax.

No model loading or experiment runner patching. The CLI defaults to a small
CPU operator check; large CUDA shapes require explicit command arguments.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from .ops import _integer, _validate_qk, attention_mean, select_fixed_budget


def execution_plan(query_shape, key_shape, query_batch_size=256):
    """Conservative live-tensor estimate, excluding input and backend workspace."""
    if len(query_shape) != 3 or len(key_shape) != 3:
        raise ValueError("shapes must be [Hq,M,D] and [Hkv,T,D]")
    hq, count, dim = [_integer(x, "query dimension", 1) for x in query_shape]
    hkv, length, kd = [_integer(x, "key dimension", 1) for x in key_shape]
    if hq % hkv or kd != dim:
        raise ValueError("native contiguous GQA groups and matching head dimensions required")
    requested = _integer(query_batch_size, "query_batch_size", 1)
    per_group = (hq // hkv) * count
    batch = min(requested, per_group)
    tile = hkv * batch * length * 4
    q_bytes, k_bytes = hq * count * dim * 4, hkv * length * dim * 4
    score_bytes = hkv * length * 4
    return {
        "query_shape": list(query_shape), "key_shape": list(key_shape),
        "queries_per_kv_head": per_group,
        "query_batch_size_requested": requested, "query_batch_size_effective": batch,
        "query_batches": math.ceil(per_group / batch),
        "logits_tile_bytes": tile, "probability_tile_bytes": tile,
        "fp32_query_staging_bytes": q_bytes, "fp32_key_staging_bytes": k_bytes,
        "score_and_reduction_bytes": 2 * score_bytes,
        "temporary_tensor_bytes_upper_estimate": 2 * tile + 2 * q_bytes + 2 * k_bytes + 2 * score_bytes,
        "estimate_scope": "conservative tensor allocation estimate; excludes inputs, allocator reserve, and GEMM/softmax backend workspace",
        "key_heads_repeated": False, "protected_keys_in_denominator": True,
    }


@torch.inference_mode()
def attention_mean_batched(queries, keys, *, attention_scale=None, query_batch_size=256):
    """Same mathematical mean as ops.attention_mean; reduction order may differ.

    Every query sees every prefix key, including protected entries. The input
    Q heads are grouped contiguously exactly as native repeat_kv would group
    them, but K itself is never repeated. CUDA callers must disable TF32 so
    'FP32' does not silently mean truncated-mantissa matrix multiplication.
    """
    hq, count, dim, hkv, length = _validate_qk(queries, keys)
    plan = execution_plan(queries.shape, keys.shape, query_batch_size)
    scale = dim**-0.5 if attention_scale is None else float(attention_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("attention_scale must be finite and positive")
    if queries.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("strict FP32 scorer requires torch.backends.cuda.matmul.allow_tf32=False")
    with torch.autocast(device_type=keys.device.type, enabled=False):
        grouped = queries.float().contiguous().reshape(hkv, (hq // hkv) * count, dim)
        key_transpose = keys.float().contiguous().transpose(1, 2)
        result = torch.zeros((hkv, length), device=queries.device, dtype=torch.float32)
        batch = plan["query_batch_size_effective"]
        for start in range(0, grouped.shape[1], batch):
            logits = torch.bmm(grouped[:, start:start + batch], key_transpose)
            logits.mul_(scale)
            probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
            result.add_(probabilities.sum(dim=1))
            # Avoid keeping the previous tile alive while allocating the next.
            del probabilities, logits
        result.div_(grouped.shape[1])
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite FP32 batched scorer output")
    return result


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--shape", choices=("small", "qwen3b_8k", "qwen3b_11589"), default="small")
    p.add_argument("--query-batch-size", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.repeats < 1 or a.warmup < 0:
        p.error("repeats must be positive and warmup nonnegative")
    if a.output and a.output.exists():
        raise FileExistsError(a.output)
    shapes = {"small": ((4, 17, 32), (2, 127, 32)),
              "qwen3b_8k": ((16, 256, 128), (2, 8192, 128)),
              "qwen3b_11589": ((16, 256, 128), (2, 11589, 128))}
    qshape, kshape = shapes[a.shape]
    report = {"scope": "random-tensor operator timing and numerical agreement only; no model or generation quality evidence",
              "device": a.device, "shape_name": a.shape, "seed": a.seed,
              "plan": execution_plan(qshape, kshape, a.query_batch_size),
              "fp32_matmul": "IEEE; TF32 disabled for both implementations",
              "dry_run": a.dry_run}
    if not a.dry_run:
        if a.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing a silent CPU fallback")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device(a.device)
        generator = torch.Generator(device=device).manual_seed(a.seed)
        queries = torch.randn(qshape, device=device, generator=generator)
        keys = torch.randn(kshape, device=device, generator=generator)
        outputs, timings = {}, {}
        for name, scorer in (("reference", lambda: attention_mean(queries, keys)),
                             ("batched", lambda: attention_mean_batched(queries, keys, query_batch_size=a.query_batch_size))):
            for _ in range(a.warmup):
                scorer()
            _sync(device)
            baseline_bytes = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            elapsed = []
            for _ in range(a.repeats):
                _sync(device)
                start = time.perf_counter()
                output = scorer()
                _sync(device)
                elapsed.append(time.perf_counter() - start)
            outputs[name] = output
            timings[name] = {"seconds": elapsed, "median_seconds": sorted(elapsed)[len(elapsed)//2],
                             "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                             "baseline_cuda_allocated_bytes": baseline_bytes}
        reference, batched = outputs["reference"], outputs["batched"]
        delta = (reference - batched).abs()
        budget = kshape[1] // 4
        sinks, recent = (1, 2) if a.shape == "small" else (4, 256)
        reference_keep = select_fixed_budget(reference, budget, sink_tokens=sinks, recent_tokens=recent)
        batched_keep = select_fixed_budget(batched, budget, sink_tokens=sinks, recent_tokens=recent)
        swapped = [len(set(x.tolist()) - set(y.tolist())) for x, y in zip(reference_keep, batched_keep)]
        report.update(timings=timings, max_abs_error=float(delta.max()), mean_abs_error=float(delta.mean()),
                      max_relative_error=float((delta / reference.abs().clamp_min(1e-8)).max()),
                      reference_score_sums=reference.sum(-1).tolist(), batched_score_sums=batched.sum(-1).tolist(),
                      top_budget=budget, changed_keep_entries_per_kv_head=swapped,
                      warning="FP32 summation order differs; near-tie keep-set equality is not universally guaranteed",
                      torch_version=torch.__version__, cuda_device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
