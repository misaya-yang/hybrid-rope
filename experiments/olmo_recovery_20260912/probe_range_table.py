#!/usr/bin/env python3
"""One-step 16K answer+EOS gradient probe for the frozen OLMo range solver."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np


TASKS = ("niah_single_1", "niah_multikey_1", "qa_2")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--initial-arm", choices=("BM_g4", "BetaSym_gamma3_g4"), default="BM_g4")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = read_rows(args.panel)
    selected = [next(row for row in rows if row["task"] == task and row["length_cap"] == 16384) for task in TASKS]
    plan = {
        "status": "PLAN_ONLY" if not args.execute else "STARTING",
        "initial_arm": args.initial_arm,
        "rows": [row["row_id"] for row in selected],
        "objective": "mean canonical answer plus terminal EOS cross entropy",
        "model_weight_updates": 0,
        "purpose": "gradient, memory, and fixed-table feasibility only; not a candidate result",
    }
    if not args.execute:
        print(json.dumps(plan, sort_keys=True))
        return

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from experiments.olmo_recovery_20260912.native_relative_allocation import install_native_relative_allocation
    from experiments.olmo_recovery_20260912.recovery_v2_runtime import table_for_config
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from scripts.experiments.cross_audit.tables import native_table

    environment = validate_cuda()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    table = table_for_config(config, args.initial_arm)
    base = getattr(config, "rope_theta", None) or getattr(config, "rope_parameters", {}).get("rope_theta")
    native = native_table(config.hidden_size // config.num_attention_heads, base).astype(np.float64)
    values = np.asarray(table["values_float32"], dtype=np.float64)
    exponents = -np.log(values / native) / math.log(4.0)
    exponents[:15] = 0.0
    exponents[32:] = 1.0
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
    )
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()):
        raise ValueError("range probe requires a deterministic zero-dropout checkpoint")
    allocation = install_native_relative_allocation(
        model,
        low=14,
        high=32,
        scale=4.0,
        initial_exponents=exponents,
        initial_gain=float(table["gain"]),
        initial_inv_freq=values.astype(np.float32),
    )
    initial_values = allocation.realized_inv_freq().detach().cpu().numpy()
    parity = float(np.max(np.abs(initial_values - values.astype(np.float32))))
    if parity > 2e-10:
        raise RuntimeError(f"differentiable initial table differs from frozen arm: {parity}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    row_results = []
    for row in selected:
        if len(row["references"]) != 1:
            raise ValueError("probe requires one canonical reference per row")
        target = tokenizer.encode(" " + row["references"][0], add_special_tokens=False)
        target.append(int(tokenizer.eos_token_id))
        ids = torch.tensor([row["prompt_ids"] + target[:-1]], dtype=torch.long, device="cuda")
        labels = torch.tensor(target, dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=len(target), return_dict=True).logits[0].float()
            if logits.shape[0] != len(target):
                raise RuntimeError("logits_to_keep did not return the answer trajectory")
            loss = torch.nn.functional.cross_entropy(logits, labels)
        (loss / len(selected)).backward()
        row_results.append({
            "row_id": row["row_id"],
            "task": row["task"],
            "input_tokens": len(row["prompt_ids"]),
            "answer_tokens_including_eos": len(target),
            "nll": float(loss.detach()),
        })
        del ids, labels, logits, loss
    shape_grad = allocation.increment_logits.grad
    gain_grad = allocation.log_gain.grad
    if shape_grad is None or gain_grad is None or not torch.isfinite(shape_grad).all() or not torch.isfinite(gain_grad):
        raise RuntimeError("range allocation gradient is missing or non-finite")
    if any(parameter.grad is not None for name, parameter in model.named_parameters() if "increment_logits" not in name and "log_gain" not in name):
        raise RuntimeError("a frozen model weight received a gradient")
    receipt = {
        **plan,
        "status": "COMPLETE",
        "environment": environment,
        "table_parity_max_abs": parity,
        "allocation": allocation.receipt(),
        "rows_detail": row_results,
        "mean_nll": sum(row["nll"] for row in row_results) / len(row_results),
        "shape_gradient_norm": float(torch.linalg.vector_norm(shape_grad.detach())),
        "gain_gradient": float(gain_grad.detach()),
        "gradient_sum_translation_null": float(shape_grad.detach().sum()),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.out.mkdir(parents=True)
    (args.out / "probe.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": receipt["status"],
        "mean_nll": receipt["mean_nll"],
        "shape_gradient_norm": receipt["shape_gradient_norm"],
        "gain_gradient": receipt["gain_gradient"],
        "peak_memory_allocated_bytes": receipt["peak_memory_allocated_bytes"],
        "elapsed_seconds": receipt["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
