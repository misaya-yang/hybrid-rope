#!/usr/bin/env python3
"""Inference-only native-Q/K residual EVQ gate on frozen 16K passkey rows."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lora_evq_v2.eval_official_yarn_capability import (
    generate_answer_details,
    score_answer,
    score_generation_metrics,
)
from experiments.lora_evq_v2.eval_sparse_conversion import (
    _answer_ids,
    _select_passkey_rows,
)
from experiments.lora_evq_v2.residual_rope_adapter import (
    attach_residual_rope,
    attention_modules,
    self_test,
    set_residual_alpha,
    set_residual_enabled,
    set_residual_method,
)
from experiments.lora_evq_v2.run_residual_rope_pilot import _prepare_evaluation


SCHEMA = "evq_cosh.native_qk_residual_gate.v1"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _aggregate_nll(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    tokens = sum(int(record["answer_tokens"]) for record in records)
    return {
        "examples": len(records),
        "answer_tokens": tokens,
        "nll": sum(float(record["nll_sum"]) for record in records) / tokens,
    }


@torch.inference_mode()
def _score_rows(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    records = []
    for index, row in enumerate(rows, start=1):
        answer_ids = _answer_ids(tokenizer, str(row["answers"][0]))
        score = score_answer(
            model.model,
            model.lm_head,
            prompt_ids=row["prompt_ids"],
            answer_ids=answer_ids,
            device=torch.device("cuda"),
        )
        record = {
            "example_id": row["example_id"],
            "depth_percent": float(row["depth_percent"]),
            "nll_sum": float(score["nll_sum"]),
            "answer_tokens": int(score["answer_tokens"]),
            "nll": -float(score["mean_logprob"]),
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "score": f"{index}/{len(rows)}",
                    "depth": record["depth_percent"],
                    "nll": record["nll"],
                }
            ),
            flush=True,
        )
    return records, _aggregate_nll(records)


@torch.inference_mode()
def _generate_rows(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    records = []
    for index, row in enumerate(rows, start=1):
        details = generate_answer_details(
            model,
            tokenizer,
            prompt_ids=row["prompt_ids"],
            metric=row["metric"],
            generation_tokens=16,
            device=torch.device("cuda"),
        )
        metrics = score_generation_metrics(
            str(details["prediction"]),
            row["answers"],
            eos_terminated=bool(details["eos_terminated"]),
            generated_token_count=int(details["generated_token_count"]),
        )
        records.append(
            {
                "example_id": row["example_id"],
                "depth_percent": float(row["depth_percent"]),
                "prediction": str(details["prediction"]),
                "references": list(row["answers"]),
                **metrics,
            }
        )
        print(
            json.dumps(
                {
                    "arm": arm,
                    "generate": f"{index}/{len(rows)}",
                    "first_value_exact": metrics["first_value_exact"],
                }
            ),
            flush=True,
        )
    return records, {
        "examples": len(records),
        "first_value_exact": statistics.fmean(
            float(record["first_value_exact"]) for record in records
        ),
        "strict_exact": statistics.fmean(float(record["strict_exact"]) for record in records),
    }


@torch.inference_mode()
def _short_context_parity(
    model: torch.nn.Module,
    tokenizer: Any,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device="cuda")
    mask = torch.ones_like(ids)
    set_residual_enabled(model, False)
    native_hidden = model.model(
        input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
    ).last_hidden_state[:, -1]
    native_logits = model.lm_head(native_hidden).float()
    set_residual_enabled(model, True)
    residual_hidden = model.model(
        input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
    ).last_hidden_state[:, -1]
    residual_logits = model.lm_head(residual_hidden).float()
    difference = (native_logits - residual_logits).abs()

    set_residual_enabled(model, False)
    native_generation = generate_answer_details(
        model,
        tokenizer,
        prompt_ids=row["prompt_ids"],
        metric=row["metric"],
        generation_tokens=16,
        device=torch.device("cuda"),
    )
    set_residual_enabled(model, True)
    residual_generation = generate_answer_details(
        model,
        tokenizer,
        prompt_ids=row["prompt_ids"],
        metric=row["metric"],
        generation_tokens=16,
        device=torch.device("cuda"),
    )
    result = {
        "prompt_tokens": len(row["prompt_ids"]),
        "max_abs_logit_diff": float(difference.max()),
        "mean_abs_logit_diff": float(difference.mean()),
        "generated_ids_equal": native_generation["generated_ids"]
        == residual_generation["generated_ids"],
    }
    if result != {
        "prompt_tokens": len(row["prompt_ids"]),
        "max_abs_logit_diff": 0.0,
        "mean_abs_logit_diff": 0.0,
        "generated_ids_equal": True,
    }:
        raise RuntimeError(f"short-context bypass parity failed: {result}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("native-Q/K residual gate requires exactly one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    passkey_eval, qasper_rows, all_passkey_rows = _prepare_evaluation(
        args.passkey_root, args.qa_root
    )
    dev_rows = _select_passkey_rows(
        all_passkey_rows, lengths=(16384,), trials=args.dev_trials
    )
    test_rows = _select_passkey_rows(
        all_passkey_rows, lengths=(16384,), trials=args.test_trials
    )
    if {row["example_id"] for row in dev_rows} & {row["example_id"] for row in test_rows}:
        raise RuntimeError("dev and test passkey rows overlap")
    short_rows = [row for row in passkey_eval if int(row["target_length"]) == 8192]
    covered = [*short_rows, *qasper_rows]
    if any(len(row["prompt_ids"]) + 16 > args.query_gate_start for row in covered):
        raise RuntimeError("a registered short-context row crosses the query gate")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
    contract = {
        "hidden_size": int(model.config.hidden_size),
        "query_heads": int(model.config.num_attention_heads),
        "key_value_heads": int(model.config.num_key_value_heads),
        "head_dim": head_dim,
        "rope_theta": float(model.config.rope_theta),
    }
    if contract != {
        "hidden_size": 4096,
        "query_heads": 32,
        "key_value_heads": 8,
        "head_dim": 128,
        "rope_theta": 500000.0,
    }:
        raise RuntimeError(f"model architecture contract mismatch: {contract}")
    attach_residual_rope(
        model,
        branch_dim=head_dim,
        method="evq_cosh",
        tau=args.tau,
        projection_mode="native_qk",
        query_gate_start=args.query_gate_start,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = attention_modules(model)
    expected_scale = 1.0 / math.sqrt(head_dim)
    if any(not math.isclose(float(module.scaling), expected_scale, abs_tol=1e-12) for module in modules):
        raise RuntimeError("attention scaling differs from native 1/sqrt(head_dim)")
    if any(module.residual_rope_branch.inv_freq.dtype != torch.float32 for module in modules):
        raise RuntimeError("residual inverse frequencies are not FP32")
    if any(
        module.residual_rope_branch.q_proj is not None
        or module.residual_rope_branch.k_proj is not None
        for module in modules
    ):
        raise RuntimeError("native-Q/K residual unexpectedly allocated projection weights")

    set_residual_alpha(model, 1.0)
    set_residual_method(model, "evq_cosh")
    parity = _short_context_parity(model, tokenizer, short_rows[0])
    print(json.dumps({"short_context_parity": parity}), flush=True)

    set_residual_enabled(model, False)
    base_dev_records, base_dev = _score_rows(
        model, tokenizer, dev_rows, arm="base_native_dev"
    )
    dev_records: dict[str, Any] = {"base_native": base_dev_records}
    dev_aggregates: dict[str, Any] = {"base_native": base_dev}
    for alpha in args.alphas:
        set_residual_enabled(model, True)
        set_residual_method(model, "evq_cosh")
        set_residual_alpha(model, alpha)
        label = f"evq_alpha_{alpha:g}"
        records, aggregate = _score_rows(model, tokenizer, dev_rows, arm=label)
        dev_records[label] = records
        dev_aggregates[label] = aggregate
    selected_alpha = min(
        args.alphas, key=lambda value: float(dev_aggregates[f"evq_alpha_{value:g}"]["nll"])
    )
    best_dev = dev_aggregates[f"evq_alpha_{selected_alpha:g}"]
    dev_nll_gain = float(base_dev["nll"]) - float(best_dev["nll"])
    dev_gate = dev_nll_gain > args.minimum_dev_nll_gain

    test_records: dict[str, Any] = {}
    test_aggregates: dict[str, Any] = {}
    if dev_gate:
        for arm, enabled, method in (
            ("base_native", False, "evq_cosh"),
            ("residual_native", True, "native_geo"),
            ("residual_evq", True, "evq_cosh"),
        ):
            set_residual_enabled(model, enabled)
            set_residual_method(model, method)
            set_residual_alpha(model, selected_alpha)
            nll_records, nll = _score_rows(model, tokenizer, test_rows, arm=f"{arm}_test")
            generation_records, generation = _generate_rows(
                model, tokenizer, test_rows, arm=f"{arm}_test"
            )
            test_records[arm] = {"nll": nll_records, "generation": generation_records}
            test_aggregates[arm] = {"nll": nll, "generation": generation}

    heldout_nll_gate = bool(
        dev_gate
        and float(test_aggregates["residual_evq"]["nll"]["nll"])
        < float(test_aggregates["residual_native"]["nll"]["nll"])
    )
    heldout_exact_gate = bool(
        dev_gate
        and float(test_aggregates["residual_evq"]["generation"]["first_value_exact"])
        > float(test_aggregates["base_native"]["generation"]["first_value_exact"])
    )
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "single_seed_supporting": True,
        "paper_claim": False,
        "inference_only": True,
        "protocol": {
            "base": "Meta-Llama-3-8B-Instruct",
            "base_frozen": True,
            "projection_mode": "reuse_native_qk",
            "learned_parameters": 0,
            "query_gate_start": args.query_gate_start,
            "alpha_grid": list(args.alphas),
            "selected_alpha": selected_alpha,
            "dev_trials": list(args.dev_trials),
            "test_trials": list(args.test_trials),
            "tau": args.tau,
            "branch_dim": head_dim,
            "kernel_qkv_dim": 2 * head_dim,
            "attention_scaling": expected_scale,
            "inv_freq_dtype": "torch.float32",
            "activation_dtype": "torch.bfloat16",
            "operator": "native score plus query-gated EVQ score",
        },
        "short_context": {
            "covered_frozen_rows": len(covered),
            "parity": parity,
        },
        "dev": {
            "aggregates": dev_aggregates,
            "base_minus_best_evq_nll": dev_nll_gain,
            "minimum_nll_gain": args.minimum_dev_nll_gain,
            "gate": dev_gate,
        },
        "test": test_aggregates,
        "gates": {
            "short_context_exact": True,
            "dev_evq_nll_improves_base": dev_gate,
            "heldout_evq_nll_beats_native_temperature_control": heldout_nll_gate,
            "heldout_evq_first_value_exact_beats_base": heldout_exact_gate,
        },
        "gate": "positive" if heldout_nll_gate and heldout_exact_gate else "negative",
        "runtime": {
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    raw = {
        "summary": summary,
        "dev": {"records": dev_records, "aggregates": dev_aggregates},
        "test": {"records": test_records, "aggregates": test_aggregates},
    }
    _atomic_json(args.output_dir / "raw.json", raw)
    _atomic_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--passkey-root", type=Path)
    parser.add_argument("--qa-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.03, 0.1, 0.3, 1.0))
    parser.add_argument("--tau", type=float, default=1.414)
    parser.add_argument("--query-gate-start", type=int, default=8224)
    parser.add_argument("--dev-trials", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--test-trials", type=int, nargs="+", default=(4, 5))
    parser.add_argument("--minimum-dev-nll-gain", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("native-Q/K residual evaluator self-test ok")
        return
    required = (args.model, args.passkey_root, args.qa_root, args.output_dir)
    if any(value is None for value in required):
        raise ValueError("model, passkey-root, qa-root, and output-dir are required")
    if len(set(args.alphas)) != len(args.alphas) or any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("alphas must be unique and positive")
    if set(args.dev_trials) & set(args.test_trials):
        raise ValueError("dev and test trials must be disjoint")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    summary = run(args)
    print(
        json.dumps(
            {"status": summary["status"], "gate": summary["gate"], "seconds": time.time() - started},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
