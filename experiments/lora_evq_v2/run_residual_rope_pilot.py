#!/usr/bin/env python3
"""Matched residual-Native/EVQ retrieval pilot on a frozen Llama-3-8B base."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
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
    load_capability_suite,
    score_capability_prediction,
    score_generation_metrics,
)
from experiments.lora_evq_v2.eval_sparse_conversion import (
    _answer_ids,
    _select_passkey_rows,
    load_passkey_rows,
)
from experiments.lora_evq_v2.gen_retrieval_mix import gen_kv_retrieval, gen_single_needle
from experiments.lora_evq_v2.residual_rope_adapter import (
    alpha_summary,
    attention_modules,
    attach_residual_rope,
    freeze_base,
    load_residual_state_dict,
    residual_state_dict,
    self_test,
    set_residual_active_layers,
    set_residual_enabled,
    set_residual_method,
)


SCHEMA = "evq_cosh.residual_rope_pilot.v1"
ARMS = ("base_native", "residual_native", "residual_evq")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _chat_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]], generation: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=generation
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return [int(token_id) for token_id in encoded]


def build_training_rows(tokenizer: Any, *, count: int, max_length: int, seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    rows = []
    attempts = 0
    while len(rows) < count and attempts < count * 20:
        attempts += 1
        words = random.randint(700, 1200)
        item = gen_single_needle(words) if len(rows) % 2 == 0 else gen_kv_retrieval(8, words)
        messages = item["messages"]
        prompt = _chat_ids(tokenizer, messages[:-1], True)
        full = _chat_ids(tokenizer, messages, False)
        if full[: len(prompt)] != prompt or not 128 <= len(full) <= max_length:
            continue
        rows.append(
            {
                "input_ids": full,
                "labels": [-100] * len(prompt) + full[len(prompt) :],
                "task": "single_needle" if len(rows) % 2 == 0 else "kv_retrieval",
            }
        )
    if len(rows) != count:
        raise RuntimeError(f"built {len(rows)} training rows; expected {count}")
    return rows


def build_passkey_training_rows(
    tokenizer: Any, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected = _select_passkey_rows(rows, lengths=(16384,), trials=(2, 3))
    training = []
    for row in selected:
        answer_ids = _answer_ids(tokenizer, str(row["answers"][0]))
        prompt_ids = [int(token_id) for token_id in row["prompt_ids"]]
        training.append(
            {
                "input_ids": prompt_ids + answer_ids,
                "labels": [-100] * len(prompt_ids) + answer_ids,
                "task": "passkey_16k",
            }
        )
    return training


def _prepare_evaluation(
    passkey_root: Path, qa_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _, passkey_rows = load_passkey_rows(passkey_root)
    passkey = _select_passkey_rows(passkey_rows, lengths=(8192, 16384), trials=(0, 1))
    _, qa_rows = load_capability_suite(qa_root)
    qasper = sorted(
        (
            row
            for row in qa_rows
            if row["task"] == "qasper" and len(row["prompt_ids"]) <= 8192
        ),
        key=lambda row: (len(row["prompt_ids"]), row["example_id"]),
    )[:20]
    if len(passkey) != 20 or len(qasper) != 20:
        raise RuntimeError("evaluation selection differs from the registered 20+20 pilot")
    return passkey, qasper, passkey_rows


@torch.inference_mode()
def evaluate_arm(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    arm: str,
    passkey_rows: Sequence[Mapping[str, Any]],
    qasper_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model.eval()
    model.config.use_cache = True
    results = []
    for index, row in enumerate([*passkey_rows, *qasper_rows], start=1):
        details = generate_answer_details(
            model,
            tokenizer,
            prompt_ids=row["prompt_ids"],
            metric=row["metric"],
            generation_tokens=int(row.get("generation_tokens") or 16),
            device=torch.device("cuda"),
        )
        prediction = str(details["prediction"])
        diagnostics = score_generation_metrics(
            prediction,
            row["answers"],
            eos_terminated=bool(details["eos_terminated"]),
            generated_token_count=int(details["generated_token_count"]),
        )
        score = score_capability_prediction(
            row["metric"], prediction, row["answers"], source=row.get("source")
        )
        result = {
            "example_id": row["example_id"],
            "task": row["task"],
            "target_length": int(row["target_length"]),
            "prompt_tokens": len(row["prompt_ids"]),
            "depth_percent": row.get("depth_percent"),
            "metric_score": float(score),
            "prediction": prediction,
            "references": list(row["answers"]),
            **diagnostics,
        }
        results.append(result)
        print(json.dumps({"arm": arm, "eval": f"{index}/40", "task": row["task"], "score": score}), flush=True)

    passkey_by_length: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if row["task"] == "passkey_retrieval":
            passkey_by_length[int(row["target_length"])].append(row)
    qasper = [row for row in results if row["task"] == "qasper"]
    aggregate = {
        "passkey": {
            str(length): {
                "examples": len(rows),
                "strict_exact": statistics.fmean(float(row["strict_exact"]) for row in rows),
                "first_value_exact": statistics.fmean(float(row["first_value_exact"]) for row in rows),
            }
            for length, rows in sorted(passkey_by_length.items())
        },
        "qasper": {
            "examples": len(qasper),
            "f1": statistics.fmean(float(row["metric_score"]) for row in qasper),
            "strict_exact": statistics.fmean(float(row["strict_exact"]) for row in qasper),
        },
    }
    return {"aggregate": aggregate, "results": results}


def _train_arm(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    projection_lr: float,
    alpha_lr: float,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    with torch.no_grad():
        for module in attention_modules(model):
            if module.residual_rope_branch.enabled:
                module.residual_rope_branch.alpha.fill_(1e-3)
    trainable = freeze_base(model)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    alpha = [parameter for name, parameter in named if name.endswith(".alpha")]
    projections = [parameter for name, parameter in named if not name.endswith(".alpha")]
    if set(trainable) != set(alpha + projections) or not alpha or not projections:
        raise RuntimeError("residual trainable parameter partition is invalid")
    optimizer = torch.optim.AdamW(
        [
            {"params": projections, "lr": projection_lr},
            {"params": alpha, "lr": alpha_lr, "weight_decay": 0.0},
        ],
        weight_decay=0.01,
    )
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    losses = []
    started = time.time()
    for step in range(steps):
        row = rows[step % len(rows)]
        input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
        labels = torch.tensor([row["labels"]], dtype=torch.long, device="cuda")
        output = model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
        target_positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
        if target_positions.numel() == 0 or int(target_positions.min()) <= 0:
            raise RuntimeError("answer-only row has no valid causal targets")
        prediction_hidden = output.last_hidden_state[0, target_positions - 1]
        answer_logits = model.lm_head(prediction_hidden).float()
        loss = torch.nn.functional.cross_entropy(answer_logits, labels[0, target_positions])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite training loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        print(json.dumps({"step": step + 1, "loss": losses[-1], "alpha": alpha_summary(model)}), flush=True)
    model.gradient_checkpointing_disable()
    return {
        "steps": steps,
        "alpha_initial": 1e-3,
        "loss_first": losses[0],
        "loss_final": losses[-1],
        "loss_mean": statistics.fmean(losses),
        "seconds": time.time() - started,
        "alpha": alpha_summary(model),
    }


@torch.inference_mode()
def _zero_gate_parity(model: torch.nn.Module, prompt_ids: Sequence[int]) -> dict[str, float]:
    model.eval()
    ids = torch.tensor([list(prompt_ids)[:64]], dtype=torch.long, device="cuda")
    set_residual_enabled(model, False)
    native = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits.float()
    set_residual_enabled(model, True)
    residual = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits.float()
    difference = (native - residual).abs()
    result = {"max_abs_logit_diff": float(difference.max()), "mean_abs_logit_diff": float(difference.mean())}
    if result["max_abs_logit_diff"] > 0.25 or result["mean_abs_logit_diff"] > 0.01:
        raise RuntimeError(f"zero-gate parity failed: {result}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("pilot requires exactly one CUDA GPU")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    passkey_rows, qasper_rows, all_passkey_rows = _prepare_evaluation(
        args.passkey_root, args.qa_root
    )
    training_rows = (
        build_passkey_training_rows(tokenizer, all_passkey_rows)
        if args.train_mode == "passkey_16k"
        else build_training_rows(
            tokenizer, count=max(args.steps, 24), max_length=args.train_length, seed=args.seed
        )
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad = False
    attach_residual_rope(
        model,
        branch_dim=args.branch_dim,
        method="native_geo",
        tau=args.tau,
        seed=args.seed,
    )
    active_layers = set(
        range(int(model.config.num_hidden_layers) - args.active_last_layers, int(model.config.num_hidden_layers))
    )
    set_residual_active_layers(model, active_layers)
    initial_state = residual_state_dict(model)
    parity = _zero_gate_parity(model, qasper_rows[0]["prompt_ids"])
    print(json.dumps({"zero_gate_parity": parity}), flush=True)

    outputs: dict[str, Any] = {}
    set_residual_enabled(model, False)
    outputs["base_native"] = evaluate_arm(
        model, tokenizer, arm="base_native", passkey_rows=passkey_rows, qasper_rows=qasper_rows
    )
    for arm, method in (("residual_native", "native_geo"), ("residual_evq", "evq_cosh")):
        load_residual_state_dict(model, initial_state)
        set_residual_method(model, method)
        set_residual_active_layers(model, active_layers)
        training = _train_arm(
            model,
            training_rows,
            steps=args.steps,
            projection_lr=args.projection_lr,
            alpha_lr=args.alpha_lr,
            seed=args.seed,
        )
        artifact = {
            "schema": "evq_cosh.residual_rope_adapter.v1",
            "method": method,
            "branch_dim": args.branch_dim,
            "tau": args.tau,
            "seed": args.seed,
            "training": training,
            "state_dict": residual_state_dict(model),
        }
        torch.save(artifact, args.output_dir / f"{arm}.pt")
        outputs[arm] = {
            "training": training,
            **evaluate_arm(
                model, tokenizer, arm=arm, passkey_rows=passkey_rows, qasper_rows=qasper_rows
            ),
        }

    absolute = {arm: outputs[arm]["aggregate"] for arm in ARMS}
    base_qasper = absolute["base_native"]["qasper"]["f1"]
    evq_qasper = absolute["residual_evq"]["qasper"]["f1"]
    native_16k = absolute["residual_native"]["passkey"]["16384"]["first_value_exact"]
    evq_16k = absolute["residual_evq"]["passkey"]["16384"]["first_value_exact"]
    gates = {
        "zero_gate_parity": True,
        "evq_retains_90pct_base_qasper": evq_qasper >= 0.9 * base_qasper,
        "evq_16k_passkey_beats_residual_native": evq_16k > native_16k,
    }
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "single_seed_supporting": True,
        "paper_claim": False,
        "protocol": {
            "base": "Meta-Llama-3-8B-Instruct",
            "base_frozen": True,
            "branch_dim": args.branch_dim,
            "tau": args.tau,
            "steps": args.steps,
            "train_mode": args.train_mode,
            "objective": "answer_only_sparse_lm_head",
            "train_max_length": max(len(row["input_ids"]) for row in training_rows),
            "active_layers": sorted(active_layers),
            "training_rows_sha256": _sha256_json(training_rows),
            "matched_initial_residual_state": True,
            "only_variable": "residual_branch_frequency",
        },
        "zero_gate_parity": parity,
        "absolute": absolute,
        "gates": gates,
        "gate": "positive" if all(gates.values()) else "negative",
        "runtime": {
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    _atomic_json(args.output_dir / "raw.json", {"summary": summary, "arms": outputs})
    _atomic_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--passkey-root", type=Path)
    parser.add_argument("--qa-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--branch-dim", type=int, default=8)
    parser.add_argument("--tau", type=float, default=1.414)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--train-length", type=int, default=2048)
    parser.add_argument("--train-mode", choices=("synthetic_2k", "passkey_16k"), default="synthetic_2k")
    parser.add_argument("--active-last-layers", type=int, default=32)
    parser.add_argument("--projection-lr", type=float, default=1e-4)
    parser.add_argument("--alpha-lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("residual pilot self-test ok")
        return
    required = (args.model, args.passkey_root, args.qa_root, args.output_dir)
    if any(value is None for value in required):
        raise ValueError("model, passkey-root, qa-root, and output-dir are required")
    if not 1 <= args.active_last_layers <= 32:
        raise ValueError("active-last-layers must be in [1, 32]")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = run(args)
    print(json.dumps({"status": summary["status"], "gate": summary["gate"]}, indent=2))


if __name__ == "__main__":
    main()
