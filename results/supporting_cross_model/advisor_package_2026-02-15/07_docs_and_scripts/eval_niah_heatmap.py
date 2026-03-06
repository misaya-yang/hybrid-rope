#!/usr/bin/env python3
"""
Needle-In-A-Haystack (NIAH) evaluation for Llama-3-8B + Hybrid-LoRA.

Highlights:
- Loads base model and optionally a LoRA adapter.
- Uses torch.bfloat16 explicitly.
- Tries FlashAttention2 first, then falls back to SDPA.
- Evaluates a depth x context-length matrix and saves a green heatmap PDF.

Default matrix:
- Context lengths: [2048, 4096, 8192, 16384, 32768, 65536]
- Depths: [0, 10, 20, ..., 100]

Outputs:
- results/niah_results.json
- results/niah_heatmap.pdf
- results/niah_heatmap.png
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # isort:skip

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    sns = None

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional dependency at runtime
    PeftModel = None


LOG = logging.getLogger("eval_niah")


def parse_int_list(csv_values: str) -> List[int]:
    return [int(x.strip()) for x in csv_values.split(",") if x.strip()]


def enforce_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    LOG.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NIAH depth-length matrix and save heatmap."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
        help="Base model path (HF local dir or model id).",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="/root/autodl-tmp/dfrope/hybrid-rope/results/llama3_hybrid_lora/final_lora",
        help="LoRA adapter path. Ignored if --base_only is set.",
    )
    parser.add_argument(
        "--base_only",
        action="store_true",
        help="Evaluate base model only (do not load LoRA).",
    )
    parser.add_argument(
        "--no_merge_lora",
        action="store_true",
        help="Do not merge LoRA weights after loading.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for JSON and heatmap outputs.",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="2048,4096,8192,16384,32768,65536",
        help="Comma-separated context lengths.",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="0,10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated depth percentages.",
    )
    parser.add_argument(
        "--trials_per_cell",
        type=int,
        default=1,
        help="How many trials to run for each (length, depth) cell.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Max generated tokens for passkey extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention implementation. 'auto' tries flash_attention_2 then sdpa.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="device_map for model loading.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    return parser


def enable_perf_flags() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def pick_attn_candidates(mode: str) -> List[Optional[str]]:
    if mode == "auto":
        return ["flash_attention_2", "sdpa", None]
    return [mode]


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[torch.nn.Module, AutoTokenizer, str]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.model_max_length = 10_000_000

    model = None
    chosen_attn = "default"
    load_errors: List[str] = []
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )

    for attn_impl in pick_attn_candidates(args.attn_implementation):
        try:
            if attn_impl is None:
                LOG.info("Trying model load with default attention backend.")
                model = AutoModelForCausalLM.from_pretrained(
                    args.base_model_path,
                    **model_kwargs,
                )
                chosen_attn = "default"
            else:
                LOG.info("Trying model load with attn_implementation=%s", attn_impl)
                model = AutoModelForCausalLM.from_pretrained(
                    args.base_model_path,
                    attn_implementation=attn_impl,
                    **model_kwargs,
                )
                chosen_attn = attn_impl
            break
        except Exception as exc:
            msg = f"attn={attn_impl}: {type(exc).__name__}: {exc}"
            load_errors.append(msg)
            LOG.warning("Model load failed: %s", msg)

    if model is None:
        raise RuntimeError(
            "Failed to load model with all attention backends.\n" + "\n".join(load_errors)
        )

    if not args.base_only:
        if PeftModel is None:
            raise RuntimeError("peft is not installed but --base_only is not set.")
        adapter_path = Path(args.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
        has_weight = (adapter_path / "adapter_model.safetensors").exists() or (
            adapter_path / "adapter_model.bin"
        ).exists()
        if not has_weight:
            raise FileNotFoundError(
                f"Adapter directory exists but no LoRA weights found in {adapter_path}. "
                "Expected adapter_model.safetensors or adapter_model.bin."
            )
        LOG.info("Loading LoRA adapter from: %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        if not args.no_merge_lora and hasattr(model, "merge_and_unload"):
            LOG.info("Merging LoRA weights into base model for faster inference.")
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer, chosen_attn


def get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def make_filler_token_pool(tokenizer: AutoTokenizer, min_tokens: int) -> List[int]:
    paragraph = (
        "In the central archive, analysts documented climate, policy, economics, medicine, "
        "history, astronomy, linguistics, and engineering details for future retrieval. "
        "Each section was written in clear prose, with neutral wording and consistent style. "
        "The report discussed timelines, hypotheses, assumptions, constraints, and outcomes. "
        "A review committee periodically inspected the record and appended correction notes. "
    )
    unit_tokens = tokenizer.encode(paragraph, add_special_tokens=False)
    if not unit_tokens:
        raise RuntimeError("Tokenizer produced empty filler token sequence.")

    pool: List[int] = []
    while len(pool) < min_tokens:
        pool.extend(unit_tokens)
    return pool


def build_prompt_ids(
    tokenizer: AutoTokenizer,
    filler_pool: List[int],
    context_len: int,
    depth_percent: int,
    passkey: str,
) -> torch.Tensor:
    prefix_text = (
        "You will read a long context. The passkey appears exactly once.\n"
        "Return only the passkey digits in your final answer.\n\n"
        "[CONTEXT_START]\n"
    )
    suffix_text = (
        "\n[CONTEXT_END]\n"
        "Question: What is the passkey?\n"
        "Answer with digits only.\n"
        "Answer:"
    )
    needle_text = f" Important: the passkey is {passkey}. Remember this exact value. "

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)

    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    non_filler_tokens = len(bos) + len(prefix_ids) + len(suffix_ids) + len(needle_ids)
    filler_budget = context_len - non_filler_tokens
    if filler_budget <= 32:
        raise ValueError(
            f"context_len={context_len} is too small for prompt overhead ({non_filler_tokens})."
        )
    if len(filler_pool) < filler_budget:
        raise ValueError("Filler pool is shorter than filler budget.")

    body_filler = filler_pool[:filler_budget]
    insert_at = int(round((depth_percent / 100.0) * filler_budget))
    insert_at = max(0, min(insert_at, filler_budget))

    context_ids = body_filler[:insert_at] + needle_ids + body_filler[insert_at:]
    input_ids = bos + prefix_ids + context_ids + suffix_ids
    if len(input_ids) != context_len:
        # Keep exact-length control to make matrix comparable.
        if len(input_ids) > context_len:
            input_ids = input_ids[:context_len]
        else:
            pad_id = tokenizer.pad_token_id
            input_ids = input_ids + [pad_id] * (context_len - len(input_ids))

    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)


def extract_passkey(text: str) -> Optional[str]:
    match = re.search(r"\d{4,12}", text)
    return match.group(0) if match else None


def run_one_trial(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    filler_pool: List[int],
    context_len: int,
    depth_percent: int,
    trial_idx: int,
    rng: random.Random,
    max_new_tokens: int,
) -> Dict[str, object]:
    passkey = str(rng.randint(10000, 999999))
    input_ids = build_prompt_ids(
        tokenizer=tokenizer,
        filler_pool=filler_pool,
        context_len=context_len,
        depth_percent=depth_percent,
        passkey=passkey,
    )
    device = get_model_device(model)
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    t0 = time.time()
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return {
                "status": "oom",
                "error": str(exc),
                "trial": trial_idx,
            }
        return {
            "status": "error",
            "error": str(exc),
            "trial": trial_idx,
        }

    gen_ids = output_ids[0, input_ids.shape[1] :]
    generation = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    pred = extract_passkey(generation)
    correct = pred == passkey
    elapsed = time.time() - t0

    return {
        "status": "ok",
        "trial": trial_idx,
        "passkey": passkey,
        "prediction": pred,
        "correct": bool(correct),
        "generated_text": generation,
        "elapsed_sec": round(elapsed, 3),
    }


def run_matrix(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    lengths: List[int],
    depths: List[int],
    trials_per_cell: int,
    max_new_tokens: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rng = random.Random(seed)
    max_length = max(lengths)
    filler_pool = make_filler_token_pool(tokenizer, min_tokens=max_length + 4096)

    matrix = np.full((len(depths), len(lengths)), np.nan, dtype=np.float32)
    raw: Dict[str, object] = {"cells": {}}

    for col, context_len in enumerate(lengths):
        length_key = str(context_len)
        raw["cells"][length_key] = {}
        LOG.info("==== Context length: %s ====", context_len)
        length_failed_oom = False

        for row, depth in enumerate(depths):
            depth_key = str(depth)
            raw["cells"][length_key][depth_key] = {"trials": []}

            if length_failed_oom:
                raw["cells"][length_key][depth_key]["status"] = "skipped_due_to_oom"
                continue

            correct = 0
            total = 0
            cell_status = "ok"
            LOG.info("Running cell length=%s, depth=%s%%", context_len, depth)

            for trial_idx in range(trials_per_cell):
                rec = run_one_trial(
                    model=model,
                    tokenizer=tokenizer,
                    filler_pool=filler_pool,
                    context_len=context_len,
                    depth_percent=depth,
                    trial_idx=trial_idx,
                    rng=rng,
                    max_new_tokens=max_new_tokens,
                )
                raw["cells"][length_key][depth_key]["trials"].append(rec)

                if rec["status"] == "oom":
                    cell_status = "oom"
                    length_failed_oom = True
                    LOG.warning(
                        "OOM at length=%s depth=%s%% trial=%s; skipping remaining depths at this length.",
                        context_len,
                        depth,
                        trial_idx,
                    )
                    break
                if rec["status"] != "ok":
                    cell_status = "error"
                    LOG.error(
                        "Error at length=%s depth=%s%% trial=%s: %s",
                        context_len,
                        depth,
                        trial_idx,
                        rec.get("error"),
                    )
                    break

                total += 1
                correct += int(rec["correct"])

            raw["cells"][length_key][depth_key]["status"] = cell_status
            raw["cells"][length_key][depth_key]["correct"] = correct
            raw["cells"][length_key][depth_key]["total"] = total

            if total > 0:
                acc = correct / total
                matrix[row, col] = acc
                raw["cells"][length_key][depth_key]["accuracy"] = acc
                LOG.info(
                    "Cell done length=%s depth=%s%% | acc=%.3f (%d/%d)",
                    context_len,
                    depth,
                    acc,
                    correct,
                    total,
                )
            else:
                raw["cells"][length_key][depth_key]["accuracy"] = None

            # keep GPU memory stable in long runs
            torch.cuda.empty_cache()
            gc.collect()

    df = pd.DataFrame(
        matrix,
        index=[f"{d}%" for d in depths],
        columns=[str(l) for l in lengths],
    )
    return df, raw


def save_heatmap(df: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    plt.figure(figsize=(max(10, len(df.columns) * 1.45), max(7, len(df.index) * 0.55)))
    if sns is not None:
        cmap = sns.light_palette("#16a34a", as_cmap=True)
        sns.heatmap(
            df,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Passkey Accuracy"},
        )
    else:
        ax = plt.gca()
        img = ax.imshow(df.values, cmap="Greens", vmin=0.0, vmax=1.0, aspect="auto")
        plt.colorbar(img, ax=ax, label="Passkey Accuracy")
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.index)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                text = "nan" if pd.isna(val) else f"{val:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)
        LOG.warning("seaborn is unavailable; used matplotlib fallback for heatmap.")
    plt.title("Needle In A Haystack (Llama-3-8B + Hybrid-LoRA)", fontsize=14)
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Needle Depth in Context")
    plt.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, dpi=300)
    plt.savefig(output_png, dpi=300)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    enforce_offline_mode()
    output_dir = Path(args.output_dir)
    setup_logging(output_dir / "niah_eval.log")
    enable_perf_flags()

    lengths = parse_int_list(args.lengths)
    depths = parse_int_list(args.depths)

    # Sort and deduplicate for stable plotting.
    lengths = sorted(set(lengths))
    depths = sorted(set(depths))

    if lengths[0] <= 0:
        raise ValueError("All lengths must be positive.")
    if any(d < 0 or d > 100 for d in depths):
        raise ValueError("Depth values must be within [0, 100].")
    if args.trials_per_cell <= 0:
        raise ValueError("--trials_per_cell must be > 0.")

    LOG.info("Starting NIAH evaluation.")
    LOG.info("Lengths: %s", lengths)
    LOG.info("Depths: %s", depths)
    LOG.info("Trials per cell: %d", args.trials_per_cell)
    LOG.info("Using torch dtype: bfloat16")

    model, tokenizer, chosen_attn = load_model_and_tokenizer(args)
    LOG.info("Model loaded. Chosen attention backend: %s", chosen_attn)
    LOG.info(
        "Model max_position_embeddings=%s",
        getattr(getattr(model, "config", None), "max_position_embeddings", "unknown"),
    )

    df, raw = run_matrix(
        model=model,
        tokenizer=tokenizer,
        lengths=lengths,
        depths=depths,
        trials_per_cell=args.trials_per_cell,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "niah_results.json"
    output_pdf = output_dir / "niah_heatmap.pdf"
    output_png = output_dir / "niah_heatmap.png"

    payload = {
        "meta": {
            "base_model_path": args.base_model_path,
            "adapter_path": None if args.base_only else args.adapter_path,
            "attn_implementation_requested": args.attn_implementation,
            "attn_implementation_used": chosen_attn,
            "dtype": "bfloat16",
            "lengths": lengths,
            "depths": depths,
            "trials_per_cell": args.trials_per_cell,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "accuracy_matrix": df.to_dict(orient="index"),
        "raw": raw,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    save_heatmap(df, output_pdf=output_pdf, output_png=output_png)

    LOG.info("Saved JSON: %s", output_json)
    LOG.info("Saved heatmap PDF: %s", output_pdf)
    LOG.info("Saved heatmap PNG: %s", output_png)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
