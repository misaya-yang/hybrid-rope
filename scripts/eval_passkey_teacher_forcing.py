#!/usr/bin/env python3
"""
Passkey evaluation with teacher-forcing true-vs-false NLL gap.

Why this script:
- Avoids relying only on generation substring matching.
- Uses controlled true/false candidate comparison under identical prompt.
- Supports exact custom inv_freq patching for fair-method adapters.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_niah_recall import enforce_offline_mode, load_model_and_tokenizer
from eval_niah_recall import resolve_inv_metadata


def parse_int_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Passkey teacher-forcing evaluation for long-context adapters.")
    ap.add_argument(
        "--base_model_path",
        type=str,
        default="/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
    )
    ap.add_argument("--adapter_path", type=str, default="")
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument(
        "--variant",
        type=str,
        default="auto",
        choices=["auto", "base", "hybrid", "yarn", "pi", "pi_soft", "custom"],
    )
    ap.add_argument("--custom_inv_freq_path", type=str, default="")
    ap.add_argument("--rope_factor", type=float, default=8.0)
    ap.add_argument("--orig_ctx", type=int, default=8192)
    ap.add_argument("--rope_theta", type=float, default=0.0)
    ap.add_argument("--hybrid_split_ratio", type=float, default=0.5)
    ap.add_argument("--hybrid_alpha", type=float, default=0.2)
    ap.add_argument("--hybrid_p", type=float, default=3.9)
    ap.add_argument("--hybrid_min_freq_scale", type=float, default=4.0)
    ap.add_argument("--attn_implementation", type=str, default="sdpa", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--trust_remote_code", action="store_true", default=True)

    ap.add_argument("--lengths", type=str, default="1024,2048,4096,8192,16384")
    ap.add_argument("--depths", type=str, default="10,50,90")
    ap.add_argument("--trials_per_cell", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--run_generation_probe", action="store_true")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--manifest_json", type=str, default="")
    return ap


def repeat_to_len(tokens: Sequence[int], n: int) -> List[int]:
    if n <= 0:
        return []
    base = list(tokens)
    if not base:
        return [0] * n
    rep = n // len(base) + 1
    return (base * rep)[:n]


def build_passkey_prompt_ids(
    tokenizer,
    seq_len: int,
    depth_pct: int,
    passkey: str,
) -> List[int]:
    """
    Build prompt directly in token space for strict length control.
    """
    prefix_text = (
        "Read the long document below. Memorize the special magic number exactly.\n\n"
        "Document:\n"
    )
    suffix_text = "\n\nQuestion: What is the special magic number?\nAnswer:"
    filler_text = (
        "This section discusses long-context memory, retrieval, and reasoning. "
        "Careful reading is required. "
    )
    needle_text = f"The special magic number is {passkey}. "

    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)

    if len(filler_ids) == 0 or len(needle_ids) == 0:
        raise RuntimeError("Tokenizer returned empty ids for filler/needle.")

    bos_extra = 1 if tokenizer.bos_token_id is not None else 0
    doc_budget = int(seq_len) - bos_extra - len(prefix_ids) - len(suffix_ids)
    if doc_budget <= len(needle_ids) + 16:
        raise RuntimeError(
            f"seq_len={seq_len} too short for passkey prompt. "
            f"Need > {bos_extra + len(prefix_ids) + len(suffix_ids) + len(needle_ids) + 16}"
        )

    filler_budget = doc_budget - len(needle_ids)
    filler_stream = repeat_to_len(filler_ids, filler_budget)
    depth_ratio = max(0.0, min(1.0, depth_pct / 100.0))
    pos = int(round(depth_ratio * len(filler_stream)))
    pos = min(max(pos, 0), len(filler_stream))

    doc_ids = filler_stream[:pos] + needle_ids + filler_stream[pos:]
    prompt_ids = prefix_ids + doc_ids + suffix_ids
    if tokenizer.bos_token_id is not None:
        prompt_ids = [int(tokenizer.bos_token_id)] + prompt_ids

    # Final exact-length guard.
    if len(prompt_ids) > int(seq_len):
        prompt_ids = prompt_ids[-int(seq_len) :]
    elif len(prompt_ids) < int(seq_len):
        prompt_ids = prompt_ids + repeat_to_len(filler_ids, int(seq_len) - len(prompt_ids))
    return prompt_ids


def sample_false_passkey(tokenizer, true_passkey: str, rng: random.Random) -> str:
    true_len = len(tokenizer.encode(true_passkey, add_special_tokens=False))
    fallback = true_passkey
    for _ in range(256):
        cand = f"{rng.randint(10000, 99999)}"
        if cand == true_passkey:
            continue
        fallback = cand
        if len(tokenizer.encode(cand, add_special_tokens=False)) == true_len:
            return cand
    if fallback == true_passkey:
        fallback = f"{(int(true_passkey) + 7) % 100000:05d}"
    return fallback


def answer_nll(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
    device: torch.device,
) -> float:
    x_ids = list(prompt_ids) + list(answer_ids)
    x = torch.tensor([x_ids], dtype=torch.long, device=device)
    labels = torch.full_like(x, -100)
    labels[:, len(prompt_ids) :] = x[:, len(prompt_ids) :]
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=x, labels=labels)
    return float(out.loss.detach().item())


@torch.no_grad()
def generation_probe(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: Sequence[int],
    device: torch.device,
    max_new_tokens: int,
) -> Tuple[str, Optional[str]]:
    x = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    out = model.generate(
        input_ids=x,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    gen_ids = out[0][x.shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    m = re.search(r"\d{5}", text)
    return text, (m.group(0) if m else None)


def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, out_png: Path, out_pdf: Path) -> None:
    lengths = sorted(df["length"].unique())
    depths = sorted(df["depth"].unique())
    mat = np.full((len(depths), len(lengths)), np.nan, dtype=np.float32)
    for i, d in enumerate(depths):
        for j, L in enumerate(lengths):
            sub = df[(df["depth"] == d) & (df["length"] == L)]
            if not sub.empty:
                mat[i, j] = float(sub[value_col].iloc[0])

    fig, ax = plt.subplots(figsize=(1.8 + 1.2 * len(lengths), 2.2 + 0.55 * len(depths)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0 if value_col == "tf_accuracy" else None, vmax=1.0 if value_col == "tf_accuracy" else None)
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([str(x) for x in lengths], rotation=45, ha="right")
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([str(x) for x in depths])
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Needle Depth (%)")
    ax.set_title(title)

    for i in range(len(depths)):
        for j in range(len(lengths)):
            if math.isfinite(float(mat[i, j])):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def main() -> None:
    args = make_parser().parse_args()
    enforce_offline_mode()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = sorted(set(parse_int_csv(args.lengths)))
    depths = sorted(set(parse_int_csv(args.depths)))
    if not lengths or not depths:
        raise ValueError("lengths/depths cannot be empty")

    # Reuse the tested model-loading path from NIAH evaluator.
    model, tokenizer, attn_used, variant_used, rope_used = load_model_and_tokenizer(args)
    device = next(model.parameters()).device

    rows: List[Dict[str, object]] = []
    rng_global = random.Random(int(args.seed))

    for L in lengths:
        for d in depths:
            for t in range(int(args.trials_per_cell)):
                local_seed = rng_global.randint(0, 2**31 - 1) ^ (L * 131 + d * 17 + t)
                rng = random.Random(local_seed)
                passkey = f"{rng.randint(10000, 99999)}"
                false_key = sample_false_passkey(tokenizer, passkey, rng)

                prompt_ids = build_passkey_prompt_ids(
                    tokenizer=tokenizer,
                    seq_len=L,
                    depth_pct=d,
                    passkey=passkey,
                )
                true_ids = tokenizer.encode(passkey, add_special_tokens=False)
                false_ids = tokenizer.encode(false_key, add_special_tokens=False)
                if len(true_ids) == 0 or len(false_ids) == 0:
                    continue

                nll_true = answer_nll(model, prompt_ids, true_ids, device)
                nll_false = answer_nll(model, prompt_ids, false_ids, device)
                margin = float(nll_false - nll_true)
                tf_correct = bool(margin > 0.0)

                gen_text = ""
                gen_ans: Optional[str] = None
                gen_correct: Optional[bool] = None
                if args.run_generation_probe:
                    gen_text, gen_ans = generation_probe(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                    )
                    gen_correct = bool(gen_ans == passkey) if gen_ans is not None else False

                rows.append(
                    {
                        "length": int(L),
                        "depth": int(d),
                        "trial": int(t),
                        "passkey_true": passkey,
                        "passkey_false": false_key,
                        "nll_true": float(nll_true),
                        "nll_false": float(nll_false),
                        "margin_false_minus_true": margin,
                        "tf_correct": int(tf_correct),
                        "gen_answer": gen_ans if gen_ans is not None else "",
                        "gen_correct": int(gen_correct) if gen_correct is not None else -1,
                        "gen_text": gen_text,
                    }
                )

    if not rows:
        raise RuntimeError("No passkey samples were evaluated.")

    df = pd.DataFrame(rows)
    df_cell = (
        df.groupby(["length", "depth"], as_index=False)
        .agg(
            tf_accuracy=("tf_correct", "mean"),
            margin_mean=("margin_false_minus_true", "mean"),
            margin_std=("margin_false_minus_true", "std"),
            n=("tf_correct", "count"),
            gen_accuracy=("gen_correct", lambda s: float((s[s >= 0]).mean()) if (s >= 0).any() else float("nan")),
        )
        .sort_values(["length", "depth"])
    )
    df_len = (
        df.groupby(["length"], as_index=False)
        .agg(
            tf_accuracy=("tf_correct", "mean"),
            margin_mean=("margin_false_minus_true", "mean"),
            margin_std=("margin_false_minus_true", "std"),
            n=("tf_correct", "count"),
            gen_accuracy=("gen_correct", lambda s: float((s[s >= 0]).mean()) if (s >= 0).any() else float("nan")),
        )
        .sort_values(["length"])
    )

    df.to_csv(out_dir / "passkey_tf_results.csv", index=False)
    df_cell.to_csv(out_dir / "passkey_tf_summary_by_cell.csv", index=False)
    df_len.to_csv(out_dir / "passkey_tf_summary_by_length.csv", index=False)

    plot_heatmap(
        df=df_cell,
        value_col="tf_accuracy",
        title="Passkey TF Accuracy",
        out_png=out_dir / "passkey_tf_accuracy_heatmap.png",
        out_pdf=out_dir / "passkey_tf_accuracy_heatmap.pdf",
    )
    plot_heatmap(
        df=df_cell,
        value_col="margin_mean",
        title="Passkey TF Margin (NLL_false - NLL_true)",
        out_png=out_dir / "passkey_tf_margin_heatmap.png",
        out_pdf=out_dir / "passkey_tf_margin_heatmap.pdf",
    )
    inv_sha256, inv_path = resolve_inv_metadata(
        base_only=bool(args.base_only),
        adapter_path=args.adapter_path,
        custom_inv_freq_path=args.custom_inv_freq_path,
        rope_used=rope_used if isinstance(rope_used, dict) else None,
    )
    per_sample_scores_raw = [float(x) for x in df["tf_correct"].tolist()]
    manifest_json = Path(args.manifest_json).resolve().as_posix() if args.manifest_json else ""
    protocol_lock = {
        "base_model_path": args.base_model_path,
        "adapter_path": None if args.base_only else args.adapter_path,
        "variant": args.variant,
        "custom_inv_freq_path": args.custom_inv_freq_path,
        "attn_implementation": args.attn_implementation,
        "lengths": lengths,
        "depths": depths,
        "trials_per_cell": int(args.trials_per_cell),
        "seed": int(args.seed),
        "decode": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "use_cache": True,
        },
    }

    summary = {
        "protocol_lock": protocol_lock,
        "manifest_json": manifest_json,
        "per_sample_scores_raw": per_sample_scores_raw,
        "inv_sha256": inv_sha256,
        "meta": {
            "timestamp": now(),
            "base_model_path": args.base_model_path,
            "adapter_path": None if args.base_only else args.adapter_path,
            "variant_used": variant_used,
            "rope_used": rope_used,
            "attn_used": attn_used,
            "custom_inv_freq_path": args.custom_inv_freq_path,
            "lengths": lengths,
            "depths": depths,
            "trials_per_cell": int(args.trials_per_cell),
            "seed": int(args.seed),
            "run_generation_probe": bool(args.run_generation_probe),
            "manifest_json": manifest_json,
            "inv_sha256": inv_sha256,
            "inv_freq_path": inv_path,
            "protocol_lock": protocol_lock,
        },
        "overall": {
            "tf_accuracy": float(df["tf_correct"].mean()),
            "margin_mean": float(df["margin_false_minus_true"].mean()),
            "margin_std": float(df["margin_false_minus_true"].std(ddof=1)),
            "n": int(len(df)),
        },
        "by_length": {
            str(int(r["length"])): {
                "tf_accuracy": float(r["tf_accuracy"]),
                "margin_mean": float(r["margin_mean"]),
                "margin_std": float(r["margin_std"]) if math.isfinite(float(r["margin_std"])) else None,
                "n": int(r["n"]),
                "gen_accuracy": float(r["gen_accuracy"]) if math.isfinite(float(r["gen_accuracy"])) else None,
            }
            for _, r in df_len.iterrows()
        },
    }
    (out_dir / "passkey_tf_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
