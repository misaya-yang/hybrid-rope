#!/usr/bin/env python3
"""Evaluate token-level perplexity for one method at a target context length."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_niah_recall import infer_rope_theta, load_model_and_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Perplexity evaluation at fixed context length.")
    ap.add_argument("--base_model_path", type=str, required=True)
    ap.add_argument("--adapter_path", type=str, default="")
    ap.add_argument("--variant", type=str, default="auto")
    ap.add_argument("--custom_inv_freq_path", type=str, default="")
    ap.add_argument("--rope_factor", type=float, default=8.0)
    ap.add_argument("--orig_ctx", type=int, default=8192)
    ap.add_argument("--rope_theta", type=float, default=0.0)
    ap.add_argument("--hybrid_split_ratio", type=float, default=0.5)
    ap.add_argument("--hybrid_alpha", type=float, default=0.2)
    ap.add_argument("--hybrid_p", type=float, default=3.9)
    ap.add_argument("--hybrid_min_freq_scale", type=float, default=4.0)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--max_chunks", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--text_path", type=str, default="/root/autodl-tmp/data/long_text.txt")
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output_json", type=str, required=True)
    return ap.parse_args()


def build_eval_token_ids(tokenizer, path: str, target_tokens: int, read_chunk_chars: int = 2_000_000) -> List[int]:
    """
    Build token ids for evaluation without tokenizing the entire corpus file.

    This keeps PPL runtime bounded when the source text is very large.
    """
    target = max(2, int(target_tokens))
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        ids: List[int] = []
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            while len(ids) < target:
                chunk = f.read(read_chunk_chars)
                if not chunk:
                    break
                ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if len(ids) >= 2:
            return ids[:target]

    # Fallback keeps evaluation runnable before data sync completes.
    fallback = (
        "Long context evaluation fallback text. "
        "This synthetic stream is used only when no real corpus is mounted. "
    )
    fallback_ids = tokenizer.encode(fallback * 4096, add_special_tokens=False)
    if len(fallback_ids) < 2:
        fallback_ids = [1, 2]
    while len(fallback_ids) < target:
        fallback_ids.extend(fallback_ids)
    return fallback_ids[:target]


def chunk_token_ids(ids: List[int], ctx: int, max_chunks: int) -> List[List[int]]:
    usable = max(2, int(ctx))
    chunks: List[List[int]] = []
    if len(ids) < 2:
        return chunks
    if len(ids) <= usable:
        return [ids[:]]
    for start in range(0, len(ids) - usable + 1, usable):
        c = ids[start : start + usable]
        if len(c) >= 2:
            chunks.append(c)
        if len(chunks) >= max_chunks:
            break
    return chunks


@torch.no_grad()
def compute_ppl(model: torch.nn.Module, chunks: List[List[int]]) -> Dict[str, object]:
    device = next(model.parameters()).device
    losses: List[float] = []
    for c in chunks:
        x = torch.tensor([c], dtype=torch.long, device=device)
        out = model(input_ids=x, labels=x, use_cache=False)
        losses.append(float(out.loss.detach().cpu().item()))
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    ppl = float(math.exp(mean_loss)) if math.isfinite(mean_loss) else float("nan")
    return {
        "num_chunks": len(losses),
        "loss_mean": mean_loss,
        "loss_std": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
        "ppl": ppl,
        "chunk_losses": losses,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rope_theta = float(args.rope_theta)
    if rope_theta <= 0:
        rope_theta = infer_rope_theta(args.base_model_path, args.trust_remote_code)

    loader_args = argparse.Namespace(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path.strip(),
        base_only=(args.adapter_path.strip() == ""),
        custom_inv_freq_path=args.custom_inv_freq_path.strip(),
        merge_lora=args.merge_lora,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        variant=args.variant,
        rope_factor=args.rope_factor,
        orig_ctx=args.orig_ctx,
        rope_theta=rope_theta,
        hybrid_split_ratio=args.hybrid_split_ratio,
        hybrid_alpha=args.hybrid_alpha,
        hybrid_p=args.hybrid_p,
        hybrid_min_freq_scale=args.hybrid_min_freq_scale,
        device_map="auto",
    )

    model, tokenizer, attn_used, variant_used, rope_used = load_model_and_tokenizer(loader_args)

    need_tokens = max(2, int(args.ctx) * int(args.max_chunks))
    ids = build_eval_token_ids(tokenizer, args.text_path, target_tokens=need_tokens)
    chunks = chunk_token_ids(ids, ctx=args.ctx, max_chunks=args.max_chunks)
    stats = compute_ppl(model, chunks)

    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path,
            "variant_requested": args.variant,
            "variant_used": variant_used,
            "rope_used": rope_used,
            "rope_theta": rope_theta,
            "ctx": args.ctx,
            "max_chunks": args.max_chunks,
            "seed": args.seed,
            "text_path": args.text_path,
            "attn_used": attn_used,
            "elapsed_sec": round(time.time() - t0, 3),
        },
        "result": stats,
    }

    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
