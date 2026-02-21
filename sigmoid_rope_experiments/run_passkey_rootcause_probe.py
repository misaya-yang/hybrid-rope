#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import run_phase4 as p4
from src.rope import RoPEFrequencyAllocator
from src.utils import cleanup_cuda, set_seed


def build_prompt(
    tokenizer: p4.TokenizerBase,
    passkey: str,
    mode: str,
    context_length: int,
) -> List[int] | None:
    prefix = "This is a controlled memory test. "
    insertion = f"The special number is {passkey}. "
    extraction = "As stated above, the special number is "
    filler = (
        "In the vast expanse of the universe, countless stars illuminate the darkness of space. "
        "Researchers carefully evaluate long-context reasoning under controlled settings. "
    )

    p_ids = tokenizer.encode(prefix)
    i_ids = tokenizer.encode(insertion)
    e_ids = tokenizer.encode(extraction)
    f_ids = tokenizer.encode(filler)

    if mode == "direct":
        seq = p_ids + i_ids + e_ids
    elif mode == "near":
        gap = (f_ids * 2)[:64]
        seq = p_ids + i_ids + gap + e_ids
    elif mode == "far":
        budget = context_length - len(p_ids) - len(i_ids) - len(e_ids) - 2
        if budget < 128:
            return None
        rep = budget // max(1, len(f_ids)) + 2
        body = (f_ids * rep)[:budget]
        seq = p_ids + body[: len(body) // 3] + i_ids + body[len(body) // 3 :] + e_ids
    else:
        raise ValueError(mode)

    seq = seq[:context_length]
    if tokenizer.bos_token_id is not None:
        seq = [int(tokenizer.bos_token_id)] + seq
    return seq


def greedy_generate(model: p4.GPTSmall, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    model.eval()
    out = x
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(out)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
    return out


def next_token_true_id_prob(model: p4.GPTSmall, prompt_ids: Sequence[int], true_first_id: int) -> Tuple[float, int, float]:
    device = next(model.parameters()).device
    x = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_id = int(torch.argmax(probs, dim=-1).item())
        top_prob = float(probs[0, top_id].item())
        true_prob = float(probs[0, int(true_first_id)].item())
    return true_prob, top_id, top_prob


def load_best_model(
    device: torch.device,
    vocab_size: int,
    inv_freq: torch.Tensor,
    ckpt_path: Path,
) -> p4.GPTSmall:
    model = p4.GPTSmall(
        vocab_size=vocab_size,
        n_layers=12,
        n_heads=12,
        d_model=768,
        d_ff=3072,
        inv_freq=inv_freq,
        gradient_checkpointing=False,
    ).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Root-cause probe for passkey failure")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--context_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--tokenizer_mode", type=str, default="auto", choices=["auto", "hf", "byte"])
    ap.add_argument("--tokenizer_path", type=str, default="")
    ap.add_argument("--local_model_candidates", type=str, default="")
    ap.add_argument("--root_dir", type=str, default=".")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed + 99)
    digit_re = re.compile(r"\d+")

    root_dir = Path(args.root_dir).resolve()
    out_dir = root_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, tok_name = p4.resolve_tokenizer(args.tokenizer_mode, args.tokenizer_path, args.local_model_candidates)
    print(f"[probe] device={device}, tokenizer={tok_name}, vocab={tokenizer.vocab_size}")

    # Match phase4-corrected model variants.
    d = 64
    alloc = RoPEFrequencyAllocator(d=d, base=10000.0)
    inv_std = alloc.standard()
    N = d // 2
    k = 16.05 / d
    x0 = 0.47 * N
    inv_sig = alloc.sigmoid(k=k, x0=x0)
    inv_a20 = alloc.anchored_sigmoid(k=k, j0=x0, anchor_factor=20.0)
    theta_min_std = float(inv_std[-1].item())
    theta_target = float(2.0 * math.pi / 8192.0)
    alpha_star = max(1.0, theta_target / max(theta_min_std, 1e-12))
    inv_astar = alloc.anchored_sigmoid(k=k, j0=x0, anchor_factor=alpha_star)

    specs: List[Tuple[str, torch.Tensor, Path]] = [
        ("Standard", inv_std, root_dir / "checkpoints" / "standard_best" / "checkpoint.pt"),
        ("Sigmoid", inv_sig, root_dir / "checkpoints" / "sigmoid_best" / "checkpoint.pt"),
        ("Anchored-20", inv_a20, root_dir / "checkpoints" / "anchored20_best" / "checkpoint.pt"),
        ("Anchored-alpha*", inv_astar, root_dir / "checkpoints" / "anchored_alpha_best" / "checkpoint.pt"),
    ]

    modes = ["direct", "near", "far"]
    rows: List[Dict] = []
    nt_rows: List[Dict] = []

    for model_name, inv, ckpt in specs:
        if not ckpt.exists():
            print(f"[probe] skip {model_name}, missing {ckpt}")
            continue
        model = load_best_model(device=device, vocab_size=tokenizer.vocab_size, inv_freq=inv, ckpt_path=ckpt)
        pbar = tqdm(total=len(modes) * args.repeats, desc=f"probe-{model_name}", dynamic_ncols=True)
        for mode in modes:
            for rep in range(args.repeats):
                key = f"{rng.randint(10000, 99999)}"
                prompt = build_prompt(tokenizer, key, mode=mode, context_length=args.context_length)
                if prompt is None:
                    rows.append({"model": model_name, "mode": mode, "rep": rep, "ok": 0, "status": "invalid"})
                    pbar.update(1)
                    continue

                key_ids = tokenizer.encode(key)
                true_first = int(key_ids[0]) if key_ids else None
                if true_first is not None:
                    tp, top_id, top_p = next_token_true_id_prob(model, prompt, true_first)
                    nt_rows.append({
                        "model": model_name,
                        "mode": mode,
                        "rep": rep,
                        "true_first_token_prob": tp,
                        "top1_token_id": top_id,
                        "top1_token_prob": top_p,
                        "true_first_token_id": int(true_first),
                    })

                x = torch.tensor([prompt], dtype=torch.long, device=device)
                out = greedy_generate(model, x, max_new_tokens=args.max_new_tokens)
                gen_ids = out[0, x.size(1):].detach().cpu().tolist()
                gen_txt = tokenizer.decode(gen_ids)
                pred_digits = "".join(digit_re.findall(gen_txt))
                ok = int((key in pred_digits) or (key in gen_txt))
                rows.append({
                    "model": model_name,
                    "mode": mode,
                    "rep": rep,
                    "ok": ok,
                    "status": "ok",
                    "passkey": key,
                    "pred_digits": pred_digits,
                    "generated_text": gen_txt,
                })
                pbar.update(1)
        pbar.close()
        del model
        cleanup_cuda()

    df = pd.DataFrame(rows)
    nt_df = pd.DataFrame(nt_rows)
    out_main = out_dir / "passkey_rootcause_probe.csv"
    out_nt = out_dir / "passkey_rootcause_nexttoken.csv"
    df.to_csv(out_main, index=False, encoding="utf-8")
    nt_df.to_csv(out_nt, index=False, encoding="utf-8")

    ok_df = df[df["status"] == "ok"]
    summary = ok_df.groupby(["model", "mode"], as_index=False)["ok"].mean().rename(columns={"ok": "accuracy"})
    print("\n[probe] generation accuracy")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not nt_df.empty:
        nt_sum = nt_df.groupby(["model", "mode"], as_index=False).agg(
            mean_true_first_prob=("true_first_token_prob", "mean"),
            mean_top1_prob=("top1_token_prob", "mean"),
        )
        print("\n[probe] next-token prob")
        print(nt_sum.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print(f"\n[probe] saved: {out_main}")
    print(f"[probe] saved: {out_nt}")


if __name__ == "__main__":
    main()

