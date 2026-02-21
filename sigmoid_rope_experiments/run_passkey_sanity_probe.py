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
from tqdm import tqdm

import run_phase4 as p4
from src.rope import RoPEFrequencyAllocator
from src.utils import cleanup_cuda, set_seed


def build_prompt(
    tokenizer: p4.TokenizerBase,
    context_length: int,
    passkey: str,
    mode: str,
) -> List[int] | None:
    filler = (
        "In the vast expanse of the universe, countless stars illuminate the darkness of space. "
        "Researchers carefully evaluate long-context reasoning under controlled settings. "
    )
    insertion = f"REMEMBER: The special number is {passkey}. REMEMBER: The special number is {passkey}. "
    extraction = "As stated above, the special number is "

    filler_ids = tokenizer.encode(filler)
    insertion_ids = tokenizer.encode(insertion)
    extraction_ids = tokenizer.encode(extraction)

    budget = context_length - len(insertion_ids) - len(extraction_ids) - 2
    if budget < 128:
        return None
    rep = budget // max(1, len(filler_ids)) + 3
    body = (filler_ids * rep)[:budget]

    if mode == "near":
        # Keep insertion very close to extraction to validate minimal retrieval.
        tail = min(48, max(8, len(body) // 20))
        pos = max(0, len(body) - tail)
    elif mode == "mid":
        pos = len(body) // 2
    elif mode == "far":
        pos = max(0, len(body) // 10)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    seq = body[:pos] + insertion_ids + body[pos:] + extraction_ids
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


def key_nll(model: p4.GPTSmall, prompt_ids: Sequence[int], key_ids: Sequence[int]) -> float:
    if len(key_ids) == 0:
        return float("nan")
    device = next(model.parameters()).device
    arr = np.array(list(prompt_ids) + list(key_ids), dtype=np.int64)
    x = torch.tensor(arr[None, :], dtype=torch.long, device=device)
    with torch.no_grad():
        tok_loss = model.compute_per_token_loss(x).detach().float().cpu().numpy()[0]  # len(arr)-1
    # key tokens correspond to targets starting at index len(prompt_ids)-1
    start = len(prompt_ids) - 1
    end = start + len(key_ids)
    seg = tok_loss[start:end]
    return float(np.mean(seg))


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
    ap = argparse.ArgumentParser(description="Passkey sanity probe: near/mid/far + teacher forcing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context_length", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--tokenizer_mode", type=str, default="auto", choices=["auto", "hf", "byte"])
    ap.add_argument("--tokenizer_path", type=str, default="")
    ap.add_argument("--local_model_candidates", type=str, default="")
    ap.add_argument("--root_dir", type=str, default=".")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root_dir = Path(args.root_dir).resolve()
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, tok_name = p4.resolve_tokenizer(args.tokenizer_mode, args.tokenizer_path, args.local_model_candidates)
    print(f"[sanity] device={device}, tokenizer={tok_name}, vocab={tokenizer.vocab_size}")

    # Match run_phase4_corrected frequency setup.
    head_dim = 64
    allocator = RoPEFrequencyAllocator(d=head_dim, base=10000.0)
    inv_std = allocator.standard()
    N = head_dim // 2
    k_sig = 16.05 / head_dim
    x0 = 0.47 * N
    j0 = 0.47 * N
    inv_sig = allocator.sigmoid(k=k_sig, x0=x0)
    inv_anchor20 = allocator.anchored_sigmoid(k=k_sig, j0=j0, anchor_factor=20.0)
    theta_min_std = float(inv_std[-1].item())
    theta_target = float(2.0 * math.pi / 8192.0)
    alpha_star = max(1.0, theta_target / max(theta_min_std, 1e-12))
    inv_anchor_star = allocator.anchored_sigmoid(k=k_sig, j0=j0, anchor_factor=alpha_star)

    specs: List[Tuple[str, torch.Tensor, Path]] = [
        ("Standard", inv_std, root_dir / "checkpoints" / "standard_best" / "checkpoint.pt"),
        ("Sigmoid", inv_sig, root_dir / "checkpoints" / "sigmoid_best" / "checkpoint.pt"),
        ("Anchored-20", inv_anchor20, root_dir / "checkpoints" / "anchored20_best" / "checkpoint.pt"),
        ("Anchored-alpha*", inv_anchor_star, root_dir / "checkpoints" / "anchored_alpha_best" / "checkpoint.pt"),
    ]

    modes = ["near", "mid", "far"]
    digit_re = re.compile(r"\d+")
    rows: List[Dict] = []
    rng = random.Random(args.seed + 1234)

    for model_name, inv, ckpt in specs:
        if not ckpt.exists():
            print(f"[sanity] skip {model_name}, missing ckpt: {ckpt}")
            continue
        model = load_best_model(device=device, vocab_size=tokenizer.vocab_size, inv_freq=inv, ckpt_path=ckpt)
        pbar = tqdm(total=len(modes) * args.repeats, desc=f"sanity-{model_name}", dynamic_ncols=True)
        for mode in modes:
            for rep in range(args.repeats):
                key = f"{rng.randint(10000, 99999)}"
                prompt = build_prompt(tokenizer, args.context_length, key, mode)
                if prompt is None:
                    rows.append({
                        "model": model_name,
                        "mode": mode,
                        "rep": rep,
                        "ok": 0,
                        "status": "invalid",
                    })
                    pbar.update(1)
                    continue

                x = torch.tensor([prompt], dtype=torch.long, device=device)
                out = greedy_generate(model, x, max_new_tokens=args.max_new_tokens)
                gen_ids = out[0, x.size(1):].detach().cpu().tolist()
                gen_txt = tokenizer.decode(gen_ids)
                pred_digits = "".join(digit_re.findall(gen_txt))
                ok = int((key in pred_digits) or (key in gen_txt))

                # Teacher-forcing preference: true key NLL vs random key NLL
                key_ids = tokenizer.encode(key)
                fake = key
                while fake == key:
                    fake = f"{rng.randint(10000, 99999)}"
                fake_ids = tokenizer.encode(fake)
                true_nll = key_nll(model, prompt, key_ids)
                fake_nll = key_nll(model, prompt, fake_ids)
                margin = fake_nll - true_nll

                rows.append({
                    "model": model_name,
                    "mode": mode,
                    "rep": rep,
                    "ok": ok,
                    "status": "ok",
                    "passkey": key,
                    "generated_text": gen_txt,
                    "pred_digits": pred_digits,
                    "true_key_nll": true_nll,
                    "fake_key_nll": fake_nll,
                    "nll_margin_fake_minus_true": margin,
                })
                pbar.update(1)
        pbar.close()
        del model
        cleanup_cuda()

    df = pd.DataFrame(rows)
    out_csv = data_dir / "passkey_sanity_probe.csv"
    out_csv_summary = data_dir / "passkey_sanity_probe_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    ok_df = df[df["status"] == "ok"].copy()
    summary = (
        ok_df.groupby(["model", "mode"], as_index=False)
        .agg(
            accuracy=("ok", "mean"),
            n=("ok", "count"),
            mean_true_nll=("true_key_nll", "mean"),
            mean_fake_nll=("fake_key_nll", "mean"),
            mean_margin=("nll_margin_fake_minus_true", "mean"),
        )
    )
    summary.to_csv(out_csv_summary, index=False, encoding="utf-8")

    print("\n[sanity] summary by model/mode")
    if not summary.empty:
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\n[sanity] saved: {out_csv}")
    print(f"[sanity] saved: {out_csv_summary}")


if __name__ == "__main__":
    main()

