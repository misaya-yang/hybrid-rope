#!/usr/bin/env python3
"""E3-lite: online attention-distance histogram and power-law fit."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rope.attn_hist import accumulate_distance_histogram, bootstrap_alpha_ci, fit_power_law  # noqa: E402
from scripts.eval_niah_recall import infer_rope_theta, load_model_and_tokenizer  # noqa: E402


def parse_int_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run E3-lite attention distance prior estimation.")
    ap.add_argument("--exp", type=str, default="E3")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--layers", type=str, default="2,16,30")
    ap.add_argument("--heads", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--bins", type=int, default=256)
    ap.add_argument("--query_stride", type=int, default=16)
    ap.add_argument("--query_block", type=int, default=128)
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--adapter_path", type=str, default="")
    ap.add_argument("--variant", type=str, default="auto")
    ap.add_argument("--custom_inv_freq_path", type=str, default="")
    ap.add_argument("--text_path", type=str, default="/root/autodl-tmp/data/long_text.txt")
    ap.add_argument("--out_fig", type=str, default="artifacts/figures/Dhat_loglog.png")
    ap.add_argument("--out_json", type=str, default="artifacts/results/prior_fit.json")
    ap.add_argument(
        "--save_hist",
        action="store_true",
        help="Include rebinned/overall histogram arrays in JSON output for downstream bridge analysis.",
    )
    return ap.parse_args()


def find_transformer_layers(model: torch.nn.Module):
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("base_model.model.model.layers", lambda m: m.base_model.model.model.layers),
        ("base_model.model.layers", lambda m: m.base_model.model.layers),
    ]
    for _, getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                return layers
        except Exception:
            continue
    raise RuntimeError("Cannot locate transformer layers in model object.")


def build_input_pool(tokenizer, text_path: str, ctx: int, n: int) -> List[List[int]]:
    p = Path(text_path)
    if p.exists() and p.stat().st_size > 0:
        text = p.read_text(encoding="utf-8", errors="ignore")
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= ctx + 2:
            chunks = []
            max_start = max(1, len(ids) - ctx - 1)
            rng = random.Random(1337)
            for _ in range(n):
                s = rng.randint(0, max_start - 1)
                chunks.append(ids[s : s + ctx])
            return chunks

    rng = np.random.default_rng(1337)
    vocab = int(getattr(tokenizer, "vocab_size", 128256))
    chunks = []
    for _ in range(n):
        arr = rng.integers(10, max(11, vocab - 1), size=ctx, dtype=np.int64)
        chunks.append(arr.tolist())
    return chunks


def rebin_hist(hist: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0:
        raise ValueError("bins must be >0")
    if bins >= len(hist):
        return hist.copy()
    edges = np.linspace(0, len(hist), bins + 1, dtype=np.int64)
    out = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        s, e = int(edges[i]), int(edges[i + 1])
        if e <= s:
            e = min(len(hist), s + 1)
        out[i] = float(np.sum(hist[s:e]))
    return out


def plot_loglog(layer_curves: Dict[int, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for layer, hist in sorted(layer_curves.items()):
        d = np.arange(1, len(hist), dtype=np.float64)
        y = hist[1:]
        mask = np.isfinite(y) & (y > 0)
        if int(mask.sum()) < 10:
            continue
        ax.plot(np.log10(d[mask]), np.log10(y[mask]), linewidth=1.2, alpha=0.8, label=f"L{layer}")
    ax.set_xlabel("log10(Δ)")
    ax.set_ylabel("log10(D̂(Δ))")
    ax.set_title("E3-lite: attention distance prior")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    layers_sel = parse_int_csv(args.layers)
    heads_sel = parse_int_csv(args.heads)
    if not layers_sel or not heads_sel:
        raise ValueError("layers/heads cannot be empty")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rope_theta = infer_rope_theta(args.model, True)
    model, tokenizer, _, _, _ = load_model_and_tokenizer(
        base_model_path=args.model,
        adapter_path=args.adapter_path or None,
        custom_inv_freq_path=args.custom_inv_freq_path or None,
        merge_lora=False,
        attn_mode=args.attn_implementation,
        trust_remote_code=True,
        variant=args.variant,
        rope_factor=8.0,
        orig_ctx=8192,
        rope_theta=rope_theta,
        hybrid_split_ratio=0.5,
        hybrid_alpha=0.2,
        hybrid_p=3.9,
        hybrid_min_freq_scale=4.0,
    )

    model.eval()
    layers = find_transformer_layers(model)
    max_layer = len(layers) - 1
    layers_sel = [l for l in layers_sel if 0 <= l <= max_layer]
    if not layers_sel:
        raise ValueError(f"No valid layers left; max layer index={max_layer}")

    input_chunks = build_input_pool(tokenizer, args.text_path, args.ctx, args.N)
    max_dist = args.ctx - 1
    per_layer_hists: Dict[int, List[np.ndarray]] = {l: [] for l in layers_sel}

    for s_idx, ids in enumerate(input_chunks):
        x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
        with torch.no_grad():
            out = model(input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("Model did not return hidden_states.")

        for layer_idx in layers_sel:
            layer = layers[layer_idx]
            attn_mod = layer.self_attn
            h_in = hs[layer_idx]  # input of layer_idx
            bsz, seqlen, hidden = h_in.shape
            if bsz != 1:
                raise RuntimeError("Expected batch size 1 for E3-lite.")

            num_heads = int(getattr(attn_mod, "num_heads", getattr(model.config, "num_attention_heads")))
            num_kv_heads = int(getattr(attn_mod, "num_key_value_heads", getattr(model.config, "num_key_value_heads", num_heads)))
            head_dim = hidden // num_heads
            group = max(1, num_heads // num_kv_heads)

            q = attn_mod.q_proj(h_in).view(1, seqlen, num_heads, head_dim).permute(0, 2, 1, 3)[0]
            k = attn_mod.k_proj(h_in).view(1, seqlen, num_kv_heads, head_dim).permute(0, 2, 1, 3)[0]
            if num_kv_heads != num_heads:
                k = k.repeat_interleave(group, dim=0)

            valid_heads = [h for h in heads_sel if 0 <= h < num_heads]
            if not valid_heads:
                continue

            q_sel = q[valid_heads]
            k_sel = k[valid_heads]
            q_pos = torch.arange(0, seqlen, args.query_stride, device=q_sel.device, dtype=torch.long)
            if q_pos.numel() == 0:
                q_pos = torch.arange(seqlen, device=q_sel.device, dtype=torch.long)

            hist = torch.zeros(max_dist + 1, device=q_sel.device, dtype=torch.float32)
            accumulate_distance_histogram(
                q=q_sel[:, q_pos, :],
                k=k_sel,
                query_positions=q_pos,
                max_distance=max_dist,
                hist=hist,
                block_q=max(1, int(args.query_block)),
            )
            hist_np = hist.detach().cpu().numpy().astype(np.float64)
            denom = float(np.sum(hist_np))
            if denom > 0:
                hist_np = hist_np / denom
            per_layer_hists[layer_idx].append(hist_np)

        if (s_idx + 1) % 4 == 0:
            print(f"[E3-lite] processed {s_idx+1}/{len(input_chunks)} samples", flush=True)

    layer_mean_hists: Dict[int, np.ndarray] = {}
    layer_fits: Dict[int, Dict[str, object]] = {}
    for l in layers_sel:
        if not per_layer_hists[l]:
            continue
        mean_hist = np.mean(per_layer_hists[l], axis=0)
        layer_mean_hists[l] = mean_hist
        fit = fit_power_law(mean_hist, d_min=8, d_max=min(args.ctx // 2, 4096))
        ci = bootstrap_alpha_ci(per_layer_hists[l], n_bootstrap=400, seed=args.seed)
        layer_fits[l] = {
            "alpha": fit.get("alpha"),
            "r2": fit.get("r2"),
            "n_points": fit.get("n_points"),
            "alpha_ci_low": ci.get("alpha_ci_low"),
            "alpha_ci_high": ci.get("alpha_ci_high"),
            "n_samples": len(per_layer_hists[l]),
        }

    if not layer_mean_hists:
        raise RuntimeError("No histograms were collected.")

    overall_hist = np.mean(list(layer_mean_hists.values()), axis=0)
    overall_fit = fit_power_law(overall_hist, d_min=8, d_max=min(args.ctx // 2, 4096))
    overall_ci = bootstrap_alpha_ci(list(layer_mean_hists.values()), n_bootstrap=600, seed=args.seed)

    plot_curves = {l: rebin_hist(h, args.bins) for l, h in layer_mean_hists.items()}
    out_fig = Path(args.out_fig)
    plot_loglog(plot_curves, out_fig)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "exp": args.exp,
            "model": args.model,
            "ctx": args.ctx,
            "seed": args.seed,
            "N": args.N,
            "layers": layers_sel,
            "heads": heads_sel,
            "bins": args.bins,
            "query_stride": args.query_stride,
            "query_block": args.query_block,
            "hist_length": int(overall_hist.shape[0]),
        },
        "overall": {
            "alpha": overall_fit.get("alpha"),
            "r2": overall_fit.get("r2"),
            "n_points": overall_fit.get("n_points"),
            "alpha_ci_low": overall_ci.get("alpha_ci_low"),
            "alpha_ci_high": overall_ci.get("alpha_ci_high"),
        },
        "by_layer": {str(k): v for k, v in layer_fits.items()},
        "outputs": {
            "figure": str(out_fig),
            "prior_fit_json": str(out_json),
        },
    }
    if args.save_hist:
        payload["overall_hist"] = overall_hist.astype(float).tolist()
        payload["overall_hist_rebinned"] = rebin_hist(overall_hist, args.bins).astype(float).tolist()
        payload["by_layer_hist_rebinned"] = {
            str(layer): rebin_hist(hist, args.bins).astype(float).tolist()
            for layer, hist in layer_mean_hists.items()
        }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
