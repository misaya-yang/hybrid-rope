#!/usr/bin/env python3
"""
Attention Distance Visualization: EVQ vs Geometric RoPE (750M)
==============================================================
Produces publication-quality figures comparing attention distance
distributions between EVQ-Cosh and Geometric frequency allocation.

Outputs:
  1. attn_weight_vs_distance.pdf   — Attention weight vs distance curve, per layer
  2. attn_head_avg_distance.pdf    — Per-head average attention distance scatter
  3. attn_distance_heatmap.pdf     — Layer × distance heatmap (side-by-side)

Usage:
  python visualize_attention_distance.py \
      --geo_ckpt  /path/to/geo_750m_checkpoint.pt \
      --evq_ckpt  /path/to/evq_750m_checkpoint.pt \
      --tier 750m \
      --tau 1.5 \
      --seq_len 8192 \
      --output_dir ./figures/attention_viz

  # Quick test with shorter context:
  python visualize_attention_distance.py \
      --geo_ckpt  /path/to/geo.pt \
      --evq_ckpt  /path/to/evq.pt \
      --seq_len 2048 \
      --output_dir ./figures/attention_viz

Notes:
  - For 750M @ 8K: ~6GB VRAM per model (R6000 24GB is fine)
  - Attention weights are computed manually (not via F.scaled_dot_product_attention)
    so we can capture per-head weight matrices.
  - Statistics are accumulated on-the-fly; full attention matrices are NOT stored
    in memory to keep VRAM usage manageable.
"""

import argparse
import math
import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# ── Project imports ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_evq_sweep import (
    GPT,
    RotaryEmbedding,
    Attention,
    apply_rope,
    rotate_half,
    evq_cosh_inv_freq,
)

# ── Tier configs (mirrors phase21b_quality_eval_clean.py) ────────────
TIER_CONFIGS = {
    "454m": dict(
        vocab_size=50304, hidden_size=1024, num_layers=24,
        num_heads=16, head_dim=64, intermediate_size=4096,
    ),
    "750m": dict(
        vocab_size=50304, hidden_size=1536, num_layers=18,
        num_heads=24, head_dim=64, intermediate_size=6144,
    ),
}


def geometric_inv_freq(dim=64, base=500000.0):
    n = dim // 2
    return torch.tensor(
        [1.0 / (base ** (2 * i / dim)) for i in range(n)],
        dtype=torch.float32,
    )


def _load_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    rope_keys = [k for k in state if ".rope." in k]
    for key in rope_keys:
        del state[key]
    return state


# ── Attention statistics collector ───────────────────────────────────

class AttentionStatsCollector:
    """
    Collects attention distance statistics WITHOUT storing full attention matrices.

    For each layer & head, we accumulate:
      - distance_histogram: attention weight binned by query-key distance
      - avg_distance: weighted average attention distance
      - total_weight: total attention mass (for normalization)

    Memory: O(num_layers * num_heads * num_bins) ≈ negligible
    """

    def __init__(self, num_layers: int, num_heads: int,
                 seq_len: int, num_bins: int = 256):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.num_bins = num_bins

        # Log-spaced bins from 1 to seq_len
        self.bin_edges = np.concatenate([
            [0],
            np.logspace(0, np.log10(seq_len), num_bins).astype(int)
        ])
        self.bin_edges = np.unique(self.bin_edges)
        self.num_bins = len(self.bin_edges) - 1
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        # Accumulators: [num_layers, num_heads, num_bins]
        self.distance_hist = np.zeros((num_layers, num_heads, self.num_bins), dtype=np.float64)
        # Per-head average distance: [num_layers, num_heads]
        self.weighted_dist_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
        self.weight_sum = np.zeros((num_layers, num_heads), dtype=np.float64)

        self._hooks = []

    def register_hooks(self, model: GPT):
        """Register forward hooks on all Attention modules."""
        for layer_idx, block in enumerate(model.blocks):
            hook = self._make_hook(layer_idx, block.attn)
            h = block.attn.register_forward_hook(hook)
            self._hooks.append(h)

    def _make_hook(self, layer_idx: int, attn_module: Attention):
        """Create a hook that recomputes attention weights and accumulates stats."""
        collector = self

        def hook_fn(module, inputs, output):
            x = inputs[0]  # (B, L, H)
            B, L, _ = x.shape
            nh = module.nh
            hd = module.hd

            with torch.no_grad():
                # Recompute QKV and apply RoPE (same as Attention.forward)
                qkv = module.qkv(x).view(B, L, 3, nh, hd).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]  # (B, nh, L, hd)
                cos, sin = module.rope(L)
                cos, sin = cos[None, None], sin[None, None]
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)

                # Compute attention scores manually
                scale = 1.0 / math.sqrt(hd)

                # Process in chunks along query dimension to save memory
                # For each chunk of queries, compute attention over ALL keys
                chunk_size = min(512, L)
                for q_start in range(0, L, chunk_size):
                    q_end = min(q_start + chunk_size, L)
                    q_chunk = q[:, :, q_start:q_end, :]  # (B, nh, chunk, hd)

                    # Scores: (B, nh, chunk, L)
                    scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale

                    # Causal mask: query at position q_pos can only attend to k_pos <= q_pos
                    q_positions = torch.arange(q_start, q_end, device=x.device)
                    k_positions = torch.arange(L, device=x.device)
                    # mask[i, j] = True if k_positions[j] > q_positions[i]
                    causal_mask = k_positions[None, :] > q_positions[:, None]
                    scores.masked_fill_(causal_mask[None, None, :, :], float('-inf'))

                    # Softmax → attention weights
                    weights = F.softmax(scores, dim=-1)  # (B, nh, chunk, L)

                    # Compute distances: |q_pos - k_pos| for valid (non-masked) positions
                    # distances[i, j] = q_positions[i] - k_positions[j]  (always >= 0 for causal)
                    distances = q_positions[:, None] - k_positions[None, :]  # (chunk, L)
                    distances = distances.clamp(min=0)

                    # Accumulate statistics (move to CPU for numpy)
                    weights_cpu = weights.float().mean(dim=0).cpu().numpy()  # (nh, chunk, L)
                    distances_cpu = distances.cpu().numpy()  # (chunk, L)

                    for h in range(nh):
                        w_h = weights_cpu[h]  # (chunk, L)
                        # Flatten for histogram
                        d_flat = distances_cpu.ravel()
                        w_flat = w_h.ravel()

                        # Remove -inf/NaN positions (where mask was applied)
                        valid = np.isfinite(w_flat) & (w_flat > 0)
                        d_valid = d_flat[valid]
                        w_valid = w_flat[valid]

                        if len(w_valid) == 0:
                            continue

                        # Bin into distance histogram
                        hist, _ = np.histogram(d_valid, bins=collector.bin_edges,
                                               weights=w_valid)
                        collector.distance_hist[layer_idx, h] += hist

                        # Accumulate for weighted average distance
                        collector.weighted_dist_sum[layer_idx, h] += np.sum(d_valid * w_valid)
                        collector.weight_sum[layer_idx, h] += np.sum(w_valid)

        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_avg_distance(self):
        """Per-head average attention distance. Shape: (num_layers, num_heads)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            avg = self.weighted_dist_sum / np.maximum(self.weight_sum, 1e-12)
        return avg

    def get_normalized_hist(self):
        """Distance histogram normalized per head. Shape: (num_layers, num_heads, num_bins)"""
        totals = self.distance_hist.sum(axis=-1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = self.distance_hist / np.maximum(totals, 1e-12)
        return normalized


# ── Model loading ────────────────────────────────────────────────────

def load_model(ckpt_path, tier, rope_type, tau, base, seq_len, device):
    """Load model with specified RoPE configuration."""
    cfg = TIER_CONFIGS[tier].copy()
    dim = cfg["head_dim"]

    if rope_type == "evq":
        inv_freq = evq_cosh_inv_freq(head_dim=dim, tau=tau, base=base)
    else:
        inv_freq = geometric_inv_freq(dim, base)

    cfg["max_position_embeddings"] = seq_len + 128
    cfg["seq_len"] = seq_len + 128

    model = GPT(cfg, inv_freq)
    state = _load_state(ckpt_path)
    missing, _ = model.load_state_dict(state, strict=False)
    other_missing = [k for k in missing if ".rope." not in k]
    if other_missing:
        print(f"  WARNING: missing non-rope keys: {other_missing}")

    model = model.to(device)
    for block in model.blocks:
        block.attn.rope.inv_freq.copy_(inv_freq.to(device))
    model.extend_rope(seq_len + 128)
    model.eval()
    del state
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return model


# ── Input data preparation ───────────────────────────────────────────

def prepare_input(seq_len: int, device, tokenizer=None, text_file=None):
    """
    Prepare input token IDs for attention analysis.

    Priority:
      1. If text_file is provided, tokenize from file
      2. If tokenizer available, use a built-in sample text
      3. Fallback: random token IDs (attention patterns still meaningful for PE comparison)
    """
    if text_file and os.path.exists(text_file):
        print(f"  Loading text from {text_file}")
        with open(text_file, "r") as f:
            text = f.read()
        if tokenizer is not None:
            ids = tokenizer.encode(text, add_special_tokens=True)
            ids = ids[:seq_len]
            if len(ids) < seq_len:
                print(f"  WARNING: text only {len(ids)} tokens, padding to {seq_len}")
                ids = ids + [tokenizer.eos_token_id] * (seq_len - len(ids))
            return torch.tensor([ids], dtype=torch.long, device=device)

    if tokenizer is not None:
        try:
            from datasets import load_dataset
            print("  Loading sample from FineWeb-Edu for attention analysis...")
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                              split="train", streaming=True)
            texts = []
            total_tokens = 0
            for sample in ds:
                texts.append(sample["text"])
                total_tokens += len(sample["text"].split())
                if total_tokens > seq_len * 2:
                    break
            full_text = "\n\n".join(texts)
            ids = tokenizer.encode(full_text, add_special_tokens=True)[:seq_len]
            print(f"  Tokenized {len(ids)} tokens from FineWeb-Edu")
            return torch.tensor([ids], dtype=torch.long, device=device)
        except Exception as e:
            print(f"  Could not load FineWeb-Edu: {e}")

    # Fallback: random tokens (still valid for PE structural comparison)
    print("  Using random token IDs (structural PE comparison still valid)")
    ids = torch.randint(100, 50000, (1, seq_len), device=device)
    return ids


# ── Visualization ────────────────────────────────────────────────────

def plot_attention_weight_vs_distance(geo_collector, evq_collector, output_path, tier):
    """
    Figure 1: Attention weight vs distance curve.
    One subplot per layer, EVQ vs Geo lines overlaid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    num_layers = geo_collector.num_layers
    geo_hist = geo_collector.get_normalized_hist()  # (L, H, bins)
    evq_hist = evq_collector.get_normalized_hist()
    bin_centers = geo_collector.bin_centers

    # Average over heads for per-layer curves
    geo_layer = geo_hist.mean(axis=1)  # (L, bins)
    evq_layer = evq_hist.mean(axis=1)

    cols = 3
    rows = math.ceil(num_layers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows),
                              sharex=True, sharey=False)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for layer_idx in range(num_layers):
        ax = axes_flat[layer_idx]
        ax.plot(bin_centers, geo_layer[layer_idx], color="#2196F3",
                alpha=0.85, linewidth=1.5, label="Geometric")
        ax.plot(bin_centers, evq_layer[layer_idx], color="#FF5722",
                alpha=0.85, linewidth=1.5, label="EVQ-Cosh")
        ax.set_xscale("log")
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.tick_params(labelsize=8)
        if layer_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Hide unused subplots
    for idx in range(num_layers, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.supxlabel("Attention Distance (tokens)", fontsize=12)
    fig.supylabel("Normalized Attention Weight", fontsize=12)
    fig.suptitle(f"Attention Weight vs Distance — {tier.upper()} (head-averaged)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_head_avg_distance(geo_collector, evq_collector, output_path, tier):
    """
    Figure 2: Per-head average attention distance.
    Scatter plot: x = Geo avg distance, y = EVQ avg distance.
    Points above diagonal → EVQ attends farther.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geo_avg = geo_collector.get_avg_distance()  # (L, H)
    evq_avg = evq_collector.get_avg_distance()
    num_layers = geo_avg.shape[0]

    # Color by layer
    cmap = plt.cm.viridis
    colors = [cmap(i / max(num_layers - 1, 1)) for i in range(num_layers)]

    fig, ax = plt.subplots(figsize=(7, 7))

    for layer_idx in range(num_layers):
        ax.scatter(geo_avg[layer_idx], evq_avg[layer_idx],
                   c=[colors[layer_idx]] * geo_avg.shape[1],
                   s=30, alpha=0.7, edgecolors="white", linewidths=0.3,
                   label=f"L{layer_idx}" if layer_idx % 3 == 0 else None)

    # Diagonal reference line
    all_vals = np.concatenate([geo_avg.ravel(), evq_avg.ravel()])
    lo, hi = all_vals.min() * 0.8, all_vals.max() * 1.2
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Geometric — Avg Attention Distance (tokens)", fontsize=11)
    ax.set_ylabel("EVQ-Cosh — Avg Attention Distance (tokens)", fontsize=11)
    ax.set_title(f"Per-Head Average Attention Distance — {tier.upper()}\n"
                 f"(above diagonal = EVQ attends farther)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_distance_heatmap(geo_collector, evq_collector, output_path, tier):
    """
    Figure 3: Distance heatmap (layer × distance bins).
    Side-by-side: Geo | EVQ | Difference.
    Head-averaged, log-distance x-axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    geo_hist = geo_collector.get_normalized_hist().mean(axis=1)  # (L, bins)
    evq_hist = evq_collector.get_normalized_hist().mean(axis=1)
    diff = evq_hist - geo_hist  # positive = EVQ has more weight at this distance
    bin_centers = geo_collector.bin_centers

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Shared colormap for Geo and EVQ
    vmax = max(geo_hist.max(), evq_hist.max())
    for ax_idx, (data, title, cmap_name) in enumerate([
        (geo_hist, "Geometric", "Blues"),
        (evq_hist, "EVQ-Cosh", "Oranges"),
    ]):
        ax = axes[ax_idx]
        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap=cmap_name, vmin=0, vmax=vmax,
                       extent=[0, data.shape[1], -0.5, data.shape[0] - 0.5])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Distance Bin Index (log-spaced)", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Layer", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Difference plot
    ax = axes[2]
    abs_max = max(abs(diff.min()), abs(diff.max()))
    if abs_max < 1e-12:
        abs_max = 1.0
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    im = ax.imshow(diff, aspect="auto", origin="lower",
                   cmap="RdBu_r", norm=norm,
                   extent=[0, diff.shape[1], -0.5, diff.shape[0] - 0.5])
    ax.set_title("Δ (EVQ − Geo)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Distance Bin Index (log-spaced)", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Attention Distance Distribution Heatmap — {tier.upper()} (head-averaged)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_layer_aggregate_comparison(geo_collector, evq_collector, output_path, tier):
    """
    Figure 4 (bonus): Single-panel summary.
    Aggregate across all layers: attention weight vs distance for Geo and EVQ.
    Clearest view of the PE difference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geo_hist = geo_collector.get_normalized_hist().mean(axis=(0, 1))  # (bins,)
    evq_hist = evq_collector.get_normalized_hist().mean(axis=(0, 1))
    bin_centers = geo_collector.bin_centers

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(bin_centers, geo_hist, color="#2196F3", linewidth=2, label="Geometric")
    ax.plot(bin_centers, evq_hist, color="#FF5722", linewidth=2, label="EVQ-Cosh")

    ax.fill_between(bin_centers, geo_hist, evq_hist,
                     where=(evq_hist > geo_hist), alpha=0.15, color="#FF5722",
                     label="EVQ > Geo")
    ax.fill_between(bin_centers, geo_hist, evq_hist,
                     where=(geo_hist > evq_hist), alpha=0.15, color="#2196F3",
                     label="Geo > EVQ")

    ax.set_xscale("log")
    ax.set_xlabel("Attention Distance (tokens)", fontsize=12)
    ax.set_ylabel("Normalized Attention Weight", fontsize=12)
    ax.set_title(f"Aggregate Attention Distance Distribution — {tier.upper()}\n"
                 f"(averaged across all layers and heads)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def collect_attention_stats(model, input_ids, tier_cfg):
    """Run forward pass and collect attention statistics."""
    num_layers = tier_cfg["num_layers"]
    num_heads = tier_cfg["num_heads"]
    seq_len = input_ids.shape[1]

    collector = AttentionStatsCollector(num_layers, num_heads, seq_len)
    collector.register_hooks(model)

    print(f"  Running forward pass (seq_len={seq_len})...")
    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids)
    t1 = time.time()
    print(f"  Forward pass done in {t1 - t0:.1f}s")

    collector.remove_hooks()
    return collector


def main():
    parser = argparse.ArgumentParser(
        description="Attention distance visualization for EVQ vs Geometric RoPE")
    parser.add_argument("--geo_ckpt", type=str, required=True,
                        help="Path to Geometric RoPE checkpoint")
    parser.add_argument("--evq_ckpt", type=str, required=True,
                        help="Path to EVQ-Cosh RoPE checkpoint")
    parser.add_argument("--tier", type=str, default="750m",
                        choices=["454m", "750m"],
                        help="Model tier (default: 750m)")
    parser.add_argument("--tau", type=float, default=1.5,
                        help="EVQ tau parameter (default: 1.5)")
    parser.add_argument("--base", type=float, default=500000.0,
                        help="RoPE base frequency (default: 500000)")
    parser.add_argument("--seq_len", type=int, default=8192,
                        help="Sequence length for attention analysis (default: 8192)")
    parser.add_argument("--text_file", type=str, default=None,
                        help="Optional text file for input (otherwise uses FineWeb or random)")
    parser.add_argument("--output_dir", type=str, default="./figures/attention_viz",
                        help="Output directory for figures")
    parser.add_argument("--num_bins", type=int, default=256,
                        help="Number of distance histogram bins (default: 256)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "bf16", "fp16"],
                        help="Compute dtype (default: bf16)")
    args = parser.parse_args()

    # ── Device setup ─────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    compute_dtype = dtype_map[args.dtype]
    print(f"Device: {device}  |  Dtype: {args.dtype}")

    tier_cfg = TIER_CONFIGS[args.tier]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Prepare input ────────────────────────────────────────────────
    print("\n=== Preparing input tokens ===")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    except Exception:
        tokenizer = None
        print("  transformers not available, will use random tokens")

    input_ids = prepare_input(args.seq_len, device, tokenizer, args.text_file)
    print(f"  Input shape: {input_ids.shape}")

    # ── Collect Geometric attention stats ────────────────────────────
    print("\n=== Loading Geometric model ===")
    geo_model = load_model(
        args.geo_ckpt, args.tier, "geo", args.tau, args.base, args.seq_len, device)
    if compute_dtype != torch.float32:
        geo_model = geo_model.to(dtype=compute_dtype)
        # Hooks do manual attention in float32 for numerical stability

    print("\n=== Collecting Geometric attention statistics ===")
    geo_collector = collect_attention_stats(geo_model, input_ids, tier_cfg)
    del geo_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Collect EVQ attention stats ──────────────────────────────────
    print("\n=== Loading EVQ model ===")
    evq_model = load_model(
        args.evq_ckpt, args.tier, "evq", args.tau, args.base, args.seq_len, device)
    if compute_dtype != torch.float32:
        evq_model = evq_model.to(dtype=compute_dtype)

    print("\n=== Collecting EVQ attention statistics ===")
    evq_collector = collect_attention_stats(evq_model, input_ids, tier_cfg)
    del evq_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Generate figures ─────────────────────────────────────────────
    print("\n=== Generating figures ===")

    plot_attention_weight_vs_distance(
        geo_collector, evq_collector,
        os.path.join(args.output_dir, "attn_weight_vs_distance.pdf"),
        args.tier)

    plot_head_avg_distance(
        geo_collector, evq_collector,
        os.path.join(args.output_dir, "attn_head_avg_distance.pdf"),
        args.tier)

    plot_distance_heatmap(
        geo_collector, evq_collector,
        os.path.join(args.output_dir, "attn_distance_heatmap.pdf"),
        args.tier)

    plot_layer_aggregate_comparison(
        geo_collector, evq_collector,
        os.path.join(args.output_dir, "attn_aggregate_comparison.pdf"),
        args.tier)

    # ── Save raw statistics for later use ────────────────────────────
    stats_path = os.path.join(args.output_dir, "attention_stats.npz")
    np.savez_compressed(
        stats_path,
        geo_distance_hist=geo_collector.distance_hist,
        evq_distance_hist=evq_collector.distance_hist,
        geo_avg_distance=geo_collector.get_avg_distance(),
        evq_avg_distance=evq_collector.get_avg_distance(),
        bin_edges=geo_collector.bin_edges,
        bin_centers=geo_collector.bin_centers,
    )
    print(f"  Raw stats saved: {stats_path}")

    # ── Summary ──────────────────────────────────────────────────────
    geo_avg = geo_collector.get_avg_distance()
    evq_avg = evq_collector.get_avg_distance()
    print("\n" + "=" * 64)
    print("SUMMARY: Per-Layer Average Attention Distance")
    print("=" * 64)
    print(f"{'Layer':<8} {'Geo':>10} {'EVQ':>10} {'Δ':>10} {'EVQ/Geo':>10}")
    print("-" * 48)
    for layer_idx in range(tier_cfg["num_layers"]):
        g = geo_avg[layer_idx].mean()
        e = evq_avg[layer_idx].mean()
        delta = e - g
        ratio = e / max(g, 1e-12)
        print(f"  L{layer_idx:<4}  {g:>10.1f} {e:>10.1f} {delta:>+10.1f} {ratio:>10.2f}x")
    g_all = geo_avg.mean()
    e_all = evq_avg.mean()
    print("-" * 48)
    print(f"  {'ALL':<6} {g_all:>10.1f} {e_all:>10.1f} {e_all - g_all:>+10.1f} {e_all / max(g_all, 1e-12):>10.2f}x")
    print("=" * 64)

    print(f"\nAll figures saved to: {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
