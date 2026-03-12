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
  4. attn_aggregate_comparison.pdf — Single-panel summary

Usage:
  python visualize_attention_distance.py \
      --geo_ckpt  /path/to/geo_750m_checkpoint.pt \
      --evq_ckpt  /path/to/evq_750m_checkpoint.pt \
      --val_pt    /path/to/val_fineweb-edu_5000000.pt \
      --num_samples 16 \
      --tier 750m --tau 1.5 --seq_len 8192 \
      --output_dir ./figures/attention_viz
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
                 seq_len: int, num_bins: int = 128):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_len = seq_len

        # Uniform-in-log bins: equal width on log scale
        self.bin_edges = np.geomspace(1, seq_len, num_bins + 1)
        self.bin_edges = np.concatenate([[0], self.bin_edges])  # prepend 0 for distance=0
        self.num_bins = len(self.bin_edges) - 1
        self.bin_centers = np.sqrt(
            np.maximum(self.bin_edges[:-1], 0.5) * self.bin_edges[1:]
        )  # geometric mean for log-spaced centers

        # Accumulators: [num_layers, num_heads, num_bins]
        self.distance_hist = np.zeros((num_layers, num_heads, self.num_bins), dtype=np.float64)
        # Per-head average distance: [num_layers, num_heads]
        self.weighted_dist_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
        self.weight_sum = np.zeros((num_layers, num_heads), dtype=np.float64)

        self._hooks = []
        self.num_samples_accumulated = 0

    def register_hooks(self, model: GPT):
        """Register forward hooks on all Attention modules."""
        for layer_idx, block in enumerate(model.blocks):
            hook = self._make_hook(layer_idx, block.attn)
            h = block.attn.register_forward_hook(hook)
            self._hooks.append(h)

    def _make_hook(self, layer_idx: int, attn_module: Attention):
        """Create a hook that computes attention weights and accumulates stats.

        Optimizations:
          - All histogram computation on GPU via torch.bucketize + scatter_add
          - No Python loop over heads (fully vectorized)
          - Single CPU transfer per layer
        """
        collector = self
        bin_edges_t = torch.from_numpy(self.bin_edges.astype(np.float32))
        num_bins = self.num_bins

        def hook_fn(module, inputs, output):
            x = inputs[0]  # (B, L, H)
            B, L, _ = x.shape
            nh = module.nh
            hd = module.hd

            with torch.no_grad():
                dev = x.device

                # Recompute QKV + RoPE (needed because SDPA doesn't expose weights)
                qkv = module.qkv(x).view(B, L, 3, nh, hd).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]  # (B, nh, L, hd)
                del qkv
                cos, sin = module.rope(L)
                cos, sin = cos[None, None], sin[None, None]
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)

                scale = 1.0 / math.sqrt(hd)
                bin_edges_dev = bin_edges_t.to(dev)

                # GPU accumulators for this layer
                hist_gpu = torch.zeros(nh, num_bins, device=dev, dtype=torch.float64)
                wdist_gpu = torch.zeros(nh, device=dev, dtype=torch.float64)
                wsum_gpu = torch.zeros(nh, device=dev, dtype=torch.float64)

                chunk_size = min(1024, L)
                for q_start in range(0, L, chunk_size):
                    q_end = min(q_start + chunk_size, L)
                    q_chunk = q[:, :, q_start:q_end, :]  # (B, nh, chunk, hd)

                    scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale

                    q_pos = torch.arange(q_start, q_end, device=dev)
                    k_pos = torch.arange(L, device=dev)
                    causal_mask = k_pos[None, :] > q_pos[:, None]
                    scores.masked_fill_(causal_mask[None, None], float('-inf'))

                    # (B, nh, chunk, L) → average over batch → (nh, chunk, L)
                    weights = F.softmax(scores, dim=-1).float().mean(dim=0)
                    del scores

                    distances = (q_pos[:, None] - k_pos[None, :]).clamp(min=0).float()  # (chunk, L)

                    # ── Vectorized histogram: all heads at once via scatter_add ──
                    bin_idx = torch.bucketize(distances, bin_edges_dev) - 1
                    bin_idx = bin_idx.clamp(0, num_bins - 1).long()  # (chunk, L)

                    # Expand bin_idx to (nh, chunk*L) and flatten weights
                    bin_flat = bin_idx.unsqueeze(0).expand(nh, -1, -1).reshape(nh, -1)
                    w_flat = weights.reshape(nh, -1).double()
                    hist_gpu.scatter_add_(1, bin_flat, w_flat)

                    # Weighted distance sum (vectorized, no head loop)
                    d_exp = distances.unsqueeze(0).expand(nh, -1, -1).double()
                    wdist_gpu += (weights.double() * d_exp).sum(dim=(1, 2))
                    wsum_gpu += weights.double().sum(dim=(1, 2))

                # Single CPU transfer per layer
                collector.distance_hist[layer_idx] += hist_gpu.cpu().numpy()
                collector.weighted_dist_sum[layer_idx] += wdist_gpu.cpu().numpy()
                collector.weight_sum[layer_idx] += wsum_gpu.cpu().numpy()

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

    def get_density_hist(self):
        """Density histogram: normalized AND divided by bin width (for proper PDF).
        This corrects for unequal bin widths in log-spaced bins."""
        normalized = self.get_normalized_hist()
        bin_widths = np.diff(self.bin_edges)
        return normalized / bin_widths[None, None, :]


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

def prepare_inputs(seq_len: int, device, val_pt=None, num_samples=16):
    """
    Prepare multiple input segments for robust attention analysis.

    If val_pt is provided, loads pre-tokenized data and splits into segments.
    Otherwise falls back to random tokens.

    Returns: list of (1, seq_len) tensors
    """
    if val_pt and os.path.exists(val_pt):
        print(f"  Loading pre-tokenized data from {val_pt}")
        data = torch.load(val_pt, map_location="cpu")
        if isinstance(data, dict):
            data = data.get("val", data.get("data", next(iter(data.values()))))
        data = data.flatten()
        total_tokens = data.shape[0]
        max_segments = total_tokens // seq_len
        num_samples = min(num_samples, max_segments)
        print(f"  {total_tokens} tokens → {max_segments} possible segments, using {num_samples}")

        segments = []
        for i in range(num_samples):
            start = i * seq_len
            seg = data[start:start + seq_len].unsqueeze(0).to(device)
            segments.append(seg)
        return segments

    # Fallback: try streaming from HF
    try:
        from transformers import AutoTokenizer
        from datasets import load_dataset
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        print("  Loading from FineWeb-Edu (streaming)...")
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                          split="train", streaming=True)
        all_ids = []
        for sample in ds:
            ids = tokenizer.encode(sample["text"], add_special_tokens=False)
            all_ids.extend(ids)
            if len(all_ids) >= seq_len * num_samples:
                break
        all_ids = all_ids[:seq_len * num_samples]
        segments = []
        for i in range(num_samples):
            seg = torch.tensor(all_ids[i * seq_len:(i + 1) * seq_len],
                               dtype=torch.long, device=device).unsqueeze(0)
            segments.append(seg)
        print(f"  Prepared {len(segments)} segments from FineWeb-Edu")
        return segments
    except Exception as e:
        print(f"  Could not load FineWeb-Edu: {e}")

    # Final fallback: random tokens
    print(f"  Using random token IDs ({num_samples} segments)")
    segments = []
    for _ in range(num_samples):
        seg = torch.randint(100, 50000, (1, seq_len), device=device)
        segments.append(seg)
    return segments


# ── Smoothing utility ────────────────────────────────────────────────

def smooth_curve(y, sigma=2.0):
    """Gaussian smoothing for 1D curves."""
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(y, sigma=sigma)
    except ImportError:
        # Manual Gaussian smoothing fallback
        kernel_size = int(sigma * 4) * 2 + 1
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return np.convolve(y, kernel, mode='same')


# ── Visualization ────────────────────────────────────────────────────

def plot_attention_weight_vs_distance(geo_collector, evq_collector, output_path, tier,
                                       sigma=2.0):
    """
    Figure 1: Attention weight vs distance curve.
    One subplot per layer, EVQ vs Geo lines overlaid, with smoothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_layers = geo_collector.num_layers
    geo_density = geo_collector.get_density_hist().mean(axis=1)  # (L, bins)
    evq_density = evq_collector.get_density_hist().mean(axis=1)
    bin_centers = geo_collector.bin_centers

    cols = 3
    rows = math.ceil(num_layers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows),
                              sharex=True, sharey=False)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for layer_idx in range(num_layers):
        ax = axes_flat[layer_idx]
        geo_y = smooth_curve(geo_density[layer_idx], sigma)
        evq_y = smooth_curve(evq_density[layer_idx], sigma)
        ax.plot(bin_centers, geo_y, color="#2196F3",
                alpha=0.85, linewidth=1.5, label="Geometric")
        ax.plot(bin_centers, evq_y, color="#FF5722",
                alpha=0.85, linewidth=1.5, label="EVQ-Cosh")
        ax.set_xscale("log")
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.tick_params(labelsize=8)
        if layer_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    for idx in range(num_layers, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.supxlabel("Attention Distance (tokens)", fontsize=12)
    fig.supylabel("Attention Density", fontsize=12)
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
    num_heads = geo_avg.shape[1]

    # Count heads above/below diagonal
    above = np.sum(evq_avg > geo_avg)
    total = geo_avg.size

    # Color by layer
    cmap = plt.cm.viridis
    colors = [cmap(i / max(num_layers - 1, 1)) for i in range(num_layers)]

    fig, ax = plt.subplots(figsize=(7, 7))

    for layer_idx in range(num_layers):
        ax.scatter(geo_avg[layer_idx], evq_avg[layer_idx],
                   c=[colors[layer_idx]] * num_heads,
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
                 f"({above}/{total} heads: EVQ attends farther)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_distance_heatmap(geo_collector, evq_collector, output_path, tier):
    """
    Figure 3: Distance heatmap (layer × distance).
    Side-by-side: Geo | EVQ | Difference.
    X-axis: actual distance (log-scale), Y-axis: layer.
    Uses pcolormesh with log color scale to reveal long-range structure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm, SymLogNorm
    import matplotlib.ticker as mticker

    # Use raw normalized histogram (not density) for heatmap —
    # pcolormesh already renders each bin at its true log-width
    geo_hist = geo_collector.get_normalized_hist().mean(axis=1)  # (L, bins)
    evq_hist = evq_collector.get_normalized_hist().mean(axis=1)
    diff = evq_hist - geo_hist
    bin_edges = geo_collector.bin_edges
    num_layers = geo_hist.shape[0]

    x_edges = bin_edges.copy()
    x_edges[0] = max(x_edges[0], 0.5)
    y_edges = np.arange(num_layers + 1) - 0.5

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Log color scale reveals both short and long-range patterns
    vmin_log = 1e-4
    vmax_log = max(geo_hist.max(), evq_hist.max())
    for ax_idx, (data, title, cmap_name) in enumerate([
        (geo_hist, "Geometric", "Blues"),
        (evq_hist, "EVQ-Cosh", "Oranges"),
    ]):
        ax = axes[ax_idx]
        data_clipped = np.clip(data, vmin_log, None)
        im = ax.pcolormesh(x_edges, y_edges, data_clipped, cmap=cmap_name,
                           norm=LogNorm(vmin=vmin_log, vmax=vmax_log),
                           shading='flat')
        ax.set_xscale("log")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Attention Distance (tokens)", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Layer", fontsize=10)
        ax.set_yticks(range(num_layers))
        plt.colorbar(im, ax=ax, shrink=0.8, label="Attn Weight (log)")

    # Difference: symmetric log scale to show both positive/negative and small values
    ax = axes[2]
    abs_max = max(abs(diff.min()), abs(diff.max()))
    if abs_max < 1e-12:
        abs_max = 1.0
    norm = SymLogNorm(linthresh=1e-4, vmin=-abs_max, vmax=abs_max)
    im = ax.pcolormesh(x_edges, y_edges, diff, cmap="RdBu_r", norm=norm,
                       shading='flat')
    ax.set_xscale("log")
    ax.set_title("Δ (EVQ − Geo)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Attention Distance (tokens)", fontsize=10)
    ax.set_yticks(range(num_layers))
    plt.colorbar(im, ax=ax, shrink=0.8, label="Δ Weight (symlog)")

    fig.suptitle(f"Attention Distance Distribution Heatmap — {tier.upper()} (head-averaged)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_layer_aggregate_comparison(geo_collector, evq_collector, output_path, tier,
                                     sigma=2.0):
    """
    Figure 4: Two-panel summary.
    Top: attention density vs distance (smoothed, log-x).
    Bottom: EVQ/Geo ratio — makes the long-range advantage clearly visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    geo_density = geo_collector.get_density_hist().mean(axis=(0, 1))  # (bins,)
    evq_density = evq_collector.get_density_hist().mean(axis=(0, 1))
    bin_centers = geo_collector.bin_centers

    geo_smooth = smooth_curve(geo_density, sigma)
    evq_smooth = smooth_curve(evq_density, sigma)

    # Compute ratio (EVQ / Geo), safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(geo_smooth > 1e-15,
                         evq_smooth / geo_smooth, 1.0)
    ratio_smooth = smooth_curve(ratio, sigma)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

    # ── Top panel: density curves ──
    ax1.plot(bin_centers, geo_smooth, color="#2196F3", linewidth=2, label="Geometric")
    ax1.plot(bin_centers, evq_smooth, color="#FF5722", linewidth=2, label="EVQ-Cosh")

    ax1.fill_between(bin_centers, geo_smooth, evq_smooth,
                      where=(evq_smooth > geo_smooth), alpha=0.15, color="#FF5722",
                      label="EVQ > Geo")
    ax1.fill_between(bin_centers, geo_smooth, evq_smooth,
                      where=(geo_smooth > evq_smooth), alpha=0.15, color="#2196F3",
                      label="Geo > EVQ")

    # Find and annotate crossover point
    diff = evq_smooth - geo_smooth
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    cross_dist = None
    for idx in sign_changes:
        if bin_centers[idx] > 2:
            cross_dist = bin_centers[idx]
            ax1.axvline(cross_dist, color="gray", linestyle=":", alpha=0.5, linewidth=1)
            ax1.annotate(f"crossover ≈{cross_dist:.0f} tok",
                        xy=(cross_dist, geo_smooth[idx]),
                        xytext=(cross_dist * 3, max(geo_smooth) * 0.6),
                        fontsize=9, color="gray",
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))
            break

    ax1.set_xscale("log")
    ax1.set_ylabel("Attention Density", fontsize=11)
    ax1.set_title(f"Aggregate Attention Distance Distribution — {tier.upper()}\n"
                  f"(averaged over all layers, heads, and {geo_collector.num_samples_accumulated} samples)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.2)

    # ── Bottom panel: ratio ──
    ax2.plot(bin_centers, ratio_smooth, color="#333333", linewidth=1.5)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax2.fill_between(bin_centers, 1.0, ratio_smooth,
                      where=(ratio_smooth > 1.0), alpha=0.2, color="#FF5722")
    ax2.fill_between(bin_centers, 1.0, ratio_smooth,
                      where=(ratio_smooth < 1.0), alpha=0.2, color="#2196F3")

    if cross_dist is not None:
        ax2.axvline(cross_dist, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    ax2.set_ylabel("EVQ / Geo Ratio", fontsize=11)
    ax2.set_xlabel("Attention Distance (tokens)", fontsize=11)
    ax2.set_ylim(0.4, 2.5)
    ax2.grid(True, alpha=0.2)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def collect_attention_stats(model, input_segments, tier_cfg):
    """Run forward passes over multiple segments and accumulate attention statistics."""
    num_layers = tier_cfg["num_layers"]
    num_heads = tier_cfg["num_heads"]
    seq_len = input_segments[0].shape[1]

    collector = AttentionStatsCollector(num_layers, num_heads, seq_len)
    collector.register_hooks(model)

    total_samples = len(input_segments)
    t0 = time.time()
    for i, seg in enumerate(input_segments):
        with torch.no_grad():
            _ = model(seg)
        collector.num_samples_accumulated += 1
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (total_samples - i - 1)
        print(f"    sample {i+1}/{total_samples}  ({elapsed:.1f}s elapsed, ETA {eta:.0f}s)")

    t1 = time.time()
    print(f"  All {total_samples} forward passes done in {t1 - t0:.1f}s")

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
    parser.add_argument("--val_pt", type=str, default=None,
                        help="Path to pre-tokenized validation .pt file (flat 1D tensor)")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of text segments to average over (default: 16)")
    parser.add_argument("--output_dir", type=str, default="./figures/attention_viz",
                        help="Output directory for figures")
    parser.add_argument("--num_bins", type=int, default=128,
                        help="Number of distance histogram bins (default: 128)")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian smoothing sigma for curves (default: 2.0)")
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

    # ── Prepare input segments ────────────────────────────────────────
    print("\n=== Preparing input segments ===")
    input_segments = prepare_inputs(args.seq_len, device, args.val_pt, args.num_samples)
    print(f"  {len(input_segments)} segments × {args.seq_len} tokens")

    # ── Collect Geometric attention stats ────────────────────────────
    print("\n=== Loading Geometric model ===")
    geo_model = load_model(
        args.geo_ckpt, args.tier, "geo", args.tau, args.base, args.seq_len, device)
    if compute_dtype != torch.float32:
        geo_model = geo_model.to(dtype=compute_dtype)

    print("\n=== Collecting Geometric attention statistics ===")
    geo_collector = collect_attention_stats(geo_model, input_segments, tier_cfg)
    del geo_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Collect EVQ attention stats ──────────────────────────────────
    print("\n=== Loading EVQ model ===")
    evq_model = load_model(
        args.evq_ckpt, args.tier, "evq", args.tau, args.base, args.seq_len, device)
    if compute_dtype != torch.float32:
        evq_model = evq_model.to(dtype=compute_dtype)

    print("\n=== Collecting EVQ attention statistics ===")
    evq_collector = collect_attention_stats(evq_model, input_segments, tier_cfg)
    del evq_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Generate figures ─────────────────────────────────────────────
    print("\n=== Generating figures ===")

    plot_attention_weight_vs_distance(
        geo_collector, evq_collector,
        os.path.join(args.output_dir, "attn_weight_vs_distance.pdf"),
        args.tier, sigma=args.sigma)

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
        args.tier, sigma=args.sigma)

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
        num_samples=geo_collector.num_samples_accumulated,
    )
    print(f"  Raw stats saved: {stats_path}")

    # ── Summary ──────────────────────────────────────────────────────
    geo_avg = geo_collector.get_avg_distance()
    evq_avg = evq_collector.get_avg_distance()
    print("\n" + "=" * 64)
    print(f"SUMMARY: Per-Layer Average Attention Distance ({geo_collector.num_samples_accumulated} samples)")
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
