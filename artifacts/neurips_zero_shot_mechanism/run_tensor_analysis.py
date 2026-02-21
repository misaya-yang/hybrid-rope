#!/usr/bin/env python3
"""
Zero-shot tensor interference analysis for phase collision at very long context.

Outputs (all under artifacts/neurips_zero_shot_mechanism/):
- phase_collision_comparison.pdf
- phase_collision_comparison.png
- results.json
- run.log
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import seaborn as sns
import torch

# Windows + MKL/OpenMP duplicate runtime self-heal.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent
RUN_LOG = OUTPUT_DIR / "run.log"
RESULT_JSON = OUTPUT_DIR / "results.json"
FIG_PDF = OUTPUT_DIR / "phase_collision_comparison.pdf"
FIG_PNG = OUTPUT_DIR / "phase_collision_comparison.png"


def setup_logger() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tensor_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(RUN_LOG, encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def select_device_and_dtype(logger: logging.Logger) -> Tuple[torch.device, torch.dtype, str]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        logger.info("Using CUDA with torch.bfloat16 as requested.")
        return device, dtype, "cuda_bfloat16"

    logger.warning("CUDA not available in current environment. Falling back to CPU.")
    try:
        _a = torch.randn(8, dtype=torch.bfloat16)
        _b = torch.randn(8, 8, dtype=torch.bfloat16)
        _ = torch.mv(_b, _a)
        logger.info("CPU bfloat16 path is available; using bfloat16 on CPU.")
        return torch.device("cpu"), torch.bfloat16, "cpu_bfloat16"
    except Exception:
        logger.warning("CPU bfloat16 unsupported for required ops. Falling back to float32.")
        return torch.device("cpu"), torch.float32, "cpu_float32"


def standard_inv_freq(d: int, base: float = 10000.0) -> torch.Tensor:
    half = d // 2
    i = torch.arange(half, dtype=torch.float32)
    # theta_i = b^(-2i/d)
    return torch.pow(torch.tensor(base, dtype=torch.float32), -2.0 * i / float(d))


def analytic_sigmoid_inv_freq(d: int, target_length: int, base: float = 10000.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    half = d // 2
    i = torch.arange(half, dtype=torch.float32)
    x0 = float(d) / 4.0
    k = 6.0 * math.log(float(target_length)) / float(d)

    left = torch.sigmoid(torch.tensor(-k * x0, dtype=torch.float32))
    right = torch.sigmoid(torch.tensor(k * (half - 1 - x0), dtype=torch.float32))
    denom = right - left
    if float(denom.item()) == 0.0:
        raise ValueError("Invalid sigmoid mapping denominator == 0.")

    s_tilde = (torch.sigmoid(k * (i - x0)) - left) / denom
    inv_freq = torch.pow(torch.tensor(base, dtype=torch.float32), -s_tilde)
    params = {"x0": x0, "k": float(k)}
    return inv_freq, params


def apply_rope(q: torch.Tensor, k: torch.Tensor, inv_freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q/k: [L, d]
    inv_freq: [d/2]
    """
    device = q.device
    L, d = q.shape
    half = d // 2

    pos = torch.arange(L, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq.to(device=device, dtype=torch.float32))  # [L, d/2]
    cos = freqs.cos().to(dtype=q.dtype)
    sin = freqs.sin().to(dtype=q.dtype)

    q_even, q_odd = q[:, 0::2], q[:, 1::2]
    k_even, k_odd = k[:, 0::2], k[:, 1::2]

    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)
    q_rot[:, 0::2] = q_even * cos - q_odd * sin
    q_rot[:, 1::2] = q_even * sin + q_odd * cos
    k_rot[:, 0::2] = k_even * cos - k_odd * sin
    k_rot[:, 1::2] = k_even * sin + k_odd * cos

    assert q_rot.shape == (L, d) and k_rot.shape == (L, d) and half * 2 == d

    del pos, freqs, cos, sin, q_even, q_odd, k_even, k_odd
    return q_rot, k_rot


def build_distance_bin_map(length: int, num_bins: int, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    edges_np = np.logspace(0, math.log10(length), num_bins + 1, dtype=np.float64)
    edges = torch.tensor(edges_np, dtype=torch.float32, device=device)

    distances = torch.arange(length + 1, dtype=torch.float32, device=device)
    # bucket indices in [0, num_bins-1] for distance >= 1
    bin_idx = torch.bucketize(distances, edges, right=True) - 1
    bin_idx = torch.clamp(bin_idx, min=0, max=num_bins - 1)
    return bin_idx.to(torch.long), edges_np


def uniform_query_indices(length: int, samples: int) -> torch.Tensor:
    idx = torch.linspace(0, length - 1, steps=samples, dtype=torch.float64)
    idx = torch.round(idx).to(torch.long).unique(sorted=True)
    return idx


def phase_collision_curve(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    query_indices: torch.Tensor,
    distance_to_bin: torch.Tensor,
    num_bins: int,
    logger: logging.Logger,
) -> np.ndarray:
    device = q_rot.device
    L, _ = q_rot.shape
    positions = torch.arange(L, device=device, dtype=torch.long)

    counts = torch.zeros(num_bins, dtype=torch.float64)
    sums = torch.zeros(num_bins, dtype=torch.float64)
    sums_sq = torch.zeros(num_bins, dtype=torch.float64)

    t0 = time.time()
    total = int(query_indices.numel())
    for i, m in enumerate(query_indices.tolist(), start=1):
        q_vec = q_rot[m]  # [d]
        logits = torch.mv(k_rot, q_vec).to(torch.float32)  # [L]

        dist = torch.abs(positions - int(m))
        valid = dist >= 1
        bins = distance_to_bin[dist[valid]]
        vals = logits[valid]

        c = torch.bincount(bins, minlength=num_bins).to(torch.float64).cpu()
        s = torch.bincount(bins, weights=vals, minlength=num_bins).to(torch.float64).cpu()
        ss = torch.bincount(bins, weights=vals * vals, minlength=num_bins).to(torch.float64).cpu()

        counts += c
        sums += s
        sums_sq += ss

        del q_vec, logits, dist, valid, bins, vals, c, s, ss
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if i % 100 == 0 or i == total:
            logger.info("Processed %d / %d queries (%.1f%%)", i, total, 100.0 * i / total)

    logger.info("Curve accumulation finished in %.2fs", time.time() - t0)

    eps = torch.tensor(1e-12, dtype=torch.float64)
    mean = sums / torch.clamp(counts, min=1.0)
    var = (sums_sq / torch.clamp(counts, min=1.0)) - (mean * mean)
    var = torch.maximum(var, eps)
    var[counts < 1] = float("nan")
    return var.numpy()


def plot_curves(
    centers: np.ndarray,
    std_curve: np.ndarray,
    sig_curve: np.ndarray,
    out_pdf: Path,
    out_png: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
        }
    )
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(8.6, 5.1), dpi=180)
    ax.plot(centers, std_curve, color="#c44e52", lw=2.2, label="Standard RoPE")
    ax.plot(centers, sig_curve, color="#2b6cb0", lw=2.2, label="Analytic Sigmoid-RoPE")

    ax.set_xscale("log")
    ax.set_xlabel(r"Relative Distance $|m-n|$")
    ax.set_ylabel("Attention Logit Variance (Phase Collision Index)")
    ax.set_title("Zero-shot Phase-Collision Analysis at $L=131072$")
    ax.legend(loc="best", frameon=True)

    # Annotate spike region for standard RoPE beyond 32k distance.
    mask = (centers > 32000) & np.isfinite(std_curve)
    if np.any(mask):
        x_region = centers[mask]
        y_region = std_curve[mask]
        j = int(np.nanargmax(y_region))
        x_spike = float(x_region[j])
        y_spike = float(y_region[j])
        ax.annotate(
            "Phase Collision Spikes",
            xy=(x_spike, y_spike),
            xytext=(x_spike * 0.43, y_spike * 1.12),
            arrowprops=dict(arrowstyle="->", color="#7f1d1d", lw=1.2),
            fontsize=10,
            color="#7f1d1d",
        )

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_attempt(
    logger: logging.Logger,
    device: torch.device,
    dtype: torch.dtype,
    sample_count: int,
    length: int = 131072,
    d: int = 128,
    base: float = 10000.0,
    num_bins: int = 100,
    seed: int = 20260221,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(
        "Attempt with sample_count=%d, L=%d, d=%d, bins=%d, dtype=%s, device=%s",
        sample_count,
        length,
        d,
        num_bins,
        str(dtype),
        str(device),
    )

    # Required shape [1, L, d], then squeeze to [L, d] for compute.
    q = torch.randn((1, length, d), dtype=dtype, device=device)
    k = torch.randn((1, length, d), dtype=dtype, device=device)
    q = q[0].contiguous()
    k = k[0].contiguous()

    std_inv = standard_inv_freq(d=d, base=base)
    sig_inv, sig_params = analytic_sigmoid_inv_freq(d=d, target_length=length, base=base)
    query_idx = uniform_query_indices(length=length, samples=sample_count).to(device=device)
    dist2bin, edges = build_distance_bin_map(length=length, num_bins=num_bins, device=device)
    centers = np.sqrt(edges[:-1] * edges[1:])

    logger.info("Applying Standard RoPE...")
    q_std, k_std = apply_rope(q, k, std_inv)
    logger.info("Computing Standard RoPE variance curve...")
    std_curve = phase_collision_curve(
        q_rot=q_std,
        k_rot=k_std,
        query_indices=query_idx,
        distance_to_bin=dist2bin,
        num_bins=num_bins,
        logger=logger,
    )
    del q_std, k_std
    clear_cuda()

    logger.info("Applying Analytic Sigmoid-RoPE...")
    q_sig, k_sig = apply_rope(q, k, sig_inv)
    logger.info("Computing Analytic Sigmoid-RoPE variance curve...")
    sig_curve = phase_collision_curve(
        q_rot=q_sig,
        k_rot=k_sig,
        query_indices=query_idx,
        distance_to_bin=dist2bin,
        num_bins=num_bins,
        logger=logger,
    )
    del q_sig, k_sig, q, k, query_idx, dist2bin
    clear_cuda()

    plot_curves(
        centers=centers,
        std_curve=std_curve,
        sig_curve=sig_curve,
        out_pdf=FIG_PDF,
        out_png=FIG_PNG,
    )

    return {
        "config": {
            "length": length,
            "d": d,
            "base": base,
            "num_bins": num_bins,
            "sample_count_used": int(sample_count),
            "seed": seed,
            "device": str(device),
            "dtype": str(dtype),
        },
        "sigmoid_params": sig_params,
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
        "variance_standard": std_curve.tolist(),
        "variance_analytic_sigmoid": sig_curve.tolist(),
        "outputs": {
            "pdf": str(FIG_PDF),
            "png": str(FIG_PNG),
        },
    }


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda oom" in msg) or ("cublas" in msg and "alloc" in msg)


def main() -> None:
    logger = setup_logger()
    started = time.time()
    logger.info("=== Zero-shot Mechanistic Tensor Analysis START ===")

    device, dtype, runtime_mode = select_device_and_dtype(logger)
    sample_plan: List[int] = [1000, 768, 512, 384, 256, 192, 128]
    result_payload: Dict | None = None
    last_error = None

    for sample_count in sample_plan:
        try:
            logger.info("Running attempt with sample_count=%d", sample_count)
            result_payload = run_single_attempt(
                logger=logger,
                device=device,
                dtype=dtype,
                sample_count=sample_count,
                length=131072,
                d=128,
                base=10000.0,
                num_bins=100,
                seed=20260221,
            )
            logger.info("Attempt succeeded with sample_count=%d", sample_count)
            break
        except RuntimeError as exc:
            last_error = exc
            if is_oom_error(exc):
                logger.warning("OOM detected at sample_count=%d; retrying with smaller sample.", sample_count)
                clear_cuda()
                continue
            logger.error("RuntimeError (non-OOM): %s", exc)
            logger.error(traceback.format_exc())
            raise
        except Exception as exc:
            last_error = exc
            logger.error("Unexpected error: %s", exc)
            logger.error(traceback.format_exc())
            raise

    if result_payload is None:
        raise RuntimeError(f"All retry attempts failed. Last error: {last_error}")

    finished = time.time()
    result_payload["runtime"] = {
        "mode": runtime_mode,
        "started_at_unix": started,
        "finished_at_unix": finished,
        "elapsed_sec": finished - started,
    }
    RESULT_JSON.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Saved %s", RESULT_JSON)
    logger.info("Saved %s", FIG_PDF)
    logger.info("Saved %s", FIG_PNG)
    logger.info("Total elapsed: %.2fs", finished - started)
    logger.info("=== Zero-shot Mechanistic Tensor Analysis DONE ===")


if __name__ == "__main__":
    main()
