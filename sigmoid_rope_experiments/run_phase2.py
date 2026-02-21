#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import hilbert
from tqdm import tqdm

from src.fitting import choose_best_fit, fit_k_models, fit_x0_models
from src.grid_search import GridSearchConfig, _build_sigmoid_freqs_batch, run_grid_search
from src.metrics import compute_attention_score_decay, compute_phase_collision_curve, phase_collision_score, phase_collision_score_batch
from src.rope import RoPEFrequencyAllocator
from src.utils import cleanup_cuda, env_info, get_device, load_json, save_json, set_seed
from src.visualization import save_fig_both, set_plot_style

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def ensure_dependencies() -> None:
    required = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
        ("seaborn", "seaborn"),
    ]
    missing: List[Tuple[str, str]] = []
    for mod, pip_name in required:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append((mod, pip_name))
    if not missing:
        return
    print("[deps] missing:", ", ".join(m for m, _ in missing))
    for _, pip_name in missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sigmoid-RoPE phase2: fine search + formula refit.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="Force CPU mode.")
    ap.add_argument("--fine_num_samples", type=int, default=5000)
    ap.add_argument("--max_hours_threshold", type=float, default=3.0)
    return ap.parse_args()


def ensure_coarse_centers(root_dir: Path, device: torch.device) -> pd.DataFrame:
    data_dir = root_dir / "data"
    coarse_csv = data_dir / "grid_search_results.csv"
    if coarse_csv.exists():
        df = pd.read_csv(coarse_csv)
        if not df.empty and {"d", "L", "k_optimal", "x0_optimal", "score_optimal"}.issubset(df.columns):
            centers = (
                df.sort_values(["d", "L", "score_optimal"])
                .groupby(["d", "L"], as_index=False)
                .first()[["d", "L", "k_optimal", "x0_optimal", "score_optimal"]]
            )
            expected = {(d, L) for d in [64, 128, 256] for L in [4096, 8192, 16384, 32768, 65536, 131072]}
            got = {(int(r.d), int(r.L)) for r in centers.itertuples()}
            if got == expected:
                print(f"[phase2] Loaded coarse centers from {coarse_csv}")
                return centers

    print("[phase2] coarse centers missing/incomplete, running bootstrap coarse search ...")
    cfg = GridSearchConfig(
        mode="coarse",
        coarse_k_step=0.01,
        coarse_x0_step=2.0,
        num_samples_coarse=1500,
        checkpoint_path=str(data_dir / "grid_search_checkpoint_bootstrap.json"),
        csv_path=str(data_dir / "grid_search_results.csv"),
    )
    df = run_grid_search(cfg, device=device)
    centers = (
        df.sort_values(["d", "L", "score_optimal"])
        .groupby(["d", "L"], as_index=False)
        .first()[["d", "L", "k_optimal", "x0_optimal", "score_optimal"]]
    )
    return centers


def _fine_search_one_config(
    d: int,
    L: int,
    center_k: float,
    center_x0: float,
    device: torch.device,
    num_samples: int,
    k_step: float,
    x0_step: float,
    base: float = 10000.0,
) -> Dict:
    allocator = RoPEFrequencyAllocator(d=d, base=base)
    n = d // 2

    k_lo = max(1e-4, float(center_k) - 0.05)
    k_hi = float(center_k) + 0.05
    x0_lo = max(0.0, float(center_x0) - 5.0)
    x0_hi = min(float(n - 1), float(center_x0) + 5.0)

    k_values = np.arange(k_lo, k_hi + 1e-12, k_step, dtype=np.float64)
    x0_values = np.arange(x0_lo, x0_hi + 1e-12, x0_step, dtype=np.float64)

    score_std, _ = phase_collision_score(
        freqs=allocator.standard(),
        L=L,
        d=d,
        num_samples=num_samples,
        device=device,
    )

    best_score = float("inf")
    best_k = None
    best_x0 = None
    for k in k_values:
        fb = _build_sigmoid_freqs_batch(d=d, base=base, k=float(k), x0_values=x0_values)
        scores, _ = phase_collision_score_batch(
            freqs_batch=fb,
            L=L,
            num_samples=num_samples,
            device=device,
        )
        idx = int(torch.argmin(scores).item())
        s = float(scores[idx].item())
        if s < best_score:
            best_score = s
            best_k = float(k)
            best_x0 = float(x0_values[idx])
        del fb, scores
        cleanup_cuda()

    return {
        "d": int(d),
        "L": int(L),
        "k_center": float(center_k),
        "x0_center": float(center_x0),
        "k_optimal": float(best_k),
        "x0_optimal": float(best_x0),
        "score_optimal": float(best_score),
        "score_standard": float(score_std),
        "k_step": float(k_step),
        "x0_step": float(x0_step),
        "num_samples": int(num_samples),
    }


def run_fine_search(root_dir: Path, centers: pd.DataFrame, device: torch.device, num_samples: int, max_hours_threshold: float) -> pd.DataFrame:
    data_dir = root_dir / "data"
    cp_path = data_dir / "fine_search_checkpoint.json"
    out_csv = data_dir / "fine_search_results.csv"
    cp = load_json(cp_path, default={"rows": [], "completed_keys": [], "profile": "fine"})
    rows: List[Dict] = list(cp.get("rows", []))
    completed = set(cp.get("completed_keys", []))

    work = centers.sort_values(["d", "L"]).to_dict(orient="records")
    k_step = 0.001
    x0_step = 0.5
    profile = "fine"
    first_timing_done = False
    start = time.time()

    pbar = tqdm(work, desc="FineSearch", dynamic_ncols=True)
    for item in pbar:
        d = int(item["d"])
        L = int(item["L"])
        key = f"d={d}|L={L}"
        if key in completed:
            continue

        t0 = time.time()
        result = _fine_search_one_config(
            d=d,
            L=L,
            center_k=float(item["k_optimal"]),
            center_x0=float(item["x0_optimal"]),
            device=device,
            num_samples=num_samples,
            k_step=k_step,
            x0_step=x0_step,
        )
        elapsed = time.time() - t0

        if not first_timing_done:
            remaining = len([w for w in work if f"d={int(w['d'])}|L={int(w['L'])}" not in completed]) - 1
            est_total = elapsed * max(1, remaining + 1)
            if est_total > max_hours_threshold * 3600:
                profile = "medium"
                k_step = 0.002
                x0_step = 1.0
                print(
                    f"[phase2] Estimated fine runtime {est_total/3600:.2f}h > {max_hours_threshold:.1f}h. "
                    "Switching remaining configs to medium granularity."
                )
            first_timing_done = True

        rows.append(result)
        completed.add(key)
        cp_payload = {
            "updated_at_unix": time.time(),
            "elapsed_sec": time.time() - start,
            "profile": profile,
            "rows": rows,
            "completed_keys": sorted(completed),
        }
        save_json(cp_path, cp_payload)
        pbar.set_postfix(
            d=d,
            L=L,
            k=result["k_optimal"],
            x0=result["x0_optimal"],
            score=f"{result['score_optimal']:.5f}",
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["d", "L"]).drop_duplicates(["d", "L"], keep="last").reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[phase2] Fine search done. Saved {out_csv}")
    return df


def fit_formula_models(root_dir: Path, fine_df: pd.DataFrame) -> Dict:
    data_dir = root_dir / "data"
    L_arr = fine_df["L"].to_numpy(dtype=np.float64)
    d_arr = fine_df["d"].to_numpy(dtype=np.float64)
    k_true = fine_df["k_optimal"].to_numpy(dtype=np.float64)
    x0_true = fine_df["x0_optimal"].to_numpy(dtype=np.float64)

    k_results = fit_k_models(L_arr=L_arr, d_arr=d_arr, y=k_true)
    x0_results = fit_x0_models(L_arr=L_arr, d_arr=d_arr, y=x0_true)

    best_k = choose_best_fit(k_results, min_r2=0.95)
    best_x0 = choose_best_fit(x0_results, min_r2=0.95)

    print("\n[phase2] k-model fitting results:")
    for r in k_results:
        print(
            f"  k-{r.name}: ok={r.ok} r2={r.r2:.6f} params={r.params} "
            f"mean_rel={r.mean_rel_error_pct:.2f}% max_rel={r.max_rel_error_pct:.2f}%"
        )
    print("[phase2] x0-model fitting results:")
    for r in x0_results:
        print(
            f"  x0-{r.name}: ok={r.ok} r2={r.r2:.6f} params={r.params} "
            f"mean_rel={r.mean_rel_error_pct:.2f}% max_rel={r.max_rel_error_pct:.2f}%"
        )

    payload = {
        "k_candidates": [r.to_dict() for r in k_results],
        "x0_candidates": [r.to_dict() for r in x0_results],
        "selected": {
            "k": best_k.to_dict(),
            "x0": best_x0.to_dict(),
        },
    }
    save_json(data_dir / "fitting_results.json", payload)
    save_json(
        data_dir / "formula_v2_params.json",
        {
            "k_form": best_k.name,
            "k_params": best_k.params,
            "x0_form": best_x0.name,
            "x0_params": best_x0.params,
        },
    )

    print("\n推荐公式:")
    if best_k.name == "A" and best_x0.name == "A":
        c1 = best_k.params["c1"]
        c3 = best_x0.params["c3"]
        print(
            f"  k = {c1:.4f} * ln(L) / d      "
            f"(R² = {best_k.r2:.4f}, max_error = {best_k.max_rel_error_pct:.1f}%)"
        )
        print(
            f"  x0 = {c3:.4f} * d             "
            f"(R² = {best_x0.r2:.4f}, max_error = {best_x0.max_rel_error_pct:.1f}%)"
        )
    else:
        print(
            f"  k-form={best_k.name}, params={best_k.params}, "
            f"R²={best_k.r2:.4f}, max_error={best_k.max_rel_error_pct:.1f}%"
        )
        print(
            f"  x0-form={best_x0.name}, params={best_x0.params}, "
            f"R²={best_x0.r2:.4f}, max_error={best_x0.max_rel_error_pct:.1f}%"
        )

    return payload


def _compute_formula_comparison(root_dir: Path, fine_df: pd.DataFrame, fit_payload: Dict, device: torch.device) -> pd.DataFrame:
    data_dir = root_dir / "data"
    best_k = fit_payload["selected"]["k"]
    best_x0 = fit_payload["selected"]["x0"]
    k_form = best_k["name"]
    k_params = best_k["params"]
    x0_form = best_x0["name"]
    x0_params = best_x0["params"]

    rows: List[Dict] = []
    for r in fine_df.sort_values(["d", "L"]).itertuples():
        allocator = RoPEFrequencyAllocator(d=int(r.d), base=10000.0)
        std_score = float(r.score_standard)

        f1, k1, x1 = allocator.sigmoid_analytical_v1(L=int(r.L))
        score_v1, _ = phase_collision_score(f1, L=int(r.L), d=int(r.d), num_samples=5000, device=device)

        f2, k2, x2 = allocator.sigmoid_analytical_v2(
            L=int(r.L),
            k_form=k_form,
            k_params=k_params,
            x0_form=x0_form,
            x0_params=x0_params,
        )
        score_v2, _ = phase_collision_score(f2, L=int(r.L), d=int(r.d), num_samples=5000, device=device)
        cleanup_cuda()

        rows.append(
            {
                "d": int(r.d),
                "L": int(r.L),
                "k_optimal": float(r.k_optimal),
                "x0_optimal": float(r.x0_optimal),
                "k_formula_v1": float(k1),
                "x0_formula_v1": float(x1),
                "k_formula_v2": float(k2),
                "x0_formula_v2": float(x2),
                "score_standard": std_score,
                "score_grid_opt": float(r.score_optimal),
                "score_formula_v1": float(score_v1),
                "score_formula_v2": float(score_v2),
            }
        )
    df = pd.DataFrame(rows).sort_values(["d", "L"]).reset_index(drop=True)
    df["k_error_pct_v2"] = (np.abs(df["k_formula_v2"] - df["k_optimal"]) / np.clip(np.abs(df["k_optimal"]), 1e-12, None)) * 100.0
    df["x0_error_pct_v2"] = (np.abs(df["x0_formula_v2"] - df["x0_optimal"]) / np.clip(np.abs(df["x0_optimal"]), 1e-12, None)) * 100.0
    df["improve_v2_pct"] = (df["score_standard"] - df["score_formula_v2"]) / np.clip(df["score_standard"], 1e-12, None) * 100.0
    df.to_csv(data_dir / "formula_comparison_v2.csv", index=False, encoding="utf-8")
    return df


def _plot_formula_validation_k_v2(df: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    colors = {64: "#1f77b4", 128: "#d62728", 256: "#2ca02c"}
    for d in sorted(df["d"].unique()):
        sub = df[df["d"] == d].sort_values("L")
        c = colors.get(int(d), None)
        ax.plot(sub["L"], sub["k_formula_v2"], color=c, linestyle="-", label=f"d={d} formula-v2")
        ax.scatter(sub["L"], sub["k_optimal"], color=c, marker="o", s=34, label=f"d={d} grid-opt")
    ax.set_xscale("log")
    ax.set_xlabel("L")
    ax.set_ylabel("k")
    ax.set_title("k: Formula-v2 Prediction vs Grid-Optimal")
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    save_fig_both(fig, out_dir / "formula_validation_k_v2")
    plt.close(fig)


def _plot_formula_validation_x0_v2(df: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    l_values = sorted(df["L"].unique())
    cmap = plt.get_cmap("viridis")
    for idx, L in enumerate(l_values):
        sub = df[df["L"] == L].sort_values("d")
        c = cmap(idx / max(1, len(l_values) - 1))
        ax.plot(sub["d"], sub["x0_formula_v2"], color=c, linestyle="-", label=f"L={L} formula-v2")
        ax.scatter(sub["d"], sub["x0_optimal"], color=c, marker="o", s=32, label=f"L={L} grid-opt")
    ax.set_xlabel("d")
    ax.set_ylabel("x0")
    ax.set_title("x0: Formula-v2 Prediction vs Grid-Optimal")
    ax.legend(ncol=2, fontsize=8, frameon=True)
    fig.tight_layout()
    save_fig_both(fig, out_dir / "formula_validation_x0_v2")
    plt.close(fig)


def _plot_score_compare_v2(df: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    dd = df.sort_values(["d", "L"]).copy()
    dd["cfg"] = dd.apply(lambda r: f"d{int(r['d'])}-L{int(r['L'])//1024}k", axis=1)
    x = np.arange(len(dd))
    w = 0.2
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.bar(x - 1.5 * w, dd["score_standard"], width=w, color="#d62728", label="Standard")
    ax.bar(x - 0.5 * w, dd["score_grid_opt"], width=w, color="#2ca02c", label="Sigmoid(Grid-Opt)")
    ax.bar(x + 0.5 * w, dd["score_formula_v1"], width=w, color="#9467bd", label="Sigmoid(Formula-v1)")
    ax.bar(x + 1.5 * w, dd["score_formula_v2"], width=w, color="#1f77b4", label="Sigmoid(Formula-v2)")
    ax.set_xticks(x)
    ax.set_xticklabels(dd["cfg"], rotation=45, ha="right")
    ax.set_ylabel("Phase Collision Score")
    ax.set_title("Score Comparison Across Configurations")
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    save_fig_both(fig, out_dir / "formula_vs_gridsearch_score_v2")
    plt.close(fig)


def _plot_formula_v1_v2_compare(df: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    dd = df.sort_values(["d", "L"]).copy()
    dd["cfg"] = dd.apply(lambda r: f"d{int(r['d'])}-L{int(r['L'])//1024}k", axis=1)
    x = np.arange(len(dd))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    ax.bar(x - w, dd["score_standard"], width=w, color="#d62728", label="Standard")
    ax.bar(x, dd["score_formula_v1"], width=w, color="#9467bd", label="Formula-v1")
    ax.bar(x + w, dd["score_formula_v2"], width=w, color="#1f77b4", label="Formula-v2")
    ax.set_xticks(x)
    ax.set_xticklabels(dd["cfg"], rotation=45, ha="right")
    ax.set_ylabel("Phase Collision Score")
    ax.set_title("Formula-v1 vs Formula-v2")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig_both(fig, out_dir / "formula_v1_vs_v2_comparison")
    plt.close(fig)


def _plot_fitting_quality(df: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.0))
    markers = {64: "o", 128: "s", 256: "^"}
    colors = {64: "#1f77b4", 128: "#d62728", 256: "#2ca02c"}

    ax = axes[0]
    for d in sorted(df["d"].unique()):
        sub = df[df["d"] == d]
        ax.scatter(sub["k_optimal"], sub["k_formula_v2"], label=f"d={d}", marker=markers[int(d)], color=colors[int(d)], s=50)
    min_k = min(df["k_optimal"].min(), df["k_formula_v2"].min())
    max_k = max(df["k_optimal"].max(), df["k_formula_v2"].max())
    ax.plot([min_k, max_k], [min_k, max_k], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("k_optimal")
    ax.set_ylabel("k_formula_v2")
    ax.set_title("k Fitting Quality")
    ax.legend(frameon=True)

    ax2 = axes[1]
    for d in sorted(df["d"].unique()):
        sub = df[df["d"] == d]
        ax2.scatter(sub["x0_optimal"], sub["x0_formula_v2"], label=f"d={d}", marker=markers[int(d)], color=colors[int(d)], s=50)
    min_x = min(df["x0_optimal"].min(), df["x0_formula_v2"].min())
    max_x = max(df["x0_optimal"].max(), df["x0_formula_v2"].max())
    ax2.plot([min_x, max_x], [min_x, max_x], linestyle="--", color="black", linewidth=1.0)
    ax2.set_xlabel("x0_optimal")
    ax2.set_ylabel("x0_formula_v2")
    ax2.set_title("x0 Fitting Quality")
    ax2.legend(frameon=True)

    fig.tight_layout()
    save_fig_both(fig, out_dir / "fitting_quality")
    plt.close(fig)


def _plot_phase_collision_v2(fit_payload: Dict, out_dir: Path, device: torch.device) -> Dict[int, Dict]:
    set_plot_style()
    d = 128
    l_values = [4096, 16384, 65536, 131072]
    allocator = RoPEFrequencyAllocator(d=d, base=10000.0)
    std = allocator.standard()

    k_form = fit_payload["selected"]["k"]["name"]
    k_params = fit_payload["selected"]["k"]["params"]
    x0_form = fit_payload["selected"]["x0"]["name"]
    x0_params = fit_payload["selected"]["x0"]["params"]

    curves: Dict[int, Dict] = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    for ax, L in zip(axes, l_values):
        f2, k2, x2 = allocator.sigmoid_analytical_v2(
            L=L,
            k_form=k_form,
            k_params=k_params,
            x0_form=x0_form,
            x0_params=x0_params,
        )
        ds, cs = compute_phase_collision_curve(std, L=L, num_points=2000, device=device)
        d2, c2 = compute_phase_collision_curve(f2, L=L, num_points=2000, device=device)
        curves[L] = {"k": float(k2), "x0": float(x2), "standard": (ds, cs), "v2": (d2, c2)}
        ax.plot(ds, cs, color="#d62728", label="Standard")
        ax.plot(d2, c2, color="#1f77b4", label="Sigmoid-v2")
        ax.set_xscale("log")
        ax.set_title(f"L={L}")
        ax.set_xlabel(r"$|m-n|$")
        ax.text(
            0.04,
            0.95,
            f"k={k2:.4f}\nx0={x2:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    axes[0].set_ylabel("Phase Collision")
    axes[2].set_ylabel("Phase Collision")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig_both(fig, out_dir / "phase_collision_multi_L_v2")
    plt.close(fig)
    return curves


def _plot_phase_envelope_v2(curves: Dict[int, Dict], out_dir: Path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    l_values = sorted(curves.keys())
    cmap = plt.get_cmap("viridis")
    for i, L in enumerate(l_values):
        color = cmap(i / max(1, len(l_values) - 1))
        ds, cs = curves[L]["standard"]
        d2, c2 = curves[L]["v2"]
        env_s = np.abs(hilbert(cs))
        env_v = np.abs(hilbert(c2))
        ax.plot(ds / float(L), env_s, color=color, linestyle="--", label=f"Std L={L}")
        ax.plot(d2 / float(L), env_v, color=color, linestyle="-", label=f"V2 L={L}")
    ax.set_xscale("log")
    ax.set_xlabel(r"Normalized Distance $|m-n|/L$")
    ax.set_ylabel("Envelope Amplitude")
    ax.set_title("Phase-Collision Envelope (v2)")
    ax.legend(ncol=2, fontsize=8, frameon=True)
    fig.tight_layout()
    save_fig_both(fig, out_dir / "phase_collision_envelope_v2")
    plt.close(fig)


def _plot_attention_v2(fit_payload: Dict, out_dir: Path, device: torch.device) -> None:
    set_plot_style()
    d = 128
    L = 131072
    allocator = RoPEFrequencyAllocator(d=d, base=10000.0)
    std = allocator.standard()
    f2, k2, x2 = allocator.sigmoid_analytical_v2(
        L=L,
        k_form=fit_payload["selected"]["k"]["name"],
        k_params=fit_payload["selected"]["k"]["params"],
        x0_form=fit_payload["selected"]["x0"]["name"],
        x0_params=fit_payload["selected"]["x0"]["params"],
    )
    ds, ss = compute_attention_score_decay(std, L=L, d=d, max_distance=50000, device=device)
    d2, s2 = compute_attention_score_decay(f2, L=L, d=d, max_distance=50000, device=device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    m = ds <= 500
    axes[0].plot(ds[m], ss[m], color="#d62728", label="Standard")
    axes[0].plot(d2[m], s2[m], color="#1f77b4", label="Sigmoid-v2")
    axes[0].set_xlabel(r"$|m-n|$")
    axes[0].set_ylabel("Expected Attention Score")
    axes[0].set_title("Short-Range (0~500)")
    axes[0].legend(frameon=True)

    nz = ds > 0
    axes[1].plot(ds[nz], ss[nz], color="#d62728", label="Standard")
    axes[1].plot(d2[nz], s2[nz], color="#1f77b4", label="Sigmoid-v2")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$|m-n|$ (log)")
    axes[1].set_title("Global View")
    axes[1].text(
        0.03,
        0.95,
        f"k={k2:.4f}, x0={x2:.2f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    fig.tight_layout()
    save_fig_both(fig, out_dir / "attention_decay_v2")
    plt.close(fig)

    diff = s2 - ss
    fig2, ax2 = plt.subplots(figsize=(8.2, 4.6))
    ax2.plot(ds, diff, color="#333333", linewidth=1.2)
    ax2.fill_between(ds, 0.0, diff, where=(diff >= 0), color="#1f77b4", alpha=0.35, label="Sigmoid-v2 better")
    ax2.fill_between(ds, 0.0, diff, where=(diff < 0), color="#d62728", alpha=0.30, label="Standard better")
    ax2.axhline(0.0, linestyle="--", color="black", linewidth=0.8)
    ax2.set_xlabel(r"$|m-n|$")
    ax2.set_ylabel("Sigmoid-v2 - Standard")
    ax2.set_title("Attention Decay Difference (v2)")
    ax2.legend(frameon=True)
    fig2.tight_layout()
    save_fig_both(fig2, out_dir / "attention_decay_difference_v2")
    plt.close(fig2)


def _plot_frequency_distribution_v2(fit_payload: Dict, out_dir: Path) -> None:
    set_plot_style()
    d = 128
    L = 131072
    allocator = RoPEFrequencyAllocator(d=d, base=10000.0)
    std = allocator.standard().cpu().numpy()
    f2, k2, x2 = allocator.sigmoid_analytical_v2(
        L=L,
        k_form=fit_payload["selected"]["k"]["name"],
        k_params=fit_payload["selected"]["k"]["params"],
        x0_form=fit_payload["selected"]["x0"]["name"],
        x0_params=fit_payload["selected"]["x0"]["params"],
    )
    sig = f2.cpu().numpy()
    idx = np.arange(len(std))
    rs = std[1:] / std[:-1]
    rv = sig[1:] / sig[:-1]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    axes[0].plot(idx, std, color="#d62728", label="Standard")
    axes[0].plot(idx, sig, color="#1f77b4", label="Sigmoid-v2")
    axes[0].fill_between(idx, std, sig, color="#999999", alpha=0.2)
    axes[0].set_xlabel("i")
    axes[0].set_ylabel(r"$\theta_i$")
    axes[0].set_title("Frequency Distribution")
    axes[0].legend(frameon=True)

    axes[1].plot(idx, np.log(std), color="#d62728", label="Standard")
    axes[1].plot(idx, np.log(sig), color="#1f77b4", label="Sigmoid-v2")
    axes[1].set_xlabel("i")
    axes[1].set_ylabel(r"$\log(\theta_i)$")
    axes[1].set_title("Log-Frequency")

    ridx = np.arange(len(rs))
    axes[2].plot(ridx, rs, color="#d62728", label="Standard")
    axes[2].plot(ridx, rv, color="#1f77b4", label="Sigmoid-v2")
    axes[2].set_xlabel("i")
    axes[2].set_ylabel(r"$\theta_{i+1}/\theta_i$")
    axes[2].set_title("Adjacent Ratio")
    axes[2].text(
        0.03,
        0.95,
        f"k={k2:.4f}, x0={x2:.2f}",
        transform=axes[2].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    fig.tight_layout()
    save_fig_both(fig, out_dir / "frequency_distribution_v2")
    plt.close(fig)


def _plot_scaling_law_summary_v2(df: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    d_values = sorted(df["d"].unique())
    l_values = sorted(df["L"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
    axa, axb, axc, axd = axes.flatten()
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for idx, d in enumerate(d_values):
        sub = df[df["d"] == d].sort_values("L")
        c = colors[idx % len(colors)]
        axa.plot(sub["L"], sub["k_formula_v2"], color=c, linestyle="-", label=f"d={d} formula-v2")
        axa.plot(sub["L"], sub["k_optimal"], color=c, linestyle="--", marker="o", label=f"d={d} grid-opt")
    axa.set_xscale("log")
    axa.set_xlabel("L")
    axa.set_ylabel("k")
    axa.set_title("(a) k vs L")
    axa.legend(ncol=2, fontsize=8, frameon=True)

    for idx, L in enumerate(l_values):
        sub = df[df["L"] == L].sort_values("d")
        c = plt.get_cmap("viridis")(idx / max(1, len(l_values) - 1))
        axb.plot(sub["d"], sub["x0_formula_v2"], color=c, linestyle="-", label=f"L={L} formula-v2")
        axb.scatter(sub["d"], sub["x0_optimal"], color=c, marker="o", s=30, label=f"L={L} grid-opt")
    axb.set_xlabel("d")
    axb.set_ylabel("x0")
    axb.set_title("(b) x0 vs d")
    axb.legend(ncol=2, fontsize=7, frameon=True)

    g = df.groupby("L", as_index=False).agg(
        score_standard=("score_standard", "mean"),
        score_grid_opt=("score_grid_opt", "mean"),
        score_formula_v1=("score_formula_v1", "mean"),
        score_formula_v2=("score_formula_v2", "mean"),
    )
    axc.plot(g["L"], g["score_standard"], color="#d62728", marker="o", label="Standard")
    axc.plot(g["L"], g["score_grid_opt"], color="#2ca02c", marker="o", label="Grid-Opt")
    axc.plot(g["L"], g["score_formula_v1"], color="#9467bd", marker="o", label="Formula-v1")
    axc.plot(g["L"], g["score_formula_v2"], color="#1f77b4", marker="o", label="Formula-v2")
    axc.set_xscale("log")
    axc.set_xlabel("L")
    axc.set_ylabel("Phase Collision Score")
    axc.set_title("(c) Score vs L")
    axc.legend(ncol=2, fontsize=8, frameon=True)

    imp_v2 = (g["score_standard"] - g["score_formula_v2"]) / np.clip(g["score_standard"], 1e-12, None)
    imp_opt = (g["score_standard"] - g["score_grid_opt"]) / np.clip(g["score_standard"], 1e-12, None)
    axd.plot(g["L"], imp_v2, color="#1f77b4", marker="o", label="Formula-v2")
    axd.plot(g["L"], imp_opt, color="#2ca02c", marker="o", label="Grid-Opt")
    axd.axhline(0.0, linestyle="--", color="black", linewidth=0.8)
    axd.set_xscale("log")
    axd.set_xlabel("L")
    axd.set_ylabel(r"$(S_{std}-S_{sig})/S_{std}$")
    axd.set_title("(d) Relative Improvement")
    axd.legend(frameon=True)

    fig.tight_layout()
    save_fig_both(fig, out_dir / "scaling_law_summary_v2")
    plt.close(fig)


def _print_summary_table(df: pd.DataFrame, fit_payload: Dict, root_dir: Path) -> Dict:
    rows = []
    for r in df.sort_values(["d", "L"]).itertuples():
        k_err = abs(r.k_formula_v2 - r.k_optimal) / max(abs(r.k_optimal), 1e-12) * 100.0
        rows.append(
            {
                "d": int(r.d),
                "L": int(r.L),
                "k_optimal": float(r.k_optimal),
                "k_formula": float(r.k_formula_v2),
                "k_error_pct": float(k_err),
                "score_std": float(r.score_standard),
                "score_sigma": float(r.score_formula_v2),
            }
        )

    title = "Sigmoid-RoPE Experimental Summary"
    header = "║  d  ║   L    ║ k_optimal ║ k_formula ║ k_error(%) ║ score_std ║ score_σ ║"
    border_top = "╔═══════════════════════════════════════════════════════════════════════════╗"
    border_mid = "╠═════╦════════╦═══════════╦═══════════╦════════════╦═══════════╦═════════╣"
    border_bot = "╚═════╩════════╩═══════════╩═══════════╩════════════╩═══════════╩═════════╝"
    print("\n" + border_top)
    print(f"║{title:^75}║")
    print(border_mid)
    print(header)
    print(border_mid)
    for r in rows:
        print(
            "║"
            f"{r['d']:>4d} "
            "║"
            f"{r['L']:>7d} "
            "║"
            f"{r['k_optimal']:>9.4f} "
            "║"
            f"{r['k_formula']:>9.4f} "
            "║"
            f"{r['k_error_pct']:>10.2f} "
            "║"
            f"{r['score_std']:>9.4f} "
            "║"
            f"{r['score_sigma']:>7.4f} "
            "║"
        )
    print(border_bot)

    k_sel = fit_payload["selected"]["k"]
    x_sel = fit_payload["selected"]["x0"]
    if k_sel["name"] == "A":
        k_latex = f"{k_sel['params']['c1']:.4f} \\cdot \\frac{{\\ln L}}{{d}}"
    else:
        k_latex = f"form-{k_sel['name']} params={k_sel['params']}"
    if x_sel["name"] == "A":
        x_latex = f"{x_sel['params']['c3']:.4f} \\cdot d"
    elif x_sel["name"] == "C":
        x_latex = f"{x_sel['params']['c3']:.4f} \\cdot (\\frac{{d}}{{2}}-1)"
    else:
        x_latex = f"form-{x_sel['name']} params={x_sel['params']}"

    improve = (df["score_standard"] - df["score_formula_v2"]) / np.clip(df["score_standard"], 1e-12, None) * 100.0
    idx_max = int(np.argmax(improve))
    idx_min = int(np.argmin(improve))
    max_row = df.iloc[idx_max]
    min_row = df.iloc[idx_min]

    print("\n推荐最终公式 (LaTeX):")
    print(f"  k = {k_latex}, \\quad x_0 = {x_latex}")
    print("\n拟合质量:")
    print(f"  k:  R² = {k_sel['r2']:.4f}, 最大相对误差 = {k_sel['max_rel_error_pct']:.1f}%")
    print(f"  x₀: R² = {x_sel['r2']:.4f}, 最大相对误差 = {x_sel['max_rel_error_pct']:.1f}%")
    print("\n相对改善 (Score_standard - Score_sigmoid_v2) / Score_standard:")
    print(f"  平均: {float(np.mean(improve)):.1f}%")
    print(f"  最大: {float(np.max(improve)):.1f}% (at d={int(max_row['d'])}, L={int(max_row['L'])})")
    print(f"  最小: {float(np.min(improve)):.1f}% (at d={int(min_row['d'])}, L={int(min_row['L'])})")

    summary_payload = {
        "table_rows": rows,
        "recommended_formula": {
            "k": k_latex,
            "x0": x_latex,
            "k_model": k_sel,
            "x0_model": x_sel,
        },
        "improvement_pct": {
            "avg": float(np.mean(improve)),
            "max": float(np.max(improve)),
            "max_at": {"d": int(max_row["d"]), "L": int(max_row["L"])},
            "min": float(np.min(improve)),
            "min_at": {"d": int(min_row["d"]), "L": int(min_row["L"])},
        },
    }
    save_json(root_dir / "data" / "summary.json", summary_payload)
    return summary_payload


def main() -> None:
    args = parse_args()
    ensure_dependencies()

    root_dir = Path(__file__).resolve().parent
    (root_dir / "data").mkdir(parents=True, exist_ok=True)
    (root_dir / "results").mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    print("[env]", env_info())
    print("[run_phase2] device:", device)

    t0 = time.time()
    centers = ensure_coarse_centers(root_dir=root_dir, device=device)
    fine_df = run_fine_search(
        root_dir=root_dir,
        centers=centers,
        device=device,
        num_samples=args.fine_num_samples,
        max_hours_threshold=args.max_hours_threshold,
    )
    fit_payload = fit_formula_models(root_dir=root_dir, fine_df=fine_df)
    compare_df = _compute_formula_comparison(root_dir=root_dir, fine_df=fine_df, fit_payload=fit_payload, device=device)

    result_dir = root_dir / "results"
    _plot_formula_validation_k_v2(compare_df, result_dir)
    _plot_formula_validation_x0_v2(compare_df, result_dir)
    _plot_score_compare_v2(compare_df, result_dir)
    _plot_formula_v1_v2_compare(compare_df, result_dir)
    _plot_fitting_quality(compare_df, result_dir)
    curves = _plot_phase_collision_v2(fit_payload, result_dir, device=device)
    _plot_phase_envelope_v2(curves, result_dir)
    _plot_attention_v2(fit_payload, result_dir, device=device)
    _plot_frequency_distribution_v2(fit_payload, result_dir)
    _plot_scaling_law_summary_v2(compare_df, result_dir)
    _print_summary_table(compare_df, fit_payload, root_dir=root_dir)

    print(f"\n[run_phase2] Done in {time.time() - t0:.2f}s")
    print(f"[run_phase2] data dir: {root_dir / 'data'}")
    print(f"[run_phase2] results dir: {root_dir / 'results'}")


if __name__ == "__main__":
    main()
