#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from tqdm import tqdm

from src.grid_search import _build_sigmoid_freqs_batch
from src.metrics import compute_phase_collision_curve, phase_collision_score, phase_collision_score_batch
from src.rope import RoPEFrequencyAllocator
from src.utils import cleanup_cuda, env_info, get_device, save_json, set_seed
from src.visualization import save_fig_both, set_plot_style

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass
class FitModel:
    name: str
    n_params: int
    ok: bool
    params: Dict[str, float]
    stderr: Dict[str, float]
    r2: float
    aic: float
    bic: float
    mae: float
    max_rel_error_pct: float
    message: str

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "n_params": self.n_params,
            "ok": self.ok,
            "params": self.params,
            "stderr": self.stderr,
            "r2": self.r2,
            "aic": self.aic,
            "bic": self.bic,
            "mae": self.mae,
            "max_rel_error_pct": self.max_rel_error_pct,
            "message": self.message,
        }


def ensure_dependencies() -> None:
    required = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
        ("seaborn", "seaborn"),
        ("transformers", "transformers"),
    ]
    missing: List[str] = []
    for mod, pip_name in required:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pip_name)
    for pip_name in missing:
        print(f"[deps] installing {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sigmoid-RoPE Phase-3 pipeline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--num_samples", type=int, default=5000)
    ap.add_argument("--passkey_budget", type=int, default=100)
    ap.add_argument("--force_skip_passkey", action="store_true")
    ap.add_argument(
        "--local_model_candidates",
        type=str,
        default=(
            "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct,"
            "/root/autodl-tmp/dfrope/ms_models/Qwen/Qwen2___5-7B-Instruct"
        ),
    )
    return ap.parse_args()


def compute_aic_bic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> Tuple[float, float]:
    n = len(y_true)
    rss = float(np.sum((y_true - y_pred) ** 2))
    sigma2 = max(rss / max(n, 1), 1e-18)
    ll = -n / 2.0 * (math.log(2.0 * math.pi * sigma2) + 1.0)
    aic = 2.0 * n_params - 2.0 * ll
    bic = n_params * math.log(max(n, 1)) - 2.0 * ll
    return float(aic), float(bic)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-18:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _fit(
    name: str,
    n_params: int,
    fn: Callable,
    xdata: np.ndarray,
    y: np.ndarray,
    p0: Tuple[float, ...],
    bounds: Tuple[Tuple[float, ...], Tuple[float, ...]],
    param_names: List[str],
) -> FitModel:
    try:
        popt, pcov = curve_fit(fn, xdata, y, p0=p0, bounds=bounds, maxfev=300000)
        pred = fn(xdata, *popt)
        aic, bic = compute_aic_bic(y, pred, n_params=n_params)
        abs_err = np.abs(pred - y)
        rel_err = abs_err / np.clip(np.abs(y), 1e-12, None)
        perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
        return FitModel(
            name=name,
            n_params=n_params,
            ok=True,
            params={k: float(v) for k, v in zip(param_names, popt)},
            stderr={k: float(v) for k, v in zip(param_names, perr)},
            r2=float(_r2(y, pred)),
            aic=float(aic),
            bic=float(bic),
            mae=float(np.mean(abs_err)),
            max_rel_error_pct=float(np.max(rel_err) * 100.0),
            message="ok",
        )
    except Exception as ex:
        return FitModel(
            name=name,
            n_params=n_params,
            ok=False,
            params={},
            stderr={},
            r2=float("-inf"),
            aic=float("inf"),
            bic=float("inf"),
            mae=float("inf"),
            max_rel_error_pct=float("inf"),
            message=str(ex),
        )


def fit_k_models(df: pd.DataFrame) -> List[FitModel]:
    L = df["L"].to_numpy(dtype=np.float64)
    d = df["d"].to_numpy(dtype=np.float64)
    y = df["k_optimal"].to_numpy(dtype=np.float64)
    x = np.vstack([L, d])

    def m0(xx, c):
        _, dd = xx
        return c / dd

    def m1(xx, c):
        LL, dd = xx
        return c * np.log(LL) / dd

    def m2(xx, c1, c2):
        LL, dd = xx
        return c1 * np.log(LL) / dd + c2

    def m3(xx, c1, c2):
        LL, dd = xx
        return c1 * (LL**c2) / dd

    def m4(xx, c1, c2):
        LL, dd = xx
        return c1 * np.log(LL / np.clip(c2, 1e-6, None)) / dd

    return [
        _fit("M0", 1, m0, x, y, p0=(20.0,), bounds=((0.0,), (1000.0,)), param_names=["c"]),
        _fit("M1", 1, m1, x, y, p0=(1.5,), bounds=((0.0,), (20.0,)), param_names=["c"]),
        _fit("M2", 2, m2, x, y, p0=(1.3, 0.0), bounds=((0.0, -1.0), (20.0, 1.0)), param_names=["c1", "c2"]),
        _fit("M3", 2, m3, x, y, p0=(5.0, 0.05), bounds=((0.0, 0.0), (100.0, 2.0)), param_names=["c1", "c2"]),
        _fit("M4", 2, m4, x, y, p0=(1.0, 2.0), bounds=((0.0, 1e-4), (20.0, 1e6)), param_names=["c1", "c2"]),
    ]


def fit_x0_models(df: pd.DataFrame) -> List[FitModel]:
    L = df["L"].to_numpy(dtype=np.float64)
    d = df["d"].to_numpy(dtype=np.float64)
    y = df["x0_optimal"].to_numpy(dtype=np.float64)
    x = np.vstack([L, d])

    def x0(xx, c):
        _, dd = xx
        return c * dd

    def x1(xx, c):
        _, dd = xx
        return c * (dd / 2.0 - 1.0)

    def x2(xx, c1, c2):
        LL, dd = xx
        return c1 * dd + c2 * np.log(LL)

    return [
        _fit("X0", 1, x0, x, y, p0=(0.47,), bounds=((0.0,), (2.0,)), param_names=["c"]),
        _fit("X1", 1, x1, x, y, p0=(0.95,), bounds=((0.0,), (3.0,)), param_names=["c"]),
        _fit("X2", 2, x2, x, y, p0=(0.46, 0.1), bounds=((0.0, -5.0), (2.0, 5.0)), param_names=["c1", "c2"]),
    ]


def choose_k_model(fits: List[FitModel]) -> FitModel:
    ok = {m.name: m for m in fits if m.ok}
    if not ok:
        raise RuntimeError("No valid k model")
    m0 = ok.get("M0")
    m1 = ok.get("M1")
    m3 = ok.get("M3")
    if m0 and m3 and abs(m0.bic - m3.bic) < 2.0:
        return m0
    if m1 and m3 and abs(m1.bic - m3.bic) < 2.0:
        return m1
    if m0 and m1 and m3 and (m0.bic - m3.bic > 6.0) and (m1.bic - m3.bic > 6.0):
        return m3
    return sorted(ok.values(), key=lambda x: x.bic)[0]


def choose_x0_model(fits: List[FitModel]) -> FitModel:
    ok = {m.name: m for m in fits if m.ok}
    if not ok:
        raise RuntimeError("No valid x0 model")
    x0 = ok.get("X0")
    x1 = ok.get("X1")
    x2 = ok.get("X2")
    if x0 and x1 and abs(x0.bic - x1.bic) < 2.0:
        return x0
    if x0 and x2 and (x0.bic - x2.bic) > 6.0:
        return x2
    return sorted(ok.values(), key=lambda x: x.bic)[0]


def k_formula(model: FitModel, L: int, d: int) -> float:
    p = model.params
    if model.name == "M0":
        return float(p["c"]) / float(d)
    if model.name == "M1":
        return float(p["c"]) * math.log(float(L)) / float(d)
    if model.name == "M2":
        return float(p["c1"]) * math.log(float(L)) / float(d) + float(p["c2"])
    if model.name == "M3":
        return float(p["c1"]) * (float(L) ** float(p["c2"])) / float(d)
    if model.name == "M4":
        return float(p["c1"]) * math.log(float(L) / max(float(p["c2"]), 1e-6)) / float(d)
    raise ValueError(model.name)


def x0_formula(model: FitModel, L: int, d: int) -> float:
    p = model.params
    if model.name == "X0":
        return float(p["c"]) * float(d)
    if model.name == "X1":
        return float(p["c"]) * (float(d) / 2.0 - 1.0)
    if model.name == "X2":
        return float(p["c1"]) * float(d) + float(p["c2"]) * math.log(float(L))
    raise ValueError(model.name)


def print_model_selection_table(k_fits: List[FitModel], x_fits: List[FitModel], k_best: FitModel, x_best: FitModel) -> None:
    print("\n+--------+----------+---------+---------+-----------+------------+")
    print("| Model  | n_params |   R2    |   AIC   |    BIC    | Recommend? |")
    print("+--------+----------+---------+---------+-----------+------------+")
    for m in k_fits:
        rec = "YES" if m.name == k_best.name else ""
        print(f"| {m.name:<6} | {m.n_params:^8d} | {m.r2:>7.4f} | {m.aic:>7.2f} | {m.bic:>9.2f} | {rec:^10} |")
    print("+--------+----------+---------+---------+-----------+------------+")
    for m in x_fits:
        rec = "YES" if m.name == x_best.name else ""
        print(f"| {m.name:<6} | {m.n_params:^8d} | {m.r2:>7.4f} | {m.aic:>7.2f} | {m.bic:>9.2f} | {rec:^10} |")
    print("+--------+----------+---------+---------+-----------+------------+")
    print("-> selected by BIC with your simplification rules.")


def run_model_selection(root_dir: Path, fine_df: pd.DataFrame) -> Dict:
    data_dir = root_dir / "data" / "phase3"
    result_dir = root_dir / "results" / "phase3"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    k_fits = fit_k_models(fine_df)
    x_fits = fit_x0_models(fine_df)
    k_best = choose_k_model(k_fits)
    x_best = choose_x0_model(x_fits)
    print_model_selection_table(k_fits, x_fits, k_best, x_best)

    payload = {
        "k_models": [m.to_dict() for m in k_fits],
        "x0_models": [m.to_dict() for m in x_fits],
        "recommended": {"k": k_best.to_dict(), "x0": x_best.to_dict()},
    }
    save_json(data_dir / "model_selection.json", payload)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    k_names = [m.name for m in k_fits]
    x_names = [m.name for m in x_fits]
    k_bic = [m.bic for m in k_fits]
    x_bic = [m.bic for m in x_fits]
    axes[0].bar(k_names, k_bic, color=["#2ca02c" if n == k_best.name else "#1f77b4" for n in k_names])
    axes[0].set_title("k-model BIC")
    axes[0].set_ylabel("BIC")
    axes[1].bar(x_names, x_bic, color=["#2ca02c" if n == x_best.name else "#1f77b4" for n in x_names])
    axes[1].set_title("x0-model BIC")
    fig.tight_layout()
    save_fig_both(fig, result_dir / "model_selection_comparison")
    plt.close(fig)
    return payload


def run_sensitivity(
    root_dir: Path,
    device: torch.device,
    k_model: FitModel,
    x0_model: FitModel,
    num_samples: int,
) -> Dict:
    data_dir = root_dir / "data" / "phase3"
    result_dir = root_dir / "results" / "phase3"
    fine_df = pd.read_csv(root_dir / "data" / "fine_search_results.csv")
    row = fine_df[(fine_df["d"] == 128) & (fine_df["L"] == 131072)]
    if row.empty:
        raise RuntimeError("Missing d=128,L=131072 in fine_search_results.csv")
    k_opt = float(row.iloc[0]["k_optimal"])
    x0_opt = float(row.iloc[0]["x0_optimal"])
    d = 128
    L = 131072
    n = d // 2

    allocator = RoPEFrequencyAllocator(d=d, base=10000.0)
    score_std, _ = phase_collision_score(allocator.standard(), L=L, d=d, num_samples=num_samples, device=device)
    k_formula_val = k_formula(k_model, L=L, d=d)
    x0_formula_val = x0_formula(x0_model, L=L, d=d)

    k_values = np.linspace(0.5 * k_opt, 2.0 * k_opt, 50, dtype=np.float64)
    k_scores = []
    for kv in tqdm(k_values, desc="sens-k", dynamic_ncols=True):
        s, _ = phase_collision_score(allocator.sigmoid(float(kv), x0_opt), L=L, d=d, num_samples=num_samples, device=device)
        k_scores.append(float(s))
    k_scores = np.array(k_scores)

    x0_values = np.linspace(0.2 * n, 0.95 * n, 50, dtype=np.float64)
    x0_scores = []
    for xv in tqdm(x0_values, desc="sens-x0", dynamic_ncols=True):
        s, _ = phase_collision_score(allocator.sigmoid(k_opt, float(xv)), L=L, d=d, num_samples=num_samples, device=device)
        x0_scores.append(float(s))
    x0_scores = np.array(x0_scores)

    k_grid = np.linspace(0.5 * k_opt, 2.0 * k_opt, 50, dtype=np.float64)
    x0_grid = np.linspace(0.3 * n, 0.8 * n, 50, dtype=np.float64)
    heat = np.zeros((len(k_grid), len(x0_grid)), dtype=np.float64)
    for i, kv in enumerate(tqdm(k_grid, desc="sens-2d", dynamic_ncols=True)):
        fb = _build_sigmoid_freqs_batch(d=d, base=10000.0, k=float(kv), x0_values=x0_grid)
        scores, _ = phase_collision_score_batch(fb, L=L, num_samples=num_samples, device=device)
        heat[i, :] = scores.numpy()
        del fb, scores
        cleanup_cuda()

    pd.DataFrame({"k": k_values, "k_over_kopt": k_values / k_opt, "score": k_scores}).to_csv(
        data_dir / "sensitivity_k.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame({"x0": x0_values, "x0_over_x0opt": x0_values / x0_opt, "score": x0_scores}).to_csv(
        data_dir / "sensitivity_x0.csv", index=False, encoding="utf-8"
    )
    np.save(data_dir / "sensitivity_heatmap.npy", heat)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(k_values / k_opt, k_scores, color="#1f77b4")
    ax.axvline(1.0, linestyle="--", color="#2ca02c", label="k_opt")
    ax.axvline(k_formula_val / k_opt, linestyle="--", color="#ff7f0e", label="k_formula")
    ax.axhline(score_std, linestyle=":", color="#d62728", label="standard")
    ax.set_xlabel("k / k_optimal")
    ax.set_ylabel("Phase Collision Score")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig_both(fig, result_dir / "sensitivity_k")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7.6, 4.6))
    ax2.plot(x0_values / x0_opt, x0_scores, color="#1f77b4")
    ax2.axvline(1.0, linestyle="--", color="#2ca02c", label="x0_opt")
    ax2.axvline(x0_formula_val / x0_opt, linestyle="--", color="#ff7f0e", label="x0_formula")
    ax2.axhline(score_std, linestyle=":", color="#d62728", label="standard")
    ax2.set_xlabel("x0 / x0_optimal")
    ax2.set_ylabel("Phase Collision Score")
    ax2.legend(frameon=True)
    fig2.tight_layout()
    save_fig_both(fig2, result_dir / "sensitivity_x0")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(7.2, 5.6))
    im = ax3.imshow(
        heat.T,
        origin="lower",
        aspect="auto",
        extent=[k_grid.min(), k_grid.max(), x0_grid.min(), x0_grid.max()],
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax3, label="Phase Collision Score")
    xx, yy = np.meshgrid(k_grid, x0_grid)
    ax3.contour(xx, yy, heat.T, levels=[score_std], colors="white", linewidths=1.2)
    ax3.scatter([k_opt], [x0_opt], marker="*", s=180, color="#ffdd00", edgecolor="black", label="optimal")
    ax3.scatter([k_formula_val], [x0_formula_val], marker="^", s=90, color="#ff7f0e", edgecolor="black", label="formula")
    ax3.set_xlabel("k")
    ax3.set_ylabel("x0")
    ax3.legend(frameon=True)
    fig3.tight_layout()
    save_fig_both(fig3, result_dir / "sensitivity_heatmap")
    plt.close(fig3)

    good_k = k_values[k_scores < score_std]
    good_x0 = x0_values[x0_scores < score_std]
    return {
        "d": d,
        "L": L,
        "k_opt": k_opt,
        "x0_opt": x0_opt,
        "k_formula": float(k_formula_val),
        "x0_formula": float(x0_formula_val),
        "score_standard": float(score_std),
        "k_good_range": [float(good_k.min()) if len(good_k) else None, float(good_k.max()) if len(good_k) else None],
        "x0_good_range": [float(good_x0.min()) if len(good_x0) else None, float(good_x0.max()) if len(good_x0) else None],
    }


def _placeholder_passkey_outputs(result_dir: Path, data_dir: Path, reason: str) -> Dict:
    df = pd.DataFrame(
        [{"method": "N/A", "context_length": None, "position_ratio": None, "repeat": None, "correct": None, "status": "skipped", "reason": reason}]
    )
    df.to_csv(data_dir / "passkey_results.csv", index=False, encoding="utf-8")

    set_plot_style()
    for stem in ["passkey_retrieval", "passkey_retrieval_by_length"]:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.axis("off")
        ax.text(0.5, 0.55, "Passkey evaluation skipped", ha="center", va="center", fontsize=14)
        ax.text(0.5, 0.40, reason, ha="center", va="center", fontsize=10)
        fig.tight_layout()
        save_fig_both(fig, result_dir / stem)
        plt.close(fig)
    return {"mode": "skipped", "reason": reason}


def run_passkey_or_fallback(
    root_dir: Path,
    device: torch.device,
    k_formula_val: float,
    x0_formula_val: float,
    k_opt: float,
    x0_opt: float,
    args: argparse.Namespace,
) -> Dict:
    data_dir = root_dir / "data" / "phase3"
    result_dir = root_dir / "results" / "phase3"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.force_skip_passkey:
        return _placeholder_passkey_outputs(result_dir, data_dir, "force_skip_passkey=True")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as ex:
        return _placeholder_passkey_outputs(result_dir, data_dir, f"transformers import failed: {ex}")

    candidates = [p.strip() for p in str(args.local_model_candidates).split(",") if p.strip()]
    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break
    if model_path is None:
        return _placeholder_passkey_outputs(result_dir, data_dir, "no local model path found")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs = {"torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32, "trust_remote_code": True, "local_files_only": True}
        if device.type == "cuda":
            load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model.eval()
    except Exception as ex:
        return _placeholder_passkey_outputs(result_dir, data_dir, f"model load failed: {type(ex).__name__}: {ex}")

    # prepare rotary patch
    rotary = [m for _, m in model.named_modules() if hasattr(m, "inv_freq")]
    if not rotary:
        return _placeholder_passkey_outputs(result_dir, data_dir, "no rotary inv_freq modules found")
    orig_inv = [m.inv_freq.detach().clone() for m in rotary]
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    allocator = RoPEFrequencyAllocator(d=head_dim, base=10000.0)

    def apply_freq(freqs: torch.Tensor) -> None:
        for m, old in zip(rotary, orig_inv):
            nv = freqs.to(device=old.device, dtype=old.dtype)
            if isinstance(m.inv_freq, torch.nn.Parameter):
                m.inv_freq = torch.nn.Parameter(nv, requires_grad=False)
            else:
                m.inv_freq = nv
            if hasattr(m, "original_inv_freq"):
                m.original_inv_freq = nv.clone()
            if hasattr(m, "max_seq_len_cached"):
                m.max_seq_len_cached = 0

    def restore_std() -> None:
        for m, old in zip(rotary, orig_inv):
            nv = old.to(device=old.device, dtype=old.dtype)
            if isinstance(m.inv_freq, torch.nn.Parameter):
                m.inv_freq = torch.nn.Parameter(nv, requires_grad=False)
            else:
                m.inv_freq = nv
            if hasattr(m, "original_inv_freq"):
                m.original_inv_freq = nv.clone()
            if hasattr(m, "max_seq_len_cached"):
                m.max_seq_len_cached = 0

    max_pos = int(getattr(model.config, "max_position_embeddings", 8192))
    lengths = [L for L in [4096, 8192, 16384, 32768] if L <= max_pos]
    if not lengths:
        return _placeholder_passkey_outputs(result_dir, data_dir, f"model max_position_embeddings={max_pos} too small")
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    repeats = max(1, min(10, int(args.passkey_budget // (len(lengths) * len(ratios)))))
    filler = "The grass is green. The sky is blue. The sun is yellow. "
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)

    methods = [
        ("Standard", None),
        ("Sigmoid(Formula)", (k_formula_val, x0_formula_val)),
        ("Sigmoid(Grid-Optimal)", (k_opt, x0_opt)),
    ]

    rows: List[Dict] = []
    for method_name, kv in methods:
        if kv is None:
            restore_std()
        else:
            apply_freq(allocator.sigmoid(k=float(kv[0]), x0=float(kv[1])))
        pbar = tqdm(total=len(lengths) * len(ratios) * repeats, desc=f"passkey-{method_name}", dynamic_ncols=True)
        for L in lengths:
            for r in ratios:
                for t in range(repeats):
                    passkey = f"{np.random.randint(10000, 99999)}"
                    ask = tokenizer.encode("\nWhat is the pass key? The pass key is ", add_special_tokens=False)
                    needle = tokenizer.encode(f" The pass key is {passkey}. Remember it. {passkey} is the pass key. ", add_special_tokens=False)
                    budget = L - len(ask) - len(needle) - 4
                    if budget < 32:
                        rows.append({"method": method_name, "context_length": L, "position_ratio": r, "repeat": t, "correct": 0, "status": "invalid"})
                        pbar.update(1)
                        continue
                    rep = budget // len(filler_ids) + 2
                    body = (filler_ids * rep)[:budget]
                    pos = int(r * max(1, len(body) - 1))
                    seq = body[:pos] + needle + body[pos:] + ask
                    seq = seq[:L]
                    if tokenizer.bos_token_id is not None:
                        seq = [tokenizer.bos_token_id] + seq
                    x = torch.tensor([seq], dtype=torch.long, device=next(model.parameters()).device)
                    try:
                        with torch.no_grad():
                            out = model.generate(
                                input_ids=x,
                                max_new_tokens=8,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                        gen = tokenizer.decode(out[0, x.shape[1]:], skip_special_tokens=True)
                        ok = int(passkey in "".join(ch for ch in gen if ch.isdigit()))
                        rows.append(
                            {"method": method_name, "context_length": L, "position_ratio": r, "repeat": t, "correct": ok, "status": "ok"}
                        )
                    except RuntimeError as ex:
                        rows.append(
                            {
                                "method": method_name,
                                "context_length": L,
                                "position_ratio": r,
                                "repeat": t,
                                "correct": 0,
                                "status": "oom" if "out of memory" in str(ex).lower() else "error",
                            }
                        )
                        cleanup_cuda()
                    pbar.update(1)
        pbar.close()

    pass_df = pd.DataFrame(rows)
    pass_df.to_csv(data_dir / "passkey_results.csv", index=False, encoding="utf-8")
    ok_df = pass_df[pass_df["status"] == "ok"].copy()
    if ok_df.empty:
        return _placeholder_passkey_outputs(result_dir, data_dir, "all passkey trials failed")

    # heatmap by method
    methods_order = ["Standard", "Sigmoid(Formula)", "Sigmoid(Grid-Optimal)"]
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9), sharey=True)
    for ax, m in zip(axes, methods_order):
        sub = ok_df[ok_df["method"] == m]
        pivot = sub.groupby(["context_length", "position_ratio"], as_index=False)["correct"].mean().pivot(
            index="context_length", columns="position_ratio", values="correct"
        )
        if pivot.empty:
            ax.axis("off")
            ax.set_title(m)
            continue
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="RdYlGn")
        ax.set_title(m)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(int(v)) for v in pivot.index])
        ax.set_xlabel("Passkey Position Ratio")
    axes[0].set_ylabel("Context Length")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Accuracy")
    fig.tight_layout()
    save_fig_both(fig, result_dir / "passkey_retrieval")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7.4, 4.4))
    mean_len = ok_df.groupby(["method", "context_length"], as_index=False)["correct"].mean()
    for m in methods_order:
        sub = mean_len[mean_len["method"] == m].sort_values("context_length")
        ax2.plot(sub["context_length"], sub["correct"], marker="o", label=m)
    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("Average Accuracy")
    ax2.set_title("Passkey Retrieval by Length")
    ax2.legend(frameon=True)
    fig2.tight_layout()
    save_fig_both(fig2, result_dir / "passkey_retrieval_by_length")
    plt.close(fig2)
    return {"mode": "passkey", "model_path": model_path, "repeats": repeats, "lengths": lengths}


def run_figure1_overview(root_dir: Path, k_model: FitModel, x0_model: FitModel) -> Dict:
    result_dir = root_dir / "results" / "phase3"
    result_dir.mkdir(parents=True, exist_ok=True)

    d = 128
    L = 131072
    allocator = RoPEFrequencyAllocator(d=d, base=10000.0)
    n = allocator.N
    i = np.arange(n, dtype=np.float64)

    std = allocator.standard().numpy()
    kf = k_formula(k_model, L=L, d=d)
    x0f = x0_formula(x0_model, L=L, d=d)
    sig = allocator.sigmoid(k=kf, x0=x0f).numpy()

    log_std = np.log10(std)
    log_sig = np.log10(sig)
    wl_std = 2.0 * math.pi / np.clip(std, 1e-30, None)
    wl_sig = 2.0 * math.pi / np.clip(sig, 1e-30, None)

    # local density: -d(log(theta))/di
    dens_std = -np.gradient(np.log(np.clip(std, 1e-30, None)))
    dens_sig = -np.gradient(np.log(np.clip(sig, 1e-30, None)))

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

    ax = axes[0]
    ax.plot(i, log_std, "--", color="#d62728", label="Standard RoPE")
    ax.plot(i, log_sig, "-", color="#1f77b4", label="Sigmoid-RoPE")
    ax.fill_between(i, log_std, log_sig, color="#1f77b4", alpha=0.15)
    ax.set_xlabel("Dimension Pair Index $i$")
    ax.set_ylabel(r"$\log_{10}(\theta_i)$")
    ax.set_title("(a) Frequency Allocation")
    ax.legend(frameon=True, loc="best")
    ax.annotate("High-frequency plateau", xy=(n * 0.16, log_sig[int(n * 0.16)]), xytext=(n * 0.02, log_sig[int(n * 0.16)] + 0.35), arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax.annotate("Low-frequency plateau", xy=(n * 0.84, log_sig[int(n * 0.84)]), xytext=(n * 0.52, log_sig[int(n * 0.84)] - 0.45), arrowprops={"arrowstyle": "->", "lw": 1.0})

    ax = axes[1]
    ax.plot(i, wl_std, "--", color="#d62728", label="Standard RoPE")
    ax.plot(i, wl_sig, "-", color="#1f77b4", label="Sigmoid-RoPE")
    ax.set_yscale("log")
    ax.axhline(L, color="#2ca02c", linestyle=":", label=f"$L={L}$")
    ax.axhline(2.0 * math.pi, color="#9467bd", linestyle=":", label=r"$\lambda=2\pi$")
    ax.set_xlabel("Dimension Pair Index $i$")
    ax.set_ylabel(r"Wavelength $\lambda_i = 2\pi/\theta_i$")
    ax.set_title("(b) Wavelength Coverage")
    ax.legend(frameon=True, loc="best")

    ax = axes[2]
    ax.plot(i, dens_std, "--", color="#d62728", label="Standard RoPE")
    ax.plot(i, dens_sig, "-", color="#1f77b4", label="Sigmoid-RoPE")
    ax.set_xlabel("Dimension Pair Index $i$")
    ax.set_ylabel(r"$-\frac{d\log \theta}{di}$")
    ax.set_title("(c) Local Frequency Density")
    ax.legend(frameon=True, loc="best")

    fig.tight_layout()
    save_fig_both(fig, result_dir / "figure1_overview")
    plt.close(fig)
    return {"d": d, "L": L, "k_formula": float(kf), "x0_formula": float(x0f)}


def _build_fit_from_dict(payload: Dict) -> FitModel:
    return FitModel(
        name=str(payload["name"]),
        n_params=int(payload["n_params"]),
        ok=bool(payload["ok"]),
        params={k: float(v) for k, v in payload.get("params", {}).items()},
        stderr={k: float(v) for k, v in payload.get("stderr", {}).items()},
        r2=float(payload["r2"]),
        aic=float(payload["aic"]),
        bic=float(payload["bic"]),
        mae=float(payload["mae"]),
        max_rel_error_pct=float(payload["max_rel_error_pct"]),
        message=str(payload.get("message", "")),
    )


def evaluate_formula_scores(
    root_dir: Path,
    fine_df: pd.DataFrame,
    device: torch.device,
    k_model: FitModel,
    x0_model: FitModel,
    num_samples: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for r in tqdm(fine_df.sort_values(["d", "L"]).itertuples(index=False), total=len(fine_df), desc="eval-formula", dynamic_ncols=True):
        d = int(r.d)
        L = int(r.L)
        allocator = RoPEFrequencyAllocator(d=d, base=10000.0)
        kf = k_formula(k_model, L=L, d=d)
        x0f = x0_formula(x0_model, L=L, d=d)
        sf, _ = phase_collision_score(allocator.sigmoid(k=float(kf), x0=float(x0f)), L=L, d=d, num_samples=num_samples, device=device)
        sstd = float(getattr(r, "score_standard", np.nan))
        if not np.isfinite(sstd):
            sstd, _ = phase_collision_score(allocator.standard(), L=L, d=d, num_samples=num_samples, device=device)
        sopt = float(getattr(r, "score_optimal", np.nan))
        if not np.isfinite(sopt):
            sopt = float("nan")
        rows.append(
            {
                "d": d,
                "L": L,
                "k_optimal": float(getattr(r, "k_optimal")),
                "x0_optimal": float(getattr(r, "x0_optimal")),
                "k_formula": float(kf),
                "x0_formula": float(x0f),
                "score_standard": float(sstd),
                "score_formula": float(sf),
                "score_optimal": float(sopt),
                "impr_formula_pct": float((sstd - sf) / max(sstd, 1e-12) * 100.0),
                "impr_opt_pct": float((sstd - sopt) / max(sstd, 1e-12) * 100.0) if np.isfinite(sopt) else np.nan,
                "formula_vs_opt_gap_pct": float((sf - sopt) / max(sstd, 1e-12) * 100.0) if np.isfinite(sopt) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(root_dir / "data" / "phase3" / "formula_phase_collision_v3.csv", index=False, encoding="utf-8")
    return out


def _format_k_latex(m: FitModel) -> str:
    p = m.params
    if m.name == "M0":
        return rf"k = {p['c']:.4f} \cdot d^{{-1}}"
    if m.name == "M1":
        return rf"k = {p['c']:.4f} \cdot \frac{{\ln L}}{{d}}"
    if m.name == "M2":
        return rf"k = {p['c1']:.4f} \cdot \frac{{\ln L}}{{d}} + {p['c2']:.4f}"
    if m.name == "M3":
        return rf"k = {p['c1']:.4f} \cdot \frac{{L^{{{p['c2']:.4f}}}}}{{d}}"
    if m.name == "M4":
        return rf"k = {p['c1']:.4f} \cdot \frac{{\ln(L/{p['c2']:.4f})}}{{d}}"
    return m.name


def _format_x0_latex(m: FitModel) -> str:
    p = m.params
    if m.name == "X0":
        return rf"x_0 = {p['c']:.4f} \cdot d"
    if m.name == "X1":
        return rf"x_0 = {p['c']:.4f} \cdot (d/2 - 1)"
    if m.name == "X2":
        return rf"x_0 = {p['c1']:.4f} \cdot d + {p['c2']:.4f}\ln L"
    return m.name


def print_final_summary(
    summary: Dict,
    k_best: FitModel,
    x_best: FitModel,
    passkey_info: Dict,
    generated_pdfs: List[str],
) -> None:
    print("\n====================================================================")
    print("             Sigmoid-RoPE 最终实验报告")
    print("====================================================================")
    print("\n1. 推荐公式（基于 BIC 模型选择）:")
    print(f"   k  = {_format_k_latex(k_best)}")
    print(f"   x0 = {_format_x0_latex(x_best)}")
    print("\n2. 拟合质量:")
    print(f"   k:  R2 = {k_best.r2:.4f}, AIC = {k_best.aic:.2f}, BIC = {k_best.bic:.2f}")
    print(f"   x0: R2 = {x_best.r2:.4f}, AIC = {x_best.aic:.2f}, BIC = {x_best.bic:.2f}")
    print("\n3. Phase Collision 改善（vs Standard RoPE）:")
    print(
        "   公式参数: 平均 {avg:.1f}%, 范围 [{mn:.1f}%, {mx:.1f}%]".format(
            avg=summary["impr_formula_avg"],
            mn=summary["impr_formula_min"],
            mx=summary["impr_formula_max"],
        )
    )
    print(
        "   Grid-Opt: 平均 {avg:.1f}%, 范围 [{mn:.1f}%, {mx:.1f}%]".format(
            avg=summary["impr_opt_avg"],
            mn=summary["impr_opt_min"],
            mx=summary["impr_opt_max"],
        )
    )
    print(f"   公式 vs Grid-Opt 的差距: 平均 {summary['formula_vs_opt_gap_avg']:.1f}%")
    print("\n4. 鲁棒性:")
    print(
        "   k 在 [{a:.4f}, {b:.4f}] 范围内 Sigmoid 优于 Standard".format(
            a=summary["k_good_min"],
            b=summary["k_good_max"],
        )
    )
    print(
        "   x0 在 [{a:.4f}, {b:.4f}] 范围内 Sigmoid 优于 Standard".format(
            a=summary["x0_good_min"],
            b=summary["x0_good_max"],
        )
    )
    print("\n5. 下游任务（Passkey Retrieval / PPL）:")
    if passkey_info.get("mode") == "passkey":
        print(
            "   Passkey 完成：model={m}, lengths={ls}, repeats={r}".format(
                m=passkey_info.get("model_path"),
                ls=passkey_info.get("lengths"),
                r=passkey_info.get("repeats"),
            )
        )
    else:
        print(f"   已跳过：{passkey_info.get('reason', 'unknown')}")
    print("\n6. 生成的图表清单:")
    for p in generated_pdfs:
        print(f"   - {p}")
    print("====================================================================\n")


def main() -> None:
    args = parse_args()
    ensure_dependencies()
    set_seed(args.seed)

    root_dir = Path(__file__).resolve().parent
    (root_dir / "data" / "phase3").mkdir(parents=True, exist_ok=True)
    (root_dir / "results" / "phase3").mkdir(parents=True, exist_ok=True)

    device = get_device(prefer_cuda=not args.cpu)
    print("[env]", env_info())
    print("[phase3] device:", device)

    fine_csv = root_dir / "data" / "fine_search_results.csv"
    if not fine_csv.exists():
        raise FileNotFoundError(f"Missing required file: {fine_csv}")
    fine_df = pd.read_csv(fine_csv)
    need_cols = {"d", "L", "k_optimal", "x0_optimal", "score_optimal", "score_standard"}
    if not need_cols.issubset(fine_df.columns):
        raise RuntimeError(f"fine_search_results.csv missing columns: {sorted(list(need_cols - set(fine_df.columns)))}")

    t0 = time.time()
    model_sel = run_model_selection(root_dir, fine_df)
    k_best = _build_fit_from_dict(model_sel["recommended"]["k"])
    x_best = _build_fit_from_dict(model_sel["recommended"]["x0"])

    sensitivity = run_sensitivity(
        root_dir=root_dir,
        device=device,
        k_model=k_best,
        x0_model=x_best,
        num_samples=args.num_samples,
    )

    passkey_info = run_passkey_or_fallback(
        root_dir=root_dir,
        device=device,
        k_formula_val=float(sensitivity["k_formula"]),
        x0_formula_val=float(sensitivity["x0_formula"]),
        k_opt=float(sensitivity["k_opt"]),
        x0_opt=float(sensitivity["x0_opt"]),
        args=args,
    )

    fig1_info = run_figure1_overview(root_dir=root_dir, k_model=k_best, x0_model=x_best)
    eval_df = evaluate_formula_scores(
        root_dir=root_dir,
        fine_df=fine_df,
        device=device,
        k_model=k_best,
        x0_model=x_best,
        num_samples=args.num_samples,
    )

    summary = {
        "k_model": k_best.to_dict(),
        "x0_model": x_best.to_dict(),
        "k_formula_latex": _format_k_latex(k_best),
        "x0_formula_latex": _format_x0_latex(x_best),
        "impr_formula_avg": float(eval_df["impr_formula_pct"].mean()),
        "impr_formula_min": float(eval_df["impr_formula_pct"].min()),
        "impr_formula_max": float(eval_df["impr_formula_pct"].max()),
        "impr_opt_avg": float(eval_df["impr_opt_pct"].mean()),
        "impr_opt_min": float(eval_df["impr_opt_pct"].min()),
        "impr_opt_max": float(eval_df["impr_opt_pct"].max()),
        "formula_vs_opt_gap_avg": float(eval_df["formula_vs_opt_gap_pct"].mean()),
        "k_good_min": float(sensitivity["k_good_range"][0]) if sensitivity["k_good_range"][0] is not None else float("nan"),
        "k_good_max": float(sensitivity["k_good_range"][1]) if sensitivity["k_good_range"][1] is not None else float("nan"),
        "x0_good_min": float(sensitivity["x0_good_range"][0]) if sensitivity["x0_good_range"][0] is not None else float("nan"),
        "x0_good_max": float(sensitivity["x0_good_range"][1]) if sensitivity["x0_good_range"][1] is not None else float("nan"),
        "sensitivity": sensitivity,
        "passkey": passkey_info,
        "figure1": fig1_info,
        "elapsed_sec": float(time.time() - t0),
    }

    out_summary = root_dir / "data" / "phase3" / "summary.json"
    save_json(out_summary, summary)

    generated_pdfs = sorted(str(p.relative_to(root_dir)) for p in (root_dir / "results" / "phase3").glob("*.pdf"))
    print_final_summary(summary=summary, k_best=k_best, x_best=x_best, passkey_info=passkey_info, generated_pdfs=generated_pdfs)
    print(f"[phase3] saved summary: {out_summary}")


if __name__ == "__main__":
    main()
