from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .metrics import phase_collision_score, phase_collision_score_batch
from .rope import RoPEFrequencyAllocator
from .utils import cleanup_cuda, load_json, save_json


@dataclass
class GridSearchConfig:
    d_values: Tuple[int, ...] = (64, 128, 256)
    L_values: Tuple[int, ...] = (4096, 8192, 16384, 32768, 65536, 131072)
    base: float = 10000.0
    mode: str = "auto"  # auto / coarse / fine
    coarse_k_step: float = 0.02
    coarse_x0_step: float = 2.0
    fine_k_step: float = 0.005
    num_samples_coarse: int = 1800
    num_samples_fine: int = 5000
    checkpoint_path: str = "data/grid_search_checkpoint.json"
    csv_path: str = "data/grid_search_results.csv"


def _make_search_grid(
    d: int,
    mode: str,
    coarse_k_step: float,
    coarse_x0_step: float,
    fine_k_step: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    n = d // 2
    if mode == "fine":
        k_values = np.arange(0.01, 0.5001, fine_k_step, dtype=np.float64)
        x0_values = np.arange(int(math.floor(n * 0.1)), int(math.ceil(n * 0.9)) + 1, 1, dtype=np.float64)
        num_samples = 5000
        return k_values, x0_values, num_samples

    # coarse / auto default
    k_values = np.arange(0.02, 0.5001, coarse_k_step, dtype=np.float64)
    x0_step = float(coarse_x0_step)
    x0_values = np.arange(int(math.floor(n * 0.1)), int(math.ceil(n * 0.9)) + 1, x0_step, dtype=np.float64)
    num_samples = 1800
    return k_values, x0_values, num_samples


def _build_sigmoid_freqs_batch(
    d: int,
    base: float,
    k: float,
    x0_values: np.ndarray,
) -> torch.Tensor:
    """
    Build frequencies for all x0 values at fixed k.
    Returns: (M, N) float64
    """
    n = d // 2
    i = torch.arange(n, dtype=torch.float64).unsqueeze(0)  # (1, N)
    x0 = torch.tensor(x0_values, dtype=torch.float64).unsqueeze(-1)  # (M, 1)
    k_t = torch.tensor(float(k), dtype=torch.float64)

    raw = 1.0 / (1.0 + torch.exp(-k_t * (i - x0)))
    raw_min = 1.0 / (1.0 + torch.exp(-k_t * (torch.tensor(0.0, dtype=torch.float64) - x0)))
    raw_max = 1.0 / (1.0 + torch.exp(-k_t * (torch.tensor(float(n - 1), dtype=torch.float64) - x0)))
    denom = raw_max - raw_min
    denom = torch.clamp(denom, min=1e-18)
    s_tilde = (raw - raw_min) / denom
    freqs = torch.pow(torch.tensor(float(base), dtype=torch.float64), -s_tilde)
    return freqs


def run_grid_search(config: GridSearchConfig, device: torch.device) -> pd.DataFrame:
    t0 = time.time()
    mode = config.mode
    if mode not in {"auto", "coarse", "fine"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "auto":
        mode = "coarse"

    cp_path = Path(config.checkpoint_path)
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    cp = load_json(cp_path, default={"rows": [], "completed_keys": [], "mode": mode})
    rows: List[Dict] = list(cp.get("rows", []))
    completed = set(cp.get("completed_keys", []))

    work_items = [(d, L) for d in config.d_values for L in config.L_values]
    pbar = tqdm(work_items, desc=f"GridSearch[{mode}]", dynamic_ncols=True)
    for d, L in pbar:
        key = f"d={d}|L={L}|mode={mode}"
        if key in completed:
            continue

        allocator = RoPEFrequencyAllocator(d=d, base=config.base)
        k_values, x0_values, num_samples = _make_search_grid(
            d=d,
            mode=mode,
            coarse_k_step=config.coarse_k_step,
            coarse_x0_step=config.coarse_x0_step,
            fine_k_step=config.fine_k_step,
        )
        if mode == "fine":
            num_samples = config.num_samples_fine
        else:
            num_samples = config.num_samples_coarse

        standard_freqs = allocator.standard()
        score_standard, standard_parts = phase_collision_score(
            freqs=standard_freqs,
            L=L,
            d=d,
            num_samples=num_samples,
            device=device,
        )

        formula_freqs, k_formula, x0_formula = allocator.sigmoid_analytical(L=L)
        score_formula, formula_parts = phase_collision_score(
            freqs=formula_freqs,
            L=L,
            d=d,
            num_samples=num_samples,
            device=device,
        )

        best_score = float("inf")
        best_k = None
        best_x0 = None
        best_parts = {"short": float("nan"), "mid": float("nan"), "long": float("nan"), "total": float("inf")}

        for k in k_values:
            freqs_batch = _build_sigmoid_freqs_batch(d=d, base=config.base, k=float(k), x0_values=x0_values)
            scores, parts = phase_collision_score_batch(
                freqs_batch=freqs_batch,
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
                best_parts = {
                    "short": float(parts["short"][idx].item()),
                    "mid": float(parts["mid"][idx].item()),
                    "long": float(parts["long"][idx].item()),
                    "total": float(parts["total"][idx].item()),
                }

            del freqs_batch, scores, parts
            cleanup_cuda()

        row = {
            "d": int(d),
            "L": int(L),
            "search_mode": mode,
            "num_samples": int(num_samples),
            "k_optimal": float(best_k),
            "x0_optimal": float(best_x0),
            "k_formula": float(k_formula),
            "x0_formula": float(x0_formula),
            "score_optimal": float(best_score),
            "score_formula": float(score_formula),
            "score_standard": float(score_standard),
            "score_optimal_short": float(best_parts["short"]),
            "score_optimal_mid": float(best_parts["mid"]),
            "score_optimal_long": float(best_parts["long"]),
            "score_formula_short": float(formula_parts["short"]),
            "score_formula_mid": float(formula_parts["mid"]),
            "score_formula_long": float(formula_parts["long"]),
            "score_standard_short": float(standard_parts["short"]),
            "score_standard_mid": float(standard_parts["mid"]),
            "score_standard_long": float(standard_parts["long"]),
        }
        rows.append(row)
        completed.add(key)

        cp = {
            "mode": mode,
            "updated_at_unix": time.time(),
            "elapsed_sec": time.time() - t0,
            "rows": rows,
            "completed_keys": sorted(completed),
        }
        save_json(cp_path, cp)
        pbar.set_postfix(
            d=d,
            L=L,
            best=f"{best_score:.5f}",
            formula=f"{score_formula:.5f}",
            std=f"{score_standard:.5f}",
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["d", "L"]).reset_index(drop=True)
    out_csv = Path(config.csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df
