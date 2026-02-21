from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import curve_fit


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-18:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _error_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    rel_err = abs_err / np.clip(np.abs(y_true), 1e-12, None)
    return {
        "mae": float(np.mean(abs_err)),
        "max_abs_error": float(np.max(abs_err)),
        "mean_rel_error_pct": float(np.mean(rel_err) * 100.0),
        "max_rel_error_pct": float(np.max(rel_err) * 100.0),
    }


@dataclass
class FitResult:
    name: str
    ok: bool
    num_params: int
    params: Dict[str, float]
    stderr: Dict[str, float]
    r2: float
    mae: float
    max_abs_error: float
    mean_rel_error_pct: float
    max_rel_error_pct: float
    message: str

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "num_params": self.num_params,
            "params": self.params,
            "stderr": self.stderr,
            "r2": self.r2,
            "mae": self.mae,
            "max_abs_error": self.max_abs_error,
            "mean_rel_error_pct": self.mean_rel_error_pct,
            "max_rel_error_pct": self.max_rel_error_pct,
            "message": self.message,
        }


def fit_k_models(L_arr: np.ndarray, d_arr: np.ndarray, y: np.ndarray) -> List[FitResult]:
    xdata = np.vstack([L_arr.astype(np.float64), d_arr.astype(np.float64)])
    results: List[FitResult] = []

    def eval_fit(name: str, fn: Callable, p0: Tuple[float, ...], bounds: Tuple[Tuple[float, ...], Tuple[float, ...]], param_names: List[str]) -> None:
        try:
            popt, pcov = curve_fit(fn, xdata, y, p0=p0, bounds=bounds, maxfev=200000)
            pred = fn(xdata, *popt)
            r2 = _r2(y, pred)
            e = _error_stats(y, pred)
            perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
            params = {k: float(v) for k, v in zip(param_names, popt)}
            stderr = {k: float(v) for k, v in zip(param_names, perr)}
            results.append(
                FitResult(
                    name=name,
                    ok=True,
                    num_params=len(param_names),
                    params=params,
                    stderr=stderr,
                    r2=float(r2),
                    mae=float(e["mae"]),
                    max_abs_error=float(e["max_abs_error"]),
                    mean_rel_error_pct=float(e["mean_rel_error_pct"]),
                    max_rel_error_pct=float(e["max_rel_error_pct"]),
                    message="ok",
                )
            )
        except Exception as ex:
            results.append(
                FitResult(
                    name=name,
                    ok=False,
                    num_params=len(param_names),
                    params={},
                    stderr={},
                    r2=float("-inf"),
                    mae=float("inf"),
                    max_abs_error=float("inf"),
                    mean_rel_error_pct=float("inf"),
                    max_rel_error_pct=float("inf"),
                    message=str(ex),
                )
            )

    # A: k = c1 * ln(L) / d
    def k_a(x, c1):
        L, d = x
        return c1 * np.log(L) / d

    # B: k = c1 * ln(L)/d + c2
    def k_b(x, c1, c2):
        L, d = x
        return c1 * np.log(L) / d + c2

    # C: k = c1 * L^c2 / d
    def k_c(x, c1, c2):
        L, d = x
        return c1 * (L ** c2) / d

    # D: k = c1 * ln(L / c2) / d
    def k_d(x, c1, c2):
        L, d = x
        return c1 * np.log(L / np.clip(c2, 1e-6, None)) / d

    # E: k = c1 * ln(L) / d^c2
    def k_e(x, c1, c2):
        L, d = x
        return c1 * np.log(L) / (d ** c2)

    eval_fit("A", k_a, p0=(1.2,), bounds=((0.0,), (10.0,)), param_names=["c1"])
    eval_fit("B", k_b, p0=(1.0, 0.0), bounds=((0.0, -1.0), (10.0, 1.0)), param_names=["c1", "c2"])
    eval_fit("C", k_c, p0=(0.1, 0.2), bounds=((0.0, 0.0), (10.0, 1.5)), param_names=["c1", "c2"])
    eval_fit("D", k_d, p0=(1.0, 2.0), bounds=((0.0, 1e-3), (10.0, 1e5)), param_names=["c1", "c2"])
    eval_fit("E", k_e, p0=(1.0, 1.0), bounds=((0.0, 0.1), (10.0, 3.0)), param_names=["c1", "c2"])
    return results


def fit_x0_models(L_arr: np.ndarray, d_arr: np.ndarray, y: np.ndarray) -> List[FitResult]:
    xdata = np.vstack([L_arr.astype(np.float64), d_arr.astype(np.float64)])
    results: List[FitResult] = []

    def eval_fit(name: str, fn: Callable, p0: Tuple[float, ...], bounds: Tuple[Tuple[float, ...], Tuple[float, ...]], param_names: List[str]) -> None:
        try:
            popt, pcov = curve_fit(fn, xdata, y, p0=p0, bounds=bounds, maxfev=200000)
            pred = fn(xdata, *popt)
            r2 = _r2(y, pred)
            e = _error_stats(y, pred)
            perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
            params = {k: float(v) for k, v in zip(param_names, popt)}
            stderr = {k: float(v) for k, v in zip(param_names, perr)}
            results.append(
                FitResult(
                    name=name,
                    ok=True,
                    num_params=len(param_names),
                    params=params,
                    stderr=stderr,
                    r2=float(r2),
                    mae=float(e["mae"]),
                    max_abs_error=float(e["max_abs_error"]),
                    mean_rel_error_pct=float(e["mean_rel_error_pct"]),
                    max_rel_error_pct=float(e["max_rel_error_pct"]),
                    message="ok",
                )
            )
        except Exception as ex:
            results.append(
                FitResult(
                    name=name,
                    ok=False,
                    num_params=len(param_names),
                    params={},
                    stderr={},
                    r2=float("-inf"),
                    mae=float("inf"),
                    max_abs_error=float("inf"),
                    mean_rel_error_pct=float("inf"),
                    max_rel_error_pct=float("inf"),
                    message=str(ex),
                )
            )

    # A: x0 = c3 * d
    def x_a(x, c3):
        _, d = x
        return c3 * d

    # B: x0 = c3 * d + c4 * ln(L)
    def x_b(x, c3, c4):
        L, d = x
        return c3 * d + c4 * np.log(L)

    # C: x0 = c3 * (d/2 - 1)
    def x_c(x, c3):
        _, d = x
        return c3 * (d / 2.0 - 1.0)

    eval_fit("A", x_a, p0=(0.245,), bounds=((0.0,), (1.0,)), param_names=["c3"])
    eval_fit("B", x_b, p0=(0.24, 0.0), bounds=((0.0, -5.0), (1.0, 5.0)), param_names=["c3", "c4"])
    eval_fit("C", x_c, p0=(0.5,), bounds=((0.0,), (3.0,)), param_names=["c3"])
    return results


def choose_best_fit(results: List[FitResult], min_r2: float = 0.95) -> FitResult:
    ok = [r for r in results if r.ok]
    if not ok:
        raise RuntimeError("No valid fitting result.")
    qualified = [r for r in ok if r.r2 >= min_r2]
    if qualified:
        qualified.sort(key=lambda r: (r.num_params, -r.r2, r.max_rel_error_pct))
        return qualified[0]
    ok.sort(key=lambda r: (-r.r2, r.max_rel_error_pct, r.num_params))
    return ok[0]

