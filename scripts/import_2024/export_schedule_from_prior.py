#!/usr/bin/env python3
"""
Export discrete RoPE inv_freq schedule from empirical D(Delta) prior.

This script provides a reproducible entrypoint for anonymous code release:
- input: prior_fit JSON from run_attn_hist.py (requires overall_hist)
- output: discrete inv_freq tensor via inverse-CDF discretization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def load_prior_hist(path: Path) -> np.ndarray:
    obj = json.loads(path.read_text(encoding="utf-8"))
    hist = obj.get("overall_hist")
    if not isinstance(hist, list) or len(hist) < 8:
        raise RuntimeError("prior json must contain overall_hist from run_attn_hist.py --save_hist")
    arr = np.asarray(hist, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    arr /= max(float(np.sum(arr)), 1e-12)
    return arr


def prior_for_L(hist: np.ndarray, L: int) -> np.ndarray:
    src_x = np.arange(1, len(hist), dtype=np.float64)
    src_y = np.asarray(hist[1:], dtype=np.float64)
    src_y = np.maximum(src_y, 1e-16)
    tgt_x = np.arange(1, L + 1, dtype=np.float64)
    if L <= len(src_y):
        y = src_y[:L].copy()
    else:
        y = np.interp(tgt_x, src_x, src_y, left=src_y[0], right=src_y[-1])
    y = np.maximum(y, 1e-16)
    y /= max(float(np.sum(y)), 1e-12)
    return y


def diag_opt_density(prior: np.ndarray, b: float, phi_points: int) -> Tuple[np.ndarray, np.ndarray]:
    L = len(prior)
    phi = np.linspace(0.0, 1.0, int(phi_points), dtype=np.float64)
    omega = np.power(float(b), -phi, dtype=np.float64)

    e_diag = np.zeros_like(phi)
    chunk = 2048
    for s in range(1, L + 1, chunk):
        e = min(L, s + chunk - 1)
        delta = np.arange(s, e + 1, dtype=np.float64)
        cosv = np.cos(np.outer(omega, delta))
        w = prior[s - 1 : e]
        e_diag += np.sum((cosv * cosv) * w[None, :], axis=1)

    e_diag = np.maximum(e_diag, 1e-12)
    rho = 1.0 / e_diag
    rho = np.maximum(rho, 1e-16)
    rho /= max(float(np.sum(rho)), 1e-12)
    return phi, rho


def inverse_cdf_discretize(phi: np.ndarray, rho: np.ndarray, n_bins: int) -> np.ndarray:
    cdf = np.cumsum(rho)
    cdf = cdf / max(float(cdf[-1]), 1e-12)
    q = (np.arange(n_bins, dtype=np.float64) + 0.5) / float(n_bins)
    phi_discrete = np.interp(q, cdf, phi)
    return np.clip(phi_discrete, 0.0, 1.0)


def quantization_error(phi: np.ndarray, rho: np.ndarray, phi_discrete: np.ndarray) -> float:
    cdf = np.cumsum(rho)
    cdf = cdf / max(float(cdf[-1]), 1e-12)
    n = len(phi_discrete)
    q = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    cdf_at_discrete = np.interp(phi_discrete, phi, cdf)
    return float(np.mean(np.abs(cdf_at_discrete - q)))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export discrete inv_freq schedule from empirical prior")
    ap.add_argument("--prior_json", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, required=True)
    ap.add_argument("--L", type=int, default=16384)
    ap.add_argument("--base", type=float, default=500000.0)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--phi_points", type=int, default=256)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.head_dim % 2 != 0:
        raise ValueError("head_dim must be even")

    hist = load_prior_hist(Path(args.prior_json))
    prior = prior_for_L(hist, int(args.L))
    phi, rho = diag_opt_density(prior=prior, b=float(args.base), phi_points=int(args.phi_points))

    n_bins = args.head_dim // 2
    phi_disc = inverse_cdf_discretize(phi, rho, n_bins=n_bins)
    inv_freq = np.power(float(args.base), -phi_disc, dtype=np.float64)
    qerr = quantization_error(phi=phi, rho=rho, phi_discrete=phi_disc)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "meta": {
            "prior_json": str(Path(args.prior_json).resolve()),
            "L": int(args.L),
            "base": float(args.base),
            "head_dim": int(args.head_dim),
            "phi_points": int(args.phi_points),
            "n_bins": int(n_bins),
            "quantization_error_mean_abs_cdf": float(qerr),
        },
        "phi_discrete": [float(x) for x in phi_disc.tolist()],
        "inv_freq": [float(x) for x in inv_freq.tolist()],
    }

    out_json = out_prefix.with_suffix(".json")
    out_npy = out_prefix.with_suffix(".npy")
    out_json.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(str(out_npy), inv_freq)

    if torch is not None:
        out_pt = out_prefix.with_suffix(".pt")
        torch.save(torch.tensor(inv_freq, dtype=torch.float32), str(out_pt))

    print(json.dumps(json_payload["meta"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
