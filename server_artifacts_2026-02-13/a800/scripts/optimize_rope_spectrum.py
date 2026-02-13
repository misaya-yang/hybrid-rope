#!/usr/bin/env python3
"""Offline optimization for RoPE spectrum (COSD-RoPE).

Optimizes frequency set omega_k while preserving TI (linear phase per channel):
  phi_k(m) = omega_k * m

Objective (surrogate):
  J = lambda_coll * R_coll_soft + lambda_ood * R_ood + lambda_reg * R_reg

Outputs:
- candidates JSON with geometric baselines and optimized candidates
- plots: |S(Delta)| curves and Delta-V (OOD increment) over channels
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch


def now() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sinc(x: torch.Tensor) -> torch.Tensor:
    # sin(x)/x with stable x->0 limit.
    out = torch.ones_like(x)
    nz = x.abs() > 1e-12
    out[nz] = torch.sin(x[nz]) / x[nz]
    return out


def geometric_omega(theta: float, K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(0, K, device=device, dtype=dtype)
    return 1.0 / (theta ** (idx / K))


def make_bimodal_omega(
    theta: float,
    K: int,
    cut_low: float,
    cut_high: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not (0.0 < cut_low < cut_high < 1.0):
        raise ValueError("require 0 < cut_low < cut_high < 1")

    ref = geometric_omega(theta, K, device, dtype)
    n_left = max(1, int(round(K * cut_low)))
    n_right = max(1, int(round(K * (1.0 - cut_high))))

    left = ref[:n_left]
    right = ref[-n_right:]

    if left.numel() + right.numel() > K:
        n_right = max(1, K - left.numel())
        right = ref[-n_right:]

    keep = torch.cat([left, right], dim=0)

    if keep.numel() < K:
        # Fill remaining slots by log-linear interpolation between left end and right start,
        # but keep them compressed near edges to avoid heavy mid-band occupancy.
        need = K - keep.numel()
        lo = torch.log(right[0])
        hi = torch.log(left[-1])
        t = torch.linspace(0.0, 1.0, need + 2, device=device, dtype=dtype)[1:-1]
        t = 0.5 - 0.5 * torch.cos(math.pi * t)  # cosine easing
        mid = torch.exp(hi + (lo - hi) * t)
        keep = torch.cat([left, mid, right], dim=0)

    keep = torch.sort(keep, descending=True).values
    return keep[:K]


def parse_lambda_pairs(spec: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(":")
        lc = float(a)
        lo = float(b)
        pairs.append((lc, lo))
    if not pairs:
        raise ValueError("no valid lambda pairs")
    return pairs


def parse_theta_list(spec: str) -> List[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty theta list")
    return vals


@dataclass
class RiskConfig:
    L_train: int
    L_target: int
    tau: float = 30.0
    lambda_reg: float = 0.02
    opt_delta_stride: int = 8


class SpectrumRisk:
    def __init__(self, K: int, cfg: RiskConfig, device: torch.device, dtype: torch.dtype):
        self.K = K
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        self.delta_all = torch.arange(1, cfg.L_target + 1, device=device, dtype=dtype)
        self.delta_ood = torch.arange(cfg.L_train + 1, cfg.L_target + 1, device=device, dtype=dtype)
        stride = max(1, int(cfg.opt_delta_stride))
        self.delta_opt_all = self.delta_all[::stride]
        self.delta_opt_ood = self.delta_ood[::stride]

    def similarity(self, omega: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        # S(Delta) = mean_k cos(omega_k * Delta)
        phase = torch.outer(delta, omega)
        return torch.cos(phase).mean(dim=1)

    def V(self, omega: torch.Tensor, L: int) -> torch.Tensor:
        x = omega * (0.5 * float(L))
        return 1.0 - sinc(x).pow(2)

    def metrics(self, omega: torch.Tensor, for_opt: bool = False) -> Dict[str, torch.Tensor]:
        da = self.delta_opt_all if for_opt else self.delta_all
        do = self.delta_opt_ood if for_opt else self.delta_ood
        S_all = self.similarity(omega, da)
        S_ood = self.similarity(omega, do)

        abs_all = S_all.abs()
        abs_ood = S_ood.abs()

        r_coll_all = abs_all.max()
        r_coll_ood = abs_ood.max()

        tau = self.cfg.tau
        r_coll_soft_all = torch.logsumexp(tau * abs_all, dim=0) / tau
        r_coll_soft_ood = torch.logsumexp(tau * abs_ood, dim=0) / tau

        V_train = self.V(omega, self.cfg.L_train)
        V_target = self.V(omega, self.cfg.L_target)
        dV = (V_target - V_train).clamp(min=0.0)
        r_ood = dV.mean()

        logw = torch.log(omega)
        d2 = logw[:-2] - 2.0 * logw[1:-1] + logw[2:]
        r_reg = (d2.pow(2).mean() if d2.numel() > 0 else torch.tensor(0.0, device=self.device, dtype=self.dtype))

        return {
            "S_all": S_all,
            "S_ood": S_ood,
            "V_train": V_train,
            "V_target": V_target,
            "dV": dV,
            "r_coll_all": r_coll_all,
            "r_coll_ood": r_coll_ood,
            "r_coll_soft_all": r_coll_soft_all,
            "r_coll_soft_ood": r_coll_soft_ood,
            "r_ood": r_ood,
            "r_reg": r_reg,
        }


class OmegaParameterization(torch.nn.Module):
    def __init__(self, omega_init: torch.Tensor, noise_std: float = 0.05):
        super().__init__()
        if omega_init.ndim != 1:
            raise ValueError("omega_init must be 1D")
        if omega_init.numel() < 3:
            raise ValueError("need K>=3")

        self.K = int(omega_init.numel())
        self.logw_max = float(torch.log(omega_init[0]).item())
        self.logw_min = float(torch.log(omega_init[-1]).item())
        span = self.logw_max - self.logw_min
        if span <= 0:
            raise ValueError("omega_init must be strictly descending positive")
        self.span = span

        logw = torch.log(omega_init)
        gaps = logw[:-1] - logw[1:]
        gaps = gaps / gaps.sum() * span

        # Inverse softplus: x -> log(exp(x)-1)
        u0 = torch.log(torch.expm1(torch.clamp(gaps, min=1e-8)))
        if noise_std > 0:
            u0 = u0 + noise_std * torch.randn_like(u0)

        self.u = torch.nn.Parameter(u0)

    def forward(self) -> torch.Tensor:
        gaps = torch.nn.functional.softplus(self.u) + 1e-8
        gaps = gaps / gaps.sum() * self.span

        # logw[0] fixed at max.
        logs = [torch.tensor(self.logw_max, device=gaps.device, dtype=gaps.dtype)]
        cur = logs[0]
        for g in gaps:
            cur = cur - g
            logs.append(cur)
        logw = torch.stack(logs, dim=0)
        omega = torch.exp(logw)
        return omega


@dataclass
class Candidate:
    name: str
    source: str
    omega: List[float]
    metrics: Dict[str, float]
    extra: Dict[str, Any]


def optimize_one(
    name: str,
    omega_init: torch.Tensor,
    risk: SpectrumRisk,
    lambda_coll: float,
    lambda_ood: float,
    lambda_reg: float,
    steps: int,
    lr: float,
    noise_std: float,
    seed: int,
) -> Candidate:
    set_seed(seed)

    param = OmegaParameterization(omega_init, noise_std=noise_std).to(device=omega_init.device, dtype=omega_init.dtype)
    opt = torch.optim.Adam(param.parameters(), lr=lr)

    best = None
    best_state = None

    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        omega = param()
        m = risk.metrics(omega, for_opt=True)

        obj = lambda_coll * m["r_coll_soft_ood"] + lambda_ood * m["r_ood"] + lambda_reg * m["r_reg"]
        obj.backward()
        opt.step()

        val = float(obj.item())
        if best is None or val < best:
            best = val
            best_state = omega.detach().clone()

    assert best_state is not None
    fm = risk.metrics(best_state, for_opt=False)

    metrics = {
        "objective": float(best),
        "r_coll_all": float(fm["r_coll_all"].item()),
        "r_coll_ood": float(fm["r_coll_ood"].item()),
        "r_coll_soft_all": float(fm["r_coll_soft_all"].item()),
        "r_coll_soft_ood": float(fm["r_coll_soft_ood"].item()),
        "r_ood": float(fm["r_ood"].item()),
        "r_reg": float(fm["r_reg"].item()),
    }

    return Candidate(
        name=name,
        source="optimized",
        omega=[float(x) for x in best_state.cpu().tolist()],
        metrics=metrics,
        extra={
            "lambda_coll": lambda_coll,
            "lambda_ood": lambda_ood,
            "lambda_reg": lambda_reg,
            "seed": seed,
            "steps": steps,
            "lr": lr,
            "noise_std": noise_std,
        },
    )


def evaluate_named(name: str, source: str, omega: torch.Tensor, risk: SpectrumRisk, extra: Dict[str, Any] | None = None) -> Candidate:
    m = risk.metrics(omega)
    metrics = {
        "r_coll_all": float(m["r_coll_all"].item()),
        "r_coll_ood": float(m["r_coll_ood"].item()),
        "r_coll_soft_all": float(m["r_coll_soft_all"].item()),
        "r_coll_soft_ood": float(m["r_coll_soft_ood"].item()),
        "r_ood": float(m["r_ood"].item()),
        "r_reg": float(m["r_reg"].item()),
    }
    return Candidate(
        name=name,
        source=source,
        omega=[float(x) for x in omega.cpu().tolist()],
        metrics=metrics,
        extra=extra or {},
    )


def save_plots(
    out_dir: Path,
    risk: SpectrumRisk,
    candidates: List[Candidate],
    show_top: int,
) -> None:
    import matplotlib.pyplot as plt

    # Sort by objective-like key if present else r_coll_ood + r_ood.
    def score(c: Candidate) -> float:
        return float(c.metrics.get("objective", c.metrics["r_coll_soft_ood"] + c.metrics["r_ood"]))

    chosen = sorted(candidates, key=score)[:show_top]

    delta = risk.delta_all.cpu().numpy()

    plt.figure(figsize=(10, 5))
    for c in chosen:
        w = torch.tensor(c.omega, dtype=risk.dtype, device=risk.device)
        S = risk.similarity(w, risk.delta_all).abs().cpu().numpy()
        plt.plot(delta, S, label=c.name)
    plt.axvline(risk.cfg.L_train, color="gray", linestyle="--", linewidth=1)
    plt.ylim(0.0, 1.0)
    plt.xlim(1, risk.cfg.L_target)
    plt.xlabel("Delta")
    plt.ylabel("|S(Delta)|")
    plt.title("Collision Sidelobe Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "sidelobe_curves.png", dpi=150)
    plt.close()

    # dV per channel for selected candidates.
    K = risk.K
    x = np.arange(K)
    plt.figure(figsize=(10, 5))
    for c in chosen:
        w = torch.tensor(c.omega, dtype=risk.dtype, device=risk.device)
        dv = (risk.V(w, risk.cfg.L_target) - risk.V(w, risk.cfg.L_train)).clamp(min=0).cpu().numpy()
        plt.plot(x, dv, marker="o", markersize=2, linewidth=1.2, label=c.name)
    plt.xlabel("Channel index k (low -> high index)")
    plt.ylabel("Delta V = max(0, V_target - V_train)")
    plt.title("OOD Increment by Frequency Channel")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "ood_increment_channels.png", dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/opt/dfrope/results/cosd_search")
    ap.add_argument("--L_train", type=int, default=2048)
    ap.add_argument("--L_target", type=int, default=16384)
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--theta_ref", type=float, default=1000.0)
    ap.add_argument("--theta_sweep", type=str, default="1000,2000,5000,10000")
    ap.add_argument("--lambda_pairs", type=str, default="1:0,0.7:0.3,0.5:0.5,0.3:0.7,0:1")
    ap.add_argument("--lambda_reg", type=float, default=0.02)
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--opt_delta_stride", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--n_inits", type=int, default=3)
    ap.add_argument("--noise_std", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--show_top", type=int, default=8)
    ap.add_argument("--save_plots", action="store_true")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    if args.head_dim % 2 != 0:
        raise ValueError("head_dim must be even")

    K = args.head_dim // 2
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA unavailable")
    dtype = torch.float64

    risk = SpectrumRisk(
        K=K,
        cfg=RiskConfig(
            L_train=args.L_train,
            L_target=args.L_target,
            tau=args.tau,
            lambda_reg=args.lambda_reg,
            opt_delta_stride=args.opt_delta_stride,
        ),
        device=device,
        dtype=dtype,
    )

    out_root = Path(args.out_dir)
    run_dir = out_root / f"run_{now()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base = geometric_omega(args.theta_ref, K, device, dtype)

    candidates: List[Candidate] = []

    # Geometric baselines.
    for theta in parse_theta_list(args.theta_sweep):
        omg = geometric_omega(theta, K, device, dtype)
        candidates.append(
            evaluate_named(
                name=f"geom_theta_{theta:g}",
                source="baseline_geometric",
                omega=omg,
                risk=risk,
                extra={"theta": theta},
            )
        )

    # Bimodal heuristic baseline.
    bim = make_bimodal_omega(
        theta=args.theta_ref,
        K=K,
        cut_low=0.25,
        cut_high=0.70,
        device=device,
        dtype=dtype,
    )
    candidates.append(
        evaluate_named(
            name="bimodal_heuristic",
            source="baseline_bimodal",
            omega=bim,
            risk=risk,
            extra={"theta": args.theta_ref, "cut_low": 0.25, "cut_high": 0.70},
        )
    )

    # Random log-frequency baseline (same endpoints as theta_ref).
    log_hi = torch.log(base[0])
    log_lo = torch.log(base[-1])
    for i in range(2):
        u = torch.sort(torch.rand(K - 2, device=device, dtype=dtype), descending=True).values
        log_mid = log_lo + (log_hi - log_lo) * u
        rand_omg = torch.cat([base[:1], torch.exp(log_mid), base[-1:]], dim=0)
        rand_omg = torch.sort(rand_omg, descending=True).values
        candidates.append(
            evaluate_named(
                name=f"random_log_{i}",
                source="baseline_random",
                omega=rand_omg,
                risk=risk,
                extra={"theta_ref": args.theta_ref},
            )
        )

    # Optimized candidates over lambda grid.
    pairs = parse_lambda_pairs(args.lambda_pairs)
    for idx, (lc, lo) in enumerate(pairs):
        best_c = None
        for j in range(args.n_inits):
            s = args.seed + idx * 100 + j
            c = optimize_one(
                name=f"cosd_lc{lc:.2f}_lo{lo:.2f}_init{j}",
                omega_init=base,
                risk=risk,
                lambda_coll=lc,
                lambda_ood=lo,
                lambda_reg=args.lambda_reg,
                steps=args.steps,
                lr=args.lr,
                noise_std=args.noise_std,
                seed=s,
            )
            if best_c is None or c.metrics["objective"] < best_c.metrics["objective"]:
                best_c = c
        assert best_c is not None
        best_c.name = f"cosd_lc{lc:.2f}_lo{lo:.2f}_best"
        candidates.append(best_c)
        print(
            f"[opt] {best_c.name}: obj={best_c.metrics['objective']:.6f} "
            f"r_coll_ood={best_c.metrics['r_coll_ood']:.6f} r_ood={best_c.metrics['r_ood']:.6f}"
        )

    # Sort for display.
    def disp_score(c: Candidate) -> float:
        return float(c.metrics.get("objective", c.metrics["r_coll_soft_ood"] + c.metrics["r_ood"]))

    candidates = sorted(candidates, key=disp_score)

    payload = {
        "ts": now(),
        "config": {
            "L_train": args.L_train,
            "L_target": args.L_target,
            "head_dim": args.head_dim,
            "K": K,
            "theta_ref": args.theta_ref,
            "theta_sweep": parse_theta_list(args.theta_sweep),
            "lambda_pairs": pairs,
            "lambda_reg": args.lambda_reg,
            "tau": args.tau,
            "opt_delta_stride": args.opt_delta_stride,
            "steps": args.steps,
            "lr": args.lr,
            "n_inits": args.n_inits,
            "noise_std": args.noise_std,
            "seed": args.seed,
            "device": args.device,
        },
        "candidates": [
            {
                "name": c.name,
                "source": c.source,
                "metrics": c.metrics,
                "extra": c.extra,
                "omega": c.omega,
            }
            for c in candidates
        ],
    }

    out_json = run_dir / "candidates.json"
    out_json.write_text(json.dumps(payload, indent=2))
    (out_root / "latest.json").write_text(json.dumps(payload, indent=2))

    if args.save_plots:
        save_plots(run_dir, risk, candidates, show_top=args.show_top)

    print("\nTop candidates by objective-like score:")
    for c in candidates[: min(10, len(candidates))]:
        print(
            f"- {c.name:28s} src={c.source:18s} "
            f"Rcoll_ood={c.metrics['r_coll_ood']:.4f} Rood={c.metrics['r_ood']:.4f} "
            f"score={disp_score(c):.4f}"
        )

    print(f"\n[done] wrote {out_json}")


if __name__ == "__main__":
    main()
