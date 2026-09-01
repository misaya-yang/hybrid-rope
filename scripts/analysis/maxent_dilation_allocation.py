#!/usr/bin/env python3
"""Build and audit deterministic MaxEnt dilation RoPE tables on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.rope.schedules import (  # noqa: E402
    geometric_inv_freq,
    maxent_dilation_factors,
    maxent_dilation_inv_freq,
)


STATUS = "MAXENT_DILATION_CPU_CONTRACT_OK"
DEFAULT_LAMBDAS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-table", type=Path)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--base", type=float, default=500_000.0)
    parser.add_argument("--target-factor", type=float, default=4.0)
    parser.add_argument(
        "--lambdas",
        default=",".join(str(value) for value in DEFAULT_LAMBDAS),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _float32_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_native(args: argparse.Namespace) -> torch.Tensor:
    if args.native_table is None:
        return geometric_inv_freq(
            head_dim=int(args.head_dim),
            base=float(args.base),
            dtype=torch.float64,
        )
    value = np.load(args.native_table.expanduser().resolve(), allow_pickle=False)
    return torch.as_tensor(np.asarray(value, dtype=np.float64)).reshape(-1)


def _parse_lambdas(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(not math.isfinite(item) for item in parsed):
        raise ValueError("--lambdas must contain finite comma-separated values")
    return parsed


def audit_table(
    native: torch.Tensor,
    *,
    target_factor: float,
    lambda_: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    active_t = maxent_dilation_inv_freq(
        native,
        target_factor=target_factor,
        lambda_=lambda_,
    ).to(torch.float64)
    dilation_t = maxent_dilation_factors(
        native.numel(),
        target_factor=target_factor,
        lambda_=lambda_,
        dtype=torch.float64,
    )
    native_np = native.detach().cpu().to(torch.float64).numpy()
    active = active_t.detach().cpu().numpy()
    dilation = dilation_t.detach().cpu().numpy()
    q = (np.arange(native.numel(), dtype=np.float64) + 0.5) / native.numel()
    tau = np.log(dilation)

    if abs(lambda_) < 1e-8:
        cdf = tau / math.log(target_factor)
    else:
        cdf = np.expm1(lambda_ * tau) / math.expm1(
            lambda_ * math.log(target_factor)
        )
    log_frequency = np.log(active)
    non_affine_curvature = float(np.max(np.abs(np.diff(log_frequency, n=2))))
    phase_displacement = native_np * (1.0 - 1.0 / dilation)
    reverse_displacement = native_np * (1.0 - 1.0 / dilation[::-1])

    receipt = {
        "lambda": float(lambda_),
        "target_factor": float(target_factor),
        "pair_count": int(native.numel()),
        "formula": (
            "r_i=[1+q_i(s^lambda-1)]^(1/lambda); "
            "omega_i_prime=omega_i/r_i; q_i=(i+1/2)/K"
        ),
        "minimum_dilation": float(dilation.min()),
        "maximum_dilation": float(dilation.max()),
        "mean_log_dilation": float(tau.mean()),
        "dilation_strictly_increasing": bool(np.all(np.diff(dilation) > 0.0)),
        "frequency_strictly_decreasing": bool(np.all(np.diff(active) < 0.0)),
        "fast_frequency_ratio": float(active[0] / native_np[0]),
        "slow_frequency_ratio": float(active[-1] / native_np[-1]),
        "fast_endpoint_moved": bool(active[0] != native_np[0]),
        "slow_endpoint_moved": bool(active[-1] != native_np[-1]),
        "maximum_quantile_cdf_error": float(np.max(np.abs(cdf - q))),
        "maximum_log_frequency_second_difference": non_affine_curvature,
        "sampled_map_is_affine": bool(non_affine_curvature < 1e-10),
        "ordered_phase_displacement_sum": float(phase_displacement.sum()),
        "reverse_pairing_phase_displacement_sum": float(
            reverse_displacement.sum()
        ),
        "ordered_pairing_is_no_worse": bool(
            phase_displacement.sum() <= reverse_displacement.sum() + 1e-15
        ),
        "native_sha256_float32": _float32_sha256(native_np),
        "active_sha256_float32": _float32_sha256(active),
    }
    return np.ascontiguousarray(active, dtype="<f4"), receipt


def run_contract(
    native: torch.Tensor,
    *,
    target_factor: float,
    lambdas: tuple[float, ...],
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    if target_factor <= 1.0:
        raise ValueError("contract requires target_factor > 1")
    reports = [
        audit_table(native, target_factor=target_factor, lambda_=value)
        for value in lambdas
    ]
    means = [receipt["mean_log_dilation"] for _, receipt in reports]
    if any(
        not receipt["dilation_strictly_increasing"]
        or not receipt["frequency_strictly_decreasing"]
        or not receipt["ordered_pairing_is_no_worse"]
        or receipt["maximum_quantile_cdf_error"] > 1e-12
        or not (1.0 < receipt["minimum_dilation"])
        or not (receipt["maximum_dilation"] < target_factor)
        for _, receipt in reports
    ):
        raise RuntimeError("MaxEnt dilation identity contract failed")
    if any(left >= right for left, right in zip(means, means[1:])):
        raise RuntimeError("mean log dilation must increase with lambda")

    lambda_zero = next(
        (table for table, receipt in reports if receipt["lambda"] == 0.0), None
    )
    if lambda_zero is None:
        raise RuntimeError("contract lambda grid must include zero")
    q = (np.arange(native.numel(), dtype=np.float64) + 0.5) / native.numel()
    expected_zero = native.detach().cpu().numpy() / np.power(target_factor, q)
    if not np.allclose(lambda_zero, expected_zero, rtol=1e-6, atol=0.0):
        raise RuntimeError("lambda=0 log-uniform limit drift")
    for _, receipt in reports:
        expected_affine = receipt["lambda"] == 0.0
        if bool(receipt["sampled_map_is_affine"]) != expected_affine:
            raise RuntimeError("affine/non-affine identity drift")
    return reports


def main() -> int:
    args = parse_args()
    native = _load_native(args)
    lambdas = _parse_lambdas(args.lambdas)
    reports = run_contract(
        native,
        target_factor=float(args.target_factor),
        lambdas=lambdas,
    )

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    if output_dir is not None:
        if output_dir.exists():
            raise FileExistsError("output-dir must be a new path")
        output_dir.mkdir(parents=True)
        for table, receipt in reports:
            label = str(receipt["lambda"]).replace("-", "m").replace(".", "p")
            np.save(output_dir / f"maxent_lambda_{label}.npy", table, allow_pickle=False)
        (output_dir / "receipt.json").write_text(
            json.dumps(
                {"status": STATUS, "tables": [item[1] for item in reports]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    print(json.dumps({
        "status": STATUS,
        "target_factor": float(args.target_factor),
        "lambdas": list(lambdas),
        "tables": [receipt for _, receipt in reports],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
