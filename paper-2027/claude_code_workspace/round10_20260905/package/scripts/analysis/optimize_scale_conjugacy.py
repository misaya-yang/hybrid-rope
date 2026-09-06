#!/usr/bin/env python3
"""Fit bounded-condition real conjugacies for finite-window RoPE scaling.

This is a numerical candidate search for the operator premise in the finite
scale-covariance theorem.  It never treats the result as a proof or as an LM
selector.  Preflight validates inputs without importing PyTorch; execution is
CUDA-only and explicitly authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.finite_scale_covariance import (  # noqa: E402
    _load_manifest_tables,
    _matching_for_threshold,
    fourier_orbit_rank_report,
    phase_character_sup_error,
    sha256_file,
)


METHOD_ID = "bounded_condition_scale_conjugacy_v1"
AUTHORIZATION_ENV = "SCALE_CONJUGACY_GPU_AUTHORIZED"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".incomplete",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def float32_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def sampled_positions(horizon: float, count: int, seed: int) -> np.ndarray:
    if horizon <= 0 or count < 5:
        raise ValueError("positive horizon and at least five positions required")
    rng = np.random.default_rng(seed)
    grid_count = max(3, count // 2)
    values = np.concatenate((
        np.linspace(-horizon, horizon, grid_count),
        rng.uniform(-horizon, horizon, count - grid_count),
        np.asarray((-horizon, 0.0, horizon)),
    ))
    return np.unique(values.astype(np.float64))


def exact_identity_error(
    frequencies: Sequence[float], scale: float, level: int, horizon: float
) -> float:
    return max(
        phase_character_sup_error(frequency, scale**level * frequency, horizon)
        for frequency in frequencies
    )


def exact_block_permutation_error(
    frequencies: Sequence[float], scale: float, level: int, horizon: float
) -> dict[str, Any]:
    source = np.asarray(frequencies, dtype=np.float64)
    target = scale**level * source
    cost = np.asarray([
        [phase_character_sup_error(left, right, horizon) for left in target]
        for right in source
    ])
    candidates = np.unique(cost)
    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        middle = (lo + hi) // 2
        if _matching_for_threshold(cost, float(candidates[middle])) is None:
            lo = middle + 1
        else:
            hi = middle
    permutation = _matching_for_threshold(cost, float(candidates[lo]))
    if permutation is None:
        raise AssertionError("bottleneck permutation reconstruction failed")
    return {"error": float(candidates[lo]), "permutation": permutation}


def _rotation_stack(torch: Any, frequencies: Any, positions: Any) -> Any:
    angles = positions[:, None] * frequencies[None, :]
    cosine, sine = torch.cos(angles), torch.sin(angles)
    batch, pairs = angles.shape
    matrices = torch.zeros(
        (batch, 2 * pairs, 2 * pairs), dtype=angles.dtype, device=angles.device
    )
    index = torch.arange(pairs, device=angles.device)
    matrices[:, 2 * index, 2 * index] = cosine
    matrices[:, 2 * index, 2 * index + 1] = -sine
    matrices[:, 2 * index + 1, 2 * index] = sine
    matrices[:, 2 * index + 1, 2 * index + 1] = cosine
    return matrices


def _project_condition(torch: Any, matrix: Any, cap: float) -> None:
    with torch.no_grad():
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        log_singular = torch.log(singular) - torch.log(singular).mean()
        half_width = 0.5 * math.log(cap)
        singular = torch.exp(torch.clamp(log_singular, -half_width, half_width))
        matrix.copy_((left * singular[None, :]) @ right)


def _power_norm(torch: Any, matrices: Any, vectors: Any, iterations: int) -> Any:
    vector = vectors
    for _ in range(iterations):
        left = torch.nn.functional.normalize(matrices @ vector, dim=-2)
        vector = torch.nn.functional.normalize(
            matrices.transpose(-1, -2) @ left, dim=-2
        )
    return torch.linalg.vector_norm(matrices @ vector, dim=(-2, -1))


def _sampled_exact_error(
    torch: Any,
    matrix: Any,
    frequencies: Any,
    scale: float,
    level: int,
    positions: Any,
    chunk: int = 16,
) -> float:
    inverse = torch.linalg.inv(matrix)
    maximum = 0.0
    for start in range(0, len(positions), chunk):
        times = positions[start : start + chunk]
        source = _rotation_stack(torch, frequencies, times)
        target = _rotation_stack(torch, frequencies, scale**level * times)
        error = matrix[None] @ source @ inverse[None] - target
        values = torch.linalg.matrix_norm(error, ord=2)
        maximum = max(maximum, float(values.max().item()))
    return maximum


def optimize_one(
    torch: Any,
    frequencies: np.ndarray,
    scale: float,
    level: int,
    horizon: int,
    condition_cap: float,
    steps: int,
    train_count: int,
    eval_count: int,
    power_iterations: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    device = torch.device("cuda")
    dtype = torch.float32
    train_positions = sampled_positions(horizon, train_count, seed)
    eval_positions = sampled_positions(horizon, eval_count, seed + 1)
    torch.manual_seed(seed)
    frequency_tensor = torch.as_tensor(frequencies, dtype=dtype, device=device)
    train_tensor = torch.as_tensor(train_positions, dtype=dtype, device=device)
    eval_tensor = torch.as_tensor(eval_positions, dtype=dtype, device=device)
    source = _rotation_stack(torch, frequency_tensor, train_tensor)
    target = _rotation_stack(torch, frequency_tensor, scale**level * train_tensor)
    dimension = 2 * len(frequencies)
    initial = torch.eye(dimension, dtype=dtype, device=device)
    initial += 0.01 * torch.randn_like(initial)
    matrix = torch.nn.Parameter(initial)
    _project_condition(torch, matrix, condition_cap)
    optimizer = torch.optim.Adam((matrix,), lr=0.02)
    generator = torch.Generator(device=device).manual_seed(seed + 2)
    vectors = torch.randn(
        (len(train_tensor), dimension, 1),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    vectors = torch.nn.functional.normalize(vectors, dim=-2)
    best_loss = math.inf
    best_matrix = matrix.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        inverse = torch.linalg.inv(matrix)
        error = matrix[None] @ source @ inverse[None] - target
        norms = _power_norm(torch, error, vectors, power_iterations)
        loss = 0.05 * torch.logsumexp(norms / 0.05, dim=0)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite scale-conjugacy objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((matrix,), 10.0)
        optimizer.step()
        _project_condition(torch, matrix, condition_cap)
        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_matrix = matrix.detach().clone()
    singular = torch.linalg.svdvals(best_matrix)
    condition = float((singular.max() / singular.min()).item())
    sampled_error = _sampled_exact_error(
        torch,
        best_matrix,
        frequency_tensor,
        scale,
        level,
        eval_tensor,
    )
    identity = torch.eye(dimension, dtype=dtype, device=device)
    identity_sampled_error = _sampled_exact_error(
        torch,
        identity,
        frequency_tensor,
        scale,
        level,
        eval_tensor,
    )
    selected_source = "optimized"
    if identity_sampled_error < sampled_error:
        best_matrix = identity
        condition = 1.0
        sampled_error = identity_sampled_error
        selected_source = "identity_control"
    return ({
        "seed": seed,
        "condition_cap": condition_cap,
        "realized_condition_number": condition,
        "steps": steps,
        "train_positions": len(train_positions),
        "train_positions_sha256": hashlib.sha256(train_positions.tobytes()).hexdigest(),
        "eval_positions": len(eval_positions),
        "eval_positions_sha256": hashlib.sha256(eval_positions.tobytes()).hexdigest(),
        "best_smoothed_training_objective": best_loss,
        "identity_sampled_operator_error": identity_sampled_error,
        "selected_source": selected_source,
        "sampled_operator_error": sampled_error,
        "sampling_boundary": "Lower bound on the true continuous supremum; not a certificate.",
    }, best_matrix.cpu().numpy())


def _selected_tables(
    manifest_path: Path, requested: Sequence[str] | None
) -> dict[str, np.ndarray]:
    tables = _load_manifest_tables(manifest_path)
    names = sorted(tables) if not requested else list(requested)
    missing = sorted(set(names) - set(tables))
    if missing:
        raise ValueError(f"unknown table names: {missing}")
    return {name: tables[name] for name in names}


def _validate_args(args: argparse.Namespace) -> None:
    if any(value <= 1 for value in args.scales):
        raise ValueError("all scales must exceed one")
    if any(value <= 0 for value in args.levels + args.horizons):
        raise ValueError("levels and horizons must be positive")
    if any(value < 1 for value in args.condition_caps):
        raise ValueError("condition caps must be at least one")
    if min(args.steps, args.restarts, args.train_positions, args.eval_positions) <= 0:
        raise ValueError("steps, restarts, and position counts must be positive")
    if args.power_iterations <= 0:
        raise ValueError("power-iterations must be positive")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get(AUTHORIZATION_ENV) != "YES":
        raise RuntimeError(f"set {AUTHORIZATION_ENV}=YES for CUDA execution")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for scale-conjugacy optimization")
    tables = _selected_tables(args.table_manifest.resolve(), args.tables)
    output_path = args.output.resolve()
    matrix_root = output_path.with_suffix("").with_name(output_path.stem + "_matrices")
    incomplete_root = matrix_root.with_name(matrix_root.name + ".incomplete")
    if output_path.exists() or matrix_root.exists() or incomplete_root.exists():
        raise FileExistsError("output JSON or matrix directory already exists")
    incomplete_root.mkdir(parents=True)
    records = []
    for table_name, frequencies in tables.items():
        for scale in args.scales:
            for levels in args.levels:
                for horizon in args.horizons:
                    theorem = fourier_orbit_rank_report(
                        frequencies,
                        scale,
                        levels,
                        horizon,
                        real=True,
                        discrete=False,
                    )
                    for condition_cap in args.condition_caps:
                        per_level = []
                        for level in range(1, levels + 1):
                            restarts = []
                            best: tuple[dict[str, Any], np.ndarray] | None = None
                            for restart in range(args.restarts):
                                result = optimize_one(
                                    torch,
                                    frequencies,
                                    scale,
                                    level,
                                    horizon,
                                    condition_cap,
                                    args.steps,
                                    args.train_positions,
                                    args.eval_positions,
                                    args.power_iterations,
                                    args.seed + 1009 * restart + 9176 * level,
                                )
                                restarts.append(result[0])
                                if best is None or result[0]["sampled_operator_error"] < best[0][
                                    "sampled_operator_error"
                                ]:
                                    best = result
                            assert best is not None
                            filename = (
                                f"{table_name}_s{scale:g}_N{levels}_j{level}_"
                                f"L{horizon}_kappa{condition_cap:g}.npy"
                            )
                            matrix_path = incomplete_root / filename
                            np.save(matrix_path, best[1], allow_pickle=False)
                            per_level.append({
                                "level": level,
                                "identity_error": exact_identity_error(
                                    frequencies, scale, level, horizon
                                ),
                                "best_block_permutation": exact_block_permutation_error(
                                    frequencies, scale, level, horizon
                                ),
                                "restarts": restarts,
                                "selected_restart": best[0],
                                "matrix_file": filename,
                                "matrix_file_sha256": sha256_file(matrix_path),
                            })
                        sampled_error = max(
                            row["selected_restart"]["sampled_operator_error"]
                            for row in per_level
                        )
                        lower = theorem["ky_fan"]["epsilon_lower_bound"]
                        records.append({
                            "table": table_name,
                            "table_float32_sha256": float32_sha256(frequencies),
                            "scale": scale,
                            "levels": levels,
                            "horizon": horizon,
                            "condition_cap": condition_cap,
                            "theorem_ky_fan_lower_bound": lower,
                            "fixed_native_subspace_lower_bound": math.sqrt(
                                theorem["fixed_native_subspace_max_squared_residual"]
                            ),
                            "sampled_best_conjugacy_error": sampled_error,
                            "sampled_minus_ky_fan_gap": sampled_error - lower,
                            "per_level": per_level,
                        })
    payload = {
        "status": "SCALE_CONJUGACY_OPTIMIZATION_COMPLETE",
        "method_id": METHOD_ID,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "table_manifest_sha256": sha256_file(args.table_manifest.resolve()),
        "protocol": {
            "tables": list(tables),
            "scales": args.scales,
            "levels": args.levels,
            "horizons": args.horizons,
            "condition_caps": args.condition_caps,
            "steps": args.steps,
            "restarts": args.restarts,
            "train_positions": args.train_positions,
            "eval_positions": args.eval_positions,
            "power_iterations": args.power_iterations,
            "seed": args.seed,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "records": records,
        "claim_limit": (
            "Optimized sampled errors are numerical lower estimates of the true supremum. "
            "They test tightness but neither prove an upper bound nor predict LM behavior."
        ),
    }
    incomplete_root.replace(matrix_root)
    atomic_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tables", nargs="+")
    parser.add_argument("--scales", type=float, nargs="+", default=[4.0])
    parser.add_argument("--levels", type=int, nargs="+", default=[1])
    parser.add_argument("--horizons", type=int, nargs="+", default=[4096])
    parser.add_argument("--condition-caps", type=float, nargs="+", default=[1.0, 8.0, 64.0])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--train-positions", type=int, default=64)
    parser.add_argument("--eval-positions", type=int, default=256)
    parser.add_argument("--power-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    _validate_args(args)
    if not args.table_manifest.is_file():
        parser.error("table manifest does not exist")
    if not args.preflight_only and args.output is None:
        parser.error("--output is required unless --preflight-only is used")
    return args


def main() -> int:
    args = parse_args()
    tables = _selected_tables(args.table_manifest.resolve(), args.tables)
    if args.preflight_only:
        print(json.dumps({
            "status": "SCALE_CONJUGACY_PREFLIGHT_COMPLETE",
            "method_id": METHOD_ID,
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "table_manifest_sha256": sha256_file(args.table_manifest.resolve()),
            "torch_imported": "torch" in sys.modules,
            "cuda_initialized": False,
            "planned": {
                "tables": {name: float32_sha256(values) for name, values in tables.items()},
                "scales": args.scales,
                "levels": args.levels,
                "horizons": args.horizons,
                "condition_caps": args.condition_caps,
                "configurations": (
                    len(tables)
                    * len(args.scales)
                    * len(args.levels)
                    * len(args.horizons)
                    * len(args.condition_caps)
                ),
                "optimizations": (
                    len(tables)
                    * len(args.scales)
                    * sum(args.levels)
                    * len(args.horizons)
                    * len(args.condition_caps)
                    * args.restarts
                ),
            },
            "claim_limit": "Preflight loads no model and does not import PyTorch.",
        }, indent=2, sort_keys=True))
        return 0
    try:
        payload = run(args)
    except Exception as error:
        failure_path = args.output.resolve().with_suffix(args.output.suffix + ".failed.json")
        if not failure_path.exists():
            atomic_json(failure_path, {
                "status": "SCALE_CONJUGACY_OPTIMIZATION_FAILED",
                "method_id": METHOD_ID,
                "source_sha256": sha256_file(Path(__file__).resolve()),
                "error_type": type(error).__name__,
                "error": str(error),
            })
        raise
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
        "records": len(payload["records"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
