#!/usr/bin/env python3
"""GPU positive control for the bounded-condition conjugacy optimizer."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.optimize_scale_conjugacy import (  # noqa: E402
    _power_norm,
    _project_condition,
    _rotation_stack,
    atomic_json,
    sampled_positions,
    sha256_file,
)


AUTHORIZATION_ENV = "SCALE_CONJUGACY_GPU_AUTHORIZED"


def run(output: Path, steps: int, seed: int) -> dict[str, object]:
    if os.environ.get(AUTHORIZATION_ENV) != "YES":
        raise RuntimeError(f"set {AUTHORIZATION_ENV}=YES for CUDA execution")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(seed)
    device, dtype = torch.device("cuda"), torch.float32
    frequencies = torch.tensor((0.3, 0.7, 1.1, 1.7), device=device, dtype=dtype)
    positions = torch.as_tensor(
        sampled_positions(8.0, 64, seed), device=device, dtype=dtype
    )
    source = _rotation_stack(torch, frequencies, positions)
    raw = torch.randn((8, 8), device=device, dtype=dtype)
    known, _ = torch.linalg.qr(raw)
    if torch.linalg.det(known) < 0:
        known[:, 0] *= -1
    target = known[None] @ source @ known.T[None]
    candidate = torch.nn.Parameter(
        torch.eye(8, device=device, dtype=dtype) + 0.01 * torch.randn((8, 8), device=device)
    )
    _project_condition(torch, candidate, 1.0)
    optimizer = torch.optim.Adam((candidate,), lr=0.03)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    vectors = torch.nn.functional.normalize(
        torch.randn((len(positions), 8, 1), device=device, generator=generator), dim=-2
    )
    best_loss, best = math.inf, candidate.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        inverse = torch.linalg.inv(candidate)
        error = candidate[None] @ source @ inverse[None] - target
        norms = _power_norm(torch, error, vectors, 10)
        loss = 0.05 * torch.logsumexp(norms / 0.05, dim=0)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite positive-control objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((candidate,), 10.0)
        optimizer.step()
        _project_condition(torch, candidate, 1.0)
        value = float(loss.detach().item())
        if value < best_loss:
            best_loss, best = value, candidate.detach().clone()

    def exact(matrix: object) -> float:
        inverse = torch.linalg.inv(matrix)
        error = matrix[None] @ source @ inverse[None] - target
        return float(torch.linalg.matrix_norm(error, ord=2).max().item())

    identity_error, known_error, recovered_error = (
        exact(torch.eye(8, device=device)),
        exact(known),
        exact(best),
    )
    passed = known_error <= 1e-5 and recovered_error <= 0.1 * identity_error
    matrix_path = output.with_suffix(".matrix.npy")
    if output.exists() or matrix_path.exists():
        raise FileExistsError("positive-control output already exists")
    np.save(matrix_path, best.cpu().numpy(), allow_pickle=False)
    payload: dict[str, object] = {
        "status": "PASS" if passed else "FAIL",
        "method_id": "scale_conjugacy_positive_control_v1",
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "seed": seed,
        "steps": steps,
        "identity_error": identity_error,
        "known_witness_error": known_error,
        "recovered_error": recovered_error,
        "required_recovered_error": 0.1 * identity_error,
        "matrix_file": matrix_path.name,
        "matrix_file_sha256": sha256_file(matrix_path),
        "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    result = run(args.output.resolve(), args.steps, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
