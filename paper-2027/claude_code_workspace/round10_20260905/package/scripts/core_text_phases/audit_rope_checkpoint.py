#!/usr/bin/env python3
"""Audit RoPE inverse-frequency provenance for EVQ-Cosh checkpoints.

This is a lightweight, read-only checker. It does not instantiate a model; it
only inspects a checkpoint/state_dict or an inv_freq file and compares the
actual frequencies against canonical schedules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.rope.schedules import (  # noqa: E402
    evq_cosh_inv_freq,
    geometric_inv_freq,
)


def _to_tensor(x: Any) -> torch.Tensor | None:
    if torch.is_tensor(x):
        return x.detach().cpu().to(torch.float64).view(-1)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).detach().cpu().to(torch.float64).view(-1)
    return None


def _load_state_dict(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                return nested
        return obj
    raise TypeError(f"{path} did not load to a dict-like checkpoint")


def _candidate_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    names = [
        "inv_freq.npy",
        "custom_inv_freq.pt",
        "model.pt",
        "model_50pct.pt",
        "model_75pct.pt",
        "model_state_dict.pt",
    ]
    return [path / name for name in names if (path / name).exists()]


def load_inv_freq(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load inv_freq from a file or directory.

    Returns the first representative inv_freq tensor and metadata about all
    buffers found.
    """
    tried: list[str] = []
    for candidate in _candidate_paths(path):
        tried.append(str(candidate))
        if candidate.suffix == ".npy":
            inv = _to_tensor(np.load(candidate))
            if inv is not None:
                return inv, {"source": str(candidate), "kind": "npy", "num_buffers": 1}

        if candidate.suffix in {".pt", ".pth", ".bin"}:
            obj = torch.load(candidate, map_location="cpu", weights_only=True)
            if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
                inv = _to_tensor(obj)
                if inv is not None:
                    return inv, {"source": str(candidate), "kind": "tensor", "num_buffers": 1}
            if isinstance(obj, dict) and torch.is_tensor(obj.get("inv_freq")):
                inv = _to_tensor(obj["inv_freq"])
                return inv, {
                    "source": str(candidate),
                    "kind": "dict_inv_freq",
                    "num_buffers": 1,
                    "metadata": {k: v for k, v in obj.items() if k != "inv_freq"},
                }

            state = _load_state_dict(candidate)
            buffers: list[tuple[str, torch.Tensor]] = []
            for name, value in state.items():
                if "inv_freq" not in name:
                    continue
                inv = _to_tensor(value)
                if inv is not None and inv.ndim == 1 and inv.numel() > 0:
                    buffers.append((name, inv))
            if buffers:
                first_name, first = buffers[0]
                max_mismatch = 0.0
                mismatched: list[str] = []
                for name, inv in buffers[1:]:
                    if inv.numel() != first.numel():
                        mismatched.append(name)
                        max_mismatch = float("inf")
                        continue
                    diff = float((inv - first).abs().max().item())
                    if diff > 1e-8:
                        mismatched.append(name)
                        max_mismatch = max(max_mismatch, diff)
                return first, {
                    "source": str(candidate),
                    "kind": "state_dict",
                    "first_buffer": first_name,
                    "num_buffers": len(buffers),
                    "mismatched_buffers": mismatched,
                    "max_buffer_mismatch": max_mismatch,
                }

    raise FileNotFoundError(
        "Could not find inv_freq in path. Tried: " + ", ".join(tried)
    )


def infer_tau_from_path(path: Path) -> float | None:
    match = re.search(r"tau([0-9]+(?:\.[0-9]+)?)", str(path))
    if match:
        return float(match.group(1))
    return None


def load_nearby_metadata(path: Path) -> dict[str, Any]:
    base = path if path.is_dir() else path.parent
    out: dict[str, Any] = {}
    for name in ("results.json", "config.json", "metadata.json"):
        p = base / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        out[name] = data
    return out


def tensor_hash(x: torch.Tensor) -> str:
    arr = x.detach().cpu().to(torch.float64).contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()


def compare(actual: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = actual - ref.to(actual.dtype)
    return {
        "l2": float(torch.linalg.vector_norm(diff).item()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
    }


def estimate_tau(
    actual: torch.Tensor,
    rope_dim: int,
    base: float,
    tau_max: float,
    steps: int,
) -> tuple[float, float]:
    best_tau = 0.0
    best_err = float("inf")
    for i in range(steps + 1):
        tau = tau_max * i / steps
        ref = evq_cosh_inv_freq(rope_dim, tau=tau, base=base).to(torch.float64)
        err = float((actual - ref).abs().max().item())
        if err < best_err:
            best_tau = tau
            best_err = err
    return best_tau, best_err


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.checkpoint).expanduser()
    actual, source_meta = load_inv_freq(path)
    n_freqs = int(actual.numel())
    rope_dim = args.rope_dim or args.d_rope or (2 * n_freqs)
    d_head = args.d_head or rope_dim
    d_eff = args.d_eff or d_head
    tau = args.tau
    if tau is None:
        tau = infer_tau_from_path(path)

    if rope_dim != 2 * n_freqs:
        raise ValueError(
            f"rope_dim={rope_dim} implies {rope_dim // 2} frequencies, "
            f"but actual inv_freq has {n_freqs}"
        )

    base = float(args.base)
    phi = -torch.log(actual.clamp_min(torch.finfo(torch.float64).tiny)) / math.log(base)
    phi_checksum = tensor_hash(phi)

    geo_endpoint = geometric_inv_freq(rope_dim, base=base).to(torch.float64)
    geo_midpoint = evq_cosh_inv_freq(rope_dim, tau=0.0, base=base).to(torch.float64)

    comparisons: dict[str, Any] = {
        "geo_endpoint": compare(actual, geo_endpoint),
        "geo_midpoint_tau0": compare(actual, geo_midpoint),
    }
    if tau is not None:
        evq_ref = evq_cosh_inv_freq(rope_dim, tau=tau, base=base).to(torch.float64)
        comparisons[f"evq_cosh_tau_{tau:g}"] = compare(actual, evq_ref)

    best_tau, best_tau_maxdiff = estimate_tau(
        actual, rope_dim=rope_dim, base=base, tau_max=args.estimate_tau_max,
        steps=args.estimate_tau_steps,
    )

    closest_name = min(comparisons, key=lambda k: comparisons[k]["max_abs"])
    closest = comparisons[closest_name]
    evq_status = "unknown"
    if tau is not None:
        evq_name = f"evq_cosh_tau_{tau:g}"
        evq_err = comparisons[evq_name]["max_abs"]
        geo_err = min(
            comparisons["geo_endpoint"]["max_abs"],
            comparisons["geo_midpoint_tau0"]["max_abs"],
        )
        if evq_err <= args.tolerance:
            evq_status = "matches_requested_evq" if tau > 0 else "matches_tau0_geo_midpoint"
        elif evq_err < geo_err:
            evq_status = "closer_to_requested_evq_than_geo"
        else:
            evq_status = "does_not_match_requested_evq"

    metadata = load_nearby_metadata(path)
    return {
        "checkpoint": str(path),
        "source": source_meta,
        "nearby_metadata": metadata,
        "rope_base": base,
        "tau_requested_or_inferred": tau,
        "d_eff": d_eff,
        "d_rope_or_drot": rope_dim,
        "d_head": d_head,
        "n_freqs": n_freqs,
        "actual_inv_freq": {
            "first_8": [float(x) for x in actual[:8]],
            "last_8": [float(x) for x in actual[-8:]],
            "min": float(actual.min().item()),
            "max": float(actual.max().item()),
            "sha256": tensor_hash(actual),
        },
        "phi_grid": {
            "first_8": [float(x) for x in phi[:8]],
            "last_8": [float(x) for x in phi[-8:]],
            "checksum_sha256": phi_checksum,
        },
        "comparisons": comparisons,
        "closest_reference": {"name": closest_name, **closest},
        "estimated_tau_grid": {
            "tau": best_tau,
            "max_abs": best_tau_maxdiff,
            "grid_max": args.estimate_tau_max,
            "grid_steps": args.estimate_tau_steps,
        },
        "classification": evq_status,
    }


def print_report(report: dict[str, Any]) -> None:
    print("=" * 78)
    print("RoPE checkpoint audit")
    print("=" * 78)
    print(f"checkpoint: {report['checkpoint']}")
    print(f"source: {report['source']}")
    print(
        "config: "
        f"base={report['rope_base']} tau={report['tau_requested_or_inferred']} "
        f"d_eff={report['d_eff']} d_rope/drot={report['d_rope_or_drot']} "
        f"d_head={report['d_head']} n_freqs={report['n_freqs']}"
    )
    inv = report["actual_inv_freq"]
    print(f"inv_freq sha256: {inv['sha256']}")
    print(f"inv_freq first8: {inv['first_8']}")
    print(f"inv_freq last8:  {inv['last_8']}")
    phi = report["phi_grid"]
    print(f"phi checksum: {phi['checksum_sha256']}")
    print(f"phi first8: {phi['first_8']}")
    print(f"phi last8:  {phi['last_8']}")
    print("\ncomparisons:")
    for name, cmp_data in report["comparisons"].items():
        print(
            f"  {name:<22} "
            f"l2={cmp_data['l2']:.6g} "
            f"max={cmp_data['max_abs']:.6g} "
            f"mean={cmp_data['mean_abs']:.6g}"
        )
    est = report["estimated_tau_grid"]
    print(
        f"\nestimated tau grid: tau={est['tau']:.6g}, "
        f"max_abs={est['max_abs']:.6g}"
    )
    closest = report["closest_reference"]
    print(
        f"closest reference: {closest['name']} "
        f"(max_abs={closest['max_abs']:.6g})"
    )
    print(f"classification: {report['classification']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit actual RoPE inv_freq values in a checkpoint or run directory."
    )
    parser.add_argument("checkpoint", help="model.pt, inv_freq.npy, custom_inv_freq.pt, or run directory")
    parser.add_argument("--base", type=float, default=500000.0, help="RoPE base/theta")
    parser.add_argument("--tau", type=float, default=None, help="Expected EVQ tau")
    parser.add_argument("--rope-dim", type=int, default=None, help="Full rotary dimension; defaults to 2*len(inv_freq)")
    parser.add_argument("--d-rope", type=int, default=None, help="Alias for MLA d_rope / d_rot")
    parser.add_argument("--d-head", type=int, default=None, help="Attention head dimension")
    parser.add_argument("--d-eff", type=int, default=None, help="Operating-rule d_eff")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--estimate-tau-max", type=float, default=8.0)
    parser.add_argument("--estimate-tau-steps", type=int, default=800)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
