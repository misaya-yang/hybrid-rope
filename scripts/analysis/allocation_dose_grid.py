#!/usr/bin/env python3
"""Freeze a fixed-support allocation dose-response family and its static prediction.

CPU only.  No checkpoint, no GPU, no LM loss.

This script does **not** propose a new static score that selects a table.  The
closed-route ledger (`INDEX.md` §3.4) rejects that class, and this file stays
outside it: every static quantity here is computed on a *pre-declared* dose
grid whose members are fixed by construction, and the quantities are written
into a prediction receipt that is frozen **before** any language-model number
is read.  The behavioural sweep then either confirms or falsifies the
prediction.  A selector claims a table is good; this is a stake on the theory
that an independent measurement can take away.

Two dose paths share exact Native endpoints, so only the interior allocation
`z` in `x_k = -log omega_k = a + R z_k` moves:

*   **Path A (theory).**  `z(lambda) = (1 - lambda) z_geo + lambda z_cosh(tau)`
    interpolates the geometric Native table toward anchored EVQ-Cosh quantiles
    normalised to the Native endpoints.  `lambda = 0` is bitwise Native and
    `lambda = 1` is the anchored construction the manuscript already owns.
    Nothing between those two endpoints has ever been measured.
*   **Path B (empirical).**  `gaps proportional to exp(lambda * delta)` scales
    the direction that the 2026-08-25 co-adaptive oracle actually learned.
    `lambda = 1` reproduces that table bitwise, so the sweep says whether its
    measured effect grows with displacement or is a fixed-point artefact.

Every emitted table is float32, shape `(K,)`, strictly decreasing, with both
endpoints bitwise equal to Native, which is the identity contract
`scripts/eval/target_free_ruler_smoke.py` enforces for `--table`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.third_axis_ceiling import (  # noqa: E402
    characteristic,
    measure_weights,
    recurrence_gap,
    stable_rank,
)
from scripts.lib.rope.schedules import evq_cosh_phi, geometric_inv_freq  # noqa: E402

torch.set_default_dtype(torch.float64)

STATUS = "ALLOCATION_DOSE_GRID_FROZEN_V1"
PATH_A_LAMBDAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00)
PATH_A_SHAPE_CONTROL_LAMBDAS = (0.10, 0.35, 0.70)
PATH_B_LAMBDAS = (1.0, 4.0, 16.0, 64.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-count", type=int, default=64)
    parser.add_argument("--base", type=float, default=500_000.0)
    parser.add_argument("--native-length", type=int, default=4096)
    parser.add_argument("--primary-tau", type=float, default=4.0)
    parser.add_argument("--shape-control-tau", type=float, default=2.0)
    parser.add_argument(
        "--learned-table",
        type=Path,
        help="oracle allocation_table.npy; enables Path B when supplied",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def float32_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def anchored_cosh_z(pairs: int, tau: float) -> torch.Tensor:
    """EVQ-Cosh quantiles renormalised so both Native endpoints are exact."""

    phi = evq_cosh_phi(pairs, tau=tau, midpoint=True)
    z = (phi - phi[0]) / (phi[-1] - phi[0])
    if not bool((z[1:] > z[:-1]).all()):
        raise RuntimeError(f"anchored Cosh quantiles are not increasing at tau={tau}")
    return z


def learned_direction(path: Path, native: torch.Tensor) -> torch.Tensor:
    """Recover centred gap logits from a realised oracle table."""

    learned = torch.as_tensor(np.load(path, allow_pickle=False), dtype=torch.float64)
    if learned.shape != native.shape:
        raise ValueError("learned table shape does not match the Native table")
    x_native = -native.log()
    span = x_native[-1] - x_native[0]
    z = (-learned.log() - x_native[0]) / span
    gaps = torch.diff(z)
    if not bool((gaps > 0).all()):
        raise RuntimeError("learned table is not strictly ordered")
    logits = (gaps * (native.numel() - 1)).log()
    return logits - logits.mean()


def z_from_gap_logits(logits: torch.Tensor) -> torch.Tensor:
    gaps = torch.softmax(logits, dim=0)
    return torch.cat((torch.zeros(1, dtype=gaps.dtype), torch.cumsum(gaps, dim=0)))


def realise(z: torch.Tensor, native: torch.Tensor) -> np.ndarray:
    """Map normalised coordinates back to float32 frequencies with exact endpoints."""

    x_native = -native.log()
    omega = torch.exp(-(x_native[0] + (x_native[-1] - x_native[0]) * z))
    table = omega.to(torch.float32).numpy().astype("<f4")
    table[0] = np.float32(native[0].item())
    table[-1] = np.float32(native[-1].item())
    if not np.all(table[:-1] > table[1:]):
        raise RuntimeError("realised float32 table is not strictly decreasing")
    if not np.isfinite(table).all():
        raise RuntimeError("realised table is not finite")
    return table


def static_diagnostics(
    table: np.ndarray, native: torch.Tensor, length: int
) -> dict[str, float]:
    """Canonical static geometry, computed by the tracked owner, not re-derived."""

    omega = torch.as_tensor(table, dtype=torch.float64)
    weights = measure_weights(length, "causal")
    r2, cbar = stable_rank(omega, weights)

    x_native = -native.log()
    span = x_native[-1] - x_native[0]
    z = (-omega.log() - x_native[0]) / span
    z_native = (x_native - x_native[0]) / span

    # Pair Gram spectrum of a single rotary pair under the exact causal measure.
    chi = characteristic(weights, 2.0 * omega).abs()
    pair_volume = float((1.0 - chi**2).clamp_min(0.0).mean())
    isotropy = float(((1.0 - chi) / (1.0 + chi)).mean())

    return {
        "r2_causal": float(r2),
        "cbar_causal": float(cbar),
        "median_normalized_coordinate": float(z.median()),
        "max_abs_coordinate_shift": float((z - z_native).abs().max()),
        "min_normalized_gap": float(torch.diff(z).min()),
        "mean_pair_volume": pair_volume,
        "mean_phase_isotropy": isotropy,
        "recurrence_gap_1x": recurrence_gap(omega, length),
        "recurrence_gap_4x": recurrence_gap(omega, 4 * length),
    }


def direction_alignment(
    table: np.ndarray, native: torch.Tensor, reference_z: torch.Tensor
) -> dict[str, float]:
    """Cosine alignment of a table's z-displacement with a reference direction.

    Both arguments are displacements from the geometric Native coordinates, so
    this compares *shape of reallocation* independently of how far the table
    travelled.  It is a descriptive geometric quantity, not a score.
    """

    x_native = -native.log()
    span = x_native[-1] - x_native[0]
    z_native = (x_native - x_native[0]) / span
    z = (-torch.as_tensor(table, dtype=torch.float64).log() - x_native[0]) / span
    moved = z - z_native
    reference = reference_z - z_native
    norm = float(moved.norm()) * float(reference.norm())
    interior = slice(1, -1)
    return {
        "displacement_norm": float(moved.norm()),
        "cosine_with_reference": float(moved @ reference) / norm if norm > 0 else 0.0,
        "displacement_norm_ratio_to_reference": (
            float(moved.norm()) / float(reference.norm()) if float(reference.norm()) > 0 else 0.0
        ),
        "interior_sign_agreement_with_reference": float(
            ((moved[interior] * reference[interior]) > 0).to(torch.float64).mean()
        ),
    }


def main() -> int:
    args = parse_args()
    pairs = int(args.pair_count)
    length = int(args.native_length)
    native = geometric_inv_freq(head_dim=2 * pairs, base=float(args.base))
    z_geo = torch.linspace(0.0, 1.0, pairs, dtype=torch.float64)

    output = args.output.resolve()
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, str, float, torch.Tensor]] = []
    for lam in PATH_A_LAMBDAS:
        z = (1.0 - lam) * z_geo + lam * anchored_cosh_z(pairs, float(args.primary_tau))
        plan.append((f"coshA_tau{args.primary_tau:g}_lam{lam:g}", "path_a_theory", lam, z))
    for lam in PATH_A_SHAPE_CONTROL_LAMBDAS:
        z = (1.0 - lam) * z_geo + lam * anchored_cosh_z(pairs, float(args.shape_control_tau))
        plan.append(
            (f"coshA_tau{args.shape_control_tau:g}_lam{lam:g}", "path_a_shape_control", lam, z)
        )
    if args.learned_table is not None:
        delta = learned_direction(args.learned_table.resolve(), native)
        for lam in PATH_B_LAMBDAS:
            plan.append(
                (f"learnedB_lam{lam:g}", "path_b_empirical", lam, z_from_gap_logits(lam * delta))
            )

    reference_z = anchored_cosh_z(pairs, float(args.primary_tau))
    native_float32 = native.to(torch.float32).numpy().astype("<f4")
    entries: list[dict[str, Any]] = []
    for name, family, lam, z in plan:
        table = realise(z, native)
        if not (
            table[0] == native_float32[0] and table[-1] == native_float32[-1]
        ):
            raise RuntimeError(f"{name}: sampled support endpoints moved")
        path = tables_dir / f"{name}.npy"
        np.save(path, table, allow_pickle=False)
        entry = {
            "name": name,
            "family": family,
            "lambda": float(lam),
            "path": str(path),
            "table_sha256_float32": float32_sha256(table),
            "table_file_sha256": sha256_file(path),
            "is_bitwise_native": bool(np.array_equal(table, native_float32)),
            **static_diagnostics(table, native, length),
        }
        if np.array_equal(table, native_float32):
            entry.update(
                {
                    "displacement_norm": 0.0,
                    "cosine_with_reference": None,
                    "displacement_norm_ratio_to_reference": 0.0,
                    "interior_sign_agreement_with_reference": None,
                }
            )
        else:
            entry.update(direction_alignment(table, native, reference_z))
        entries.append(entry)
        print(
            f"{name:28s} lam={lam:6.3f}  r2={entry['r2_causal']:8.4f}  "
            f"medz={entry['median_normalized_coordinate']:.4f}  "
            f"maxdz={entry['max_abs_coordinate_shift']:.5f}"
            + (
                "  cos_evq=   n/a"
                if entry["cosine_with_reference"] is None
                else f"  cos_evq={entry['cosine_with_reference']:+.4f}"
            )
        )

    primary = [e for e in entries if e["family"] == "path_a_theory"]
    primary.sort(key=lambda e: e["lambda"])
    best_r2 = max(primary, key=lambda e: e["r2_causal"])
    monotone_r2 = all(
        primary[i]["r2_causal"] <= primary[i + 1]["r2_causal"] + 1e-9
        for i in range(len(primary) - 1)
    )

    prediction = {
        "status": "ALLOCATION_DOSE_PREDICTION_FROZEN_V1",
        "declared_before_any_language_model_number": True,
        "role": "falsifiable prediction, not a table selector",
        "primary_family": "path_a_theory",
        "static_owner": "scripts/analysis/third_axis_ceiling.py",
        "measure": "causal triangular p(d) proportional to (L-d)",
        "length": length,
        "P1_r2_monotone_increasing_in_lambda": monotone_r2,
        "P1_argmax_r2_lambda": best_r2["lambda"],
        "P2_predicted_in_window_4k_nll_monotone_increasing_in_lambda": True,
        "P3_predicted_interior_minimum_of_16k_tail_nll_at_lambda_gt_0": True,
        "P4_predicted_lambda_star_closer_to_argmax_r2_than_to_zero": True,
        "falsifier_closing_the_route": (
            "If 16K tail NLL is monotone non-decreasing in lambda over the whole "
            "Path A grid, then zero-training fixed-support reallocation buys no "
            "target-free long-range gain on this checkpoint and the route closes."
        ),
        "path_b_alignment_with_theory_direction": {
            entry["name"]: {
                "cosine_with_anchored_cosh": entry["cosine_with_reference"],
                "displacement_norm_ratio": entry["displacement_norm_ratio_to_reference"],
                "interior_sign_agreement": entry["interior_sign_agreement_with_reference"],
            }
            for entry in entries
            if entry["family"] == "path_b_empirical"
        },
        "falsifier_for_path_b": (
            "If the measured effect of learnedB_lam1 does not grow in magnitude "
            "through lam4 and lam16, the 2026-08-25 oracle table's long-range "
            "effect is not a dose effect and must not carry a mechanism claim."
        ),
    }

    manifest = {
        "status": STATUS,
        "pair_count": pairs,
        "base": float(args.base),
        "native_length": length,
        "primary_tau": float(args.primary_tau),
        "shape_control_tau": float(args.shape_control_tau),
        "native_table_sha256_float32": float32_sha256(native_float32),
        "learned_table": str(args.learned_table.resolve()) if args.learned_table else None,
        "learned_table_file_sha256": (
            sha256_file(args.learned_table.resolve()) if args.learned_table else None
        ),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "tables": entries,
    }
    (output / "dose_grid_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "prediction.json").write_text(
        json.dumps(prediction, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nfrozen {len(entries)} tables -> {output}")
    print(f"P1 r2 monotone in lambda: {monotone_r2}; argmax r2 at lambda={best_r2['lambda']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
