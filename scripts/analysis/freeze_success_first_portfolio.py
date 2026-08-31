#!/usr/bin/env python3
"""Freeze the success-first zero-training portfolio before any model output.

CPU only.  No checkpoint, no GPU, no language-model number.

This realises the ``R0`` readiness stage of the success-first tournament
(``ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830`` §10).  It materialises the
four candidate families' *construction inputs* and the frozen selection rule, so
that development, selection and confirmation all run against one hash-bound
portfolio:

*   **F1 PC-MORPH** — the full seven-point log-frequency morph grid from Native
    to the reconstructed phase-chord target;
*   **F2 PC-RETENTION-PROJECT** — the hat-basis projection of the phase-chord
    direction and the frozen trust-region constant (the calibrated step is found
    in development);
*   **F3 Z5-BEHAVIOUR** — the knot-parameterisation constants, the frozen
    optimiser/budget, and three predeclared initialisations (Native, the
    learned-direction teacher, the coarse budgeted direction); the fourth init,
    the F1 family winner, is bound after F1 development selects it;
*   **F4 SR-Z5** — F3 plus the frozen support-factor grid.

The portfolio is not a result.  It binds identities; a missing raw owner, a
hash drift, a non-monotone table, or an unsupported endpoint stops the freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.rope.hat_projection import (  # noqa: E402
    hat_basis,
    phase_chord_direction,
    project_direction_onto_basis,
)
from scripts.lib.rope.knot_allocation import (  # noqa: E402
    INTERIOR_KNOTS,
    init_gap_logits_from_table,
    knot_positions,
)

STATUS = "SUCCESS_FIRST_PORTFOLIO_FROZEN_V1"

F1_MORPH_GRID = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50)
F4_SUPPORT_GRID = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
PHASE_LAMBDA = 0.1
PHASE_BINS = 64
GUARD_MARGIN = 0.01
PARITY_PER_TOKEN_TOL = 1e-3
PARITY_PREFIX_MEAN_TOL = 1e-4

# Frozen F3/F4 development constants (optimiser identity and budget).
F3_OPTIMIZER = {
    "name": "AdamW",
    "learning_rate": 3e-3,
    "weight_decay": 0.0,
    "steps": 40,
    "batch_documents": 8,
    "guard_margin": GUARD_MARGIN,
    "initialisations": ["native", "f1_winner", "learned_teacher", "coarse_budgeted"],
}
# Frozen F2 trust-region constant for the Native-loss quadratic model.
F2_TRUST_REGION = {
    "native_prefix_mean_delta": 0.5 * GUARD_MARGIN,
    "line_search_alpha_grid": [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
}
TIEBREAK_DOF = {"F1": 1, "F2": 1, "F3": INTERIOR_KNOTS, "F4": INTERIOR_KNOTS + 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-collection", type=Path, required=True)
    parser.add_argument("--r0-expected-sha256", default=None)
    parser.add_argument("--learned-table", type=Path, required=True)
    parser.add_argument("--budgeted-table", type=Path, required=True)
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


def _strict_quantiles(grid: np.ndarray, density: np.ndarray, count: int) -> np.ndarray:
    area = float(np.sum(0.5 * (density[:-1] + density[1:]) * np.diff(grid)))
    if not math.isfinite(area) or area <= 0.0:
        raise RuntimeError("phase-chord density has no finite positive integral")
    rho = density / area
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (rho[:-1] + rho[1:]) * np.diff(grid))))
    cdf /= cdf[-1]
    probabilities = np.linspace(0.0, 1.0, count, dtype=np.float64)
    phi = np.interp(probabilities, cdf, grid)
    phi[0], phi[-1] = 0.0, 1.0
    if not np.all(np.diff(phi) > 0.0):
        raise RuntimeError("phase-chord quantiles are not strictly increasing")
    return phi


def reconstruct_phase_chord_target(collection: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reconstruct Native and the lambda=0.1 phase-chord target from the R0 npz."""

    with np.load(collection, allow_pickle=False) as payload:
        native = np.asarray(payload["inv_freq"], dtype=np.float64)
        mass = np.asarray(payload["mass"], dtype=np.float64).sum(axis=(0, 1))
        metadata = json.loads(str(payload["metadata"].item()))
    if not np.all(native[:-1] > native[1:]) or native.size < 4:
        raise RuntimeError("R0 Native inv_freq is not a valid table")
    mass = mass.copy()
    mass[0] = 0.0
    if mass.sum() <= 0.0:
        raise RuntimeError("R0 has no non-self attention mass")
    distance_probability = mass / mass.sum()
    grid = np.linspace(0.0, 1.0, PHASE_BINS, dtype=np.float64)
    log_span = float(np.log(native[0] / native[-1]))
    probe_frequency = native[0] * np.exp(-log_span * grid)
    distance = np.arange(mass.size, dtype=np.float64)
    kernel = 1.0 - np.cos(distance[:, None] * probe_frequency[None, :])
    demand = np.sum(distance_probability[:, None] * kernel, axis=0)
    demand = np.maximum(demand, np.finfo(np.float64).tiny)
    demand /= float(np.sum(0.5 * (demand[:-1] + demand[1:]) * np.diff(grid)))
    density = np.cbrt((1.0 - PHASE_LAMBDA) * demand + PHASE_LAMBDA)
    phi = _strict_quantiles(grid, density, native.size)
    inv_freq = inv_freq_from_phi(phi, native)
    return native, inv_freq, {
        "kernel": "1-cos(omega(phi)*Delta)",
        "self_distance_excluded": True,
        "density": "((1-lambda)*normalised_phase_demand + lambda)^(1/3)",
        "bins": PHASE_BINS,
        "lambda": PHASE_LAMBDA,
        "r0_metadata_model_revision": metadata.get("model_revision"),
        "r0_metadata_tokens_sha256": metadata.get("tokens_sha256"),
    }


def inv_freq_from_phi(phi: np.ndarray, native: np.ndarray) -> np.ndarray:
    log_high = math.log(float(native[0]))
    log_low = math.log(float(native[-1]))
    table = np.exp(log_high + phi * (log_low - log_high))
    table[0], table[-1] = native[0], native[-1]
    if not np.all(table[:-1] > table[1:]):
        raise RuntimeError("phase-chord target is not strictly decreasing")
    return table


def log_morph(native: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
    amount = float(t)
    if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise RuntimeError("morph amount t must lie in [0,1]")
    if amount == 0.0:
        return native.astype(np.float64).copy()
    table = np.exp((1.0 - amount) * np.log(native) + amount * np.log(target))
    table[0], table[-1] = native[0], native[-1]
    if not np.all(table[:-1] > table[1:]):
        raise RuntimeError(f"morph t={amount:g} is not strictly decreasing")
    return table


def table_receipt(name: str, table: np.ndarray, native: np.ndarray) -> dict[str, Any]:
    log_displacement = np.log(table) - np.log(native)
    return {
        "name": name,
        "inv_freq_float32_sha256": float32_sha256(table),
        "is_bitwise_native": bool(np.array_equal(table.astype("<f4"), native.astype("<f4"))),
        "fast_endpoint_fixed": bool(table[0] == native[0]),
        "slow_endpoint_fixed": bool(table[-1] == native[-1]),
        "strictly_decreasing": bool(np.all(table[:-1] > table[1:])),
        "log_displacement_rms": float(np.sqrt(np.mean(np.square(log_displacement)))),
        "minimum_adjacent_log_gap": float(np.min(-np.diff(np.log(table)))),
    }


def main() -> int:
    args = parse_args()
    collection = args.r0_collection.resolve()
    r0_sha = sha256_file(collection)
    if args.r0_expected_sha256 and r0_sha != args.r0_expected_sha256:
        raise RuntimeError(
            f"R0 collection hash drift: expected {args.r0_expected_sha256}, got {r0_sha}"
        )

    output = args.output.resolve()
    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    native, target, chord_meta = reconstruct_phase_chord_target(collection)
    native_f32 = native.astype("<f4")

    # F1: full morph grid.
    f1_entries: list[dict[str, Any]] = []
    for t in F1_MORPH_GRID:
        table = log_morph(native, target, t)
        name = f"F1_pcmorph_t{t:g}".replace(".", "p")
        path = tables_dir / f"{name}.npy"
        np.save(path, table.astype("<f4"), allow_pickle=False)
        f1_entries.append({**table_receipt(name, table, native), "morph_t": float(t), "path": str(path)})

    # F2: hat-basis projection of the phase-chord direction.
    basis = hat_basis(native.size)
    direction = phase_chord_direction(native, target)
    coefficients = project_direction_onto_basis(direction, basis)
    reconstructed = basis.numpy() @ coefficients.numpy()
    f2_projection = {
        "basis_rows": int(basis.shape[0]),
        "basis_cols": int(basis.shape[1]),
        "direction_rms": float(np.sqrt(np.mean(np.square(direction)))),
        "coefficients": [float(v) for v in coefficients.numpy()],
        "coefficients_sha256": hashlib.sha256(
            np.ascontiguousarray(coefficients.numpy().astype("<f8")).tobytes()
        ).hexdigest(),
        "basis_projection_residual_rms": float(
            np.sqrt(np.mean(np.square(direction - reconstructed)))
        ),
        "trust_region": F2_TRUST_REGION,
        "construction_label": "GRADIENT_CALIBRATED_Z",
    }

    # F3: knot initialisations.
    learned = np.load(args.learned_table.resolve(), allow_pickle=False)
    budgeted = np.load(args.budgeted_table.resolve(), allow_pickle=False)
    f3_inits = {
        "native": {"source": "bitwise_native", "gap_logits": [0.0] * 6},
        "f1_winner": {"source": "bound_after_F1_development", "gap_logits": None},
        "learned_teacher": {
            "source": str(args.learned_table.resolve()),
            "source_sha256": sha256_file(args.learned_table.resolve()),
            "gap_logits": [float(v) for v in init_gap_logits_from_table(learned, native).numpy()],
        },
        "coarse_budgeted": {
            "source": str(args.budgeted_table.resolve()),
            "source_sha256": sha256_file(args.budgeted_table.resolve()),
            "gap_logits": [float(v) for v in init_gap_logits_from_table(budgeted, native).numpy()],
        },
    }

    manifest = {
        "status": STATUS,
        "schema_version": 1,
        "preflight": "ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830",
        "model_weight_updates": 0,
        "source": {
            "r0_collection": str(collection),
            "r0_collection_sha256": r0_sha,
        },
        "native_table_sha256_float32": float32_sha256(native_f32),
        "phase_chord_target": {
            **table_receipt("phase_chord_target", target, native),
            "construction": chord_meta,
        },
        "families": {
            "F1_PC_MORPH": {
                "morph_grid": list(F1_MORPH_GRID),
                "zero_search_reference_t": 0.05,
                "tables": f1_entries,
            },
            "F2_PC_RETENTION_PROJECT": f2_projection,
            "F3_Z5_BEHAVIOUR": {
                "pair_positions": [float(v) for v in knot_positions(native.size).numpy()],
                "optimizer_budget": F3_OPTIMIZER,
                "initialisations": f3_inits,
            },
            "F4_SR_Z5": {
                "support_grid": list(F4_SUPPORT_GRID),
                "inherits": "F3_Z5_BEHAVIOUR",
            },
        },
        "selection_rule": {
            "native_prefix_guard": GUARD_MARGIN,
            "long_dense_guard": GUARD_MARGIN,
            "primary_metric": "lowest far-tail mean NLL among feasible candidates",
            "tiebreak_order": [
                "fewer calibrated degrees of freedom",
                "no long-range outcomes in construction",
                "smaller exact operator-chord displacement",
                "smaller support movement",
            ],
            "tiebreak_dof": TIEBREAK_DOF,
        },
        "endpoint_contract": {
            "native_prefix_targets": "positions 1..L_native-1",
            "position_bin_width": 1024,
            "long_dense_targets": "every valid next-token target in the 4x sequence",
            "far_tail_targets": "final 1024 target tokens",
            "single_4x_forward": True,
        },
        "parity_smoke": {
            "per_token_abs_tol": PARITY_PER_TOKEN_TOL,
            "prefix_mean_abs_tol": PARITY_PREFIX_MEAN_TOL,
            "comparison": "standalone 1x forward vs prefix of the 4x forward, Native table",
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (output / "portfolio_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"froze {len(f1_entries)} F1 tables + F2 projection + F3/F4 construction "
        f"-> {output / 'portfolio_manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
