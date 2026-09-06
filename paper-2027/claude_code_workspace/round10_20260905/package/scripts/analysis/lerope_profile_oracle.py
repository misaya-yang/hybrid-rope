#!/usr/bin/env python3
"""CPU-only pilot: does rho(phi) propto w(phi)^(1/3) predict LeRoPE?

The primary weight is the per-band diagonal of

    J_logomega^T (diag(p) - p p^T) J_logomega,

averaged over a frozen trained checkpoint.  Multiplying by log(base)^2 would
convert the derivative from log(omega) to phi; that constant cancels after
normalizing rho.  The high-rate density is discretized by inverse-CDF midpoint
quantiles.  No training or parameter update is performed.

LeRoPE's 32 frequencies are transcribed from the printed headers of Figure 6
in arXiv:2607.10134v1.  They are rounded display values, not checkpoint data.

Fast path (the canonical probe already exists):

  conda run --no-capture-output -n aidemo python \
    scripts/analysis/lerope_profile_oracle.py \
    --probe-json /tmp/attention_fisher_50m_probe_20260819.json

Portable path (recomputes one CPU-only self-consistent cell):

  conda run --no-capture-output -n aidemo python \
    scripts/analysis/lerope_profile_oracle.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "paper-2027/figs/fig_lerope_profile_oracle.pdf"
CANONICAL_PROBE_SHA256 = (
    "32770ef134a457de8f4878f6d669f499defedafd079f96da625503b2660bc2c9"
)
PROBE_ARM = "geometric_tau0_seed42__runtime_geometric_tau0"

# Figure 6 header transcription, L=2048.  Suffixes m and micro in the paper
# mean 1e-3 and 1e-6.  The half-width is one half of the final displayed digit;
# it is a display-rounding envelope, not statistical uncertainty.
LEROPE_THETA = np.array([
    1.9, 1.1, 0.67, 0.45, 0.31, 0.24, 0.16, 0.11,
    0.084, 0.059, 0.044, 0.031, 0.023, 0.016, 0.0094, 0.0042,
    0.0015, 0.0011, 0.000100, 0.000049, 0.000036, 0.000024,
    0.000020, 0.000020, 0.000012, 0.000018, 0.000010, 0.000015,
    0.000012, 0.0000099, 0.0000094, 0.000015,
], dtype=np.float64)
LEROPE_THETA_HALF_ROUNDING = np.array([
    0.05, 0.05, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
    0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.00005, 0.00005,
    0.00005, 0.00005, 0.0000005, 0.0000005, 0.0000005, 0.0000005,
    0.0000005, 0.0000005, 0.0000005, 0.0000005, 0.0000005, 0.0000005,
    0.0000005, 0.00000005, 0.00000005, 0.0000005,
], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_canonical_probe(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    actual_hash = sha256(path)
    if actual_hash != CANONICAL_PROBE_SHA256:
        raise ValueError(
            f"probe JSON hash mismatch: {actual_hash}; expected {CANONICAL_PROBE_SHA256}"
        )
    payload = json.loads(path.read_text())
    arm = payload["arms"][PROBE_ARM]
    w = np.asarray(
        arm["fisher"]["frequency_softmax"]["pair_diagonal"], dtype=np.float64
    )
    inv_freq = np.asarray(arm["identity"]["runtime_inv_freq"], dtype=np.float64)
    phi = -np.log(inv_freq) / math.log(500_000.0)
    identity = {
        "source": str(path.resolve()),
        "source_sha256": actual_hash,
        "checkpoint_sha256": arm["identity"]["checkpoint_sha256"],
        "validation_sha256": payload["protocol"]["validation_sha256"],
        "length": payload["protocol"]["length"],
        "windows": payload["protocol"]["validation_windows"],
        "query_positions": payload["protocol"]["query_positions"],
        "head_query_observations": arm["runtime"]["head_query_observations"],
        "manual_vs_sdpa_max_abs": arm["runtime"]["manual_vs_sdpa_max_abs"],
        "log_frequency_jacobian_finite_difference_max_abs": arm["runtime"][
            "log_frequency_jacobian_finite_difference_max_abs"
        ],
    }
    return phi, w, identity


def measure_probe(windows: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Recompute only the Geo-weights/Geo-table CPU cell."""
    import torch

    from scripts.analysis.attention_fisher_50m_probe import ARMS, RUN_ROOT, _probe_arm

    length = 512
    validation_path = RUN_ROOT / "val_tinystories_5000000.pt"
    validation = torch.load(validation_path, map_location="cpu", weights_only=True)
    starts = np.linspace(0, len(validation) - length - 1, windows, dtype=np.int64)
    query_positions = (63, 127, 255, 383, 511)
    inv_path = ARMS["geometric_tau0_seed42"] / "inv_freq.npy"
    inv_freq = torch.from_numpy(np.load(inv_path)).float()
    public, _ = _probe_arm(
        ARMS["geometric_tau0_seed42"],
        inv_freq,
        "geometric_tau0",
        validation,
        starts,
        query_positions,
        length,
    )
    w = np.asarray(
        public["fisher"]["frequency_softmax"]["pair_diagonal"], dtype=np.float64
    )
    inv = np.asarray(public["identity"]["runtime_inv_freq"], dtype=np.float64)
    phi = -np.log(inv) / math.log(500_000.0)
    identity = {
        "source": "CPU recomputation",
        "checkpoint_sha256": public["identity"]["checkpoint_sha256"],
        "validation_sha256": sha256(validation_path),
        "length": length,
        "windows": windows,
        "query_positions": list(query_positions),
        "head_query_observations": public["runtime"]["head_query_observations"],
        "manual_vs_sdpa_max_abs": public["runtime"]["manual_vs_sdpa_max_abs"],
        "log_frequency_jacobian_finite_difference_max_abs": public["runtime"][
            "log_frequency_jacobian_finite_difference_max_abs"
        ],
    }
    return phi, w, identity


def high_rate_profile(
    phi_nodes: np.ndarray, w_nodes: np.ndarray, bands: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Piecewise-log-linear w, rho proportional to cube root, midpoint I-CDF."""
    if np.any(w_nodes <= 0) or np.any(np.diff(phi_nodes) <= 0):
        raise ValueError("phi must increase and w must be strictly positive")
    grid = np.linspace(0.0, 1.0, 20_001)
    log_w = np.interp(grid, phi_nodes, np.log(w_nodes))
    rho = np.exp(log_w / 3.0)
    increments = 0.5 * (rho[:-1] + rho[1:]) * np.diff(grid)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    cdf /= cdf[-1]
    quantiles = np.interp((np.arange(bands) + 0.5) / bands, cdf, grid)
    return grid, rho / np.trapezoid(rho, grid), quantiles


def evq_phi(bands: int, tau: float) -> np.ndarray:
    u = (np.arange(bands, dtype=np.float64) + 0.5) / bands
    if abs(tau) < 1e-12:
        return u
    return 1.0 - np.arcsinh((1.0 - u) * np.sinh(tau)) / tau


def endpoint_normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        raise ValueError("profile support collapsed")
    return (values - lo) / (hi - lo)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def rounding_rmse_envelope(
    predicted_log_wavelength: np.ndarray,
) -> tuple[float, float]:
    theta_lo = LEROPE_THETA - LEROPE_THETA_HALF_ROUNDING
    theta_hi = LEROPE_THETA + LEROPE_THETA_HALF_ROUNDING
    if np.any(theta_lo <= 0):
        raise ValueError("invalid LeRoPE rounding interval")
    observed_lo = math.log(2.0 * math.pi) - np.log(theta_hi)
    observed_hi = math.log(2.0 * math.pi) - np.log(theta_lo)
    below = np.maximum(observed_lo - predicted_log_wavelength, 0.0)
    above = np.maximum(predicted_log_wavelength - observed_hi, 0.0)
    nearest = below + above
    farthest = np.maximum(
        np.abs(predicted_log_wavelength - observed_lo),
        np.abs(predicted_log_wavelength - observed_hi),
    )
    return rmse(nearest, np.zeros_like(nearest)), rmse(
        farthest, np.zeros_like(farthest)
    )


def compute_metrics(
    oracle_phi: np.ndarray, evq_public_phi: np.ndarray
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    bands = len(oracle_phi)
    base = 10_000.0
    geo_phi = np.arange(bands, dtype=np.float64) / bands
    profiles_phi = {
        "Geo": geo_phi,
        "EVQ-Cosh": evq_public_phi,
        "rho*": oracle_phi,
    }
    log_wavelength = {
        name: math.log(2.0 * math.pi) + math.log(base) * phi
        for name, phi in profiles_phi.items()
    }
    log_wavelength["LeRoPE"] = math.log(2.0 * math.pi) - np.log(LEROPE_THETA)
    shape = {name: endpoint_normalize(value) for name, value in log_wavelength.items()}

    support_to_lerope = {
        name: rmse(value, log_wavelength["LeRoPE"])
        for name, value in log_wavelength.items()
        if name != "LeRoPE"
    }
    shape_to_lerope = {
        name: rmse(value, shape["LeRoPE"])
        for name, value in shape.items()
        if name != "LeRoPE"
    }
    shape_oracle_to_evq = rmse(shape["rho*"], shape["EVQ-Cosh"])
    shape_oracle_to_lerope = shape_to_lerope["rho*"]
    direction = shape["LeRoPE"] - shape["EVQ-Cosh"]
    alpha = float(
        np.dot(shape["rho*"] - shape["EVQ-Cosh"], direction)
        / np.dot(direction, direction)
    )
    segment_residual = rmse(
        shape["rho*"], shape["EVQ-Cosh"] + alpha * direction
    )
    if alpha < 0.0:
        classification = "beyond_EVQ_away_from_LeRoPE_along_projection"
    elif alpha <= 1.0:
        classification = "projects_between_EVQ_and_LeRoPE"
    else:
        classification = "beyond_LeRoPE_along_projection"

    metrics = {
        "support_aware_log_wavelength_rmse_nats_to_LeRoPE": support_to_lerope,
        "support_normalized_shape_rmse_to_LeRoPE": shape_to_lerope,
        "support_normalized_rho_star_rmse_to_EVQ": shape_oracle_to_evq,
        "EVQ_to_LeRoPE_segment_projection_alpha": alpha,
        "segment_projection_residual_rmse": segment_residual,
        "classification": classification,
        "display_rounding_rmse_envelope_nats": {
            name: list(rounding_rmse_envelope(value))
            for name, value in log_wavelength.items()
            if name != "LeRoPE"
        },
    }
    return metrics, {"log_wavelength": log_wavelength, "shape": shape}


def plot(
    output: Path,
    phi_nodes: np.ndarray,
    w_nodes: np.ndarray,
    density_grid: np.ndarray,
    density: np.ndarray,
    curves: dict[str, np.ndarray],
    metrics: dict[str, object],
) -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "font.family": "DejaVu Sans",
    })
    colors = {
        "Geo": "#7A7F87",
        "EVQ-Cosh": "#2356A8",
        "rho*": "#D06B25",
        "LeRoPE": "#20242A",
    }
    styles = {"Geo": "--", "EVQ-Cosh": "-", "rho*": "-.", "LeRoPE": ":"}
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))

    ax = axes[0]
    ax.semilogy(phi_nodes, w_nodes / np.max(w_nodes), "o", ms=3.1,
                color=colors["rho*"], label=r"measured $w/\max w$")
    ax.semilogy(density_grid, density / np.max(density), color=colors["EVQ-Cosh"],
                lw=1.5, label=r"$\rho^*\!\propto w^{1/3}$")
    ax.set_title("(a) Checkpoint-local utility")
    ax.set_xlabel(r"normalized log-frequency $\phi$")
    ax.set_ylabel("relative magnitude (log)")
    ax.grid(True, which="major", color="#D9DDE3", lw=0.5)
    ax.legend(frameon=False, loc="lower left")

    bands = np.arange(len(LEROPE_THETA))
    ax = axes[1]
    for name in ("Geo", "EVQ-Cosh", "rho*", "LeRoPE"):
        ax.plot(
            bands,
            np.exp(curves["log_wavelength"][name]),
            linestyle=styles[name],
            color=colors[name],
            lw=1.45,
            marker="o" if name in {"rho*", "LeRoPE"} else None,
            ms=2.2,
            label=name,
        )
    ax.set_yscale("log")
    ax.set_title("(b) Published-profile coordinates")
    ax.set_xlabel("frequency-band index")
    ax.set_ylabel("wavelength (tokens)")
    ax.grid(True, which="major", color="#D9DDE3", lw=0.5)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    ax = axes[2]
    for name in ("Geo", "EVQ-Cosh", "rho*", "LeRoPE"):
        ax.plot(
            bands,
            curves["shape"][name],
            linestyle=styles[name],
            color=colors[name],
            lw=1.45,
            marker="o" if name in {"rho*", "LeRoPE"} else None,
            ms=2.2,
            label=name,
        )
    ax.set_title("(c) Shape after support removal")
    ax.set_xlabel("frequency-band index")
    ax.set_ylabel("normalized log-wavelength")
    ax.grid(True, which="major", color="#D9DDE3", lw=0.5)
    d_evq = metrics["support_normalized_rho_star_rmse_to_EVQ"]
    d_lerope = metrics["support_normalized_shape_rmse_to_LeRoPE"]["rho*"]
    ax.text(
        0.03,
        0.96,
        f"RMSE(rho*, EVQ)={d_evq:.3f}\nRMSE(rho*, LeRoPE)={d_lerope:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox={"facecolor": "white", "edgecolor": "#C8CDD5", "pad": 2.0},
    )

    fig.text(
        0.5,
        0.005,
        "LeRoPE: arXiv:2607.10134v1 Fig. 6 header values (rounded); "
        "rho*: frozen 50M Geo checkpoint, L=512 structural softmax curvature.",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color="#4A4F57",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-json", type=Path)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.windows < 2:
        raise ValueError("at least two windows are required")

    if args.probe_json:
        phi_nodes, w_nodes, identity = load_canonical_probe(args.probe_json)
    else:
        phi_nodes, w_nodes, identity = measure_probe(args.windows)
    if len(phi_nodes) != len(LEROPE_THETA):
        raise ValueError("the oracle and LeRoPE comparison both require K=32")

    grid, density, oracle_phi = high_rate_profile(phi_nodes, w_nodes, len(phi_nodes))
    public_tau = 64.0 / math.sqrt(2048.0)
    evq_public_phi = evq_phi(len(phi_nodes), public_tau)
    metrics, curves = compute_metrics(oracle_phi, evq_public_phi)
    plot(args.output, phi_nodes, w_nodes, grid, density, curves, metrics)

    result = {
        "status": "CPU_ONLY_COMPLETE",
        "measurement": identity,
        "definition": {
            "w": "mean diag(J_logomega^T (diag(p)-pp^T) J_logomega)",
            "density": "rho(phi) proportional to w(phi)^(1/3)",
            "interpolation": "piecewise linear in log w on phi in [0,1]",
            "quantization": "32 inverse-CDF midpoint quantiles",
            "comparison_EVQ_tau": public_tau,
            "comparison_support": "base=10000, head_dim=64, L_train=2048",
        },
        "oracle_phi_midpoint_quantiles": oracle_phi.tolist(),
        "metrics": metrics,
        "figure": str(args.output.resolve()),
        "figure_sha256": sha256(args.output),
        "training_or_parameter_updates": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
