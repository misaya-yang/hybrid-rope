#!/usr/bin/env python3
"""Freeze low-dimensional G_4(x) candidates and evaluate a Qwen geometry holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np


CANDIDATES = ("C0_exact_step", "C1_movable_step", "C2_clipped_affine", "C3_logistic")
MODELS = ("olmo", "qwen")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = average_ranks(a), average_ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def transition_summary(x: np.ndarray, movement: np.ndarray) -> dict:
    centered = movement - 0.5
    crossings = [
        [i, i + 1]
        for i in range(len(movement) - 1)
        if centered[i] == 0 or centered[i] * centered[i + 1] < 0
    ]
    nearest = int(np.argmin(np.abs(centered)))
    soft = np.flatnonzero((movement > 0.01) & (movement < 0.99)).tolist()
    return {
        "half_threshold_crossing_brackets": crossings,
        "nearest_half_slot": nearest,
        "nearest_half_x": float(x[nearest]),
        "nearest_half_m": float(movement[nearest]),
        "soft_transition_slots_0p01_to_0p99": soft,
    }


def comparison(predicted: np.ndarray, target: np.ndarray, x: np.ndarray) -> dict:
    residual = predicted - target
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs_difference": float(np.max(np.abs(residual))),
        "max_abs_difference_slot": int(np.argmax(np.abs(residual))),
        "spearman_rank_agreement": spearman(predicted, target),
        "endpoint_difference": {
            "fast_slot_0": float(residual[0]),
            "slow_slot_last": float(residual[-1]),
        },
        "predicted_transition": transition_summary(x, predicted),
        "target_transition": transition_summary(x, target),
    }


def predict(x: np.ndarray, params: dict) -> dict[str, np.ndarray]:
    mu = params["mu"]
    x_h = params["x_H"]
    x_l = params["x_L"]
    logistic_mu = params["logistic_mu"]
    tau = params["tau"]
    return {
        "C0_exact_step": (x <= 0).astype(np.float64),
        "C1_movable_step": (x <= mu).astype(np.float64),
        "C2_clipped_affine": np.clip((x_h - x) / (x_h - x_l), 0.0, 1.0),
        "C3_logistic": 1.0 / (1.0 + np.exp(np.clip((x - logistic_mu) / tau, -700, 700))),
    }


def write_plots(output: Path, data: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    for model in MODELS:
        block = data[model]
        order = np.argsort(block["x"])
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(block["x"][order], block["empirical"][order], "ko-", ms=3, lw=1.5, label="empirical")
        if model == "qwen":
            ax.plot(block["x"][order], block["transport"][order], color="0.45", ls="--", lw=1.5, label="whole-profile transport")
        for name in CANDIDATES:
            ax.plot(block["x"][order], block["predictions"][name][order], lw=1.2, label=name)
        ax.set(xlabel=r"$x=\ln(c/c_{orth})$", ylabel="movement m", title=f"{model.upper()} low-dimensional coupling")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(plots / f"{model}_profiles.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        block = data[model]
        for name in CANDIDATES:
            ax.plot(np.arange(len(block["x"])), block["predictions"][name] - block["empirical"], label=name)
        ax.axhline(0, color="black", lw=0.7)
        ax.set(title=model.upper(), xlabel="rotary slot", ylabel="predicted - empirical m")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "slotwise_residuals.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "coordinate_audit": args.agent_root / "agent_a" / "coordinate_audit.json",
        "fit_0d": args.agent_root / "agent_b" / "fit_0d.json",
        "fit_1d": args.agent_root / "agent_b" / "fit_1d.json",
        "fit_clipped_affine": args.agent_root / "agent_c" / "fit_clipped_affine.json",
        "fit_logistic": args.agent_root / "agent_d" / "fit_logistic.json",
        "adversarial": args.agent_root / "agent_e" / "adversarial.json",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    audit = load_json(sources["coordinate_audit"])
    fit_1d = load_json(sources["fit_1d"])
    fit_affine = load_json(sources["fit_clipped_affine"])
    fit_logistic = load_json(sources["fit_logistic"])
    params = {
        "mu": float(fit_1d["parameters"]["mu_representative_midpoint"]),
        "x_H": float(fit_affine["parameters"]["x_H"]),
        "x_L": float(fit_affine["parameters"]["x_L"]),
        "logistic_mu": float(fit_logistic["model"]["mu"]),
        "tau": float(fit_logistic["model"]["tau"]),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    data = {}
    table_tensor = np.empty((len(MODELS), len(CANDIDATES), 64), dtype="<f4")
    table_manifest = {
        "status": "FROZEN_BEFORE_QWEN_HOLDOUT_REPORTED",
        "s": 4,
        "model_order": list(MODELS),
        "candidate_order": list(CANDIDATES),
        "parameters_fitted_on": "OLMo frozen movement only",
        "parameters": params,
        "selection": {
            "primary": "C2_clipped_affine",
            "challenger": None,
            "controls": ["C0_exact_step", "C1_movable_step", "C3_logistic"],
            "reason": "1D leaves transition residual; clipped-affine is sufficient; logistic has no reconstruction or complexity advantage; 3D not authorized.",
        },
        "tables": {},
    }

    qwen_transport = np.asarray(
        audit["current_coordinate_transport_audit"]["saved_transport_m_i_recovered_from_float32_table"],
        dtype=np.float64,
    )
    for model_index, model in enumerate(MODELS):
        arrays = audit["models"][model]["arrays"]
        x = np.asarray(arrays["x_i"], dtype=np.float64)
        empirical = np.asarray(arrays["frozen_m_i"], dtype=np.float64)
        omega = np.asarray(arrays["native_omega_i"], dtype=np.float64)
        predictions = predict(x, params)
        transport = empirical if model == "olmo" else qwen_transport
        data[model] = {"x": x, "empirical": empirical, "omega": omega, "predictions": predictions, "transport": transport}
        table_manifest["tables"][model] = {}
        table_dir = args.output / "tables" / model
        table_dir.mkdir(parents=True, exist_ok=True)
        for candidate_index, name in enumerate(CANDIDATES):
            table = np.asarray(omega * np.power(4.0, -predictions[name]), dtype="<f4")
            table_tensor[model_index, candidate_index] = table
            table_path = table_dir / f"{name}.npy"
            np.save(table_path, table, allow_pickle=False)
            table_manifest["tables"][model][name] = {
                "sha256_float32_tensor": tensor_hash(table),
                "npy_sha256_file": sha256_file(table_path),
                "path": str(table_path.relative_to(args.output)),
                "fast_ratio": float(table[0] / np.float32(omega[0])),
                "slow_ratio": float(table[-1] / np.float32(omega[-1])),
                "crossing_indices": np.flatnonzero(np.diff(table.astype(np.float64)) >= 0).astype(int).tolist(),
            }

    for name in ("coordinate_audit", "fit_0d", "fit_1d", "fit_clipped_affine", "fit_logistic"):
        shutil.copyfile(sources[name], args.output / f"{name}.json")
    np.save(args.output / "candidate_tables.npy", table_tensor, allow_pickle=False)

    qwen_holdout = {
        "status": "ZERO_REFIT_QWEN_GEOMETRY_HOLDOUT",
        "fit_data": "OLMo frozen movement only",
        "qwen_profile_use": "opened only after all candidate parameters and primary selection were frozen",
        "qwen_identity": audit["models"]["qwen"]["identity"],
        "whole_profile_transport_definition": audit["current_coordinate_transport_audit"]["registered_definition"],
        "whole_profile_transport_vs_qwen_self": comparison(qwen_transport, data["qwen"]["empirical"], data["qwen"]["x"]),
        "candidates": {},
    }
    for name in CANDIDATES:
        prediction = data["qwen"]["predictions"][name]
        qwen_holdout["candidates"][name] = {
            "vs_qwen_self_profile": comparison(prediction, data["qwen"]["empirical"], data["qwen"]["x"]),
            "vs_olmo_whole_profile_transport": comparison(prediction, qwen_transport, data["qwen"]["x"]),
        }
    (args.output / "qwen_holdout.json").write_text(json.dumps(qwen_holdout, indent=2, sort_keys=True) + "\n")

    with (args.output / "residuals.csv").open("w", newline="") as handle:
        fields = ["model", "slot", "x", "empirical_m", "whole_profile_transport_m"]
        for name in CANDIDATES:
            fields.extend((f"{name}_m", f"{name}_residual"))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            block = data[model]
            for slot in range(len(block["x"])):
                row = {
                    "model": model,
                    "slot": slot,
                    "x": float(block["x"][slot]),
                    "empirical_m": float(block["empirical"][slot]),
                    "whole_profile_transport_m": float(block["transport"][slot]),
                }
                for name in CANDIDATES:
                    value = float(block["predictions"][name][slot])
                    row[f"{name}_m"] = value
                    row[f"{name}_residual"] = value - float(block["empirical"][slot])
                writer.writerow(row)

    write_plots(args.output, data)
    table_manifest["candidate_tables_npy_sha256"] = sha256_file(args.output / "candidate_tables.npy")
    table_manifest["sources"] = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in sources.items()}
    (args.output / "candidate_tables.json").write_text(json.dumps(table_manifest, indent=2, sort_keys=True) + "\n")

    manifest = {
        "status": "CPU_LOW_DIM_COUPLING_LAW_COMPLETE",
        "cpu_only": True,
        "cuda_visible_devices": "",
        "files": {},
    }
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][str(path.relative_to(args.output))] = sha256_file(path)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
