#!/usr/bin/env python3
"""Build three frozen task-blind safety alternatives to full CA-NCP alignment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.ca_ncp_native_20260917.core import minimal_plane
from experiments.ca_ncp_native_20260917.io_utils import atomic_json, file_sha256

from . import ARMS


METHOD_ID = "CA_NCP_SAFE_FOLLOWUP_V1"


def load_table(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    return np.asarray(payload.get("values_float32", payload), dtype=np.float64)


def fit_lags(asset_manifest: dict, assets: Path) -> np.ndarray:
    values = []
    for row in asset_manifest["documents"]:
        if row["role"] != "fit":
            continue
        with np.load(assets / row["pair_file"], allow_pickle=False) as pairs:
            values.extend((pairs["query_pos"] - pairs["key_pos"]).astype(np.int64).tolist())
    result = np.asarray(values, dtype=np.float64)
    if result.size != 32 * 512 or np.any(result <= 0):
        raise ValueError("fit lag identity differs from the frozen CA-NCP statistic")
    return result


def save_alignment(
    out: Path, *, source: dict[str, np.ndarray], a: np.ndarray, b: np.ndarray,
    v: np.ndarray, u: np.ndarray, identity: np.ndarray, receipt: dict,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "a": a,
        "b": b,
        "v_real": v.real,
        "v_imag": v.imag,
        "u_real": u.real,
        "u_imag": u.imag,
        "identity": identity,
        "active_indices": source["active_indices"],
        "carrier_local": source["carrier_local"],
        "method_receipt_sha256": source["method_receipt_sha256"],
        "statistics_receipt_sha256": source["statistics_receipt_sha256"],
    }
    temporary = out / "alignment.incomplete.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(out / "alignment.npz")
    receipt = {
        **receipt,
        "status": "CA_NCP_SAFE_ALIGNMENT_COMPLETE_V1",
        "method_id": METHOD_ID,
        "alignment_file": "alignment.npz",
        "alignment_sha256": file_sha256(out / "alignment.npz"),
        "source_method_receipt_sha256": str(source["method_receipt_sha256"]),
        "source_statistics_receipt_sha256": str(source["statistics_receipt_sha256"]),
        "selection_uses_task_outputs": False,
        "claim_boundary": "Post-failure development diagnostic on the same Full-13x10 panel; requires independent confirmation before a paper claim.",
    }
    atomic_json(out / "ALIGNMENT_RECEIPT.json", receipt)


def angle_capped(source: dict[str, np.ndarray], angle_cap: float) -> tuple[dict, dict]:
    old_a, old_b = source["a"], source["b"]
    old_v = source["v_real"] + 1j * source["v_imag"]
    carrier = int(source["carrier_local"])
    fractions = np.empty_like(old_a, dtype=np.float64)
    a = np.empty_like(old_a, dtype=np.float64)
    b = np.empty_like(old_b, dtype=np.float64)
    v = old_v.copy()
    u = np.empty_like(v)
    identity = np.empty_like(old_a, dtype=np.bool_)
    e = np.eye(v.shape[-1], dtype=np.complex128)[:, carrier]
    for layer in range(old_a.shape[0]):
        for group in range(old_a.shape[1]):
            full_angle = math.atan2(float(old_b[layer, group]), float(old_a[layer, group]))
            angle = min(full_angle, angle_cap)
            fraction = angle / full_angle if full_angle > 0 else 0.0
            fractions[layer, group] = fraction
            a[layer, group], b[layer, group] = math.cos(angle), math.sin(angle)
            u[layer, group] = a[layer, group] * e + b[layer, group] * v[layer, group]
            identity[layer, group] = b[layer, group] <= 1e-13
    summary = {
        "angle_cap_radians": angle_cap,
        "fraction_quantiles": np.quantile(fractions, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "angle_radians_quantiles": np.quantile(
            fractions * np.arctan2(old_b, old_a), [0, 0.1, 0.5, 0.9, 1]
        ).tolist(),
        "full_strength_planes": int(np.sum(fractions == 1.0)),
        "nonidentity_planes": int(np.sum(~identity)),
    }
    return {"a": a, "b": b, "v": v, "u": u, "identity": identity}, summary


def source_moments(statistics_receipt: dict, statistics: Path, source_name: str) -> np.ndarray:
    values = []
    for row in statistics_receipt["documents"]:
        if row["role"] != "fit" or row["source"] != source_name:
            continue
        with np.load(statistics / row["moment_file"], allow_pickle=False) as payload:
            values.append(payload["moment_real"] + 1j * payload["moment_imag"])
    if len(values) != 16:
        raise ValueError(f"expected sixteen fit moments for {source_name}")
    return np.mean(values, axis=0)


def axis_consensus(source: dict[str, np.ndarray], statistics_receipt: dict, statistics: Path) -> tuple[dict, dict]:
    pg19 = source_moments(statistics_receipt, statistics, "pg19")
    proofpile = source_moments(statistics_receipt, statistics, "proofpile")
    pg_axis = np.argmax(np.real(np.diagonal(pg19, axis1=-2, axis2=-1)), axis=-1)
    pp_axis = np.argmax(np.real(np.diagonal(proofpile, axis1=-2, axis2=-1)), axis=-1)
    carrier = int(source["carrier_local"])
    shape = pg_axis.shape
    active = source["active_indices"]
    size = active.size
    a = np.empty(shape, dtype=np.float64)
    b = np.empty(shape, dtype=np.float64)
    v = np.empty(shape + (size,), dtype=np.complex128)
    u = np.empty_like(v)
    identity = np.empty(shape, dtype=np.bool_)
    chosen = np.full(shape, carrier, dtype=np.int64)
    for layer in range(shape[0]):
        for group in range(shape[1]):
            local = int(pg_axis[layer, group]) if pg_axis[layer, group] == pp_axis[layer, group] else carrier
            direction = np.eye(size, dtype=np.complex128)[:, local]
            plane = minimal_plane(direction, carrier)
            a[layer, group], b[layer, group] = plane.a, plane.b
            v[layer, group], u[layer, group] = plane.v, plane.u
            identity[layer, group] = plane.identity
            chosen[layer, group] = local
    summary = {
        "source_consensus_planes": int(np.sum(pg_axis == pp_axis)),
        "source_disagreement_identity_planes": int(np.sum(pg_axis != pp_axis)),
        "carrier_identity_planes": int(np.sum(chosen == carrier)),
        "nonidentity_planes": int(np.sum(~identity)),
        "selected_global_coordinate_counts": {
            str(int(index)): int(np.sum(active[chosen] == index)) for index in np.unique(active[chosen])
        },
        "rule": "per layer/KV group, rotate only when PG19-fit and ProofPile-fit diagonal argmax agree; otherwise identity",
    }
    return {"a": a, "b": b, "v": v, "u": u, "identity": identity}, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    construction = source_root / "construction"
    statistics = source_root / "statistics"
    assets = source_root / "assets/statistics"
    method = json.loads((construction / "METHOD_RECEIPT.json").read_text())
    stats_receipt = json.loads((statistics / "STATISTICS_RECEIPT.json").read_text())
    asset_manifest = json.loads((assets / "manifest.json").read_text())
    with np.load(source_root / "alignment/alignment.npz", allow_pickle=False) as payload:
        source = {name: payload[name] for name in payload.files}
    lags = fit_lags(asset_manifest, assets)
    active = source["active_indices"].astype(np.int64)
    native = load_table(construction / "native.json")
    ncp = load_table(construction / "ncp.json")
    carrier = load_table(construction / "carrier_ncp.json")
    distance = np.arange(1, int(method["construction"]["native_length"]), dtype=np.float64)
    carrier_local = int(source["carrier_local"])
    carrier_global = int(active[carrier_local])
    carrier_budget = float(np.max(np.abs(
        np.exp(-1j * distance * carrier[carrier_global])
        - np.exp(-1j * distance * ncp[carrier_global])
    )))
    active_spread = float(np.max(np.abs(
        np.exp(-1j * distance[:, None] * carrier[active][None, :])
        - np.exp(-1j * distance[:, None] * carrier[carrier_global])
    )))
    angle_cap = 2.0 * math.asin(min(1.0, carrier_budget / (4.0 * active_spread)))
    receipts = {}
    arrays, summary = angle_capped(source, angle_cap)
    receipt = {
        "arm": "operator_cap_shared",
        "construction": "shared maximal safe geodesic cap from identity toward the frozen full CA-NCP plane",
        "carrier_budget_max_native_window": carrier_budget,
        "carrier_active_spread_max_native_window": active_spread,
        "bound": "phi=2*asin(min(1,B/(4H))); guarantees alignment kernel perturbation no greater than the carrier snap budget",
        **summary,
    }
    save_alignment(args.out / "alignments/operator_cap", source=source, receipt=receipt, **arrays)
    for arm in ARMS[:2]:
        receipts[arm] = {**receipt, "arm": arm}
    arrays, summary = axis_consensus(source, stats_receipt, statistics)
    receipt = {
        "arm": "P_axis_consensus",
        "construction": "source-consensus sparse two-coordinate carrier assignment",
        **summary,
    }
    save_alignment(args.out / "alignments/P_axis_consensus", source=source, receipt=receipt, **arrays)
    receipts["P_axis_consensus"] = receipt
    root_receipt = {
        "status": "CA_NCP_SAFE_FOLLOWUP_CPU_READY_V1",
        "method_id": METHOD_ID,
        "source_root": str(source_root),
        "source_report_sha256": file_sha256(source_root / "reports/paired_report.json"),
        "source_alignment_sha256": file_sha256(source_root / "alignment/alignment.npz"),
        "panel_sha256": json.loads((source_root / "assets/pilot/manifest.json").read_text())["panel"]["inputs_sha256"],
        "fit_lags": int(lags.size),
        "arms": receipts,
        "selection_uses_task_outputs": False,
        "evaluation_role": "post-failure development diagnostic on the same fixed panel",
    }
    atomic_json(args.out / "METHOD_RECEIPT.json", root_receipt)
    print(json.dumps({"status": root_receipt["status"], "arms": list(receipts)}, sort_keys=True))


if __name__ == "__main__":
    main()
