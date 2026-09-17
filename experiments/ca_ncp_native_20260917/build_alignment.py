#!/usr/bin/env python3
"""Build the carrier table or solve frozen per-layer/per-group alignments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import METHOD_ID, NATIVE_LENGTH, ROPE_BASE, ROTARY_PAIRS
from .core import (
    build_carrier_table,
    choose_direction,
    minimal_plane,
    tensor_sha256,
)
from .io_utils import atomic_json, file_sha256, model_identity
from scripts.experiments.cross_audit.tables import native_table


EXPECTED_NCP_SHA256 = "54b9dd1f73aafc69f7bb5ed1b7b49d49128002371cb378d03ca1fd1d108e0cb7"
EXPECTED_CARRIER_SHA256 = "10547fcaf8d8d6f0bc3935a839f0bd81734e0fa18855603cda3686c351aa06c7"


def _table(path: Path) -> tuple[np.ndarray, float, dict]:
    payload = json.loads(path.read_text())
    table = payload.get("table", payload)
    values = np.asarray(table.get("values_float32"), dtype=np.float32)
    gain = float(table.get("gain"))
    if values.shape != (ROTARY_PAIRS,) or not np.isfinite(values).all() or not np.all(values[:-1] > values[1:]):
        raise ValueError("invalid NCP table")
    if gain != 1.0:
        raise ValueError("CA-NCP requires gain=1")
    return values, gain, table.get("construction", {})


def _write_table(path: Path, values: np.ndarray, *, label: str, construction: dict) -> None:
    atomic_json(path, {
        "status": "FROZEN_CA_NCP_TABLE_V1_1",
        "candidate_id": label,
        "values_float32": np.asarray(values, dtype=np.float32).tolist(),
        "gain": 1.0,
        "table_sha256_float32": tensor_sha256(values),
        "construction": construction,
    })


def tables_only(model: Path, ncp_path: Path, out: Path) -> dict:
    existing = out / "METHOD_RECEIPT.json"
    if existing.is_file():
        receipt = json.loads(existing.read_text())
        table_paths = [out / name for name in ("native.json", "ncp.json", "carrier_ncp.json")]
        if (
            receipt.get("status") == "METHOD_CONSTRUCTION_COMPLETE"
            and Path(receipt["model_identity"]["model_path"]).resolve() == model.resolve()
            and receipt["model_identity"]["config_sha256"] == file_sha256(model / "config.json")
            and receipt.get("ncp_source_file_sha256") == file_sha256(ncp_path)
            and all(path.is_file() for path in table_paths)
        ):
            expected = [
                receipt["native_table_sha256_float32"],
                receipt["ncp_table_sha256_float32"],
                receipt["carrier_table_sha256_float32"],
            ]
            observed = [
                tensor_sha256(np.asarray(json.loads(path.read_text())["values_float32"], dtype=np.float32))
                for path in table_paths
            ]
            if observed == expected:
                return receipt
        raise ValueError("existing construction receipt or tables drifted")
    identity = model_identity(model)
    if (
        identity["model_type"] != "olmo2"
        or identity["native_length"] != NATIVE_LENGTH
        or identity["rotary_pairs"] != ROTARY_PAIRS
        or identity["rope_theta"] != ROPE_BASE
    ):
        raise ValueError("CA-NCP v1.1 geometry differs from the requested checkpoint")
    ncp, gain, prior = _table(ncp_path)
    native = native_table(ROTARY_PAIRS * 2, ROPE_BASE).astype(np.float32)
    geometry = build_carrier_table(native, ncp, NATIVE_LENGTH)
    if geometry["ncp_table_sha256_float32"] != EXPECTED_NCP_SHA256:
        raise ValueError("NCP table differs from the frozen completed-experiment identity")
    if geometry["carrier_table_sha256_float32"] != EXPECTED_CARRIER_SHA256:
        raise AssertionError("carrier table differs from the v1.1 reference hash")
    out.mkdir(parents=True, exist_ok=True)
    construction = {
        "method_id": METHOD_ID,
        "native_length": NATIVE_LENGTH,
        "rotary_pairs": ROTARY_PAIRS,
        "rope_base": ROPE_BASE,
        "gain": gain,
        "carrier_ratio": geometry["carrier_ratio"],
        "carrier_slot_zero_based": geometry["carrier_slot"],
        "carrier_local_zero_based": geometry["carrier_local"],
        "carrier_frequency_float64": geometry["carrier_frequency_float64"],
        "carrier_frequency_float32": geometry["carrier_frequency_float32"],
        "active_indices_zero_based": geometry["active_indices"].tolist(),
        "ncp_changed_indices_zero_based": geometry["changed_indices"].tolist(),
        "activity_rule": "NCP differs from Native, interior slot, and L*omega_native <= 2*pi",
        "ncp_prior_construction": prior,
        "selection_uses_model_outputs": False,
    }
    _write_table(out / "native.json", native, label="N0_native", construction=construction)
    _write_table(out / "ncp.json", ncp, label="C0_ncp", construction=construction)
    _write_table(out / "carrier_ncp.json", geometry["carrier_table"], label="P0_carrier_ncp", construction=construction)
    receipt = {
        "status": "METHOD_CONSTRUCTION_COMPLETE",
        "method_id": METHOD_ID,
        "model_identity": identity,
        "ncp_source_path": str(ncp_path.resolve()),
        "ncp_source_file_sha256": file_sha256(ncp_path),
        "native_table_sha256_float32": geometry["native_table_sha256_float32"],
        "ncp_table_sha256_float32": geometry["ncp_table_sha256_float32"],
        "carrier_table_sha256_float32": geometry["carrier_table_sha256_float32"],
        "construction": construction,
        "public_constants_only_for_tables": True,
        "model_execution": False,
        "task_outputs_read": False,
        "claim_boundary": "Frequency construction only; complete CA-NCP additionally requires real Native Q/K statistics.",
    }
    atomic_json(existing, receipt)
    return receipt


def _load_moment(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _mean_moments(rows: list[dict], root: Path, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["role"] == role]
    if not selected:
        raise ValueError(f"statistics contain no {role} documents")
    moments, bins, counts = [], [], []
    for row in selected:
        path = root / row["moment_file"]
        if file_sha256(path) != row["moment_file_sha256"]:
            raise ValueError(f"moment file drifted: {path}")
        payload = _load_moment(path)
        moments.append(payload["moment_real"] + 1j * payload["moment_imag"])
        bins.append(payload["bin_real"] + 1j * payload["bin_imag"])
        counts.append(payload["bin_counts"])
    return np.mean(moments, axis=0), np.asarray(bins), np.asarray(counts)


def solve(statistics: Path, construction: Path, out: Path) -> dict:
    method = json.loads((construction / "METHOD_RECEIPT.json").read_text())
    stats = json.loads((statistics / "STATISTICS_RECEIPT.json").read_text())
    if method.get("method_id") != METHOD_ID or stats.get("method_id") != METHOD_ID:
        raise ValueError("method/statistics contract mismatch")
    if stats["method_receipt_sha256"] != file_sha256(construction / "METHOD_RECEIPT.json"):
        raise ValueError("statistics were captured for another method receipt")
    fit, fit_bins, fit_counts = _mean_moments(stats["documents"], statistics, "fit")
    report, report_bins, report_counts = _mean_moments(stats["documents"], statistics, "report")
    if fit.shape != report.shape or fit.ndim != 4 or fit.shape[-1] != fit.shape[-2]:
        raise ValueError("statistics moment dimensions differ")
    layers, groups, active_size, _ = fit.shape
    carrier_local = int(method["construction"]["carrier_local_zero_based"])
    a = np.empty((layers, groups), dtype=np.float64)
    b = np.empty_like(a)
    vr = np.empty((layers, groups, active_size), dtype=np.float64)
    vi = np.empty_like(vr)
    ur = np.empty_like(vr)
    ui = np.empty_like(vr)
    identity = np.empty((layers, groups), dtype=np.bool_)
    records = []
    bin_labels = list(stats["distance_bins"])
    for layer in range(layers):
        for group in range(groups):
            direction, eig = choose_direction(fit[layer, group], carrier_local)
            plane = minimal_plane(direction, carrier_local)
            a[layer, group], b[layer, group] = plane.a, plane.b
            vr[layer, group], vi[layer, group] = plane.v.real, plane.v.imag
            ur[layer, group], ui[layer, group] = plane.u.real, plane.u.imag
            identity[layer, group] = plane.identity
            report_objective = float(np.vdot(direction, report[layer, group] @ direction).real)
            bin_receipts = {}
            for role, matrices, counts in (("fit", fit_bins, fit_counts), ("report", report_bins, report_counts)):
                values = []
                for bin_index, label in enumerate(bin_labels):
                    valid = counts[:, bin_index] > 0
                    if not np.any(valid):
                        values.append({"label": label, "documents": 0, "objective": None})
                        continue
                    matrix = matrices[valid, layer, group, bin_index].mean(axis=0)
                    values.append({
                        "label": label,
                        "documents": int(valid.sum()),
                        "objective": float(np.vdot(direction, matrix @ direction).real),
                    })
                bin_receipts[role] = values
            records.append({
                "layer": layer,
                "kv_group": group,
                **eig,
                "report_objective": report_objective,
                "a": plane.a,
                "b": plane.b,
                "frobenius_distance_squared": 4.0 * (1.0 - plane.a),
                "spectral_distance": float(np.sqrt(2.0 * (1.0 - plane.a))),
                "distance_bins": bin_receipts,
            })
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "a": a,
        "b": b,
        "v_real": vr,
        "v_imag": vi,
        "u_real": ur,
        "u_imag": ui,
        "identity": identity,
        "active_indices": np.asarray(method["construction"]["active_indices_zero_based"], dtype=np.int64),
        "carrier_local": np.asarray(carrier_local, dtype=np.int64),
        "method_receipt_sha256": np.asarray(file_sha256(construction / "METHOD_RECEIPT.json")),
        "statistics_receipt_sha256": np.asarray(file_sha256(statistics / "STATISTICS_RECEIPT.json")),
    }
    alignment_path = out / "alignment.npz"
    temporary = out / "alignment.incomplete.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(alignment_path)
    identity_path = out / "identity_alignment.npz"
    identity_arrays = dict(arrays)
    identity_arrays.update(
        a=np.ones_like(a), b=np.zeros_like(b), v_real=np.zeros_like(vr),
        v_imag=np.zeros_like(vi), u_real=np.zeros_like(ur), u_imag=np.zeros_like(ui),
        identity=np.ones_like(identity),
    )
    np.savez_compressed(out / "identity_alignment.incomplete.npz", **identity_arrays)
    (out / "identity_alignment.incomplete.npz").replace(identity_path)
    receipt = {
        "status": "ALIGNMENT_COMPLETE",
        "method_id": METHOD_ID,
        "layers": layers,
        "kv_groups": groups,
        "active_dimensions": active_size,
        "carrier_local_zero_based": carrier_local,
        "nonidentity_planes": int((~identity).sum()),
        "alignment_file": alignment_path.name,
        "alignment_sha256": file_sha256(alignment_path),
        "identity_alignment_file": identity_path.name,
        "identity_alignment_sha256": file_sha256(identity_path),
        "method_receipt_sha256": file_sha256(construction / "METHOD_RECEIPT.json"),
        "statistics_receipt_sha256": file_sha256(statistics / "STATISTICS_RECEIPT.json"),
        "plane_receipts": records,
        "selection_uses_task_outputs": False,
        "selection_uses_report_documents": False,
        "claim_boundary": "Top directions optimize the declared unlabeled signed-decay statistic; no task-quality result is implied.",
    }
    atomic_json(out / "ALIGNMENT_RECEIPT.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--ncp-table", type=Path)
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--construction", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.tables_only:
        if args.model is None or args.ncp_table is None or args.statistics or args.construction:
            raise ValueError("--tables-only requires only --model and --ncp-table")
        receipt = tables_only(args.model, args.ncp_table, args.out)
    else:
        if args.statistics is None or args.construction is None or args.model or args.ncp_table:
            raise ValueError("alignment solve requires --statistics and --construction")
        receipt = solve(args.statistics, args.construction, args.out)
    print(json.dumps({"status": receipt["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
