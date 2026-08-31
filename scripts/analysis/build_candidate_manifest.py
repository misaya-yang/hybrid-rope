#!/usr/bin/env python3
"""Assemble the evaluator candidate manifest for the success-first tournament.

Turns the frozen portfolio (F1 morph grid + Native control) plus any
development-stage representative receipts (F2/F3/F4) into the generic candidate
manifest consumed by ``scripts/eval/eval_zero_training_tournament.py``.

CPU only.  Every emitted entry carries the identity fields the evaluator and the
selection rule need: table path, float32 hash, pair count, family, construction
label, calibrated degrees of freedom, whether construction read long-range
outcomes, chord displacement, and support factor.
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

TIEBREAK_DOF = {"F1_PC_MORPH": 1, "F2_PC_RETENTION_PROJECT": 1,
                "F3_Z5_BEHAVIOUR": 5, "F4_SR_Z5": 6}
USES_LONG_RANGE = {"F1_PC_MORPH": False, "F2_PC_RETENTION_PROJECT": False,
                   "F3_Z5_BEHAVIOUR": True, "F4_SR_Z5": True}
W0_S4_SHA256 = "a435d75441444bcea39b73d9cf530005249dc5afdc3cfb5a60fda10ef33312d3"

from scripts.eval.eval_zero_training_tournament import (  # noqa: E402
    checkpoint_identity,
    sha256_file,
)
from scripts.analysis.freeze_success_first_portfolio import (  # noqa: E402
    reconstruct_phase_chord_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dev-receipt", type=Path, action="append", default=[],
                        help="development receipt whose representative joins the manifest")
    parser.add_argument("--families", nargs="+", default=None,
                        help="restrict to a subset of families (e.g. F1 only)")
    parser.add_argument(
        "--w0-s4-table", type=Path, default=None,
        help="emit the W0-only Native/official-YaRN-4/frozen-s4 manifest",
    )
    parser.add_argument(
        "--w0-decompose-gain", action="store_true",
        help="add Native+matched-gain and frozen-s4+unit-gain decomposition arms",
    )
    parser.add_argument(
        "--s4-gain-coeff-grid", nargs="+", type=float, default=None,
        help="with --w0-s4-table, add combined s4 table x attention-gain candidates",
    )
    parser.add_argument(
        "--segmented-pc-cutoff", type=int, default=None,
        help="protect Native pairs through this inclusive index and morph only the PC tail",
    )
    parser.add_argument(
        "--segmented-pc-grid", nargs="+", type=float,
        default=(0.05, 0.10, 0.20, 0.35),
    )
    parser.add_argument(
        "--query-gain-coefficient", type=float, default=None,
        help="with segmented PC, add the fixed post-boundary query-only gain factorial",
    )
    parser.add_argument(
        "--w0-s4-expected-sha256", default=W0_S4_SHA256,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def float32_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype="<f4")).tobytes()
    ).hexdigest()


def chord_rms(table_path: Path, native_table: np.ndarray) -> float:
    table = np.load(table_path, allow_pickle=False).astype(np.float64)
    displacement = np.log(table) - np.log(native_table.astype(np.float64))
    return float(np.sqrt(np.mean(np.square(displacement))))


def materialize_yarn_anchor(
    checkpoint: Path,
    output: Path,
    native_table: np.ndarray,
) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    config = AutoConfig.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    if config.model_type != "olmo2" or int(config.max_position_embeddings) != 4096:
        raise RuntimeError("official YaRN anchor requires the frozen 4K OLMo-2 config")
    native_rope = dict(getattr(config, "rope_parameters", None) or {})
    native_theta = getattr(config, "rope_theta", None)
    if native_theta is None:
        native_theta = native_rope.get("rope_theta")
    if native_theta is None or float(native_theta) != 500_000.0:
        raise RuntimeError("Native OLMo-2 rope theta drift")
    native_theta = float(native_theta)
    rope = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 4096,
        "rope_theta": native_theta,
    }
    config.rope_scaling = dict(rope)
    config.rope_parameters = dict(rope)
    config.max_position_embeddings = 16_384
    table, attention_scaling = ROPE_INIT_FUNCTIONS["yarn"](
        config, torch.device("cpu")
    )
    table = np.ascontiguousarray(table.detach().cpu().numpy(), dtype="<f4")
    if table.shape != native_table.shape or not np.all(table[:-1] > table[1:]):
        raise RuntimeError("official YaRN anchor table identity drift")
    table_path = output.parent / "official_yarn_factor4.npy"
    np.save(table_path, table, allow_pickle=False)
    return {
        "name": "official_yarn_factor4",
        "family": "REFERENCE_OFFICIAL_YARN",
        "path": str(table_path.resolve()),
        "table_sha256_float32": float32_sha256(table),
        "pair_count": int(table.size),
        "is_bitwise_native": False,
        "is_reference": True,
        "is_anchor": True,
        "attention_scaling": float(attention_scaling),
        "calibrated_dof": 0,
        "used_long_range_in_construction": False,
        "chord_displacement_rms": chord_rms(table_path, native_table),
        "support_factor": 4.0,
        "transformers_version": transformers.__version__,
        "rope_scaling": rope,
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    portfolio = json.loads(args.portfolio_manifest.resolve().read_text(encoding="utf-8"))
    f1 = portfolio["families"]["F1_PC_MORPH"]
    wanted = set(args.families) if args.families else None

    native_entry = next(e for e in f1["tables"] if e["is_bitwise_native"])
    native_table = np.load(native_entry["path"], allow_pickle=False)
    pair_count = int(native_table.size)

    candidates: list[dict[str, Any]] = []

    def add(entry: dict[str, Any], family: str, *, is_native: bool, support: float,
            path: str, table_hash: str, dof: int, uses_long: bool) -> None:
        if wanted is not None and family not in wanted and not is_native:
            return
        candidates.append({
            "name": entry["name"],
            "family": family,
            "path": path,
            "table_sha256_float32": table_hash,
            "pair_count": pair_count,
            "is_bitwise_native": bool(is_native),
            "is_reference": bool(is_native),
            "attention_scaling": 1.0,
            "calibrated_dof": int(dof),
            "used_long_range_in_construction": bool(uses_long),
            "chord_displacement_rms": float(entry.get("chord_displacement_rms", 0.0)),
            "support_factor": float(support),
        })

    if args.segmented_pc_cutoff is not None:
        if args.w0_s4_table is not None:
            raise ValueError("segmented PC and W0 manifests are separate roles")
        cutoff = int(args.segmented_pc_cutoff)
        if not 0 < cutoff < pair_count - 1:
            raise ValueError("segmented PC cutoff must leave protected and active interior pairs")
        source = portfolio.get("source", {})
        collection = Path(source["r0_collection"]).resolve()
        if sha256_file(collection) != source["r0_collection_sha256"]:
            raise RuntimeError("segmented PC R0 collection identity drift")
        reconstructed_native, target, _ = reconstruct_phase_chord_target(collection)
        if float32_sha256(reconstructed_native) != native_entry["inv_freq_float32_sha256"]:
            raise RuntimeError("segmented PC Native reconstruction drift")
        if float32_sha256(target) != portfolio["phase_chord_target"]["inv_freq_float32_sha256"]:
            raise RuntimeError("segmented PC target reconstruction drift")

        add(
            native_entry, "F1_SEGMENTED_PC_Z",
            is_native=True, support=1.0, path=native_entry["path"],
            table_hash=native_entry["inv_freq_float32_sha256"], dof=0, uses_long=False,
        )
        query_gain_coefficient = args.query_gain_coefficient
        if query_gain_coefficient is not None:
            query_gain_coefficient = float(query_gain_coefficient)
            if not math.isfinite(query_gain_coefficient) or query_gain_coefficient < 0.0:
                raise ValueError("query gain coefficient must be finite and non-negative")
            candidates.append({
                "name": "native_post_boundary_query_gain",
                "family": "REFERENCE_QUERY_GAIN_ONLY",
                "path": native_entry["path"],
                "table_sha256_float32": native_entry["inv_freq_float32_sha256"],
                "pair_count": pair_count,
                "is_bitwise_native": False,
                "is_reference": True,
                "is_combined_system": True,
                "attention_scaling": 1.0,
                "query_gain_coefficient": query_gain_coefficient,
                "calibrated_dof": 0,
                "used_long_range_in_construction": False,
                "chord_displacement_rms": 0.0,
                "support_factor": 1.0,
            })
        log_fast = math.log(float(reconstructed_native[0]))
        span = math.log(float(reconstructed_native[0] / reconstructed_native[-1]))
        native_z = (log_fast - np.log(reconstructed_native)) / span
        target_z = (log_fast - np.log(target)) / span
        tail_u = (target_z[cutoff:] - target_z[cutoff]) / (1.0 - target_z[cutoff])
        segmented_target_z = native_z.copy()
        segmented_target_z[cutoff:] = (
            native_z[cutoff] + (1.0 - native_z[cutoff]) * tail_u
        )
        tables_dir = output.parent / "segmented_pc_tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for amount in args.segmented_pc_grid:
            t = float(amount)
            if not math.isfinite(t) or not 0.0 < t <= 1.0:
                raise ValueError("segmented PC morph amounts must lie in (0, 1]")
            z = (1.0 - t) * native_z + t * segmented_target_z
            table = np.exp(log_fast - span * z)
            table[: cutoff + 1] = reconstructed_native[: cutoff + 1]
            table[-1] = reconstructed_native[-1]
            table = np.ascontiguousarray(table, dtype="<f4")
            if not np.all(table[:-1] > table[1:]):
                raise RuntimeError(f"segmented PC t={t:g} is not strictly decreasing")
            name = f"segpc_c{cutoff}_t{t:g}".replace(".", "p")
            table_path = tables_dir / f"{name}.npy"
            np.save(table_path, table, allow_pickle=False)
            candidates.append({
                "name": name,
                "family": "F1_SEGMENTED_PC_Z",
                "path": str(table_path.resolve()),
                "table_sha256_float32": float32_sha256(table),
                "pair_count": pair_count,
                "is_bitwise_native": False,
                "is_reference": False,
                "attention_scaling": 1.0,
                "calibrated_dof": 1,
                "used_long_range_in_construction": False,
                "chord_displacement_rms": chord_rms(table_path, native_table),
                "support_factor": 1.0,
                "protected_through_pair": cutoff,
                "morph_t": t,
            })
            if query_gain_coefficient is not None:
                candidates.append({
                    "name": f"{name}_post_query_gain",
                    "family": "SEGMENTED_PC_Z_QUERY_GAIN",
                    "path": str(table_path.resolve()),
                    "table_sha256_float32": float32_sha256(table),
                    "pair_count": pair_count,
                    "is_bitwise_native": False,
                    "is_reference": False,
                    "is_combined_system": True,
                    "attention_scaling": 1.0,
                    "query_gain_coefficient": query_gain_coefficient,
                    "calibrated_dof": 2,
                    "used_long_range_in_construction": False,
                    "chord_displacement_rms": chord_rms(table_path, native_table),
                    "support_factor": 1.0,
                    "protected_through_pair": cutoff,
                    "morph_t": t,
                })
    else:
        # Native control + F1 grid.
        for entry in f1["tables"]:
            if args.w0_s4_table is not None and not entry["is_bitwise_native"]:
                continue
            add(
                entry, "F1_PC_MORPH",
                is_native=bool(entry["is_bitwise_native"]),
                support=1.0,
                path=entry["path"],
                table_hash=entry["inv_freq_float32_sha256"],
                dof=0 if entry["is_bitwise_native"] else TIEBREAK_DOF["F1_PC_MORPH"],
                uses_long=USES_LONG_RANGE["F1_PC_MORPH"],
            )

    if args.w0_s4_table is not None:
        s4_path = args.w0_s4_table.resolve()
        s4 = np.load(s4_path, allow_pickle=False)
        s4_hash = float32_sha256(s4)
        if s4.dtype != np.dtype("float32") or s4.shape != native_table.shape:
            raise RuntimeError("W0 frozen s4 table identity drift")
        if s4_hash != args.w0_s4_expected_sha256:
            raise RuntimeError(
                f"W0 frozen s4 hash drift: expected {args.w0_s4_expected_sha256}, got {s4_hash}"
            )
        candidates.append({
            "name": "frozen_s4",
            "family": "REFERENCE_FROZEN_S4",
            "path": str(s4_path),
            "table_sha256_float32": s4_hash,
            "pair_count": pair_count,
            "is_bitwise_native": False,
            "is_reference": True,
            "attention_scaling": 1.0 + 0.1 * math.log(4.0),
            "calibrated_dof": 0,
            "used_long_range_in_construction": False,
            "chord_displacement_rms": chord_rms(s4_path, native_table),
            "support_factor": 4.0,
        })
        if args.w0_decompose_gain:
            matched_gain = 1.0 + 0.1 * math.log(4.0)
            candidates.extend([
                {
                    "name": "native_matched_gain",
                    "family": "REFERENCE_GAIN_ONLY",
                    "path": native_entry["path"],
                    "table_sha256_float32": native_entry["inv_freq_float32_sha256"],
                    "pair_count": pair_count,
                    "is_bitwise_native": False,
                    "is_reference": True,
                    "attention_scaling": matched_gain,
                    "calibrated_dof": 0,
                    "used_long_range_in_construction": False,
                    "chord_displacement_rms": 0.0,
                    "support_factor": 1.0,
                },
                {
                    "name": "frozen_s4_gain1",
                    "family": "S4_UNIT_GAIN_DECOMPOSITION",
                    "path": str(s4_path),
                    "table_sha256_float32": s4_hash,
                    "pair_count": pair_count,
                    "is_bitwise_native": False,
                    "is_reference": False,
                    "attention_scaling": 1.0,
                    "calibrated_dof": 0,
                    "used_long_range_in_construction": False,
                    "chord_displacement_rms": chord_rms(s4_path, native_table),
                    "support_factor": 4.0,
                },
            ])
        if args.s4_gain_coeff_grid:
            for coefficient in args.s4_gain_coeff_grid:
                c = float(coefficient)
                if not math.isfinite(c) or c < 0.0:
                    raise ValueError("s4 gain coefficients must be finite and non-negative")
                name = f"frozen_s4_gain_c{c:g}".replace(".", "p")
                candidates.append({
                    "name": name,
                    "family": "S4_Z_GAIN_CALIBRATION",
                    "path": str(s4_path),
                    "table_sha256_float32": s4_hash,
                    "pair_count": pair_count,
                    "is_bitwise_native": False,
                    "is_reference": False,
                    "is_combined_system": True,
                    "attention_scaling": 1.0 + c * math.log(4.0),
                    "attention_gain_coefficient": c,
                    "calibrated_dof": 1,
                    "used_long_range_in_construction": True,
                    "chord_displacement_rms": chord_rms(s4_path, native_table),
                    "support_factor": 4.0,
                })

    # Development representatives (F2/F3/F4), when their receipts are supplied.
    for receipt_path in (
        [] if args.w0_s4_table is not None or args.segmented_pc_cutoff is not None
        else args.dev_receipt
    ):
        receipt = json.loads(receipt_path.expanduser().resolve().read_text(encoding="utf-8"))
        representative = receipt.get("representative")
        if not representative or receipt.get("family_stop"):
            continue
        family = receipt["family"]
        table_path = Path(representative["path"]).resolve()
        table_hash = representative["table_sha256_float32"]
        support = float(receipt.get("report", {}).get("support_factor", 1.0))
        name = f"{family}_rep"
        candidates.append({
            "name": name,
            "family": family,
            "path": str(table_path),
            "table_sha256_float32": table_hash,
            "pair_count": pair_count,
            "is_bitwise_native": False,
            "is_reference": False,
            "attention_scaling": 1.0,
            "calibrated_dof": int(TIEBREAK_DOF.get(family, 0)),
            "used_long_range_in_construction": bool(USES_LONG_RANGE.get(family, False)),
            "chord_displacement_rms": chord_rms(table_path, native_table),
            "support_factor": support,
        })

    candidates.append(
        materialize_yarn_anchor(
            args.checkpoint.resolve(), output, native_table
        )
    )

    names = [c["name"] for c in candidates]
    if len(set(names)) != len(names):
        raise RuntimeError(f"candidate names are not unique: {names}")
    payload = {
        "status": "SUCCESS_FIRST_CANDIDATES_ASSEMBLED_V1",
        "pair_count": pair_count,
        "manifest_role": (
            "S4_GAIN_DEVELOPMENT" if args.s4_gain_coeff_grid
            else "W0_ANCHOR" if args.w0_s4_table is not None
            else "SEGMENTED_PC_DEVELOPMENT" if args.segmented_pc_cutoff is not None
            else "TOURNAMENT"
        ),
        "portfolio_content_sha256": portfolio.get("content_sha256"),
        "checkpoint_identity": checkpoint_identity(args.checkpoint.resolve()),
        "candidates": candidates,
    }
    if args.segmented_pc_cutoff is not None:
        payload["segmented_pc"] = {
            "protected_through_pair": int(args.segmented_pc_cutoff),
            "morph_grid": [float(t) for t in args.segmented_pc_grid],
            "support_factor": 1.0,
            "attention_scaling": 1.0,
            "query_gain_coefficient": args.query_gain_coefficient,
            "construction": "Native z through cutoff; conditionally normalized phase-chord tail",
        }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"candidates": len(candidates), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
