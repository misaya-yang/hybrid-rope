#!/usr/bin/env python3
"""Zero-GPU retrofittability analysis for a mature RoPE checkpoint.

For every candidate frequency table this reports, without any model forward,
optimizer step, or benchmark:

* ``D0``  -- expected squared logit error of a hard table swap;
* ``D*``  -- the same after the best fixed, position-independent Q/K content
             maps, i.e. the exact operator class of the transplant obstruction
             and a strict superset of any Q/K LoRA;
* the minimum LoRA rank per head that reaches a given fraction of ``D*``;
* which rotary pairs carry the unrepairable residual;
* what the table can still resolve over the target range.

Fail-closed: CUDA must not be visible. This module never loads a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.rope_transport import conditioning, tables, transport, weights  # noqa: E402

METHOD_ID = "rope_transport_analysis_v1"
DEFAULT_RANKS = (1, 2, 4, 8, 16, 32, 64, 128)


def _require_no_cuda() -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "-1"):
        raise RuntimeError(
            "this analysis is CPU-only; unset CUDA_VISIBLE_DEVICES or set it to -1"
        )
    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported by the CPU analysis")
    return {"cuda_visible_devices": visible, "torch_imported": False}


def _source_hashes() -> Dict[str, str]:
    here = Path(__file__).resolve().parent
    out: Dict[str, str] = {}
    for path in sorted(here.glob("*.py")):
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def build_candidates(
    frozen: List[tables.Table],
    *,
    head_dim: int,
    rope_base: float,
    native_length: int,
    scales: List[float],
    uniqueness: np.ndarray,
    exponents: List[float],
) -> List[tables.Table]:
    by_name = {table.name: table for table in frozen}
    if "native" not in by_name:
        raise ValueError("manifest must contain a 'native' table")
    native = by_name["native"]
    out: List[tables.Table] = list(frozen)
    for scale in scales:
        out.append(tables.position_interpolation(native, scale))
        out.append(
            tables.official_yarn(
                native,
                scale=scale,
                head_dim=head_dim,
                rope_base=rope_base,
                original_max_position_embeddings=native_length,
            )
        )
        for exponent in exponents:
            out.append(tables.budgeted_transport(native, uniqueness, scale=scale, exponent=exponent))
    return out


def analyse_table(
    native: tables.Table,
    candidate: tables.Table,
    *,
    support: np.ndarray,
    weight: np.ndarray,
    far_support: np.ndarray,
    far_weight: np.ndarray,
    native_length: int,
    far_targets: List[int],
    ranks: List[int],
    max_iter: int,
) -> Dict[str, Any]:
    started = time.time()
    full = transport.transport_residual(
        native.inv_freq, candidate.inv_freq, support, weight, max_iter=max_iter
    )
    parts = transport.residual_by_pair(
        native.inv_freq, candidate.inv_freq, support, weight, full.query_map, full.key_map
    )
    rank_rows = []
    for rank in ranks:
        if rank > native.inv_freq.size * 2:
            continue
        row = transport.transport_residual(
            native.inv_freq, candidate.inv_freq, support, weight, rank=rank, max_iter=max_iter
        )
        rank_rows.append(
            {
                "rank": int(rank),
                "repaired": row.repaired,
                "relative_repaired": row.relative_repaired,
                "repairability": row.repairability,
                "fraction_of_full_rank_gain": (
                    float("nan")
                    if full.hard_swap - full.repaired <= 0.0
                    else (full.hard_swap - row.repaired) / (full.hard_swap - full.repaired)
                ),
            }
        )

    spectrum_q = np.linalg.svd(
        full.query_map - np.eye(full.query_map.shape[0]), compute_uv=False
    )
    spectrum_k = np.linalg.svd(full.key_map - np.eye(full.key_map.shape[0]), compute_uv=False)

    def effective_rank(values: np.ndarray, fraction: float) -> int:
        total = float((values ** 2).sum())
        # A numerically zero map is the identity: it needs no rank at all.
        if total <= 1e-20 * max(1.0, float(values.size)):
            return 0
        cumulative = np.cumsum(values ** 2) / total
        return int(np.searchsorted(cumulative, fraction) + 1)

    resolvability = conditioning.range_resolvability(
        candidate.inv_freq,
        far_support,
        far_weight,
        trained_omega=native.inv_freq,
        trained_length=native_length,
    )
    phase_safety = {}
    for target in far_targets:
        sup, wt, _ = weights.distance_weight("causal", length=int(target), max_points=64)
        report = conditioning.range_resolvability(
            candidate.inv_freq,
            sup,
            wt,
            trained_omega=native.inv_freq,
            trained_length=native_length,
        )
        phase_safety[str(int(target))] = {
            "phase_safe_fraction": report["phase_safe_fraction"],
            "phase_safe_count": report["phase_safe_count"],
        }
    log_shift = np.log(candidate.inv_freq) - np.log(native.inv_freq)
    return {
        "table": candidate.as_record(),
        "hard_swap": max(0.0, full.hard_swap),
        "relative_hard_swap": max(0.0, full.relative_hard_swap),
        "repaired": full.repaired,
        "relative_repaired": full.relative_repaired,
        "repairability": full.repairability,
        "converged": full.converged,
        "iterations": full.iterations,
        "selected_start": full.history[-1].get("selected_start"),
        "rank_sweep": rank_rows,
        "query_map_rank_90": effective_rank(spectrum_q, 0.90),
        "query_map_rank_99": effective_rank(spectrum_q, 0.99),
        "key_map_rank_90": effective_rank(spectrum_k, 0.90),
        "key_map_rank_99": effective_rank(spectrum_k, 0.99),
        "residual_top_query_pairs": np.argsort(-parts["query_pair_energy"])[:8].tolist(),
        "residual_query_pair_energy": parts["query_pair_energy"].tolist(),
        "log_frequency_shift": {
            "mean": float(log_shift.mean()),
            "rms": float(np.sqrt((log_shift ** 2).mean())),
            "max_abs": float(np.abs(log_shift).max()),
            "unchanged_pairs": int((np.abs(log_shift) < 1e-12).sum()),
        },
        "far_range": resolvability,
        "phase_safety": phase_safety,
        "no_positional_signal_reference": 1.0,
        "seconds": time.time() - started,
    }


def _dominance(rows: List[Dict[str, Any]], target: int) -> Dict[str, Any]:
    """Strict Pareto dominance on (in-window damage, far-range phase safety)."""
    key = str(int(target))
    points = [
        {
            "name": row["table"]["name"],
            "cost": float(row["relative_repaired"]),
            "benefit": float(row["phase_safety"][key]["phase_safe_fraction"]),
        }
        for row in rows
    ]
    out = []
    for point in points:
        dominated_by = [
            other["name"]
            for other in points
            if other["name"] != point["name"]
            and other["cost"] <= point["cost"] + 1e-12
            and other["benefit"] >= point["benefit"] - 1e-12
            and (other["cost"] < point["cost"] - 1e-12 or other["benefit"] > point["benefit"] + 1e-12)
        ]
        out.append({**point, "dominated_by": dominated_by, "on_frontier": not dominated_by})
    return {
        "target_length": int(target),
        "cost_metric": "relative_repaired (D* / d), lower is better",
        "benefit_metric": "phase_safe_fraction at target length, higher is better",
        "points": out,
        "frontier": sorted(p["name"] for p in out if p["on_frontier"]),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="frozen target manifest JSON")
    parser.add_argument("--output", required=True, help="destination receipt JSON")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rope-base", type=float, default=500000.0)
    parser.add_argument("--native-length", type=int, default=4096)
    parser.add_argument("--target-length", type=int, default=16384)
    parser.add_argument("--phase-safety-lengths", type=int, nargs="*", default=[8192, 16384])
    parser.add_argument("--weight-family", default="causal", choices=list(weights.FAMILIES))
    parser.add_argument("--weight-alpha", type=float, default=1.0)
    parser.add_argument("--support-points", type=int, default=2048)
    parser.add_argument("--scales", type=float, nargs="*", default=[2.0, 4.0])
    parser.add_argument("--exponents", type=float, nargs="*", default=[1.0, 2.0])
    parser.add_argument("--ranks", type=int, nargs="*", default=list(DEFAULT_RANKS))
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument(
        "--emit-tables",
        default=None,
        help="optional directory for derived float32 .npy tables plus a hash index",
    )
    args = parser.parse_args(argv)

    runtime = _require_no_cuda()
    started = time.time()

    frozen = tables.load_manifest_tables(args.manifest)
    by_name = {table.name: table for table in frozen}
    native = by_name["native"]
    if native.inv_freq.size * 2 != args.head_dim:
        raise ValueError(
            f"head_dim {args.head_dim} does not match manifest pairs {native.inv_freq.size}"
        )

    support, weight, weight_meta = weights.distance_weight(
        args.weight_family,
        length=args.native_length,
        alpha=args.weight_alpha,
        max_points=args.support_points,
    )
    far_support, far_weight, far_meta = weights.distance_weight(
        args.weight_family,
        length=args.target_length,
        alpha=args.weight_alpha,
        max_points=args.support_points,
    )

    unique = conditioning.pair_uniqueness(native.inv_freq, support, weight)
    candidates = build_candidates(
        frozen,
        head_dim=args.head_dim,
        rope_base=args.rope_base,
        native_length=args.native_length,
        scales=list(args.scales),
        uniqueness=unique["uniqueness"],
        exponents=list(args.exponents),
    )

    rows = []
    for candidate in candidates:
        rows.append(
            analyse_table(
                native,
                candidate,
                support=support,
                weight=weight,
                far_support=far_support,
                far_weight=far_weight,
                native_length=args.native_length,
                far_targets=list(args.phase_safety_lengths),
                ranks=list(args.ranks),
                max_iter=args.max_iter,
            )
        )
        row = rows[-1]
        print(
            f"{row['table']['name']:<38s} "
            f"D0={row['relative_hard_swap']:8.4f}  "
            f"D*={row['relative_repaired']:8.4f}  "
            f"repairability={row['repairability']:7.4f}  "
            f"rank90={row['query_map_rank_90']:4d}  "
            + "  ".join(
                f"safe@{name}={cell['phase_safe_fraction']:5.3f}"
                for name, cell in sorted(row["phase_safety"].items(), key=lambda kv: int(kv[0]))
            ),
            flush=True,
        )

    payload: Dict[str, Any] = {
        "method_id": METHOD_ID,
        "status": "CPU_ONLY_NO_GPU_NO_CHECKPOINT",
        "protocol": {
            "head_dim": args.head_dim,
            "rope_base": args.rope_base,
            "native_length": args.native_length,
            "target_length": args.target_length,
            "in_window_weight": weight_meta,
            "far_range_weight": far_meta,
            "scales": list(args.scales),
            "exponents": list(args.exponents),
            "ranks": list(args.ranks),
            "phase_safety_lengths": list(args.phase_safety_lengths),
            "max_iter": args.max_iter,
            "content_model": "isotropic; E[(q^T A k - q^T B k)^2] = ||A-B||_F^2",
        },
        "native_pair_uniqueness": unique["uniqueness"].tolist(),
        "native_conditional_energy": unique["conditional_energy"].tolist(),
        "results": rows,
        "dominance": _dominance(rows, int(max(args.phase_safety_lengths))),
        "runtime": {
            **runtime,
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "seconds": time.time() - started,
        },
        "source_sha256": _source_hashes(),
        "claim_boundary": (
            "D* is an upper bound on the best static Q/K repair under isotropic "
            "content on the stated distance weight. It is not a task metric, not "
            "a measured attention distribution, and not evidence about any "
            "trained checkpoint."
        ),
    }
    if args.emit_tables:
        emit_dir = Path(args.emit_tables).resolve()
        emit_dir.mkdir(parents=True, exist_ok=True)
        index = {}
        for candidate in candidates:
            if candidate.origin != "derived":
                continue
            values = np.asarray(candidate.inv_freq, dtype="<f4")
            path = emit_dir / f"{candidate.name}.npy"
            np.save(path, values)
            index[candidate.name] = {
                "path": str(path),
                "float32_sha256": candidate.sha256,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "meta": candidate.meta,
            }
        (emit_dir / "index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        payload["emitted_tables"] = index

    payload["content_sha256"] = _canonical_sha256(payload)

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nreceipt: {out}")
    print(f"receipt sha256: {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
