#!/usr/bin/env python3
"""Search the feasible set of training-free RoPE retrofit operators.

The question this answers is not "is our shape better than YaRN". It is:

    over every monotone frequency table that is fully phase-safe at the
    deployed length, how small can the unrepairable in-window residual D* be,
    and how much of that optimum does a *one-parameter* family recover?

Stage 1 sweeps one-parameter warps of the redundant band (including the
paper's EVQ-Cosh quantile warp) plus the published references.
Stage 2 runs a derivative-free search over the free monotone re-layout of the
band, giving the achievable floor for the whole operator class.
Stage 3 re-evaluates the finalists at full support resolution with the LoRA
rank sweep and per-pair residual attribution.

CPU-only and fail-closed: CUDA must not be visible and no checkpoint is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.rope_transport import conditioning, nullband, tables, transport, weights  # noqa: E402

METHOD_ID = "rope_nullband_search_v1"
TWO_PI = 2.0 * math.pi
_CTX: Dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# guards and identity
# --------------------------------------------------------------------------- #


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
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob("*.py"))}


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def phase_safe_fraction(
    omega_native: np.ndarray, omega_new: np.ndarray, native_length: int, target_length: int
) -> float:
    trained = np.asarray(omega_native, dtype=np.float64) * float(native_length)
    deployed = np.asarray(omega_new, dtype=np.float64) * float(target_length)
    safe = (deployed <= trained + 1e-12) | (trained >= TWO_PI)
    return float(safe.mean())


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #


def _init_worker(ctx: Dict[str, Any]) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    _CTX.update(ctx)
    _CTX["native"] = np.asarray(ctx["native"], dtype=np.float64)
    _CTX["support"] = np.asarray(ctx["support"], dtype=np.float64)
    _CTX["weight"] = np.asarray(ctx["weight"], dtype=np.float64)


def _score(omega_new: np.ndarray) -> Dict[str, float]:
    """Search-resolution D* and safety for one candidate table."""
    native = _CTX["native"]
    res = transport.transport_residual(
        native,
        np.asarray(omega_new, dtype=np.float64),
        _CTX["support"],
        _CTX["weight"],
        max_iter=int(_CTX["max_iter"]),
        tol=float(_CTX["tol"]),
    )
    out = {
        "d0": res.relative_hard_swap,
        "dstar": res.relative_repaired,
        "iterations": float(res.iterations),
    }
    for length in _CTX["safety_lengths"]:
        out[f"safe@{length}"] = phase_safe_fraction(
            native, omega_new, _CTX["native_length"], int(length)
        )
    return out


def _eval_parametric(spec: Dict[str, Any]) -> Dict[str, Any]:
    table = nullband.band_warp_table(
        _CTX["native_table"],
        warp=spec["warp"],
        param=spec["param"],
        scale=spec["scale"],
        band_start=spec["band_start"],
        rope_base=_CTX["rope_base"],
    )
    row = dict(spec)
    row.update(_score(table.inv_freq))
    row["sha256"] = table.sha256
    return row


def _eval_free(task: Tuple[int, np.ndarray, int, float]) -> Tuple[int, float, Dict[str, float]]:
    index, z, band_start, scale = task
    table = nullband.free_band_table(
        _CTX["native_table"],
        z,
        scale=scale,
        band_start=band_start,
        rope_base=_CTX["rope_base"],
        floor=_CTX["phi_floor"],
        name="free_candidate",
    )
    stats = _score(table.inv_freq)
    penalty = 0.0
    target = int(_CTX["target_length"])
    if stats.get(f"safe@{target}", 1.0) < 1.0 - 1e-12:
        penalty = 10.0 * (1.0 - stats[f"safe@{target}"])
    return index, stats["dstar"] + penalty, stats


# --------------------------------------------------------------------------- #
# differential evolution (numpy only, deterministic, pool-parallel)
# --------------------------------------------------------------------------- #


def differential_evolution(
    pool: Any,
    *,
    dim: int,
    band_start: int,
    scale: float,
    popsize: int,
    iters: int,
    seed: int,
    lo: float = -6.0,
    hi: float = 6.0,
    diff_weight: float = 0.6,
    crossover: float = 0.9,
    seeds_extra: List[np.ndarray] | None = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    pop = rng.uniform(lo, hi, size=(popsize, dim))
    pop[0] = 0.0  # the linear-stretch member
    for offset, extra in enumerate(seeds_extra or [], start=1):
        if offset < popsize:
            pop[offset] = np.clip(np.asarray(extra, dtype=np.float64), lo, hi)

    tasks = [(i, pop[i], band_start, scale) for i in range(popsize)]
    fitness = np.empty(popsize, dtype=np.float64)
    stats: List[Dict[str, float]] = [dict() for _ in range(popsize)]
    for index, value, st in pool.map(_eval_free, tasks, chunksize=1):
        fitness[index] = value
        stats[index] = st

    history = [{"generation": 0, "best": float(fitness.min()), "mean": float(fitness.mean())}]
    for generation in range(1, int(iters) + 1):
        trials = np.empty_like(pop)
        for i in range(popsize):
            choices = [j for j in range(popsize) if j != i]
            a, b, c = rng.choice(choices, size=3, replace=False)
            mutant = np.clip(pop[a] + diff_weight * (pop[b] - pop[c]), lo, hi)
            mask = rng.random(dim) < crossover
            if not mask.any():
                mask[rng.integers(dim)] = True
            trials[i] = np.where(mask, mutant, pop[i])
        tasks = [(i, trials[i], band_start, scale) for i in range(popsize)]
        for index, value, st in pool.map(_eval_free, tasks, chunksize=1):
            if value < fitness[index]:
                fitness[index] = value
                pop[index] = trials[index]
                stats[index] = st
        history.append(
            {"generation": generation, "best": float(fitness.min()), "mean": float(fitness.mean())}
        )

    best = int(np.argmin(fitness))
    return {
        "band_start": int(band_start),
        "scale": float(scale),
        "dim": int(dim),
        "popsize": int(popsize),
        "iterations": int(iters),
        "seed": int(seed),
        "best_value": float(fitness[best]),
        "best_stats": stats[best],
        "best_z": pop[best].tolist(),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# stage 3 verification
# --------------------------------------------------------------------------- #


def verify(
    native: tables.Table,
    candidate: tables.Table,
    *,
    support: np.ndarray,
    weight: np.ndarray,
    native_length: int,
    safety_lengths: List[int],
    ranks: List[int],
    max_iter: int,
) -> Dict[str, Any]:
    full = transport.transport_residual(
        native.inv_freq, candidate.inv_freq, support, weight, max_iter=max_iter
    )
    parts = transport.residual_by_pair(
        native.inv_freq, candidate.inv_freq, support, weight, full.query_map, full.key_map
    )
    rank_rows = []
    for rank in ranks:
        res = transport.transport_residual(
            native.inv_freq, candidate.inv_freq, support, weight, rank=rank, max_iter=max_iter
        )
        gain_full = full.relative_hard_swap - full.relative_repaired
        gain_rank = full.relative_hard_swap - res.relative_repaired
        rank_rows.append(
            {
                "rank_per_head": int(rank),
                "dstar": res.relative_repaired,
                "gain_fraction": float(gain_rank / gain_full) if gain_full > 0 else float("nan"),
            }
        )
    energy = np.asarray(parts["query_pair_energy"], dtype=np.float64)
    order = np.argsort(energy)[::-1]
    return {
        "name": candidate.name,
        "sha256": candidate.sha256,
        "origin": candidate.origin,
        "meta": candidate.meta,
        "d0": full.relative_hard_swap,
        "dstar": full.relative_repaired,
        "iterations": full.iterations,
        "converged": bool(full.converged),
        "rank_sweep": rank_rows,
        "top_residual_pairs": [int(i) for i in order[:8]],
        "phase_safe": {
            str(length): phase_safe_fraction(
                native.inv_freq, candidate.inv_freq, native_length, int(length)
            )
            for length in safety_lengths
        },
    }


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--emit-tables")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--rope-base", type=float, default=500000.0)
    ap.add_argument("--native-length", type=int, default=4096)
    ap.add_argument("--target-length", type=int, default=16384)
    ap.add_argument("--safety-lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--scales", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--search-points", type=int, default=512)
    ap.add_argument("--verify-points", type=int, default=2048)
    ap.add_argument("--search-max-iter", type=int, default=25)
    ap.add_argument("--verify-max-iter", type=int, default=60)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--uniqueness-threshold", type=float, default=0.01)
    ap.add_argument("--extra-bands", type=int, nargs="*", default=[16, 26, 40])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--de-pop", type=int, default=120)
    ap.add_argument("--de-iters", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--finalists", type=int, default=8)
    args = ap.parse_args()

    started = time.time()
    guard = _require_no_cuda()
    workers = int(args.workers) or max(1, (os.cpu_count() or 8) - 4)

    frozen = tables.load_manifest_tables(args.manifest)
    by_name = {t.name: t for t in frozen}
    native = by_name["native"]
    pairs = native.inv_freq.size

    support, weight, weight_meta = weights.distance_weight(
        "causal", length=args.native_length, max_points=args.verify_points
    )
    s_support, s_weight, s_meta = weights.distance_weight(
        "causal", length=args.native_length, max_points=args.search_points
    )

    uniq = conditioning.pair_uniqueness(native.inv_freq, support, weight)["uniqueness"]
    below = np.flatnonzero(uniq < float(args.uniqueness_threshold))
    cliff = int(below[0]) if below.size else pairs - 1
    box = nullband.phase_safety_box(
        native.inv_freq, native_length=args.native_length, target_length=args.target_length
    )
    first_unwrapped = int(box["unwrapped_index"][0]) if box["unwrapped_index"].size else pairs
    fl = nullband.phi_floor(
        native.inv_freq,
        native_length=args.native_length,
        target_length=args.target_length,
        rope_base=args.rope_base,
    )

    bands = sorted({cliff, first_unwrapped, *[int(b) for b in (args.extra_bands or [])]})
    bands = [b for b in bands if 0 <= b <= pairs - 2]
    print(
        f"[stage0] pairs={pairs} uniqueness_cliff={cliff} first_unwrapped={first_unwrapped} "
        f"bands={bands} workers={workers}",
        flush=True,
    )

    ctx = {
        "native": native.inv_freq,
        "native_table": native,
        "support": s_support,
        "weight": s_weight,
        "max_iter": int(args.search_max_iter),
        "tol": 1e-10,
        "safety_lengths": [int(x) for x in args.safety_lengths],
        "native_length": int(args.native_length),
        "target_length": int(args.target_length),
        "rope_base": float(args.rope_base),
        "phi_floor": fl,
    }

    # ---------------- stage 1: one-parameter families ---------------- #
    specs: List[Dict[str, Any]] = []
    for scale in args.scales:
        for band_start in bands:
            specs.append({"warp": "identity", "param": 0.0, "scale": scale, "band_start": band_start})
            for tau in np.round(np.arange(-4.0, 4.01, 0.25), 4):
                if abs(float(tau)) < 1e-9:
                    continue
                specs.append(
                    {"warp": "evq_cosh", "param": float(tau), "scale": scale, "band_start": band_start}
                )
            for p in np.round(np.exp(np.linspace(math.log(0.3), math.log(3.5), 21)), 4):
                specs.append(
                    {"warp": "power", "param": float(p), "scale": scale, "band_start": band_start}
                )
            for c in np.round(np.arange(-10.0, 10.01, 1.0), 4):
                if abs(float(c)) < 1e-9:
                    continue
                specs.append(
                    {"warp": "logistic", "param": float(c), "scale": scale, "band_start": band_start}
                )

    reference_tables: List[tables.Table] = []
    skipped_references: List[Dict[str, str]] = []

    def _try_reference(label: str, build) -> None:
        # A reference construction can legitimately fail its monotonicity
        # contract (the budgeted rule inherits any noise in the uniqueness
        # estimate). Record the skip rather than losing the whole run.
        try:
            reference_tables.append(build())
        except ValueError as exc:
            skipped_references.append({"label": label, "error": str(exc)})

    for scale in args.scales:
        _try_reference(f"pi_s{scale:g}", lambda s=scale: tables.position_interpolation(native, s))
        _try_reference(
            f"yarn_s{scale:g}",
            lambda s=scale: tables.official_yarn(
                native,
                scale=s,
                head_dim=args.head_dim,
                rope_base=args.rope_base,
                original_max_position_embeddings=args.native_length,
            ),
        )
        _try_reference(
            f"budgeted_s{scale:g}_p2",
            lambda s=scale: tables.budgeted_transport(native, uniq, scale=s, exponent=2.0),
        )
        _try_reference(
            f"safety_floor_s{scale:g}",
            lambda s=scale: nullband.floor_table(
                native,
                scale=s,
                native_length=args.native_length,
                target_length=int(args.native_length * s),
                rope_base=args.rope_base,
                name=f"safety_floor_s{s:g}",
            ),
        )
        for beta in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0):
            _try_reference(
                f"turnbudget_b{beta:g}_s{scale:g}",
                lambda s=scale, b=beta: nullband.turn_budget_table(
                    native,
                    scale=s,
                    beta=b,
                    ramp_turns=max(0.0, b - 1.0),
                    native_length=args.native_length,
                    rope_base=args.rope_base,
                    name=f"turnbudget_b{b:g}_s{s:g}",
                ),
            )
    reference_tables.extend(t for t in frozen if t.name != "native")

    mp_ctx = mp.get_context("fork")
    with mp_ctx.Pool(processes=workers, initializer=_init_worker, initargs=(ctx,)) as pool:
        t0 = time.time()
        parametric = pool.map(_eval_parametric, specs, chunksize=1)
        print(f"[stage1] {len(parametric)} parametric candidates in {time.time() - t0:.1f}s", flush=True)

        ref_rows = []
        for table in reference_tables:
            row = {"warp": "reference", "param": float("nan"), "scale": float("nan"),
                   "band_start": -1, "name": table.name, "sha256": table.sha256}
            row.update(pool.apply(_score, (table.inv_freq,)))
            ref_rows.append(row)
        print(f"[stage1] {len(ref_rows)} reference tables scored", flush=True)

        # ---------------- stage 2: free monotone search ---------------- #
        free_runs = []
        for scale in args.scales:
            for band_start in sorted({cliff, first_unwrapped}):
                dim = pairs - band_start - 1
                best_param = min(
                    (r for r in parametric
                     if r["band_start"] == band_start and r["scale"] == scale
                     and r.get(f"safe@{args.target_length}", 0.0) >= 1.0 - 1e-12),
                    key=lambda r: r["dstar"],
                    default=None,
                )
                t0 = time.time()
                run = differential_evolution(
                    pool,
                    dim=dim,
                    band_start=band_start,
                    scale=scale,
                    popsize=int(args.de_pop),
                    iters=int(args.de_iters),
                    seed=int(args.seed) + band_start * 100 + int(scale),
                )
                run["seeded_from_best_parametric"] = None if best_param is None else {
                    k: best_param[k] for k in ("warp", "param", "dstar")
                }
                run["wall_seconds"] = time.time() - t0
                free_runs.append(run)
                print(
                    f"[stage2] band_start={band_start} scale={scale:g} dim={dim} "
                    f"best_dstar={run['best_value']:.4f} in {run['wall_seconds']:.1f}s",
                    flush=True,
                )

    # ---------------- stage 3: verify finalists at full resolution ---------------- #
    finalists: List[tables.Table] = []
    seen: set = set()

    def _add(table: tables.Table) -> None:
        if table.sha256 in seen:
            return
        seen.add(table.sha256)
        finalists.append(table)

    _add(native)
    for table in reference_tables:
        if table.name.startswith(
            ("yarn_", "pi_", "budgeted_", "safety_floor", "turnbudget_")
        ) or table.origin == "frozen_manifest":
            _add(table)

    safe_parametric = [
        r for r in parametric if r.get(f"safe@{args.target_length}", 0.0) >= 1.0 - 1e-12
    ]
    safe_parametric.sort(key=lambda r: r["dstar"])
    for row in safe_parametric[: int(args.finalists)]:
        _add(
            nullband.band_warp_table(
                native,
                warp=row["warp"],
                param=row["param"],
                scale=row["scale"],
                band_start=row["band_start"],
                rope_base=args.rope_base,
            )
        )
    # best member of each family at each band, for the family comparison
    for scale in args.scales:
        for band_start in sorted({cliff, first_unwrapped}):
            for family in ("identity", "evq_cosh", "power", "logistic"):
                pool_rows = [
                    r for r in safe_parametric
                    if r["warp"] == family and r["band_start"] == band_start and r["scale"] == scale
                ]
                if pool_rows:
                    row = min(pool_rows, key=lambda r: r["dstar"])
                    _add(
                        nullband.band_warp_table(
                            native, warp=row["warp"], param=row["param"], scale=row["scale"],
                            band_start=row["band_start"], rope_base=args.rope_base,
                        )
                    )
    for run in free_runs:
        _add(
            nullband.free_band_table(
                native,
                np.asarray(run["best_z"], dtype=np.float64),
                scale=run["scale"],
                band_start=run["band_start"],
                rope_base=args.rope_base,
                floor=fl,
                name=f"free_band_j{run['band_start']}_s{run['scale']:g}",
            )
        )

    print(f"[stage3] verifying {len(finalists)} finalists at {support.size} support points", flush=True)
    verified = []
    for table in finalists:
        t0 = time.time()
        row = verify(
            native,
            table,
            support=support,
            weight=weight,
            native_length=args.native_length,
            safety_lengths=[int(x) for x in args.safety_lengths],
            ranks=[int(r) for r in args.ranks],
            max_iter=int(args.verify_max_iter),
        )
        row["wall_seconds"] = time.time() - t0
        verified.append(row)
        print(
            f"  {table.name:44s} D0={row['d0']:.4f} D*={row['dstar']:.4f} "
            + " ".join(f"safe@{k}={v:.3f}" for k, v in row["phase_safe"].items()),
            flush=True,
        )

    if args.emit_tables:
        out_dir = Path(args.emit_tables)
        out_dir.mkdir(parents=True, exist_ok=True)
        index = []
        for table in finalists:
            arr = np.asarray(table.inv_freq, dtype="<f4")
            np.save(out_dir / f"{table.name}.npy", arr)
            index.append(table.as_record())
        (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    payload = {
        "method_id": METHOD_ID,
        "guard": guard,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": time.time() - started,
        "host": {"node": socket.gethostname(), "platform": platform.platform(),
                 "python": sys.version.split()[0], "numpy": np.__version__, "workers": workers},
        "source_sha256": _source_hashes(),
        "config": vars(args),
        "weight_meta": {"verify": weight_meta, "search": s_meta},
        "native": native.as_record(),
        "structure": {
            "pairs": int(pairs),
            "uniqueness": uniq.tolist(),
            "uniqueness_threshold": float(args.uniqueness_threshold),
            "uniqueness_cliff_index": cliff,
            "first_unwrapped_index": first_unwrapped,
            "wrapped_count": int(box["wrapped"].sum()),
            "bands_searched": bands,
        },
        "stage1_parametric": parametric,
        "stage1_references": ref_rows,
        "skipped_references": skipped_references,
        "stage2_free": free_runs,
        "stage3_verified": verified,
    }
    payload["receipt_sha256"] = _canonical_sha256(
        {k: v for k, v in payload.items() if k != "receipt_sha256"}
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"RECEIPT {out_path} sha256={payload['receipt_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
