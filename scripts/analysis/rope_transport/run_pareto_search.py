#!/usr/bin/env python3
"""The two-objective frontier of training-free RoPE retrofit operators.

Axes, both computed in closed form with no checkpoint and no GPU:

* ``D*``   -- in-window logit energy that **no** fixed Q/K content map can
              restore after the table swap (the transplant obstruction's own
              operator class, a strict superset of any Q/K LoRA);
* ``risk`` -- mean unseen phase per channel at the deployed length, in turns.

Every published context-extension operator is a point in this plane. This
script places them, sweeps one-parameter families through the same plane, and
runs a derivative-free search for the frontier itself, so the question
"is there a better non-geometric allocation" becomes a measurement rather
than a preference.

CPU-only and fail-closed.
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
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.rope_transport import conditioning, nullband, tables, transport, weights  # noqa: E402

METHOD_ID = "rope_pareto_search_v1"
_CTX: Dict[str, Any] = {}


def _require_no_cuda() -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "-1"):
        raise RuntimeError("this analysis is CPU-only; set CUDA_VISIBLE_DEVICES=-1")
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


def _init_worker(ctx: Dict[str, Any]) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    _CTX.update(ctx)


def _score(omega_new: np.ndarray) -> Dict[str, float]:
    native = _CTX["native"]
    res = transport.transport_residual(
        native, np.asarray(omega_new, dtype=np.float64), _CTX["support"], _CTX["weight"],
        max_iter=int(_CTX["max_iter"]), tol=float(_CTX["tol"]),
    )
    out = {"d0": res.relative_hard_swap, "dstar": res.relative_repaired}
    for length in _CTX["targets"]:
        risk = nullband.phase_excess_risk(
            native, omega_new, native_length=_CTX["native_length"], target_length=int(length)
        )
        out[f"risk@{length}"] = risk["mean_turns"]
        out[f"maxrisk@{length}"] = risk["max_turns"]
    return out


def _eval_named(item: Tuple[str, np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
    name, omega, meta = item
    row = {"name": name, "meta": meta}
    row.update(_score(omega))
    return row


def _eval_free(task: Tuple[int, np.ndarray, int, float]) -> Tuple[int, float, Dict[str, float]]:
    index, vector, band_start, budget = task
    z, delta, start_delta = vector[:-2], float(vector[-2]), float(vector[-1])
    try:
        table = nullband.free_band_delta_table(
            _CTX["native_table"], z, delta, band_start=band_start,
            rope_base=_CTX["rope_base"], start_delta=start_delta, name="free_candidate",
        )
    except ValueError:
        return index, 1e6, {}
    stats = _score(table.inv_freq)
    target = int(_CTX["primary_target"])
    over = max(0.0, stats[f"risk@{target}"] - budget)
    return index, stats["dstar"] + 50.0 * over, stats


def differential_evolution(
    pool: Any, *, dim: int, lower: np.ndarray, upper: np.ndarray, band_start: int,
    budget: float, popsize: int, iters: int, seed: int,
    seeds: List[np.ndarray] | None = None,
    diff_weight: float = 0.6, crossover: float = 0.9,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    pop = rng.uniform(lower, upper, size=(popsize, dim))
    pop[0, :-2] = 0.0
    pop[0, -2] = upper[-2]
    pop[0, -1] = 0.0
    injected = 0
    for offset, vector in enumerate(seeds or [], start=1):
        if offset >= popsize:
            break
        pop[offset] = np.clip(np.asarray(vector, dtype=np.float64), lower, upper)
        injected += 1
    # jittered copies of the injected operators, so the search starts in their basin
    slot = injected + 1
    for vector in (seeds or []):
        for _ in range(2):
            if slot >= popsize:
                break
            base = np.clip(np.asarray(vector, dtype=np.float64), lower, upper)
            pop[slot] = np.clip(base + rng.normal(0.0, 0.25, size=dim), lower, upper)
            slot += 1
    fitness = np.empty(popsize)
    stats: List[Dict[str, float]] = [dict() for _ in range(popsize)]
    for i, v, st in pool.map(_eval_free, [(i, pop[i], band_start, budget) for i in range(popsize)], chunksize=1):
        fitness[i], stats[i] = v, st
    history = [{"generation": 0, "best": float(fitness.min())}]
    for generation in range(1, int(iters) + 1):
        trials = np.empty_like(pop)
        for i in range(popsize):
            a, b, c = rng.choice([j for j in range(popsize) if j != i], size=3, replace=False)
            mutant = np.clip(pop[a] + diff_weight * (pop[b] - pop[c]), lower, upper)
            mask = rng.random(dim) < crossover
            if not mask.any():
                mask[rng.integers(dim)] = True
            trials[i] = np.where(mask, mutant, pop[i])
        for i, v, st in pool.map(_eval_free, [(i, trials[i], band_start, budget) for i in range(popsize)], chunksize=1):
            if v < fitness[i]:
                fitness[i], pop[i], stats[i] = v, trials[i], st
        history.append({"generation": generation, "best": float(fitness.min())})
    best = int(np.argmin(fitness))
    return {
        "seeded_operators": int(len(seeds or [])), "seeded_slots": int(slot),
        "band_start": int(band_start), "risk_budget": float(budget), "dim": int(dim),
        "popsize": int(popsize), "iterations": int(iters), "seed": int(seed),
        "best_objective": float(fitness[best]), "best_stats": stats[best],
        "best_vector": pop[best].tolist(), "history": history,
    }


def _verify_worker(table: tables.Table) -> Dict[str, Any]:
    return verify(
        _CTX["native_table"], table, support=_CTX["support"], weight=_CTX["weight"],
        native_length=int(_CTX["native_length"]), targets=[int(x) for x in _CTX["targets"]],
        ranks=[int(r) for r in _CTX["ranks"]], max_iter=int(_CTX["max_iter"]),
    )


def verify(native: tables.Table, candidate: tables.Table, *, support, weight,
           native_length: int, targets: List[int], ranks: List[int], max_iter: int) -> Dict[str, Any]:
    full = transport.transport_residual(native.inv_freq, candidate.inv_freq, support, weight, max_iter=max_iter)
    parts = transport.residual_by_pair(native.inv_freq, candidate.inv_freq, support, weight,
                                       full.query_map, full.key_map)
    gain_full = full.relative_hard_swap - full.relative_repaired
    rank_rows = []
    for rank in ranks:
        res = transport.transport_residual(native.inv_freq, candidate.inv_freq, support, weight,
                                           rank=rank, max_iter=max_iter)
        rank_rows.append({
            "rank_per_head": int(rank), "dstar": res.relative_repaired,
            "gain_fraction": float((full.relative_hard_swap - res.relative_repaired) / gain_full)
            if gain_full > 1e-12 else float("nan"),
        })
    energy = np.asarray(parts["query_pair_energy"], dtype=np.float64)
    row = {
        "name": candidate.name, "sha256": candidate.sha256, "origin": candidate.origin,
        "meta": candidate.meta, "d0": full.relative_hard_swap, "dstar": full.relative_repaired,
        "converged": bool(full.converged), "rank_sweep": rank_rows,
        "top_residual_pairs": [int(i) for i in np.argsort(energy)[::-1][:8]],
    }
    for length in targets:
        risk = nullband.phase_excess_risk(native.inv_freq, candidate.inv_freq,
                                          native_length=native_length, target_length=int(length))
        row[f"risk@{length}"] = risk["mean_turns"]
        row[f"channels_at_risk@{length}"] = risk["channels_at_risk"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--emit-tables")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--rope-base", type=float, default=500000.0)
    ap.add_argument("--native-length", type=int, default=4096)
    ap.add_argument("--targets", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--primary-target", type=int, default=16384)
    ap.add_argument("--scales", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--search-points", type=int, default=512)
    ap.add_argument("--verify-points", type=int, default=2048)
    ap.add_argument("--search-max-iter", type=int, default=25)
    ap.add_argument("--verify-max-iter", type=int, default=60)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--uniqueness-threshold", type=float, default=0.01)
    ap.add_argument("--risk-budgets", type=float, nargs="+", default=[0.0, 0.005, 0.02, 0.05, 0.10])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--de-pop", type=int, default=96)
    ap.add_argument("--de-iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--finalists", type=int, default=10)
    args = ap.parse_args()

    started = time.time()
    guard = _require_no_cuda()
    workers = int(args.workers) or max(1, (os.cpu_count() or 8) - 8)

    frozen = tables.load_manifest_tables(args.manifest)
    native = {t.name: t for t in frozen}["native"]
    pairs = native.inv_freq.size
    support, weight, wmeta = weights.distance_weight("causal", length=args.native_length, max_points=args.verify_points)
    s_support, s_weight, smeta = weights.distance_weight("causal", length=args.native_length, max_points=args.search_points)
    uniq = conditioning.pair_uniqueness(native.inv_freq, support, weight)["uniqueness"]
    below = np.flatnonzero(uniq < float(args.uniqueness_threshold))
    cliff = int(below[0]) if below.size else pairs - 1
    box = nullband.phase_safety_box(native.inv_freq, native_length=args.native_length,
                                    target_length=args.primary_target)
    first_unwrapped = int(box["unwrapped_index"][0]) if box["unwrapped_index"].size else pairs
    native_risk = nullband.phase_excess_risk(native.inv_freq, native.inv_freq,
                                             native_length=args.native_length,
                                             target_length=args.primary_target)["mean_turns"]
    print(f"[stage0] pairs={pairs} cliff={cliff} first_unwrapped={first_unwrapped} "
          f"native_risk@{args.primary_target}={native_risk:.4f} workers={workers}", flush=True)

    # ------------------------- candidate construction ------------------------- #
    named: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
    built: Dict[str, tables.Table] = {}
    skipped: List[Dict[str, str]] = []

    def add(label: str, build: Callable[[], tables.Table], group: str, **tags: Any) -> None:
        try:
            table = build()
        except ValueError as exc:
            skipped.append({"label": label, "error": str(exc)})
            return
        built[table.name] = table
        named.append((table.name, table.inv_freq, {"group": group, **tags}))

    add("native", lambda: native, "reference")
    for t in frozen:
        if t.name != "native":
            add(t.name, lambda t=t: t, "frozen_candidate")
    for scale in args.scales:
        add(f"pi_s{scale:g}", lambda s=scale: tables.position_interpolation(native, s), "published", scale=scale)
        add(f"yarn_s{scale:g}", lambda s=scale: tables.official_yarn(
            native, scale=s, head_dim=args.head_dim, rope_base=args.rope_base,
            original_max_position_embeddings=args.native_length), "published", scale=scale)
        add(f"budgeted_s{scale:g}_p2", lambda s=scale: tables.budgeted_transport(
            native, uniq, scale=s, exponent=2.0), "derived_prior", scale=scale)
        # one-parameter turn-budget family: beta=1 is the exact safety floor,
        # beta=32 with a ramp is official YaRN's schedule shape
        for beta in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0):
            add(f"turnbudget_b{beta:g}_s{scale:g}", lambda s=scale, b=beta: nullband.turn_budget_table(
                native, scale=s, beta=b, native_length=args.native_length,
                rope_base=args.rope_base), "turn_budget", scale=scale, beta=beta, ramp=0.0)
            add(f"turnbudget_b{beta:g}_r{beta - 1:g}_s{scale:g}",
                lambda s=scale, b=beta: nullband.turn_budget_table(
                    native, scale=s, beta=b, native_length=args.native_length,
                    rope_base=args.rope_base, ramp_turns=max(b - 1.0, 0.0)),
                "turn_budget_ramped", scale=scale, beta=beta, ramp=beta - 1.0)
        # band warps, including the paper's EVQ-Cosh quantile shape
        for band_start in sorted({cliff, first_unwrapped, max(0, cliff - 6)}):
            for tau in np.round(np.arange(-4.0, 4.01, 0.5), 3):
                add(f"band_evq_t{tau:g}_j{band_start}_s{scale:g}",
                    lambda s=scale, t=float(tau), j=band_start: nullband.band_warp_table(
                        native, warp="evq_cosh", param=t, scale=s, band_start=j,
                        rope_base=args.rope_base), "band_evq_cosh",
                    scale=scale, param=float(tau), band_start=band_start)
            for p in np.round(np.exp(np.linspace(math.log(0.25), math.log(4.0), 17)), 4):
                add(f"band_pow_{p:g}_j{band_start}_s{scale:g}",
                    lambda s=scale, q=float(p), j=band_start: nullband.band_warp_table(
                        native, warp="power", param=q, scale=s, band_start=j,
                        rope_base=args.rope_base), "band_power",
                    scale=scale, param=float(p), band_start=band_start)

    ctx = {
        "native": native.inv_freq, "native_table": native,
        "support": s_support, "weight": s_weight,
        "max_iter": int(args.search_max_iter), "tol": 1e-10,
        "targets": [int(x) for x in args.targets], "native_length": int(args.native_length),
        "primary_target": int(args.primary_target), "rope_base": float(args.rope_base),
    }

    mp_ctx = mp.get_context("fork")
    with mp_ctx.Pool(processes=workers, initializer=_init_worker, initargs=(ctx,)) as pool:
        t0 = time.time()
        scored = pool.map(_eval_named, named, chunksize=1)
        print(f"[stage1] {len(scored)} named candidates in {time.time() - t0:.1f}s "
              f"({len(skipped)} skipped)", flush=True)

        free_runs = []
        delta_cap = math.log(max(args.scales) * 2.0) / math.log(args.rope_base)
        for band_start in sorted({cliff, 0}):
            dim = pairs - band_start + 1
            lower = np.concatenate([np.full(dim - 2, -8.0), [0.0, 0.0]])
            upper = np.concatenate([np.full(dim - 2, 8.0), [delta_cap, delta_cap]])
            # every named operator that encodes cleanly on this band seeds the search
            seed_vectors: List[np.ndarray] = []
            seed_labels: List[str] = []
            for row in sorted(scored, key=lambda r: r["dstar"]):
                table = built.get(row["name"])
                if table is None or table.name == "native":
                    continue
                try:
                    z, delta, start_delta = nullband.encode_band(
                        native, table, band_start=band_start, rope_base=args.rope_base)
                except ValueError:
                    continue
                if delta > delta_cap or start_delta > delta_cap:
                    continue
                seed_vectors.append(np.concatenate([z, [delta, start_delta]]))
                seed_labels.append(table.name)
                if len(seed_vectors) >= max(1, int(args.de_pop) // 4):
                    break
            for budget in args.risk_budgets:
                t0 = time.time()
                run = differential_evolution(
                    pool, dim=dim, lower=lower, upper=upper, band_start=band_start,
                    budget=float(budget), popsize=int(args.de_pop), iters=int(args.de_iters),
                    seeds=seed_vectors,
                    seed=int(args.seed) + band_start * 1000 + int(budget * 10000))
                run["seed_labels"] = seed_labels
                run["wall_seconds"] = time.time() - t0
                free_runs.append(run)
                st = run["best_stats"]
                print(f"[stage2] j={band_start} rho={budget:.3f} -> D*={st['dstar']:.4f} "
                      f"risk={st[f'risk@{args.primary_target}']:.4f} ({run['wall_seconds']:.0f}s)", flush=True)

    # ------------------------------- finalists -------------------------------- #
    finalists: List[tables.Table] = []
    seen: set = set()

    def keep(table: tables.Table) -> None:
        if table.sha256 not in seen:
            seen.add(table.sha256)
            finalists.append(table)

    keep(native)
    for row in scored:
        if row["meta"]["group"] in {"published", "frozen_candidate", "derived_prior"}:
            keep(built[row["name"]])
    for group in ("turn_budget", "turn_budget_ramped", "band_evq_cosh", "band_power"):
        rows = [r for r in scored if r["meta"]["group"] == group]
        rows.sort(key=lambda r: (r[f"risk@{args.primary_target}"] > 1e-9, r["dstar"]))
        for row in rows[:4]:
            keep(built[row["name"]])
    for run in free_runs:
        vector = np.asarray(run["best_vector"], dtype=np.float64)
        try:
            keep(nullband.free_band_delta_table(
                native, vector[:-2], float(vector[-2]), band_start=run["band_start"],
                rope_base=args.rope_base, start_delta=float(vector[-1]),
                name=f"frontier_j{run['band_start']}_rho{run['risk_budget']:g}"))
        except ValueError as exc:
            skipped.append({"label": f"frontier_j{run['band_start']}_rho{run['risk_budget']:g}",
                            "error": str(exc)})

    print(f"[stage3] verifying {len(finalists)} finalists at {support.size} points", flush=True)
    verify_ctx = {
        "native": native.inv_freq, "native_table": native,
        "support": support, "weight": weight,
        "max_iter": int(args.verify_max_iter), "tol": 1e-12,
        "targets": [int(x) for x in args.targets], "native_length": int(args.native_length),
        "primary_target": int(args.primary_target), "rope_base": float(args.rope_base),
        "ranks": [int(r) for r in args.ranks],
    }
    t0 = time.time()
    with mp_ctx.Pool(processes=min(workers, max(1, len(finalists))),
                     initializer=_init_worker, initargs=(verify_ctx,)) as vpool:
        verified = vpool.map(_verify_worker, finalists, chunksize=1)
    verified.sort(key=lambda r: (r[f"risk@{args.primary_target}"], r["dstar"]))
    print(f"[stage3] done in {time.time() - t0:.1f}s", flush=True)
    for row in verified:
        print(f"  {row['name']:42s} D0={row['d0']:.4f} D*={row['dstar']:.4f} "
              f"risk@16K={row.get('risk@16384', float('nan')):.4f} "
              f"risk@32K={row.get('risk@32768', float('nan')):.4f}", flush=True)

    # strict Pareto frontier on (D*, risk@primary), both minimised
    pts = [(r["name"], r["dstar"], r[f"risk@{args.primary_target}"]) for r in verified]
    frontier = [
        n for n, d, k in pts
        if not any((d2 <= d and k2 <= k and (d2 < d or k2 < k)) for _, d2, k2 in pts)
    ]

    if args.emit_tables:
        out_dir = Path(args.emit_tables)
        out_dir.mkdir(parents=True, exist_ok=True)
        for table in finalists:
            np.save(out_dir / f"{table.name}.npy", np.asarray(table.inv_freq, dtype="<f4"))
        (out_dir / "index.json").write_text(
            json.dumps([t.as_record() for t in finalists], indent=2), encoding="utf-8")

    payload = {
        "method_id": METHOD_ID, "guard": guard,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": time.time() - started,
        "host": {"node": socket.gethostname(), "platform": platform.platform(),
                 "python": sys.version.split()[0], "numpy": np.__version__, "workers": workers},
        "source_sha256": _source_hashes(), "config": vars(args),
        "weight_meta": {"verify": wmeta, "search": smeta},
        "native": native.as_record(),
        "structure": {"pairs": int(pairs), "uniqueness": uniq.tolist(),
                      "uniqueness_cliff_index": cliff, "first_unwrapped_index": first_unwrapped,
                      "native_risk_primary": native_risk},
        "stage1_named": scored, "skipped": skipped,
        "stage2_free": free_runs, "stage3_verified": verified,
        "pareto_frontier": frontier,
    }
    payload["receipt_sha256"] = _canonical_sha256({k: v for k, v in payload.items() if k != "receipt_sha256"})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"PARETO FRONTIER: {frontier}", flush=True)
    print(f"RECEIPT {out} sha256={payload['receipt_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
