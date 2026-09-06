#!/usr/bin/env python3
"""Export one rank-bounded static RoPE transport map with a receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import tables, transport, weights


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--support-points", type=int, default=2048)
    parser.add_argument("--max-iter", type=int, default=40)
    args = parser.parse_args()

    if args.rank <= 0:
        raise ValueError("rank must be positive")
    frozen = {table.name: table for table in tables.load_manifest_tables(args.manifest)}
    native = frozen["native"].inv_freq
    target_raw = np.load(args.target_table, allow_pickle=False)
    if target_raw.dtype != np.float32 or target_raw.shape != native.shape:
        raise ValueError("target table must be float32 and match Native shape")
    target = np.asarray(target_raw, dtype=np.float64)
    support, weight, weight_meta = weights.distance_weight(
        "causal",
        length=args.length,
        max_points=args.support_points,
    )

    started = time.time()
    result = transport.transport_residual(
        native,
        target,
        support,
        weight,
        rank=args.rank,
        max_iter=args.max_iter,
    )
    eye = np.eye(result.query_map.shape[0], dtype=np.float64)
    query_delta = result.query_map - eye
    key_delta = result.key_map - eye
    query_rank = int(np.linalg.matrix_rank(query_delta, tol=1e-7))
    key_rank = int(np.linalg.matrix_rank(key_delta, tol=1e-7))
    if query_rank > args.rank or key_rank > args.rank:
        raise RuntimeError("realized transport map exceeds registered rank")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".incomplete")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            query_map=result.query_map.astype(np.float32),
            key_map=result.key_map.astype(np.float32),
            query_delta=query_delta.astype(np.float32),
            key_delta=key_delta.astype(np.float32),
        )
    temporary.replace(output)

    receipt = {
        "status": "ROPE_TRANSPORT_MAP_EXPORTED",
        "operator": "fixed position-independent headwise Q/K post-map",
        "coordinate_order": "adjacent rotary pairs [x0,y0,x1,y1,...]",
        "rank_bound_per_head": int(args.rank),
        "realized_query_delta_rank_tol_1e-7": query_rank,
        "realized_key_delta_rank_tol_1e-7": key_rank,
        "source_table_float32_sha256": tables.float32_sha256(native),
        "target_table_file_sha256": _sha256(args.target_table.resolve()),
        "target_table_float32_sha256": tables.float32_sha256(target),
        "map_file_sha256": _sha256(output),
        "support": weight_meta,
        "transport": result.summary(),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(output.with_suffix(".json"), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
