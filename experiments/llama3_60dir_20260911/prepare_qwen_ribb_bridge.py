"""Prepare one matched-gain Qwen RIBB beta=1.5 interaction cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.run_out.exists():
        raise FileExistsError("prepared or run output already exists")

    manifest = json.loads((args.source / "manifest.json").read_text())
    for name, expected in manifest["prepared_files"].items():
        if sha_file(args.source / name) != expected:
            raise ValueError(f"source prepared file drift: {name}")
    if (float(manifest["static_scale"]), int(manifest["native_length"]),
            float(manifest["base"]), int(manifest["head_dim"])) != (
            2.0, 32768, 1_000_000.0, 128):
        raise ValueError("unexpected Qwen scale-2 geometry")

    tables = json.loads((args.source / "tables.json").read_text())
    native = np.asarray(tables["Native"]["values_float32"], dtype=np.float32)
    low, high = 23, 40
    n = high - low
    kk = np.arange(1, n + 1, dtype=np.float64)
    weights = kk * np.sqrt(n + 1.0 - kk)
    cumulative = np.concatenate([[0.0], np.cumsum(weights / weights.sum())])
    exponent = np.zeros(native.size, dtype=np.float64)
    exponent[low:high + 1] = cumulative
    exponent[high + 1:] = 1.0
    values = (native.astype(np.float64) * np.power(2.0, -exponent)).astype(np.float32)
    mr = np.asarray(tables["MrPro"]["values_float32"], dtype=np.float32)
    bm = np.asarray(tables["MrProBM"]["values_float32"], dtype=np.float32)
    interior = slice(low + 1, high)
    if not (np.all(values[interior] < mr[interior])
            and np.all(values[interior] > bm[interior])):
        raise ValueError("RIBB bridge is not pointwise between MR and BM")
    table_sha = hashlib.sha256(
        np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()
    candidate = "RIBB_A2_B1P5"
    tables[candidate] = {
        "values_float32": values.tolist(),
        "tensor_sha256": table_sha,
        "gain": float(tables["MrPro"]["gain"]),
    }

    args.out.mkdir(parents=True)
    shared = ("screen.jsonl", "qualification.jsonl", "prompts.jsonl",
              "generation_config.json")
    for name in shared:
        os.link(args.source / name, args.out / name)
    write(args.out / "tables.json", tables)
    write(args.out / "queue.json", {
        "max_candidates": 1,
        "ordered_candidates": [{
            "id": candidate,
            "eligible": True,
            "review_status": "REVIEWED_FOR_GPU",
            "definition": (
                "epsilon_q proportional to q*(N+1-q)^0.5; "
                "alpha=2,beta=1.5"),
            "hypothesis": (
                "The exact MR-to-BM bridge has a checkpoint-dependent range "
                "tradeoff on Qwen under matched scale-2 gain."),
            "failure_rule": (
                "Report both 32K and 64K against reused MR/BM/UNI; this cell "
                "is development evidence and cannot establish transfer."),
        }],
    })
    manifest = dict(manifest)
    construction = dict(manifest.get("construction", {}))
    construction["ribb_a2_b1p5"] = {
        "method": "ribb_beta_increment",
        "alpha": 2.0, "beta": 1.5, "low": low, "high": high,
        "N": n, "scale": 2.0, "exponents": exponent.tolist(),
        "pointwise_between": ["MrPro", "MrProBM"],
    }
    manifest.update(
        status="PREPARED_RIBB_MODEL_INTERACTION_CELL",
        construction=construction,
        reference_arm="MrPro",
        complete_candidate_queue=True,
        source_prepared_manifest_sha256=sha_file(args.source / "manifest.json"),
        evidence_role="development model-by-profile interaction; not confirmation",
    )
    manifest["prepared_files"] = {
        name: sha_file(args.out / name)
        for name in (*shared, "tables.json", "queue.json")
    }
    write(args.out / "manifest.json", manifest)

    args.run_out.mkdir(parents=True)
    for name in ("MrPro.json", "MrPro.jsonl"):
        os.link(args.source_run / name, args.run_out / name)
    print(json.dumps({
        "status": "READY", "candidate": candidate,
        "tensor_sha256": table_sha, "rows": sum(
            1 for _ in (args.out / "screen.jsonl").open()),
        "sum_m": float(exponent.sum()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
