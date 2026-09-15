#!/usr/bin/env python3
"""Build the matched Llama Native-8K PPL comparison from completed raw rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def point(mapping: dict[int, dict]) -> dict:
    loss = sum(float(row["whole_loss_sum"]) for row in mapping.values())
    tokens = sum(int(row["whole_target_count"]) for row in mapping.values())
    nll = loss / tokens
    return {"documents": len(mapping), "target_tokens": tokens, "nll": nll, "ppl": math.exp(nll)}


def paired_ppl(candidate, native, *, draws=20_000, seed=20260918):
    documents = sorted(native)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(len(documents), size=(draws, len(documents)))
    candidate_loss = np.asarray([candidate[index]["whole_loss_sum"] for index in documents], dtype=np.float64)
    native_loss = np.asarray([native[index]["whole_loss_sum"] for index in documents], dtype=np.float64)
    counts = np.asarray([native[index]["whole_target_count"] for index in documents], dtype=np.float64)
    candidate_draws = np.exp(candidate_loss[sampled].sum(axis=1) / counts[sampled].sum(axis=1))
    native_draws = np.exp(native_loss[sampled].sum(axis=1) / counts[sampled].sum(axis=1))
    delta = candidate_draws - native_draws
    return {
        "delta_ppl": point(candidate)["ppl"] - point(native)["ppl"],
        "paired_document_bootstrap_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        "probability_delta_lt_zero": float(np.mean(delta < 0)),
        "draws": draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="ARM=lm_rows.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {}
    for value in args.run:
        arm, raw = value.split("=", 1)
        paths[arm] = Path(raw)
    if set(paths) != {"native", "tailspline", "mrpro"}:
        raise ValueError("Native PPL summary requires native, tailspline and mrpro")
    mappings = {}
    for arm, path in paths.items():
        selected = [row for row in rows(path) if int(row["length"]) == 8192]
        mapping = {int(row["document"]): row for row in selected}
        if len(selected) != 46 or len(mapping) != 46:
            raise ValueError(f"{arm} does not contain 46 unique Native-8K documents")
        mappings[arm] = mapping
    if any(set(mapping) != set(mappings["native"]) for mapping in mappings.values()):
        raise ValueError("Native-8K document pairing drift")
    for document in mappings["native"]:
        identity = {
            (mappings[arm][document]["whole_target_count"], mappings[arm][document]["tail128_target_count"])
            for arm in mappings
        }
        if len(identity) != 1:
            raise ValueError(f"Native-8K target identity drift: {document}")
    result = {
        "status": "LLAMA_NATIVE_8K_PPL_COMPARISON_COMPLETE_V1",
        "length": 8192,
        "metric": "exp(total whole-prefix loss / total scored tokens)",
        "arms": {arm: point(mapping) for arm, mapping in mappings.items()},
        "contrasts_vs_native": {
            arm: paired_ppl(mapping, mappings["native"], seed=20260918 + offset)
            for offset, (arm, mapping) in enumerate((
                ("tailspline", mappings["tailspline"]), ("mrpro", mappings["mrpro"]),
            ))
        },
        "raw_sha256": {arm: sha256(path) for arm, path in paths.items()},
        "claim_boundary": "Matched Native-length language-model health; not a generated-task retention result.",
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
