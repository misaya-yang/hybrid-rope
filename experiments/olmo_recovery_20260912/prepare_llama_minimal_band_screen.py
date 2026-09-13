#!/usr/bin/env python3
"""Prepare a minimal Llama S=2 band screen: two PG19 docs and static tables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.rope_fast_5090_20260912.e3_tables import tables as e3_tables
from scripts.experiments.cross_audit.tables import transform
from scripts.lib.rope.boundary_matched import boundary_matched_inv_freq

from .transfer_range_profile import transfer_profile


def parse_band(value: str) -> tuple[int, int]:
    try:
        low, high = (int(item) for item in value.split(":"))
    except Exception as exc:
        raise argparse.ArgumentTypeError("band must be LOW:HIGH") from exc
    if not 0 <= low < high < 64:
        raise argparse.ArgumentTypeError("band must satisfy 0 <= LOW < HIGH < 64")
    return low, high


def native_for_config(config) -> tuple[np.ndarray, float, int]:
    """Read the standard RoPE geometry without imposing a model-name whitelist."""
    from scripts.experiments.cross_audit.tables import native_table

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    base = getattr(config, "rope_theta", None) or getattr(config, "rope_parameters", {}).get("rope_theta")
    if head_dim != 128 or base is None:
        raise ValueError("minimal band screen requires a standard 128-dimension RoPE geometry")
    return native_table(head_dim, float(base)).astype(np.float32), float(base), int(head_dim)


def build_tables(native: np.ndarray, *, native_length: int, base: float, scale: float,
                 bands: list[tuple[int, int]]) -> dict[str, dict]:
    import torch

    native = np.asarray(native, dtype=np.float32)
    bm, bm_gain, bm_meta = boundary_matched_inv_freq(
        torch.from_numpy(native.copy()), base=base, reference_length=native_length, scale=scale
    )
    mr, mr_gain, mr_meta = transform(
        native, dim=128, base=base, reference_length=native_length, scale=scale, method="mrpro"
    )
    result = {
        f"BM_s{scale:g}": {"values_float32": bm.numpy().tolist(), "gain": bm_gain, "construction": bm_meta},
        f"MrPro_s{scale:g}": {"values_float32": mr.tolist(), "gain": mr_gain, "construction": mr_meta},
    }
    record = e3_tables()["C42V24"]
    exponents = record["construction"]["cumulative_exponents_float64"]
    source = {"allocation": {
        "exponents": exponents,
        "scale": float(record["construction"]["scale"]),
        "gain": float(record["gain"]),
        "low": next(i for i, value in enumerate(exponents) if value > 0.0) - 1,
        "high": next(i for i, value in enumerate(exponents) if value >= 1.0),
    }}
    for low, high in bands:
        label = f"C42Band{low}_{high}_s{scale:g}"
        result[label] = transfer_profile(
            source, native, target_scale=scale, target_low=low, target_high=high,
            gain_policy="target_yarn",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lm-text", type=Path, action="append", required=True)
    parser.add_argument("--native-length", type=int, default=8192)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--band", type=parse_band, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or len(args.lm_text) < 2:
        raise ValueError("new output and at least two fixed LM texts are required")
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    native, base, _ = native_for_config(config)
    horizon = int(args.native_length * args.scale)
    evaluation_lengths = []
    length = args.native_length
    while length < horizon:
        evaluation_lengths.append(length)
        length *= 2
    evaluation_lengths.append(horizon)
    docs = []
    for path in args.lm_text:
        ids = tokenizer.encode(path.read_text(errors="ignore"), add_special_tokens=False)
        if len(ids) < horizon + 1:
            raise ValueError(f"LM text is shorter than {horizon + 1} tokens: {path}")
        docs.append(np.asarray(ids[: horizon + 1], dtype=np.int64))
    tables = build_tables(
        native, native_length=args.native_length,
        base=base, scale=args.scale, bands=args.band,
    )
    args.out.mkdir(parents=True)
    np.save(args.out / "lm.npy", np.stack(docs))
    table_dir = args.out / "tables"
    table_dir.mkdir()
    for label, table in tables.items():
        (table_dir / f"{label}.json").write_text(json.dumps({
            "status": "FROZEN", "label": label, "table": table,
        }, indent=2, sort_keys=True) + "\n")
    manifest = {
        "status": "READY",
        "model": str(args.model),
        "native_length": args.native_length,
        "scale": args.scale,
        "horizon": horizon,
        "evaluation_lengths": evaluation_lengths,
        "lm_evaluation": {"dev": str((args.out / "lm.npy").resolve())},
        "evaluation_panels": {"dev": []},
        "lm_sources": [str(path) for path in args.lm_text],
        "bands": [list(band) for band in args.band],
        "arms": list(tables),
        "purpose": "minimal S=2 PPL + passkey/NIAH band-position screen",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "READY", "arms": list(tables), "lm_documents": len(docs)}, sort_keys=True))


if __name__ == "__main__":
    main()
