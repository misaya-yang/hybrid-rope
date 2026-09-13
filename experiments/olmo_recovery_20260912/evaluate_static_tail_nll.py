#!/usr/bin/env python3
"""Evaluate one frozen RoPE table on paired tail-token natural-text NLL."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_table(path: Path) -> dict:
    payload = json.loads(path.read_text())
    table = payload.get("table", payload)
    values = np.asarray(table.get("values_float32"), dtype=np.float32)
    gain = float(table.get("gain"))
    if (
        values.shape != (64,)
        or not np.isfinite(values).all()
        or np.any(values[:-1] <= values[1:])
        or not math.isfinite(gain)
        or gain <= 0.0
    ):
        raise ValueError("invalid frozen RoPE table")
    return {
        "values_float32": values.tolist(),
        "gain": gain,
        "construction": table.get("construction", {}),
    }


def native_table_for_config(config) -> dict:
    from .prepare_llama_minimal_band_screen import native_for_config

    values, base, head_dim = native_for_config(config)
    return {
        "values_float32": values.tolist(),
        "gain": 1.0,
        "construction": {
            "method": "identity",
            "base": base,
            "head_dim": head_dim,
        },
    }


def summarize(rows: list[dict], lengths: list[int], documents: int) -> dict:
    result = {}
    for length in lengths:
        selected = [row for row in rows if row["length"] == length]
        if len(selected) != documents:
            raise ValueError(f"incomplete NLL rows at length {length}")
        mean_nll = float(np.mean([row["nll"] for row in selected]))
        result[str(length)] = {
            "documents": documents,
            "mean_tail_nll": mean_nll,
            "tail_ppl": math.exp(mean_nll),
            "per_document_nll": [row["nll"] for row in selected],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    table_group = parser.add_mutually_exclusive_group(required=True)
    table_group.add_argument("--table-json", type=Path)
    table_group.add_argument("--native", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--nll-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, action="append", required=True)
    parser.add_argument("--documents", type=int, default=16)
    parser.add_argument("--tail", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lengths = sorted(set(args.length))
    if len(lengths) != len(args.length) or min(lengths) <= args.tail or args.documents <= 0:
        raise ValueError("lengths must be unique and exceed a positive tail/document count")
    if args.out.exists():
        raise ValueError("output already exists")
    if args.native:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        table = native_table_for_config(config)
    else:
        table = load_table(args.table_json)
    paths = sorted(args.nll_dir.glob("doc_*.npy"))[: args.documents]
    if len(paths) != args.documents:
        raise ValueError("requested natural-text documents are unavailable")
    arrays = [np.load(path, allow_pickle=False) for path in paths]
    if any(array.ndim != 1 or len(array) < max(lengths) + 1 for array in arrays):
        raise ValueError("a natural-text document is shorter than the requested prefix")

    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM
    from scripts.experiments.cross_audit.tables import install_static

    args.out.mkdir(parents=True)
    contract = {
        "status": "FROZEN",
        "label": args.label,
        "model": str(args.model),
        "table_json": str(args.table_json) if args.table_json else None,
        "native_table": args.native,
        "documents": [path.name for path in paths],
        "lengths": lengths,
        "tail": args.tail,
        "metric": "per-document mean tail-token NLL; documents are paired across arms",
    }
    write_json(args.out / "contract.json", contract)
    write_json(args.out / "status.json", {"status": "RUNNING"})
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
    ).eval()
    model.requires_grad_(False)
    install_static(
        model,
        np.asarray(table["values_float32"], dtype=np.float32),
        table["gain"],
    )
    actual = model.model.rotary_emb.inv_freq.detach().cpu().float().numpy()
    if not np.array_equal(actual, np.asarray(table["values_float32"], dtype=np.float32)):
        raise RuntimeError("installed table differs from frozen receipt")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    rows = []
    row_path = args.out / "rows.jsonl"
    with row_path.open("x") as stream, torch.inference_mode():
        for length in lengths:
            for path, tokens in zip(paths, arrays):
                ids = torch.tensor(tokens[:length].astype(np.int64), device="cuda")[None]
                targets = torch.tensor(
                    tokens[length - args.tail + 1 : length + 1].astype(np.int64),
                    device="cuda",
                )
                logits = model(ids, use_cache=False, logits_to_keep=args.tail).logits[0].float()
                losses = functional.cross_entropy(logits, targets, reduction="none")
                row = {
                    "label": args.label,
                    "document": path.name,
                    "length": length,
                    "tail": args.tail,
                    "nll": float(losses.mean()),
                    "target_ids": targets.tolist(),
                    "token_nll": losses.tolist(),
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
                write_json(
                    args.out / "live.json",
                    {"phase": "nll", "completed": len(rows), "total": len(paths) * len(lengths)},
                )
                del ids, targets, logits, losses
    summary = {
        "status": "COMPLETE",
        "label": args.label,
        "metric": f"paired natural-text tail-{args.tail} next-token NLL",
        "by_length": summarize(rows, lengths, len(paths)),
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "table": table,
        "scope": "natural-text NLL only; not a retrieval or generation score",
    }
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "status.json", {"status": "COMPLETE", "rows": len(rows)})
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
