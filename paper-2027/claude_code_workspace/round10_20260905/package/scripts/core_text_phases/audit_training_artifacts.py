#!/usr/bin/env python3
"""Audit cached training artifacts for token-count and run-output consistency.

This is a read-only forensic helper for external training run directories.  It
does not train or evaluate models.  Given a run/work directory, it checks the
cached train/val tensors, computes nominal versus actually used tokens under the
current training loop, and lists per-run artifacts such as `results.json`,
`model.pt`, and `inv_freq.npy`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_tensor_meta(path: Path) -> dict[str, Any]:
    try:
        import torch

        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exercised by real envs
        return {"path": path.name, "exists": path.exists(), "error": str(exc)}

    shape = list(tensor.shape)
    numel = int(tensor.numel())
    return {
        "path": path.name,
        "exists": True,
        "shape": shape,
        "dtype": str(tensor.dtype),
        "numel": numel,
        "dim": int(tensor.dim()),
    }


def _cache_path(work_dir: Path, prefix: str, dataset: str, tokens: int, seq_len: int | None) -> Path:
    if prefix == "train":
        if seq_len is None:
            raise ValueError("seq_len is required for train cache names")
        return work_dir / f"train_{dataset}_{tokens}_{seq_len}.pt"
    return work_dir / f"val_{dataset}_{tokens}.pt"


def _token_math(train_meta: dict[str, Any], seq_len: int, batch_size: int, nominal_tokens: int) -> dict[str, Any]:
    if not train_meta.get("exists") or "numel" not in train_meta:
        return {"status": "missing_train_tensor"}

    actual_tokens = int(train_meta["numel"])
    if train_meta.get("dim") == 2:
        chunks = int(train_meta["shape"][0])
        row_len = int(train_meta["shape"][1])
    else:
        chunks = actual_tokens // seq_len
        row_len = seq_len

    if row_len != seq_len:
        shape_status = "seq_len_mismatch"
    else:
        shape_status = "ok"

    steps = chunks // batch_size
    used_chunks = steps * batch_size
    used_tokens = used_chunks * seq_len
    dropped_tokens = actual_tokens - used_tokens
    nominal_delta = actual_tokens - nominal_tokens

    return {
        "status": shape_status,
        "nominal_tokens": nominal_tokens,
        "actual_tokens_in_cache": actual_tokens,
        "actual_minus_nominal": nominal_delta,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "chunks": chunks,
        "optimizer_steps_from_cache": steps,
        "used_tokens_by_train_loop": used_tokens,
        "dropped_tokens_from_batch_floor": dropped_tokens,
        "used_fraction_of_cache": round(used_tokens / actual_tokens, 8) if actual_tokens else None,
    }


def _artifact_status(work_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in work_dir.iterdir() if p.is_dir()):
        row = {
            "run_id": run_dir.name,
            "results_json": (run_dir / "results.json").exists(),
            "model_pt": (run_dir / "model.pt").exists(),
            "inv_freq_npy": (run_dir / "inv_freq.npy").exists(),
            "passkey_nll_json": (run_dir / "passkey_nll.json").exists(),
        }
        results_path = run_dir / "results.json"
        if results_path.exists():
            try:
                with open(results_path, encoding="utf-8") as f:
                    result = json.load(f)
                row.update(
                    {
                        "tau": result.get("tau"),
                        "seed": result.get("seed"),
                        "attn_type": result.get("attn_type"),
                        "d_rope": result.get("d_rope"),
                        "kv_lora_rank": result.get("kv_lora_rank"),
                        "ppl_lengths": sorted((result.get("ppl") or {}).keys()),
                    }
                )
            except Exception as exc:
                row["results_error"] = str(exc)
        rows.append(row)
    return rows


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = Path(args.work_dir)
    report: dict[str, Any] = {
        "work_dir_hint": work_dir.name,
        "dataset_cache_label": args.dataset,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "nominal_train_tokens": args.train_tokens,
        "nominal_val_tokens": args.val_tokens,
    }

    train_path = _cache_path(work_dir, "train", args.dataset, args.train_tokens, args.seq_len)
    val_path = _cache_path(work_dir, "val", args.dataset, args.val_tokens, None)
    train_meta = _load_tensor_meta(train_path)
    val_meta = _load_tensor_meta(val_path)
    report["train_cache"] = train_meta
    report["val_cache"] = val_meta
    report["token_math"] = _token_math(train_meta, args.seq_len, args.batch_size, args.train_tokens)
    report["run_artifacts"] = _artifact_status(work_dir) if work_dir.exists() else []
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, help="Training work/run directory to inspect.")
    parser.add_argument("--dataset", default="fineweb-edu", help="Cache-name dataset label.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
