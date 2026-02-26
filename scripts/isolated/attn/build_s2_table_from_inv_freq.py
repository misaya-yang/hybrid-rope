#!/usr/bin/env python3
"""
Build collision-energy S^2(delta) table from an inv_freq tensor.

Definition used:
    S(delta)  = mean_i cos(inv_freq[i] * delta)
    S2(delta) = S(delta)^2

Output:
    .pt file containing {"s2_by_delta": tensor, "meta": {...}}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_inv_freq(path: Path) -> torch.Tensor:
    raw: Any = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("inv_freq", "custom_inv_freq", "tensor", "data"):
            if key in raw:
                raw = raw[key]
                break
    inv = torch.as_tensor(raw, dtype=torch.float32).view(-1)
    if inv.numel() <= 1:
        raise RuntimeError(f"Invalid inv_freq payload: {path}")
    return inv


def build_s2(inv_freq: torch.Tensor, max_delta: int) -> torch.Tensor:
    d = torch.arange(int(max_delta), dtype=torch.float32).view(-1, 1)
    s = torch.cos(d * inv_freq.view(1, -1)).mean(dim=1)
    return (s * s).clamp_min(1e-8)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build s2_by_delta tensor from inv_freq")
    ap.add_argument("--inv_freq_path", type=Path, required=True)
    ap.add_argument("--max_delta", type=int, required=True)
    ap.add_argument("--out_path", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    inv = load_inv_freq(args.inv_freq_path)
    s2 = build_s2(inv, int(args.max_delta))

    payload = {
        "s2_by_delta": s2,
        "meta": {
            "source": "build_s2_table_from_inv_freq.py",
            "inv_freq_path": str(args.inv_freq_path.resolve()),
            "inv_freq_sha256": sha256_file(args.inv_freq_path.resolve()),
            "max_delta": int(args.max_delta),
            "formula": "S2(delta) = (mean_i cos(inv_freq[i] * delta))^2",
        },
    }
    torch.save(payload, args.out_path)
    print(
        json.dumps(
            {
                "out_path": str(args.out_path.resolve()),
                "sha256": sha256_file(args.out_path.resolve()),
                "max_delta": int(args.max_delta),
                "numel": int(s2.numel()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

