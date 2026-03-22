#!/usr/bin/env python3
"""Combine 1b_diverse_4k (v1) + 1b_diverse_4k_v2 into a 2B combined dataset.

This runs on CPU (no-GPU mode). Memory: ~8GB RAM needed.
Output: /root/autodl-tmp/data/2b_diverse_4k/train_combined_2000000000_4096.pt (~7.6GB, int32)

Purpose: 125M MLA training for 1.5B tokens to test inflection point.
"""
import torch
from pathlib import Path
import time

V1 = Path("/root/autodl-tmp/data/1b_diverse_4k/train_fineweb-edu_1000000000_4096.pt")
V2 = Path("/root/autodl-tmp/data/1b_diverse_4k_v2/train_fineweb-edu_1000000000_4096.pt")
VAL_V2 = Path("/root/autodl-tmp/data/1b_diverse_4k_v2/val_fineweb-edu_5000000.pt")
OUT_DIR = Path("/root/autodl-tmp/data/2b_diverse_4k")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading v1 data (7.5GB int64)...")
t0 = time.time()
d1 = torch.load(V1, weights_only=True)
print(f"  v1: shape={d1.shape}, dtype={d1.dtype}, {time.time()-t0:.1f}s")

print("Loading v2 data (3.8GB int32)...")
t0 = time.time()
d2 = torch.load(V2, weights_only=True)
print(f"  v2: shape={d2.shape}, dtype={d2.dtype}, {time.time()-t0:.1f}s")

# Convert to int32 and concat
print("Converting and concatenating...")
t0 = time.time()
d1_32 = d1.to(torch.int32)
del d1
combined = torch.cat([d1_32, d2], dim=0)
del d1_32, d2
n_tokens = combined.numel()
print(f"  Combined: shape={combined.shape}, tokens={n_tokens/1e9:.3f}B, {time.time()-t0:.1f}s")

out_path = OUT_DIR / f"train_combined_{n_tokens}_{combined.shape[1]}.pt"
print(f"Saving to {out_path}...")
t0 = time.time()
torch.save(combined, out_path)
print(f"  Saved {out_path.stat().st_size/1e9:.1f}GB in {time.time()-t0:.1f}s")

# Canonical symlink name for run script compatibility
link = OUT_DIR / "train_fineweb-edu_2000000000_4096.pt"
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to(out_path.name)
print(f"  Symlink: {link.name} -> {out_path.name}")

# Reuse v2 val file
val_link = OUT_DIR / "val_fineweb-edu_5000000.pt"
if val_link.exists() or val_link.is_symlink():
    val_link.unlink()
val_link.symlink_to(VAL_V2)
print(f"  Val symlink: {val_link.name} -> {VAL_V2}")

print(f"\nDONE. {n_tokens/1e9:.3f}B tokens at seq_len=4096 ready in {OUT_DIR}")
print("Use --train_tokens 1500000000 to train on first 1.5B of this 2B dataset.")
