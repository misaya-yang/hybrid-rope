#!/usr/bin/env python3
"""Pre-download 2B tokens for L=2048 and L=4096 in parallel with running experiment."""
import os, sys, time
from pathlib import Path

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import load_data, load_val
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
cache_dir = "/root/autodl-tmp/evq_phase9/data"

print("=" * 60)
print("  Pre-downloading 2B tokens (L=2048)")
print("=" * 60)
t0 = time.time()
load_data(tok, 2_000_000_000, 2048, "fineweb-edu", cache_dir=cache_dir)
print(f"  L=2048 done in {(time.time()-t0)/60:.1f} min")

print("=" * 60)
print("  Pre-downloading 2B tokens (L=4096)")
print("=" * 60)
t1 = time.time()
load_data(tok, 2_000_000_000, 4096, "fineweb-edu", cache_dir=cache_dir)
print(f"  L=4096 done in {(time.time()-t1)/60:.1f} min")
print(f"  Total: {(time.time()-t0)/60:.1f} min")
