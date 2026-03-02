#!/usr/bin/env python3
"""Pre-cache FineWeb-Edu dataset for EVQ experiments.

Usage: python precache_data.py [--cache_dir DIR]
"""
import os
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# Fix hf-mirror pagination
try:
    import huggingface_hub.utils._pagination as _hf_pag
    _orig = _hf_pag._get_next_page
    def _patched(response):
        url = _orig(response)
        if url and "huggingface.co" in url:
            mirror = os.environ.get("HF_ENDPOINT", "").rstrip("/")
            if mirror and mirror != "https://huggingface.co":
                url = url.replace("https://huggingface.co", mirror)
        return url
    _hf_pag._get_next_page = _patched
except Exception:
    pass

import argparse
import time
from pathlib import Path
from transformers import AutoTokenizer

# Import load_data/load_val from run_evq_sweep
import sys
sys.path.insert(0, os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=os.path.expanduser("~/evq_500m_sweep"))
    parser.add_argument("--max_tokens", type=int, default=500_000_000)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--dataset", default="fineweb-edu")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Pre-caching dataset: {args.dataset}")
    print(f"  max_tokens: {args.max_tokens/1e6:.0f}M")
    print(f"  seq_len:    {args.seq_len}")
    print(f"  cache_dir:  {cache_dir}")

    tok = AutoTokenizer.from_pretrained("gpt2")

    from run_evq_sweep import load_data, load_val

    t0 = time.time()
    print("\n=== Training data ===")
    train = load_data(tok, args.max_tokens, args.seq_len, args.dataset, cache_dir=cache_dir)
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s: {train.shape}")

    print("\n=== Validation data ===")
    val = load_val(tok, dataset=args.dataset, cache_dir=cache_dir)
    t2 = time.time()
    print(f"  Done in {t2-t1:.1f}s: {val.shape}")

    print(f"\nTotal time: {t2-t0:.1f}s")
    print("Cache files:")
    for f in sorted(Path(cache_dir).glob("*.pt")):
        print(f"  {f.name}: {f.stat().st_size/1e6:.1f}MB")

if __name__ == "__main__":
    main()
