#!/usr/bin/env python3
"""
Download model and data via ModelScope (国内源).
Must run on the GPU server (mainland China, HF blocked).

Usage:
    python download_model_data.py                    # download all
    python download_model_data.py --model_only       # model only
    python download_model_data.py --data_only        # data only
    python download_model_data.py --verify_only      # just verify
"""

import argparse
import glob
import json
import os
import sys
import time

BASE_DIR = "/root/autodl-tmp"
MODEL_DIR = f"{BASE_DIR}/models/Meta-Llama-3-8B-Instruct"
DATA_DIR = f"{BASE_DIR}/data/longalign_10k"
WIKITEXT_DIR = f"{BASE_DIR}/data/wikitext2"
LONGBENCH_DIR = f"{BASE_DIR}/datasets/longbench"

# Cached arrow data from previous ModelScope downloads
MS_LONGALPACA_CACHE = "/root/.cache/modelscope/hub/datasets/AI-ModelScope___long_alpaca-12k"


def download_model():
    """Download LLaMA-3-8B-Instruct via ModelScope."""
    # Check completeness: need at least one safetensors file
    safetensors = glob.glob(os.path.join(MODEL_DIR, "model*.safetensors"))
    config_ok = os.path.exists(os.path.join(MODEL_DIR, "config.json"))

    if config_ok and len(safetensors) >= 4:
        total_size = sum(os.path.getsize(f) for f in safetensors)
        if total_size > 10_000_000_000:  # >10GB = complete
            print(f"[MODEL] Complete ({total_size/1e9:.1f}GB, {len(safetensors)} shards)")
            return MODEL_DIR
        print(f"[MODEL] Incomplete ({total_size/1e9:.1f}GB), re-downloading ...")
    elif config_ok:
        print(f"[MODEL] Partial download (config exists, {len(safetensors)} shards), resuming ...")
    else:
        print(f"[MODEL] Not found, starting download ...")

    print(f"[MODEL] Downloading LLM-Research/Meta-Llama-3-8B-Instruct ...")
    print(f"[MODEL] Target: {MODEL_DIR}")
    t0 = time.time()

    from modelscope import snapshot_download
    path = snapshot_download(
        "LLM-Research/Meta-Llama-3-8B-Instruct",
        local_dir=MODEL_DIR,
        ignore_file_pattern=["original/*", "*.pth"],
    )

    # Verify
    safetensors = glob.glob(os.path.join(MODEL_DIR, "model*.safetensors"))
    total_size = sum(os.path.getsize(f) for f in safetensors)
    print(f"[MODEL] Done in {(time.time()-t0)/60:.1f} min → {path}")
    print(f"[MODEL] {len(safetensors)} shards, {total_size/1e9:.1f}GB total")
    return path


def download_longalign():
    """Download long-context training data."""
    output_file = os.path.join(DATA_DIR, "longalign_10k.jsonl")
    if os.path.exists(output_file):
        n_lines = sum(1 for _ in open(output_file))
        if n_lines > 5000:
            print(f"[DATA] Already exists ({n_lines} samples), skipping")
            return output_file

    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()

    # Strategy 1: Use locally cached ModelScope arrow data (fastest)
    if os.path.exists(MS_LONGALPACA_CACHE):
        print("[DATA] Found cached LongAlpaca arrow data, converting ...")
        try:
            return _convert_cached_arrow(output_file, t0)
        except Exception as e:
            print(f"[DATA] Arrow conversion failed: {e}")

    # Strategy 2: Try ModelScope API
    print("[DATA] Trying ModelScope API ...")
    for repo_id in ["AI-ModelScope/LongAlign-10k", "THUDM/LongAlign-10k"]:
        try:
            from modelscope.msdatasets import MsDataset
            ds = MsDataset.load(repo_id, split="train")
            print(f"[DATA] Loaded {len(ds)} samples from {repo_id}")
            return _save_dataset(ds, output_file, t0)
        except Exception as e:
            print(f"[DATA] {repo_id} failed: {e}")

    # Strategy 3: HF cache (wikitext was cached, maybe LongAlpaca too)
    print("[DATA] Trying HF cache ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Yukang/LongAlpaca-12k", split="train")
        print(f"[DATA] Loaded from HF cache: {len(ds)} samples")
        return _save_dataset(ds, output_file, t0)
    except Exception as e:
        print(f"[DATA] HF cache failed: {e}")

    print("[DATA] All sources failed!")
    sys.exit(1)


def _convert_cached_arrow(output_file, t0):
    """Convert cached ModelScope arrow data to JSONL."""
    from datasets import load_dataset

    # Find the cache directory containing dataset_info.json
    cache_dirs = glob.glob(os.path.join(MS_LONGALPACA_CACHE, "**/master"), recursive=True)
    if not cache_dirs:
        raise FileNotFoundError("No cache dir found")

    cache_dir = cache_dirs[0]
    print(f"[DATA] Loading from HF datasets cache: {cache_dir}")

    # Use datasets library's own cache loading
    ds = load_dataset(cache_dir, split="train")
    print(f"[DATA] Loaded {len(ds)} samples from cache")
    return _save_dataset(ds, output_file, t0)


def _save_dataset(ds, output_file, t0):
    """Save dataset to JSONL with messages format."""
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in ds:
            item = dict(item)
            # Format 1: messages array (LongAlign style)
            if "messages" in item and item["messages"]:
                record = {"messages": item["messages"]}
            # Format 2: instruction/input/output (LongAlpaca style)
            elif "instruction" in item:
                user_text = item.get("instruction", "")
                inp = item.get("input", "")
                if inp:
                    user_text = f"{user_text}\n\n{inp}"
                record = {"messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": item.get("output", "")},
                ]}
            else:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"[DATA] Saved {count} samples -> {output_file} ({(time.time()-t0)/60:.1f} min)")
    return output_file


def download_wikitext():
    """Pre-download WikiText-2 for PPL evaluation."""
    output_file = os.path.join(WIKITEXT_DIR, "wikitext2_test.txt")
    if os.path.exists(output_file) and os.path.getsize(output_file) > 100000:
        print(f"[WIKI] Already exists ({os.path.getsize(output_file)//1024}KB), skipping")
        return output_file

    os.makedirs(WIKITEXT_DIR, exist_ok=True)
    print("[WIKI] Extracting WikiText-2 ...")

    # Try wikitext-2 first, then wikitext-103 (which is cached on this server)
    from datasets import load_dataset
    for config_name in ["wikitext-2-raw-v1", "wikitext-103-raw-v1"]:
        try:
            ds = load_dataset("wikitext", config_name, split="test",
                             trust_remote_code=True)
            text = "\n".join(ds["text"])
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[WIKI] Saved {config_name} ({len(text)//1024}KB) -> {output_file}")
            return output_file
        except Exception as e:
            print(f"[WIKI] {config_name} failed: {e}")

    # Try ModelScope
    try:
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load("modelscope/wikitext", subset_name="wikitext-2-raw-v1", split="test")
        text = "\n".join(item["text"] for item in ds)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[WIKI] Saved from ModelScope ({len(text)//1024}KB) -> {output_file}")
        return output_file
    except Exception as e:
        print(f"[WIKI] Warning: WikiText download failed: {e}")
        print("[WIKI] PPL eval will try HF cache at runtime")
        return None


def verify_longbench():
    """Verify LongBench is accessible."""
    print("[LBENCH] Checking LongBench ...")

    if os.path.exists(LONGBENCH_DIR):
        contents = os.listdir(LONGBENCH_DIR)
        if contents:
            print(f"[LBENCH] Local dir exists: {contents[:5]}")
            return True

    try:
        from datasets import load_dataset
        ds = load_dataset("THUDM/LongBench", "qasper", split="test[:2]",
                         trust_remote_code=True)
        print(f"[LBENCH] HF cache works ({len(ds)} samples)")
        return True
    except Exception as e:
        print(f"[LBENCH] Warning: LongBench not cached: {e}")
        return False


def verify_all():
    """Verify everything is ready."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    checks = []

    # Model — check safetensors, not just config
    safetensors = glob.glob(os.path.join(MODEL_DIR, "model*.safetensors"))
    config_ok = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
    total_size = sum(os.path.getsize(f) for f in safetensors) if safetensors else 0
    model_ok = config_ok and len(safetensors) >= 4 and total_size > 10_000_000_000
    print(f"  {'OK' if model_ok else 'FAIL'} Model: {MODEL_DIR} "
          f"({len(safetensors)} shards, {total_size/1e9:.1f}GB)")
    checks.append(("model", model_ok))
    if config_ok:
        with open(os.path.join(MODEL_DIR, "config.json")) as f:
            cfg = json.load(f)
        print(f"      arch={cfg.get('architectures', ['?'])[0]}, "
              f"hidden={cfg.get('hidden_size', '?')}, "
              f"heads={cfg.get('num_attention_heads', '?')}")

    # Training data
    data_file = os.path.join(DATA_DIR, "longalign_10k.jsonl")
    ok = os.path.exists(data_file)
    if ok:
        n = sum(1 for _ in open(data_file))
        ok = n > 5000
        print(f"  {'OK' if ok else 'FAIL'} Training data: {n} samples")
    else:
        print(f"  FAIL Training data: not found")
    checks.append(("train_data", ok))

    # WikiText
    wiki_file = os.path.join(WIKITEXT_DIR, "wikitext2_test.txt")
    ok = os.path.exists(wiki_file) and os.path.getsize(wiki_file) > 100000
    print(f"  {'OK' if ok else 'WARN'} WikiText-2: "
          f"{os.path.getsize(wiki_file)//1024 if os.path.exists(wiki_file) else 0}KB")
    checks.append(("wikitext", ok))

    # LongBench
    ok = verify_longbench()
    checks.append(("longbench", ok))

    # Summary
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    all_pass = n_pass == n_total
    print(f"\n  Result: {n_pass}/{n_total} passed {'(READY)' if all_pass else '(NOT READY)'}")

    # Save paths config
    paths = {
        "model_dir": MODEL_DIR,
        "train_data": os.path.join(DATA_DIR, "longalign_10k.jsonl"),
        "wikitext_path": os.path.join(WIKITEXT_DIR, "wikitext2_test.txt"),
        "longbench_available": checks[3][1],
    }
    paths_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_paths.json")
    with open(paths_file, "w") as f:
        json.dump(paths, f, indent=2)
    print(f"  Paths config -> {paths_file}")

    return all_pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_only", action="store_true")
    p.add_argument("--data_only", action="store_true")
    p.add_argument("--verify_only", action="store_true")
    args = p.parse_args()

    if args.verify_only:
        ok = verify_all()
        sys.exit(0 if ok else 1)

    if not args.data_only:
        download_model()
    if not args.model_only:
        download_longalign()
        download_wikitext()

    verify_all()


if __name__ == "__main__":
    main()
