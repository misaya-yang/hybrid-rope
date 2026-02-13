"""
Unified RoPE Frequency Distribution Search (3 configs x 3 seeds) - ModelScope edition.

Differences from unified_search_3cfg_3seed.py:
- Tokenizer is loaded from local path (no HuggingFace network dependency)
- Dataset is loaded from ModelScope MsDataset
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from modelscope.msdatasets import MsDataset
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import a100.unified_search_3cfg_3seed as base


TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_PATH",
    "/root/autodl-tmp/dfrope/ms_models/EleutherAI/gpt-neox-20b",
)
MS_DATASET_ID = os.environ.get("MS_DATASET_ID", "AI-ModelScope/TinyStories")


def load_data_ms(tokenizer, max_tokens, seq_len):
    print(f"  Loading {MS_DATASET_ID} train ({max_tokens/1e6:.0f}M tokens) from ModelScope...")
    ds = MsDataset.load(MS_DATASET_ID, subset_name="default", split="train")
    ids = []
    for row in ds:
        text = row.get("text", "")
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    n = len(ids) // seq_len
    return base.torch.tensor(ids[: n * seq_len], dtype=base.torch.long).view(n, seq_len)


def load_val_ms(tokenizer, max_tokens=5_000_000):
    print(f"  Loading {MS_DATASET_ID} validation from ModelScope...")
    ds = MsDataset.load(MS_DATASET_ID, subset_name="default", split="validation")
    ids = []
    for row in ds:
        text = row.get("text", "")
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    return base.torch.tensor(ids, dtype=base.torch.long)


def _fmt_ms(mean, std):
    return f"{mean:.3f} +- {std:.3f}"


def main():
    Path(base.WORK_DIR).mkdir(parents=True, exist_ok=True)
    out_json = f"{base.WORK_DIR}/results.json"

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_data = load_data_ms(tok, base.TRAIN_TOKENS, base.SEQ_LEN)
    val_data = load_val_ms(tok)

    K = base.MODEL_CFG["head_dim"] // 2
    configs = [
        ("geo_500k", base.geometric_freq(K, 500000)),
        (
            "hybrid_a0.2_t100k",
            base.hybrid_freq(base.geometric_freq(K, 100000), base.anchored_poly_freq(K, 100000, 3.9, 0.3), 0.2),
        ),
        ("anchpoly_p3.9_omf0.3_t500k", base.anchored_poly_freq(K, 500000, 3.9, 0.3)),
    ]

    results = {}
    rows = []

    for cfg_name, freq in configs:
        results[cfg_name] = {}
        for seed in base.SEEDS:
            base.SEED = int(seed)
            run_name = f"{cfg_name}_seed{seed}"
            r = base.run_one(run_name, freq, train_data, val_data)
            results[cfg_name][str(seed)] = r

            p2 = r.get("2048")
            p16 = r.get("16384")
            rows.append((cfg_name, seed, p2, p16))

            with open(out_json, "w") as f:
                json.dump({"results": results, "rows": rows}, f, indent=2)

    print("\nConfig              | Seed | PPL@2048 | PPL@16384")
    for cfg_name, seed, p2, p16 in rows:
        print(f"{cfg_name:<19} | {seed:>4} | {p2:>8} | {p16:>9}")

    print("\nConfig              | PPL@2048 (mean+-std) | PPL@16384 (mean+-std)")
    for cfg_name, _ in configs:
        vals2 = []
        vals16 = []
        for seed in base.SEEDS:
            rr = results[cfg_name][str(seed)]
            if "2048" in rr:
                vals2.append(float(rr["2048"]))
            if "16384" in rr:
                vals16.append(float(rr["16384"]))
        m2 = float(np.mean(vals2))
        s2 = float(np.std(vals2, ddof=0))
        m16 = float(np.mean(vals16))
        s16 = float(np.std(vals16, ddof=0))
        print(f"{cfg_name:<19} | {_fmt_ms(m2, s2):<22} | {_fmt_ms(m16, s16):<22}")

    print(f"\n[done] wrote {out_json}")


if __name__ == "__main__":
    main()
