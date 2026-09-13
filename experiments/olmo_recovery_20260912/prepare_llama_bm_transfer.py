#!/usr/bin/env python3
"""Assemble trusted Llama-tokenized BM transfer pools without byte hashing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def entry(path: Path, fmt: str, rows: int) -> dict:
    if not path.is_file() or rows <= 0:
        raise ValueError(f"missing or empty pool: {path}")
    return {"path": str(path.resolve()), "format": fmt, "rows": int(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pg19", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--longcite", type=Path, required=True)
    parser.add_argument("--short-public", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    pg19, public, longcite, short_public = (
        path.resolve() for path in (args.pg19, args.public, args.longcite, args.short_public)
    )
    if args.out.exists():
        raise FileExistsError(args.out)
    config = read(model / "config.json")
    expected = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("unexpected Llama transfer geometry")
    weight_files = sorted(model.glob("model-*-of-*.safetensors"))
    if len(weight_files) != 4 or not (model / "model.safetensors.index.json").is_file():
        raise ValueError("the trusted Llama checkpoint is incomplete")

    pg = read(pg19 / "pg19_manifest.json")
    pub = read(public / "manifest.json")
    cite = read(longcite / "manifest.json")
    short = read(short_public / "manifest.json")
    if pub.get("status") != "COMPLETE" or cite.get("status") != "COMPLETE" or short.get("status") != "COMPLETE":
        raise ValueError("one or more Llama-tokenized SFT pools are incomplete")
    if any(Path(item.get("model", "")).resolve() != model for item in (pub, cite, short)):
        raise ValueError("one or more SFT pools use another tokenizer/model identity")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    terminal_id = int(tokenizer.eos_token_id)
    if terminal_id <= 0 or not tokenizer.chat_template:
        raise ValueError("Llama tokenizer lacks the native chat/EOS contract")

    pub_rows = pub["rows"]
    cite_rows = cite["rows"]
    pg_rows = pg["rows"]
    train_windows = {
        8192: len(pg_rows["train_8192"]),
        16384: len(pg_rows["train_16384"]),
    }
    pools = {
        "long_lm_8192": [entry(pg19 / "cpt_train_8192.npy", "npy", train_windows[8192])],
        "long_lm_16384": [entry(pg19 / "cpt_train.npy", "npy", train_windows[16384])],
        "long_sft_8192": [
            entry(public / "long_sft_8192_train.jsonl", "jsonl",
                  pub_rows["longalign/8192/train"] + pub_rows["longalpaca/8192/train"]),
            entry(longcite / "longcite_8192_train.jsonl", "jsonl", cite_rows["selected/8192/train"]),
        ],
        "long_sft_16384": [
            entry(public / "long_sft_16384_train.jsonl", "jsonl",
                  pub_rows["longalign/16384/train"] + pub_rows["longalpaca/16384/train"]),
            entry(longcite / "longcite_16384_train.jsonl", "jsonl", cite_rows["selected/16384/train"]),
        ],
        "short_sft": [entry(short_public / "native_short_sft_train.jsonl", "jsonl",
                            short["rows"]["ultrachat/native/train"])],
    }

    args.out.mkdir(parents=True)
    identity = {
        "status": "READY",
        "assistant_terminal_id": terminal_id,
        "model": str(model),
        "asset_identity_policy": "user_attested_clone/no_sha_validation",
    }
    data = {
        "status": "READY",
        "model": str(model),
        "native_length": 8192,
        "target_length": 32768,
        "pools": pools,
        "lm_evaluation": {
            "dev": str((pg19 / "lm_validation.npy").resolve()),
            "test": str((pg19 / "lm_test.npy").resolve()),
        },
        "evaluation_panels": {"dev": [], "test": []},
        "asset_identity_policy": "user_attested_clone/no_sha_validation",
    }
    plan = {
        "status": "CPU_PLAN_GPU_NOT_RUN",
        "model_path": str(model),
        "model_type": "llama",
        "native_length": 8192,
        "target_length": 32768,
        "rank": 32,
        "alpha": 32,
        "dropout": 0.0,
        "modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "seed": 20260912,
        "data_manifest": str((args.out / "data_identity.json").resolve()),
        "loss_weights": {"cpt": 1.0, "sft": 1.0, "replay": 0.25, "kl": 0.25},
        "table": "exact frozen Llama BM, reference=8192, scale=4, gain=1+0.1*ln(4)",
        "asset_identity_policy": "user_attested_clone/no_sha_validation",
    }
    write(args.out / "data_identity.json", identity)
    write(args.out / "manifest.json", data)
    write(args.out / "plan.json", plan)
    print(json.dumps({
        "status": "READY",
        "terminal_id": terminal_id,
        "pool_rows": {name: sum(item["rows"] for item in records) for name, records in pools.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
