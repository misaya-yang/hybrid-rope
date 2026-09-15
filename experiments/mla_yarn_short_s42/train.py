#!/usr/bin/env python3
"""Matched seed-42 MLA K=16 pilot: train at 512 tokens without rewriting data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.native_rope_evq_150m.prepare_data import sha256_file
from experiments.native_rope_evq_150m.protocol import legacy_passkey_indices
from experiments.native_rope_evq_150m.train import trainable_state_sha256
from scripts.core_text_phases.run_gqa_evq_experiment import GPT
from scripts.core_text_phases.run_evq_sweep import (
    maybe_wrap_with_passkey_mix,
    set_seed,
    train_model,
)
from scripts.lib.rope.schedules import evq_cosh_inv_freq, geometric_inv_freq


SEED = 42
SEQ_LEN = 512
TRAIN_TOKENS = 99_999_744
TRAIN_ROWS = TRAIN_TOKENS // SEQ_LEN
BASE = 500_000.0
TAU = 1.414
PASSKEY_RATIO = 0.02
PARAMETERS = 432_194_560
ARMS = ("native_rope", "endpoint_evq_tau1p414")


def config(batch_size: int) -> dict:
    return {
        "vocab_size": 50_304,
        "hidden_size": 1_024,
        "num_layers": 24,
        "num_heads": 16,
        "head_dim": 64,
        "intermediate_size": 4_096,
        "max_position_embeddings": SEQ_LEN,
        "seq_len": SEQ_LEN,
        "batch_size": int(batch_size),
        "train_tokens": TRAIN_TOKENS,
        "lr": 2e-4,
        "eval_lengths": [1_024, 2_048, 4_096, 8_192],
        "eval_chunks": 8,
        "passkey_mix_ratio": PASSKEY_RATIO,
        "attn_type": "mla",
        "d_rope": 32,
        "d_nope": 32,
        "v_head_dim": 64,
        "kv_lora_rank": 256,
    }


def inv_freq(arm: str) -> torch.Tensor:
    if arm == "native_rope":
        return geometric_inv_freq(32, BASE, dtype=torch.float64)
    if arm == "endpoint_evq_tau1p414":
        return evq_cosh_inv_freq(
            32, TAU, BASE, midpoint=False, dtype=torch.float64
        )
    raise ValueError(f"unknown arm: {arm}")


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def load_inputs(args: argparse.Namespace):
    source = Path(args.train_npy).resolve()
    validation_path = Path(args.validation_npy).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    for path in (source, validation_path, tokenizer_path):
        if not path.exists():
            raise FileNotFoundError(path)
    actual_source_sha = sha256_file(source)
    if actual_source_sha != args.train_sha256:
        raise ValueError("training tensor SHA256 mismatch")
    actual_val_sha = sha256_file(validation_path)
    if actual_val_sha != args.validation_sha256:
        raise ValueError("validation tensor SHA256 mismatch")

    source_array = np.load(source, mmap_mode="r", allow_pickle=False)
    if source_array.dtype != np.int64 or source_array.ndim != 2:
        raise ValueError(f"unexpected training tensor: {source_array.dtype} {source_array.shape}")
    flat = source_array.reshape(-1)
    if flat.size < TRAIN_TOKENS:
        raise ValueError(f"training tensor has {flat.size} tokens, need {TRAIN_TOKENS}")
    rows = flat[:TRAIN_TOKENS].reshape(TRAIN_ROWS, SEQ_LEN)
    train = torch.from_numpy(rows)

    validation_array = np.load(validation_path, mmap_mode="r", allow_pickle=False)
    if validation_array.dtype != np.int64 or validation_array.ndim != 1:
        raise ValueError(
            f"unexpected validation tensor: {validation_array.dtype} {validation_array.shape}"
        )
    validation = torch.from_numpy(np.array(validation_array, copy=True))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    return train, validation, tokenizer, actual_source_sha, actual_val_sha


def identities(arm: str, batch_size: int) -> tuple[GPT, dict]:
    if TRAIN_ROWS % batch_size:
        raise ValueError(f"batch_size={batch_size} does not divide {TRAIN_ROWS} rows")
    cfg = config(batch_size)
    set_seed(SEED)
    base_inv = inv_freq(arm)
    model = GPT(cfg, base_inv.float())
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != PARAMETERS:
        raise RuntimeError(f"parameter count {n_params} != {PARAMETERS}")
    order_gen = torch.Generator(device="cpu").manual_seed(SEED)
    order = torch.randperm(TRAIN_ROWS, generator=order_gen)
    selected = torch.tensor(
        legacy_passkey_indices(TRAIN_ROWS, PASSKEY_RATIO), dtype=torch.int64
    )
    return model, {
        "arm": arm,
        "seed": SEED,
        "parameter_count": n_params,
        "initial_trainable_sha256": trainable_state_sha256(model),
        "inv_freq_sha256": tensor_sha256(base_inv),
        "row_order_sha256": tensor_sha256(order),
        "passkey_indices_sha256": tensor_sha256(selected),
        "passkey_rows": int(selected.numel()),
        "passkey_tokens": int(selected.numel() * SEQ_LEN),
        "passkey_actual_ratio": float(selected.numel() / TRAIN_ROWS),
        "optimizer_steps": TRAIN_ROWS // batch_size,
        "model_config": cfg,
    }


def probe(args: argparse.Namespace) -> None:
    train, _, _, train_sha, val_sha = load_inputs(args)
    model, meta = identities(args.arm, args.batch_size)
    model = model.cuda().train()
    batch = train[: args.batch_size].cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
        )
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    meta.update(
        {
            "probe": True,
            "loss": float(loss.detach().cpu()),
            "seconds": time.time() - started,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "train_sha256": train_sha,
            "validation_sha256": val_sha,
        }
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


def train(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve() / args.arm
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_data, validation, tokenizer, train_sha, val_sha = load_inputs(args)
    model, meta = identities(args.arm, args.batch_size)
    meta.update(
        {
            "schema_version": 1,
            "scope": "single-seed supporting/mechanistic MLA K=16 pilot",
            "train_length": SEQ_LEN,
            "train_tokens": TRAIN_TOKENS,
            "base": BASE,
            "tau": None if args.arm == "native_rope" else TAU,
            "frequency_grid": "endpoint u=k/K",
            "train_tensor_path": str(Path(args.train_npy).resolve()),
            "train_tensor_sha256": train_sha,
            "validation_tensor_path": str(Path(args.validation_npy).resolve()),
            "validation_tensor_sha256": val_sha,
            "tokenizer_path": str(Path(args.tokenizer).resolve()),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    (output / "train_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    np.save(output / "inv_freq.npy", inv_freq(args.arm).float().numpy())

    mixed = maybe_wrap_with_passkey_mix(
        train_data=train_data,
        filler_tokens=validation[:50_000],
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        passkey_ratio=PASSKEY_RATIO,
    )
    set_seed(SEED)
    model = model.cuda()
    started = time.time()
    model = train_model(model, mixed, config(args.batch_size), seed=SEED)
    torch.cuda.synchronize()
    meta["train_seconds"] = time.time() - started
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    state = {name: value.detach().cpu() for name, value in model.named_parameters()}
    temporary = output / "model.pt.incomplete"
    torch.save({"model": state, "metadata": meta}, temporary)
    temporary.replace(output / "model.pt")
    meta["checkpoint_sha256"] = sha256_file(output / "model.pt")
    (output / "train_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "train"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--train-npy", required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--validation-npy", required=True)
    parser.add_argument("--validation-sha256", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    probe(args) if args.mode == "probe" else train(args)


if __name__ == "__main__":
    main()
