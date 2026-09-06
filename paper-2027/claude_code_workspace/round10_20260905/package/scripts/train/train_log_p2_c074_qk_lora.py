#!/usr/bin/env python3
"""Frozen log-p2+c=.074, all-layer Q/K rank-8 LoRA screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


CONFIG_SHA = "0d15ebb6cb8d998513b46ef337214176a6fd59fe5f16b30387c70d5f87795a9c"
WEIGHT_SHA = "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
TABLE_SHA = "56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b"
DATA_STATUS = "OLMO2_PHASE_ADAROPE_IDENTIFIABLE_PAIR_VIEW_V2"
GAIN_COEFFICIENT = 0.074
ATTENTION_SCALING = 1.0 + GAIN_COEFFICIENT * math.log(4.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def load_view(root: Path, expected_length: int) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != DATA_STATUS or int(manifest.get("length", -1)) != expected_length:
        raise ValueError(f"training data contract drift: {root}")
    for name in ("input_ids.npy", "labels.npy"):
        expected = manifest["files"][name]["sha256"]
        if sha256_file(root / name) != expected:
            raise ValueError(f"training data hash drift: {root / name}")
    ids = np.load(root / "input_ids.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
    if ids.shape != (128, 2, expected_length) or labels.shape != ids.shape:
        raise ValueError(f"training data shape drift: {ids.shape}")
    return ids, labels, {
        "manifest_sha256": sha256_file(manifest_path),
        "input_ids_sha256": manifest["files"]["input_ids.npy"]["sha256"],
        "labels_sha256": manifest["files"]["labels.npy"]["sha256"],
    }


def validate(args) -> dict:
    if sha256_file(args.checkpoint / "config.json") != CONFIG_SHA:
        raise ValueError("checkpoint config drift")
    if sha256_file(args.checkpoint / "model.safetensors") != WEIGHT_SHA:
        raise ValueError("checkpoint weight drift")
    table = np.load(args.table, allow_pickle=False)
    if table.shape != (64,) or table.dtype != np.float32 or array_sha256(table) != TABLE_SHA:
        raise ValueError("log-s4 table identity drift")
    if not np.all(table[:-1] > table[1:]):
        raise ValueError("log-s4 frequency order drift")
    _, _, short_receipt = load_view(args.short_data, 4096)
    _, _, long_receipt = load_view(args.long_data, 16384)
    if args.steps <= 0 or args.learning_rate <= 0 or args.rank != 8 or args.alpha != 16:
        raise ValueError("registered rank/alpha/steps/lr drift")
    return {
        "checkpoint_config_sha256": CONFIG_SHA,
        "checkpoint_weight_sha256": WEIGHT_SHA,
        "table_sha256_float32": TABLE_SHA,
        "table_file_sha256": sha256_file(args.table),
        "gain_coefficient": GAIN_COEFFICIENT,
        "attention_scaling": ATTENTION_SCALING,
        "short_data": short_receipt,
        "long_data": long_receipt,
        "steps": args.steps,
        "rank": args.rank,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "schedule": "LLLSS repeating; each optimizer step accumulates correct+deranged variants",
        "metric_boundary": "training diagnostics only; no benchmark outcome is read",
    }


def run(args, receipt: dict) -> None:
    if not args.authorized:
        raise ValueError("GPU training requires --authorized")
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import load_model
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.evaluate_instruct_ruler_transfer import apply_frequency

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    short_ids, short_labels, _ = load_view(args.short_data, 4096)
    long_ids, long_labels, _ = load_view(args.long_data, 16384)
    table = np.load(args.table, allow_pickle=False)

    base = load_model(args.checkpoint)
    apply_frequency(base, "native")
    rotary = base.model.rotary_emb
    with torch.no_grad():
        rotary.inv_freq.copy_(torch.from_numpy(table).to(rotary.inv_freq))
        rotary.original_inv_freq = rotary.inv_freq.detach().clone()
        rotary.attention_scaling = ATTENTION_SCALING
    base.config.use_cache = False
    model = get_peft_model(
        base,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj"],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            init_lora_weights=True,
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to("cuda").train()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("trainable scope escaped LoRA")
    if any(not ("q_proj" in name or "k_proj" in name) for name, _ in trainable):
        raise RuntimeError("trainable scope escaped Q/K")
    trainable_count = sum(p.numel() for _, p in trainable)
    optimizer = torch.optim.AdamW(
        [p for _, p in trainable], lr=args.learning_rate, weight_decay=0.0, fused=True
    )
    warmup = min(10, args.steps)
    args.output.mkdir(parents=True)
    log_path = args.output / "train_log.jsonl"
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    logs = []
    for step in range(1, args.steps + 1):
        use_short = step % 5 in (4, 0)
        ids_view, labels_view = (short_ids, short_labels) if use_short else (long_ids, long_labels)
        length = 4096 if use_short else 16384
        row_index = (args.seed + 17 * step) % 128
        optimizer.zero_grad(set_to_none=True)
        losses = []
        step_started = time.perf_counter()
        for variant in (0, 1):
            ids = torch.from_numpy(np.asarray(ids_view[row_index, variant], dtype=np.int64))[None].to("cuda")
            labels = torch.from_numpy(np.asarray(labels_view[row_index, variant], dtype=np.int64))[None].to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}")
            (loss / 2.0).backward()
            losses.append(float(loss.detach()))
            del ids, labels, loss
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0))
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError(f"invalid gradient at step {step}: {grad_norm}")
        scale = step / warmup if step <= warmup else 0.5 * (
            1.0 + math.cos(math.pi * (step - warmup) / max(args.steps - warmup, 1))
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        optimizer.step()
        elapsed = time.perf_counter() - step_started
        row = {
            "step": step,
            "length": length,
            "source_row": int(row_index),
            "loss": float(np.mean(losses)),
            "gradient_norm": grad_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            "tokens_per_second": float(2 * length / elapsed),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        logs.append(row)
        with log_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    adapter_dir = args.output / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    final = {
        "status": "LOG_P2_C074_QK_RANK8_SCREEN_COMPLETE",
        "protocol": receipt,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "peft_trainable_parameters": trainable_count,
            "total_seconds": time.perf_counter() - started,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "first_step": logs[0],
        "last_step": logs[-1],
        "train_log_sha256": sha256_file(log_path),
        "adapter_files": {
            path.name: sha256_file(path) for path in sorted(adapter_dir.iterdir()) if path.is_file()
        },
        "claim_limit": "Training completion only; no task improvement is established.",
    }
    (args.output / "receipt.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--short-data", type=Path, required=True)
    parser.add_argument("--long-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    receipt = validate(args)
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", **receipt}, indent=2, sort_keys=True))
        return 0
    run(args, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
