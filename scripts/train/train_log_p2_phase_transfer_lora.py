#!/usr/bin/env python3
"""Train static-table LoRA on physical 2x/4x source-contrastive examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


PAIR_STATUS = "OLMO2_PHASE_ADAROPE_IDENTIFIABLE_PAIR_VIEW_V2"
NATIVE_LENGTH = 4096
FAMILY_PATTERN = ("near_2x", "far_4x", "near_2x", "far_4x", "replay_1x")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float32_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def family_for_step(step: int) -> str:
    if step <= 0:
        raise ValueError("step must be positive")
    return FAMILY_PATTERN[(step - 1) % len(FAMILY_PATTERN)]


def load_pair_view(
    root: Path, expected_length: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != PAIR_STATUS
        or int(manifest.get("length", -1)) != expected_length
    ):
        raise ValueError(f"paired data contract drift: {root}")
    if manifest.get("split") not in (None, "train", "train_eos"):
        raise ValueError(f"paired data split drift: expected train split, got {manifest.get('split')}")
    for name in ("input_ids.npy", "labels.npy"):
        if sha256_file(root / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"paired data hash drift: {root / name}")
    ids = np.load(root / "input_ids.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
    if ids.shape != (128, 2, expected_length) or labels.shape != ids.shape:
        raise ValueError(f"paired data shape drift: {ids.shape}")
    masks = labels != -100
    if not np.array_equal(masks[:, 0], masks[:, 1]) or np.any(masks[:, 0].sum(axis=1) < 2):
        raise ValueError("paired targets must align and include answer plus EOS")
    if not np.any(labels[:, 0][masks[:, 0]] != labels[:, 1][masks[:, 1]]):
        raise ValueError("paired views contain no counterfactual answer tokens")
    return ids, labels, {
        "manifest_sha256": sha256_file(manifest_path),
        "input_ids_sha256": manifest["files"]["input_ids.npy"]["sha256"],
        "labels_sha256": manifest["files"]["labels.npy"]["sha256"],
        "length": expected_length,
    }


def validate(args: argparse.Namespace) -> dict:
    checkpoint = args.checkpoint.expanduser().resolve()
    table_path = args.table.expanduser().resolve()
    table = np.load(table_path, allow_pickle=False)
    if (
        table.dtype != np.float32
        or table.shape != (64,)
        or not np.all(table[:-1] > table[1:])
        or not np.all(table > 0.0)
    ):
        raise ValueError("static table identity drift")
    _, _, replay = load_pair_view(args.replay_data.expanduser().resolve(), NATIVE_LENGTH)
    _, _, near = load_pair_view(args.near_data.expanduser().resolve(), 2 * NATIVE_LENGTH)
    _, _, far = load_pair_view(args.far_data.expanduser().resolve(), 4 * NATIVE_LENGTH)
    target_modules = tuple(args.target_modules)
    if target_modules not in (
        ("q_proj", "k_proj"),
        ("q_proj", "k_proj", "v_proj", "o_proj"),
    ):
        raise ValueError("target modules must be QK or QKVO")
    if (
        args.steps <= 0
        or args.rank <= 0
        or args.alpha <= 0
        or args.learning_rate <= 0
        or args.margin < 0.0
        or args.margin_weight < 0.0
        or not math.isfinite(args.attention_scaling)
        or args.attention_scaling <= 0.0
    ):
        raise ValueError("training hyperparameters are invalid")
    return {
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_weight_sha256": sha256_file(checkpoint / "model.safetensors"),
        "table_float32_sha256": float32_sha256(table),
        "table_file_sha256": sha256_file(table_path),
        "attention_scaling": float(args.attention_scaling),
        "data": {"replay_1x": replay, "near_2x": near, "far_4x": far},
        "steps": int(args.steps),
        "rank": int(args.rank),
        "alpha": int(args.alpha),
        "target_modules": list(target_modules),
        "learning_rate": float(args.learning_rate),
        "margin": float(args.margin),
        "margin_weight": float(args.margin_weight),
        "schedule": list(FAMILY_PATTERN),
        "training_physical_lengths": [NATIVE_LENGTH, 2 * NATIVE_LENGTH, 4 * NATIVE_LENGTH],
        "evaluation_length_firewall": [8, 16, 32],
    }


def paired_objective(
    *,
    model,
    ids_host: np.ndarray,
    labels_host: np.ndarray,
    counterfactual_source_host: np.ndarray,
    margin: float,
    margin_weight: float,
):
    import torch
    import torch.nn.functional as F

    positions_host = np.flatnonzero(labels_host != -100).astype(np.int64)
    targets_host = labels_host[positions_host].astype(np.int64)
    correct_host = np.asarray(ids_host, dtype=np.int64)
    counterfactual_host = np.asarray(counterfactual_source_host, dtype=np.int64).copy()
    counterfactual_host[positions_host] = correct_host[positions_host]
    if np.array_equal(correct_host, counterfactual_host):
        raise ValueError("source counterfactual changed no input token")
    ids = torch.from_numpy(correct_host)[None].to("cuda")
    counterfactual_ids = torch.from_numpy(counterfactual_host)[None].to("cuda")
    positions = torch.from_numpy(positions_host).to("cuda")
    targets = torch.from_numpy(targets_host)[None].to("cuda")
    backbone = model.get_base_model().model
    lm_head = model.get_base_model().lm_head
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = backbone(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        logits = lm_head(hidden[:, positions - 1]).float()
        counterfactual_hidden = backbone(
            input_ids=counterfactual_ids, use_cache=False, return_dict=True
        ).last_hidden_state
        counterfactual_logits = lm_head(counterfactual_hidden[:, positions - 1]).float()
        correct_logprob = logits.log_softmax(dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        counterfactual_logprob = counterfactual_logits.log_softmax(dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        ce = -correct_logprob.mean()
        source_effect = correct_logprob - counterfactual_logprob
        if source_effect.shape[1] < 2:
            raise ValueError("source effect requires answer tokens before terminal EOS")
        margin_loss = F.softplus(float(margin) - source_effect[:, :-1]).mean()
        loss = ce + float(margin_weight) * margin_loss
    return loss, {
        "ce": float(ce.detach()),
        "source_margin_loss": float(margin_loss.detach()),
        "mean_source_effect": float(source_effect[:, :-1].mean().detach()),
        "contrastive_tokens": int(source_effect.shape[1] - 1),
    }


def run(args: argparse.Namespace, receipt: dict) -> None:
    if not args.authorized:
        raise ValueError("GPU training requires --authorized")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import load_model

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    views = {
        "replay_1x": load_pair_view(args.replay_data.expanduser().resolve(), NATIVE_LENGTH)[:2],
        "near_2x": load_pair_view(args.near_data.expanduser().resolve(), 2 * NATIVE_LENGTH)[:2],
        "far_4x": load_pair_view(args.far_data.expanduser().resolve(), 4 * NATIVE_LENGTH)[:2],
    }
    table = np.load(args.table.expanduser().resolve(), allow_pickle=False)
    base = load_model(args.checkpoint.expanduser().resolve())
    rotary = base.model.rotary_emb
    with torch.no_grad():
        rotary.inv_freq.copy_(torch.from_numpy(table).to(rotary.inv_freq))
        rotary.original_inv_freq = rotary.inv_freq.detach().clone()
        rotary.attention_scaling = float(args.attention_scaling)
    base.config.max_position_embeddings = 32 * NATIVE_LENGTH
    base.config.use_cache = False
    model = get_peft_model(
        base,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(args.target_modules),
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            init_lora_weights=True,
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to("cuda").train()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("trainable scope escaped LoRA")
    if any(not any(module in name for module in args.target_modules) for name, _ in trainable):
        raise RuntimeError("trainable scope escaped requested projections")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    warmup = min(args.warmup_steps, args.steps)
    generator = np.random.default_rng(args.seed)
    output.mkdir(parents=True)
    log_path = output / "train_log.jsonl"
    exposure_digest = hashlib.sha256()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    logs: list[dict] = []

    for step in range(1, args.steps + 1):
        family = family_for_step(step)
        ids_view, labels_view = views[family]
        row_index = int(generator.integers(ids_view.shape[0]))
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        components: list[dict] = []
        step_started = time.perf_counter()
        for variant in (0, 1):
            loss, metrics = paired_objective(
                model=model,
                ids_host=np.asarray(ids_view[row_index, variant]),
                labels_host=np.asarray(labels_view[row_index, variant]),
                counterfactual_source_host=np.asarray(ids_view[row_index, 1 - variant]),
                margin=args.margin,
                margin_weight=args.margin_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}")
            (loss / 2.0).backward()
            losses.append(float(loss.detach()))
            components.append(metrics)
            del loss
        grad_norm = float(torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0))
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError(f"invalid gradient at step {step}: {grad_norm}")
        scale = step / warmup if step <= warmup else 0.5 * (
            1.0 + math.cos(math.pi * (step - warmup) / max(args.steps - warmup, 1))
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        optimizer.step()
        exposure_digest.update(np.asarray([step, row_index], dtype="<i8").tobytes())
        elapsed = time.perf_counter() - step_started
        record = {
            "step": step,
            "family": family,
            "physical_length": int(ids_view.shape[-1]),
            "source_row": row_index,
            "loss": float(np.mean(losses)),
            "answer_ce": float(np.mean([value["ce"] for value in components])),
            "counterfactual_source_margin_loss": float(
                np.mean([value["source_margin_loss"] for value in components])
            ),
            "mean_source_effect": float(
                np.mean([value["mean_source_effect"] for value in components])
            ),
            "contrastive_tokens": int(sum(value["contrastive_tokens"] for value in components)),
            "gradient_norm": grad_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            "tokens_per_second": float(2 * ids_view.shape[-1] / elapsed),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        logs.append(record)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(json.dumps(record, sort_keys=True), flush=True)

    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    final = {
        "status": "LOG_P2_PHYSICAL_2X4X_TRANSFER_LORA_COMPLETE_V1",
        "protocol": receipt,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
            "total_seconds": time.perf_counter() - started,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "exposure_sha256": exposure_digest.hexdigest(),
        "first_step": logs[0],
        "last_step": logs[-1],
        "train_log_sha256": sha256_file(log_path),
        "adapter_files": {
            path.name: sha256_file(path) for path in sorted(adapter.iterdir()) if path.is_file()
        },
        "claim_limit": "training completion only; 8x/16x/32x require held-out physical evaluation",
    }
    (output / "receipt.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--attention-scaling", type=float, required=True)
    parser.add_argument("--replay-data", type=Path, required=True)
    parser.add_argument("--near-data", type=Path, required=True)
    parser.add_argument("--far-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--target-modules", nargs="+", default=("q_proj", "k_proj", "v_proj", "o_proj"))
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260904)
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
