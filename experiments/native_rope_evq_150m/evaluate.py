#!/usr/bin/env python3
"""Four-condition NLL/PPL and Passkey evaluation for the paired 151.9M run."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from experiments.native_rope_evq_150m.prepare_data import (
    sha256_file,
    validate_data_manifest,
)
from experiments.native_rope_evq_150m.model import GPT
from experiments.native_rope_evq_150m.protocol import ARMS, SPEC, get_arm_inv_freq
from experiments.native_rope_evq_150m.train import tensor_sha256
from scripts.lib.rope.official_yarn import (
    official_yarn_on_inv_freq,
    official_yarn_on_native_grid,
    parity_vs_official_source,
)
from scripts.supporting_eval.eval_passkey_scratch import (
    PASSKEY_SUFFIX,
    build_passkey_eval_sequence,
)


EVAL_LENGTHS = (2_048, 4_096, 8_192, 16_384)
EVAL_DEPTHS = (0.10, 0.25, 0.50, 0.75, 0.90)
OPERATORS = ("raw", "yarn")


def target_yarn_factor(length: int) -> float:
    if int(length) not in EVAL_LENGTHS:
        raise ValueError(
            f"registered evaluation length must be one of {EVAL_LENGTHS}, got {length}"
        )
    return max(1.0, float(length) / float(SPEC.seq_len))


def apply_registered_operator(
    inv_freq: torch.Tensor,
    *,
    arm: str,
    operator: str,
    length: int,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    factor = target_yarn_factor(length)
    inv = inv_freq.detach().cpu().to(torch.float64).view(-1)
    if operator == "raw":
        return inv.clone(), 1.0, {
            "mode": "raw_substrate",
            "operator": "raw",
            "scale": 1.0,
            "mscale": 1.0,
            "public_label": (
                "native endpoint RoPE"
                if arm == "native_rope"
                else "endpoint EVQ-Cosh tau=1.5"
            ),
        }
    if operator != "yarn":
        raise ValueError(f"unknown operator {operator!r}; expected {OPERATORS}")

    if arm == "native_rope":
        transformed, mscale, meta = official_yarn_on_native_grid(
            head_dim=SPEC.head_dim,
            base=SPEC.rope_base,
            scale=factor,
            original_max_position_embeddings=SPEC.seq_len,
            beta_fast=32.0,
            beta_slow=1.0,
        )
        public_label = (
            "native endpoint RoPE (identity at 2K)"
            if factor == 1.0
            else "official YaRN on native endpoint RoPE"
        )
    else:
        transformed, mscale, meta = official_yarn_on_inv_freq(
            inv,
            head_dim=SPEC.head_dim,
            base=SPEC.rope_base,
            scale=factor,
            original_max_position_embeddings=SPEC.seq_len,
            beta_fast=32.0,
            beta_slow=1.0,
        )
        public_label = (
            "endpoint EVQ-Cosh (identity at 2K)"
            if factor == 1.0
            else "YaRN-derived on endpoint EVQ-Cosh"
        )
    metadata = dict(meta)
    metadata.update(
        {
            "operator": "yarn",
            "target_length": int(length),
            "scale": factor,
            "mscale": float(mscale),
            "public_label": public_label,
        }
    )
    return transformed, float(mscale), metadata


def fixed_validation_offsets(
    validation_tokens: int,
    length: int,
    *,
    chunks: int,
    seed: int = 9_999,
) -> list[int]:
    max_start = int(validation_tokens) - int(length)
    if max_start < 0:
        raise ValueError(
            f"validation has {validation_tokens} tokens but length={length} was requested"
        )
    count = min(int(chunks), max_start + 1)
    rng = np.random.RandomState(int(seed) + int(length))
    offsets = rng.choice(max_start + 1, size=count, replace=False)
    return sorted(int(value) for value in offsets.tolist())


def aggregate_nll(losses: Iterable[float]) -> dict[str, float | int]:
    values = [float(value) for value in losses]
    if not values:
        raise ValueError("cannot aggregate an empty NLL list")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite NLL value: {values}")
    mean = sum(values) / len(values)
    return {
        "mean_nll": mean,
        "ppl": math.exp(mean) if mean < 700 else float("inf"),
        "sample_count": len(values),
    }


def causal_nll_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    tail_tokens: int = SPEC.seq_len,
) -> tuple[float, float]:
    """Return full-context and final-window token NLL from one forward pass."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError(
            f"expected logits=(B,T,V) and targets=(B,T), got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    tail = min(max(int(tail_tokens), 1), targets.size(1))
    full_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )
    tail_loss = F.cross_entropy(
        logits[:, -tail:, :].reshape(-1, logits.size(-1)),
        targets[:, -tail:].reshape(-1),
    )
    return float(full_loss.detach().cpu()), float(tail_loss.detach().cpu())


def set_runtime_rope(
    model: GPT,
    inv_freq: torch.Tensor,
    *,
    max_position: int,
    mscale: float,
) -> None:
    rope = model.blocks[0].attn.rope
    if any(block.attn.rope is not rope for block in model.blocks):
        raise RuntimeError("GPT blocks do not share the registered rotary module")
    value = inv_freq.to(device=rope.inv_freq.device, dtype=rope.inv_freq.dtype)
    if value.shape != rope.inv_freq.shape:
        raise ValueError(f"inv_freq shape {value.shape} != model shape {rope.inv_freq.shape}")
    rope.inv_freq.copy_(value)
    rope._build(int(max_position))
    if abs(float(mscale) - 1.0) > 1e-12:
        rope.cos_c.mul_(float(mscale))
        rope.sin_c.mul_(float(mscale))


def _load_checkpoint(arm_dir: Path, arm: str) -> tuple[GPT, dict[str, Any], torch.Tensor]:
    inv_path = arm_dir / "inv_freq.npy"
    checkpoint_path = arm_dir / "model.pt"
    meta_path = arm_dir / "train_meta.json"
    if not inv_path.is_file() or not checkpoint_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"incomplete checkpoint directory: {arm_dir}")
    inv = torch.from_numpy(np.load(inv_path, allow_pickle=False)).to(torch.float64)
    expected = get_arm_inv_freq(arm)
    if not torch.allclose(inv, expected, atol=1e-7, rtol=1e-6):
        raise ValueError(f"{arm} inv_freq.npy does not match the registered schedule")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("arm") != arm:
        raise ValueError(f"checkpoint metadata arm mismatch in {meta_path}")
    if metadata.get("inv_freq_sha256") != tensor_sha256(expected):
        raise ValueError(f"checkpoint metadata frequency hash mismatch for {arm}")

    model = GPT(SPEC.model_config(), inv.float())
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {
        name
        for name in missing
        if name == "lm_head.weight"
        or name.endswith(".attention.rope.inv_freq")
    }
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(
            f"checkpoint state mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    return model.to("cuda"), metadata, inv


@torch.no_grad()
def evaluate_natural_text(
    model: GPT,
    validation: np.ndarray,
    base_inv: torch.Tensor,
    *,
    arm: str,
    operator: str,
    chunks: int,
) -> dict[str, Any]:
    model.eval()
    output: dict[str, Any] = {}
    for length in EVAL_LENGTHS:
        inv, mscale, operator_meta = apply_registered_operator(
            base_inv, arm=arm, operator=operator, length=length
        )
        set_runtime_rope(
            model,
            inv,
            max_position=length + 64,
            mscale=mscale,
        )
        offsets = fixed_validation_offsets(len(validation), length, chunks=chunks)
        losses: list[float] = []
        tail_losses: list[float] = []
        for offset in offsets:
            ids = torch.from_numpy(
                np.array(validation[offset : offset + length], copy=True)
            ).unsqueeze(0)
            ids = ids.to("cuda", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ids[:, :-1])
                full_value, tail_value = causal_nll_metrics(
                    logits,
                    ids[:, 1:],
                    tail_tokens=SPEC.seq_len,
                )
            if not math.isfinite(full_value) or not math.isfinite(tail_value):
                raise RuntimeError(
                    f"non-finite natural-text NLL for {arm}/{operator}/L={length}"
                )
            losses.append(full_value)
            tail_losses.append(tail_value)
            del ids, logits
        full_summary = aggregate_nll(losses)
        tail_summary = aggregate_nll(tail_losses)
        output[str(length)] = {
            **full_summary,
            "mean_tail_nll": tail_summary["mean_nll"],
            "tail_ppl": tail_summary["ppl"],
            "tail_tokens": min(SPEC.seq_len, length - 1),
            "offsets": offsets,
            "per_offset_nll": losses,
            "per_offset_tail_nll": tail_losses,
            "operator": operator_meta,
            "inv_freq_sha256": tensor_sha256(inv),
        }
        print(
            f"[{arm}/{operator}] L={length} "
            f"NLL={output[str(length)]['mean_nll']:.4f} "
            f"PPL={output[str(length)]['ppl']:.3f}",
            flush=True,
        )
    return output


def _secret_from_seed(seed: int) -> str:
    rng = random.Random(int(seed))
    return "-".join(str(rng.randint(0, 9)) for _ in range(5))


def _training_secrets(indices: np.ndarray) -> set[str]:
    return {_secret_from_seed(int(index)) for index in indices.tolist()}


def _heldout_secret(seed: int, forbidden: set[str]) -> str:
    candidate_seed = int(seed)
    while True:
        value = _secret_from_seed(candidate_seed)
        if value not in forbidden:
            return value
        candidate_seed += 1


def _wrong_secret(correct: str, seed: int) -> str:
    rng = random.Random(int(seed))
    wrong = []
    for item in correct.split("-"):
        offset = rng.randint(1, 9)
        wrong.append(str((int(item) + offset) % 10))
    return "-".join(wrong)


@torch.no_grad()
def _score_answer_batch(model: GPT, sequences: list[torch.Tensor], answer_len: int) -> list[float]:
    batch = torch.stack(sequences).to("cuda", non_blocking=True)
    start = batch.size(1) - int(answer_len) - 1
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch[:, :-1])
    answer_logits = logits[:, start : start + int(answer_len), :].float()
    targets = batch[:, start + 1 : start + 1 + int(answer_len)]
    losses = F.cross_entropy(
        answer_logits.reshape(-1, answer_logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(batch.size(0), int(answer_len))
    result = losses.mean(dim=1).detach().cpu().tolist()
    del batch, logits, answer_logits, targets, losses
    return [float(value) for value in result]


@torch.no_grad()
def evaluate_passkey(
    model: GPT,
    tokenizer,
    validation: np.ndarray,
    passkey_indices: np.ndarray,
    base_inv: torch.Tensor,
    *,
    arm: str,
    operator: str,
    trials: int,
) -> dict[str, Any]:
    model.eval()
    filler = torch.from_numpy(
        np.array(validation[: max(EVAL_LENGTHS) + 256], copy=True)
    )
    forbidden = _training_secrets(passkey_indices)
    details: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    all_gaps: list[float] = []
    all_retrieved: list[bool] = []

    for length in EVAL_LENGTHS:
        inv, mscale, operator_meta = apply_registered_operator(
            base_inv, arm=arm, operator=operator, length=length
        )
        set_runtime_rope(
            model,
            inv,
            max_position=length + 64,
            mscale=mscale,
        )
        for depth in EVAL_DEPTHS:
            sequences: list[torch.Tensor] = []
            trial_meta: list[dict[str, Any]] = []
            answer_len: int | None = None
            for trial in range(int(trials)):
                seed = 7_000_000 + length * 100 + int(depth * 1_000) + trial
                correct = _heldout_secret(seed, forbidden)
                wrong = _wrong_secret(correct, seed + 17)
                prompt, _, _ = build_passkey_eval_sequence(
                    filler,
                    correct,
                    tokenizer,
                    total_length=length,
                    depth_percent=depth,
                )
                correct_ids = tokenizer.encode(
                    f"{correct}{PASSKEY_SUFFIX}", add_special_tokens=False
                )
                wrong_ids = tokenizer.encode(
                    f"{wrong}{PASSKEY_SUFFIX}", add_special_tokens=False
                )
                if len(correct_ids) != len(wrong_ids):
                    raise RuntimeError("correct and wrong Passkey answers tokenize differently")
                if answer_len is None:
                    answer_len = len(correct_ids)
                if answer_len != len(correct_ids):
                    raise RuntimeError("Passkey answer token length changed across trials")
                sequences.append(
                    torch.cat([prompt, torch.tensor(correct_ids, dtype=torch.long)])
                )
                sequences.append(
                    torch.cat([prompt, torch.tensor(wrong_ids, dtype=torch.long)])
                )
                trial_meta.append(
                    {
                        "length": length,
                        "depth": depth,
                        "trial": trial,
                        "seed": seed,
                        "correct_passkey": correct,
                        "wrong_passkey": wrong,
                    }
                )
            if answer_len is None:
                raise RuntimeError("Passkey evaluation requires at least one trial")
            nlls = _score_answer_batch(model, sequences, answer_len)
            gaps: list[float] = []
            retrieved: list[bool] = []
            for trial, metadata in enumerate(trial_meta):
                correct_nll = nlls[2 * trial]
                wrong_nll = nlls[2 * trial + 1]
                gap = wrong_nll - correct_nll
                is_retrieved = gap > 0.0
                details.append(
                    {
                        **metadata,
                        "correct_nll": correct_nll,
                        "wrong_nll": wrong_nll,
                        "nll_gap": gap,
                        "retrieved": is_retrieved,
                    }
                )
                gaps.append(gap)
                retrieved.append(is_retrieved)
            key = f"L={length}_depth={depth:.2f}"
            summaries[key] = {
                "mean_nll_gap": sum(gaps) / len(gaps),
                "retrieval_rate": sum(retrieved) / len(retrieved),
                "retrieval_rate_chance_boundary": 0.5,
                "trials": len(gaps),
                "operator": operator_meta,
            }
            all_gaps.extend(gaps)
            all_retrieved.extend(retrieved)
            print(
                f"[{arm}/{operator}] PK L={length} depth={depth:.2f} "
                f"gap={summaries[key]['mean_nll_gap']:+.4f} "
                f"retrieval={summaries[key]['retrieval_rate']:.0%}",
                flush=True,
            )
    return {
        "metric": "teacher-forced NLL_wrong - NLL_correct",
        "primary_statistic": "continuous mean_nll_gap",
        "sign_rate_role": "diagnostic_only",
        "retrieval_rate_chance_boundary": 0.5,
        "autoregressive_exact_match_run": False,
        "training_secret_overlap": 0,
        "summary": summaries,
        "global": {
            "mean_nll_gap": sum(all_gaps) / len(all_gaps),
            "retrieval_rate": sum(all_retrieved) / len(all_retrieved),
            "retrieval_rate_chance_boundary": 0.5,
            "trials": len(all_gaps),
        },
        "details": details,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def dry_run(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    validate_data_manifest(manifest, check_files=False)
    parity = parity_vs_official_source(
        head_dim=SPEC.head_dim,
        base=SPEC.rope_base,
        scale=8.0,
        original_max_position_embeddings=SPEC.seq_len,
    )
    if not parity["parity_ok"]:
        raise RuntimeError(f"official YaRN parity failed: {parity}")
    operators: dict[str, Any] = {}
    for arm in ARMS:
        operators[arm] = {}
        base_inv = get_arm_inv_freq(arm)
        for operator in OPERATORS:
            operators[arm][operator] = {}
            for length in EVAL_LENGTHS:
                inv, mscale, meta = apply_registered_operator(
                    base_inv, arm=arm, operator=operator, length=length
                )
                operators[arm][operator][str(length)] = {
                    "inv_freq_sha256": tensor_sha256(inv),
                    "mscale": mscale,
                    "metadata": meta,
                }
    report = {
        "dry_run": True,
        "data_manifest_sha256": sha256_file(manifest_path),
        "official_yarn_parity": parity,
        "operators": operators,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    manifest_path = Path(args.data_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    validate_data_manifest(manifest, check_files=False)
    parity = parity_vs_official_source(
        head_dim=SPEC.head_dim,
        base=SPEC.rope_base,
        scale=8.0,
        original_max_position_embeddings=SPEC.seq_len,
    )
    if not parity["parity_ok"]:
        raise RuntimeError(f"official YaRN parity failed: {parity}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["tokenizer"]["path"], local_files_only=True
    )
    validation = np.load(
        manifest["validation"]["path"], mmap_mode="r", allow_pickle=False
    )
    passkey_indices = np.load(
        manifest["passkey"]["indices_path"], allow_pickle=False
    )
    work_dir = Path(args.work_dir).resolve()
    arm_metadata = {
        arm: json.loads((work_dir / arm / "train_meta.json").read_text())
        for arm in ARMS
    }
    initial_hashes = {
        arm: arm_metadata[arm]["initial_trainable_sha256"] for arm in ARMS
    }
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError(f"paired arms have different initial weights: {initial_hashes}")
    order_hashes = {arm: arm_metadata[arm]["row_order_sha256"] for arm in ARMS}
    if len(set(order_hashes.values())) != 1:
        raise RuntimeError(f"paired arms have different row orders: {order_hashes}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "paired 151.9M native-RoPE versus endpoint-EVQ diagnostic",
        "claim_boundary": (
            "Passkey is explicitly supervised at 2K. Native scaling uses official YaRN; "
            "EVQ scaling is a YaRN-derived virtual-coordinate generalization."
        ),
        "data_manifest_sha256": sha256_file(manifest_path),
        "initial_trainable_sha256": next(iter(initial_hashes.values())),
        "row_order_sha256": next(iter(order_hashes.values())),
        "official_yarn_parity": parity,
        "conditions": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for arm in ARMS:
        model, metadata, inv = _load_checkpoint(work_dir / arm, arm)
        report["conditions"][arm] = {"train_meta": metadata, "operators": {}}
        for operator in OPERATORS:
            condition_started = time.time()
            natural = evaluate_natural_text(
                model,
                validation,
                inv,
                arm=arm,
                operator=operator,
                chunks=int(args.ppl_chunks),
            )
            passkey = evaluate_passkey(
                model,
                tokenizer,
                validation,
                passkey_indices,
                inv,
                arm=arm,
                operator=operator,
                trials=int(args.pk_trials),
            )
            report["conditions"][arm]["operators"][operator] = {
                "natural_text": natural,
                "passkey": passkey,
                "seconds": time.time() - condition_started,
            }
        del model
        torch.cuda.empty_cache()
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "raw_results.json", report)
    summary = {
        "schema_version": report["schema_version"],
        "claim_boundary": report["claim_boundary"],
        "conditions": {},
    }
    for arm, arm_result in report["conditions"].items():
        summary["conditions"][arm] = {}
        for operator, condition in arm_result["operators"].items():
            summary["conditions"][arm][operator] = {
                "natural_text": {
                    length: {
                        "mean_nll": values["mean_nll"],
                        "ppl": values["ppl"],
                        "mean_tail_nll": values["mean_tail_nll"],
                        "tail_ppl": values["tail_ppl"],
                        "sample_count": values["sample_count"],
                    }
                    for length, values in condition["natural_text"].items()
                },
                "passkey_global": condition["passkey"]["global"],
            }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_dir", type=Path)
    parser.add_argument("--data_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--ppl_chunks", type=int, default=8)
    parser.add_argument("--pk_trials", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(Path(args.data_manifest).resolve())
        return
    if args.work_dir is None:
        parser.error("--work_dir is required unless --dry_run")
    if args.output_dir is None:
        args.output_dir = Path(args.work_dir) / "evaluation"
    run(args)


if __name__ == "__main__":
    main()
