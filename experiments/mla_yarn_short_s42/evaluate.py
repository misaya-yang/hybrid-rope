#!/usr/bin/env python3
"""Four-arm inference evaluation for the seed-42 short-context MLA pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.mla_yarn_short_s42.train import (
    ARMS,
    BASE,
    PARAMETERS,
    PASSKEY_RATIO,
    SEED,
    SEQ_LEN,
    TAU,
    TRAIN_ROWS,
    config,
    inv_freq,
)
from experiments.native_rope_evq_150m.evaluate import (
    fixed_validation_offsets,
    load_weights_only_checkpoint,
)
from experiments.native_rope_evq_150m.prepare_data import sha256_directory, sha256_file
from experiments.native_rope_evq_150m.protocol import legacy_passkey_indices
from experiments.native_rope_evq_150m.train import tensor_sha256
from scripts.core_text_phases.run_gqa_evq_experiment import GPT
from scripts.lib.rope.official_yarn import (
    get_mscale,
    official_yarn_on_inv_freq,
    official_yarn_on_native_grid,
)
from scripts.supporting_eval.eval_passkey_scratch import (
    PASSKEY_SUFFIX,
    build_passkey_eval_sequence,
)


LENGTHS = (1_024, 2_048, 4_096, 8_192)
DEPTHS = (0.10, 0.25, 0.50, 0.75, 0.90)
OPERATORS = (
    ("raw", False, False),
    ("freq_only", True, False),
    ("mscale_only", False, True),
    ("full", True, True),
)
SOURCE_MANIFEST_SHA256 = "7839d8dd26a46c49aa9750aba9d5a724532024dd60c6b4e6454a84ef8049c549"
FLAT_GAP_NLL = 0.02
APPROX_NLL = 0.01


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _aggregate(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid NLL values: {values}")
    mean = sum(values) / len(values)
    return {"mean_nll": mean, "ppl": math.exp(mean), "sample_count": len(values)}


def apply_operator(
    base_inv: torch.Tensor,
    *,
    arm: str,
    length: int,
    use_frequency: bool,
    use_mscale: bool,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    if arm not in ARMS or length not in LENGTHS:
        raise ValueError(f"unregistered condition: {arm}/{length}")
    scale = float(length) / float(SEQ_LEN)
    base_value = base_inv.detach().cpu().to(torch.float64).view(-1)
    if arm == "native_rope":
        transformed, official_mscale, meta = official_yarn_on_native_grid(
            head_dim=32,
            base=BASE,
            scale=scale,
            original_max_position_embeddings=SEQ_LEN,
            beta_fast=32.0,
            beta_slow=1.0,
        )
        label = "official YaRN on native endpoint RoPE"
    else:
        transformed, official_mscale, meta = official_yarn_on_inv_freq(
            base_value,
            head_dim=32,
            base=BASE,
            scale=scale,
            original_max_position_embeddings=SEQ_LEN,
            beta_fast=32.0,
            beta_slow=1.0,
        )
        label = "YaRN-derived on endpoint EVQ-Cosh"
    runtime_inv = transformed if use_frequency else base_value.clone()
    runtime_mscale = float(official_mscale) if use_mscale else 1.0
    operator = next(
        name
        for name, frequency, scaling in OPERATORS
        if (frequency, scaling) == (use_frequency, use_mscale)
    )
    operator_meta = dict(meta)
    source_yarn_mode = operator_meta["mode"]
    operator_meta.update(
        {
            "operator": operator,
            "target_length": int(length),
            "scale": scale,
            "official_mscale": float(official_mscale),
            "mscale": runtime_mscale,
            "runtime_mscale": runtime_mscale,
            "use_frequency_transform": use_frequency,
            "use_attention_scaling": use_mscale,
            "public_label": (
                "raw substrate"
                if operator == "raw"
                else f"{label} ({operator})"
            ),
        }
    )
    if operator == "raw":
        operator_meta["source_yarn_mode"] = source_yarn_mode
        operator_meta["mode"] = "raw_substrate"
    elif operator == "mscale_only":
        operator_meta["source_yarn_mode"] = source_yarn_mode
        operator_meta["mode"] = "yarn_mscale_only"
    return runtime_inv, runtime_mscale, operator_meta


def set_runtime_rope(
    model: GPT,
    runtime_inv: torch.Tensor,
    *,
    max_position: int,
    mscale: float,
) -> None:
    rope = model.blocks[0].attn.rope
    if any(block.attn.rope is not rope for block in model.blocks):
        raise RuntimeError("MLA blocks do not share one rotary module")
    value = runtime_inv.to(device=rope.inv_freq.device, dtype=rope.inv_freq.dtype)
    if value.shape != rope.inv_freq.shape:
        raise ValueError(f"frequency shape {value.shape} != {rope.inv_freq.shape}")
    with torch.no_grad():
        rope.inv_freq.copy_(value)
        rope._build(int(max_position))
        if float(mscale) != 1.0:
            rope.cos_c.mul_(float(mscale))
            rope.sin_c.mul_(float(mscale))


def _checkpoint_identity(arm_dir: Path, arm: str) -> dict[str, Any]:
    checkpoint = arm_dir / "model.pt"
    inv_path = arm_dir / "inv_freq.npy"
    meta_path = arm_dir / "train_meta.json"
    for path in (checkpoint, inv_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(meta_path.read_text())
    required = (
        "arm",
        "seed",
        "parameter_count",
        "initial_trainable_sha256",
        "inv_freq_sha256",
        "row_order_sha256",
        "passkey_indices_sha256",
        "train_tensor_sha256",
        "validation_tensor_sha256",
        "train_tensor_path",
        "validation_tensor_path",
        "tokenizer_path",
        "checkpoint_sha256",
        "model_config",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"missing checkpoint metadata for {arm}: {missing}")
    if metadata["arm"] != arm or metadata["seed"] != SEED:
        raise ValueError(f"arm/seed identity mismatch for {arm}")
    if metadata["parameter_count"] != PARAMETERS or metadata["model_config"] != config(16):
        raise ValueError(f"architecture identity mismatch for {arm}")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != metadata["checkpoint_sha256"]:
        raise ValueError(f"checkpoint SHA256 mismatch for {arm}")
    expected_f64 = inv_freq(arm)
    if metadata["inv_freq_sha256"] != tensor_sha256(expected_f64):
        raise ValueError(f"canonical frequency hash mismatch for {arm}")
    saved_inv_array = np.load(inv_path, allow_pickle=False)
    expected_f32 = expected_f64.float().numpy()
    if saved_inv_array.dtype != np.float32 or not np.array_equal(saved_inv_array, expected_f32):
        raise ValueError(f"runtime frequency tensor mismatch for {arm}")
    return {
        "arm": arm,
        "checkpoint_sha256": checkpoint_sha,
        "runtime_inv_freq_sha256": _array_sha256(saved_inv_array),
        "canonical_inv_freq_sha256": metadata["inv_freq_sha256"],
        "metadata": metadata,
        "inv_freq": torch.from_numpy(saved_inv_array.copy()).to(torch.float64),
    }


def _paired_identity(
    work_dir: Path, source_manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    identities = {arm: _checkpoint_identity(work_dir / arm, arm) for arm in ARMS}
    native = identities[ARMS[0]]["metadata"]
    evq = identities[ARMS[1]]["metadata"]
    matched = (
        "model_config",
        "initial_trainable_sha256",
        "row_order_sha256",
        "passkey_indices_sha256",
        "train_tensor_sha256",
        "validation_tensor_sha256",
        "train_tensor_path",
        "validation_tensor_path",
        "tokenizer_path",
    )
    for key in matched:
        if native[key] != evq[key]:
            raise ValueError(f"matched-control identity differs: {key}")
    if native["inv_freq_sha256"] == evq["inv_freq_sha256"]:
        raise ValueError("substrate frequency hashes must differ")
    if native["train_tensor_sha256"] != source_manifest["train"]["sha256"]:
        raise ValueError("training tensor does not match source manifest")
    if native["validation_tensor_sha256"] != source_manifest["validation"]["sha256"]:
        raise ValueError("validation tensor does not match source manifest")
    expected_paths = {
        "train_tensor_path": source_manifest["train"]["path"],
        "validation_tensor_path": source_manifest["validation"]["path"],
        "tokenizer_path": source_manifest["tokenizer"]["path"],
    }
    for key, expected in expected_paths.items():
        if Path(native[key]).resolve() != Path(expected).resolve():
            raise ValueError(f"source path identity mismatch: {key}")
    selected = np.asarray(legacy_passkey_indices(TRAIN_ROWS, PASSKEY_RATIO), dtype=np.int64)
    if _array_sha256(selected) != native["passkey_indices_sha256"]:
        raise ValueError("derived 512-token passkey selector mismatch")
    public = {
        arm: {
            "checkpoint_sha256": value["checkpoint_sha256"],
            "runtime_inv_freq_sha256": value["runtime_inv_freq_sha256"],
            "canonical_inv_freq_sha256": value["canonical_inv_freq_sha256"],
        }
        for arm, value in identities.items()
    }
    return public, identities


def preflight(source_manifest_path: Path, *, check_source_tensor: bool) -> dict[str, Any]:
    if sha256_file(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source data manifest SHA256 mismatch")
    manifest = json.loads(source_manifest_path.read_text())
    for section in ("train", "validation", "tokenizer"):
        if section not in manifest:
            raise ValueError(f"source manifest missing {section}")
    validation_path = Path(manifest["validation"]["path"])
    tokenizer_path = Path(manifest["tokenizer"]["path"])
    train_path = Path(manifest["train"]["path"])
    for path in (train_path, validation_path, tokenizer_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256_file(validation_path) != manifest["validation"]["sha256"]:
        raise ValueError("validation tensor SHA256 mismatch")
    if sha256_directory(tokenizer_path) != manifest["tokenizer"]["sha256"]:
        raise ValueError("tokenizer directory SHA256 mismatch")
    if check_source_tensor and sha256_file(train_path) != manifest["train"]["sha256"]:
        raise ValueError("source training tensor SHA256 mismatch")
    validation = np.load(validation_path, mmap_mode="r", allow_pickle=False)
    if validation.dtype != np.int64 or validation.ndim != 1:
        raise ValueError(f"invalid validation tensor: {validation.dtype} {validation.shape}")
    offsets = {
        str(length): fixed_validation_offsets(len(validation), length, chunks=8)
        for length in LENGTHS
    }
    if any(len(value) != 8 for value in offsets.values()):
        raise ValueError("each natural-text condition requires eight offsets")
    operators: dict[str, Any] = {}
    for arm in ARMS:
        base_inv = inv_freq(arm)
        operators[arm] = {}
        for name, use_frequency, use_mscale in OPERATORS:
            operators[arm][name] = {}
            for length in LENGTHS:
                runtime_inv, mscale, meta = apply_operator(
                    base_inv,
                    arm=arm,
                    length=length,
                    use_frequency=use_frequency,
                    use_mscale=use_mscale,
                )
                expected_mscale = get_mscale(length / SEQ_LEN) if use_mscale else 1.0
                if abs(mscale - expected_mscale) > 1e-12:
                    raise ValueError(f"mscale mismatch for {arm}/{name}/{length}")
                operators[arm][name][str(length)] = {
                    "inv_freq_sha256": tensor_sha256(runtime_inv),
                    "mscale": mscale,
                    "mode": meta["mode"],
                }
    return {
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_tensor_sha256_recomputed": bool(check_source_tensor),
        "source_tensor_training_access": "read-only mmap",
        "train_length": SEQ_LEN,
        "eval_lengths": list(LENGTHS),
        "natural_text_offsets": offsets,
        "passkey_cases_per_condition": len(LENGTHS) * len(DEPTHS) * 5,
        "passkey_metric": "teacher-forced NLL_wrong - NLL_correct",
        "operators": operators,
    }


def _load_model(
    arm_dir: Path, identity: dict[str, Any]
) -> tuple[GPT, torch.Tensor]:
    model = GPT(config(16), identity["inv_freq"].float())
    payload = load_weights_only_checkpoint(arm_dir / "model.pt")
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state: {arm_dir / 'model.pt'}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {
        name
        for name in missing
        if name == "head.weight" or name.endswith(".attn.rope.inv_freq")
    }
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(f"checkpoint state mismatch: missing={missing} unexpected={unexpected}")
    return model.to("cuda"), identity["inv_freq"]


@torch.no_grad()
def evaluate_natural(
    model: GPT,
    validation: np.ndarray,
    base_inv: torch.Tensor,
    *,
    arm: str,
    use_frequency: bool,
    use_mscale: bool,
) -> dict[str, Any]:
    model.eval()
    result: dict[str, Any] = {}
    for length in LENGTHS:
        runtime_inv, mscale, meta = apply_operator(
            base_inv,
            arm=arm,
            length=length,
            use_frequency=use_frequency,
            use_mscale=use_mscale,
        )
        set_runtime_rope(model, runtime_inv, max_position=length + 64, mscale=mscale)
        offsets = fixed_validation_offsets(len(validation), length, chunks=8)
        losses: list[float] = []
        tail_losses: list[float] = []
        for offset in offsets:
            ids = torch.from_numpy(np.array(validation[offset : offset + length], copy=True))
            ids = ids.unsqueeze(0).to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
                tail = min(SEQ_LEN, ids.size(1) - 1)
                tail_loss = F.cross_entropy(
                    logits[:, -tail:].reshape(-1, logits.size(-1)),
                    ids[:, 1:][:, -tail:].reshape(-1),
                )
            losses.append(float(loss.cpu()))
            tail_losses.append(float(tail_loss.cpu()))
            del ids, logits, loss, tail_loss
        operator = meta["operator"]
        summary = _aggregate(losses)
        tail_summary = _aggregate(tail_losses)
        result[str(length)] = {
            **summary,
            "mean_tail_nll": tail_summary["mean_nll"],
            "tail_ppl": tail_summary["ppl"],
            "tail_tokens": SEQ_LEN,
            "offsets": offsets,
            "per_offset_nll": losses,
            "per_offset_tail_nll": tail_losses,
            "operator": meta,
            "runtime_inv_freq_sha256": tensor_sha256(runtime_inv),
        }
        print(
            f"[{arm}/{operator}] L={length} NLL={summary['mean_nll']:.4f} PPL={summary['ppl']:.3f}",
            flush=True,
        )
    return result


def _secret(seed: int) -> str:
    rng = random.Random(int(seed))
    return "-".join(str(rng.randint(0, 9)) for _ in range(5))


def _heldout_secret(seed: int, forbidden: set[str]) -> str:
    while True:
        value = _secret(seed)
        if value not in forbidden:
            return value
        seed += 1


def _wrong_secret(correct: str, seed: int) -> str:
    rng = random.Random(seed)
    return "-".join(str((int(item) + rng.randint(1, 9)) % 10) for item in correct.split("-"))


@torch.no_grad()
def _score_answer(model: GPT, sequence: torch.Tensor, answer_len: int) -> float:
    batch = sequence.unsqueeze(0).to("cuda", non_blocking=True)
    answer_start = batch.size(1) - int(answer_len) - 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(batch[:, :-1])
    answer_logits = logits[:, answer_start : answer_start + answer_len].float()
    targets = batch[:, answer_start + 1 : answer_start + 1 + answer_len]
    loss = F.cross_entropy(
        answer_logits.reshape(-1, answer_logits.size(-1)), targets.reshape(-1)
    )
    value = float(loss.cpu())
    del batch, logits, answer_logits, targets, loss
    return value


@torch.no_grad()
def evaluate_passkey(
    model: GPT,
    tokenizer,
    validation: np.ndarray,
    base_inv: torch.Tensor,
    *,
    arm: str,
    use_frequency: bool,
    use_mscale: bool,
) -> dict[str, Any]:
    model.eval()
    filler = torch.from_numpy(np.array(validation[: max(LENGTHS) + 256], copy=True))
    selected = legacy_passkey_indices(TRAIN_ROWS, PASSKEY_RATIO)
    forbidden = {_secret(index) for index in selected}
    details: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for length in LENGTHS:
        runtime_inv, mscale, meta = apply_operator(
            base_inv,
            arm=arm,
            length=length,
            use_frequency=use_frequency,
            use_mscale=use_mscale,
        )
        set_runtime_rope(model, runtime_inv, max_position=length + 64, mscale=mscale)
        for depth in DEPTHS:
            gaps: list[float] = []
            retrieved: list[bool] = []
            for trial in range(5):
                seed = 7_000_000 + length * 100 + int(depth * 1_000) + trial
                correct = _heldout_secret(seed, forbidden)
                wrong = _wrong_secret(correct, seed + 17)
                prompt, _, _ = build_passkey_eval_sequence(
                    filler, correct, tokenizer, total_length=length, depth_percent=depth
                )
                correct_ids = tokenizer.encode(f"{correct}{PASSKEY_SUFFIX}", add_special_tokens=False)
                wrong_ids = tokenizer.encode(f"{wrong}{PASSKEY_SUFFIX}", add_special_tokens=False)
                if len(correct_ids) != len(wrong_ids):
                    raise RuntimeError("correct and wrong Passkey token counts differ")
                correct_nll = _score_answer(
                    model, torch.cat([prompt, torch.tensor(correct_ids)]), len(correct_ids)
                )
                wrong_nll = _score_answer(
                    model, torch.cat([prompt, torch.tensor(wrong_ids)]), len(wrong_ids)
                )
                gap = wrong_nll - correct_nll
                details.append(
                    {
                        "length": length,
                        "depth": depth,
                        "trial": trial,
                        "seed": seed,
                        "correct_passkey": correct,
                        "wrong_passkey": wrong,
                        "correct_nll": correct_nll,
                        "wrong_nll": wrong_nll,
                        "nll_gap": gap,
                        "retrieved": gap > 0.0,
                    }
                )
                gaps.append(gap)
                retrieved.append(gap > 0.0)
            summaries[f"L={length}_depth={depth:.2f}"] = {
                "mean_nll_gap": sum(gaps) / len(gaps),
                "retrieval_rate": sum(retrieved) / len(retrieved),
                "trials": len(gaps),
                "operator": meta,
            }
        length_gaps = [item["nll_gap"] for item in details if item["length"] == length]
        print(
            f"[{arm}/{meta['operator']}] PK L={length} gap={sum(length_gaps)/len(length_gaps):+.4f}",
            flush=True,
        )
    if len(details) != 100:
        raise RuntimeError(f"expected 100 Passkey cases, got {len(details)}")
    gaps = [item["nll_gap"] for item in details]
    return {
        "metric": "teacher-forced NLL_wrong - NLL_correct",
        "primary_statistic": "continuous mean_nll_gap",
        "autoregressive_exact_match_run": False,
        "training_secret_overlap": 0,
        "summary": summaries,
        "global": {
            "mean_nll_gap": sum(gaps) / len(gaps),
            "retrieval_rate": sum(item["retrieved"] for item in details) / len(details),
            "trials": len(details),
        },
        "details": details,
    }


def analyze(report: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for length in map(str, LENGTHS):
        native_raw = report["conditions"][ARMS[0]]["raw"]["natural_text"][length]["mean_nll"]
        evq_raw = report["conditions"][ARMS[1]]["raw"]["natural_text"][length]["mean_nll"]
        rows[length] = {}
        for operator, _, _ in OPERATORS:
            native = report["conditions"][ARMS[0]][operator]["natural_text"][length]
            evq = report["conditions"][ARMS[1]][operator]["natural_text"][length]
            rows[length][operator] = {
                "native_nll": native["mean_nll"],
                "native_ppl": native["ppl"],
                "evq_nll": evq["mean_nll"],
                "evq_ppl": evq["ppl"],
                "substrate_gap_native_minus_evq": native["mean_nll"] - evq["mean_nll"],
                "difference_in_differences":
                    (evq["mean_nll"] - evq_raw) - (native["mean_nll"] - native_raw),
            }
    long_lengths = ("4096", "8192")
    prediction_1_by_length = {
        length: abs(rows[length]["freq_only"]["substrate_gap_native_minus_evq"]) > FLAT_GAP_NLL
        for length in map(str, LENGTHS)
    }
    prediction_2_by_length = {
        length: rows[length]["mscale_only"]["evq_nll"]
        <= rows[length]["full"]["evq_nll"] + APPROX_NLL
        for length in map(str, LENGTHS)
    }
    prediction_3_by_length = {
        length: rows[length]["raw"]["evq_nll"] < rows[length]["full"]["native_nll"]
        for length in map(str, LENGTHS)
    }
    kill = all(not prediction_1_by_length[length] for length in long_lengths)
    return {
        "thresholds_preregistered_before_results": {
            "near_zero_substrate_gap_abs_nll": FLAT_GAP_NLL,
            "approximately_equal_nll_tolerance": APPROX_NLL,
            "decisive_lengths": [4_096, 8_192],
        },
        "natural_text_attribution": rows,
        "predictions": {
            "P1_freq_only_does_not_flatten_MLA_gap": {
                "by_length": prediction_1_by_length,
                "decisive": all(prediction_1_by_length[length] for length in long_lengths),
            },
            "P2_EVQ_mscale_is_approximately_or_better_than_EVQ_full": {
                "by_length": prediction_2_by_length,
                "decisive": all(prediction_2_by_length[length] for length in long_lengths),
            },
            "P3_EVQ_raw_beats_native_full": {
                "by_length": prediction_3_by_length,
                "decisive": all(prediction_3_by_length[length] for length in long_lengths),
            },
        },
        "kill_condition_triggered": kill,
    }


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    source_manifest_path = args.source_manifest.resolve()
    # Training already verified the 4 GB source SHA before opening it read-only.
    # Do not rescan it while paid GPU time is running.
    check = preflight(source_manifest_path, check_source_tensor=False)
    source_manifest = json.loads(source_manifest_path.read_text())
    paired, identity_cache = _paired_identity(args.work_dir.resolve(), source_manifest)
    validation = np.load(source_manifest["validation"]["path"], mmap_mode="r", allow_pickle=False)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source_manifest["tokenizer"]["path"], local_files_only=True
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "single-seed supporting/mechanistic MLA K=16 pilot",
        "claim_boundary": "single seed; no primary-claim promotion",
        "protocol": {
            "seed": SEED,
            "train_length": SEQ_LEN,
            "eval_lengths": list(LENGTHS),
            "natural_text_offsets": 8,
            "passkey_cases_per_condition": 100,
            "operators": [name for name, _, _ in OPERATORS],
        },
        "preflight": check,
        "checkpoint_identities": paired,
        "conditions": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        model, base_inv = _load_model(
            args.work_dir.resolve() / arm, identity_cache[arm]
        )
        report["conditions"][arm] = {}
        for operator, use_frequency, use_mscale in OPERATORS:
            started = time.time()
            report["conditions"][arm][operator] = {
                "natural_text": evaluate_natural(
                    model,
                    validation,
                    base_inv,
                    arm=arm,
                    use_frequency=use_frequency,
                    use_mscale=use_mscale,
                ),
                "passkey": evaluate_passkey(
                    model,
                    tokenizer,
                    validation,
                    base_inv,
                    arm=arm,
                    use_frequency=use_frequency,
                    use_mscale=use_mscale,
                ),
                "seconds": time.time() - started,
            }
            _json_write(output_dir / "raw_results.incomplete.json", report)
        del model
        torch.cuda.empty_cache()
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    raw_path = output_dir / "raw_results.json"
    _json_write(raw_path, report)
    (output_dir / "raw_results.incomplete.json").unlink(missing_ok=True)
    analysis_path = output_dir / "analysis.json"
    _json_write(analysis_path, analyze(report))
    manifest = {
        "schema_version": 1,
        "artifact_role": report["artifact_role"],
        "claim_boundary": report["claim_boundary"],
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "checkpoint_sha256": {
            arm: paired[arm]["checkpoint_sha256"] for arm in ARMS
        },
        "artifacts": {
            raw_path.name: sha256_file(raw_path),
            analysis_path.name: sha256_file(analysis_path),
        },
    }
    manifest_path = output_dir / "sanitized_manifest.json"
    _json_write(manifest_path, manifest)
    sums = {
        path.name: sha256_file(path)
        for path in (raw_path, analysis_path, manifest_path)
    }
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(preflight(args.source_manifest.resolve(), check_source_tensor=False), indent=2, sort_keys=True))
        return
    if args.work_dir is None or args.output_dir is None:
        parser.error("--work-dir and --output-dir are required unless --dry-run")
    run(args)


if __name__ == "__main__":
    main()
