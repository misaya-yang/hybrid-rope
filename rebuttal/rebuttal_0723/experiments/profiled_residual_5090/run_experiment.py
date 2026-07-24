#!/usr/bin/env python3
"""Profile pure RoPE residuals on frozen 500M-token checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from experiments.native_rope_evq_150m.model import GPT
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.prepare import (
    sha256_file,
    validate_experiment_manifest,
)
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.protocol import (
    ARMS as SOURCE_ARMS,
    SPEC as SOURCE_SPEC,
    training_inv_freq,
)
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment import (
    _state_inv_freq,
    build_model,
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.profiled_residual_5090.protocol import (
    GATE_ARM,
    PROFILE_FAMILIES,
    SPEC,
    candidate_grid,
    candidate_inv_freq,
    decomposition_report,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent
    for parent in (PACKAGE_DIR, *PACKAGE_DIR.parents)
    if (parent / "AGENTS.md").is_file()
    and (parent / "scripts/lib/rope/schedules.py").is_file()
)
SOURCE_CHECKPOINT_CODE_SHA256 = (
    "351419fcda6d3bd32e57dec122fdd7cf892b17b96c3b1d9bfa845949e6797e14"
)


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        PACKAGE_DIR / "protocol.py",
        PACKAGE_DIR / "run_experiment.py",
        PACKAGE_DIR / "run_5090.sh",
        REPO_ROOT / "tests/test_profiled_residual_5090.py",
        REPO_ROOT / "experiments/native_rope_evq_150m/model.py",
        REPO_ROOT / "rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/protocol.py",
        REPO_ROOT / "rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/run_experiment.py",
        REPO_ROOT / "rebuttal/rebuttal_0723/experiments/reviewer27be_shape_base/real_rope_schedules.py",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(row)
    return rows


def _small_file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def _checkpoint_record(source_work_dir: Path, arm: str) -> dict[str, Any]:
    run_dir = source_work_dir.resolve() / "runs" / arm
    checkpoint = run_dir / "model.pt"
    metadata_path = run_dir / "train_meta.json"
    inv_path = run_dir / "inv_freq.npy"
    for path in (checkpoint, metadata_path, inv_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "arm": arm,
        "protocol_sha256": SOURCE_SPEC.fingerprint(),
        "code_sha256": SOURCE_CHECKPOINT_CODE_SHA256,
        "seed": SOURCE_SPEC.seed,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{arm} metadata {key} mismatch")
    saved_inv = torch.from_numpy(np.load(inv_path, allow_pickle=False)).contiguous()
    canonical_inv = training_inv_freq(arm)
    if saved_inv.dtype != torch.float32 or not torch.equal(saved_inv, canonical_inv):
        raise ValueError(f"{arm} inv_freq sidecar differs from canonical schedule")
    if tensor_sha256(saved_inv) != metadata.get("training_inv_freq_sha256"):
        raise ValueError(f"{arm} inv_freq hash differs from train metadata")
    stat = checkpoint.stat()
    return {
        "arm": arm,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "checkpoint_sha256_from_train_receipt": metadata.get("checkpoint_sha256"),
        "metadata": _small_file_record(metadata_path),
        "inv_freq": _small_file_record(inv_path),
        "training_inv_freq_sha256": tensor_sha256(saved_inv),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if torch.cuda.is_available():
        raise RuntimeError("preflight must run before the paid GPU is enabled")
    source_manifest_path = args.source_data_manifest.resolve()
    source_manifest = json.loads(source_manifest_path.read_text())
    validate_experiment_manifest(
        source_manifest, check_files=True, check_content_hashes=False
    )
    anchor_manifest_path = args.anchor_manifest.resolve()
    anchor_manifest = json.loads(anchor_manifest_path.read_text())
    if anchor_manifest.get("validation", {}).get("sha256") != source_manifest.get(
        "validation", {}
    ).get("sha256"):
        raise ValueError("anchor and source manifests use different validation tensors")

    anchor_records: dict[str, Any] = {}
    all_anchors = []
    for split, expected_count in (
        ("selection", SPEC.selection_anchors),
        ("test", SPEC.test_anchors),
    ):
        record = anchor_manifest[f"{split}_anchors"]
        path = Path(record["path"]).resolve()
        anchors = np.load(path, allow_pickle=False)
        if anchors.dtype != np.int64 or anchors.shape != (expected_count,):
            raise ValueError(f"{split} anchor tensor shape/dtype mismatch")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"{split} anchor hash mismatch")
        if int(anchors.min()) < max(SPEC.eval_lengths):
            raise ValueError(f"{split} anchors cannot supply 8K windows")
        all_anchors.extend(int(value) for value in anchors)
        anchor_records[split] = {
            **_small_file_record(path),
            "count": len(anchors),
            "values": anchors.tolist(),
        }
    ordered = np.asarray(sorted(all_anchors), dtype=np.int64)
    if len(set(ordered.tolist())) != len(ordered):
        raise ValueError("selection/test anchors overlap")
    if np.any(np.diff(ordered) < max(SPEC.eval_lengths)):
        raise ValueError("registered 8K anchor windows overlap")

    validation_path = Path(source_manifest["validation"]["path"]).resolve()
    validation = np.load(validation_path, mmap_mode="r", allow_pickle=False).reshape(-1)
    if validation.dtype != np.int64 or int(validation.max()) >= SOURCE_SPEC.vocab_size:
        raise ValueError("validation tensor is incompatible with the model vocabulary")

    schedules = {}
    for length in SPEC.eval_lengths:
        rows = candidate_grid(length)
        schedules[str(length)] = {
            "valid_candidates": len(rows),
            "candidates": rows,
        }
        if not any(row["family"] == "base" for row in rows):
            raise RuntimeError(f"no valid base candidate at length {length}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    decomposition = decomposition_report()
    _atomic_json(output / "decomposition.json", decomposition)
    receipt = {
        "status": "READY",
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "checkpoint_training_code_sha256": SOURCE_CHECKPOINT_CODE_SHA256,
        "source_data_manifest": _small_file_record(source_manifest_path),
        "anchor_manifest": _small_file_record(anchor_manifest_path),
        "validation": {
            "path": str(validation_path),
            "sha256_from_manifest": source_manifest["validation"]["sha256"],
            "tokens": int(validation.size),
        },
        "anchors": anchor_records,
        "checkpoints": {
            arm: _checkpoint_record(args.source_work_dir, arm) for arm in SOURCE_ARMS
        },
        "schedules": schedules,
        "gate_arm": GATE_ARM,
        "claim_boundary": (
            "single-seed checkpoint-only supporting diagnostic; anchor bootstrap "
            "does not replace independent training seeds"
        ),
    }
    _atomic_json(output / "READY.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _load_ready(output_dir: Path) -> dict[str, Any]:
    path = output_dir.resolve() / "READY.json"
    ready = json.loads(path.read_text())
    if ready.get("status") != "READY":
        raise ValueError("READY receipt does not have READY status")
    if ready.get("protocol_sha256") != SPEC.fingerprint():
        raise ValueError("protocol changed after preflight")
    if ready.get("code_sha256") != code_fingerprint():
        raise ValueError("profile code changed after preflight")
    if ready.get("checkpoint_training_code_sha256") != SOURCE_CHECKPOINT_CODE_SHA256:
        raise ValueError("checkpoint training-code identity changed after preflight")
    for record in ready["checkpoints"].values():
        path = Path(record["checkpoint"])
        stat = path.stat()
        if stat.st_size != record["checkpoint_size"] or stat.st_mtime_ns != record[
            "checkpoint_mtime_ns"
        ]:
            raise ValueError(f"checkpoint changed after preflight: {path}")
    return ready


def _validate_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    props = torch.cuda.get_device_properties(0)
    total_memory = getattr(props, "total_memory", getattr(props, "total_mem", 0))
    if int(total_memory) < 28 * 2**30:
        raise RuntimeError(f"need >=28 GiB VRAM, found {total_memory / 2**30:.1f}")
    return {
        "name": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "memory_bytes": int(total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _load_model(ready: dict[str, Any], arm: str) -> tuple[GPT, dict[str, Any]]:
    record = ready["checkpoints"][arm]
    payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
    metadata = payload.get("metadata")
    state = payload.get("model")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError(f"invalid checkpoint payload for {arm}")
    expected = {
        "arm": arm,
        "protocol_sha256": SOURCE_SPEC.fingerprint(),
        "code_sha256": SOURCE_CHECKPOINT_CODE_SHA256,
        "seed": SOURCE_SPEC.seed,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"embedded {arm} metadata {key} mismatch")
    checkpoint_inv = _state_inv_freq(state, label=record["checkpoint"])
    if tensor_sha256(checkpoint_inv) != record["training_inv_freq_sha256"]:
        raise ValueError(f"embedded {arm} inv_freq differs from preflight")
    model = build_model(arm)
    model.load_state_dict(state, strict=True)
    del payload, state
    gc.collect()
    return model, metadata


def _set_rope(model: GPT, inv_freq: torch.Tensor, length: int) -> None:
    rope = model.blocks[0].attention.rope
    if any(block.attention.rope is not rope for block in model.blocks):
        raise RuntimeError("model blocks do not share one rotary module")
    if inv_freq.shape != rope.inv_freq.shape:
        raise ValueError("runtime inverse-frequency shape mismatch")
    rope.inv_freq.copy_(inv_freq.to(device=rope.inv_freq.device, dtype=rope.inv_freq.dtype))
    rope.attention_scaling = 1.0
    rope._build(int(length))


@torch.inference_mode()
def _evaluate_schedule(
    model: GPT,
    validation: np.ndarray,
    anchors: np.ndarray,
    *,
    inv_freq: torch.Tensor,
    length: int,
    max_batch_tokens: int,
) -> list[float]:
    _set_rope(model, inv_freq, length)
    batch_size = max(1, min(len(anchors), int(max_batch_tokens) // int(length)))
    values: list[float] = []
    for start in range(0, len(anchors), batch_size):
        endpoints = anchors[start : start + batch_size]
        windows = np.stack(
            [
                np.asarray(validation[int(end) - length : int(end)], dtype=np.int64)
                for end in endpoints
            ]
        )
        batch = torch.from_numpy(windows).to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        tail = min(SPEC.tail_tokens, targets.shape[1])
        per_token = F.cross_entropy(
            logits[:, -tail:].float().reshape(-1, logits.size(-1)),
            targets[:, -tail:].reshape(-1),
            reduction="none",
        ).reshape(len(endpoints), tail)
        means = per_token.mean(dim=1)
        if not torch.isfinite(means).all():
            raise RuntimeError("non-finite profile NLL")
        values.extend(float(value) for value in means.cpu())
        del batch, logits, targets, per_token, means
    return values


def _candidate_key(arm: str, length: int, row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        arm,
        int(length),
        str(row["family"]),
        float(row["base_multiplier"]),
        float(row["lambda_ratio"]),
    )


def _test_candidates(selection: dict[str, Any], arm: str, length: int) -> list[dict[str, Any]]:
    selected = selection["selected"][arm][str(length)]
    return [selected[family] for family in PROFILE_FAMILIES]


def profile(args: argparse.Namespace) -> None:
    runtime = _validate_cuda()
    output = args.output_dir.resolve()
    ready = _load_ready(output)
    arms = tuple(args.arms) if args.arms else tuple(SOURCE_ARMS)
    if any(arm not in SOURCE_ARMS for arm in arms):
        raise ValueError(f"unknown source arm in {arms}")
    selection = None
    if args.split == "test":
        selection = json.loads((output / "selection.json").read_text())
        if selection.get("protocol_sha256") != SPEC.fingerprint():
            raise ValueError("selection receipt protocol mismatch")
    anchor_record = ready["anchors"][args.split]
    anchors = np.load(anchor_record["path"], allow_pickle=False)
    validation = np.load(ready["validation"]["path"], mmap_mode="r", allow_pickle=False).reshape(-1)
    result_path = output / "profile" / f"{args.split}.jsonl"
    completed = {
        _candidate_key(row["arm"], row["length"], row) for row in _read_jsonl(result_path)
    } if result_path.exists() else set()
    print(json.dumps({"runtime": runtime, "split": args.split, "resume_rows": len(completed)}))

    for arm in arms:
        model, metadata = _load_model(ready, arm)
        model = model.to("cuda").eval()
        for length in SPEC.eval_lengths:
            candidates = (
                candidate_grid(length)
                if args.split == "selection"
                else _test_candidates(selection, arm, length)  # type: ignore[arg-type]
            )
            for candidate in candidates:
                key = _candidate_key(arm, length, candidate)
                if key in completed:
                    continue
                inv = candidate_inv_freq(
                    family=candidate["family"],
                    length=length,
                    base_multiplier=candidate["base_multiplier"],
                    lambda_ratio=candidate["lambda_ratio"],
                )
                started = time.time()
                nll = _evaluate_schedule(
                    model,
                    validation,
                    anchors,
                    inv_freq=inv,
                    length=length,
                    max_batch_tokens=args.max_batch_tokens,
                )
                row = {
                    "arm": arm,
                    "checkpoint_sha256": ready["checkpoints"][arm][
                        "checkpoint_sha256_from_train_receipt"
                    ],
                    "split": args.split,
                    "length": length,
                    "family": candidate["family"],
                    "base_multiplier": float(candidate["base_multiplier"]),
                    "lambda_ratio": float(candidate["lambda_ratio"]),
                    "strictly_monotonic": bool(candidate["strictly_monotonic"]),
                    "inv_freq_sha256": tensor_sha256(inv),
                    "anchor_sha256": anchor_record["sha256"],
                    "tail_nll": nll,
                    "mean_tail_nll": statistics.fmean(nll),
                    "elapsed_seconds": time.time() - started,
                    "metric": "mean teacher-forced NLL on final 128 targets",
                }
                _append_jsonl(result_path, row)
                completed.add(key)
                print(
                    f"[{args.split}] {arm} L={length} {candidate['family']} "
                    f"c={candidate['base_multiplier']} lambda={candidate['lambda_ratio']} "
                    f"NLL={row['mean_tail_nll']:.5f}",
                    flush=True,
                )
        del model, metadata
        gc.collect()
        torch.cuda.empty_cache()


def select_candidates(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for arm in SOURCE_ARMS:
        arm_rows = [row for row in records if row.get("arm") == arm]
        if not arm_rows:
            continue
        selected[arm] = {}
        curves[arm] = {}
        for length in SPEC.eval_lengths:
            rows = [row for row in arm_rows if int(row["length"]) == length]
            selected[arm][str(length)] = {}
            curves[arm][str(length)] = {}
            for family in PROFILE_FAMILIES:
                family_rows = [row for row in rows if row["family"] == family]
                if not family_rows:
                    raise ValueError(f"missing {arm}/L={length}/{family} selection rows")
                family_rows.sort(
                    key=lambda row: (
                        float(row["mean_tail_nll"]),
                        abs(float(row["lambda_ratio"])),
                        float(row["base_multiplier"]),
                    )
                )
                best = family_rows[0]
                selected[arm][str(length)][family] = {
                    key: best[key]
                    for key in (
                        "family",
                        "base_multiplier",
                        "lambda_ratio",
                        "strictly_monotonic",
                        "inv_freq_sha256",
                        "mean_tail_nll",
                    )
                }
                if family.endswith("_residual"):
                    per_lambda = {}
                    for ratio in SPEC.lambda_ratios:
                        candidates = [
                            row
                            for row in family_rows
                            if float(row["lambda_ratio"]) == ratio
                        ]
                        if candidates:
                            winner = min(candidates, key=lambda row: row["mean_tail_nll"])
                            per_lambda[str(ratio)] = {
                                "base_multiplier": winner["base_multiplier"],
                                "mean_tail_nll": winner["mean_tail_nll"],
                            }
                    curves[arm][str(length)][family] = per_lambda
    return {"selected": selected, "per_lambda_reprofiled": curves}


def select(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    ready = _load_ready(output)
    records = _read_jsonl(output / "profile" / "selection.jsonl")
    result = {
        "status": "PASS",
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "selection_anchor_sha256": ready["anchors"]["selection"]["sha256"],
        "test_not_read": True,
        **select_candidates(records),
    }
    _atomic_json(output / "selection.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _bootstrap_ci(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SPEC.bootstrap_seed)
    indices = rng.integers(0, len(array), size=(SPEC.bootstrap_samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    ready = _load_ready(output)
    selection = json.loads((output / "selection.json").read_text())
    records = _read_jsonl(output / "profile" / "test.jsonl")
    summary: dict[str, Any] = {}
    for arm in selection["selected"]:
        summary[arm] = {}
        for length in SPEC.eval_lengths:
            rows = [
                row for row in records if row["arm"] == arm and int(row["length"]) == length
            ]
            by_family = {row["family"]: row for row in rows}
            if set(by_family) != set(PROFILE_FAMILIES):
                raise ValueError(f"incomplete test matrix for {arm}/L={length}")
            baseline = by_family["base"]
            result = {}
            for family, row in by_family.items():
                differences = [
                    left - right
                    for left, right in zip(row["tail_nll"], baseline["tail_nll"], strict=True)
                ]
                result[family] = {
                    "family": family,
                    "mean_tail_nll": row["mean_tail_nll"],
                    "tail_ppl": math.exp(row["mean_tail_nll"]),
                    "base_multiplier": row["base_multiplier"],
                    "lambda_ratio": row["lambda_ratio"],
                    "strictly_monotonic": row["strictly_monotonic"],
                    "delta_vs_profiled_base": statistics.fmean(differences),
                    "paired_anchor_bootstrap_95ci": _bootstrap_ci(differences),
                }
            summary[arm][str(length)] = result

    gate_lengths = (4_096, 8_192)
    gate_rows = summary[GATE_ARM]
    winners = {}
    conditions = []
    for length in gate_lengths:
        candidates = [gate_rows[str(length)][name] for name in ("cosh_residual", "band_residual")]
        winner = min(candidates, key=lambda row: row["mean_tail_nll"])
        winners[str(length)] = winner
        raw_delta = gate_rows[str(length)]["raw_evq"]["delta_vs_profiled_base"]
        raw_gain = max(0.0, -raw_delta)
        residual_gain = max(0.0, -winner["delta_vs_profiled_base"])
        winner["retained_raw_deformation_gain_fraction"] = (
            residual_gain / raw_gain if raw_gain > 0.0 else None
        )
        conditions.extend(
            [
                winner["lambda_ratio"] not in (0.0, min(SPEC.lambda_ratios), max(SPEC.lambda_ratios)),
                winner["delta_vs_profiled_base"] < 0.0,
                winner["paired_anchor_bootstrap_95ci"][1] < 0.0,
            ]
        )
        if raw_gain > 0.0:
            conditions.append(residual_gain / raw_gain >= 0.25)
    in_domain_winner = min(
        (
            gate_rows[str(SOURCE_SPEC.train_length)][name]
            for name in ("cosh_residual", "band_residual")
        ),
        key=lambda row: row["mean_tail_nll"],
    )
    long_gain = -statistics.fmean(
        winner["delta_vs_profiled_base"] for winner in winners.values()
    )
    conditions.append(
        in_domain_winner["delta_vs_profiled_base"] <= max(0.02, long_gain)
    )
    proceed = all(conditions)
    gate = {
        "decision": "PROCEED_TO_MATCHED_TRAINING" if proceed else "STOP_RESIDUAL_TRACK",
        "proceed": proceed,
        "gate_arm": GATE_ARM,
        "long_length_winners": winners,
        "in_domain_best_residual": in_domain_winner,
        "conditions_passed": sum(bool(value) for value in conditions),
        "conditions_total": len(conditions),
        "claim_boundary": (
            "single-seed screening gate; bootstrap units are held-out windows, "
            "not independent training seeds"
        ),
    }
    result = {
        "status": "PASS",
        "protocol_sha256": SPEC.fingerprint(),
        "code_sha256": code_fingerprint(),
        "test_anchor_sha256": ready["anchors"]["test"]["sha256"],
        "summary": summary,
        "gate": gate,
    }
    _atomic_json(output / "summary.json", result)
    lines = [
        "# Profiled residual diagnostic",
        "",
        f"Decision: **{gate['decision']}**.",
        "",
        "Negative deltas favor the residual after re-profiling target base.",
        "",
        "| checkpoint | length | family | lambda | c | delta NLL | 95% window CI |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for arm, lengths in summary.items():
        for length, families in lengths.items():
            for family in ("raw_evq", "cosh_residual", "band_residual"):
                row = families[family]
                lines.append(
                    f"| {arm} | {length} | {family} | {row['lambda_ratio']:+.1f} | "
                    f"{row['base_multiplier']:.1f} | {row['delta_vs_profiled_base']:+.5f} | "
                    f"[{row['paired_anchor_bootstrap_95ci'][0]:+.5f}, "
                    f"{row['paired_anchor_bootstrap_95ci'][1]:+.5f}] |"
                )
    lines.extend(
        [
            "",
            "Single-seed checkpoint-only supporting evidence. Window bootstrap is not a seed-level CI.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(gate, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--source-work-dir", type=Path, required=True)
    pre.add_argument("--source-data-manifest", type=Path, required=True)
    pre.add_argument("--anchor-manifest", type=Path, required=True)
    pre.add_argument("--output-dir", type=Path, required=True)

    run = sub.add_parser("profile")
    run.add_argument("--split", choices=("selection", "test"), required=True)
    run.add_argument("--arms", nargs="*", choices=SOURCE_ARMS)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--max-batch-tokens", type=int, default=8_192)

    choose = sub.add_parser("select")
    choose.add_argument("--output-dir", type=Path, required=True)
    report = sub.add_parser("summarize")
    report.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "profile":
        profile(args)
    elif args.command == "select":
        select(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
