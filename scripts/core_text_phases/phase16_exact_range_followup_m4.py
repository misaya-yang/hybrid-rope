#!/usr/bin/env python3
"""Post-registered follow-up for the running M4 exact-range factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = (
    REPO_ROOT
    / "results"
    / "theory"
    / "phase16_exact_range_factorial_m4_20260724"
)
TRACKED_ARMS = {
    "native_geo",
    "anchored_cosh_rule",
    "anchored_exp_rule",
}
FINAL_EVAL_STREAMS = ("validation", "test")
FINAL_ANCHORS = 4
MILESTONE_ANCHORS = 2
TAIL_TOKENS = 128
EVAL_PROTOCOL = {
    "registered_at": "2026-07-25",
    "final_checkpoint_conditions": (
        "own interior shape under fixed training range and target-matched "
        "range with runtime base equal to target window T"
    ),
    "cross_swap": (
        "Geo-trained and formula-Cosh-trained checkpoints x Geo/Cosh runtime "
        "interior grids, under both registered range conditions"
    ),
    "length_ratios": [1, 2, 4, 8],
    "streams": list(FINAL_EVAL_STREAMS),
    "anchors_per_stream_length": FINAL_ANCHORS,
    "anchor_seed_rule": "20260725 + 100000*stream_index + window_length",
    "position_bins": "four contiguous equal-count bins over L-1 prediction positions",
    "final_tail": f"last min({TAIL_TOKENS}, L-1) prediction positions",
    "milestone_eval": {
        "arms": sorted(TRACKED_ARMS),
        "fractions": [25, 50, 75, 100],
        "range": "fixed training range",
        "stream": "validation",
        "length_ratios": [2, 4, 8],
        "anchors": MILESTONE_ANCHORS,
    },
    "aggregation": (
        "average anchors within each checkpoint/condition first; pair the "
        "three seeds within each of 12 structural configs; then average the "
        "12 config means equally. Streams and length ratios remain separate "
        "conditions and never increase n."
    ),
    "bootstrap": "10000 config-level resamples, seed 20260725",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def capture_milestones(root: Path) -> None:
    state_path = root / "analysis" / "milestone_capture_state.json"
    state = read_json(state_path)
    state.setdefault("runs", {})
    while True:
        completed = len(list((root / "runs").glob("*/result.json")))
        for spec_path in (root / "runs").glob("*/spec.json"):
            spec = read_json(spec_path)
            if spec.get("arm") not in TRACKED_ARMS:
                continue
            run_dir = spec_path.parent
            if (run_dir / "result.json").exists():
                continue
            checkpoint = run_dir / "checkpoint_last.pt"
            progress = read_json(run_dir / "progress.json")
            if not checkpoint.exists() or int(progress.get("step", 0)) < 24:
                continue
            record = state["runs"].setdefault(run_dir.name, {"inodes": []})
            inode = checkpoint.stat().st_ino
            if inode in record["inodes"]:
                continue
            slot = len(record["inodes"])
            if slot >= 3:
                continue
            milestone = (25, 50, 75)[slot]
            destination = run_dir / f"checkpoint_{milestone:03d}.pt"
            if not destination.exists():
                os.link(checkpoint, destination)
            record["inodes"].append(inode)
            record[f"checkpoint_{milestone:03d}"] = {
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_inode": inode,
                "progress_step_observed": progress.get("step"),
            }
            write_json(state_path, state)
            print(f"[capture] {run_dir.name} {milestone}%", flush=True)

        if completed >= 180:
            state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            write_json(state_path, state)
            return
        lock = read_json(root / "sweep.lock")
        pid = int(lock.get("pid", -1))
        if pid <= 0:
            raise RuntimeError(f"main sweep stopped at {completed}/180")
        try:
            os.kill(pid, 0)
        except OSError as exc:
            raise RuntimeError(f"main sweep pid {pid} stopped at {completed}/180") from exc
        time.sleep(5)


def experiment_modules():
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import phase16_exact_range_factorial_m4 as experiment
    import phase16_formula_optimality_sweep as harness

    return harness, experiment


def write_preregistration(root: Path) -> None:
    write_json(root / "analysis" / "evaluation_protocol.json", EVAL_PROTOCOL)


def frequency_audit(root: Path) -> list[dict[str, Any]]:
    _, experiment = experiment_modules()
    specs = experiment.build_specs(8_388_608, FINAL_ANCHORS)
    receipts = experiment.validate_schedules(specs)
    if len(receipts) != 12:
        raise AssertionError(f"expected 12 frequency receipts, got {len(receipts)}")
    if max(float(row["absolute_matching_error"]) for row in receipts) > 1e-12:
        raise AssertionError("matched exponential exceeds frozen RMS tolerance")
    write_json(
        root / "analysis" / "frequency_audit.json",
        {
            "definition": (
                "normalized log-frequency node RMS displacement from uniform"
            ),
            "tolerance": 1e-12,
            "receipts": receipts,
        },
    )
    return receipts


def model_state_sha256(path: Path) -> str:
    harness, _ = experiment_modules()
    state = harness.torch.load(path, map_location="cpu")
    digest = hashlib.sha256()
    for key, tensor in sorted(state["model"].items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    del state
    return digest.hexdigest()


def build_context(root: Path):
    harness, experiment = experiment_modules()
    args = SimpleNamespace(
        work_root=root,
        train_tokens=8_388_608,
        val_tokens=1_048_576,
        eval_chunks=FINAL_ANCHORS,
    )
    return harness, experiment, experiment.build_context(args)


def run_extreme_arms(root: Path) -> list[Any]:
    harness, experiment, ctx = build_context(root)
    specs = experiment.build_extreme_specs(8_388_608, FINAL_ANCHORS)
    write_json(
        root / "analysis" / "extreme_plan.json",
        {
            "reason": "formula-tau minimum and maximum at canonical base 500K",
            "runs": len(specs),
            "specs": [asdict(spec) for spec in specs],
        },
    )
    with harness.SweepLock(root / "augment_train.lock"):
        experiment.run_specs(ctx, specs)
    return specs


def replay_missing_milestones(root: Path) -> dict[str, Any]:
    harness, experiment = experiment_modules()
    specs = [
        spec
        for spec in experiment.build_specs(8_388_608, FINAL_ANCHORS)
        if spec.arm in TRACKED_ARMS
    ]
    missing = [
        spec
        for spec in specs
        if any(
            not (root / "runs" / spec.run_id / f"checkpoint_{fraction:03d}.pt").exists()
            for fraction in (25, 50, 75)
        )
    ]
    receipt: dict[str, Any] = {
        "missing_runs_before_replay": len(missing),
        "runs": [],
    }
    if not missing:
        write_json(root / "analysis" / "milestone_replay_receipt.json", receipt)
        return receipt

    replay_root = root / "milestone_replays"
    _, _, ctx = build_context(replay_root)
    with harness.SweepLock(replay_root / "sweep.lock"):
        for spec in missing:
            replay_spec = replace(spec, stage="milestone_replay", save_milestones=True)
            experiment.run_specs(ctx, [replay_spec])
            original_dir = root / "runs" / spec.run_id
            replay_dir = replay_root / "runs" / spec.run_id
            original_hash = model_state_sha256(original_dir / "checkpoint_last.pt")
            replay_hash = model_state_sha256(replay_dir / "checkpoint_last.pt")
            if original_hash != replay_hash:
                raise RuntimeError(f"deterministic replay mismatch for {spec.run_id}")
            promoted = []
            for fraction in (25, 50, 75):
                destination = original_dir / f"checkpoint_{fraction:03d}.pt"
                source = replay_dir / f"checkpoint_{fraction:03d}.pt"
                if not destination.exists():
                    os.link(source, destination)
                    promoted.append(fraction)
            receipt["runs"].append(
                {
                    "run_id": spec.run_id,
                    "model_state_sha256": original_hash,
                    "promoted_fractions": promoted,
                }
            )
            write_json(root / "analysis" / "milestone_replay_receipt.json", receipt)
    return receipt


def evaluation_streams(max_tokens: int = 1_048_576):
    harness, _ = experiment_modules()
    tokenizer = harness.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    return {
        split: harness.load_local_wikitext_tokens(tokenizer, split, max_tokens)
        for split in FINAL_EVAL_STREAMS
    }


def build_anchor_manifest(
    root: Path, streams: dict[str, Any], lengths: list[int]
) -> dict[str, Any]:
    manifest_path = root / "analysis" / "evaluation_anchors.json"
    existing = read_json(manifest_path)
    if existing:
        return existing
    manifest: dict[str, Any] = {
        "protocol": EVAL_PROTOCOL,
        "streams": {},
    }
    for stream_index, (split, tokens) in enumerate(streams.items()):
        stream_record = {
            "tokens": int(tokens.numel()),
            "lengths": {},
        }
        for length in lengths:
            rng = random.Random(20_260_725 + 100_000 * stream_index + length)
            max_start = int(tokens.numel()) - length
            starts = sorted(rng.sample(range(max_start), FINAL_ANCHORS))
            hashes = []
            for start in starts:
                window = tokens[start : start + length].contiguous().numpy()
                hashes.append(hashlib.sha256(window.tobytes()).hexdigest())
            stream_record["lengths"][str(length)] = {
                "starts": starts,
                "window_token_sha256": hashes,
            }
        manifest["streams"][split] = stream_record
    write_json(manifest_path, manifest)
    return manifest


def runtime_inv_freq(spec, runtime_shape: str, range_condition: str, length: int):
    harness, experiment = experiment_modules()
    if runtime_shape == "geo":
        grid = experiment.normalized_native_grid(spec.head_dim)
    elif runtime_shape == "cosh_rule":
        grid = experiment.normalized_cosh_grid(spec.head_dim, spec.theory_tau)
    elif runtime_shape == "own":
        grid, _ = experiment.schedule_grid(spec)
    else:
        raise ValueError(f"unknown runtime shape {runtime_shape}")
    base = spec.rope_base if range_condition == "fixed_training_range" else float(length)
    channels = spec.head_dim // 2
    log_span = (channels - 1) / channels * math.log(base)
    return harness.torch.exp(-log_span * grid).float()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def load_jsonl_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        keys.add(
            (
                row["run_id"],
                row["fraction"],
                row["range_condition"],
                row["runtime_shape"],
                row["stream"],
                row["length"],
                row["anchor_index"],
            )
        )
    return keys


def load_model(spec, checkpoint: Path):
    harness, experiment = experiment_modules()
    cfg = harness.build_cfg(harness.TIER_CONFIGS["50m"], spec.seq_len, spec.num_heads)
    model = harness.GPT(cfg, experiment.build_inv_freq(spec))
    state = harness.torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    del state
    return model.to(harness.DEVICE).eval()


def evaluate_checkpoint(
    output: Path,
    seen: set[tuple],
    model,
    spec,
    fraction: int,
    streams: dict[str, Any],
    anchors: dict[str, Any],
    *,
    range_conditions: tuple[str, ...],
    runtime_shapes: tuple[str, ...],
    length_ratios: tuple[int, ...],
    stream_names: tuple[str, ...],
    anchor_count: int,
) -> None:
    harness, _ = experiment_modules()
    torch = harness.torch
    for range_condition in range_conditions:
        for runtime_shape in runtime_shapes:
            for ratio in length_ratios:
                length = spec.seq_len * ratio
                inv = runtime_inv_freq(spec, runtime_shape, range_condition, length)
                rope = model.blocks[0].attn.rope
                rope.inv_freq.copy_(inv.to(rope.inv_freq.device))
                model.extend_rope(length + 64)
                for split in stream_names:
                    starts = anchors["streams"][split]["lengths"][str(length)]["starts"]
                    for anchor_index, start in enumerate(starts[:anchor_count]):
                        key = (
                            spec.run_id,
                            fraction,
                            range_condition,
                            runtime_shape,
                            split,
                            length,
                            anchor_index,
                        )
                        if key in seen:
                            continue
                        window = streams[split][start : start + length].unsqueeze(0)
                        inputs = window[:, :-1].to(harness.DEVICE)
                        targets = window[:, 1:].to(harness.DEVICE)
                        with torch.inference_mode():
                            logits = model(inputs)
                            token_nll = harness.F.cross_entropy(
                                logits.reshape(-1, logits.size(-1)),
                                targets.reshape(-1),
                                reduction="none",
                            )
                        bins = [
                            float(chunk.mean().cpu())
                            for chunk in torch.tensor_split(token_nll, 4)
                        ]
                        tail = min(TAIL_TOKENS, token_nll.numel())
                        row = {
                            "run_id": spec.run_id,
                            "fraction": fraction,
                            "train_arm": spec.arm,
                            "rope_base": spec.rope_base,
                            "train_length": spec.seq_len,
                            "head_dim": spec.head_dim,
                            "seed": spec.seed,
                            "range_condition": range_condition,
                            "runtime_shape": runtime_shape,
                            "stream": split,
                            "length": length,
                            "length_ratio": ratio,
                            "anchor_index": anchor_index,
                            "full_nll": float(token_nll.mean().cpu()),
                            "position_bin_nll": bins,
                            "final_tail_nll": float(token_nll[-tail:].mean().cpu()),
                            "tail_tokens": tail,
                            "runtime_inv_freq_sha256": hashlib.sha256(
                                inv.numpy().tobytes()
                            ).hexdigest(),
                        }
                        if not all(
                            math.isfinite(value)
                            for value in [row["full_nll"], row["final_tail_nll"], *bins]
                        ):
                            raise RuntimeError(f"non-finite NLL for {key}")
                        append_jsonl(output, row)
                        seen.add(key)
                        del inputs, targets, logits, token_nll
                        harness.maybe_clear_device_cache()


def evaluate_final_checkpoints(root: Path, extreme_specs: list[Any]) -> None:
    harness, experiment = experiment_modules()
    specs = experiment.build_specs(8_388_608, FINAL_ANCHORS) + extreme_specs
    lengths = sorted({spec.seq_len * ratio for spec in specs for ratio in (1, 2, 4, 8)})
    streams = evaluation_streams()
    anchors = build_anchor_manifest(root, streams, lengths)
    output = root / "analysis" / "final_eval.jsonl"
    seen = load_jsonl_keys(output)
    for index, spec in enumerate(specs, 1):
        checkpoint = root / "runs" / spec.run_id / "checkpoint_last.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = load_model(spec, checkpoint)
        runtime_shapes = ("own",)
        if spec.arm == "native_geo":
            runtime_shapes = ("geo", "cosh_rule")
        elif spec.arm == "anchored_cosh_rule":
            runtime_shapes = ("cosh_rule", "geo")
        evaluate_checkpoint(
            output,
            seen,
            model,
            spec,
            100,
            streams,
            anchors,
            range_conditions=("fixed_training_range", "target_matched_range"),
            runtime_shapes=runtime_shapes,
            length_ratios=(1, 2, 4, 8),
            stream_names=FINAL_EVAL_STREAMS,
            anchor_count=FINAL_ANCHORS,
        )
        print(f"[final-eval] {index}/{len(specs)} {spec.run_id}", flush=True)
        del model
        harness.maybe_clear_device_cache()


def evaluate_milestones(root: Path) -> None:
    harness, experiment = experiment_modules()
    specs = [
        spec
        for spec in experiment.build_specs(8_388_608, FINAL_ANCHORS)
        if spec.arm in TRACKED_ARMS
    ]
    lengths = sorted({spec.seq_len * ratio for spec in specs for ratio in (2, 4, 8)})
    streams = evaluation_streams()
    anchors = build_anchor_manifest(
        root,
        streams,
        sorted({256, 512, 1024, 2048, 4096, 8192}),
    )
    output = root / "analysis" / "milestone_eval.jsonl"
    seen = load_jsonl_keys(output)
    for index, spec in enumerate(specs, 1):
        for fraction in (25, 50, 75):
            checkpoint = root / "runs" / spec.run_id / f"checkpoint_{fraction:03d}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model = load_model(spec, checkpoint)
            evaluate_checkpoint(
                output,
                seen,
                model,
                spec,
                fraction,
                streams,
                anchors,
                range_conditions=("fixed_training_range",),
                runtime_shapes=("own",),
                length_ratios=(2, 4, 8),
                stream_names=("validation",),
                anchor_count=MILESTONE_ANCHORS,
            )
            del model
            harness.maybe_clear_device_cache()
        print(f"[milestone-eval] {index}/{len(specs)} {spec.run_id}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def mean_rows(
    rows: list[dict[str, Any]], metric: str = "full_nll"
) -> dict[tuple, float]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["rope_base"],
            row["train_length"],
            row["head_dim"],
            row["seed"],
            row["train_arm"],
            row["range_condition"],
            row["runtime_shape"],
            row["stream"],
            row["length_ratio"],
            row["fraction"],
        )
        value = (
            row["position_bin_nll"][int(metric.rsplit("_", 1)[1]) - 1]
            if metric.startswith("position_bin_")
            else row[metric]
        )
        grouped[key].append(float(value))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def config_bootstrap(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = random.Random(20_260_725)
    draws = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(10_000)
    )
    return {
        "mean": statistics.fmean(values),
        "ci_low": draws[249],
        "ci_high": draws[9749],
    }


def analyze(root: Path) -> dict[str, Any]:
    final_rows = read_jsonl(root / "analysis" / "final_eval.jsonl")
    metrics = (
        "full_nll",
        "final_tail_nll",
        "position_bin_1",
        "position_bin_2",
        "position_bin_3",
        "position_bin_4",
    )
    metric_contrasts = []
    final_by_metric = {metric: mean_rows(final_rows, metric) for metric in metrics}
    for metric, final in final_by_metric.items():
        for left_arm, left_shape, contrast_name in (
            ("anchored_cosh_rule", "cosh_rule", "formula_cosh_minus_geo"),
            ("anchored_exp_rule", "own", "matched_exp_minus_geo"),
        ):
            for condition in ("fixed_training_range", "target_matched_range"):
                for stream in FINAL_EVAL_STREAMS:
                    for ratio in (1, 2, 4, 8):
                        config_values = []
                        for base in (500_000.0, 1_000_000.0):
                            for train_length in (256, 1024):
                                for head_dim in (128, 64, 32):
                                    seed_values = []
                                    for seed in (42, 137, 256):
                                        prefix = (base, train_length, head_dim, seed)
                                        geo = final[
                                            prefix
                                            + (
                                                "native_geo",
                                                condition,
                                                "geo",
                                                stream,
                                                ratio,
                                                100,
                                            )
                                        ]
                                        left = final[
                                            prefix
                                            + (
                                                left_arm,
                                                condition,
                                                left_shape,
                                                stream,
                                                ratio,
                                                100,
                                            )
                                        ]
                                        seed_values.append(left - geo)
                                    config_values.append(statistics.fmean(seed_values))
                        metric_contrasts.append(
                            {
                                "contrast": contrast_name,
                                "metric": metric,
                                "range_condition": condition,
                                "stream": stream,
                                "length_ratio": ratio,
                                "n_structural_configs": len(config_values),
                                "seeds_per_config": 3,
                                **config_bootstrap(config_values),
                            }
                        )

    final = final_by_metric["full_nll"]
    coadaptation = []
    for condition in ("fixed_training_range", "target_matched_range"):
        for stream in FINAL_EVAL_STREAMS:
            for ratio in (1, 2, 4, 8):
                config_values = []
                for base in (500_000.0, 1_000_000.0):
                    for train_length in (256, 1024):
                        for head_dim in (128, 64, 32):
                            seed_values = []
                            for seed in (42, 137, 256):
                                prefix = (base, train_length, head_dim, seed)
                                geo_geo = final[
                                    prefix + ("native_geo", condition, "geo", stream, ratio, 100)
                                ]
                                geo_cosh = final[
                                    prefix
                                    + ("native_geo", condition, "cosh_rule", stream, ratio, 100)
                                ]
                                cosh_geo = final[
                                    prefix
                                    + (
                                        "anchored_cosh_rule",
                                        condition,
                                        "geo",
                                        stream,
                                        ratio,
                                        100,
                                    )
                                ]
                                cosh_cosh = final[
                                    prefix
                                    + (
                                        "anchored_cosh_rule",
                                        condition,
                                        "cosh_rule",
                                        stream,
                                        ratio,
                                        100,
                                    )
                                ]
                                seed_values.append(
                                    0.5 * ((geo_cosh - geo_geo) + (cosh_geo - cosh_cosh))
                                )
                            config_values.append(statistics.fmean(seed_values))
                coadaptation.append(
                    {
                        "metric": "mean_wrong_runtime_shape_penalty",
                        "range_condition": condition,
                        "stream": stream,
                        "length_ratio": ratio,
                        "n_structural_configs": len(config_values),
                        "seeds_per_config": 3,
                        **config_bootstrap(config_values),
                    }
                )

    extreme_curves = []
    arm_multiplier = {
        "anchored_cosh_m050": 0.5,
        "anchored_cosh_m075": 0.75,
        "anchored_cosh_rule": 1.0,
        "anchored_cosh_m125": 1.25,
        "anchored_cosh_m150": 1.5,
    }
    for train_length, head_dim in ((1024, 32), (256, 128)):
        for condition in ("fixed_training_range", "target_matched_range"):
            for stream in FINAL_EVAL_STREAMS:
                for ratio in (1, 2, 4, 8):
                    curve = {}
                    for arm, multiplier in arm_multiplier.items():
                        values = []
                        for seed in (42, 137, 256):
                            runtime_shape = "cosh_rule" if arm == "anchored_cosh_rule" else "own"
                            values.append(
                                final[
                                    (
                                        500_000.0,
                                        train_length,
                                        head_dim,
                                        seed,
                                        arm,
                                        condition,
                                        runtime_shape,
                                        stream,
                                        ratio,
                                        100,
                                    )
                                ]
                            )
                        curve[str(multiplier)] = statistics.fmean(values)
                    extreme_curves.append(
                        {
                            "train_length": train_length,
                            "head_dim": head_dim,
                            "rule_tau": head_dim / math.sqrt(train_length),
                            "range_condition": condition,
                            "stream": stream,
                            "length_ratio": ratio,
                            "seed_mean_full_nll_by_multiplier": curve,
                            "winner_multiplier": min(curve, key=curve.get),
                            "seeds": 3,
                        }
                    )

    milestone_rows = read_jsonl(root / "analysis" / "milestone_eval.jsonl")
    final_milestone_rows = [
        {**row, "fraction": 100}
        for row in final_rows
        if row["train_arm"] in TRACKED_ARMS
        and row["range_condition"] == "fixed_training_range"
        and row["stream"] == "validation"
        and row["length_ratio"] in (2, 4, 8)
        and row["anchor_index"] < MILESTONE_ANCHORS
        and (
            (row["train_arm"] == "native_geo" and row["runtime_shape"] == "geo")
            or (
                row["train_arm"] == "anchored_cosh_rule"
                and row["runtime_shape"] == "cosh_rule"
            )
            or (
                row["train_arm"] == "anchored_exp_rule"
                and row["runtime_shape"] == "own"
            )
        )
    ]
    milestone_means = mean_rows(milestone_rows + final_milestone_rows)
    milestone_rankings = []
    for fraction in (25, 50, 75, 100):
        for ratio in (2, 4, 8):
            winner_counts = {arm: 0 for arm in TRACKED_ARMS}
            config_rows = []
            for base in (500_000.0, 1_000_000.0):
                for train_length in (256, 1024):
                    for head_dim in (128, 64, 32):
                        arm_values = {}
                        for arm in TRACKED_ARMS:
                            runtime_shape = (
                                "own"
                                if fraction < 100
                                else "geo"
                                if arm == "native_geo"
                                else "cosh_rule"
                                if arm == "anchored_cosh_rule"
                                else "own"
                            )
                            arm_values[arm] = statistics.fmean(
                                milestone_means[
                                    (
                                        base,
                                        train_length,
                                        head_dim,
                                        seed,
                                        arm,
                                        "fixed_training_range",
                                        runtime_shape,
                                        "validation",
                                        ratio,
                                        fraction,
                                    )
                                ]
                                for seed in (42, 137, 256)
                            )
                        winner = min(arm_values, key=arm_values.get)
                        winner_counts[winner] += 1
                        config_rows.append(
                            {
                                "rope_base": base,
                                "train_length": train_length,
                                "head_dim": head_dim,
                                "seed_mean_nll": arm_values,
                                "winner": winner,
                            }
                        )
            milestone_rankings.append(
                {
                    "fraction": fraction,
                    "length_ratio": ratio,
                    "n_structural_configs": 12,
                    "seeds_per_config": 3,
                    "winner_counts": winner_counts,
                    "configs": config_rows,
                }
            )

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "statistical_unit": (
            "12 structural configs; three paired seeds within config; "
            "lengths/anchors/runs are not independent samples"
        ),
        "final_metric_contrasts": metric_contrasts,
        "cross_swap_coadaptation": coadaptation,
        "extreme_runs": 12,
        "extreme_tau_curves": extreme_curves,
        "milestone_rows": len(milestone_rows),
        "milestone_rankings": milestone_rankings,
    }
    write_json(root / "analysis" / "summary.json", summary)
    lines = [
        "# M4 exact-range follow-up",
        "",
        "- Evidence tier: supporting/mechanistic, 50M local WikiText.",
        "- Statistical unit: 12 structural configurations, each paired over three seeds.",
        "- Anchors, lengths, and 180 training runs are not treated as independent samples.",
        "",
        "## Formula Cosh minus Geo full-NLL",
        "",
        "| Range | Stream | Ratio | Mean | 95% config bootstrap CI |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metric_contrasts:
        if row["metric"] not in {"full_nll", "final_tail_nll"}:
            continue
        lines.append(
            f"| {row['contrast']} {row['metric']} / {row['range_condition']} "
            f"| {row['stream']} | {row['length_ratio']}x "
            f"| {row['mean']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Cross-swap co-adaptation",
            "",
            "Positive values mean that the wrong runtime interior grid is worse on average.",
            "",
            "| Range | Stream | Ratio | Mean wrong-shape penalty | 95% config bootstrap CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in coadaptation:
        lines.append(
            f"| {row['range_condition']} | {row['stream']} | {row['length_ratio']}x "
            f"| {row['mean']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "Position-bin 1--4 contrasts, extreme tau curves, and milestone rankings",
            "are frozen in `summary.json`; they use the same config-level aggregation.",
        ]
    )
    (root / "analysis" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def augment(root: Path) -> None:
    write_preregistration(root)
    frequency_audit(root)
    extreme_specs = run_extreme_arms(root)
    replay_missing_milestones(root)
    evaluate_final_checkpoints(root, extreme_specs)
    evaluate_milestones(root)
    print(json.dumps(analyze(root), indent=2), flush=True)


def self_test(root: Path) -> None:
    _, experiment = experiment_modules()
    main_specs = experiment.build_specs(8_388_608, FINAL_ANCHORS)
    extreme_specs = experiment.build_extreme_specs(8_388_608, FINAL_ANCHORS)
    assert len(main_specs) == 180
    assert len(extreme_specs) == 12
    assert {spec.tau for spec in extreme_specs if spec.tau_multiplier == 0.5} == {0.5, 4.0}
    assert {spec.tau for spec in extreme_specs if spec.tau_multiplier == 1.5} == {1.5, 12.0}
    write_preregistration(root)
    frequency_audit(root)
    with tempfile.TemporaryDirectory() as directory:
        test_root = Path(directory)
        final_path = test_root / "analysis" / "final_eval.jsonl"
        for spec in main_specs + extreme_specs:
            shapes = (
                ("geo", "cosh_rule")
                if spec.arm == "native_geo"
                else ("cosh_rule", "geo")
                if spec.arm == "anchored_cosh_rule"
                else ("own",)
            )
            for condition in ("fixed_training_range", "target_matched_range"):
                for shape in shapes:
                    for split in FINAL_EVAL_STREAMS:
                        for ratio in (1, 2, 4, 8):
                            for anchor in range(FINAL_ANCHORS):
                                append_jsonl(
                                    final_path,
                                    {
                                        "run_id": spec.run_id,
                                        "fraction": 100,
                                        "train_arm": spec.arm,
                                        "rope_base": spec.rope_base,
                                        "train_length": spec.seq_len,
                                        "head_dim": spec.head_dim,
                                        "seed": spec.seed,
                                        "range_condition": condition,
                                        "runtime_shape": shape,
                                        "stream": split,
                                        "length": spec.seq_len * ratio,
                                        "length_ratio": ratio,
                                        "anchor_index": anchor,
                                        "full_nll": 5.0,
                                        "position_bin_nll": [5.0] * 4,
                                        "final_tail_nll": 5.0,
                                    },
                                )
        milestone_path = test_root / "analysis" / "milestone_eval.jsonl"
        for spec in main_specs:
            if spec.arm not in TRACKED_ARMS:
                continue
            for fraction in (25, 50, 75):
                for ratio in (2, 4, 8):
                    for anchor in range(MILESTONE_ANCHORS):
                        append_jsonl(
                            milestone_path,
                            {
                                "run_id": spec.run_id,
                                "fraction": fraction,
                                "train_arm": spec.arm,
                                "rope_base": spec.rope_base,
                                "train_length": spec.seq_len,
                                "head_dim": spec.head_dim,
                                "seed": spec.seed,
                                "range_condition": "fixed_training_range",
                                "runtime_shape": "own",
                                "stream": "validation",
                                "length": spec.seq_len * ratio,
                                "length_ratio": ratio,
                                "anchor_index": anchor,
                                "full_nll": 5.0,
                                "position_bin_nll": [5.0] * 4,
                                "final_tail_nll": 5.0,
                            },
                        )
        summary = analyze(test_root)
        assert len(summary["final_metric_contrasts"]) == 192
        assert len(summary["cross_swap_coadaptation"]) == 16
        assert len(summary["milestone_rankings"]) == 12
    print("[self-test] schedules, RMS audit, frozen aggregation, and report: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("capture", "audit", "self-test", "augment", "report"),
        required=True,
    )
    parser.add_argument("--work-root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.work_root.expanduser().resolve()
    if args.mode == "capture":
        capture_milestones(root)
    elif args.mode == "audit":
        write_preregistration(root)
        frequency_audit(root)
    elif args.mode == "self-test":
        self_test(root)
    elif args.mode == "augment":
        augment(root)
    else:
        print(json.dumps(analyze(root), indent=2))


if __name__ == "__main__":
    main()
