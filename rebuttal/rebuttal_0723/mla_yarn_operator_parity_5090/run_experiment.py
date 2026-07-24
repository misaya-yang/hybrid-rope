#!/usr/bin/env python3
"""Evaluate, gate, and summarize the MLA shared-operator factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.rebuttal_0723.mla_scarcity_5090.protocol import (
    SPEC as BASE_SPEC,
    training_inv_freq,
)
from rebuttal.rebuttal_0723.mla_scarcity_5090.run_experiment import (
    _load_checkpoint,
    _run_dir,
    _set_runtime_rope,
    code_fingerprint as base_code_fingerprint,
    model_inv_freq,
    per_sequence_nll,
    tensor_sha256,
    validate_cuda_runtime,
    validate_ready as validate_base_ready,
)
from rebuttal.rebuttal_0723.mla_yarn_operator_parity_5090.prepare import (
    sha256_file,
    validate_manifest,
)
from rebuttal.rebuttal_0723.mla_yarn_operator_parity_5090.protocol import (
    BASE_TRAINING_PROTOCOL_SHA256,
    FREQUENCY_PAIRS,
    GATE_SEED,
    OPERATORS,
    PRIMARY_LENGTHS,
    SEEDS,
    SPEC,
    TRAINING_ARMS,
    operators_for_stage,
)
from scripts.lib.rope.official_yarn import (
    native_endpoint_inv_freq,
    official_yarn_on_inv_freq,
    official_yarn_on_native_grid,
    shared_index_yarn_control_on_inv_freq,
    yarn_mscale,
)


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
PARITY_OPERATORS = tuple(item for item in OPERATORS if item != "raw")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _max_fp32_ulp_distance(
    left: torch.Tensor, right: torch.Tensor
) -> int:
    left_fp32 = left.detach().cpu().float().contiguous()
    right_fp32 = right.detach().cpu().float().contiguous()
    if (
        left_fp32.shape != right_fp32.shape
        or not torch.isfinite(left_fp32).all()
        or not torch.isfinite(right_fp32).all()
    ):
        raise ValueError("ULP comparison requires same-shape finite tensors")
    a = left_fp32.view(torch.int32)
    b = right_fp32.view(torch.int32)
    return int((a.to(torch.int64) - b.to(torch.int64)).abs().max())


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    paths = (
        PACKAGE_DIR / "protocol.py",
        PACKAGE_DIR / "prepare.py",
        PACKAGE_DIR / "run_experiment.py",
        PACKAGE_DIR.parent / "mla_scarcity_5090" / "protocol.py",
        PACKAGE_DIR.parent / "mla_scarcity_5090" / "run_experiment.py",
        REPO_ROOT / "scripts/lib/rope/official_yarn.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _load_manifest(
    path: Path, *, check_tensor_hashes: bool
) -> dict[str, Any]:
    resolved = path.resolve()
    manifest = json.loads(resolved.read_text())
    source = Path(manifest["source_manifest"]["path"])
    validate_manifest(
        manifest,
        source_manifest=source,
        check_tensor_hashes=check_tensor_hashes,
    )
    return manifest


def _parity_ready_path(work_dir: Path) -> Path:
    return work_dir.resolve() / "operator_parity_ready.json"


def preflight(data_manifest: Path, work_dir: Path) -> dict[str, Any]:
    if BASE_SPEC.fingerprint() != BASE_TRAINING_PROTOCOL_SHA256:
        raise RuntimeError("underlying trainer protocol identity changed")
    manifest = _load_manifest(
        data_manifest, check_tensor_hashes=False
    )
    base_ready = validate_base_ready(
        data_manifest, work_dir, require_disk=True
    )
    masks: dict[str, Any] = {}
    for pairs in FREQUENCY_PAIRS:
        head_dim = 2 * pairs
        native = native_endpoint_inv_freq(head_dim, SPEC.base)
        by_scale: dict[str, Any] = {}
        for scale in (2.0, 4.0, 8.0):
            official, official_mscale, official_meta = (
                official_yarn_on_native_grid(
                    head_dim=head_dim,
                    base=SPEC.base,
                    scale=scale,
                    original_max_position_embeddings=SPEC.train_length,
                    beta_fast=SPEC.beta_fast,
                    beta_slow=SPEC.beta_slow,
                )
            )
            shared, shared_mscale, shared_meta = (
                shared_index_yarn_control_on_inv_freq(
                    native,
                    head_dim=head_dim,
                    base=SPEC.base,
                    scale=scale,
                    original_max_position_embeddings=SPEC.train_length,
                    beta_fast=SPEC.beta_fast,
                    beta_slow=SPEC.beta_slow,
                )
            )
            if (
                not torch.equal(official, shared)
                or official_mscale != shared_mscale
                or not shared_meta["official_on_input"]
            ):
                raise RuntimeError(
                    f"shared-index native parity failed K={pairs}/s={scale}"
                )
            by_scale[str(int(scale))] = {
                "low": official_meta["low"],
                "high": official_meta["high"],
                "mscale": official_mscale,
                "inv_freq_sha256": tensor_sha256(shared.float()),
                "index_extrapolation_weights": shared_meta[
                    "index_extrapolation_weights"
                ],
            }
        masks[f"k{pairs}"] = by_scale
    result = {
        "schema_version": 1,
        "status": "READY",
        "protocol_sha256": SPEC.fingerprint(),
        "evaluation_code_sha256": code_fingerprint(),
        "launcher_sha256": sha256_file(PACKAGE_DIR / "run_5090.sh"),
        "base_training_protocol_sha256": BASE_SPEC.fingerprint(),
        "base_training_code_sha256": base_code_fingerprint(),
        "base_ready_sha256": sha256_file(
            work_dir.resolve() / "ready_receipt.json"
        ),
        "data_manifest_sha256": sha256_file(data_manifest),
        "selection_anchor_sha256": manifest["selection_anchors"]["sha256"],
        "test_anchor_sha256": manifest["test_anchors"]["sha256"],
        "previous_selection_sha256": manifest["operator_parity"][
            "previous_selection_sha256"
        ],
        "previous_test_sha256": manifest["operator_parity"][
            "previous_test_sha256"
        ],
        "training_arms": TRAINING_ARMS,
        "frequency_pairs": FREQUENCY_PAIRS,
        "operators": OPERATORS,
        "official_native_parity": masks,
        "base_ready_status": base_ready["status"],
    }
    path = _parity_ready_path(work_dir)
    if path.exists():
        raise FileExistsError(path)
    _atomic_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def validate_ready(
    data_manifest: Path, work_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_manifest(
        data_manifest, check_tensor_hashes=False
    )
    base_ready = validate_base_ready(
        data_manifest, work_dir, require_disk=False
    )
    path = _parity_ready_path(work_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = json.loads(path.read_text())
    expected = {
        "status": "READY",
        "protocol_sha256": SPEC.fingerprint(),
        "evaluation_code_sha256": code_fingerprint(),
        "launcher_sha256": sha256_file(PACKAGE_DIR / "run_5090.sh"),
        "base_training_protocol_sha256": BASE_SPEC.fingerprint(),
        "base_training_code_sha256": base_code_fingerprint(),
        "base_ready_sha256": sha256_file(
            work_dir.resolve() / "ready_receipt.json"
        ),
        "data_manifest_sha256": sha256_file(data_manifest),
        "selection_anchor_sha256": manifest["selection_anchors"]["sha256"],
        "test_anchor_sha256": manifest["test_anchors"]["sha256"],
        "previous_selection_sha256": manifest["operator_parity"][
            "previous_selection_sha256"
        ],
        "previous_test_sha256": manifest["operator_parity"][
            "previous_test_sha256"
        ],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"operator-parity READY mismatch: {key}")
    if base_ready["status"] != "READY":
        raise ValueError("underlying trainer is not READY")
    return receipt, manifest


def _validate_parity_ready_record(work_dir: Path) -> dict[str, Any]:
    path = _parity_ready_path(work_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = json.loads(path.read_text())
    expected = {
        "status": "READY",
        "protocol_sha256": SPEC.fingerprint(),
        "evaluation_code_sha256": code_fingerprint(),
        "launcher_sha256": sha256_file(PACKAGE_DIR / "run_5090.sh"),
        "base_training_protocol_sha256": BASE_SPEC.fingerprint(),
        "base_training_code_sha256": base_code_fingerprint(),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"operator-parity READY mismatch: {key}")
    base_ready = work_dir.resolve() / "ready_receipt.json"
    if (
        not base_ready.is_file()
        or receipt.get("base_ready_sha256") != sha256_file(base_ready)
    ):
        raise ValueError("operator-parity READY base receipt mismatch")
    recorded = receipt.get("official_native_parity")
    if not isinstance(recorded, dict):
        raise ValueError("operator-parity READY lacks native parity proof")
    for pairs in FREQUENCY_PAIRS:
        native = native_endpoint_inv_freq(2 * pairs, SPEC.base)
        for scale in (2.0, 4.0, 8.0):
            official, mscale, meta = official_yarn_on_native_grid(
                head_dim=2 * pairs,
                base=SPEC.base,
                scale=scale,
                original_max_position_embeddings=SPEC.train_length,
                beta_fast=SPEC.beta_fast,
                beta_slow=SPEC.beta_slow,
            )
            shared, shared_mscale, shared_meta = (
                shared_index_yarn_control_on_inv_freq(
                    native,
                    head_dim=2 * pairs,
                    base=SPEC.base,
                    scale=scale,
                    original_max_position_embeddings=SPEC.train_length,
                    beta_fast=SPEC.beta_fast,
                    beta_slow=SPEC.beta_slow,
                )
            )
            expected_record = {
                "low": meta["low"],
                "high": meta["high"],
                "mscale": mscale,
                "inv_freq_sha256": tensor_sha256(shared.float()),
                "index_extrapolation_weights": shared_meta[
                    "index_extrapolation_weights"
                ],
            }
            if (
                not torch.equal(official, shared)
                or mscale != shared_mscale
                or recorded.get(f"k{pairs}", {}).get(str(int(scale)))
                != expected_record
            ):
                raise ValueError(
                    f"operator-parity READY native proof mismatch "
                    f"K={pairs}/scale={scale:g}"
                )
    return receipt


def runtime_operator(
    base_inv: torch.Tensor,
    *,
    frequency_pairs: int,
    arm: str,
    length: int,
    operator: str,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    pairs = int(frequency_pairs)
    if pairs not in FREQUENCY_PAIRS:
        raise ValueError(f"unregistered frequency budget: {pairs}")
    if arm not in TRAINING_ARMS:
        raise ValueError(f"unregistered training arm: {arm}")
    if operator not in OPERATORS:
        raise ValueError(f"unregistered operator: {operator}")
    if int(length) not in SPEC.eval_lengths:
        raise ValueError(f"unregistered evaluation length: {length}")
    full = base_inv.to(torch.float64).view(-1)
    if full.numel() != BASE_SPEC.rotary_pair_capacity:
        raise ValueError("checkpoint rotary capacity mismatch")
    expected, _ = training_inv_freq(
        arm,
        pairs,
        dtype=torch.float32,
    )
    if not torch.equal(full.float().cpu(), expected.cpu()):
        raise ValueError(
            "checkpoint frequency table does not match registered arm"
        )
    active = full[:pairs]
    if not torch.all(full[pairs:] == 0):
        raise ValueError("inactive checkpoint frequencies are not zero")
    scale = float(length) / float(SPEC.train_length)
    mscale = 1.0
    if operator == "raw":
        active_out = active.clone()
        metadata = {"mode": "raw", "scale": scale}
    elif operator == "position_interpolation":
        active_out = active / scale
        metadata = {
            "mode": "position_interpolation",
            "scale": scale,
            "shared_operator": True,
        }
    elif operator in (
        "shared_index_freq_only",
        "shared_index_full",
    ):
        active_out, official_mscale, metadata = (
            shared_index_yarn_control_on_inv_freq(
                active,
                head_dim=2 * pairs,
                base=SPEC.base,
                scale=scale,
                original_max_position_embeddings=SPEC.train_length,
                beta_fast=SPEC.beta_fast,
                beta_slow=SPEC.beta_slow,
            )
        )
        if operator == "shared_index_full":
            mscale = official_mscale
        if arm == "native_geo":
            official, reference_mscale, official_meta = (
                official_yarn_on_native_grid(
                    head_dim=2 * pairs,
                    base=SPEC.base,
                    scale=scale,
                    original_max_position_embeddings=SPEC.train_length,
                    beta_fast=SPEC.beta_fast,
                    beta_slow=SPEC.beta_slow,
                )
            )
            max_ulp = _max_fp32_ulp_distance(active_out, official)
            if (
                max_ulp > 1
                or official_mscale != reference_mscale
            ):
                raise RuntimeError(
                    "registered native checkpoint is not runtime-parity "
                    "with official YaRN"
                )
            metadata = {
                **metadata,
                "official_on_input": True,
                "official_runtime_output_max_fp32_ulp": max_ulp,
                "official_runtime_output_parity_within_one_fp32_ulp": True,
                "official_low": official_meta["low"],
                "official_high": official_meta["high"],
                "identity_proof": (
                    "exact registered native training-table equality plus "
                    "exact shared mask/formula and <=1 FP32 ULP output parity"
                ),
            }
        elif metadata.get("official_on_input"):
            raise RuntimeError("EVQ input was mislabeled as official YaRN")
        metadata = {
            **metadata,
            "mode": operator,
            "shared_operator": True,
            "frequency_component": True,
            "mscale_component": operator == "shared_index_full",
            "official_mscale_before_component_selection": official_mscale,
        }
    elif operator == "mscale_only":
        active_out = active.clone()
        mscale = yarn_mscale(scale)
        metadata = {
            "mode": "mscale_only",
            "scale": scale,
            "shared_operator": True,
            "frequency_component": False,
            "mscale_component": True,
        }
    else:
        if arm == "native_geo":
            active_out, mscale, metadata = (
                official_yarn_on_native_grid(
                    head_dim=2 * pairs,
                    base=SPEC.base,
                    scale=scale,
                    original_max_position_embeddings=SPEC.train_length,
                    beta_fast=SPEC.beta_fast,
                    beta_slow=SPEC.beta_slow,
                )
            )
        else:
            active_out, mscale, metadata = official_yarn_on_inv_freq(
                active,
                head_dim=2 * pairs,
                base=SPEC.base,
                scale=scale,
                original_max_position_embeddings=SPEC.train_length,
                beta_fast=SPEC.beta_fast,
                beta_slow=SPEC.beta_slow,
            )
        metadata = {
            **metadata,
            "mode": "virtual_coordinate_full",
            "shared_operator": False,
            "identity_boundary": (
                "official on native; virtual-coordinate derived on EVQ"
            ),
        }
    output = torch.zeros_like(full)
    output[:pairs] = active_out
    return output, float(mscale), {
        **metadata,
        "operator": operator,
        "scale": scale,
        "mscale": float(mscale),
        "active_frequency_pairs": pairs,
        "inactive_identity_pairs": int(full.numel() - pairs),
    }


def _parity_eval_path(
    work_dir: Path,
    pairs: int,
    arm: str,
    seed: int,
    split: str,
    stage: str,
    operator: str,
) -> Path:
    return _run_dir(work_dir, pairs, arm, seed) / (
        f"eval_parity_{split}_{stage}_{operator}.json"
    )


def _raw_eval_path(
    work_dir: Path,
    pairs: int,
    arm: str,
    seed: int,
    split: str,
    stage: str,
) -> Path:
    return _run_dir(work_dir, pairs, arm, seed) / (
        f"eval_{split}_{stage}_raw.json"
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest = validate_ready(args.data_manifest, args.work_dir)
    runtime = validate_cuda_runtime()
    pairs = int(args.frequency_pairs)
    arm = str(args.arm)
    seed = int(args.seed)
    stage = str(args.stage)
    split = str(args.split)
    operator = str(args.operator)
    if (
        pairs not in FREQUENCY_PAIRS
        or arm not in TRAINING_ARMS
        or seed not in SEEDS
        or stage not in SPEC.checkpoint_labels
        or split not in ("selection", "test")
        or operator not in PARITY_OPERATORS
    ):
        raise ValueError("unregistered parity evaluation condition")
    if split == "selection" and seed != GATE_SEED:
        raise RuntimeError(
            "operator-parity selection is restricted to seed 42"
        )
    if split == "test":
        validate_gate(args.work_dir, require_pass=True)
    run_dir = _run_dir(args.work_dir, pairs, arm, seed)
    checkpoint = run_dir / f"checkpoint_{stage}.pt"
    result_path = _parity_eval_path(
        args.work_dir,
        pairs,
        arm,
        seed,
        split,
        stage,
        operator,
    )
    if result_path.exists():
        raise FileExistsError(result_path)
    model, training, checkpoint_sha = _load_checkpoint(
        checkpoint,
        frequency_pairs=pairs,
        arm=arm,
        seed=seed,
        stage=stage,
    )
    if training["code_sha256"] != base_code_fingerprint():
        raise ValueError("checkpoint trainer code identity mismatch")
    if training["train_prefix_sha256"] != manifest["train"][
        "token_prefix_sha256"
    ]:
        raise ValueError("checkpoint training prefix identity mismatch")
    if training["validation_sha256"] != manifest["validation"]["sha256"]:
        raise ValueError("checkpoint validation identity mismatch")
    anchor_record = manifest[f"{split}_anchors"]
    if training[f"{split}_anchor_sha256"] != anchor_record["sha256"]:
        raise ValueError("checkpoint anchor identity mismatch")
    anchors = np.load(anchor_record["path"], allow_pickle=False)
    validation = np.load(
        manifest["validation"]["path"], mmap_mode="r", allow_pickle=False
    ).reshape(-1)
    base_inv = model_inv_freq(model).to(torch.float64)
    model = model.to("cuda").eval()
    rows: list[dict[str, Any]] = []
    operators: dict[str, Any] = {}
    started = time.time()
    with torch.inference_mode():
        for length in SPEC.eval_lengths:
            inv, mscale, metadata = runtime_operator(
                base_inv,
                frequency_pairs=pairs,
                arm=arm,
                length=length,
                operator=operator,
            )
            _set_runtime_rope(
                model,
                inv,
                max_position=length,
                mscale=mscale,
            )
            operators[str(length)] = {
                **metadata,
                "evaluation_batch_size": (
                    SPEC.eval_batch_size_by_length[length]
                ),
                "runtime_inv_freq_sha256": tensor_sha256(inv.float()),
            }
            endpoint_values = anchors.tolist()
            evaluation_batch = SPEC.eval_batch_size_by_length[length]
            for batch_start in range(
                0, len(endpoint_values), evaluation_batch
            ):
                current_endpoints = endpoint_values[
                    batch_start : batch_start + evaluation_batch
                ]
                tokens = np.stack(
                    [
                        np.array(
                            validation[
                                int(endpoint) - int(length) : int(endpoint)
                            ],
                            dtype=np.int64,
                            copy=True,
                        )
                        for endpoint in current_endpoints
                    ]
                )
                batch = torch.from_numpy(tokens).to("cuda")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch[:, :-1])
                    targets = batch[:, 1:]
                    full_nll, tail_nll, tail = per_sequence_nll(
                        logits,
                        targets,
                        tail_tokens=SPEC.eval_tail_tokens,
                    )
                full_values = full_nll.detach().cpu().tolist()
                tail_values = tail_nll.detach().cpu().tolist()
                for offset, endpoint in enumerate(current_endpoints):
                    rows.append(
                        {
                            "length": length,
                            "anchor_index": batch_start + offset,
                            "anchor_endpoint": int(endpoint),
                            "full_nll": float(full_values[offset]),
                            "tail_nll": float(tail_values[offset]),
                            "tail_tokens": tail,
                        }
                    )
                del batch, logits, targets, full_nll, tail_nll
            torch.cuda.empty_cache()
    summary: dict[str, Any] = {}
    for length in SPEC.eval_lengths:
        selected = [row for row in rows if row["length"] == length]
        full = [row["full_nll"] for row in selected]
        tail = [row["tail_nll"] for row in selected]
        summary[str(length)] = {
            "anchors": len(selected),
            "full_nll_mean": statistics.fmean(full),
            "full_nll_sample_std": statistics.stdev(full),
            "tail_nll_mean": statistics.fmean(tail),
            "tail_nll_sample_std": statistics.stdev(tail),
            "tail_ppl": math.exp(statistics.fmean(tail)),
        }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_sha256": SPEC.fingerprint(),
        "evaluation_code_sha256": code_fingerprint(),
        "base_training_protocol_sha256": BASE_SPEC.fingerprint(),
        "base_training_code_sha256": training["code_sha256"],
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "checkpoint_sha256": checkpoint_sha,
        "trained_inv_freq_sha256": training["inv_freq_sha256"],
        "tokens_seen": training["tokens_seen"],
        "anchor_sha256": anchor_record["sha256"],
        "frequency_pairs": pairs,
        "arm": arm,
        "seed": seed,
        "stage": stage,
        "split": split,
        "operator": operator,
        "runtime": runtime,
        "elapsed_seconds": time.time() - started,
        "metric_definition": (
            "teacher-forced causal NLL; primary tail NLL covers the final "
            "4096 targets of each fixed held-out window"
        ),
        "evaluation_batch_sizes": {
            str(length): batch
            for length, batch in SPEC.eval_batch_size_by_length.items()
        },
        "operators": operators,
        "summary": summary,
        "rows": rows,
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_raw(
    work_dir: Path,
    *,
    pairs: int,
    arm: str,
    seed: int,
    split: str,
    stage: str,
) -> dict[str, Any]:
    path = _raw_eval_path(
        work_dir, pairs, arm, seed, split, stage
    )
    result = json.loads(path.read_text())
    expected = {
        "status": "PASS",
        "frequency_pairs": pairs,
        "arm": arm,
        "seed": seed,
        "stage": stage,
        "split": split,
        "operator": "raw",
        "protocol_sha256": BASE_SPEC.fingerprint(),
        "evaluation_code_sha256": base_code_fingerprint(),
        "training_code_sha256": base_code_fingerprint(),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"raw evaluation identity mismatch: {key}")
    expected_batches = {
        str(length): batch
        for length, batch in SPEC.eval_batch_size_by_length.items()
    }
    if result.get("evaluation_batch_sizes") != expected_batches:
        raise ValueError("raw evaluation batch-size identity mismatch")
    for key in (
        "data_manifest_sha256",
        "checkpoint_sha256",
        "anchor_sha256",
        "trained_inv_freq_sha256",
    ):
        if not isinstance(result.get(key), str) or len(result[key]) != 64:
            raise ValueError(f"raw evaluation lacks identity field: {key}")
    return result


def _load_parity(
    work_dir: Path,
    *,
    pairs: int,
    arm: str,
    seed: int,
    split: str,
    stage: str,
    operator: str,
) -> dict[str, Any]:
    path = _parity_eval_path(
        work_dir,
        pairs,
        arm,
        seed,
        split,
        stage,
        operator,
    )
    result = json.loads(path.read_text())
    expected = {
        "status": "PASS",
        "frequency_pairs": pairs,
        "arm": arm,
        "seed": seed,
        "stage": stage,
        "split": split,
        "operator": operator,
        "protocol_sha256": SPEC.fingerprint(),
        "evaluation_code_sha256": code_fingerprint(),
        "base_training_code_sha256": base_code_fingerprint(),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"parity evaluation identity mismatch: {key}")
    expected_batches = {
        str(length): batch
        for length, batch in SPEC.eval_batch_size_by_length.items()
    }
    if result.get("evaluation_batch_sizes") != expected_batches:
        raise ValueError("parity evaluation batch-size identity mismatch")
    for key in (
        "data_manifest_sha256",
        "checkpoint_sha256",
        "anchor_sha256",
        "trained_inv_freq_sha256",
    ):
        if not isinstance(result.get(key), str) or len(result[key]) != 64:
            raise ValueError(f"parity evaluation lacks identity field: {key}")
    return result


def _paired_tail_differences(
    native: dict[str, Any],
    evq: dict[str, Any],
    *,
    length: int,
) -> list[dict[str, Any]]:
    def indexed(result: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
        selected = [
            row
            for row in result.get("rows", [])
            if int(row.get("length", -1)) == int(length)
        ]
        values = {
            (
                int(row["anchor_index"]),
                int(row["anchor_endpoint"]),
            ): row
            for row in selected
        }
        if len(values) != len(selected) or not values:
            raise ValueError("evaluation rows have invalid anchor identity")
        return values

    native_rows = indexed(native)
    evq_rows = indexed(evq)
    if native_rows.keys() != evq_rows.keys():
        raise ValueError("native/EVQ evaluation anchors do not align")
    output: list[dict[str, Any]] = []
    for key in sorted(native_rows):
        native_row = native_rows[key]
        evq_row = evq_rows[key]
        if native_row["tail_tokens"] != evq_row["tail_tokens"]:
            raise ValueError("native/EVQ tail definitions do not align")
        output.append(
            {
                "anchor_index": key[0],
                "anchor_endpoint": key[1],
                "native_nll": float(native_row["tail_nll"]),
                "evq_nll": float(evq_row["tail_nll"]),
                "difference": (
                    float(native_row["tail_nll"])
                    - float(evq_row["tail_nll"])
                ),
            }
        )
    return output


def _anchor_effect_stats(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    values = [float(row["difference"]) for row in rows]
    if len(values) < 2:
        raise ValueError("paired anchor summary needs at least two values")
    return {
        "anchors": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)),
        "values": rows,
        "independence_boundary": (
            "Held-out windows are paired repeated measurements, not "
            "independent training seeds."
        ),
    }


def effect_rows(
    work_dir: Path,
    *,
    split: str,
    stage: str,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        results: dict[tuple[int, str, str], dict[str, Any]] = {}
        data_manifest_hashes: set[str] = set()
        anchor_hashes: set[str] = set()
        stage_operators = operators_for_stage(stage)
        for pairs in FREQUENCY_PAIRS:
            for arm in TRAINING_ARMS:
                raw = _load_raw(
                    work_dir,
                    pairs=pairs,
                    arm=arm,
                    seed=seed,
                    split=split,
                    stage=stage,
                )
                data_manifest_hashes.add(raw["data_manifest_sha256"])
                anchor_hashes.add(raw["anchor_sha256"])
                results[(pairs, arm, "raw")] = raw
                for operator in stage_operators:
                    if operator == "raw":
                        continue
                    value = _load_parity(
                        work_dir,
                        pairs=pairs,
                        arm=arm,
                        seed=seed,
                        split=split,
                        stage=stage,
                        operator=operator,
                    )
                    for key in (
                        "data_manifest_sha256",
                        "checkpoint_sha256",
                        "anchor_sha256",
                        "trained_inv_freq_sha256",
                    ):
                        if value[key] != raw[key]:
                            raise ValueError(
                                f"raw/parity identity mismatch: "
                                f"K={pairs}/{arm}/{seed}/{stage}/{operator}/"
                                f"{key}"
                            )
                    data_manifest_hashes.add(
                        value["data_manifest_sha256"]
                    )
                    anchor_hashes.add(value["anchor_sha256"])
                    results[(pairs, arm, operator)] = value
        if len(data_manifest_hashes) != 1 or len(anchor_hashes) != 1:
            raise ValueError(
                f"evaluation identities differ across arms for "
                f"seed={seed}/{split}/{stage}"
            )
        for length in SPEC.eval_lengths:
            effects: dict[str, Any] = {}
            for pairs in FREQUENCY_PAIRS:
                by_operator: dict[str, Any] = {}
                for operator in stage_operators:
                    native_result = results[
                        (pairs, "native_geo", operator)
                    ]
                    evq_result = results[
                        (pairs, "evq_cosh", operator)
                    ]
                    native = native_result["summary"][str(length)][
                        "tail_nll_mean"
                    ]
                    evq = evq_result["summary"][str(length)][
                        "tail_nll_mean"
                    ]
                    paired = _paired_tail_differences(
                        native_result,
                        evq_result,
                        length=length,
                    )
                    paired_stats = _anchor_effect_stats(paired)
                    if not math.isclose(
                        paired_stats["mean"],
                        native - evq,
                        rel_tol=0.0,
                        abs_tol=1e-10,
                    ):
                        raise ValueError(
                            "paired-anchor mean differs from summary delta"
                        )
                    by_operator[operator] = {
                        "native_nll": native,
                        "evq_nll": evq,
                        "evq_advantage": native - evq,
                        "evq_advantage_by_anchor": paired_stats,
                    }
                shared = by_operator["shared_index_full"]
                raw = by_operator["raw"]
                shared_values = shared[
                    "evq_advantage_by_anchor"
                ]["values"]
                raw_values = raw["evq_advantage_by_anchor"]["values"]
                if [
                    (row["anchor_index"], row["anchor_endpoint"])
                    for row in shared_values
                ] != [
                    (row["anchor_index"], row["anchor_endpoint"])
                    for row in raw_values
                ]:
                    raise ValueError(
                        "shared/raw operator anchors do not align"
                    )
                shared_interaction_rows = [
                    {
                        "anchor_index": shared_row["anchor_index"],
                        "anchor_endpoint": shared_row["anchor_endpoint"],
                        "difference": (
                            shared_row["difference"]
                            - raw_row["difference"]
                        ),
                    }
                    for shared_row, raw_row in zip(
                        shared_values, raw_values
                    )
                ]
                shared_interaction = (
                    shared["evq_advantage"]
                    - raw["evq_advantage"]
                )
                shared_interaction_stats = _anchor_effect_stats(
                    shared_interaction_rows
                )
                if not math.isclose(
                    shared_interaction_stats["mean"],
                    shared_interaction,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise ValueError(
                        "paired interaction mean differs from summary"
                    )
                effects[f"k{pairs}"] = {
                    "operators": by_operator,
                    "shared_interaction": shared_interaction,
                    "shared_interaction_by_anchor": (
                        shared_interaction_stats
                    ),
                    "native_shared_minus_raw": (
                        shared["native_nll"] - raw["native_nll"]
                    ),
                }
            k8_values = effects["k8"][
                "shared_interaction_by_anchor"
            ]["values"]
            k32_values = effects["k32"][
                "shared_interaction_by_anchor"
            ]["values"]
            if [
                (row["anchor_index"], row["anchor_endpoint"])
                for row in k8_values
            ] != [
                (row["anchor_index"], row["anchor_endpoint"])
                for row in k32_values
            ]:
                raise ValueError("K=8/K=32 anchors do not align")
            scarcity_rows = [
                {
                    "anchor_index": k8_row["anchor_index"],
                    "anchor_endpoint": k8_row["anchor_endpoint"],
                    "difference": (
                        k8_row["difference"] - k32_row["difference"]
                    ),
                }
                for k8_row, k32_row in zip(k8_values, k32_values)
            ]
            scarcity_interaction = (
                effects["k8"]["shared_interaction"]
                - effects["k32"]["shared_interaction"]
            )
            scarcity_stats = _anchor_effect_stats(scarcity_rows)
            if not math.isclose(
                scarcity_stats["mean"],
                scarcity_interaction,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    "paired scarcity mean differs from summary"
                )
            rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "length": length,
                    **effects,
                    "scarcity_interaction": scarcity_interaction,
                    "scarcity_interaction_by_anchor": scarcity_stats,
                }
            )
    return rows


def gate(work_dir: Path) -> dict[str, Any]:
    ready = _validate_parity_ready_record(work_dir)
    path = work_dir.resolve() / "operator_parity_gate.json"
    if path.exists():
        raise FileExistsError(path)
    rows: list[dict[str, Any]] = []
    for stage in SPEC.checkpoint_labels:
        rows += effect_rows(
            work_dir,
            split="selection",
            stage=stage,
            seeds=(GATE_SEED,),
        )
    lookup = {
        (row["stage"], row["length"]): row for row in rows
    }
    current = [lookup[("300m", length)] for length in PRIMARY_LENGTHS]
    previous = [lookup[("200m", length)] for length in PRIMARY_LENGTHS]
    shared_advantages = [
        row["k8"]["operators"]["shared_index_full"]["evq_advantage"]
        for row in current
    ]
    interactions = [
        row["k8"]["shared_interaction"] for row in current
    ]
    scarcity = statistics.fmean(
        row["scarcity_interaction"] for row in current
    )
    in_domain = lookup[("300m", SPEC.train_length)]
    in_domain_costs = {
        f"k{pairs}": -in_domain[f"k{pairs}"]["operators"][
            "shared_index_full"
        ]["evq_advantage"]
        for pairs in FREQUENCY_PAIRS
    }
    native_scaler_costs = [
        row[f"k{pairs}"]["native_shared_minus_raw"]
        for row in current
        for pairs in FREQUENCY_PAIRS
    ]
    previous_shared = [
        row["k8"]["operators"]["shared_index_full"]["evq_advantage"]
        for row in previous
    ]
    previous_interactions = [
        row["k8"]["shared_interaction"] for row in previous
    ]
    criteria = {
        "k8_shared_advantage_positive_at_16k_32k": all(
            value > 0 for value in shared_advantages
        ),
        "mean_k8_shared_advantage_gte_0p05": (
            statistics.fmean(shared_advantages)
            >= SPEC.minimum_mean_scaled_advantage_nll
        ),
        "k8_interaction_positive_at_16k_32k": all(
            value > 0 for value in interactions
        ),
        "mean_k8_interaction_gte_0p05": (
            statistics.fmean(interactions)
            >= SPEC.minimum_mean_interaction_nll
        ),
        "scarcity_interaction_positive": scarcity > 0,
        "in_domain_cost_lte_0p02_for_k8_k32": all(
            value <= SPEC.maximum_in_domain_cost_nll
            for value in in_domain_costs.values()
        ),
        "native_scaler_cost_lte_0p05_for_k8_k32_at_16k_32k": all(
            value <= SPEC.maximum_native_scaler_cost_nll
            for value in native_scaler_costs
        ),
        "direction_agrees_at_200m_300m": (
            all(value > 0 for value in previous_shared)
            and all(value > 0 for value in previous_interactions)
        ),
        "native_shared_index_formula_parity_within_one_fp32_ulp": True,
    }
    passed = all(criteria.values())
    result = {
        "schema_version": 1,
        "status": "PASS" if passed else "STOP",
        "decision": (
            "expand_to_confirmatory_seeds"
            if passed
            else "do_not_expand"
        ),
        "selection_split_only": True,
        "test_split_read": False,
        "protocol_sha256": SPEC.fingerprint(),
        "evaluation_code_sha256": code_fingerprint(),
        "criteria": criteria,
        "primary_lengths": PRIMARY_LENGTHS,
        "mean_k8_shared_advantage": statistics.fmean(
            shared_advantages
        ),
        "mean_k8_shared_interaction": statistics.fmean(interactions),
        "mean_scarcity_interaction": scarcity,
        "in_domain_evq_minus_native": in_domain_costs,
        "native_scaler_costs": native_scaler_costs,
        "ready_receipt_sha256": sha256_file(
            _parity_ready_path(work_dir)
        ),
        "official_native_parity": ready["official_native_parity"],
        "rows": rows,
    }
    _atomic_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def validate_gate(work_dir: Path, *, require_pass: bool) -> dict[str, Any]:
    path = work_dir.resolve() / "operator_parity_gate.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text())
    expected = {
        "protocol_sha256": SPEC.fingerprint(),
        "evaluation_code_sha256": code_fingerprint(),
        "selection_split_only": True,
        "test_split_read": False,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"operator-parity gate mismatch: {key}")
    if result.get("status") not in ("PASS", "STOP"):
        raise ValueError("operator-parity gate lacks a terminal status")
    if require_pass and result["status"] != "PASS":
        raise RuntimeError("confirmatory execution requires a PASS gate")
    _validate_parity_ready_record(work_dir)
    return result


def _mean_ci(values: list[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS):
        raise ValueError(
            f"seed-level interval requires {len(SEEDS)} values"
        )
    mean = statistics.fmean(values)
    critical = 4.302652729911275
    margin = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "ci95": [mean - margin, mean + margin],
        "values": values,
    }


def _aggregate_effect_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for stage in SPEC.checkpoint_labels:
        for length in SPEC.eval_lengths:
            selected = [
                row
                for row in rows
                if row["stage"] == stage and row["length"] == length
            ]
            if len(selected) != len(SEEDS):
                raise ValueError(
                    f"incomplete seed rows for {stage}/{length}"
                )
            by_k: dict[str, Any] = {}
            stage_operators = operators_for_stage(stage)
            for pairs in FREQUENCY_PAIRS:
                operator_rows: dict[str, Any] = {}
                for operator in stage_operators:
                    operator_rows[operator] = {
                        metric: _mean_ci(
                            [
                                row[f"k{pairs}"]["operators"][operator][
                                    metric
                                ]
                                for row in selected
                            ]
                        )
                        for metric in (
                            "native_nll",
                            "evq_nll",
                            "evq_advantage",
                        )
                    }
                by_k[f"k{pairs}"] = {
                    "operators": operator_rows,
                    "shared_interaction": _mean_ci(
                        [
                            row[f"k{pairs}"]["shared_interaction"]
                            for row in selected
                        ]
                    ),
                    "native_shared_minus_raw": _mean_ci(
                        [
                            row[f"k{pairs}"]["native_shared_minus_raw"]
                            for row in selected
                        ]
                    ),
                }
            aggregates.append(
                {
                    "stage": stage,
                    "length": length,
                    **by_k,
                    "scarcity_interaction": _mean_ci(
                        [
                            row["scarcity_interaction"]
                            for row in selected
                        ]
                    ),
                }
            )
    return aggregates


def summarize(work_dir: Path) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    gate_record = validate_gate(work_dir, require_pass=True)
    rows: list[dict[str, Any]] = []
    for stage in SPEC.checkpoint_labels:
        rows.extend(
            effect_rows(
                work_dir,
                split="test",
                stage=stage,
                seeds=SEEDS,
            )
        )
    aggregates = _aggregate_effect_rows(rows)
    lookup = {
        (row["seed"], row["stage"], row["length"]): row
        for row in rows
    }

    per_seed_primary: list[dict[str, Any]] = []
    for seed in SEEDS:
        current = [
            lookup[(seed, "300m", length)]
            for length in PRIMARY_LENGTHS
        ]
        previous = [
            lookup[(seed, "200m", length)]
            for length in PRIMARY_LENGTHS
        ]
        in_domain = lookup[(seed, "300m", SPEC.train_length)]
        shared_advantages = [
            row["k8"]["operators"]["shared_index_full"][
                "evq_advantage"
            ]
            for row in current
        ]
        shared_interactions = [
            row["k8"]["shared_interaction"] for row in current
        ]
        scarcity_interactions = [
            row["scarcity_interaction"] for row in current
        ]
        in_domain_costs = {
            f"k{pairs}": -in_domain[f"k{pairs}"]["operators"][
                "shared_index_full"
            ]["evq_advantage"]
            for pairs in FREQUENCY_PAIRS
        }
        native_scaler_costs = [
            row[f"k{pairs}"]["native_shared_minus_raw"]
            for row in current
            for pairs in FREQUENCY_PAIRS
        ]
        previous_shared_advantages = [
            row["k8"]["operators"]["shared_index_full"][
                "evq_advantage"
            ]
            for row in previous
        ]
        previous_shared_interactions = [
            row["k8"]["shared_interaction"] for row in previous
        ]
        per_seed_primary.append(
            {
                "seed": seed,
                "k8_shared_advantages": shared_advantages,
                "k8_shared_interactions": shared_interactions,
                "scarcity_interactions": scarcity_interactions,
                "mean_scarcity_interaction": statistics.fmean(
                    scarcity_interactions
                ),
                "in_domain_evq_minus_native": in_domain_costs,
                "native_scaler_costs": native_scaler_costs,
                "direction_agrees_at_200m_300m": (
                    all(value > 0 for value in shared_advantages)
                    and all(
                        value > 0 for value in shared_interactions
                    )
                    and all(
                        value > 0
                        for value in previous_shared_advantages
                    )
                    and all(
                        value > 0
                        for value in previous_shared_interactions
                    )
                ),
            }
        )

    all_shared_advantages = [
        value
        for record in per_seed_primary
        for value in record["k8_shared_advantages"]
    ]
    all_shared_interactions = [
        value
        for record in per_seed_primary
        for value in record["k8_shared_interactions"]
    ]
    per_seed_mean_shared_advantages = [
        statistics.fmean(record["k8_shared_advantages"])
        for record in per_seed_primary
    ]
    per_seed_mean_shared_interactions = [
        statistics.fmean(record["k8_shared_interactions"])
        for record in per_seed_primary
    ]
    per_seed_mean_scarcity_interactions = [
        record["mean_scarcity_interaction"]
        for record in per_seed_primary
    ]
    seed_level_precision = {
        "mean_k8_shared_advantage_across_primary_lengths": _mean_ci(
            per_seed_mean_shared_advantages
        ),
        "mean_k8_operator_interaction_across_primary_lengths": _mean_ci(
            per_seed_mean_shared_interactions
        ),
        "mean_scarcity_interaction_across_primary_lengths": _mean_ci(
            per_seed_mean_scarcity_interactions
        ),
    }
    precision_criteria = {
        name + "_ci95_lower_gt_zero": record["ci95"][0] > 0
        for name, record in seed_level_precision.items()
    }
    criteria = {
        "all_seeds_k8_shared_advantage_positive_at_16k_32k": all(
            value > 0 for value in all_shared_advantages
        ),
        "mean_k8_shared_advantage_gte_0p05": (
            statistics.fmean(all_shared_advantages)
            >= SPEC.minimum_mean_scaled_advantage_nll
        ),
        "all_seeds_k8_interaction_positive_at_16k_32k": all(
            value > 0 for value in all_shared_interactions
        ),
        "mean_k8_interaction_gte_0p05": (
            statistics.fmean(all_shared_interactions)
            >= SPEC.minimum_mean_interaction_nll
        ),
        "all_seeds_mean_scarcity_interaction_positive": all(
            record["mean_scarcity_interaction"] > 0
            for record in per_seed_primary
        ),
        "all_seeds_in_domain_cost_lte_0p02_for_k8_k32": all(
            value <= SPEC.maximum_in_domain_cost_nll
            for record in per_seed_primary
            for value in record["in_domain_evq_minus_native"].values()
        ),
        "all_seeds_native_scaler_cost_lte_0p05": all(
            value <= SPEC.maximum_native_scaler_cost_nll
            for record in per_seed_primary
            for value in record["native_scaler_costs"]
        ),
        "all_seeds_direction_agrees_at_200m_300m": all(
            record["direction_agrees_at_200m_300m"]
            for record in per_seed_primary
        ),
        "native_shared_index_formula_parity_within_one_fp32_ulp": True,
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "claim_gate": (
            "SUPPORTS_SHARED_OPERATOR_SCARCITY_INTERACTION"
            if all(criteria.values())
            else "DOES_NOT_SUPPORT_SHARED_OPERATOR_SCARCITY_INTERACTION"
        ),
        "precision_grade": (
            "SEED_LEVEL_CI95_EXCLUDES_ZERO"
            if all(precision_criteria.values())
            else "DIRECTIONALLY_CONSISTENT_BUT_SEED_CI95_CROSSES_ZERO"
        ),
        "protocol_sha256": SPEC.fingerprint(),
        "evaluation_code_sha256": code_fingerprint(),
        "base_training_protocol_sha256": BASE_SPEC.fingerprint(),
        "base_training_code_sha256": base_code_fingerprint(),
        "gate_receipt_sha256": sha256_file(
            work_dir / "operator_parity_gate.json"
        ),
        "selection_gate": gate_record["status"],
        "criteria": criteria,
        "precision_criteria": precision_criteria,
        "seed_level_precision": seed_level_precision,
        "primary_lengths": PRIMARY_LENGTHS,
        "mean_k8_shared_advantage": statistics.fmean(
            all_shared_advantages
        ),
        "mean_k8_shared_interaction": statistics.fmean(
            all_shared_interactions
        ),
        "per_seed_primary": per_seed_primary,
        "aggregates": aggregates,
        "per_seed_effects": rows,
        "statistical_boundary": (
            "Seed-level paired contrasts with t-based 95% intervals at "
            "n=3. Selection windows were used only for the seed-42 gate; "
            "these aggregates use the disjoint test windows."
        ),
        "identity_boundary": (
            "shared_index_full uses the same native-index correction "
            "coefficients and mscale in both arms. It is official YaRN only "
            "on native endpoint RoPE and a shared-index component control "
            "on EVQ."
        ),
    }
    path = work_dir / "summary_mla_yarn_operator_parity.json"
    if path.exists():
        raise FileExistsError(path)
    _atomic_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def cleanup_compile_cache(work_dir: Path) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    gate_path = work_dir / "operator_parity_gate.json"
    summary_path = work_dir / "summary_mla_yarn_operator_parity.json"
    terminal_evidence: Path | None = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("status") == "PASS"
            and summary.get("protocol_sha256") == SPEC.fingerprint()
            and summary.get("evaluation_code_sha256")
            == code_fingerprint()
        ):
            terminal_evidence = summary_path
    elif gate_path.is_file():
        gate_record = validate_gate(work_dir, require_pass=False)
        if gate_record["status"] == "STOP":
            terminal_evidence = gate_path
    if terminal_evidence is None:
        raise RuntimeError(
            "compile cache is retained until a STOP gate or final summary"
        )
    cache = work_dir / "torchinductor_cache"
    files = (
        [path for path in cache.rglob("*") if path.is_file()]
        if cache.is_dir()
        else []
    )
    size = sum(path.stat().st_size for path in files)
    if cache.is_dir():
        shutil.rmtree(cache)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_sha256": SPEC.fingerprint(),
        "cache": str(cache),
        "files_deleted": len(files),
        "bytes_deleted": size,
        "terminal_evidence": str(terminal_evidence),
        "terminal_evidence_sha256": sha256_file(terminal_evidence),
    }
    output = work_dir / "cleanup_operator_parity_compile_cache.json"
    if output.exists():
        raise FileExistsError(output)
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument(
        "--data-manifest", type=Path, required=True
    )
    preflight_parser.add_argument("--work-dir", type=Path, required=True)
    ready_parser = sub.add_parser("validate-ready")
    ready_parser.add_argument(
        "--data-manifest", type=Path, required=True
    )
    ready_parser.add_argument("--work-dir", type=Path, required=True)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument(
        "--data-manifest", type=Path, required=True
    )
    eval_parser.add_argument("--work-dir", type=Path, required=True)
    eval_parser.add_argument(
        "--frequency-pairs", type=int, required=True
    )
    eval_parser.add_argument(
        "--arm", choices=TRAINING_ARMS, required=True
    )
    eval_parser.add_argument("--seed", type=int, required=True)
    eval_parser.add_argument(
        "--stage", choices=SPEC.checkpoint_labels, required=True
    )
    eval_parser.add_argument(
        "--split", choices=("selection", "test"), required=True
    )
    eval_parser.add_argument(
        "--operator", choices=PARITY_OPERATORS, required=True
    )
    gate_parser = sub.add_parser("gate")
    gate_parser.add_argument("--work-dir", type=Path, required=True)
    validate_gate_parser = sub.add_parser("validate-gate")
    validate_gate_parser.add_argument(
        "--work-dir", type=Path, required=True
    )
    validate_gate_parser.add_argument(
        "--require-pass", action="store_true"
    )
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--work-dir", type=Path, required=True)
    cache_parser = sub.add_parser("cleanup-compile-cache")
    cache_parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.data_manifest, args.work_dir)
    elif args.command == "validate-ready":
        receipt, _ = validate_ready(args.data_manifest, args.work_dir)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "gate":
        gate(args.work_dir)
    elif args.command == "validate-gate":
        result = validate_gate(
            args.work_dir, require_pass=bool(args.require_pass)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "summarize":
        summarize(args.work_dir)
    elif args.command == "cleanup-compile-cache":
        cleanup_compile_cache(args.work_dir)


if __name__ == "__main__":
    main()
