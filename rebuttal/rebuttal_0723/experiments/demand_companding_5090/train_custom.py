#!/usr/bin/env python3
"""Run one frozen custom frequency table through the canonical 151.9M pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


AUTH_ENV = "DEMAND_COMPANDING_GPU_AUTHORIZED"
PACKAGE_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_schedule(path: Path, arm: str) -> tuple[torch.Tensor, dict[str, Any]]:
    manifest = json.loads(path.read_text())
    receipts = manifest.get("schedule_receipts")
    if not isinstance(receipts, dict) or arm not in receipts:
        raise KeyError(f"schedule arm {arm!r} is absent from {path}")
    receipt = receipts[arm]
    inv = np.asarray(receipt["inv_freq"], dtype=np.float32)
    if inv.shape != (32,) or not np.isfinite(inv).all() or not np.all(inv > 0):
        raise ValueError("custom 151.9M inv_freq must contain 32 finite positive values")
    if not np.all(np.diff(inv) < 0):
        raise ValueError("custom inv_freq must be strictly decreasing")
    observed = hashlib.sha256(inv.tobytes()).hexdigest()
    if observed != receipt["inv_freq_float32_sha256"]:
        raise ValueError("frequency hash does not match the dry-run manifest")
    return torch.from_numpy(inv.copy()), receipt


def patch_canonical(
    schedule_manifest: Path,
    arm: str,
    seed: int,
    micro_batch: int,
):
    os.environ["FMR_SEED"] = str(seed)
    os.environ["FMR_MICRO_BATCH_SIZE"] = str(micro_batch)
    from experiments.native_rope_evq_150m.model import GPT
    from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m import (
        run_experiment as canonical,
    )

    inv_freq, receipt = load_schedule(schedule_manifest, arm)
    if canonical.SPEC.seed != seed or canonical.SPEC.micro_batch_size != micro_batch:
        raise RuntimeError("canonical protocol was imported before seed/micro-batch binding")
    custom_arm = f"custom_{arm}"

    def training_inv_freq(_arm: str, **_kwargs) -> torch.Tensor:
        if _arm != custom_arm:
            raise ValueError(f"unexpected arm {_arm!r}")
        return inv_freq.clone()

    def build_model(_arm: str, **_kwargs) -> GPT:
        if _arm != custom_arm:
            raise ValueError(f"unexpected arm {_arm!r}")
        canonical.seed_everything(seed)
        return GPT(canonical.SPEC.model_config(), inv_freq.clone())

    def meta_parameter_count(_arm: str, **_kwargs) -> int:
        if _arm != custom_arm:
            raise ValueError(f"unexpected arm {_arm!r}")
        with torch.device("meta"):
            model = GPT(canonical.SPEC.model_config(), inv_freq.to("meta"))
        return sum(parameter.numel() for parameter in model.parameters())

    def runtime_frequency(
        _arm: str,
        condition: str,
        length: int,
        **_kwargs,
    ):
        if _arm != custom_arm or condition != "raw":
            raise ValueError(f"unexpected runtime request: {_arm}/{condition}")
        return inv_freq.clone(), 1.0, {
            "source": "frozen demand-companding schedule",
            "schedule_manifest": str(schedule_manifest.resolve()),
            "schedule_manifest_sha256": sha256(schedule_manifest),
            "schedule_arm": arm,
            "length": int(length),
            "runtime_table_changed": False,
        }

    wrapper_hash = hashlib.sha256(
        (PACKAGE_DIR / "train_custom.py").read_bytes()
        + schedule_manifest.read_bytes()
    ).hexdigest()
    canonical.training_inv_freq = training_inv_freq
    canonical.build_model = build_model
    canonical.meta_parameter_count = meta_parameter_count
    canonical.runtime_frequency = runtime_frequency
    canonical.ARM_CONDITIONS = {custom_arm: ("raw",)}
    canonical.code_fingerprint = lambda: wrapper_hash
    return canonical, custom_arm, receipt


def require_authorization(args: argparse.Namespace) -> None:
    if not args.authorize or os.environ.get(AUTH_ENV) != "YES":
        raise PermissionError(
            f"GPU action requires --authorize and {AUTH_ENV}=YES"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("command", choices=("preflight", "probe", "train", "evaluate"))
    result.add_argument("--schedule-manifest", type=Path, required=True)
    result.add_argument("--arm", required=True)
    result.add_argument("--seed", type=int, choices=(42, 137, 256), required=True)
    result.add_argument("--micro-batch", type=int, choices=(64, 128, 256), default=128)
    result.add_argument("--data-manifest", type=Path, required=True)
    result.add_argument("--work-dir", type=Path, required=True)
    result.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    result.add_argument("--timed-steps", type=int, default=20)
    result.add_argument("--num-workers", type=int, default=8)
    result.add_argument("--log-every", type=int, default=25)
    result.add_argument("--eval-batch-size", type=int, default=2)
    result.add_argument("--full-hash-check", action="store_true")
    result.add_argument("--authorize", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    canonical, custom_arm, receipt = patch_canonical(
        args.schedule_manifest.resolve(),
        args.arm,
        args.seed,
        args.micro_batch,
    )
    common = {
        "arm": custom_arm,
        "data_manifest": args.data_manifest.resolve(),
        "work_dir": args.work_dir.resolve(),
    }
    if args.command == "preflight":
        report = canonical.run_preflight(
            common["data_manifest"],
            full_hash_check=bool(args.full_hash_check),
            verify_full_initialization=False,
            arms=(custom_arm,),
        )
        print(json.dumps({
            "status": report["status"],
            "custom_arm": custom_arm,
            "frequency_hash": receipt["inv_freq_float32_sha256"],
            "training_started": False,
        }, sort_keys=True))
        return

    require_authorization(args)
    if args.command == "probe":
        canonical.probe_gpu(argparse.Namespace(
            **common,
            timed_steps=args.timed_steps,
            compile_mode=args.compile_mode,
        ))
    elif args.command == "train":
        canonical.train_arm(argparse.Namespace(
            **common,
            num_workers=args.num_workers,
            log_every=args.log_every,
            compile_mode=args.compile_mode,
            no_compile=False,
            full_hash_check=bool(args.full_hash_check),
        ))
    else:
        canonical.evaluate(argparse.Namespace(
            data_manifest=common["data_manifest"],
            work_dir=common["work_dir"],
            eval_batch_size=args.eval_batch_size,
            full_hash_check=bool(args.full_hash_check),
            arms=[custom_arm],
        ))


if __name__ == "__main__":
    main()
