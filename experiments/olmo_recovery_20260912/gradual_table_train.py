#!/usr/bin/env python3
"""Matched 8K Cosh LoRA with one change: a gradual Native-to-Cosh table install."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .recovery_v2_runtime import table_for_config
from .recovery_v2_train import Pool
from .runtime import load_model, set_static_in_place, step, validate_cuda
from .train import recoverable_checkpoint, restore_rng, save_bundle
from .two_stage_train import (
    LOSS_WEIGHTS,
    PEAK_LR,
    WARMUP_UPDATES,
    contract as stage_contract,
    short_replay,
    validate_long,
)


LENGTH = 8192
TOTAL_UPDATES = 300
NATIVE_HOLD_END = 20
TRANSITION_END = 200
ARM = "Cosh_tau_sqrt2_gradual_install"
TARGET_ARM = "Cosh_tau_sqrt2"


def table_at(update_number: int, native: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Return the exact table used by one-based optimizer update ``update_number``."""
    if not 1 <= update_number <= TOTAL_UPDATES:
        raise ValueError("update is outside the locked gradual-install schedule")
    if update_number <= NATIVE_HOLD_END:
        return native.copy(), 0.0, "native_hold"
    if update_number <= TRANSITION_END:
        fraction = (update_number - NATIVE_HOLD_END) / (TRANSITION_END - NATIVE_HOLD_END)
        values = np.exp(
            (1.0 - fraction) * np.log(native.astype(np.float64))
            + fraction * np.log(target.astype(np.float64))
        ).astype(np.float32)
        values[[0, -1]] = native[[0, -1]]
        if fraction == 1.0:
            values = target.copy()
        return values, float(fraction), "log_frequency_transition"
    return target.copy(), 1.0, "target_hold"


def scientific_contract(source_plan: dict, data: dict, target_table: dict, terminal_id: int) -> dict:
    base = stage_contract(source_plan, data, "8k", target_table, terminal_id)
    base.update(
        profile="expanded_data_8k300_gradual_table_install",
        arm=ARM,
        target_arm=TARGET_ARM,
        optimizer_state="fresh AdamW from the released base; same LR state as abrupt 8K training",
        table_installation={
            "updates_1_20": "exact Native FP32 table",
            "updates_21_200": "log-frequency interpolation; lambda=(update-20)/180",
            "updates_201_300": "exact Cosh_tau_sqrt2 FP32 table",
            "gain": 1.0,
            "intermediate_endpoints": "copied exactly from Native",
            "single_changed_factor": "table installation schedule",
        },
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--until-updates", type=int, default=TOTAL_UPDATES)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not 0 < args.until_updates <= TOTAL_UPDATES:
        raise ValueError("requested endpoint exceeds the locked 300-update schedule")
    source_plan = json.loads((args.source_root / "plan.json").read_text())
    data = json.loads(args.data.read_text())
    data["_manifest_path"] = args.data.resolve()
    if data.get("status") != "READY":
        raise ValueError("expanded data manifest is not READY")
    if source_plan["rank"] != 32 or source_plan["alpha"] != 32 or source_plan["dropout"] != 0:
        raise ValueError("source plan is not the locked R0 all-linear LoRA recipe")
    source_data = json.loads(Path(source_plan["data_manifest"]).read_text())
    terminal_id = int(source_data["assistant_terminal_id"])

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(source_plan["model_path"], local_files_only=True)
    native_table = table_for_config(config, "Native")
    target_table = table_for_config(config, TARGET_ARM)
    scientific = scientific_contract(source_plan, data, target_table, terminal_id)
    metadata = {
        "status": "METADATA_ONLY" if not args.execute else "STARTING",
        "scientific_config": scientific,
        "asset_identity_policy": "user_attested_clone/no_sha_validation",
    }
    print(json.dumps(metadata), flush=True)
    if not args.execute:
        return

    resume = recoverable_checkpoint(args.resume.resolve()) if args.resume else None
    if args.out.exists() and resume is None:
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=bool(resume))
    (args.out / "plan.json").write_text(json.dumps(metadata, indent=2) + "\n")

    pools = data["pools"]
    long_lm = Pool(pools["long_lm_8192"])
    long_sft = Pool(pools["long_sft_8192"])
    short_sft = Pool(pools["short_sft"])
    if min(long_lm.total, long_sft.total, short_sft.total) <= 0:
        raise ValueError("one or more expanded pools are empty")

    validate_cuda()
    model, wrapper = load_model(source_plan, native_table, checkpoint=resume)
    model._inplace_rope_table_updates = True
    model._frozen_native_teacher_cache_path = str(args.out.parent / "native_teacher_trajectories.jsonl")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )
    start = 0
    prior_state = None
    if resume:
        prior_state = json.loads((resume / "state.json").read_text())
        if prior_state["scientific_config"] != scientific or prior_state.get("arm") != ARM:
            raise ValueError("resume scientific contract mismatch")
        start = int(prior_state["stage_update"])
        saved = torch.load(resume / "training.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
    if start > args.until_updates:
        raise ValueError("resume checkpoint is beyond the requested endpoint")

    native = np.asarray(native_table["values_float32"], dtype=np.float32)
    target = np.asarray(target_table["values_float32"], dtype=np.float32)
    stage_input_tokens = int(prior_state.get("stage_input_tokens", 0)) if prior_state else 0
    started = time.monotonic()
    log_path = args.out / "steps.jsonl"
    with log_path.open("a") as log:
        for update in range(start, args.until_updates):
            update_number = update + 1
            installed, table_lambda, table_phase = table_at(update_number, native, target)
            set_static_in_place(model, installed, 1.0)

            cpt = long_lm.get(update, 3000 + LENGTH)
            sft = long_sft.get(update, 4000 + LENGTH)
            replay = short_replay(short_sft.get(update, 5000), terminal_id)
            validate_long(cpt, LENGTH, terminal_id, lm=True)
            validate_long(sft, LENGTH, terminal_id, lm=False)
            update_input_tokens = (
                int(cpt.shape[0] - 1)
                + len(sft["input_ids"]) - 1
                + len(replay["input_ids"]) - 1
            )
            stage_input_tokens += update_input_tokens
            lr = PEAK_LR * min(1.0, update_number / WARMUP_UPDATES)
            for group in optimizer.param_groups:
                group["lr"] = lr
            record = step(
                model,
                wrapper,
                optimizer,
                cpt,
                sft,
                replay,
                native_table=native,
                weights=LOSS_WEIGHTS,
                chunk_size=128,
                amp=True,
            )
            actual = model.model.rotary_emb.inv_freq.detach().cpu().numpy()
            if not np.array_equal(actual, installed) or float(model.model.rotary_emb.attention_scaling) != 1.0:
                raise RuntimeError("student table changed during the optimizer update")
            state = {
                "arm": ARM,
                "target_arm": TARGET_ARM,
                "stage": "8k_gradual_install",
                "stage_update": update_number,
                "input_tokens": stage_input_tokens,
                "stage_input_tokens": stage_input_tokens,
                "stage_cpt_tokens": update_number * LENGTH,
                "seed": source_plan["seed"],
                "table_lambda": table_lambda,
                "table_phase": table_phase,
                "installed_table_values_float32": installed.tolist(),
                "scientific_config": scientific,
                "asset_identity_policy": "user_attested_clone/no_sha_validation",
            }
            record.update(
                stage="8k_gradual_install",
                stage_update=update_number,
                length=LENGTH,
                lr=lr,
                table_lambda=table_lambda,
                table_phase=table_phase,
                table_linf_to_target=float(np.max(np.abs(installed - target))),
                update_primary_input_tokens=update_input_tokens,
                long_lm_pool_index=update,
                long_sft_pool_index=update,
                short_sft_pool_index=update,
            )
            log.write(json.dumps(record) + "\n")
            log.flush()
            if update_number % 50 == 0:
                save_bundle(args.out / "resume", wrapper, optimizer, state)

    endpoint = args.out / f"checkpoint-{args.until_updates}"
    save_bundle(endpoint, wrapper, optimizer, state)
    if args.until_updates == TOTAL_UPDATES and (args.out / "resume").is_dir():
        shutil.rmtree(args.out / "resume")
    completion = {
        "status": (
            "TRAINING_ENDPOINT_COMPLETE_NEEDS_EVALUATION"
            if args.until_updates == TOTAL_UPDATES
            else "INTERMEDIATE_STOP_RESUMABLE_NEEDS_EVALUATION"
        ),
        "stage": "8k_gradual_install",
        "arm": ARM,
        "target_arm": TARGET_ARM,
        "updates": args.until_updates,
        "schedule_updates": TOTAL_UPDATES,
        "length": LENGTH,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(endpoint.resolve()),
        "final_table_lambda": state["table_lambda"],
        "final_table_phase": state["table_phase"],
        "scientific_config": scientific,
    }
    (args.out / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
