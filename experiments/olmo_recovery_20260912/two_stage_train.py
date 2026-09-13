#!/usr/bin/env python3
"""Train the authorized staged 8K -> 16K OLMo experiment on expanded data."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .recovery_v2_runtime import table_for_config
from .recovery_v2_train import Pool
from .runtime import load_model, step, validate_cuda
from .train import recoverable_checkpoint, restore_rng, save_bundle


STAGES = {
    "8k": {"arm": "Cosh_tau_sqrt2", "length": 8192, "short_offset": 0, "schedule_updates": 300},
    "8k_native": {"arm": "Native", "length": 8192, "short_offset": 0, "schedule_updates": 300},
    "8k_coshdeploy": {"arm": "CoshDeploy_tau1_g4", "length": 8192, "short_offset": 0, "schedule_updates": 300},
    "8k_bm": {"arm": "BM_g4", "length": 8192, "short_offset": 0, "schedule_updates": 300},
    "16k": {"arm": "Cosh_tau1", "length": 16384, "short_offset": 300, "schedule_updates": 200},
    "16k_coshdeploy": {"arm": "CoshDeploy_tau1_g4", "length": 16384, "short_offset": 200, "schedule_updates": 200},
    "16k_bm": {"arm": "BM_g4", "length": 16384, "short_offset": 200, "schedule_updates": 200},
    "16k_bm_llama": {"arm": "BM_g4", "length": 16384, "short_offset": 100, "schedule_updates": 100},
}
WARMUP_UPDATES = 20
PEAK_LR = 2e-5
LOSS_WEIGHTS = {"cpt": 1.0, "sft": 1.0, "replay": .25, "kl": .25}
STAGE2_SFT_SOURCE_COUNTS = (43, 144, 13)
COSHDEPLOY_STAGE2_SFT_SOURCE_COUNTS = (36, 149, 15)
LLAMA_BM_STAGE2_SFT_SOURCE_COUNTS = (88, 12)


def stage2_source_counts(stage: str) -> tuple[int, ...] | None:
    if stage == "16k": return STAGE2_SFT_SOURCE_COUNTS
    if stage in ("16k_coshdeploy", "16k_bm"): return COSHDEPLOY_STAGE2_SFT_SOURCE_COUNTS
    if stage == "16k_bm_llama": return LLAMA_BM_STAGE2_SFT_SOURCE_COUNTS
    return None


def short_replay(row: dict, terminal_id: int) -> dict:
    replay = dict(row)
    start, total = int(replay["target_start"]), len(replay["input_ids"])
    if not 0 < start < total or total > 4096 or replay["input_ids"][-1] != terminal_id:
        raise ValueError("expanded short SFT row violates answer/EOS contract")
    positions = list(range(start - 1, total - 1))
    if len(positions) > 128:
        take = np.linspace(0, len(positions) - 1, 128, dtype=np.int64)
        positions = [positions[int(index)] for index in take]
    replay["task"] = "text"
    replay["kl_positions"] = positions
    return replay


def contract(source_plan: dict, data: dict, stage: str, table: dict, terminal_id: int) -> dict:
    spec = STAGES[stage]; length = spec["length"]
    pools = data["pools"]
    return {
        "profile": {
            "8k_native": "expanded_data_8k_native_control",
            "8k_coshdeploy": "expanded_data_8k_coshdeploy_development",
            "8k_bm": "expanded_data_8k_bm_development",
            "16k_coshdeploy": "expanded_data_16k_coshdeploy_continuation",
            "16k_bm": "expanded_data_16k_bm_continuation",
            "16k_bm_llama": "llama_bm_8k100_to_16k100_continuation",
        }.get(stage, "expanded_data_two_stage_8k200_16k200"),
        "stage": stage,
        "arm": spec["arm"],
        "table": table,
        "length": length,
        "schedule_updates": spec["schedule_updates"],
        "warmup_updates": WARMUP_UPDATES,
        "peak_lr": PEAK_LR,
        "optimizer_state": (
            "fresh AdamW from the released base; same LR state as other 8K arms"
            if stage in ("8k_coshdeploy", "8k_bm")
            else "inherited across stages; LR warmup restarts per stage"
        ),
        "loss_weights": LOSS_WEIGHTS,
        "assistant_terminal_id": terminal_id,
        "lora": {"rank": 32, "alpha": 32, "dropout": 0.0, "modules": source_plan["modules"]},
        "data_manifest": str(data["_manifest_path"]),
        "pools": {
            "long_lm": pools[f"long_lm_{length}"],
            "long_sft": pools[f"long_sft_{length}"],
            "short_sft": pools["short_sft"],
        },
        "maximum_data_visits": {name: spec["schedule_updates"] for name in ("long_lm", "long_sft", "short_sft")},
        "short_sft_counter_offset": spec["short_offset"],
        "fixed_long_sft_source_counts": (
            list(stage2_source_counts(stage)) if stage2_source_counts(stage) else None
        ),
        "seed": int(source_plan["seed"]),
    }


def fixed_stage2_sft_selection(
    pool: Pool, records: list[dict], source_counts: tuple[int, ...], schedule_updates: int
) -> list[tuple[int, int]]:
    names = [Path(record["path"]).parent.name for record in records]
    expected_names = {
        2: ["data_expanded_public_8k16k", "data_expanded_longcite_8k16k"],
        3: ["data_expanded_longalign", "data_expanded_longalpaca", "data_expanded_longcite"],
    }
    if (names != expected_names.get(len(names)) or len(source_counts) != len(pool.sources)
            or sum(source_counts) != schedule_updates):
        raise ValueError("16K long-SFT sources differ from the fixed stage-1 mixture")
    source_order = [index for index, count in enumerate(source_counts) for _ in range(count)]
    random.Random(20261600).shuffle(source_order)
    local_orders = []
    for index, (source, count) in enumerate(zip(pool.sources, source_counts)):
        if len(source) < count: raise ValueError("16K long-SFT source is smaller than its fixed allocation")
        order = list(range(len(source))); random.Random(20261610 + index).shuffle(order)
        local_orders.append(order)
    counters = [0] * len(pool.sources); selected = []
    for source_index in source_order:
        selected.append((source_index, local_orders[source_index][counters[source_index]]))
        counters[source_index] += 1
    return selected


def validate_long(example, length: int, terminal_id: int, *, lm: bool) -> None:
    if lm:
        if not isinstance(example, np.ndarray) or example.shape != (length + 1,):
            raise ValueError("expanded long LM row has wrong physical length")
        return
    total, start = len(example["input_ids"]), int(example["target_start"])
    if not 0 < start < total <= length or example["input_ids"][-1] != terminal_id:
        raise ValueError("expanded long SFT row violates answer/EOS contract")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--until-updates", type=int, default=200)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--initialize-from", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    source_plan = json.loads((args.source_root / "plan.json").read_text())
    data = json.loads(args.data.read_text()); data["_manifest_path"] = args.data.resolve()
    if data.get("status") != "READY": raise ValueError("expanded data manifest is not READY")
    if source_plan["rank"] != 32 or source_plan["alpha"] != 32 or source_plan["dropout"] != 0:
        raise ValueError("source plan is not the locked R0 all-linear LoRA recipe")
    source_data = json.loads(Path(source_plan["data_manifest"]).read_text())
    terminal_id = int(source_data["assistant_terminal_id"])
    spec = STAGES[args.stage]; length = spec["length"]
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(source_plan["model_path"], local_files_only=True)
    table = table_for_config(config, spec["arm"])
    scientific = contract(source_plan, data, args.stage, table, terminal_id)
    if not 0 < args.until_updates <= spec["schedule_updates"]:
        raise ValueError("requested endpoint exceeds the stage schedule")
    metadata = {"status": "METADATA_ONLY" if not args.execute else "STARTING",
                "scientific_config": scientific, "asset_identity_policy": "user_attested_clone/no_sha_validation"}
    print(json.dumps(metadata), flush=True)
    if not args.execute: return

    continuation_stage = args.stage in ("16k", "16k_coshdeploy", "16k_bm", "16k_bm_llama")
    if not continuation_stage and args.initialize_from:
        raise ValueError("8K stages must start from the base model")
    if continuation_stage and not (args.initialize_from or args.resume):
        raise ValueError("16K stage must inherit the 8K optimizer or resume itself")
    resume = recoverable_checkpoint(args.resume.resolve()) if args.resume else None
    initialize = recoverable_checkpoint(args.initialize_from.resolve()) if args.initialize_from else None
    if args.out.exists() and resume is None: raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=bool(resume))
    (args.out / "plan.json").write_text(json.dumps(metadata, indent=2) + "\n")

    pools = data["pools"]
    long_lm = Pool(pools[f"long_lm_{length}"])
    long_sft = Pool(pools[f"long_sft_{length}"])
    short_sft = Pool(pools["short_sft"])
    source_counts = stage2_source_counts(args.stage)
    fixed_sft = (
        fixed_stage2_sft_selection(
            long_sft, pools[f"long_sft_{length}"], source_counts, spec["schedule_updates"]
        )
        if source_counts else None
    )
    if min(long_lm.total, long_sft.total, short_sft.total) <= 0:
        raise ValueError("one or more expanded pools are empty")

    validate_cuda()
    checkpoint = resume or initialize
    model, wrapper = load_model(source_plan, table, checkpoint=checkpoint)
    model._frozen_native_teacher_cache_path = str(args.out.parent / "native_teacher_trajectories.jsonl")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=PEAK_LR,
                                  betas=(.9, .95), weight_decay=0., fused=True)
    start = 0
    if checkpoint:
        saved = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"]); restore_rng(saved["rng"])
    if resume:
        previous = json.loads((resume / "state.json").read_text())
        if previous["scientific_config"] != scientific: raise ValueError("resume scientific contract mismatch")
        start = int(previous["stage_update"])
    elif initialize:
        previous = json.loads((initialize / "state.json").read_text())
        expected = {
            "16k_coshdeploy": ("CoshDeploy_tau1_g4", (200,)),
            "16k_bm": ("BM_g4", (200,)),
            "16k_bm_llama": ("BM_g4", (100,)),
        }.get(args.stage, ("Cosh_tau_sqrt2", (200, 300)))
        if previous.get("arm") != expected[0] or previous.get("stage_update") not in expected[1]:
            raise ValueError("16K initialization is not the matching completed 8K stage")

    native = np.asarray(table_for_config(config, "Native")["values_float32"], dtype=np.float32)
    stage_input_tokens = int(previous.get("stage_input_tokens", 0)) if resume else 0
    started = time.monotonic(); log_path = args.out / "steps.jsonl"
    with log_path.open("a") as log:
        for update in range(start, args.until_updates):
            cpt = long_lm.get(update, 3000 + length)
            if fixed_sft:
                long_sft_source, long_sft_local_index = fixed_sft[update]
                sft = long_sft.sources[long_sft_source][long_sft_local_index]
            else:
                long_sft_source, long_sft_local_index = None, None
                sft = long_sft.get(update, 4000 + length)
            replay = short_replay(short_sft.get(update + spec["short_offset"], 5000), terminal_id)
            validate_long(cpt, length, terminal_id, lm=True)
            validate_long(sft, length, terminal_id, lm=False)
            update_input_tokens = int(cpt.shape[0] - 1) + len(sft["input_ids"]) - 1 + len(replay["input_ids"]) - 1
            stage_input_tokens += update_input_tokens
            lr = PEAK_LR * min(1.0, (update + 1) / WARMUP_UPDATES)
            for group in optimizer.param_groups: group["lr"] = lr
            record = step(model, wrapper, optimizer, cpt, sft, replay, native_table=native,
                          weights=LOSS_WEIGHTS, chunk_size=128, amp=True)
            state = {"arm": spec["arm"], "stage": args.stage, "stage_update": update + 1,
                     "input_tokens": stage_input_tokens, "stage_input_tokens": stage_input_tokens,
                     "stage_cpt_tokens": (update + 1) * length, "seed": source_plan["seed"],
                     "scientific_config": scientific,
                     "initialized_from": str(initialize) if initialize else None,
                     "asset_identity_policy": "user_attested_clone/no_sha_validation"}
            record.update(stage=args.stage, stage_update=update + 1, length=length, lr=lr,
                          update_primary_input_tokens=update_input_tokens,
                          long_lm_pool_index=update, long_sft_pool_index=update,
                          long_sft_source_index=long_sft_source,
                          long_sft_source_local_index=long_sft_local_index,
                          short_sft_pool_index=update + spec["short_offset"])
            log.write(json.dumps(record) + "\n"); log.flush()
            if (update + 1) % 50 == 0: save_bundle(args.out / "resume", wrapper, optimizer, state)
    endpoint = args.out / f"checkpoint-{args.until_updates}"
    save_bundle(endpoint, wrapper, optimizer, state)
    if (args.out / "resume").is_dir(): shutil.rmtree(args.out / "resume")
    final_stage_endpoint = args.until_updates == spec["schedule_updates"] or continuation_stage
    completion = {"status": "TRAINING_ENDPOINT_COMPLETE_NEEDS_EVALUATION" if final_stage_endpoint else "INTERMEDIATE_STOP_RESUMABLE_NEEDS_EVALUATION", "stage": args.stage,
                  "arm": spec["arm"], "updates": args.until_updates,
                  "schedule_updates": spec["schedule_updates"], "length": length,
                  "elapsed_seconds": time.monotonic() - started,
                  "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                  "checkpoint": str(endpoint.resolve()), "scientific_config": scientific}
    (args.out / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
