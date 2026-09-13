#!/usr/bin/env python3
"""Token-budgeted, resumable OLMo recovery-v2 training."""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.evq_recovery.data import JsonlIndex
from .recovery_v2_runtime import ARMS, backward_family, group_optimizer, load_model, table_for_config

FAMILIES = ("short_lm", "short_sft", "long_lm", "long_sft", "long_synthetic")
COUNTS_A = (5, 5, 5, 4, 1)
COUNTS_B = (4, 4, 6, 4, 2)
UPDATE_INPUT_TARGET = 65_536
DEFAULT_INPUT_BUDGET = 262_144_000


def phase_cycle(phase: str, cycle: int) -> list[str]:
    counts = COUNTS_A if phase == "A" else COUNTS_B
    values = [family for family, count in zip(FAMILIES, counts) for _ in range(count)]
    random.Random(42 + cycle + (0 if phase == "A" else 1_000_000)).shuffle(values)
    return values


def family_at(update: int, actual_input_tokens: int, total_budget: int) -> tuple[str, str]:
    phase = "A" if actual_input_tokens < total_budget * .25 else "B"
    cycle, offset = divmod(update, 20)
    return phase_cycle(phase, cycle)[offset], phase


def lr_scale(actual_tokens: int, total: int) -> float:
    warmup = total * .05
    if actual_tokens <= warmup: return max(0.0, actual_tokens / warmup)
    progress = min(1.0, (actual_tokens - warmup) / (total - warmup))
    return .1 + .45 * (1 + math.cos(math.pi * progress))


class Pool:
    def __init__(self, records: list[dict]):
        self.sources = []
        self.records = []
        for record in records:
            path = Path(record["path"])
            if record["format"] == "npy": source = np.load(path, mmap_mode="r", allow_pickle=False)
            elif record["format"] == "jsonl": source = JsonlIndex(path)
            else: raise ValueError("unknown pool format")
            if len(source) != int(record["rows"]) or len(source) == 0: raise ValueError("pool row count mismatch")
            self.sources.append(source); self.records.append({"format": record["format"], "rows": int(record["rows"])})
        self.ends = []; total = 0
        for source in self.sources: total += len(source); self.ends.append(total)
        self.total = total; self.order_epoch = None; self.order = None

    def get(self, counter: int, seed: int):
        epoch, position = divmod(counter, self.total)
        if self.order_epoch != epoch:
            self.order = list(range(self.total)); random.Random(seed + epoch).shuffle(self.order); self.order_epoch = epoch
        global_index = self.order[position]
        source_index = bisect.bisect_right(self.ends, global_index)
        previous = 0 if source_index == 0 else self.ends[source_index - 1]
        return self.sources[source_index][global_index - previous]


def pool_name(family: str, phase: str) -> str:
    if family.startswith("short_"): return family
    return family + ("_8192" if phase == "A" else "_16384")


def input_tokens(example) -> int:
    return int(example.shape[0] - 1 if isinstance(example, np.ndarray) else len(example["input_ids"]) - 1)


def semantic_config(arm, manifest, schedule_total):
    pool_layout = {name: [{"format": row["format"], "rows": int(row["rows"])} for row in records]
                   for name, records in sorted(manifest["pools"].items())}
    return {"arm": arm, "seed": 42, "schedule_input_tokens": schedule_total, "pool_layout": pool_layout,
        "phase_boundary": .25, "cycles": {"A": list(COUNTS_A), "B": list(COUNTS_B)},
        "phase_cycle_semantics": "family offset is global update modulo 20; crossing the 25% token boundary may enter B in a partial cycle",
        "update_input_target": UPDATE_INPUT_TARGET, "lora": {"qkvo": [64, 128], "ffn": [16, 32], "dropout": .05},
        "group_lrs": {"qk": 5e-5, "vo": 2.5e-5, "ffn": 1.25e-5}, "short_kl_weight": .2}


def rng_state(): return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "numpy": np.random.get_state(), "python": random.getstate()}
def restore_rng(state): torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"]); np.random.set_state(state["numpy"]); random.setstate(state["python"])


def save(path, wrapper, optimizer, state):
    temporary = path.with_name(path.name + ".incomplete"); old = path.with_name(path.name + ".old")
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True); wrapper.save_pretrained(temporary, safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "rng": rng_state()}, temporary / "training.pt")
    (temporary / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    if old.exists(): shutil.rmtree(old)
    if path.exists(): os.replace(path, old)
    os.replace(temporary, path)
    if old.exists(): shutil.rmtree(old)


def recoverable(path: Path) -> Path:
    if path.is_dir() and (path / "state.json").is_file() and (path / "training.pt").is_file(): return path
    old = path.with_name(path.name + ".old")
    if old.is_dir() and (old / "state.json").is_file() and (old / "training.pt").is_file(): return old
    raise FileNotFoundError(f"no complete checkpoint at {path} or {old}")


def validate_cuda():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 CUDA required")
    if torch.cuda.get_device_capability() != (12, 0) or "5090" not in torch.cuda.get_device_name().upper(): raise RuntimeError("requires RTX 5090 sm120")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.enable_flash_sdp(True); torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_cudnn_sdp(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train"); train.add_argument("--data", type=Path, required=True); train.add_argument("--model", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True); train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--schedule-input-tokens", type=int, default=DEFAULT_INPUT_BUDGET)
    train.add_argument("--until-input-tokens", type=int); train.add_argument("--resume", type=Path); train.add_argument("--execute", action="store_true")
    args = parser.parse_args(); manifest = json.loads(args.data.read_text())
    missing = [name for name in ("short_lm", "short_sft", "long_lm_8192", "long_lm_16384", "long_sft_8192", "long_sft_16384", "long_synthetic_8192", "long_synthetic_16384") if name not in manifest["pools"]]
    if missing: raise ValueError("missing pools: " + ",".join(missing))
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model, local_files_only=True); table = table_for_config(config, args.arm)
    until = args.schedule_input_tokens if args.until_input_tokens is None else args.until_input_tokens
    if not 0 < until <= args.schedule_input_tokens: raise ValueError("until must be within the fixed schedule token budget")
    contract = semantic_config(args.arm, manifest, args.schedule_input_tokens)
    metadata = {"status": "METADATA_ONLY" if not args.execute else "STARTING", "training_type": "recovery_v2_multifamily_token_budgeted_lora", "scientific_config": contract,
        "table": table, "asset_identity_policy": "user_attested_clone/no_sha_validation"}
    print(json.dumps(metadata), flush=True)
    if not args.execute: return
    resume = recoverable(args.resume.resolve()) if args.resume else None
    if (args.out / "steps.jsonl").is_file() and (args.out / "steps.jsonl").stat().st_size and resume is None:
        raise FileExistsError("nonempty steps.jsonl requires --resume")
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "plan.json").write_text(json.dumps(metadata, indent=2) + "\n")
    validate_cuda(); pools = {name: Pool(records) for name, records in manifest["pools"].items()}
    model, wrapper, table = load_model(args.model, args.arm, resume, training=True); optimizer = group_optimizer(model)
    state = {"semantic_config": contract, "arm": args.arm, "seed": 42,
             "profile": "recovery_v2_multifamily_token_budgeted_lora", "update": 0,
             "input_tokens": 0, "actual_input_tokens": 0,
             "pool_counters": {name: 0 for name in pools}, "prediction_tokens": 0}
    if resume:
        previous = json.loads((resume / "state.json").read_text())
        if previous["semantic_config"] != contract: raise ValueError("resume semantic configuration mismatch")
        if previous["actual_input_tokens"] > until: raise ValueError("resume is beyond requested stop")
        state = previous; saved = torch.load(resume / "training.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"]); restore_rng(saved["rng"])
    thresholds = [args.schedule_input_tokens * fraction for fraction in (.25, .5, .75, 1.)]
    crossed = {int(value) for value in state.get("saved_thresholds", [])}; started = time.monotonic()
    native_values = np.asarray(table_for_config(config, "Native")["values_float32"], dtype=np.float32)
    with (args.out / "steps.jsonl").open("a") as log:
        while state["actual_input_tokens"] < until:
            family, phase = family_at(state["update"], state["actual_input_tokens"], args.schedule_input_tokens)
            name = pool_name(family, phase); examples = []; batch_tokens = 0
            while batch_tokens < UPDATE_INPUT_TARGET:
                counter = state["pool_counters"][name]; example = pools[name].get(counter, 42 + sum(map(ord, name)))
                state["pool_counters"][name] = counter + 1; examples.append(example); batch_tokens += input_tokens(example)
            before = state["actual_input_tokens"]; optimizer.zero_grad(set_to_none=True)
            record = backward_family(model, wrapper, examples, family, native_values)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            state["actual_input_tokens"] += batch_tokens; state["input_tokens"] = state["actual_input_tokens"]
            scale = lr_scale(state["actual_input_tokens"], args.schedule_input_tokens)
            for group in optimizer.param_groups: group["lr"] = group["peak_lr"] * scale
            optimizer.step(); state["update"] += 1; state["prediction_tokens"] += record["prediction_tokens"]
            record.update(update=state["update"], phase=phase, pool=name, actual_input_tokens=batch_tokens,
                          cumulative_actual_input_tokens=state["actual_input_tokens"], lr_scale=scale, clip_norm=float(norm),
                          phase_cycle_offset=(state["update"] - 1) % 20,
                          phase_cycle_semantics="global update modulo 20; boundary may begin B mid-cycle")
            log.write(json.dumps(record) + "\n"); log.flush()
            newly = [int(value) for value in thresholds if before < value <= state["actual_input_tokens"] and int(value) not in crossed]
            for value in newly: crossed.add(value); state["saved_thresholds"] = sorted(crossed); save(args.out / f"checkpoint-{value}", wrapper, optimizer, state)
            if state["update"] % 64 == 0: state["saved_thresholds"] = sorted(crossed); save(args.out / "resume", wrapper, optimizer, state)
    state["saved_thresholds"] = sorted(crossed)
    final_schedule = state["actual_input_tokens"] >= args.schedule_input_tokens
    endpoint = args.out / ("final" if final_schedule else f'stop-{state["actual_input_tokens"]}')
    save(endpoint, wrapper, optimizer, state)
    completion = {"status": "TRAINING_ENDPOINT_COMPLETE_NEEDS_EVALUATION" if final_schedule else "INTERMEDIATE_STOP_RESUMABLE",
        "schedule_input_tokens": args.schedule_input_tokens, "requested_stop_input_tokens": until,
        "actual_input_tokens": state["actual_input_tokens"], "overshoot_tokens": state["actual_input_tokens"] - until,
        "checkpoint": str(endpoint), "updates": state["update"], "prediction_tokens": state["prediction_tokens"], "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()), "scientific_config": contract}
    (args.out / "completion.json").write_text(json.dumps(completion, indent=2) + "\n"); print(json.dumps(completion))


if __name__ == "__main__": main()
