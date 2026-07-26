#!/usr/bin/env python3
"""Train one fresh Llama-3-8B EVQ-LoRA with paired counterfactual supervision.

The learned LoRA state is initialized from scratch.  The historical seed-42
adapter is not loaded.  Its fixed ``custom_inv_freq.pt`` may be supplied only
as the exact, non-learned EVQ frequency identity.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AttentionMaskInterface, AutoConfig, AutoModelForCausalLM

from experiments.lora_evq_v2.train_evq_lora import (
    PACKED_FREE_CAUSAL_SDPA_BACKEND,
    configure_packed_free_causal_sdpa,
    inject_inv_freq,
    load_frequency_artifact,
    verify_model_inv_freq,
)

from .common import (
    TRAIN_LENGTH,
    append_jsonl,
    atomic_json,
    canonical_json_sha256,
    configure_cuda,
    sha256_file,
)
from .train import (
    FixedView,
    TrainingBackbone,
    causal_parts,
    cosine_lr,
    fused_loss_module,
    packed_free_causal_mask,
    seed_everything,
    trainable_parameters,
)


PAIR_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
)
FAMILY_PATTERN = ("counterfactual", "counterfactual", "natural")
PAIR_STATUS = "LLAMA8B_8K_COUNTERFACTUAL_PAIRS_READY_V1"
READY_STATUS = "LLAMA8B_FRESH_EVQ_COUNTERFACTUAL_NO_GPU_READY_V1"
RESULT_STATUS = "LLAMA8B_FRESH_EVQ_COUNTERFACTUAL_COMPLETE_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("prepare", "preflight", "probe", "train"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frequency-artifact", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--pair-data", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--counterfactual-margin", type=float, default=1.0)
    parser.add_argument(
        "--counterfactual-margin-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--discarded-steady-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--required-gpu-substring", default="RTX PRO 6000")
    return parser.parse_args()


def load_metadata(view: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (view / "metadata.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def subsequence_starts(haystack: np.ndarray, needle: np.ndarray) -> list[int]:
    if len(needle) == 0 or len(needle) > len(haystack):
        return []
    first = np.flatnonzero(haystack[: len(haystack) - len(needle) + 1] == needle[0])
    return [
        int(start)
        for start in first
        if np.array_equal(haystack[start : start + len(needle)], needle)
    ]


def _pair_candidates(
    records: list[dict[str, Any]],
    input_ids: np.ndarray,
    assistant_mask: np.ndarray,
) -> dict[int, tuple[np.ndarray, int]]:
    candidates: dict[int, tuple[np.ndarray, int]] = {}
    for record in records:
        row = int(record["row"])
        length = int(record["encoded_length"])
        mask = np.asarray(assistant_mask[row, :length], dtype=np.uint8)
        target_positions = np.flatnonzero(mask)
        if len(target_positions) < 2:
            raise RuntimeError(f"counterfactual row {row} has no answer")
        if not np.array_equal(
            target_positions,
            np.arange(target_positions[0], target_positions[-1] + 1),
        ):
            raise RuntimeError(f"counterfactual row {row} target is not contiguous")
        answer = np.asarray(
            input_ids[row, target_positions[:-1]],
            dtype=np.int64,
        )
        prompt = np.asarray(
            input_ids[row, : int(target_positions[0])],
            dtype=np.int64,
        )
        starts = subsequence_starts(prompt, answer)
        if len(starts) != 1:
            raise RuntimeError(
                f"row {row} source answer occurrence count is {len(starts)}"
            )
        candidates[row] = (answer, starts[0])
    return candidates


def build_pair_split(
    *,
    destination: Path,
    records: list[dict[str, Any]],
    input_ids: np.ndarray,
    assistant_mask: np.ndarray,
    pad_id: int,
) -> dict[str, Any]:
    candidates = _pair_candidates(records, input_ids, assistant_mask)
    by_task_and_tokens: dict[tuple[str, int], list[int]] = {}
    for record in records:
        row = int(record["row"])
        answer, _ = candidates[row]
        by_task_and_tokens.setdefault(
            (str(record["task"]), len(answer)),
            [],
        ).append(row)
    all_by_task_and_tokens: dict[tuple[str, int], list[int]] = {}
    all_records = [
        record
        for record in load_metadata(Path(records[0]["_view_root"]))
        if record.get("source") == "official_ruler"
        and record.get("task") in PAIR_TASKS
    ]
    all_candidates = _pair_candidates(all_records, input_ids, assistant_mask)
    for record in all_records:
        row = int(record["row"])
        answer, _ = all_candidates[row]
        all_by_task_and_tokens.setdefault(
            (str(record["task"]), len(answer)),
            [],
        ).append(row)

    destination.mkdir(parents=True)
    pair_ids = np.lib.format.open_memmap(
        destination / "input_ids.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(len(records), 2, TRAIN_LENGTH),
    )
    pair_labels = np.lib.format.open_memmap(
        destination / "labels.npy",
        mode="w+",
        dtype=np.int32,
        shape=(len(records), 2, TRAIN_LENGTH),
    )
    pair_ids[:] = int(pad_id)
    pair_labels[:] = -100
    rows_path = destination / "rows.jsonl"
    used_alternates: set[tuple[int, int]] = set()
    supervised = 0
    with rows_path.open("w", encoding="utf-8") as handle:
        for pair_index, record in enumerate(records):
            row = int(record["row"])
            length = int(record["encoded_length"])
            gold, source_start = candidates[row]
            key = (str(record["task"]), len(gold))
            pool = all_by_task_and_tokens[key]
            alternate_row = next(
                (
                    candidate
                    for candidate in pool
                    if candidate != row
                    and (row, candidate) not in used_alternates
                    and not subsequence_starts(
                        np.asarray(
                            input_ids[row, : int(np.flatnonzero(
                                assistant_mask[row, :length]
                            )[0])],
                            dtype=np.int64,
                        ),
                        all_candidates[candidate][0],
                    )
                ),
                None,
            )
            if alternate_row is None:
                fallback_rows = [
                    candidate
                    for (task, _), local_rows
                    in all_by_task_and_tokens.items()
                    if task == key[0]
                    for candidate in local_rows
                    if candidate != row
                ]
                if not fallback_rows:
                    raise RuntimeError(
                        f"no alternate source for task={key[0]}"
                    )
                alternate_row = int(fallback_rows[0])
                donor = all_candidates[alternate_row][0]
                alternate = gold.copy()
                changed = False
                for position in range(len(alternate) - 1, -1, -1):
                    for donor_token in donor:
                        if int(donor_token) != int(alternate[position]):
                            alternate[position] = donor_token
                            changed = True
                            break
                    if changed:
                        break
                if not changed:
                    raise RuntimeError(
                        f"could not synthesize token-matched alternate "
                        f"for task={key[0]}"
                    )
            else:
                alternate = all_candidates[int(alternate_row)][0]
            used_alternates.add((row, int(alternate_row)))
            if len(alternate) != len(gold) or np.array_equal(alternate, gold):
                raise RuntimeError("counterfactual alternate geometry drift")

            original = np.asarray(input_ids[row], dtype=np.uint32)
            swapped = original.copy()
            swapped[source_start : source_start + len(gold)] = alternate
            answer_positions = np.flatnonzero(assistant_mask[row])
            answer_without_eos = answer_positions[:-1]
            if len(answer_without_eos) != len(gold):
                raise RuntimeError("answer-mask geometry drift")
            swapped[answer_without_eos] = alternate
            pair_ids[pair_index, 0] = original
            pair_ids[pair_index, 1] = swapped
            pair_labels[pair_index, 0, answer_without_eos] = gold
            pair_labels[pair_index, 1, answer_without_eos] = alternate
            supervised += 2 * len(gold)
            handle.write(
                json.dumps(
                    {
                        "pair": pair_index,
                        "source_view_row": row,
                        "alternate_view_row": int(alternate_row),
                        "task": record["task"],
                        "split": record["split"],
                        "source_row_sha256": record["source_row_sha256"],
                        "answer_tokens": len(gold),
                        "source_token_start": int(source_start),
                        "answer_token_start": int(answer_without_eos[0]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    pair_ids.flush()
    pair_labels.flush()
    files = {
        name: {
            "sha256": sha256_file(destination / name),
            "size_bytes": (destination / name).stat().st_size,
        }
        for name in ("input_ids.npy", "labels.npy", "rows.jsonl")
    }
    manifest = {
        "status": PAIR_STATUS,
        "shape": [len(records), 2, TRAIN_LENGTH],
        "physical_training_length": TRAIN_LENGTH,
        "virtual_position_ids": False,
        "tasks": list(PAIR_TASKS),
        "variant_names": ["sourced", "value_swapped"],
        "answer_only": True,
        "source_content_swap": True,
        "supervised_answer_tokens": int(supervised),
        "files": files,
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def prepare_pairs(args: argparse.Namespace) -> dict[str, Any]:
    root = args.pair_data.resolve()
    if root.exists():
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("status") != PAIR_STATUS:
            raise RuntimeError("existing counterfactual data status drift")
        return manifest
    incomplete = root.with_name(root.name + ".incomplete")
    if incomplete.exists():
        shutil.rmtree(incomplete)
    incomplete.mkdir(parents=True)
    view = FixedView(args.training_view.resolve())
    metadata = load_metadata(args.training_view.resolve())
    if len(metadata) != len(view.input_ids):
        raise RuntimeError("training-view metadata row-count drift")
    selected = [
        {**record, "_view_root": str(args.training_view.resolve())}
        for record in metadata
        if record.get("source") == "official_ruler"
        and record.get("task") in PAIR_TASKS
    ]
    train = [record for record in selected if record["split"] == "train"]
    calibration = [
        record for record in selected if record["split"] == "validation"
    ]
    if (len(train), len(calibration)) != (576, 24):
        raise RuntimeError(
            f"counterfactual split drift: {(len(train), len(calibration))}"
        )
    pad_id = int(view.manifest["tokenizer"]["pad_token_id"])
    train_manifest = build_pair_split(
        destination=incomplete / "train",
        records=train,
        input_ids=view.input_ids,
        assistant_mask=view.assistant_mask,
        pad_id=pad_id,
    )
    calibration_manifest = build_pair_split(
        destination=incomplete / "calibration",
        records=calibration,
        input_ids=view.input_ids,
        assistant_mask=view.assistant_mask,
        pad_id=pad_id,
    )
    train_hashes = {record["source_row_sha256"] for record in train}
    calibration_hashes = {
        record["source_row_sha256"] for record in calibration
    }
    if train_hashes & calibration_hashes:
        raise RuntimeError("counterfactual train/calibration source overlap")
    manifest = {
        "status": PAIR_STATUS,
        "source_training_view": str(args.training_view.resolve()),
        "source_training_view_manifest_sha256": sha256_file(
            args.training_view.resolve() / "manifest.json"
        ),
        "physical_training_length": TRAIN_LENGTH,
        "virtual_position_ids": False,
        "tasks": list(PAIR_TASKS),
        "train_source_rows": len(train),
        "calibration_source_rows": len(calibration),
        "train_calibration_source_overlap": 0,
        "sets": {
            "train": {
                "manifest": train_manifest,
                "manifest_sha256": sha256_file(
                    incomplete / "train" / "manifest.json"
                ),
            },
            "calibration": {
                "manifest": calibration_manifest,
                "manifest_sha256": sha256_file(
                    incomplete / "calibration" / "manifest.json"
                ),
            },
        },
    }
    atomic_json(incomplete / "manifest.json", manifest)
    incomplete.replace(root)
    return manifest


class PairView:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("status") != PAIR_STATUS:
            raise RuntimeError("counterfactual pair-set status drift")
        expected = tuple(int(value) for value in self.manifest["shape"])
        if expected[1:] != (2, TRAIN_LENGTH):
            raise RuntimeError("counterfactual pair shape drift")
        for name, receipt in self.manifest["files"].items():
            path = root / name
            if (
                path.stat().st_size != int(receipt["size_bytes"])
                or sha256_file(path) != receipt["sha256"]
            ):
                raise RuntimeError(f"counterfactual pair file drift: {path}")
        self.input_ids = np.load(
            root / "input_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.labels = np.load(
            root / "labels.npy", mmap_mode="r", allow_pickle=False
        )
        if self.input_ids.dtype != np.uint32 or self.labels.dtype != np.int32:
            raise RuntimeError("counterfactual pair dtype drift")
        if tuple(self.input_ids.shape) != expected:
            raise RuntimeError("counterfactual pair array shape drift")


def protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "parent_adapter": None,
        "fresh_lora_initialization": True,
        "method": "evq_cosh",
        "physical_training_length": TRAIN_LENGTH,
        "virtual_position_ids": False,
        "long_context_backward": False,
        "steps": int(args.steps),
        "family_pattern": list(FAMILY_PATTERN),
        "micro_batch_size": int(args.micro_batch_size),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "global_batch_size": int(
            args.micro_batch_size * args.gradient_accumulation_steps
        ),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "counterfactual_margin": float(args.counterfactual_margin),
        "counterfactual_margin_weight": float(
            args.counterfactual_margin_weight
        ),
        "compile_mode": args.compile_mode,
        "seed": int(args.seed),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    config = AutoConfig.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    if (
        config.model_type != "llama"
        or int(config.hidden_size) != 4096
        or int(config.num_hidden_layers) != 32
        or int(config.max_position_embeddings) != TRAIN_LENGTH
    ):
        raise RuntimeError("Llama-3-8B checkpoint geometry drift")
    view = FixedView(args.training_view.resolve())
    pairs = json.loads(
        (args.pair_data.resolve() / "manifest.json").read_text()
    )
    if pairs.get("status") != PAIR_STATUS:
        raise RuntimeError("counterfactual data is not READY")
    inv_freq, metadata, frequency = load_frequency_artifact(
        args.frequency_artifact.resolve(),
        expected_method="evq_cosh",
    )
    if (
        inv_freq.shape != (64,)
        or int(metadata["head_dim"]) != 128
        or not math.isclose(float(metadata["base"]), 500_000.0)
        or not math.isclose(float(metadata["tau"]), 1.414)
        or metadata.get("midpoint") is not True
    ):
        raise RuntimeError("EVQ frequency identity drift")
    if args.output.exists() or args.output.with_name(
        args.output.name + ".incomplete"
    ).exists():
        raise FileExistsError(args.output)
    if int(args.micro_batch_size) != 2 or int(
        args.gradient_accumulation_steps
    ) != 4:
        raise RuntimeError("registered Blackwell execution shape drift")
    if int(args.steps) != 300 or int(args.seed) != 42:
        raise RuntimeError("registered scientific schedule drift")
    receipt = {
        "status": READY_STATUS,
        "reviewer_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "existing_evidence": (
            "seed-42 EVQ-LoRA improves external 16K/32K NLL but lacks "
            "matched downstream capability"
        ),
        "smallest_missing_evidence": (
            "one fresh EVQ-LoRA trained with source-dependent supervision"
        ),
        "stop_condition": (
            "stop on non-finite loss/gradients, Flash fallback, identity "
            "drift, OOM, or no finite first optimizer step"
        ),
        "inputs": {
            "checkpoint": str(checkpoint),
            "training_view": str(args.training_view.resolve()),
            "training_view_manifest_sha256": sha256_file(
                args.training_view.resolve() / "manifest.json"
            ),
            "pair_data": str(args.pair_data.resolve()),
            "pair_data_manifest_sha256": sha256_file(
                args.pair_data.resolve() / "manifest.json"
            ),
            "frequency_artifact": str(
                args.frequency_artifact.resolve()
            ),
            "frequency_artifact_sha256": sha256_file(
                args.frequency_artifact.resolve()
            ),
            "frequency": frequency,
        },
        "protocol": protocol(args),
        "code": {
            "trainer": str(Path(__file__).resolve()),
            "trainer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "training_view_protocol": view.manifest["protocol"],
    }
    atomic_json(args.ready_receipt.resolve(), receipt)
    return receipt


def validate_ready(args: argparse.Namespace) -> dict[str, Any]:
    receipt = json.loads(
        args.ready_receipt.resolve().read_text(encoding="utf-8")
    )
    if receipt.get("status") != READY_STATUS:
        raise RuntimeError("fresh Counterfactual READY status drift")
    if receipt["protocol"] != protocol(args):
        raise RuntimeError("fresh Counterfactual READY protocol drift")
    checks = {
        "training_view_manifest_sha256": (
            args.training_view.resolve() / "manifest.json"
        ),
        "pair_data_manifest_sha256": (
            args.pair_data.resolve() / "manifest.json"
        ),
        "frequency_artifact_sha256": args.frequency_artifact.resolve(),
    }
    for key, path in checks.items():
        if sha256_file(path) != receipt["inputs"][key]:
            raise RuntimeError(f"READY input hash drift: {path}")
    if (
        sha256_file(Path(__file__).resolve())
        != receipt["code"]["trainer_sha256"]
    ):
        raise RuntimeError("READY trainer hash drift")
    return receipt


def pair_batch(
    view: PairView,
    pair_indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    ids = np.asarray(view.input_ids[pair_indices], dtype=np.int64)
    labels = np.asarray(view.labels[pair_indices], dtype=np.int64)
    alternates = labels[:, ::-1, :].copy()
    contexts = torch.from_numpy(
        ids.reshape(-1, TRAIN_LENGTH).copy()
    ).to("cuda", non_blocking=True)
    targets = torch.from_numpy(
        labels.reshape(-1, TRAIN_LENGTH).copy()
    ).to("cuda", non_blocking=True)
    alternate_targets = torch.from_numpy(
        alternates.reshape(-1, TRAIN_LENGTH)
    ).to("cuda", non_blocking=True)
    supervised = int((targets[:, 1:] != -100).sum())
    return contexts, targets, alternate_targets, supervised


def natural_batch(
    view: FixedView,
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    ids = np.asarray(view.input_ids[indices], dtype=np.int64)
    mask = np.asarray(view.assistant_mask[indices], dtype=np.uint8)
    contexts = torch.from_numpy(ids.copy()).to("cuda", non_blocking=True)
    labels = torch.full_like(contexts, -100)
    next_ids = torch.from_numpy(ids[:, 1:].copy()).to(
        "cuda", non_blocking=True
    )
    next_mask = torch.from_numpy(mask[:, 1:].copy()).to(
        "cuda", non_blocking=True
    ).bool()
    labels[:, :-1] = torch.where(
        next_mask,
        next_ids,
        torch.full_like(next_ids, -100),
    )
    supervised = int(next_mask.sum())
    return contexts, labels, supervised


def counterfactual_loss(
    *,
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    labels: torch.Tensor,
    alternate_labels: torch.Tensor,
    margin: float,
    margin_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = labels[:, 1:]
    alternate = alternate_labels[:, 1:]
    mask = target != -100
    selected = hidden[:, :-1][mask]
    gold = target[mask]
    other = alternate[mask]
    logits = F.linear(selected, lm_head.weight).float()
    answer_ce = F.cross_entropy(logits, gold)
    source_mask = gold != other
    source_logits = logits[source_mask]
    source_gold = gold[source_mask]
    source_other = other[source_mask]
    preference = (
        source_logits.gather(1, source_gold[:, None]).squeeze(1)
        - source_logits.gather(1, source_other[:, None]).squeeze(1)
    )
    margin_loss = F.softplus(float(margin) - preference).mean()
    total = answer_ce + float(margin_weight) * margin_loss
    return total, {
        "answer_ce": float(answer_ce.detach()),
        "counterfactual_loss": float(margin_loss.detach()),
        "preference_mean": float(preference.detach().mean()),
        "preference_positive_fraction": float(
            (preference.detach() > 0).float().mean()
        ),
        "token_exact": float(
            logits.detach().argmax(dim=-1).eq(gold).float().mean()
        ),
    }


@torch.inference_mode()
def evaluate_pairs(
    model: PeftModel,
    view: PairView,
    *,
    pair_batch_size: int = 1,
) -> dict[str, Any]:
    model.eval()
    backbone_module, lm_head = causal_parts(model)
    backbone = TrainingBackbone(backbone_module)
    totals = {
        "tokens": 0,
        "nll": 0.0,
        "exact": 0,
        "preference": 0.0,
        "preference_positive": 0,
        "sequences": 0,
        "sequence_exact": 0,
    }
    for start in range(0, len(view.input_ids), pair_batch_size):
        indices = np.arange(
            start,
            min(len(view.input_ids), start + pair_batch_size),
        )
        contexts, labels, alternates, _ = pair_batch(view, indices)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = backbone(contexts)
        target = labels[:, 1:]
        other = alternates[:, 1:]
        mask = target != -100
        logits = F.linear(
            hidden[:, :-1][mask], lm_head.weight
        ).float()
        gold = target[mask]
        alternate = other[mask]
        nll = F.cross_entropy(logits, gold, reduction="none")
        predicted = logits.argmax(dim=-1)
        preference = (
            logits.gather(1, gold[:, None]).squeeze(1)
            - logits.gather(1, alternate[:, None]).squeeze(1)
        )
        cursor = 0
        for sequence_mask in mask:
            count = int(sequence_mask.sum())
            totals["sequences"] += 1
            totals["sequence_exact"] += int(
                predicted[cursor : cursor + count]
                .eq(gold[cursor : cursor + count])
                .all()
            )
            cursor += count
        count = int(gold.numel())
        totals["tokens"] += count
        totals["nll"] += float(nll.sum())
        totals["exact"] += int(predicted.eq(gold).sum())
        totals["preference"] += float(preference.sum())
        totals["preference_positive"] += int((preference > 0).sum())
        del contexts, labels, alternates, hidden, logits
    model.train()
    count = int(totals["tokens"])
    return {
        "pairs": int(len(view.input_ids)),
        "variants": int(totals["sequences"]),
        "answer_tokens": count,
        "mean_answer_nll": totals["nll"] / count,
        "token_exact": totals["exact"] / count,
        "sequence_exact": (
            totals["sequence_exact"] / totals["sequences"]
        ),
        "source_preference_mean": totals["preference"] / count,
        "source_preference_positive_fraction": (
            totals["preference_positive"] / count
        ),
    }


def ordered_cycles(
    values: np.ndarray,
    needed: int,
    *,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    chunks = []
    remaining = int(needed)
    while remaining > 0:
        permutation = generator.permutation(values)
        take = min(remaining, len(permutation))
        chunks.append(permutation[:take])
        remaining -= take
    return np.concatenate(chunks)


def run_training(args: argparse.Namespace, probe_only: bool) -> dict[str, Any]:
    ready = validate_ready(args)
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    incomplete.mkdir(parents=True)
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        str(
            output.parent.parent
            / "cache"
            / "torchinductor_pro6000"
        ),
    )
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(
        parents=True, exist_ok=True
    )
    runtime = configure_cuda()
    if args.required_gpu_substring not in runtime["gpu_name"]:
        raise RuntimeError(
            f"wrong GPU: {runtime['gpu_name']} does not contain "
            f"{args.required_gpu_substring!r}"
        )
    seed_everything(int(args.seed))
    base = AutoModelForCausalLM.from_pretrained(
        args.checkpoint.resolve(),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    base.config.use_cache = False
    attention_backend = configure_packed_free_causal_sdpa(base)
    AttentionMaskInterface.register(
        PACKED_FREE_CAUSAL_SDPA_BACKEND,
        packed_free_causal_mask,
    )
    inv_freq, frequency_metadata, frequency_receipt = (
        load_frequency_artifact(
            args.frequency_artifact.resolve(),
            expected_method="evq_cosh",
        )
    )
    inject_inv_freq(base, inv_freq)
    verify_model_inv_freq(base, inv_freq)
    lora_config = LoraConfig(
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)
    inject_inv_freq(model, inv_freq)
    frequency_verification = verify_model_inv_freq(model, inv_freq)
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    parameters = trainable_parameters(model)
    trainable_count = sum(parameter.numel() for parameter in parameters)
    if trainable_count != 54_525_952:
        raise RuntimeError(
            f"fresh QKVO LoRA trainable-count drift: {trainable_count}"
        )

    pair_root = args.pair_data.resolve()
    routing = PairView(pair_root / "train")
    calibration = PairView(pair_root / "calibration")
    natural = FixedView(args.training_view.resolve())
    metadata = load_metadata(args.training_view.resolve())
    natural_rows = np.asarray(
        [
            int(record["row"])
            for record in metadata
            if record.get("source") == "longalpaca_instruction_replay"
        ],
        dtype=np.int64,
    )
    if len(natural_rows) != 128:
        raise RuntimeError("natural replay row-count drift")

    backbone_module, lm_head = causal_parts(model)
    eager_backbone = TrainingBackbone(backbone_module)
    compiled_backbone = torch.compile(
        eager_backbone,
        fullgraph=False,
        dynamic=False,
        mode=args.compile_mode,
    )
    natural_loss_module = fused_loss_module()
    pair_indices = np.arange(len(routing.input_ids), dtype=np.int64)
    probe_pairs = pair_indices[: int(args.micro_batch_size) // 2]
    if len(probe_pairs) != 1:
        raise RuntimeError("registered probe requires one semantic pair")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    probe_durations = []
    probe_losses = []
    for _ in range(1 + int(args.discarded_steady_steps)):
        for parameter in parameters:
            parameter.grad = None
        contexts, labels, alternates, _ = pair_batch(
            routing, probe_pairs
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = compiled_backbone(contexts)
            loss, metrics = counterfactual_loss(
                hidden=hidden,
                lm_head=lm_head,
                labels=labels,
                alternate_labels=alternates,
                margin=float(args.counterfactual_margin),
                margin_weight=float(args.counterfactual_margin_weight),
            )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite compile/probe loss")
        loss.backward()
        torch.cuda.synchronize()
        probe_durations.append(time.perf_counter() - started)
        probe_losses.append(float(loss.detach()))
        del contexts, labels, alternates, hidden, loss
    for parameter in parameters:
        parameter.grad = None
    probe = {
        "compile_plus_first_step_seconds": probe_durations[0],
        "steady_step_seconds": probe_durations[1:],
        "steady_tokens_per_second": (
            int(args.micro_batch_size)
            * TRAIN_LENGTH
            * int(args.discarded_steady_steps)
            / sum(probe_durations[1:])
        ),
        "losses": probe_losses,
        "peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "compile_mode": args.compile_mode,
        "compile_cache": os.environ["TORCHINDUCTOR_CACHE_DIR"],
    }
    if probe_only:
        result = {
            "status": "LLAMA8B_FRESH_EVQ_COUNTERFACTUAL_PROBE_COMPLETE_V1",
            "runtime": runtime,
            "attention_backend": attention_backend,
            "frequency": frequency_receipt,
            "trainable_parameters": trainable_count,
            "probe": probe,
        }
        atomic_json(output, result)
        return result

    seed_everything(int(args.seed))
    initial_calibration = evaluate_pairs(model, calibration)
    seed_everything(int(args.seed))
    counterfactual_steps = sum(
        FAMILY_PATTERN[index % len(FAMILY_PATTERN)] == "counterfactual"
        for index in range(int(args.steps))
    )
    natural_steps = int(args.steps) - counterfactual_steps
    pair_order = ordered_cycles(
        pair_indices,
        counterfactual_steps
        * int(args.micro_batch_size)
        * int(args.gradient_accumulation_steps)
        // 2,
        seed=int(args.seed) + 81_001,
    )
    natural_order = ordered_cycles(
        natural_rows,
        natural_steps
        * int(args.micro_batch_size)
        * int(args.gradient_accumulation_steps),
        seed=int(args.seed) + 82_001,
    )
    pair_cursor = 0
    natural_cursor = 0
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    log_path = incomplete / "train_log.jsonl"
    started = time.perf_counter()
    last_time = started
    last_tokens = 0
    processed_tokens = 0
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, int(args.steps) + 1):
        family = FAMILY_PATTERN[(step - 1) % len(FAMILY_PATTERN)]
        lr = cosine_lr(
            step,
            int(args.steps),
            int(args.warmup_steps),
            float(args.learning_rate),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        step_metrics = []
        step_supervised = 0
        for _ in range(int(args.gradient_accumulation_steps)):
            if family == "counterfactual":
                pair_count = int(args.micro_batch_size) // 2
                indices = pair_order[
                    pair_cursor : pair_cursor + pair_count
                ]
                pair_cursor += pair_count
                contexts, labels, alternates, supervised = pair_batch(
                    routing, indices
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = compiled_backbone(contexts)
                    value, metrics = counterfactual_loss(
                        hidden=hidden,
                        lm_head=lm_head,
                        labels=labels,
                        alternate_labels=alternates,
                        margin=float(args.counterfactual_margin),
                        margin_weight=float(
                            args.counterfactual_margin_weight
                        ),
                    )
                del alternates, hidden
            else:
                count = int(args.micro_batch_size)
                indices = natural_order[
                    natural_cursor : natural_cursor + count
                ]
                natural_cursor += count
                contexts, labels, supervised = natural_batch(
                    natural, indices
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = compiled_backbone(contexts)
                    fused = natural_loss_module(
                        lm_head.weight,
                        hidden.reshape(-1, hidden.shape[-1]),
                        labels.reshape(-1),
                    )
                    value = (
                        fused.loss if hasattr(fused, "loss") else fused
                    )
                metrics = {"answer_ce": float(value.detach())}
                del hidden
            scaled = value / int(args.gradient_accumulation_steps)
            if not torch.isfinite(scaled):
                raise RuntimeError(f"non-finite loss at step {step}")
            scaled.backward()
            step_losses.append(float(value.detach()))
            step_metrics.append(metrics)
            step_supervised += supervised
            del contexts, labels, value, scaled
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        processed_tokens += (
            int(args.micro_batch_size)
            * int(args.gradient_accumulation_steps)
            * TRAIN_LENGTH
        )
        if step == 1 or step % 10 == 0 or step == int(args.steps):
            torch.cuda.synchronize()
            now = time.perf_counter()
            record = {
                "step": step,
                "total_steps": int(args.steps),
                "family": family,
                "loss": float(np.mean(step_losses)),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "supervised_tokens": int(step_supervised),
                "processed_input_tokens": processed_tokens,
                "elapsed_seconds": now - started,
                "interval_tokens_per_second": (
                    (processed_tokens - last_tokens)
                    / max(now - last_time, 1e-9)
                ),
                "peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
            }
            for key in step_metrics[0]:
                record[key] = float(
                    np.mean(
                        [
                            metric[key]
                            for metric in step_metrics
                            if key in metric
                        ]
                    )
                )
            append_jsonl(log_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)
            last_time = now
            last_tokens = processed_tokens
        if step in {100, 200}:
            recovery = incomplete / "recovery"
            temporary = incomplete / "recovery.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            model.save_pretrained(temporary / "adapter")
            torch.save(
                {
                    "completed_step": step,
                    "optimizer": optimizer.state_dict(),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                    "pair_cursor": pair_cursor,
                    "natural_cursor": natural_cursor,
                },
                temporary / "state.pt",
            )
            if recovery.exists():
                shutil.rmtree(recovery)
            temporary.replace(recovery)

    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    final_calibration = evaluate_pairs(model, calibration)
    model.save_pretrained(incomplete, safe_serialization=True)
    shutil.copy2(
        args.frequency_artifact.resolve(),
        incomplete / "custom_inv_freq.pt",
    )
    torch.save(
        {
            "completed_step": int(args.steps),
            "optimizer": optimizer.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "pair_cursor": pair_cursor,
            "natural_cursor": natural_cursor,
        },
        incomplete / "final_training_state.pt",
    )
    adapter = incomplete / "adapter_model.safetensors"
    result = {
        "status": RESULT_STATUS,
        "reviewer_concerns": ["R27bE.2", "R27bE.5", "AC.2"],
        "scientific_contract": protocol(args),
        "runtime": {
            **runtime,
            "python": platform.python_version(),
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "liger_kernel": importlib.metadata.version("liger-kernel"),
        },
        "execution": {
            "attention_backend": attention_backend,
            "flash_only": True,
            "precision": "bf16_autocast",
            "gradient_checkpointing": False,
            "optimizer": "fused_adamw",
            "optimizer_betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "compile_mode": args.compile_mode,
            "probe": probe,
            "training_seconds": training_seconds,
            "tokens_per_second": processed_tokens / training_seconds,
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
        },
        "frequency": {
            "artifact": frequency_receipt,
            "metadata": {
                key: value
                for key, value in frequency_metadata.items()
                if key != "inv_freq"
            },
            "verification": frequency_verification,
        },
        "parent_adapter": None,
        "fresh_lora_initialization": True,
        "trainable_parameters": trainable_count,
        "pair_data_manifest_sha256": sha256_file(
            pair_root / "manifest.json"
        ),
        "training_view_manifest_sha256": sha256_file(
            args.training_view.resolve() / "manifest.json"
        ),
        "ready_receipt_sha256": sha256_file(
            args.ready_receipt.resolve()
        ),
        "ready_protocol_sha256": canonical_json_sha256(
            ready["protocol"]
        ),
        "initial_counterfactual_calibration": initial_calibration,
        "final_counterfactual_calibration": final_calibration,
        "adapter_sha256": sha256_file(adapter),
        "frequency_artifact_sha256": sha256_file(
            incomplete / "custom_inv_freq.pt"
        ),
        "final_training_state_sha256": sha256_file(
            incomplete / "final_training_state.pt"
        ),
    }
    atomic_json(incomplete / "result.json", result)
    atomic_json(
        incomplete / "experiment_meta.json",
        {
            "status": "complete",
            "stage": "fresh_evq_counterfactual_8k",
            "method": "evq_cosh",
            "global_step": int(args.steps),
            "max_seq_len": TRAIN_LENGTH,
            "seed": int(args.seed),
            "parent_adapter": None,
            "adapter_sha256": result["adapter_sha256"],
            "frequency_sha256": result["frequency_artifact_sha256"],
        },
    )
    recovery = incomplete / "recovery"
    if recovery.exists():
        shutil.rmtree(recovery)
    incomplete.replace(output)
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        result = prepare_pairs(args)
    elif args.mode == "preflight":
        result = preflight(args)
    elif args.mode == "probe":
        result = run_training(args, probe_only=True)
    else:
        result = run_training(args, probe_only=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
