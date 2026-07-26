#!/usr/bin/env python3
"""Evaluate native or EVQ-adapted OLMo-2 Instruct on a RULER screen."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.models.olmo2 import modeling_olmo2

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
    configure_cuda,
    configure_ruler_flash_attention,
    greedy_generate,
    row_sha256,
    score_prediction,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    tensor_sha256,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_conversion import (
    install_adaptation,
    load_model,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_ood_factorial import (
    load_adapter,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.prepare_data import (
    atomic_json,
    sha256_file,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_4k_stage_a import (
    ready_checkpoint_digest,
)
from rebuttal.rebuttal_0723.experiments.olmo2_lora_maturity.train_screen import (
    apply_frequency,
)


SUPPORTED_DATA_STATUSES = (
    "OLMO2_INSTRUCT_RULER_SCREEN_PREPARED",
    "OLMO2_INSTRUCT_RULER_LONG_GAP_SCREEN_PREPARED",
)
SUPPORTED_TASKS = ("niah_single_1", "niah_single_2")
GENERATION_TOKENS = 128
NUMBER_PATTERN = re.compile(r"\b[0-9]+\b")
FREQUENCIES = (
    "native",
    "evq",
    "hybrid_native_low8",
    "hybrid_native_low16",
    "hybrid_native_low32",
    "hybrid_native_high16",
    "hybrid_native_ends16",
    "hybrid_evq_low4",
    "hybrid_evq_low8",
    "hybrid_evq_low12",
    "hybrid_evq_low16",
    "hybrid_evq_low24",
    "hybrid_evq_low32",
    "hybrid_evq_low40",
    "hybrid_evq_high8",
    "hybrid_evq_mid8",
    "hybrid_blend10",
    "hybrid_blend25",
    "hybrid_blend_evq_0p1pct",
    "hybrid_blend_evq_0p5pct",
    "hybrid_blend_evq_1pct",
    "hybrid_blend_evq_2pct",
    "hybrid_blend_evq_5pct",
    "hybrid_heads_evq1",
    "hybrid_heads_evq2",
    "hybrid_heads_evq4",
    "hybrid_heads_evq8",
    "hybrid_heads_custom",
)

LOG_FREQUENCY_BLEND_WEIGHTS = {
    "hybrid_blend10": 0.10,
    "hybrid_blend25": 0.25,
    "hybrid_blend_evq_0p1pct": 0.001,
    "hybrid_blend_evq_0p5pct": 0.005,
    "hybrid_blend_evq_1pct": 0.01,
    "hybrid_blend_evq_2pct": 0.02,
    "hybrid_blend_evq_5pct": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frequency", choices=FREQUENCIES, required=True
    )
    parser.add_argument(
        "--task",
        choices=SUPPORTED_TASKS,
        default="niah_single_1",
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--evq-head-indices", type=int, nargs="*")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[4_096, 8_192, 16_384],
    )
    parser.add_argument("--limit-per-length", type=int, default=20)
    return parser.parse_args()


def validate_data(
    root: Path,
    checkpoint: Path,
    task: str,
    lengths: tuple[int, ...],
    limit_per_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in SUPPORTED_DATA_STATUSES:
        raise RuntimeError("RULER screen data is not prepared")
    if manifest.get("task") != task:
        raise RuntimeError("RULER screen task drift")
    if Path(manifest["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError("RULER screen tokenizer checkpoint drift")
    tokenizer_digest = sha256_file(checkpoint / "tokenizer.json")
    if manifest.get("tokenizer_sha256") != tokenizer_digest:
        raise RuntimeError("RULER screen tokenizer hash drift")
    manifest_lengths = tuple(
        int(value) for value in manifest["lengths"]
    )
    if (
        not manifest_lengths
        or tuple(sorted(set(manifest_lengths))) != manifest_lengths
        or any(
            length not in {4_096, 8_192, 16_384}
            for length in manifest_lengths
        )
    ):
        raise RuntimeError("RULER screen manifest length drift")
    if any(length not in set(manifest_lengths) for length in lengths):
        raise RuntimeError("requested length is absent from data manifest")
    if not 1 <= limit_per_length <= int(manifest["samples_per_length"]):
        raise RuntimeError("limit-per-length exceeds prepared rows")

    selected: list[dict[str, Any]] = []
    file_receipts: dict[str, Any] = {}
    for length in lengths:
        entry = manifest["files"][str(length)]
        path = root / entry["relative_path"]
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise RuntimeError(f"RULER screen hash drift at L={length}")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != int(entry["rows"]):
            raise RuntimeError(f"RULER screen row-count drift at L={length}")
        for local_index, row in enumerate(rows[:limit_per_length]):
            row["_nominal_length"] = int(length)
            row["_local_index"] = int(local_index)
            selected.append(row)
        file_receipts[str(length)] = {
            "path": str(path),
            "sha256": digest,
            "rows": len(rows),
        }
    return (
        {
            "preparation_status": manifest["status"],
            "manifest_sha256": sha256_file(manifest_path),
            "ruler_commit": manifest["ruler_commit"],
            "tokenizer_sha256": tokenizer_digest,
            "files": file_receipts,
        },
        selected,
    )


def first_number_exact(prediction: str, references: list[str]) -> float:
    match = NUMBER_PATTERN.search(prediction)
    if match is None:
        return 0.0
    return float(match.group(0) in {str(value).strip() for value in references})


def load_completed(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["nominal_length"]), int(row["local_index"]))
        if key in completed:
            raise RuntimeError(f"duplicate completed RULER row: {key}")
        completed[key] = row
    return completed


def validate_adapter_metadata(
    metadata: dict[str, Any],
    *,
    checkpoint_digest: str,
    frequency: dict[str, Any],
    frequency_name: str,
    rank: int,
    alpha: float,
) -> None:
    expected = {
        "base_checkpoint_sha256": checkpoint_digest,
        "frequency": frequency_name,
        "frequency_sha256_float32": frequency[
            "active_sha256_float32"
        ],
        "adaptation": "qkvo_answer",
        "rank": int(rank),
        "alpha": float(alpha),
        "training_sequence_length": 4_096,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(
                f"adapter metadata drift for {name}: "
                f"{metadata.get(name)!r} != {value!r}"
            )


class HeadHybridRotaryEmbedding(nn.Module):
    """Return native/EVQ rotary phases independently for each head."""

    def __init__(
        self,
        native: torch.Tensor,
        evq: torch.Tensor,
        *,
        head_count: int,
        evq_head_indices: tuple[int, ...],
    ) -> None:
        super().__init__()
        inv_freq = native.repeat(int(head_count), 1)
        inv_freq[list(evq_head_indices)] = evq
        self.register_buffer(
            "inv_freq_by_head",
            inv_freq.to(torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = position_ids[:, None, :, None].float()
        frequencies = (
            self.inv_freq_by_head[None, :, None, :].float()
            * positions
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            return embedding.cos(), embedding.sin()


_ORIGINAL_APPLY_ROTARY_POS_EMB = modeling_olmo2.apply_rotary_pos_emb


def apply_head_aware_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.ndim != 4:
        return _ORIGINAL_APPLY_ROTARY_POS_EMB(
            q,
            k,
            cos,
            sin,
            position_ids=position_ids,
            unsqueeze_dim=unsqueeze_dim,
        )
    if (
        cos.shape != sin.shape
        or cos.shape[1] != q.shape[1]
        or cos.shape[-1] != q.shape[-1]
    ):
        raise RuntimeError("head-hybrid rotary shape drift")
    q_type, k_type = q.dtype, k.dtype
    q_embed = (q * cos) + (modeling_olmo2.rotate_half(q) * sin)
    k_embed = (k * cos) + (modeling_olmo2.rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)


def apply_screen_frequency(
    model: Any,
    frequency_name: str,
    *,
    custom_evq_head_indices: tuple[int, ...] = (),
) -> dict[str, Any]:
    if frequency_name in {"native", "evq"}:
        return apply_frequency(model, frequency_name)
    native = (
        model.model.rotary_emb.inv_freq.detach()
        .cpu()
        .to(torch.float32)
        .clone()
    )
    receipt = apply_frequency(model, "evq")
    evq = (
        model.model.rotary_emb.inv_freq.detach()
        .cpu()
        .to(torch.float32)
        .clone()
    )
    if torch.equal(native, evq):
        raise RuntimeError(
            "Native and EVQ frequency snapshots unexpectedly alias or match"
        )
    pair_count = int(native.numel())
    if pair_count != 64:
        raise RuntimeError(f"expected 64 rotary pairs, got {pair_count}")
    if frequency_name.startswith("hybrid_heads_"):
        head_count = int(model.config.num_attention_heads)
        if frequency_name == "hybrid_heads_custom":
            evq_head_indices = custom_evq_head_indices
        else:
            evq_head_count = int(
                frequency_name.removeprefix("hybrid_heads_evq")
            )
            evq_head_indices = tuple(
                range(head_count - evq_head_count, head_count)
            )
        if (
            not evq_head_indices
            or len(set(evq_head_indices)) != len(evq_head_indices)
            or any(
                index < 0 or index >= head_count
                for index in evq_head_indices
            )
            or len(evq_head_indices) >= head_count
        ):
            raise RuntimeError("invalid head-hybrid EVQ head indices")
        evq_head_indices = tuple(sorted(evq_head_indices))
        evq_head_count = len(evq_head_indices)
        rotary = HeadHybridRotaryEmbedding(
            native,
            evq,
            head_count=head_count,
            evq_head_indices=evq_head_indices,
        )
        model.model.rotary_emb = rotary
        modeling_olmo2.apply_rotary_pos_emb = (
            apply_head_aware_rotary_pos_emb
        )
        return {
            **receipt,
            "active_frequency": frequency_name,
            "active_sha256_float32": tensor_sha256(
                rotary.inv_freq_by_head
            ),
            "hybrid_axis": "attention_head",
            "hybrid_native_head_indices": [
                index
                for index in range(head_count)
                if index not in set(evq_head_indices)
            ],
            "hybrid_evq_head_indices": list(evq_head_indices),
            "hybrid_native_head_count": (
                head_count - evq_head_count
            ),
            "hybrid_evq_head_count": evq_head_count,
            "hybrid_flash_attention_compatible": True,
        }
    blend_weight = None
    if frequency_name == "hybrid_native_low8":
        native_indices = tuple(range(56, 64))
    elif frequency_name == "hybrid_native_low16":
        native_indices = tuple(range(48, 64))
    elif frequency_name == "hybrid_native_low32":
        native_indices = tuple(range(32, 64))
    elif frequency_name == "hybrid_native_high16":
        native_indices = tuple(range(0, 16))
    elif frequency_name == "hybrid_native_ends16":
        native_indices = tuple(range(0, 8)) + tuple(range(56, 64))
    elif frequency_name == "hybrid_evq_low4":
        native_indices = tuple(range(0, 60))
    elif frequency_name == "hybrid_evq_low8":
        native_indices = tuple(range(0, 56))
    elif frequency_name == "hybrid_evq_low12":
        native_indices = tuple(range(0, 52))
    elif frequency_name == "hybrid_evq_low16":
        native_indices = tuple(range(0, 48))
    elif frequency_name == "hybrid_evq_low24":
        native_indices = tuple(range(0, 40))
    elif frequency_name == "hybrid_evq_low32":
        native_indices = tuple(range(0, 32))
    elif frequency_name == "hybrid_evq_low40":
        native_indices = tuple(range(0, 24))
    elif frequency_name == "hybrid_evq_high8":
        native_indices = tuple(range(8, 64))
    elif frequency_name == "hybrid_evq_mid8":
        native_indices = tuple(range(0, 28)) + tuple(range(36, 64))
    elif frequency_name in LOG_FREQUENCY_BLEND_WEIGHTS:
        native_indices = ()
        blend_weight = LOG_FREQUENCY_BLEND_WEIGHTS[frequency_name]
    else:
        raise ValueError(f"unknown frequency {frequency_name!r}")
    if blend_weight is None:
        hybrid = evq.clone()
        hybrid[list(native_indices)] = native[list(native_indices)]
    else:
        hybrid = torch.exp(
            (1.0 - blend_weight) * torch.log(native)
            + blend_weight * torch.log(evq)
        )
    with torch.no_grad():
        model.model.rotary_emb.inv_freq.copy_(
            hybrid.to(
                device=model.model.rotary_emb.inv_freq.device,
                dtype=model.model.rotary_emb.inv_freq.dtype,
            )
        )
    receipt.update(
        {
            "active_frequency": frequency_name,
            "active_sha256_float32": tensor_sha256(hybrid),
            "hybrid_native_pair_indices": list(native_indices),
            "hybrid_evq_pair_indices": (
                [
                    index
                    for index in range(pair_count)
                    if index not in set(native_indices)
                ]
                if blend_weight is None
                else []
            ),
            "hybrid_native_pair_count": (
                len(native_indices) if blend_weight is None else 0
            ),
            "hybrid_evq_pair_count": (
                pair_count - len(native_indices)
                if blend_weight is None
                else 0
            ),
            "hybrid_blended_pair_count": (
                pair_count if blend_weight is not None else 0
            ),
            "hybrid_log_frequency_evq_weight": blend_weight,
            "hybrid_pair_order": (
                "index 0 is highest frequency; index 63 is lowest"
            ),
        }
    )
    return receipt


def main() -> None:
    args = parse_args()
    lengths = tuple(int(value) for value in args.lengths)
    if not lengths or tuple(sorted(set(lengths))) != lengths:
        raise RuntimeError("lengths must be unique and ascending")
    if (
        args.adapter is not None
        and args.frequency not in {"native", "evq"}
    ):
        raise RuntimeError("hybrid screen does not accept an adapter")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples_path = output / "examples.jsonl"
    checkpoint = args.checkpoint.resolve()
    ready_receipt = args.ready_receipt.resolve()
    checkpoint_digest = ready_checkpoint_digest(
        checkpoint, ready_receipt
    )
    data_receipt, rows = validate_data(
        args.data_root.resolve(),
        checkpoint,
        str(args.task),
        lengths,
        int(args.limit_per_length),
    )

    configure_cuda()
    runtime = {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "mem_efficient_sdp_enabled": (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
    }
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = load_model(checkpoint)
    frequency = apply_screen_frequency(
        model,
        args.frequency,
        custom_evq_head_indices=tuple(args.evq_head_indices or ()),
    )
    adapter_receipt = None
    if args.adapter is not None:
        readout = install_adaptation(
            model,
            "qkvo_answer",
            rank=int(args.rank),
            alpha=float(args.alpha),
        )
        if readout is not None:
            raise RuntimeError("RULER screen does not admit a readout")
        adapter_path = args.adapter.resolve()
        adapter_metadata = load_adapter(adapter_path, model, None)
        validate_adapter_metadata(
            adapter_metadata,
            checkpoint_digest=checkpoint_digest,
            frequency=frequency,
            frequency_name=args.frequency,
            rank=int(args.rank),
            alpha=float(args.alpha),
        )
        adapter_receipt = {
            "path": str(adapter_path),
            "sha256": sha256_file(adapter_path),
            "metadata": adapter_metadata,
        }

    configure_ruler_flash_attention(model)
    model.config.use_cache = True
    model.eval()
    model.to("cuda")
    completed = load_completed(examples_path)
    expected = len(rows)
    torch.cuda.reset_peak_memory_stats()

    with examples_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            length = int(row["_nominal_length"])
            local_index = int(row["_local_index"])
            key = (length, local_index)
            if key in completed:
                continue
            chat_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["input"]}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            prefix_ids = tokenizer(
                row.get("answer_prefix", ""),
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids
            input_ids = torch.cat((chat_ids, prefix_ids), dim=1).to(
                "cuda"
            )
            if input_ids.shape[1] + GENERATION_TOKENS > length:
                raise RuntimeError(
                    f"RULER prompt exceeds L={length}: "
                    f"{input_ids.shape[1]}+{GENERATION_TOKENS}"
                )
            started = time.perf_counter()
            output_ids = greedy_generate(
                model,
                input_ids,
                max_new_tokens=GENERATION_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            prediction = tokenizer.decode(
                output_ids[0].detach().cpu(),
                skip_special_tokens=True,
            )
            references = [str(value) for value in row["outputs"]]
            result = {
                "task": str(args.task),
                "nominal_length": length,
                "local_index": local_index,
                "source_row_index": int(row["index"]),
                "source_token_position_answer": int(
                    row["token_position_answer"]
                ),
                "row_sha256": row_sha256(
                    {
                        key: value
                        for key, value in row.items()
                        if not key.startswith("_")
                    }
                ),
                "input_tokens": int(input_ids.numel()),
                "generated_tokens": int(output_ids.numel()),
                "prediction": prediction,
                "references": references,
                "official_string_match": score_prediction(
                    prediction, references
                ),
                "first_number_exact": first_number_exact(
                    prediction, references
                ),
                "elapsed_seconds": elapsed,
            }
            handle.write(
                json.dumps(result, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
            handle.flush()
            completed[key] = result
            print(
                f"{len(completed)}/{expected} L={length} "
                f"match={result['official_string_match']:.3f} "
                f"first={result['first_number_exact']:.3f} "
                f"seconds={elapsed:.2f}",
                flush=True,
            )

    relevant = [
        completed[(length, local_index)]
        for length in lengths
        for local_index in range(int(args.limit_per_length))
    ]
    if len(relevant) != expected:
        raise RuntimeError("RULER screen result-count drift")
    cells = {}
    for length in lengths:
        selected = [
            row
            for row in relevant
            if int(row["nominal_length"]) == length
        ]
        cells[str(length)] = {
            "examples": len(selected),
            "official_string_match": sum(
                float(row["official_string_match"]) for row in selected
            )
            / len(selected),
            "first_number_exact": sum(
                float(row["first_number_exact"]) for row in selected
            )
            / len(selected),
            "mean_elapsed_seconds": sum(
                float(row["elapsed_seconds"]) for row in selected
            )
            / len(selected),
        }
    macro = sum(
        float(cell["official_string_match"]) for cell in cells.values()
    ) / len(cells)
    if not math.isfinite(macro):
        raise RuntimeError("non-finite RULER screen result")
    receipt = {
        "status": "OLMO2_INSTRUCT_RULER_SCREEN_COMPLETE",
        "metric_boundary": (
            "official RULER NIAH single-key autoregressive string-match "
            "screen; not the full 13-task RULER suite"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "ready_receipt_sha256": sha256_file(ready_receipt),
        "frequency": frequency,
        "adapter": adapter_receipt,
        "data": data_receipt,
        "protocol": {
            "task": str(args.task),
            "lengths": list(lengths),
            "limit_per_length": int(args.limit_per_length),
            "greedy": True,
            "maximum_new_tokens": GENERATION_TOKENS,
            "training_length_if_adapted": (
                4_096 if adapter_receipt is not None else None
            ),
            "attention": "flash_only_custom_kv_cache",
            "precision": "bf16_weights_and_autocast",
        },
        "runtime": {
            **runtime,
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "peak_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved()
            ),
        },
        "results": {
            "cells": cells,
            "macro_official_string_match": macro,
            "examples": len(relevant),
            "examples_sha256": sha256_file(examples_path),
        },
    }
    atomic_json(output / "results.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output / "results.json"),
                "cells": cells,
                "macro_official_string_match": macro,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
