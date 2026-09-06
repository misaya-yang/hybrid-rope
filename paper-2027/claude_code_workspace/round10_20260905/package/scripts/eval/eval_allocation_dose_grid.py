#!/usr/bin/env python3
"""Frozen-checkpoint dose-response evaluation of a fixed-support allocation grid.

No training, no learned parameter, no attention-amplitude change, no routing,
and no target length.  Model weights stay bitwise frozen; the only thing that
moves between arms is the shared rotary inverse-frequency buffer emitted by
``scripts/analysis/allocation_dose_grid.py``.

The evaluation protocol is deliberately the one the fresh-FineWeb session-s4
owner already uses, so the numbers land in the same frame as its published
Native row: identical documents, identical `1x/2x/4x` multipliers, and the
identical final-1024-token teacher-forced tail metric.  Two additional
endpoints come free from the same forward pass:

*   **dense whole-sequence NLL**, which separates a table that helps the tail
    from one that merely moves loss around inside the sequence; and
*   **per-1024-position-bin NLL**, which reports *where* in the sequence a
    reallocation starts paying.  The in-window/long-range trade-off predicted
    by the spectral-budget account is a statement about that profile, and no
    existing owner measures it.

Per-row values are written for every cell so a paired bootstrap conditions on
documents rather than on pooled tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.rope.inject import (  # noqa: E402
    apply_inv_freq_inplace,
    find_rotary_modules_with_inv_freq,
)

STATUS = "ALLOCATION_DOSE_GRID_EVAL_COMPLETE_V1"
BIN_WIDTH = 1024
LOGIT_CHUNK = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dose-manifest", type=Path, required=True)
    parser.add_argument("--long-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multipliers", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--native-length", type=int, default=4096)
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="0 evaluates every document; a small value gives a fast screen",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        help="table names from the manifest; default evaluates the whole grid",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_826)
    parser.add_argument("--authorize", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float32_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype="<f4")).tobytes()
    ).hexdigest()


def configure_flash_only() -> dict[str, Any]:
    """Flash-only SDPA with math attention explicitly disabled.

    Disabling the math and memory-efficient backends means an unsupported shape
    raises instead of silently falling back to the quadratic kernel, which is
    the Blackwell requirement in ``AGENTS.md`` §4.
    """

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    capability = torch.cuda.get_device_capability()
    return {
        "name": torch.cuda.get_device_name(),
        "capability": list(capability),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
    }


def load_rows(path: Path, multipliers: Iterable[int], limit: int) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {int(m): [] for m in multipliers}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            multiplier = int(row["multiplier"])
            if multiplier in grouped:
                grouped[multiplier].append(row)
    for multiplier, rows in grouped.items():
        rows.sort(key=lambda r: int(r["source_row"]))
        if limit > 0:
            grouped[multiplier] = rows[:limit]
        if not grouped[multiplier]:
            raise RuntimeError(f"no evaluation rows for multiplier {multiplier}")
    counts = {m: len(r) for m, r in grouped.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"multipliers must share a document count, got {counts}")
    return grouped


def load_model(checkpoint: Path, max_positions: int):
    from transformers import AutoModelForCausalLM

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    except TypeError:  # transformers < 5 spells the argument differently
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    model.config.max_position_embeddings = int(max_positions)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to("cuda")


@torch.inference_mode()
def score_row(*, model: Any, input_ids: torch.Tensor) -> np.ndarray:
    """Return the per-target-token NLL vector for one sequence."""

    length = int(input_ids.shape[1])
    position_ids = torch.arange(length, device="cuda", dtype=torch.long).unsqueeze(0)
    hidden = model.model(input_ids=input_ids, position_ids=position_ids).last_hidden_state[0]
    targets = input_ids[0, 1:]
    losses = torch.empty(length - 1, dtype=torch.float32, device="cuda")
    for start in range(0, length - 1, LOGIT_CHUNK):
        stop = min(start + LOGIT_CHUNK, length - 1)
        logits = model.lm_head(hidden[start:stop]).float()
        losses[start:stop] = F.cross_entropy(
            logits, targets[start:stop], reduction="none"
        )
        del logits
    del hidden
    return losses.detach().cpu().numpy().astype(np.float64)


def summarise_row(losses: np.ndarray, target_start: int, target_tokens: int) -> dict[str, Any]:
    """Tail metric, dense metric, and the per-position-bin profile.

    ``losses[i]`` is the NLL of the token at absolute position ``i + 1``.
    """

    positions = np.arange(1, losses.size + 1)
    tail = (positions >= int(target_start)) & (positions < int(target_start) + int(target_tokens))
    if not tail.any():
        raise RuntimeError("declared tail target does not intersect the scored positions")
    bins: dict[str, float] = {}
    for start in range(0, losses.size + 1, BIN_WIDTH):
        mask = (positions >= start) & (positions < start + BIN_WIDTH)
        if mask.any():
            bins[str(start)] = float(losses[mask].mean())
    return {
        "tail_nll": float(losses[tail].mean()),
        "tail_tokens": int(tail.sum()),
        "dense_nll": float(losses.mean()),
        "dense_tokens": int(losses.size),
        "position_bin_nll": bins,
    }


def paired_bootstrap(
    treatment: np.ndarray, control: np.ndarray, *, resamples: int, seed: int
) -> dict[str, float]:
    difference = treatment - control
    generator = np.random.default_rng(seed)
    index = generator.integers(0, difference.size, size=(resamples, difference.size))
    draws = difference[index].mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "rows_favouring_treatment": int((difference < 0).sum()),
        "rows": int(difference.size),
    }


def main() -> int:
    args = parse_args()
    if not args.authorize or os.environ.get("ALLOCATION_DOSE_GRID_GPU_AUTHORIZED") != "YES":
        raise PermissionError(
            "GPU evaluation requires --authorize and "
            "ALLOCATION_DOSE_GRID_GPU_AUTHORIZED=YES"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("this evaluation requires CUDA")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output path must be new: {output}")

    manifest_path = args.dose_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = list(manifest["tables"])
    if args.tables:
        wanted = set(args.tables)
        selected = [entry for entry in selected if entry["name"] in wanted]
        missing = wanted - {entry["name"] for entry in selected}
        if missing:
            raise ValueError(f"unknown table names: {sorted(missing)}")
    if not selected:
        raise ValueError("no tables selected")

    rows_path = args.long_rows.resolve()
    grouped = load_rows(rows_path, args.multipliers, int(args.limit_rows))
    native_length = int(args.native_length)
    max_positions = native_length * max(int(m) for m in args.multipliers)

    environment = configure_flash_only()
    model = load_model(args.checkpoint.resolve(), max_positions)
    rotary_modules = find_rotary_modules_with_inv_freq(model)
    if not rotary_modules:
        raise RuntimeError("checkpoint exposes no shared rotary inverse-frequency buffer")
    baseline_inv_freq = (
        rotary_modules[0][1].inv_freq.detach().clone().cpu().to(torch.float64).numpy()
    )

    output.mkdir(parents=True, exist_ok=True)
    per_row_path = output / "per_row.jsonl"
    results: dict[str, Any] = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    with per_row_path.open("w", encoding="utf-8") as sink:
        for entry in selected:
            table = np.load(Path(entry["path"]).resolve(), allow_pickle=False)
            if (
                table.dtype != np.dtype("float32")
                or table.shape != baseline_inv_freq.shape
                or not np.all(table[:-1] > table[1:])
                or float32_sha256(table) != entry["table_sha256_float32"]
            ):
                raise RuntimeError(f"{entry['name']}: table identity drift")
            report = apply_inv_freq_inplace(model, torch.as_tensor(table, dtype=torch.float64))
            for name, module in rotary_modules:
                installed = module.inv_freq.detach().cpu().numpy().astype("<f4")
                if float32_sha256(installed) != entry["table_sha256_float32"]:
                    raise RuntimeError(f"{entry['name']}: installed table hash drift at {name}")

            cell: dict[str, Any] = {
                "table_sha256_float32": entry["table_sha256_float32"],
                "patched_rotary_modules": int(report["patched_count"]),
                "lengths": {},
            }
            for multiplier in sorted(grouped):
                length = native_length * multiplier
                summaries = []
                for row in grouped[multiplier]:
                    ids = torch.as_tensor(
                        np.asarray(row["input_ids"], dtype=np.int64), device="cuda"
                    ).unsqueeze(0)
                    if int(ids.shape[1]) != length:
                        raise RuntimeError(
                            f"row {row['source_row']} length {int(ids.shape[1])} != {length}"
                        )
                    losses = score_row(model=model, input_ids=ids)
                    summary = summarise_row(
                        losses, int(row["nll_target_start"]), int(row["nll_target_tokens"])
                    )
                    summary.update(
                        {
                            "table": entry["name"],
                            "length": length,
                            "source_row": int(row["source_row"]),
                            "row_sha256": row.get("row_sha256"),
                        }
                    )
                    sink.write(json.dumps(summary, sort_keys=True) + "\n")
                    summaries.append(summary)
                    del ids
                bin_keys = sorted({k for s in summaries for k in s["position_bin_nll"]}, key=int)
                cell["lengths"][str(length)] = {
                    "rows": len(summaries),
                    "mean_tail_nll": float(np.mean([s["tail_nll"] for s in summaries])),
                    "mean_dense_nll": float(np.mean([s["dense_nll"] for s in summaries])),
                    "per_row_tail_nll": [s["tail_nll"] for s in summaries],
                    "per_row_dense_nll": [s["dense_nll"] for s in summaries],
                    "mean_position_bin_nll": {
                        key: float(
                            np.mean(
                                [s["position_bin_nll"][key] for s in summaries if key in s["position_bin_nll"]]
                            )
                        )
                        for key in bin_keys
                    },
                }
                sink.flush()
                print(
                    f"{entry['name']:28s} L={length:6d} "
                    f"tail={cell['lengths'][str(length)]['mean_tail_nll']:.5f} "
                    f"dense={cell['lengths'][str(length)]['mean_dense_nll']:.5f} "
                    f"({time.perf_counter() - started:.0f}s)",
                    flush=True,
                )
            results[entry["name"]] = cell

    control_name = next(
        (entry["name"] for entry in selected if entry.get("is_bitwise_native")), None
    )
    contrasts: dict[str, Any] = {}
    if control_name is not None:
        for name, cell in results.items():
            if name == control_name:
                continue
            contrasts[name] = {
                length: {
                    metric: paired_bootstrap(
                        np.asarray(cell["lengths"][length][f"per_row_{metric}"]),
                        np.asarray(results[control_name]["lengths"][length][f"per_row_{metric}"]),
                        resamples=int(args.bootstrap_resamples),
                        seed=int(args.bootstrap_seed),
                    )
                    for metric in ("tail_nll", "dense_nll")
                }
                for length in cell["lengths"]
            }

    receipt = {
        "status": STATUS,
        "checkpoint": str(args.checkpoint.resolve()),
        "dose_manifest_sha256": sha256_file(manifest_path),
        "long_rows_sha256": sha256_file(rows_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "environment": environment,
        "native_length": native_length,
        "multipliers": sorted(int(m) for m in args.multipliers),
        "documents_per_length": len(next(iter(grouped.values()))),
        "position_bin_width": BIN_WIDTH,
        "frozen_weights": True,
        "attention_scaling": 1.0,
        "learned_parameters": 0,
        "target_length_used": False,
        "control_table": control_name,
        "results": results,
        "paired_bootstrap_vs_control": contrasts,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (output / "dose_eval.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwrote {output / 'dose_eval.json'} and {per_row_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
