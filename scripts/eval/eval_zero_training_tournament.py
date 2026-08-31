#!/usr/bin/env python3
"""Single-4x-forward evaluation of success-first zero-training candidates.

Frozen-checkpoint, zero-weight-update evaluation for the success-first
tournament (``ZERO_TRAINING_FOLLOWUP_SPRINT_PREFLIGHT_20260830`` §5–§6).  With a
static table and gain ``1``, one physical ``4x`` causal forward per document
yields every endpoint the protocol needs, so development, selection and
confirmation all condition on the same rows:

*   ``native_prefix_nll`` — target positions ``1..L_native-1``;
*   fixed 1024-token position-bin NLL across the whole ``4x`` sequence;
*   ``long_dense_nll`` — every valid next-token target in the ``4x`` sequence;
*   ``far_tail_nll`` — the final 1024 target tokens.

Three modes share one contract:

*   ``--contract`` — CPU-only identity check of the candidate manifest, tables
    and rows schema.  No model, no CUDA, no authorisation.
*   ``--parity-smoke`` — GPU.  Verifies the ``4x``-prefix reuse against a
    standalone ``1x`` forward on the Native table within predeclared tolerance.
*   ``--evaluate`` — GPU.  Scores every candidate on the split's rows, writes
    per-row numerators/denominators and paired bootstrap versus the Native control.

The ``1x``-versus-``4x``-prefix parity smoke must pass its predeclared numerical
tolerance before this reuse is accepted; otherwise fall back to separate forwards.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STATUS = "ZERO_TRAINING_TOURNAMENT_EVAL_COMPLETE_V1"
BIN_WIDTH = 1024
LOGIT_CHUNK = 512
PARITY_PER_TOKEN_TOL = 1e-3
PARITY_PREFIX_MEAN_TOL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--native-length", type=int, default=4096)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--tables", nargs="+", default=None)
    parser.add_argument("--parity-docs", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_831)
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--parity-smoke", action="store_true")
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


def load_candidates(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload["candidates"] if isinstance(payload, dict) else payload
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate manifest must list at least one candidate")
    return candidates, payload if isinstance(payload, dict) else {}


def validate_table_identity(entry: dict[str, Any], pair_count: int) -> np.ndarray:
    table = np.load(Path(entry["path"]).resolve(), allow_pickle=False)
    if table.dtype != np.dtype("float32"):
        raise RuntimeError(f"{entry['name']}: table is not float32")
    if table.shape != (pair_count,):
        raise RuntimeError(f"{entry['name']}: table shape {table.shape} != ({pair_count},)")
    if not np.isfinite(table).all() or not np.all(table > 0.0):
        raise RuntimeError(f"{entry['name']}: table has non-finite or non-positive values")
    if not np.all(table[:-1] > table[1:]):
        raise RuntimeError(f"{entry['name']}: table is not strictly decreasing")
    if float32_sha256(table) != entry["table_sha256_float32"]:
        raise RuntimeError(f"{entry['name']}: table hash drift")
    support_factor = float(entry.get("support_factor", 1.0))
    if support_factor < 1.0 or not math.isfinite(support_factor):
        raise RuntimeError(f"{entry['name']}: invalid support_factor")
    return table


def is_config_arm(entry: dict[str, Any]) -> bool:
    """True when the arm is realised by model configuration, not a table swap."""

    return bool(entry.get("hf_rope_scaling"))


def validate_arm_contract(entry: dict[str, Any], pair_count: int) -> None:
    """Enforce the serving invariant: only reference arms may leave gain 1.

    A deployable candidate is always one static table installed at attention
    scaling 1.  Reference arms exist to anchor a comparison, so they may carry a
    published configuration and attention scaling instead.
    """

    reference = bool(entry.get("is_reference"))
    scaling = float(entry.get("attention_scaling", 1.0))
    if not reference:
        if is_config_arm(entry):
            raise RuntimeError(
                f"{entry['name']}: hf_rope_scaling is only allowed on a reference arm"
            )
        if scaling != 1.0:
            raise RuntimeError(
                f"{entry['name']}: candidate arms must keep attention_scaling 1, got {scaling}"
            )
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise RuntimeError(f"{entry['name']}: invalid attention_scaling {scaling}")
    if is_config_arm(entry):
        if entry.get("path"):
            raise RuntimeError(
                f"{entry['name']}: a configuration arm must not also declare a table path"
            )
        return
    validate_table_identity(entry, pair_count)


def validate_rows_schema(path: Path, native_length: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            multiplier = int(row["multiplier"])
            expected = native_length * multiplier
            if len(row["input_ids"]) != expected:
                raise RuntimeError(
                    f"row {row.get('source_row')} multiplier {multiplier}: "
                    f"length {len(row['input_ids'])} != {expected}"
                )
            counts[multiplier] = counts.get(multiplier, 0) + 1
    return counts


def contract_check(args: argparse.Namespace) -> int:
    candidates, _ = load_candidates(args.candidates.resolve())
    pair_counts = {int(entry.get("pair_count", 0)) for entry in candidates}
    if len(pair_counts) != 1 or 0 in pair_counts:
        raise RuntimeError("all candidates must declare one common pair_count")
    pair_count = pair_counts.pop()
    names = [entry["name"] for entry in candidates]
    if len(set(names)) != len(names):
        raise RuntimeError("candidate names are not unique")
    native_controls = [entry["name"] for entry in candidates if entry.get("is_bitwise_native")]
    if len(native_controls) != 1:
        raise RuntimeError("exactly one candidate must be the bitwise Native control")
    for entry in candidates:
        validate_arm_contract(entry, pair_count)
    counts = validate_rows_schema(args.rows.resolve(), int(args.native_length))
    if 4 not in counts:
        raise RuntimeError("rows must contain multiplier-4 documents for the 4x forward")
    references = [entry["name"] for entry in candidates if entry.get("is_reference")]
    anchors = [entry["name"] for entry in candidates if entry.get("is_anchor")]
    if len(anchors) > 1:
        raise RuntimeError(f"at most one arm may be the anchor, got {anchors}")
    if anchors and anchors[0] not in references:
        raise RuntimeError("the anchor arm must also be marked as a reference arm")
    receipt = {
        "status": "ZERO_TRAINING_TOURNAMENT_CONTRACT_OK",
        "candidates": len(candidates),
        "pair_count": pair_count,
        "native_control": native_controls[0],
        "reference_arms": references,
        "anchor_arm": anchors[0] if anchors else None,
        "candidate_sha256": sha256_file(args.candidates.resolve()),
        "rows_sha256": sha256_file(args.rows.resolve()),
        "rows_by_multiplier": counts,
        "endpoint_contract": {
            "native_prefix": "positions 1..L_native-1",
            "position_bin_width": BIN_WIDTH,
            "long_dense": "every target in the 4x sequence",
            "far_tail": "final 1024 targets",
        },
        "parity_tolerance": {
            "per_token_abs": PARITY_PER_TOKEN_TOL,
            "prefix_mean_abs": PARITY_PREFIX_MEAN_TOL,
        },
        "model_loaded": False,
        "cuda_initialised": False,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    (args.output.resolve() / "contract_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "candidates": len(candidates)}, sort_keys=True))
    return 0


# --- GPU machinery -------------------------------------------------------


def configure_flash_only() -> dict[str, Any]:
    import torch

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return {
        "name": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }


def load_model(checkpoint: Path, max_positions: int, rope_scaling: dict[str, Any] | None = None):
    import torch
    from transformers import AutoModelForCausalLM

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    extra: dict[str, Any] = {}
    if rope_scaling:
        extra["rope_scaling"] = dict(rope_scaling)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", low_cpu_mem_usage=True, **extra,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", low_cpu_mem_usage=True, **extra,
        )
    if rope_scaling:
        installed = getattr(model.config, "rope_scaling", None)
        if not installed:
            raise RuntimeError("requested rope_scaling was not installed on the model config")
        for key, value in rope_scaling.items():
            if str(installed.get(key)) != str(value):
                raise RuntimeError(
                    f"rope_scaling drift: {key}={installed.get(key)!r} != requested {value!r}"
                )
    model.config.max_position_embeddings = int(max_positions)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to("cuda")


def score_row(model: Any, input_ids: Any) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    length = int(input_ids.shape[1])
    position_ids = torch.arange(length, device="cuda", dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        hidden = model.model(input_ids=input_ids, position_ids=position_ids).last_hidden_state[0]
        targets = input_ids[0, 1:]
        losses = torch.empty(length - 1, dtype=torch.float32, device="cuda")
        for start in range(0, length - 1, LOGIT_CHUNK):
            stop = min(start + LOGIT_CHUNK, length - 1)
            logits = model.lm_head(hidden[start:stop]).float()
            losses[start:stop] = F.cross_entropy(logits, targets[start:stop], reduction="none")
            del logits
    del hidden
    return losses.detach().cpu().numpy().astype(np.float64)


def endpoint_summary(losses: np.ndarray, native_length: int) -> dict[str, Any]:
    """Numerator/denominator for each endpoint from one ``4x`` loss vector.

    ``losses[i]`` is the NLL of the token at absolute position ``i + 1``.
    """

    positions = np.arange(1, losses.size + 1)
    total = int(losses.size)

    prefix_mask = positions <= native_length - 1
    far_start = total - BIN_WIDTH + 1
    far_mask = positions >= far_start
    if not prefix_mask.any() or not far_mask.any():
        raise RuntimeError("endpoint masks do not intersect the scored positions")

    bins: dict[str, dict[str, float]] = {}
    for start in range(0, total + 1, BIN_WIDTH):
        mask = (positions > start) & (positions <= start + BIN_WIDTH)
        if mask.any():
            bins[str(start)] = {
                "numerator": float(losses[mask].sum()),
                "denominator": int(mask.sum()),
            }

    def endpoint(mask: np.ndarray) -> dict[str, float]:
        return {
            "numerator": float(losses[mask].sum()),
            "denominator": int(mask.sum()),
            "nll": float(losses[mask].mean()),
        }

    return {
        "native_prefix": endpoint(prefix_mask),
        "long_dense": endpoint(np.ones(total, dtype=bool)),
        "far_tail": endpoint(far_mask),
        "position_bins": bins,
    }


def paired_bootstrap(treatment: np.ndarray, control: np.ndarray, *, resamples: int, seed: int) -> dict[str, float]:
    difference = treatment - control
    generator = np.random.default_rng(seed)
    index = generator.integers(0, difference.size, size=(resamples, difference.size))
    draws = difference[index].mean(axis=1)
    return {
        "mean_delta": float(difference.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "rows": int(difference.size),
    }


def _install(model: Any, table: np.ndarray) -> None:
    import torch

    from scripts.lib.rope.inject import apply_inv_freq_inplace

    apply_inv_freq_inplace(model, torch.as_tensor(table, dtype=torch.float64))


def _row_tensors(rows: list[dict[str, Any]]):
    import torch

    return [
        torch.as_tensor(np.asarray(row["input_ids"], dtype=np.int64), device="cuda").unsqueeze(0)
        for row in rows
    ]


def _load_rows(path: Path, multiplier: int, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["multiplier"]) == multiplier:
                rows.append(row)
    rows.sort(key=lambda r: int(r["source_row"]))
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"no rows for multiplier {multiplier}")
    return rows


def parity_smoke(args: argparse.Namespace) -> int:
    import torch

    _require_gpu_authorization(args)
    candidates, _ = load_candidates(args.candidates.resolve())
    native = next(e for e in candidates if e.get("is_bitwise_native"))
    pair_count = int(native["pair_count"])
    validate_table_identity(native, pair_count)
    rows4 = _load_rows(args.rows.resolve(), 4, int(args.parity_docs))
    rows1 = _load_rows(args.rows.resolve(), 1, int(args.parity_docs))
    by_source = {int(r["source_row"]): r for r in rows1}

    environment = configure_flash_only()
    native_length = int(args.native_length)
    model = load_model(args.checkpoint.resolve(), native_length * 4)
    _install(model, np.load(Path(native["path"]).resolve(), allow_pickle=False))

    records = []
    for row in rows4:
        source = int(row["source_row"])
        if source not in by_source:
            raise RuntimeError(f"parity needs a matched 1x row for source {source}")
        ids4 = torch.as_tensor(np.asarray(row["input_ids"], dtype=np.int64), device="cuda").unsqueeze(0)
        ids1 = torch.as_tensor(np.asarray(by_source[source]["input_ids"], dtype=np.int64), device="cuda").unsqueeze(0)
        losses4 = score_row(model, ids4)
        losses1 = score_row(model, ids1)
        prefix = losses4[: native_length - 1]
        if prefix.shape != losses1.shape:
            raise RuntimeError("parity prefix length mismatch")
        diff = np.abs(prefix - losses1)
        records.append(
            {
                "source_row": source,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
            }
        )
        del ids4, ids1

    worst_max = max(r["max_abs_diff"] for r in records)
    worst_mean = max(r["mean_abs_diff"] for r in records)
    passed = worst_max <= PARITY_PER_TOKEN_TOL and worst_mean <= PARITY_PREFIX_MEAN_TOL
    receipt = {
        "status": "ZERO_TRAINING_TOURNAMENT_PARITY_SMOKE",
        "passed": passed,
        "per_token_abs_tol": PARITY_PER_TOKEN_TOL,
        "prefix_mean_abs_tol": PARITY_PREFIX_MEAN_TOL,
        "worst_max_abs_diff": worst_max,
        "worst_mean_abs_diff": worst_mean,
        "documents": len(records),
        "records": records,
        "environment": environment,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    (args.output.resolve() / "parity_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "passed": passed}, sort_keys=True))
    return 0 if passed else 2


def evaluate(args: argparse.Namespace) -> int:
    import torch

    _require_gpu_authorization(args)
    candidates, _ = load_candidates(args.candidates.resolve())
    pair_count = int(candidates[0]["pair_count"])
    selected = candidates
    if args.tables:
        wanted = set(args.tables)
        selected = [e for e in candidates if e["name"] in wanted]
        missing = wanted - {e["name"] for e in selected}
        if missing:
            raise ValueError(f"unknown table names: {sorted(missing)}")
    for entry in selected:
        validate_arm_contract(entry, pair_count)

    rows4 = _load_rows(args.rows.resolve(), 4, int(args.limit_rows))
    native_length = int(args.native_length)
    environment = configure_flash_only()
    model = load_model(args.checkpoint.resolve(), native_length * 4)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_row_path = output / "per_row.jsonl"
    results: dict[str, Any] = {}
    control_name = next(e["name"] for e in selected if e.get("is_bitwise_native"))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    with per_row_path.open("w", encoding="utf-8") as sink:
        for entry in selected:
            config_arm = is_config_arm(entry)
            if config_arm:
                arm_model = load_model(
                    args.checkpoint.resolve(), native_length * 4, entry["hf_rope_scaling"]
                )
            else:
                arm_model = model
                table = np.load(Path(entry["path"]).resolve(), allow_pickle=False)
                _install(arm_model, table)
            summaries = []
            for row in rows4:
                ids = torch.as_tensor(
                    np.asarray(row["input_ids"], dtype=np.int64), device="cuda"
                ).unsqueeze(0)
                losses = score_row(arm_model, ids)
                summary = endpoint_summary(losses, native_length)
                summary.update(
                    {
                        "candidate": entry["name"],
                        "source_row": int(row["source_row"]),
                        "row_sha256": row.get("row_sha256"),
                    }
                )
                sink.write(json.dumps(summary, sort_keys=True) + "\n")
                summaries.append(summary)
                del ids
            if config_arm:
                del arm_model
                torch.cuda.empty_cache()
            results[entry["name"]] = {
                "candidate": entry["name"],
                "family": entry.get("family"),
                "is_reference": bool(entry.get("is_reference")),
                "attention_scaling": float(entry.get("attention_scaling", 1.0)),
                "realised_by": "hf_rope_scaling" if config_arm else "table_swap",
                "rows": len(summaries),
                "mean_native_prefix_nll": float(np.mean([s["native_prefix"]["nll"] for s in summaries])),
                "mean_long_dense_nll": float(np.mean([s["long_dense"]["nll"] for s in summaries])),
                "mean_far_tail_nll": float(np.mean([s["far_tail"]["nll"] for s in summaries])),
                "per_row_native_prefix_nll": [s["native_prefix"]["nll"] for s in summaries],
                "per_row_long_dense_nll": [s["long_dense"]["nll"] for s in summaries],
                "per_row_far_tail_nll": [s["far_tail"]["nll"] for s in summaries],
            }
            sink.flush()
            print(
                f"{entry['name']:32s} prefix={results[entry['name']]['mean_native_prefix_nll']:.5f} "
                f"dense={results[entry['name']]['mean_long_dense_nll']:.5f} "
                f"tail={results[entry['name']]['mean_far_tail_nll']:.5f} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

    control = results[control_name]
    anchor_entry = next((e for e in selected if e.get("is_anchor")), None)
    anchor_name = anchor_entry["name"] if anchor_entry else None
    anchor_cell = results.get(anchor_name) if anchor_name else None
    anchor_prefix_delta = (
        anchor_cell["mean_native_prefix_nll"] - control["mean_native_prefix_nll"]
        if anchor_cell else None
    )
    anchor_long_dense_delta = (
        anchor_cell["mean_long_dense_nll"] - control["mean_long_dense_nll"]
        if anchor_cell else None
    )
    selection_rows = []
    bootstrap = {}
    bootstrap_vs_anchor = {}
    for name, cell in results.items():
        if name == control_name:
            native_prefix_delta = 0.0
            long_dense_delta = 0.0
        else:
            native_prefix_delta = cell["mean_native_prefix_nll"] - control["mean_native_prefix_nll"]
            long_dense_delta = cell["mean_long_dense_nll"] - control["mean_long_dense_nll"]
        entry = next(e for e in selected if e["name"] == name)
        row = {
            "name": name,
            "family": entry.get("family"),
            "is_reference": bool(entry.get("is_reference")),
            "is_anchor": bool(entry.get("is_anchor")),
            "native_prefix_delta": native_prefix_delta,
            "long_dense_delta": long_dense_delta,
            "far_tail_nll": cell["mean_far_tail_nll"],
            "calibrated_dof": int(entry.get("calibrated_dof", 0)),
            "used_long_range_in_construction": bool(entry.get("used_long_range_in_construction", False)),
            "chord_displacement_rms": float(entry.get("chord_displacement_rms", 0.0)),
            "support_movement": float(math.log(float(entry.get("support_factor", 1.0)))),
        }
        if anchor_prefix_delta is not None:
            row["yarn_native_prefix_delta"] = float(anchor_prefix_delta)
            row["yarn_long_dense_delta"] = float(anchor_long_dense_delta)
        selection_rows.append(row)
        if name != control_name:
            bootstrap[name] = {
                endpoint: paired_bootstrap(
                    np.asarray(cell[f"per_row_{endpoint}_nll"]),
                    np.asarray(control[f"per_row_{endpoint}_nll"]),
                    resamples=int(args.bootstrap_resamples),
                    seed=int(args.bootstrap_seed),
                )
                for endpoint in ("native_prefix", "long_dense", "far_tail")
            }
        if anchor_cell is not None and name != anchor_name:
            bootstrap_vs_anchor[name] = {
                endpoint: paired_bootstrap(
                    np.asarray(cell[f"per_row_{endpoint}_nll"]),
                    np.asarray(anchor_cell[f"per_row_{endpoint}_nll"]),
                    resamples=int(args.bootstrap_resamples),
                    seed=int(args.bootstrap_seed),
                )
                for endpoint in ("native_prefix", "long_dense", "far_tail")
            }

    receipt = {
        "status": STATUS,
        "candidates": len(selected),
        "rows": len(rows4),
        "native_length": native_length,
        "control": control_name,
        "anchor": anchor_name,
        "environment": environment,
        "selection_rows": selection_rows,
        "paired_bootstrap_vs_control": bootstrap,
        "paired_bootstrap_vs_anchor": bootstrap_vs_anchor,
        "frozen_weights": True,
        "attention_scaling_by_arm": {
            name: cell["attention_scaling"] for name, cell in results.items()
        },
        "realised_by_arm": {name: cell["realised_by"] for name, cell in results.items()},
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    (output / "eval_summary.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {output / 'eval_summary.json'} and {per_row_path}")
    return 0


def _require_gpu_authorization(args: argparse.Namespace) -> None:
    import torch

    if not args.authorize or os.environ.get("ZERO_TRAINING_TOURNAMENT_GPU_AUTHORIZED") != "YES":
        raise PermissionError(
            "GPU evaluation requires --authorize and "
            "ZERO_TRAINING_TOURNAMENT_GPU_AUTHORIZED=YES"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("this evaluation requires CUDA")


def main() -> int:
    args = parse_args()
    if args.contract:
        return contract_check(args)
    if args.parity_smoke:
        return parity_smoke(args)
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --evaluate")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
