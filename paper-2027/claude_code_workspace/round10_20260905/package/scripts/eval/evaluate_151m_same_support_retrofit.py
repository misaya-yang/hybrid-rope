#!/usr/bin/env python3
"""Evaluate fixed long-profile controls on paired 151.9M checkpoints.

This is inference only.  It reuses the frozen exact-range validation anchors
and evaluates each checkpoint's own Native table plus four deterministic
factor-four controls derived without task labels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.protocol import (
    ExperimentSpec,
    training_inv_freq,
)
from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment import (
    _load_checkpoint,
    _model_inv_freq,
    build_eval_windows,
    set_runtime_rope,
    sha256_file,
    tensor_sha256,
)
from scripts.analysis.rope_transport.same_support_controls import (
    build_same_support_controls,
)
from scripts.lib.rope.length_conditioned_budgeted import matched_attention_scaling


STATUS = "SAME_SUPPORT_151M_RETROFIT_COMPLETE"
ARMS = ("fmrope_base256", "anchored_cosh_tau4_fmrope_range")
METHODS = (
    "fmrope_native_raw",
    "anchored_cosh_native_raw",
    "fmrope_native_plus_s4_amplitude",
    "anchored_cosh_native_plus_s4_amplitude",
    "fmrope_budgeted_s4",
    "anchored_cosh_budgeted_s4",
    "same_support_geometric_s4",
    "fmrope_nearest_yarn_ramp_s4",
    "anchored_cosh_nearest_yarn_ramp_s4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(137, 256))
    parser.add_argument("--lengths", type=int, nargs="+", default=(512, 1024))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".incomplete",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def target_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<i8").tobytes()).hexdigest()


def method_tables(
    *,
    spec: ExperimentSpec,
    native_context_length: int,
) -> dict[str, tuple[torch.Tensor, float, dict[str, Any]]]:
    amplitude = matched_attention_scaling(4.0)
    source_tables = {
        "fmrope": training_inv_freq("fmrope_base256", spec=spec).cpu().float().contiguous(),
        "anchored_cosh": training_inv_freq(
            "anchored_cosh_tau4_fmrope_range", spec=spec
        ).cpu().float().contiguous(),
    }
    controls = {
        source: build_same_support_controls(
            table.numpy().astype(np.float64),
            native_context_length=int(native_context_length),
            factor=4.0,
        )
        for source, table in source_tables.items()
    }
    geometric_fmrope = controls["fmrope"]["same_support_geometric_s4"][0]
    geometric_cosh = controls["anchored_cosh"]["same_support_geometric_s4"][0]
    if not np.array_equal(geometric_fmrope, geometric_cosh):
        raise RuntimeError("same-support geometric table differs across equal-endpoint sources")
    result: dict[str, tuple[torch.Tensor, float, dict[str, Any]]] = {}
    for source, table in source_tables.items():
        result[f"{source}_native_raw"] = (
            table.clone(),
            1.0,
            {
                "construction": "training_table_cross_assignment",
                "source_training_table": source,
                "factor": 1.0,
            },
        )
        result[f"{source}_native_plus_s4_amplitude"] = (
            table.clone(),
            amplitude,
            {
                "construction": "training_table_plus_scaling_only",
                "source_training_table": source,
                "factor": 1.0,
            },
        )
        budgeted, budgeted_receipt = controls[source]["converged_budgeted_s4"]
        result[f"{source}_budgeted_s4"] = (
            torch.from_numpy(budgeted.copy()),
            amplitude,
            {**budgeted_receipt, "source_training_table": source},
        )
        ramp, ramp_receipt = controls[source]["nearest_yarn_ramp_s4"]
        result[f"{source}_nearest_yarn_ramp_s4"] = (
            torch.from_numpy(ramp.copy()),
            amplitude,
            {**ramp_receipt, "source_training_table": source},
        )
    geometric, geometric_receipt = controls["fmrope"]["same_support_geometric_s4"]
    result["same_support_geometric_s4"] = (
        torch.from_numpy(geometric.copy()),
        amplitude,
        {
            **geometric_receipt,
            "source_training_table": "shared_equal_endpoints",
        },
    )
    if set(result) != set(METHODS):
        raise RuntimeError(f"151M method set drift: {sorted(result)}")
    return result


@torch.no_grad()
def evaluate_table(
    model: torch.nn.Module,
    validation: np.ndarray,
    anchors: np.ndarray,
    *,
    table: torch.Tensor,
    amplitude: float,
    length: int,
    batch_size: int,
    tail_tokens: int,
) -> list[dict[str, Any]]:
    set_runtime_rope(model, table, length=int(length), mscale=float(amplitude))
    rows: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(anchors), int(batch_size)):
        selected = anchors[start : start + int(batch_size)]
        inputs_np, targets_np = build_eval_windows(
            validation,
            selected,
            length=int(length),
        )
        inputs = torch.from_numpy(inputs_np).to("cuda", non_blocking=True)
        targets = torch.from_numpy(targets_np).to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs)
        token_nll = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        full = token_nll.mean(dim=1)
        tail = token_nll[:, -int(tail_tokens) :].mean(dim=1)
        if not torch.isfinite(full).all() or not torch.isfinite(tail).all():
            raise RuntimeError("non-finite 151M retrofit NLL")
        for index, anchor in enumerate(selected.tolist()):
            rows.append(
                {
                    "anchor": int(anchor),
                    "full_nll": float(full[index].cpu()),
                    "tail_nll": float(tail[index].cpu()),
                    "tail_target_sha256": target_sha256(
                        targets_np[index, -int(tail_tokens) :]
                    ),
                }
            )
        del inputs, targets, logits, token_nll, full, tail
    return rows


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = args.root.resolve()
    manifest_path = args.data_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != "2c4b1c0ec6993a4065a666dd04c26f1c3439e812de25e49cbf5d21b106ab9433":
        raise RuntimeError("151M data manifest drift")
    validation_path = Path(manifest["validation"]["path"])
    anchors_path = Path(manifest["anchors"]["path"])
    if sha256_file(validation_path) != str(manifest["validation"]["sha256"]):
        raise RuntimeError("151M validation drift")
    if sha256_file(anchors_path) != str(manifest["anchors"]["sha256"]):
        raise RuntimeError("151M anchor drift")
    validation = np.load(validation_path, mmap_mode="r", allow_pickle=False)
    anchors = np.load(anchors_path, allow_pickle=False)
    lengths = tuple(int(value) for value in args.lengths)
    if not lengths or any(value not in (512, 1024) for value in lengths):
        raise ValueError("this audit is frozen to 2x/4x lengths 512 and 1024")
    seeds = tuple(int(value) for value in args.seeds)
    if not seeds or any(value not in (137, 256) for value in seeds):
        raise ValueError("available paired checkpoints are seeds 137 and 256")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    receipt: dict[str, Any] = {
        "status": "SAME_SUPPORT_151M_RETROFIT_RUNNING",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_manifest_sha256": sha256_file(manifest_path),
        "validation_sha256": sha256_file(validation_path),
        "anchors_sha256": sha256_file(anchors_path),
        "seeds": list(seeds),
        "arms": list(ARMS),
        "methods": list(METHODS),
        "lengths": list(lengths),
        "evaluation_anchor_count": int(len(anchors)),
        "tail_tokens": 128,
        "results": {},
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }

    for seed in seeds:
        spec = ExperimentSpec(seed=int(seed), micro_batch_size=128)
        seed_root = root / f"seed_{seed}"
        receipt["results"][str(seed)] = {}
        for arm in ARMS:
            print(f"[load] seed={seed} arm={arm}", flush=True)
            model, checkpoint_meta = _load_checkpoint(seed_root, arm, spec=spec)
            model = model.to("cuda")
            native = _model_inv_freq(model).detach().cpu().float().clone()
            tables = method_tables(
                spec=spec,
                native_context_length=spec.train_length,
            )
            arm_result: dict[str, Any] = {
                "checkpoint_sha256": str(checkpoint_meta["checkpoint_sha256"]),
                "training_inv_freq_sha256": tensor_sha256(native),
                "methods": {},
            }
            for method in METHODS:
                table, amplitude, method_meta = tables[method]
                method_result: dict[str, Any] = {
                    "inv_freq_sha256": tensor_sha256(table),
                    "attention_scaling": float(amplitude),
                    "construction": method_meta,
                    "lengths": {},
                }
                for length in lengths:
                    print(
                        f"[eval] seed={seed} arm={arm} method={method} L={length}",
                        flush=True,
                    )
                    rows = evaluate_table(
                        model,
                        validation,
                        anchors,
                        table=table,
                        amplitude=amplitude,
                        length=length,
                        batch_size=int(args.batch_size),
                        tail_tokens=spec.eval_tail_tokens,
                    )
                    method_result["lengths"][str(length)] = {
                        "rows": rows,
                        "mean_full_nll": float(np.mean([row["full_nll"] for row in rows])),
                        "mean_tail_nll": float(np.mean([row["tail_nll"] for row in rows])),
                        "ppl_from_mean_tail_nll": float(
                            math.exp(np.mean([row["tail_nll"] for row in rows]))
                        ),
                    }
                arm_result["methods"][method] = method_result
            receipt["results"][str(seed)][arm] = arm_result
            del model
            gc.collect()
            torch.cuda.empty_cache()

    receipt["status"] = STATUS
    receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(output, receipt)
    print(json.dumps({"status": STATUS, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
