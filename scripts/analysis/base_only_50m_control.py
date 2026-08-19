#!/usr/bin/env python3
"""CPU-only base-adjustment controls for the frozen L=512 50M checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.attention_fisher_50m_probe import (
    ARMS,
    RUN_ROOT,
    _load_arm,
    _sha256,
    _static_geometry,
)


def _tables() -> dict[str, dict[str, object]]:
    geo = np.load(ARMS["geometric_tau0_seed42"] / "inv_freq.npy").astype(np.float64)
    evq = np.load(ARMS["evq_cosh_tau2.83_seed42"] / "inv_freq.npy").astype(np.float64)
    pairs = len(geo)
    quantile = (np.arange(pairs, dtype=np.float64) + 0.5) / pairs

    log_base_ls = -float(quantile @ np.log(evq)) / float(quantile @ quantile)
    bases = {
        "geometric_base500k": 500_000.0,
        "geometric_base_ls_to_evq": math.exp(log_base_ls),
        "geometric_base_low_endpoint_match": math.exp(-math.log(evq[-1]) / quantile[-1]),
        "geometric_base_span_match": math.exp(
            math.log(evq[0] / evq[-1]) / (quantile[-1] - quantile[0])
        ),
    }
    result = {
        name: {
            "kind": "geometric_base_only",
            "base": base,
            "inv_freq": np.power(base, -quantile),
        }
        for name, base in bases.items()
    }
    result["evq_cosh_tau2.83"] = {
        "kind": "historical_evq_table",
        "base": 500_000.0,
        "inv_freq": evq,
    }
    for value in result.values():
        inv = np.asarray(value["inv_freq"], dtype=np.float64)
        value.update({
            "first_frequency": float(inv[0]),
            "last_frequency": float(inv[-1]),
            "log_rms_to_evq": float(np.sqrt(np.mean((np.log(inv) - np.log(evq)) ** 2))),
            "log_rms_to_geo": float(np.sqrt(np.mean((np.log(inv) - np.log(geo)) ** 2))),
        })
    return result


@torch.inference_mode()
def _loss(model: torch.nn.Module, validation: torch.Tensor, starts: np.ndarray, length: int) -> dict[str, object]:
    window_losses = []
    token_losses = []
    for start in starts:
        tokens = validation[int(start) : int(start) + length].view(1, length)
        targets = validation[int(start) + 1 : int(start) + length + 1]
        hidden = model.emb(tokens)
        for block in model.blocks:
            hidden = block(hidden)
        hidden = model.ln(hidden)[0]
        losses = []
        for offset in range(0, length, 64):
            losses.append(F.cross_entropy(
                model.head(hidden[offset : offset + 64]),
                targets[offset : offset + 64],
                reduction="none",
            ))
        losses = torch.cat(losses).double()
        token_losses.extend(losses.tolist())
        window_losses.append(float(losses.mean()))
    mean = float(np.mean(token_losses))
    return {
        "loss": mean,
        "ppl": math.exp(mean),
        "tokens": len(token_losses),
        "window_losses": window_losses,
        "window_loss_median": float(np.median(window_losses)),
        "window_loss_iqr": float(np.quantile(window_losses, 0.75) - np.quantile(window_losses, 0.25)),
    }


def run(length: int = 512, windows: int = 8) -> dict[str, object]:
    if length != 512 or windows < 2:
        raise ValueError("this frozen control requires length=512 and at least two windows")
    validation_path = RUN_ROOT / "val_tinystories_5000000.pt"
    validation = torch.load(validation_path, map_location="cpu", weights_only=True)
    starts = np.linspace(0, len(validation) - length - 1, windows, dtype=np.int64)
    tables = _tables()
    arms = {}
    for weights, path in ARMS.items():
        training_table = torch.from_numpy(
            np.load(path / "inv_freq.npy")
        ).float()
        model, _, identity = _load_arm(
            path, length, training_table, "training_table"
        )
        table_results = {}
        for table_name, table in tables.items():
            inv_freq = torch.from_numpy(
                np.asarray(table["inv_freq"], dtype=np.float32)
            )
            ropes = {id(block.attn.rope): block.attn.rope for block in model.blocks}
            for rope in ropes.values():
                rope.inv_freq.copy_(inv_freq)
                rope._build(length)
            table_results[table_name] = {
                "lm": _loss(model, validation, starts, length),
                "static_geometry": _static_geometry(inv_freq, length),
            }
        arms[weights] = {"identity": identity, "tables": table_results}
    public_tables = {
        name: {key: value for key, value in table.items() if key != "inv_freq"}
        for name, table in tables.items()
    }
    return {
        "status": "CPU_ONLY_COMPLETE",
        "protocol": {
            "length": length,
            "windows": windows,
            "window_starts": starts.tolist(),
            "validation_sha256": _sha256(validation_path),
            "training_or_parameter_updates": False,
            "loss_based_base_search": False,
        },
        "tables": public_tables,
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/base_only_50m_control_20260819.json"),
    )
    args = parser.parse_args()
    result = run(windows=args.windows)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
