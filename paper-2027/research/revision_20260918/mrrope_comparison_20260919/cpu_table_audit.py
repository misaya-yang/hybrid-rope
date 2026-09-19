#!/usr/bin/env python3
"""Compare the reviewed author constructors with the live project CPU table."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[3]
sys.path.insert(0, str(ROOT))

from experiments.fixed_rope_three_interfaces_20260913.tables import build_analytic
from scripts.experiments.cross_audit.tables import native_table


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(left, right):
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    ulps = np.abs(left.view(np.uint32).astype(np.int64) - right.view(np.uint32).astype(np.int64))
    return {
        "bitwise_equal": bool(np.array_equal(left.view(np.uint32), right.view(np.uint32))),
        "equal_entries": int(np.count_nonzero(left == right)),
        "total_entries": len(left),
        "differing_indices": np.flatnonzero(left != right).tolist(),
        "max_absolute_delta": float(delta.max()),
        "max_relative_delta": float((delta / right).max()),
        "max_ulp": int(ulps.max()),
        "max_unwrapped_phase_delta_at_distance_131071": float(delta.max() * 131071),
    }


def main():
    torch.set_num_threads(2)
    config = {"hidden_size": 4096, "num_attention_heads": 32, "rope_theta": 500000,
              "max_position_embeddings": 8192, "model_type": "llama"}
    ours, gain, meta = build_analytic(config, method="mrpro", scale=16, low=None,
                                     high=None, depth=1, gain=None)
    native = native_table(128, 500000)
    exponent = np.asarray(meta["cumulative_exponents"], dtype=np.float64)
    # The author divides an FP32 tensor by a Python scalar, which is converted
    # to FP32 by Torch. The project divides in NumPy FP64 and casts once.
    fp32_denominator = torch.from_numpy(np.power(16.0, exponent).astype(np.float32))
    fp32_replay = (torch.from_numpy(native) / fp32_denominator).numpy()
    out = OWNER / "cpu_table_audit"
    out.mkdir(exist_ok=True)
    sources = OWNER / "ruler_protocol_sources"
    arrays = {}
    reports = {}
    for label, filename, classname in (
        ("github", "github_LlamaYaRNRadix.py", "LlamaYaRNRadix"),
        ("conference_supplement", "supplement_LlamaMrRoPE.py", "LlamaMrRoPE"),
    ):
        path = sources / filename
        spec = importlib.util.spec_from_file_location("audit_" + label, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            obj = getattr(module, classname)(
                dim=128, max_position_embeddings=131072, base=500000, scale=16,
                original_max_position_embeddings=8192, device="cpu")
        values = obj.inv_freq.detach().cpu().numpy().copy()
        arrays[label] = values
        reports[label] = {
            "source": path.relative_to(ROOT).as_posix(), "source_sha256": digest(path),
            "band": list(module.find_correction_range(32, 1, 128, 500000, 8192)),
            "gain": obj.mscale, "gain_exactly_equal": obj.mscale == gain,
            "versus_project": compare(values, ours),
            "versus_fp32_division_replay": compare(values, fp32_replay),
            "constructor_stdout": log.getvalue(),
        }
        del obj
    report = {
        "status": "CPU_CONSTRUCTOR_AUDIT_COMPLETE", "model_execution": False,
        "config": config, "scale": 16, "torch_version": torch.__version__,
        "numpy_version": np.__version__, "python_version": sys.version,
        "project_band": [meta["low"], meta["high"]], "project_gain": gain,
        "project_constructor": "experiments/fixed_rope_three_interfaces_20260913/tables.py",
        "project_transform_sha256": digest(ROOT / "scripts/experiments/cross_audit/tables.py"),
        "github_revision": "d1a6f31eba7fbf5f6ad5e31c948a6f8f9f148c73",
        "authors": reports,
        "author_implementations_equal": compare(arrays["github"], arrays["conference_supplement"]),
        "values_float32": {"project": ours.tolist(), **{k: v.tolist() for k, v in arrays.items()}},
        "scope": "CPU constructor values and gain only. Does not certify GPU runtime installation, generation parity, actual historic checkpoint identity, or historical saved table bytes.",
    }
    for label, values in {"project": ours, **arrays}.items():
        receipt = {"values_float32": values.tolist(), "gain": gain,
                   "construction": {"method": "mrpro", "scale": 16, "base": 500000,
                                    "reference_length": 8192, "band": [18, 35], "source": label}}
        (out / f"{label}_table.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "authors": reports,
                      "author_implementations_equal": report["author_implementations_equal"]}, indent=2))


if __name__ == "__main__":
    main()
