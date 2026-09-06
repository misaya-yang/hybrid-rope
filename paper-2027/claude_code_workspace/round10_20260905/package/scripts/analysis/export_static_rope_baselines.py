#!/usr/bin/env python3
"""Freeze official-equation YaRN and static NTK tables; no fitting or GPU use.

NumPy-only by default. --verify-transformers additionally checks installed HF
CPU initializers and exports their actual float32 YaRN tensor. Tolerance parity
of equations is not bitwise identity or GPU runtime/attention parity.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.export_frozen_coupling_transport import (  # noqa: E402
    config_identity,
    sha256_bytes,
    sha256_file,
)

PARITY_RTOL = 1e-6
NATIVE_ROTARY_CLASSES = {
    "gemma": "GemmaRotaryEmbedding",
    "qwen2": "Qwen2RotaryEmbedding",
    "olmo2": "Olmo2RotaryEmbedding",
    "llama": "LlamaRotaryEmbedding",
}


def tensor_hash(table: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(table, dtype="<f4").tobytes())


def load_inputs(config_path: Path, native_path: Path) -> tuple[dict, np.ndarray]:
    identity = config_identity(config_path)
    config = json.loads(config_path.read_text())
    parameters = config.get("rope_parameters") or {}
    if set(parameters) - {"rope_theta", "rope_type", "partial_rotary_factor"}:
        raise ValueError("scaled or nested rope_parameters are not admitted")
    if parameters.get("rope_type", "default") != "default":
        raise ValueError("checkpoint already has RoPE scaling in rope_parameters")
    if float(parameters.get("partial_rotary_factor", 1.0)) != 1.0:
        raise ValueError("only full-head RoPE checkpoints are admitted")
    base = identity["rope_theta"]
    if "rope_theta" in parameters and float(parameters["rope_theta"]) != base:
        raise ValueError("conflicting rope_theta values")
    if not math.isfinite(base) or base <= 1 or identity["native_length"] <= 0:
        raise ValueError("invalid RoPE base or native length")
    if identity["head_dim"] <= 2:
        raise ValueError("static NTK requires head_dim > 2")
    native = np.load(native_path, allow_pickle=False)
    if native.dtype != np.dtype("float32"):
        raise ValueError("runtime Native tensor must be float32")
    if (
        native.shape != (identity["pairs"],)
        or not np.isfinite(native).all()
        or not (native > 0).all()
        or not np.all(native[:-1] > native[1:])
    ):
        raise ValueError("invalid runtime Native tensor")
    formula = base ** (-np.arange(identity["pairs"], dtype=np.float64) / identity["pairs"])
    if not np.allclose(native, formula, rtol=PARITY_RTOL, atol=0):
        raise ValueError("runtime Native tensor does not match config geometric grid")
    identity.update({
        "K": identity["pairs"], "b": base, "L": identity["native_length"],
        "native_tensor_source": "supplied runtime rotary initializer float32 tensor",
        "native_tensor_file_sha256": sha256_file(native_path),
        "native_sha256_float32": tensor_hash(native),
        "formula_native_max_abs_difference": float(np.max(np.abs(native - formula))),
        "formula_native_comparison": {"rtol": PARITY_RTOL, "atol": 0.0},
    })
    return identity, native


def build_tables(identity: dict, native: np.ndarray, factor: float) -> tuple[dict, dict]:
    if not math.isfinite(factor) or factor <= 1:
        raise ValueError("factor must be finite and exceed one")
    dim, pairs = identity["head_dim"], identity["pairs"]
    base = identity["rope_theta"]
    length = identity.get("reference_length", identity["native_length"])
    slots = np.arange(pairs, dtype=np.float64)
    low = max(math.floor(dim * math.log(length / (32 * 2 * math.pi)) / (2 * math.log(base))), 0)
    high = min(math.ceil(dim * math.log(length / (2 * math.pi)) / (2 * math.log(base))), dim - 1)
    high_float = high + 0.001 if high == low else float(high)
    ramp = np.clip((slots - low) / (high_float - low), 0, 1)
    omega = native.astype(np.float64)
    yarn = omega / factor * ramp + omega * (1 - ramp)
    ntk = omega * np.power(factor, -slots / (pairs - 1))
    multiplier = factor ** (dim / (dim - 2))
    new_base = base * multiplier
    if not math.isfinite(new_base):
        raise ValueError("static NTK new base is non-finite")
    ntk_new_base = new_base ** (-slots / pairs)
    tables = {
        "official_equation_yarn": np.ascontiguousarray(yarn, dtype="<f4"),
        "static_ntk": np.ascontiguousarray(ntk, dtype="<f4"),
    }
    metadata = {
        "official_equation_yarn": {
            "beta_fast": 32.0, "beta_slow": 1.0,
            "original_max_position_embeddings": length,
            "low": low, "high": high, "ramp": "linear index ramp, floor/ceil bounds",
            "attention_scaling": 1 + 0.1 * math.log(factor),
            "gain_semantics": "cos/sin amplitude; QK logit multiplier is its square",
            "formula": "native_i / factor * ramp_i + native_i * (1-ramp_i)",
            "rounding": "supplied native float32 promoted to float64; blend then cast float32",
            "source": "jquesnelle/yarn@995db5b:scaled_rope/LlamaYaRNScaledRotaryEmbedding.py",
            "table_source": "local NumPy equation implementation; HF parity unverified",
        },
        "static_ntk": {
            "attention_scaling": 1.0,
            "gain_semantics": "unmodified cos/sin amplitude",
            "formula": "native_i * factor ** (-i/(K-1))",
            "base_multiplier": multiplier, "effective_base": new_base,
            "source": "static NTK base multiplier factor ** (d/(d-2)); not dynamic NTK",
            "rounding": "native-relative float64 multiply then float32; algebraically equivalent to new-base formula only before rounding",
            "new_base_formula_max_abs_difference": float(np.max(np.abs(tables["static_ntk"].astype(np.float64) - ntk_new_base))),
            "new_base_formula_float32_tensor_sha256": tensor_hash(ntk_new_base),
            "endpoints": "fast=native[0], slow=native[-1]/factor, before float32 rounding",
        },
    }
    return tables, metadata


def bind_reference(identity: dict, receipt_path: Path | None,
                   target_length: int | None, factor: float | None) -> tuple[float, dict | None]:
    """Bind a confirmed Native-only length without rewriting checkpoint geometry."""
    reference = None
    length = identity["native_length"]
    if receipt_path is not None:
        reference = json.loads(receipt_path.read_text())
        if not isinstance(reference, dict) or reference.get("status") != "NATIVE_REFERENCE_CONFIRMED":
            raise ValueError("reference receipt must be NATIVE_REFERENCE_CONFIRMED")
        length = reference.get("reference_length")
        if type(length) is not int or length <= 0 or length & (length - 1) or length > identity["native_length"]:
            raise ValueError("reference_length must be a positive power of two <= config length")
        for key in ("checkpoint_weight_sha256", "config_sha256", "native_sha256_float32",
                    "confirmation_decision_sha256", "data_manifest_sha256"):
            if not isinstance(reference.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", reference[key]):
                raise ValueError(f"reference receipt requires {key}")
        for key in ("config_sha256", "native_sha256_float32"):
            if reference[key] != identity[key]:
                raise ValueError(f"reference receipt {key} mismatch")
        if type(target_length) is not int or target_length <= length:
            raise ValueError("--target-length > reference_length is required with a reference receipt")
        realized_factor = target_length / length
        if factor is not None and float(factor) != realized_factor:
            raise ValueError("factor must equal target_length / reference_length exactly")
        factor = realized_factor
        reference = {**reference, "receipt_sha256": sha256_file(receipt_path)}
    elif target_length is not None:
        raise ValueError("--target-length requires a confirmed --reference-receipt")
    if factor is None or not math.isfinite(float(factor)) or float(factor) <= 1:
        raise ValueError("factor must be finite and exceed one")
    identity["reference_length"] = length
    return float(factor), reference


def resolve_native_initializer(registry: dict, model_type: str):
    """HF 5.x moved default RoPE from the registry to model-specific static methods."""
    if "default" in registry:
        return registry["default"]
    class_name = NATIVE_ROTARY_CLASSES.get(model_type)
    if class_name is None:
        raise ValueError(f"unverified model-specific Native initializer: {model_type}")
    module = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
    rotary_class = getattr(module, class_name, None)
    initializer = getattr(rotary_class, "compute_default_rope_parameters", None)
    if not callable(initializer):
        raise ValueError(f"{class_name} has no verified default RoPE initializer")
    return initializer


def compare_tables(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.dtype != np.dtype("float32") or actual.shape != expected.shape:
        raise ValueError("HF initializer must return matching float32 tensor")
    if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=PARITY_RTOL, atol=0):
        raise ValueError("HF initializer parity failed")
    return {
        "passed": True, "rtol": PARITY_RTOL, "atol": 0.0,
        "max_abs_difference": float(np.max(np.abs(actual.astype(np.float64) - expected))),
        "exact_tensor_hash_match": tensor_hash(actual) == tensor_hash(expected),
        "hf_tensor_sha256": tensor_hash(actual),
        "equation_tensor_sha256": tensor_hash(expected),
    }


def verify_transformers(config_path: Path, native: np.ndarray, equation: np.ndarray,
                        factor: float, gain: float,
                        reference_length: int | None = None) -> tuple[np.ndarray, dict]:
    """CPU-only, no model weights/network; fail closed on incompatible HF APIs."""
    import torch
    import transformers
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    raw = json.loads(config_path.read_text())
    model_type = raw.pop("model_type")
    native_config = AutoConfig.for_model(model_type, **raw)
    native_initializer = resolve_native_initializer(ROPE_INIT_FUNCTIONS, model_type)
    default_inv, default_gain = native_initializer(native_config, torch.device("cpu"))
    native_parity = compare_tables(default_inv.detach().cpu().numpy(), native)
    if not native_parity["exact_tensor_hash_match"]:
        raise ValueError("HF Native tensor hash drift")
    if float(default_gain) != 1.0:
        raise ValueError("HF Native amplitude must equal one")
    scaling = {"rope_type": "yarn", "factor": factor, "beta_fast": 32.0,
               "beta_slow": 1.0, "attention_factor": gain,
               "original_max_position_embeddings": (
                   raw["max_position_embeddings"] if reference_length is None else reference_length
               )}
    raw["rope_scaling"] = scaling
    base = raw.get("rope_theta", (raw.get("rope_parameters") or {}).get("rope_theta"))
    raw["rope_parameters"] = {**scaling, "rope_theta": base}
    yarn_config = AutoConfig.for_model(model_type, **raw)
    initializer = ROPE_INIT_FUNCTIONS["yarn"]
    inv, actual_gain = initializer(yarn_config, torch.device("cpu"))
    actual = inv.detach().cpu().numpy()
    parity = compare_tables(actual, equation)
    if not math.isclose(float(actual_gain), gain, rel_tol=0, abs_tol=1e-12):
        raise ValueError("HF YaRN amplitude parity failed")
    parity.update({
        "native_parity": native_parity,
        "native_initializer": f"{native_initializer.__module__}.{native_initializer.__qualname__}",
        "native_initializer_source_sha256": sha256_bytes(inspect.getsource(native_initializer).encode()),
        "native_initializer_file_sha256": sha256_file(Path(inspect.getfile(native_initializer))),
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "initializer": f"{initializer.__module__}.{initializer.__name__}",
        "initializer_source_sha256": sha256_bytes(inspect.getsource(initializer).encode()),
        "initializer_file_sha256": sha256_file(Path(inspect.getfile(initializer))),
        "attention_scaling": float(actual_gain), "device": "cpu",
        "scope": "CPU initializer tolerance parity only; exported tensor is exact installed-HF float32 output",
    })
    return np.ascontiguousarray(actual, dtype="<f4"), parity


def export(args: argparse.Namespace) -> dict[str, Any]:
    identity, native = load_inputs(args.config, args.native_inv)
    factor, reference = bind_reference(
        identity, getattr(args, "reference_receipt", None),
        getattr(args, "target_length", None), args.factor,
    )
    tables, metadata = build_tables(identity, native, factor)
    verification: dict[str, Any] = {"passed": False, "status": "NOT_RUN"}
    if args.verify_transformers:
        tables["official_equation_yarn"], verification = verify_transformers(
            args.config, native, tables["official_equation_yarn"], factor,
            metadata["official_equation_yarn"]["attention_scaling"],
            reference_length=identity["reference_length"],
        )
        metadata["official_equation_yarn"]["table_source"] = "installed HF ROPE_INIT_FUNCTIONS yarn, CPU float32"
        metadata["official_equation_yarn"]["rounding"] = "actual installed-HF initializer float32 output; equation tolerance check passed"
    for name, table in tables.items():
        if not np.isfinite(table).all() or not (table > 0).all() or not np.all(table[:-1] > table[1:]):
            raise ValueError(f"invalid or crossing {name} table")
    receipt = {
        "status": "STATIC_ROPE_BASELINES_EXPORTED", "benchmark_scores_used": reference is not None,
        "long_benchmark_scores_used": False,
        "reference_receipt": reference,
        "reference_length": identity["reference_length"],
        "target_length": getattr(args, "target_length", None),
        "weight_identity_verification": "not performed; no weights loaded; receipt binds declared identity only",
        "runtime_evaluation_status": "NOT_RUN",
        "search_performed": False, "checkpoint": identity, "factor": factor,
        "base_and_length_semantics": "checkpoint b and native_length remain config inputs; confirmed reference_length controls YaRN ramp and target/reference factor only; static NTK changes effective base; neither arm claims fixed-support pure-allocation identification",
        "numpy_version": np.__version__,
        "implementation_sha256": sha256_file(Path(__file__)),
        "config_identity_implementation_sha256": sha256_file(ROOT / "scripts/analysis/export_frozen_coupling_transport.py"),
        "official_equation_reference_sha256": sha256_file(ROOT / "scripts/lib/rope/official_yarn.py"),
        "transformers_verification": verification, "tables": {},
    }
    # Never overwrite an earlier frozen artifact, even at a different factor.
    args.output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        path = args.output / f"{name}_s{factor:g}.npy"
        np.save(path, table, allow_pickle=False)
        receipt["tables"][name] = {
            **metadata[name], "path": path.name, "shape": list(table.shape), "dtype": "<f4",
            "tensor_sha256": tensor_hash(table), "file_sha256": sha256_file(path),
            "fast_endpoint": float(table[0]), "slow_endpoint": float(table[-1]),
        }
    (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-inv", type=Path, required=True)
    parser.add_argument("--factor", "--scale", type=float,
                        help="required without a reference receipt; otherwise must equal target/reference")
    parser.add_argument("--reference-receipt", type=Path)
    parser.add_argument("--target-length", type=int,
                        help="required with a confirmed reference receipt; fixes factor=target/reference")
    parser.add_argument("--verify-transformers", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="new output directory (must not exist)")
    print(json.dumps(export(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
