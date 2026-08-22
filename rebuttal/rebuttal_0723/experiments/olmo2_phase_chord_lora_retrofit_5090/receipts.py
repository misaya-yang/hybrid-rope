"""Receipt, authorization, morph, and standard-PEFT artifact contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping


AUTH_ENV = "OLMO2_PHASE_CHORD_LORA_GPU_AUTHORIZED"
READY_STATUS = "OLMO2_PHASE_CHORD_LORA_RETROFIT_READY_V1"
GPU_READY_STATUS = "OLMO2_PHASE_CHORD_LORA_RETROFIT_GPU_READY_V1"
COMPLETE_STATUS = "OLMO2_PHASE_CHORD_LORA_RETROFIT_COMPLETE_V1"
EXPECTED_CHECKPOINT_SHA256 = (
    "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
)
ARMS = (
    "native",
    "anchored_evq_cosh_tau_2",
    "phase_chord_olmo_r0_lambda_0p1",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def require_gpu_authorization(
    *, cli_authorize: bool, environment: Mapping[str, str]
) -> None:
    if not cli_authorize or environment.get(AUTH_ENV) != "1":
        raise RuntimeError(
            f"GPU execution requires --authorize and {AUTH_ENV}=1"
        )


def tree_hashes(root: str | Path) -> dict[str, dict[str, Any]]:
    directory = Path(root)
    return {
        str(path.relative_to(directory)): {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.name.endswith(".incomplete")
    }


def tensor_bundle_sha256(values: Mapping[str, Any]) -> str:
    import numpy as np
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        value = torch.as_tensor(tensor).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def smoothstep_log_morph(
    native: Any,
    target: Any,
    *,
    step: int,
    morph_steps: int,
) -> tuple[Any, dict[str, Any]]:
    """Return Native->target log-frequency smoothstep for a training step."""
    import torch

    native_tensor = torch.as_tensor(native, dtype=torch.float32)
    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    if native_tensor.shape != target_tensor.shape or native_tensor.ndim != 1:
        raise ValueError("Native and target frequency tables must share shape [P]")
    if int(morph_steps) <= 0 or int(step) <= 0:
        raise ValueError("step and morph_steps must be positive")
    if (
        not torch.isfinite(native_tensor).all()
        or not torch.isfinite(target_tensor).all()
        or not (native_tensor > 0).all()
        or not (target_tensor > 0).all()
    ):
        raise ValueError("frequency tables must be finite and positive")
    if not torch.all(native_tensor[:-1] > native_tensor[1:]):
        raise ValueError("Native frequency table must be strictly decreasing")
    if not torch.all(target_tensor[:-1] > target_tensor[1:]):
        raise ValueError("target frequency table must be strictly decreasing")
    progress = min(float(step) / float(morph_steps), 1.0)
    amount = progress * progress * (3.0 - 2.0 * progress)
    value = torch.exp(
        (1.0 - amount) * native_tensor.log()
        + amount * target_tensor.log()
    )
    if int(step) >= int(morph_steps):
        # Avoid retaining a rounded interpolation when the scientific
        # contract says the deployed table is the exact frozen target.
        value = target_tensor.clone()
    return value, {
        "step": int(step),
        "morph_steps": int(morph_steps),
        "linear_progress": progress,
        "smoothstep_amount": amount,
        "is_exact_target": bool(step >= morph_steps),
    }


def checkpoint_receipt(checkpoint: Path, ready_path: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("status") not in {
        "VERIFIED_READY",
        "OLMO2_INSTRUCT_4K_CONVERSION_READY",
        "OLMO2_INSTRUCT_PRO6000_RULER_MATRIX_READY",
    }:
        raise RuntimeError("checkpoint READY status drift")
    weight = checkpoint / "model.safetensors"
    config_path = checkpoint / "config.json"
    if not weight.is_file():
        raise FileNotFoundError(weight)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    weight_sha = sha256_file(weight)
    if weight_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("released checkpoint SHA drift")
    if ready.get("status") == "VERIFIED_READY":
        recorded = ready.get("canonical_identity", {}).get(
            "historical_model_sha256"
        )
        files = ready.get("repository_files", {}).get("files", [])
        weight_rows = [
            row
            for row in files
            if Path(str(row.get("path", ""))).name == "model.safetensors"
        ]
        if len(weight_rows) != 1 or weight_rows[0].get("sha256") != weight_sha:
            raise RuntimeError("VERIFIED_READY repository weight digest drift")
        recorded_path = ready.get("model_dir")
    else:
        entry = ready.get("checkpoint", {})
        recorded = entry.get("composite_sha256")
        recorded_path = entry.get("checkpoint_path")
    if recorded != weight_sha:
        raise RuntimeError("checkpoint READY canonical digest drift")
    if recorded_path is not None and Path(recorded_path).resolve() != checkpoint:
        raise RuntimeError("checkpoint READY path drift")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = {
        "model_type": "olmo2",
        "hidden_size": 2_048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 100_352,
    }
    if any(config.get(name) != value for name, value in expected_config.items()):
        raise RuntimeError("released OLMo configuration drift")
    return {
        "path": str(checkpoint),
        "weight_sha256": weight_sha,
        "weight_bytes": int(weight.stat().st_size),
        "ready_receipt_sha256": sha256_file(ready_path),
        "config_sha256": sha256_file(config_path),
        "config": expected_config,
    }


def write_ready_receipt(
    *,
    path: Path,
    protocol: Mapping[str, Any],
    inputs: Mapping[str, Any],
    code_sha256: Mapping[str, str],
    run_output: Path,
    cuda_available: bool,
    cuda_initialized: bool,
) -> dict[str, Any]:
    if cuda_initialized:
        raise RuntimeError("preflight must not run after CUDA initialization")
    if path.exists():
        raise FileExistsError(path)
    receipt = {
        "schema_version": 1,
        "status": READY_STATUS,
        "protocol": dict(protocol),
        "inputs": dict(inputs),
        "code_sha256": dict(code_sha256),
        "run_output": str(run_output.resolve()),
        "execution_proof": {
            "cuda_available": bool(cuda_available),
            "cuda_initialized": bool(cuda_initialized),
            "model_loaded": False,
            "optimizer_created": False,
            "training_attempted": False,
        },
    }
    atomic_json(path, receipt)
    return receipt


def validate_ready_receipt(
    *,
    path: Path,
    protocol: Mapping[str, Any],
    inputs: Mapping[str, Any],
    code_sha256: Mapping[str, str],
    run_output: Path,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != READY_STATUS
        or receipt.get("protocol") != dict(protocol)
        or receipt.get("inputs") != dict(inputs)
        or receipt.get("code_sha256") != dict(code_sha256)
        or Path(receipt.get("run_output", "")).resolve()
        != run_output.resolve()
    ):
        raise RuntimeError("READY receipt drift")
    return receipt


def save_standard_peft_bundle(
    *,
    model: Any,
    output: Path,
    inv_freq: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Save a stock PEFT directory plus the static frequency sidecar."""
    import torch

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    config = adapter / "adapter_config.json"
    if not config.is_file():
        raise RuntimeError("standard PEFT adapter_config.json was not saved")
    model_files = list(adapter.glob("adapter_model.*"))
    if len(model_files) != 1:
        raise RuntimeError("standard PEFT adapter must contain one adapter_model file")
    frequency_path = output / "custom_inv_freq.pt"
    temporary = output / "custom_inv_freq.pt.incomplete"
    torch.save(torch.as_tensor(inv_freq).detach().cpu().float().contiguous(), temporary)
    temporary.replace(frequency_path)
    atomic_json(output / "metadata.json", metadata)
    return {
        "adapter_relative_path": "adapter",
        "frequency_relative_path": "custom_inv_freq.pt",
        "metadata_relative_path": "metadata.json",
        "files": tree_hashes(output),
    }


def load_standard_peft_bundle(
    *,
    base_model: Any,
    bundle: Path,
    peft_loader: Callable[[Any, str], Any] | None = None,
    torch_load: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load the standard adapter and inject its exact static HF RoPE table."""
    import torch

    if peft_loader is None:
        from peft import PeftModel

        peft_loader = lambda model, path: PeftModel.from_pretrained(  # noqa: E731
            model, path, is_trainable=False
        )
    if torch_load is None:
        torch_load = torch.load
    adapter = bundle / "adapter"
    frequency_path = bundle / "custom_inv_freq.pt"
    if not (adapter / "adapter_config.json").is_file() or not frequency_path.is_file():
        raise RuntimeError("PEFT bundle is incomplete")
    model = peft_loader(base_model, str(adapter))
    value = torch_load(frequency_path, map_location="cpu", weights_only=True)
    value = torch.as_tensor(value).float().contiguous()
    rotary = model.base_model.model.model.rotary_emb
    if rotary.inv_freq.shape != value.shape:
        raise RuntimeError("saved frequency shape does not match OLMo rotary")
    with torch.no_grad():
        rotary.inv_freq.copy_(value.to(rotary.inv_freq))
        rotary.original_inv_freq = rotary.inv_freq
    from .frequency_assets import float32_sha256

    return model, {
        "adapter_files": tree_hashes(adapter),
        "custom_inv_freq_sha256": sha256_file(frequency_path),
        "active_inv_freq_float32_sha256": float32_sha256(value.numpy()),
        "active_inv_freq_sha256": tensor_bundle_sha256({"inv_freq": value}),
    }


def assert_peft_state_roundtrip(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, Any]:
    expected_sha = tensor_bundle_sha256(expected)
    observed_sha = tensor_bundle_sha256(observed)
    if set(expected) != set(observed) or expected_sha != observed_sha:
        raise RuntimeError("standard PEFT save/load state roundtrip drift")
    return {
        "parameter_names": sorted(expected),
        "parameter_tensors": len(expected),
        "state_sha256": expected_sha,
        "bitwise_equal": True,
    }


__all__ = [
    "ARMS",
    "AUTH_ENV",
    "COMPLETE_STATUS",
    "GPU_READY_STATUS",
    "READY_STATUS",
    "assert_peft_state_roundtrip",
    "atomic_json",
    "checkpoint_receipt",
    "load_standard_peft_bundle",
    "require_gpu_authorization",
    "save_standard_peft_bundle",
    "sha256_file",
    "smoothstep_log_morph",
    "tensor_bundle_sha256",
    "tree_hashes",
    "validate_ready_receipt",
    "write_ready_receipt",
]
