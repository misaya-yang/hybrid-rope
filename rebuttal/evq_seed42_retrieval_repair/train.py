#!/usr/bin/env python3
"""Continue the validated EVQ seed-42 adapter on answer-only retrieval."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.lora_evq_v2.legacy_lora_protocol import (
    sha256_file,
    validate_paper_longalpaca_receipt,
)
from experiments.lora_evq_v2.train_evq_lora import (
    build_training_inv_freq,
    evaluation_strategy_kwargs,
    find_rotary_modules,
    load_frequency_artifact,
    resolve_model_rope_geometry,
    verify_model_inv_freq,
)
from experiments.lora_evq_v2.validate_legacy_lora_artifact import validate_artifact
from rebuttal.frequency_adaptation_8b.train import (
    LoraPairDiagnostics,
    TensorAnswerDataset,
    load_model_identity,
    require_tail_logits_support,
    tail_answer_cross_entropy,
)
from scripts.lib.rope.official_yarn import official_yarn_on_inv_freq

from .prepare_data import validate_bundle, validate_prepared_dir
from .protocol import get_stage, segment_contract


PURPOSE = "evq_seed42_retrieval_repair"
GATE_PURPOSE = "evq_seed42_retrieval_repair_gate"
EVQ_TAU = 1.414
ROPE_BASE = 500_000.0
HEAD_DIM = 128
LORA_CONFIG = {
    "r": 64,
    "alpha": 128,
    "dropout": 0.05,
    "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash a frequency tensor in a dtype-stable canonical representation."""
    tensor = value.detach().cpu().to(torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def runtime_frequency_contract(
    canonical_evq: torch.Tensor,
    *,
    stage: str,
) -> dict[str, Any]:
    """Derive the registered runtime tensor directly from canonical EVQ."""
    spec = get_stage(stage)
    substrate = canonical_evq.detach().cpu().to(torch.float64).view(-1)
    if substrate.numel() != HEAD_DIM // 2:
        raise ValueError("canonical EVQ tensor must contain 64 rotary frequencies")
    if not torch.isfinite(substrate).all() or not torch.all(substrate > 0):
        raise ValueError("canonical EVQ frequencies must be finite and positive")
    runtime, mscale, operator = official_yarn_on_inv_freq(
        substrate,
        head_dim=HEAD_DIM,
        base=ROPE_BASE,
        scale=spec.factor,
        original_max_position_embeddings=8192,
        beta_fast=32.0,
        beta_slow=1.0,
        extrapolation_factor=1.0,
        attn_factor=1.0,
    )
    label = (
        "identity EVQ substrate"
        if spec.factor == 1.0
        else "YaRN-derived generalization on the EVQ substrate"
    )
    operator = dict(operator)
    if spec.factor > 1.0:
        operator["mode"] = "yarn_derived_virtual_dim"
        operator["label"] = label
    return {
        "substrate_inv_freq": substrate,
        "runtime_inv_freq": runtime.detach().cpu().to(torch.float64),
        "factor": spec.factor,
        "mscale": float(mscale),
        "label": label,
        "operator": operator,
        "substrate_tensor_sha256": tensor_sha256(substrate),
        "runtime_tensor_sha256": tensor_sha256(runtime),
    }


def apply_runtime_frequency(
    model: torch.nn.Module,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the runtime frequency and YaRN amplitude to every rotary module."""
    runtime = contract.get("runtime_inv_freq")
    if not torch.is_tensor(runtime):
        raise ValueError("runtime frequency contract is missing runtime_inv_freq")
    mscale = float(contract.get("mscale", math.nan))
    if not math.isfinite(mscale) or mscale <= 0:
        raise ValueError("runtime frequency contract has an invalid mscale")
    modules = find_rotary_modules(model)
    if not modules:
        raise RuntimeError("model exposes no rotary modules")
    changed = []
    for name, module in modules:
        if module.inv_freq.numel() != runtime.numel():
            raise RuntimeError(f"rotary frequency shape mismatch at {name}")
        target = runtime.to(device=module.inv_freq.device, dtype=module.inv_freq.dtype)
        with torch.no_grad():
            module.inv_freq.copy_(target)
            original = getattr(module, "original_inv_freq", None)
            if torch.is_tensor(original):
                if original.shape != target.shape:
                    raise RuntimeError(f"original rotary frequency shape mismatch at {name}")
                original.copy_(target.to(device=original.device, dtype=original.dtype))
        if not hasattr(module, "attention_scaling"):
            raise RuntimeError(f"rotary module {name} has no attention_scaling field")
        module.attention_scaling = mscale
        for attribute in (
            "_cos_cached",
            "_sin_cached",
            "cos_cached",
            "sin_cached",
            "_cos_cache",
            "_sin_cache",
        ):
            if hasattr(module, attribute):
                setattr(module, attribute, None)
        if hasattr(module, "max_seq_len_cached"):
            module.max_seq_len_cached = 0
        changed.append(name)
    verify_model_inv_freq(model, runtime)
    return {
        "patched_modules": len(changed),
        "module_names": changed,
        "factor": float(contract["factor"]),
        "mscale": mscale,
        "runtime_tensor_sha256": tensor_sha256(runtime),
    }


def validate_parent_adapter(
    adapter_dir: Path,
    *,
    longalpaca_manifest: Path,
    model_manifest: Path,
) -> dict[str, Any]:
    """Validate the complete paper-lineage EVQ seed-42 parent artifact."""
    adapter_dir = Path(adapter_dir)
    longalpaca_manifest = Path(longalpaca_manifest)
    model_manifest = Path(model_manifest)
    if not longalpaca_manifest.is_file():
        raise FileNotFoundError(longalpaca_manifest)
    if not model_manifest.is_file():
        raise FileNotFoundError(model_manifest)
    data_record = json.loads(longalpaca_manifest.read_text(encoding="utf-8"))
    validate_paper_longalpaca_receipt(data_record.get("source", {}))
    data_sha = sha256_file(longalpaca_manifest)
    model_sha = sha256_file(model_manifest)
    metadata = validate_artifact(
        adapter_dir,
        expected_method="evq_cosh",
        expected_seed=42,
        expected_data_manifest_sha256=data_sha,
    )
    if metadata.get("data_manifest_sha256") != data_sha:
        raise RuntimeError("parent adapter LongAlpaca manifest hash mismatch")
    if metadata.get("model_manifest_sha256") != model_sha:
        raise RuntimeError("parent adapter model manifest hash mismatch")
    inv_freq, frequency, provenance = load_frequency_artifact(
        adapter_dir / "custom_inv_freq.pt",
        expected_method="evq_cosh",
    )
    canonical, _ = build_training_inv_freq("evq_cosh", HEAD_DIM, ROPE_BASE, EVQ_TAU)
    if not torch.allclose(
        inv_freq.to(torch.float64),
        canonical.to(torch.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("parent adapter frequency is not canonical EVQ-Cosh")
    if frequency.get("midpoint") is not True:
        raise RuntimeError("parent adapter does not record midpoint EVQ quantization")
    return {
        "kind": "legacy_longalpaca_evq_seed42",
        "directory": adapter_dir.name,
        "adapter_sha256": metadata["adapter_sha256"],
        "data_manifest_sha256": data_sha,
        "model_manifest_sha256": model_sha,
        "frequency": provenance,
        "substrate_tensor_sha256": tensor_sha256(inv_freq),
    }


def validate_gate_transition(
    gate_path: Path,
    *,
    allowed_statuses: set[str],
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a stale or weakly bound gate before another GPU stage."""
    gate_path = Path(gate_path)
    if not gate_path.is_file():
        raise FileNotFoundError(gate_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("format_version") != 1 or gate.get("purpose") != GATE_PURPOSE:
        raise RuntimeError("repair gate identity mismatch")
    if gate.get("status") not in allowed_statuses:
        raise RuntimeError(
            f"repair gate status {gate.get('status')!r} is not one of {sorted(allowed_statuses)}"
        )
    bindings = gate.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("repair gate has no bindings")
    for field, expected in expected_bindings.items():
        if bindings.get(field) != expected:
            raise RuntimeError(f"repair gate binding mismatch for {field}")
    extra = sorted(set(bindings) - set(expected_bindings))
    if extra:
        raise RuntimeError(f"repair gate has unregistered bindings: {', '.join(extra)}")
    return gate


def _adapter_model_path(adapter_dir: Path) -> Path:
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        path = Path(adapter_dir) / filename
        if path.is_file():
            return path
    raise FileNotFoundError("adapter directory is missing adapter_model.safetensors/bin")


def validate_repair_parent(adapter_dir: Path) -> dict[str, Any]:
    """Validate one completed repair checkpoint and its two frequency tensors."""
    adapter_dir = Path(adapter_dir)
    protocol_path = adapter_dir / "run_protocol.json"
    frequency_path = adapter_dir / "frequency_artifact.pt"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    if not frequency_path.is_file():
        raise FileNotFoundError(frequency_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format_version") != 1
        or protocol.get("purpose") != PURPOSE
        or protocol.get("status") != "complete_uninterpreted"
    ):
        raise RuntimeError("repair parent protocol identity/status mismatch")
    if int(protocol.get("seed", -1)) != 42:
        raise RuntimeError("repair parent seed mismatch")
    lora = protocol.get("lora")
    if not isinstance(lora, Mapping):
        raise RuntimeError("repair parent protocol has no LoRA contract")
    for field, expected in LORA_CONFIG.items():
        if lora.get(field) != expected:
            raise RuntimeError(f"repair parent LoRA {field} mismatch")
    adapter_path = _adapter_model_path(adapter_dir)
    adapter_sha = sha256_file(adapter_path)
    artifacts = protocol.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("repair parent protocol has no artifact map")
    if artifacts.get("adapter_sha256") != adapter_sha:
        raise RuntimeError("repair parent adapter SHA-256 mismatch")
    if artifacts.get("frequency_sha256") != sha256_file(frequency_path):
        raise RuntimeError("repair parent frequency artifact SHA-256 mismatch")
    frequency = torch.load(frequency_path, map_location="cpu", weights_only=True)
    if not isinstance(frequency, Mapping):
        raise RuntimeError("repair parent frequency artifact must be a mapping")
    substrate = frequency.get("substrate_inv_freq")
    runtime = frequency.get("runtime_inv_freq")
    if not torch.is_tensor(substrate) or not torch.is_tensor(runtime):
        raise RuntimeError("repair parent frequency artifact is missing tensors")
    if tensor_sha256(substrate) != frequency.get("substrate_tensor_sha256"):
        raise RuntimeError("repair parent substrate tensor hash mismatch")
    if tensor_sha256(runtime) != frequency.get("runtime_tensor_sha256"):
        raise RuntimeError("repair parent runtime tensor hash mismatch")
    canonical, _ = build_training_inv_freq("evq_cosh", HEAD_DIM, ROPE_BASE, EVQ_TAU)
    if not torch.allclose(
        substrate.to(torch.float64), canonical.to(torch.float64), rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("repair parent substrate is not canonical EVQ")
    return {
        "kind": "repair_checkpoint",
        "directory": adapter_dir.name,
        "adapter_sha256": adapter_sha,
        "protocol_sha256": sha256_file(protocol_path),
        "frequency_sha256": sha256_file(frequency_path),
        "stage": protocol.get("stage"),
        "segment": int(protocol.get("segment", -1)),
        "factor": float(frequency.get("factor", math.nan)),
        "substrate_inv_freq": substrate.detach().cpu().to(torch.float64),
        "runtime_inv_freq": runtime.detach().cpu().to(torch.float64),
        "model_manifest_sha256": protocol.get("model", {}).get("manifest_sha256"),
        "longalpaca_manifest_sha256": protocol.get("longalpaca_manifest_sha256"),
        "repair_manifest_sha256": protocol.get("repair_manifest_sha256"),
    }


def _load_training_bundle(
    data_dir: Path,
    *,
    stage: str,
    segment: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest = validate_prepared_dir(data_dir)
    filename = f"train_{stage}_segment{int(segment)}.pt"
    record = manifest["files"].get(filename)
    if not isinstance(record, Mapping):
        raise FileNotFoundError(f"repair manifest has no {filename}")
    spec = get_stage(stage)
    path = Path(data_dir) / filename
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    validate_bundle(
        bundle,
        stage=stage,
        split="train",
        segment=int(segment),
        expected_seq_len=spec.seq_len,
        expected_rows=spec.examples_per_segment,
    )
    if sha256_file(path) != record.get("sha256"):
        raise RuntimeError("training shard hash differs from prepared manifest")
    return dict(bundle), dict(record), sha256_file(Path(data_dir) / "manifest.json")


def _runtime_identity() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate"):
        try:
            import importlib.metadata

            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    identity: dict[str, Any] = {
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        identity["cuda_device"] = torch.cuda.get_device_name(0)
        identity["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return identity


def _code_sha256() -> str:
    project_root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("protocol.py").resolve(),
        Path(__file__).with_name("prepare_data.py").resolve(),
        (project_root / "scripts/lib/rope/official_yarn.py").resolve(),
        (project_root / "rebuttal/frequency_adaptation_8b/train.py").resolve(),
        (project_root / "experiments/lora_evq_v2/validate_legacy_lora_artifact.py").resolve(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.incomplete")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.incomplete")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gate_for_parent(
    gate_path: Path,
    *,
    stage: str,
    segment: int,
    parent: Mapping[str, Any],
    factor: float,
    model_manifest_sha256: str,
    longalpaca_manifest_sha256: str,
    repair_manifest_sha256: str,
) -> dict[str, Any]:
    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    bindings = gate.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("repair parent gate has no bindings")
    expected_statuses = {"rescue_allowed"} if int(segment) == 2 else {"pass"}
    expected_stage = stage if int(segment) == 2 else "r8"
    expected_segment = 1 if int(segment) == 2 else int(parent["segment"])
    expected = dict(bindings)
    expected.update(
        {
            "stage": expected_stage,
            "segment": expected_segment,
            "factor": 1.0 if expected_stage == "r8" else float(factor),
            "checkpoint_adapter_sha256": parent["adapter_sha256"],
            "model_manifest_sha256": model_manifest_sha256,
            "longalpaca_manifest_sha256": longalpaca_manifest_sha256,
            "repair_manifest_sha256": repair_manifest_sha256,
        }
    )
    return validate_gate_transition(
        gate_path,
        allowed_statuses=expected_statuses,
        expected_bindings=expected,
    )


def build_run_protocol(
    *,
    stage: str,
    segment: int,
    seed: int,
    model_identity: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    longalpaca_manifest_sha256: str,
    repair_manifest_sha256: str,
    training_file: str,
    training_file_sha256: str,
    frequency: Mapping[str, Any],
    gate_sha256: str | None,
) -> dict[str, Any]:
    """Build the immutable pre-training protocol written inside incomplete output."""
    spec = get_stage(stage)
    return {
        "format_version": 1,
        "purpose": PURPOSE,
        "status": "running_no_result",
        "stage": stage,
        "segment": int(segment),
        "seed": int(seed),
        "model": dict(model_identity),
        "longalpaca_manifest_sha256": longalpaca_manifest_sha256,
        "repair_manifest_sha256": repair_manifest_sha256,
        "parent": {key: value for key, value in parent_identity.items() if not torch.is_tensor(value)},
        "gate_sha256": gate_sha256,
        "training": {
            **segment_contract(stage, segment),
            "file": training_file,
            "file_sha256": training_file_sha256,
            "seq_len": spec.seq_len,
            "distance": [spec.min_distance, spec.max_distance],
            "objective": "answer_only_tail_logits_causal_cross_entropy",
            "precision": "bfloat16",
            "gradient_checkpointing": True,
        },
        "frequency": {
            "factor": float(frequency["factor"]),
            "mscale": float(frequency["mscale"]),
            "label": str(frequency["label"]),
            "operator": dict(frequency["operator"]),
            "substrate_tensor_sha256": str(frequency["substrate_tensor_sha256"]),
            "runtime_tensor_sha256": str(frequency["runtime_tensor_sha256"]),
            "derivation": "factor applied directly to canonical EVQ substrate",
        },
        "lora": dict(LORA_CONFIG),
        "code_sha256": _code_sha256(),
        "runtime": _runtime_identity(),
    }


def _validate_completed_output(output_dir: Path) -> dict[str, Any]:
    protocol_path = output_dir / "run_protocol.json"
    frequency_path = output_dir / "frequency_artifact.pt"
    diagnostics_path = output_dir / "frequency_diagnostics.json"
    state_path = output_dir / "trainer_state.json"
    for path in (protocol_path, frequency_path, diagnostics_path, state_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "complete_uninterpreted":
        raise RuntimeError("completed repair output has non-complete status")
    adapter_path = _adapter_model_path(output_dir)
    artifacts = protocol.get("artifacts", {})
    expected = {
        "adapter_sha256": sha256_file(adapter_path),
        "frequency_sha256": sha256_file(frequency_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "trainer_state_sha256": sha256_file(state_path),
    }
    for key, value in expected.items():
        if artifacts.get(key) != value:
            raise RuntimeError(f"completed repair artifact hash mismatch for {key}")
    validate_repair_parent(output_dir)
    return protocol


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--longalpaca-manifest", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("r8", "r16"), required=True)
    parser.add_argument("--segment", choices=(1, 2), type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seed != 42:
        raise ValueError("retrieval-repair v1 is registered only for seed 42")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("retrieval-repair v1 requires one process")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    bundle, training_record, repair_manifest_sha = _load_training_bundle(
        args.data_dir,
        stage=args.stage,
        segment=args.segment,
    )
    longalpaca_sha = sha256_file(args.longalpaca_manifest)
    model_identity = load_model_identity(args.model_name, args.model_manifest)
    model_manifest_sha = model_identity["manifest_sha256"]

    if args.stage == "r8" and args.segment == 1:
        if args.gate is not None:
            raise ValueError("r8 segment 1 starts from the legacy parent and must not receive a gate")
        parent = validate_parent_adapter(
            args.parent_adapter,
            longalpaca_manifest=args.longalpaca_manifest,
            model_manifest=args.model_manifest,
        )
        canonical, _, _ = load_frequency_artifact(
            args.parent_adapter / "custom_inv_freq.pt",
            expected_method="evq_cosh",
        )
        gate_sha = None
    else:
        parent = validate_repair_parent(args.parent_adapter)
        if parent.get("model_manifest_sha256") != model_manifest_sha:
            raise RuntimeError("repair parent model manifest differs from this run")
        if parent.get("longalpaca_manifest_sha256") != longalpaca_sha:
            raise RuntimeError("repair parent LongAlpaca manifest differs from this run")
        if parent.get("repair_manifest_sha256") != repair_manifest_sha:
            raise RuntimeError("repair parent data manifest differs from this run")
        expected_parent_factor = 2.0 if args.stage == "r16" and args.segment == 2 else 1.0
        if not math.isclose(
            float(parent.get("factor", math.nan)),
            expected_parent_factor,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("repair parent runtime factor differs from the stage transition")
        expected_prior = {
            ("r8", 2): ("r8", 1),
            ("r16", 2): ("r16", 1),
        }.get((args.stage, args.segment))
        if args.stage == "r16" and args.segment == 1:
            if parent.get("stage") != "r8" or parent.get("segment") not in {1, 2}:
                raise RuntimeError("r16 segment 1 requires the selected passing r8 checkpoint")
        elif expected_prior != (parent.get("stage"), parent.get("segment")):
            raise RuntimeError(
                f"{args.stage} segment {args.segment} requires parent {expected_prior}, "
                f"found {(parent.get('stage'), parent.get('segment'))}"
            )
        if args.gate is None:
            raise ValueError("continuing from a repair checkpoint requires --gate")
        _gate_for_parent(
            args.gate,
            stage=args.stage,
            segment=args.segment,
            parent=parent,
            factor=get_stage(args.stage).factor,
            model_manifest_sha256=model_manifest_sha,
            longalpaca_manifest_sha256=longalpaca_sha,
            repair_manifest_sha256=repair_manifest_sha,
        )
        gate_sha = sha256_file(args.gate)
        canonical = parent["substrate_inv_freq"]

    expected_canonical, _ = build_training_inv_freq(
        "evq_cosh", HEAD_DIM, ROPE_BASE, EVQ_TAU
    )
    if not torch.allclose(
        canonical.to(torch.float64),
        expected_canonical.to(torch.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("training parent does not carry the canonical EVQ substrate")
    frequency = runtime_frequency_contract(canonical, stage=args.stage)
    protocol = build_run_protocol(
        stage=args.stage,
        segment=args.segment,
        seed=args.seed,
        model_identity=model_identity,
        parent_identity=parent,
        longalpaca_manifest_sha256=longalpaca_sha,
        repair_manifest_sha256=repair_manifest_sha,
        training_file=f"train_{args.stage}_segment{args.segment}.pt",
        training_file_sha256=str(training_record["sha256"]),
        frequency=frequency,
        gate_sha256=gate_sha,
    )
    preview = {
        "stage": args.stage,
        "segment": args.segment,
        "parent_adapter_sha256": parent["adapter_sha256"],
        "repair_manifest_sha256": repair_manifest_sha,
        "factor": frequency["factor"],
        "mscale": frequency["mscale"],
        "label": frequency["label"],
        "steps": protocol["training"]["steps"],
        "physical_tokens": protocol["training"]["tokens"],
        "output": args.output_dir.name,
    }
    print(json.dumps(preview, indent=2, sort_keys=True), flush=True)
    require_tail_logits_support()
    if args.dry_run:
        print(json.dumps({"status": "dry_run_valid", "protocol": protocol}, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("retrieval-repair protocol requires BF16-capable CUDA")

    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        local_files_only=True,
    )
    geometry = resolve_model_rope_geometry(model.config)
    if geometry.head_dim != HEAD_DIM or not math.isclose(
        geometry.rope_base, ROPE_BASE, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RuntimeError("loaded model geometry differs from the registered LLaMA-3-8B contract")
    model = PeftModel.from_pretrained(
        model,
        str(args.parent_adapter),
        is_trainable=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    runtime_application = apply_runtime_frequency(model, frequency)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable <= 0:
        raise RuntimeError("continued adapter exposes no trainable parameters")
    forward_target = model.get_base_model() if hasattr(model, "get_base_model") else model
    if "logits_to_keep" not in inspect.signature(forward_target.forward).parameters:
        raise RuntimeError("loaded model does not expose logits_to_keep")
    diagnostics = LoraPairDiagnostics(model, HEAD_DIM)

    incomplete = args.output_dir.with_name(f"{args.output_dir.name}.incomplete")
    if incomplete.exists():
        raise FileExistsError(incomplete)
    incomplete.mkdir(parents=True, exist_ok=False)
    protocol["lora"]["trainable_parameters"] = int(trainable)
    protocol["frequency"]["runtime_application"] = runtime_application
    _atomic_json_dump(protocol, incomplete / "run_protocol.json")

    training_kwargs: dict[str, Any] = {
        "output_dir": str(incomplete / "trainer"),
        "max_steps": 32,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": get_stage(args.stage).accumulation,
        "learning_rate": 2e-5,
        "warmup_steps": 4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "logging_steps": 1,
        "save_strategy": "no",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }
    training_kwargs.update(evaluation_strategy_kwargs(TrainingArguments, "no"))

    class TailAnswerTrainer(Trainer):
        def compute_loss(
            self,
            training_model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            del num_items_in_batch
            labels = inputs["labels"]
            answer_length = int(labels.ne(-100).sum(dim=1)[0])
            outputs = training_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                logits_to_keep=answer_length + 1,
            )
            loss = tail_answer_cross_entropy(outputs.logits, inputs["input_ids"], labels)
            return (loss, outputs) if return_outputs else loss

    trainer = TailAnswerTrainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=TensorAnswerDataset(bundle),
        data_collator=default_data_collator,
    )
    trainer.model_accepts_loss_kwargs = False
    started = time.time()
    trainer.train()
    elapsed = time.time() - started
    if int(trainer.state.global_step) != 32:
        raise RuntimeError(f"training ended at step {trainer.state.global_step}, expected 32")
    apply_runtime_frequency(model, frequency)
    diagnostic_record = diagnostics.finalize(
        model,
        frequency["runtime_inv_freq"],
        frequency["runtime_inv_freq"],
    )
    diagnostic_record.update(
        {
            "training_seconds": elapsed,
            "global_step": int(trainer.state.global_step),
            "train_log_history": trainer.state.log_history,
        }
    )
    model.save_pretrained(incomplete)
    tokenizer.save_pretrained(incomplete)
    trainer.state.save_to_json(str(incomplete / "trainer_state.json"))
    frequency_record = {
        "format_version": 1,
        "purpose": PURPOSE,
        "stage": args.stage,
        "segment": args.segment,
        "factor": float(frequency["factor"]),
        "mscale": float(frequency["mscale"]),
        "label": frequency["label"],
        "operator": frequency["operator"],
        "substrate_inv_freq": frequency["substrate_inv_freq"],
        "runtime_inv_freq": frequency["runtime_inv_freq"],
        "substrate_tensor_sha256": frequency["substrate_tensor_sha256"],
        "runtime_tensor_sha256": frequency["runtime_tensor_sha256"],
    }
    _atomic_torch_save(frequency_record, incomplete / "frequency_artifact.pt")
    _atomic_json_dump(diagnostic_record, incomplete / "frequency_diagnostics.json")
    adapter_path = _adapter_model_path(incomplete)
    protocol["status"] = "complete_uninterpreted"
    protocol["artifacts"] = {
        "adapter_file": adapter_path.name,
        "adapter_sha256": sha256_file(adapter_path),
        "frequency_file": "frequency_artifact.pt",
        "frequency_sha256": sha256_file(incomplete / "frequency_artifact.pt"),
        "diagnostics_file": "frequency_diagnostics.json",
        "diagnostics_sha256": sha256_file(incomplete / "frequency_diagnostics.json"),
        "trainer_state_file": "trainer_state.json",
        "trainer_state_sha256": sha256_file(incomplete / "trainer_state.json"),
    }
    _atomic_json_dump(protocol, incomplete / "run_protocol.json")
    _validate_completed_output(incomplete)
    os.replace(incomplete, args.output_dir)
    print(
        json.dumps(
            {
                "status": "complete_uninterpreted",
                "stage": args.stage,
                "segment": args.segment,
                "global_step": 32,
                "training_seconds": elapsed,
                "adapter_sha256": protocol["artifacts"]["adapter_sha256"],
                "output": args.output_dir.name,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
