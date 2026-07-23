#!/usr/bin/env python3
"""Train one gated phase of the matched Geo/EVQ 8B adaptation experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from experiments.lora_evq_v2.train_evq_lora import (
    build_training_inv_freq,
    evaluation_strategy_kwargs,
    find_rotary_modules,
    inject_inv_freq,
    load_frequency_artifact,
    resolve_model_rope_geometry,
    verify_model_inv_freq,
)
from experiments.lora_evq_v2.prepare_legacy_model_manifest import validate_model_manifest

from .curriculum import (
    answer_only_labels,
    get_phase,
    half_split_pair_energy,
    log_frequency_homotopy,
)
from .prepare_data import sha256_file, validate_bundle

EVQ_TAU = 1.414
LORA_DROPOUT = 0.05


def phase_frequency_endpoints(
    native_inv_freq: torch.Tensor,
    evq_inv_freq: torch.Tensor,
    phase_name: str,
    arm: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return exact start/end tensors for one matched curriculum arm."""
    get_phase(phase_name)
    if arm not in {"geo", "evq"}:
        raise ValueError("arm must be 'geo' or 'evq'")
    if phase_name == "warmup":
        if arm != "geo":
            raise ValueError("warmup is shared native Geo and cannot use the EVQ arm")
        return native_inv_freq.clone(), native_inv_freq.clone()
    target = native_inv_freq if arm == "geo" else evq_inv_freq
    if phase_name == "transition":
        return native_inv_freq.clone(), target.clone()
    return target.clone(), target.clone()


@dataclass(frozen=True)
class FrequencyTransition:
    """Step-indexed smooth path with exact tensors at both optimizer endpoints."""

    start: torch.Tensor
    end: torch.Tensor
    steps: int

    def __post_init__(self) -> None:
        if int(self.steps) < 2:
            raise ValueError("frequency transition requires at least two optimizer steps")
        # Validate tensor contracts now rather than during the first GPU step.
        log_frequency_homotopy(self.start, self.end, 0.5)

    def at_step(self, step: int) -> torch.Tensor:
        step = int(step)
        if not 0 <= step < int(self.steps):
            raise ValueError(f"transition step must lie in [0, {self.steps - 1}]")
        progress = step / float(self.steps - 1)
        return log_frequency_homotopy(self.start, self.end, progress)


def effective_lora_pair_energy(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    scaling: float,
    head_dim: int,
) -> torch.Tensor:
    """Per-rotary-pair squared energy of one effective LoRA matrix."""
    if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_b.shape[1] != lora_a.shape[0]:
        raise ValueError("LoRA matrices must have shapes [r,in] and [out,r]")
    a = lora_a.to(torch.float64)
    b = lora_b.to(torch.float64)
    gram = a.matmul(a.t())
    row_energy = (b.matmul(gram) * b).sum(dim=1).clamp_min(0.0) * float(scaling) ** 2
    return half_split_pair_energy(row_energy, head_dim=head_dim)


def _effective_delta_row_energy(
    start_a: torch.Tensor,
    start_b: torch.Tensor,
    end_a: torch.Tensor,
    end_b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Row norms of ``scale * (B1 A1 - B0 A0)`` without an out-by-in matrix."""
    a0 = start_a.to(torch.float64)
    b0 = start_b.to(torch.float64)
    a1 = end_a.to(torch.float64)
    b1 = end_b.to(torch.float64)
    if a0.shape != a1.shape or b0.shape != b1.shape or b0.shape[1] != a0.shape[0]:
        raise ValueError("starting and ending LoRA factors must have matching shapes")
    gram_11 = a1.matmul(a1.t())
    gram_00 = a0.matmul(a0.t())
    gram_10 = a1.matmul(a0.t())
    term_11 = (b1.matmul(gram_11) * b1).sum(dim=1)
    term_00 = (b0.matmul(gram_00) * b0).sum(dim=1)
    cross = (b1.matmul(gram_10) * b0).sum(dim=1)
    return (term_11 + term_00 - 2.0 * cross).clamp_min(0.0) * float(scaling) ** 2


class TensorAnswerDataset(torch.utils.data.Dataset):
    """Fixed token tensors whose labels are materialized answer-only on access."""

    def __init__(self, bundle: Dict[str, Any]):
        input_ids = bundle.get("input_ids")
        answer_start = bundle.get("answer_start")
        answer_end = bundle.get("answer_end")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("bundle input_ids must be a two-dimensional tensor")
        rows = input_ids.shape[0]
        if not torch.is_tensor(answer_start) or answer_start.numel() != rows:
            raise ValueError("bundle answer_start must have one entry per row")
        if not torch.is_tensor(answer_end) or answer_end.numel() != rows:
            raise ValueError("bundle answer_end must have one entry per row")
        self.input_ids = input_ids
        self.answer_start = answer_start
        self.answer_end = answer_end

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        input_ids = self.input_ids[index].to(torch.long)
        labels = answer_only_labels(
            input_ids,
            int(self.answer_start[index]),
            int(self.answer_end[index]),
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }


def tail_answer_cross_entropy(
    tail_logits: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute causal CE from only the answer-predecessor tail logits."""
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("input_ids and labels must have the same [batch, sequence] shape")
    supervised = labels.ne(-100)
    answer_counts = supervised.sum(dim=1)
    if answer_counts.numel() == 0 or int(answer_counts.min()) <= 0:
        raise ValueError("every row must contain a non-empty answer")
    if not torch.all(answer_counts == answer_counts[0]):
        raise ValueError("every row in a batch must use the same answer length")
    answer_length = int(answer_counts[0])
    expected_mask = torch.zeros_like(supervised)
    expected_mask[:, -answer_length:] = True
    if not torch.equal(supervised, expected_mask):
        raise ValueError("answer supervision must be one contiguous tail span")
    if not torch.equal(labels[:, -answer_length:], input_ids[:, -answer_length:]):
        raise ValueError("supervised answer labels must equal the input answer tokens")
    expected_shape = (input_ids.shape[0], answer_length + 1)
    if tail_logits.ndim != 3 or tail_logits.shape[:2] != expected_shape:
        raise ValueError(f"tail_logits must have shape [batch, answer_length+1, vocab], got {tuple(tail_logits.shape)}")
    predictors = tail_logits[:, :-1, :].float().reshape(-1, tail_logits.shape[-1])
    targets = input_ids[:, -answer_length:].to(predictors.device, dtype=torch.long).reshape(-1)
    return torch.nn.functional.cross_entropy(predictors, targets, reduction="mean")


def _iter_lora_components(model) -> Iterable[Tuple[str, str, torch.Tensor, torch.Tensor, float]]:
    """Yield q/k LoRA A/B factors using the active PEFT adapter."""
    for module_name, module in model.named_modules():
        projection = module_name.rsplit(".", 1)[-1]
        if projection not in {"q_proj", "k_proj"}:
            continue
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        keys = list(lora_a.keys())
        if not keys:
            continue
        active_value = getattr(module, "active_adapters", None)
        if active_value is None:
            active_value = getattr(module, "active_adapter", None)
        if isinstance(active_value, str):
            active = [active_value]
        else:
            active = list(active_value or [])
        adapter_name = active[0] if active and active[0] in keys else keys[0]
        if adapter_name not in lora_b:
            raise RuntimeError(f"LoRA A/B adapter mismatch at {module_name}")
        scaling_map = getattr(module, "scaling", {})
        scaling = float(scaling_map[adapter_name])
        kind = "q" if projection == "q_proj" else "k"
        yield (
            module_name,
            kind,
            lora_a[adapter_name].weight,
            lora_b[adapter_name].weight,
            scaling,
        )


class LoraPairDiagnostics:
    """Accumulate q/k gradient energy and phase-local effective update energy."""

    def __init__(self, model, head_dim: int):
        self.head_dim = int(head_dim)
        self.gradient: Dict[str, Optional[torch.Tensor]] = {"q": None, "k": None}
        self.hook_calls = {"q": 0, "k": 0}
        self.handles = []
        self.start_factors: Dict[str, Tuple[str, torch.Tensor, torch.Tensor, float]] = {}
        components = list(_iter_lora_components(model))
        if not components:
            raise RuntimeError("no active q/k LoRA components found for diagnostics")
        for name, kind, a_weight, b_weight, scaling in components:
            self.start_factors[name] = (
                kind,
                a_weight.detach().cpu().float().clone(),
                b_weight.detach().cpu().float().clone(),
                scaling,
            )

            def hook(gradient, *, pair_kind=kind):
                row_energy = gradient.detach().float().square().sum(dim=1)
                pair_energy = half_split_pair_energy(row_energy, head_dim=self.head_dim)
                if self.gradient[pair_kind] is None:
                    self.gradient[pair_kind] = torch.zeros_like(pair_energy)
                self.gradient[pair_kind].add_(pair_energy)
                self.hook_calls[pair_kind] += 1
                return gradient

            self.handles.append(b_weight.register_hook(hook))

    def finalize(self, model, frequency_start: torch.Tensor, frequency_end: torch.Tensor) -> Dict[str, Any]:
        final_energy = {
            "q": torch.zeros(self.head_dim // 2, dtype=torch.float64),
            "k": torch.zeros(self.head_dim // 2, dtype=torch.float64),
        }
        phase_delta_energy = {
            "q": torch.zeros(self.head_dim // 2, dtype=torch.float64),
            "k": torch.zeros(self.head_dim // 2, dtype=torch.float64),
        }
        seen = set()
        for name, kind, a_weight, b_weight, scaling in _iter_lora_components(model):
            if name not in self.start_factors:
                raise RuntimeError(f"LoRA component appeared during training: {name}")
            start_kind, start_a, start_b, start_scaling = self.start_factors[name]
            if kind != start_kind or not math.isclose(scaling, start_scaling):
                raise RuntimeError(f"LoRA component identity changed during training: {name}")
            end_a = a_weight.detach().cpu().float()
            end_b = b_weight.detach().cpu().float()
            final_energy[kind] += effective_lora_pair_energy(
                end_a,
                end_b,
                scaling=scaling,
                head_dim=self.head_dim,
            )
            delta_rows = _effective_delta_row_energy(
                start_a,
                start_b,
                end_a,
                end_b,
                scaling,
            )
            phase_delta_energy[kind] += half_split_pair_energy(delta_rows, head_dim=self.head_dim)
            seen.add(name)
        if seen != set(self.start_factors):
            raise RuntimeError("a q/k LoRA component disappeared during training")
        for handle in self.handles:
            handle.remove()

        shift = (
            frequency_end.detach().cpu().to(torch.float64).log()
            - frequency_start.detach().cpu().to(torch.float64).log()
        ).abs()
        output: Dict[str, Any] = {
            "pairing": "llama_rotate_half_i_plus_head_dim_over_2",
            "head_dim": self.head_dim,
            "absolute_log_frequency_shift": shift.tolist(),
            "hook_calls": dict(self.hook_calls),
        }
        for kind in ("q", "k"):
            gradient_device = self.gradient[kind]
            gradient = (
                torch.zeros(self.head_dim // 2, dtype=torch.float64)
                if gradient_device is None
                else gradient_device.detach().cpu().to(torch.float64)
            )
            output[f"{kind}_pair_gradient_energy"] = gradient.tolist()
            output[f"{kind}_gradient_pair_coverage"] = float((gradient > 0).double().mean())
            output[f"{kind}_final_effective_pair_energy"] = final_energy[kind].tolist()
            output[f"{kind}_phase_delta_pair_energy"] = phase_delta_energy[kind].tolist()
        return output


def _runtime_identity() -> Dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    identity = {
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        identity["cuda_device"] = torch.cuda.get_device_name(0)
        identity["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return identity


def load_model_identity(model_name: str, manifest_path: Path) -> Dict[str, Any]:
    """Validate a precomputed full-byte manifest without rehashing on paid GPU time."""
    model_dir = Path(model_name).expanduser()
    if not model_dir.is_dir():
        raise ValueError("frequency-adaptation training requires a local model directory")
    if not manifest_path.is_file():
        raise FileNotFoundError("precomputed model manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_model_manifest(model_dir, manifest, verify_hashes=False)
    return {
        "identifier": model_dir.name,
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "files": [
            {
                "name": record["name"],
                "size_bytes": int(record["size_bytes"]),
                "sha256": record["sha256"],
            }
            for record in manifest["files"]
        ],
    }


def require_tail_logits_support() -> None:
    """Fail during CPU preflight if Transformers would compute full-sequence logits."""
    try:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
    except ImportError as exc:
        raise RuntimeError("Transformers with LLaMA support is required") from exc
    if "logits_to_keep" not in inspect.signature(LlamaForCausalLM.forward).parameters:
        raise RuntimeError("installed Transformers lacks logits_to_keep; refusing full-sequence 128K-vocab logits")


def _atomic_json_dump(value: Dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(value: Dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_bundle(path: Path, phase_name: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"training bundle not found: {path.name}")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    phase = get_phase(phase_name)
    validate_bundle(
        bundle,
        expected_phase=phase_name,
        expected_split="train",
        expected_seq_len=phase.seq_len,
    )
    if int(bundle["input_ids"].shape[0]) != phase.training_examples:
        raise ValueError(f"{phase_name} bundle must contain exactly {phase.training_examples} examples")
    answer_lengths = bundle["answer_end"] - bundle["answer_start"]
    if not torch.all(answer_lengths == 13):
        raise ValueError("registered training bundles must supervise 12 value tokens plus EOS")
    return bundle


def _adapter_model_path(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError("adapter source is missing adapter_model.safetensors/bin")


def _validate_adapter_source(
    adapter_dir: Path,
    phase_name: str,
    arm: str,
    seed: int,
) -> Dict[str, Any]:
    prior_phase = {
        "transition": "warmup",
        "exact_8k": "transition",
        "exact_16k": "exact_8k",
    }.get(phase_name)
    if prior_phase is None:
        raise ValueError("warmup must not load an adapter source")
    protocol_path = adapter_dir / "run_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("adapter source is missing run_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "complete_uninterpreted":
        raise RuntimeError("adapter source phase is not complete_uninterpreted")
    expected_arm = "geo" if prior_phase == "warmup" else arm
    if protocol.get("phase") != prior_phase or protocol.get("arm") != expected_arm:
        raise RuntimeError(
            f"adapter source must be {prior_phase}/{expected_arm}, found {protocol.get('phase')}/{protocol.get('arm')}"
        )
    if int(protocol.get("seed", -1)) != int(seed):
        raise RuntimeError("adapter source seed does not match this phase")
    expected_method = "evq_cosh" if expected_arm == "evq" else "native_geo"
    _, frequency_artifact, frequency_provenance = load_frequency_artifact(
        adapter_dir / "custom_inv_freq.pt",
        expected_method=expected_method,
    )
    if frequency_artifact.get("phase") != prior_phase or frequency_artifact.get("arm") != expected_arm:
        raise RuntimeError("adapter source frequency artifact phase/arm does not match its protocol")
    adapter_path = _adapter_model_path(adapter_dir)
    return {
        "directory": adapter_dir.name,
        "adapter_file": adapter_path.name,
        "adapter_sha256": sha256_file(adapter_path),
        "protocol_sha256": sha256_file(protocol_path),
        "frequency": frequency_provenance,
        "lora": protocol.get("lora"),
        "model": protocol.get("model"),
    }


def _capture_native_inv_freq(model) -> torch.Tensor:
    modules = find_rotary_modules(model)
    if not modules:
        raise RuntimeError("model exposes no rotary inv_freq tensor")
    native = modules[0][1].inv_freq.detach().cpu().to(torch.float64).clone()
    for name, module in modules[1:]:
        current = module.inv_freq.detach().cpu().to(torch.float64)
        if current.shape != native.shape or not torch.allclose(current, native, rtol=1e-6, atol=1e-9):
            raise RuntimeError(f"loaded model has inconsistent native frequencies at {name}")
    return native


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one phase of the LLaMA-3-8B RoPE frequency curriculum")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("warmup", "transition", "exact_8k", "exact_16k"),
    )
    parser.add_argument("--arm", required=True, choices=("geo", "evq"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-from", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase = get_phase(args.phase)
    if args.phase == "warmup":
        if args.arm != "geo":
            raise ValueError("warmup must use --arm geo")
        if args.adapter_from is not None:
            raise ValueError("warmup must start from a fresh base model, not an adapter")
        adapter_source = None
    else:
        if args.adapter_from is None:
            raise ValueError(f"{args.phase} requires --adapter-from")
        adapter_source = _validate_adapter_source(args.adapter_from, args.phase, args.arm, args.seed)
    if args.lora_r <= 0:
        raise ValueError("lora-r must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("v1 protocol is single-process so matched effective batch stays exact")
    if args.output_dir.exists():
        raise FileExistsError("output directory must not already exist")

    bundle = _load_bundle(args.data, args.phase)
    if adapter_source is not None:
        source_lora = adapter_source.get("lora") or {}
        expected_lora = {
            "r": int(args.lora_r),
            "alpha": int(2 * args.lora_r),
            "dropout": LORA_DROPOUT,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        for key, expected in expected_lora.items():
            if source_lora.get(key) != expected:
                raise RuntimeError(f"adapter source LoRA {key}={source_lora.get(key)!r} does not match {expected!r}")
    configuration = {
        "phase": args.phase,
        "arm": args.arm,
        "seed": int(args.seed),
        "seq_len": phase.seq_len,
        "distance": [phase.min_distance, phase.max_distance],
        "steps": phase.steps,
        "effective_batch": phase.effective_batch,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": phase.effective_batch,
        "gradient_accumulation_loss_scaling": "mean_loss_divided_by_accumulation_steps",
        "tokens_per_step": phase.tokens_per_step,
        "training_tokens": phase.training_tokens,
        "learning_rate": phase.learning_rate,
        "supervised_tokens_per_example": 13,
        "data": {
            "name": args.data.name,
            "sha256": sha256_file(args.data),
            "generation_seed": int(bundle["seed"]),
            "trainer_shuffle_seed": int(args.seed),
        },
        "adapter_source": adapter_source,
    }
    model_identity = load_model_identity(args.model_name, args.model_manifest)
    configuration["model"] = model_identity
    if adapter_source is not None and adapter_source.get("model") != model_identity:
        raise RuntimeError("adapter source and current run do not use the same model bytes")
    require_tail_logits_support()
    if args.dry_run:
        print(json.dumps(configuration, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("default protocol requires BF16-capable CUDA hardware")

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.model_name).expanduser().is_dir(),
    )
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": {"": 0},
        "local_files_only": Path(args.model_name).expanduser().is_dir(),
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.config.use_cache = False
    geometry = resolve_model_rope_geometry(model.config)
    native_inv_freq = _capture_native_inv_freq(model)
    canonical_native, _ = build_training_inv_freq("native_geo", geometry.head_dim, geometry.rope_base, EVQ_TAU)
    if not torch.allclose(native_inv_freq, canonical_native, rtol=1e-5, atol=1e-9):
        raise RuntimeError(
            "loaded checkpoint frequencies are not canonical native Geo; "
            "do not silently bridge from a scaled or modified model"
        )
    evq_inv_freq, _ = build_training_inv_freq("evq_cosh", geometry.head_dim, geometry.rope_base, EVQ_TAU)
    frequency_start, frequency_end = phase_frequency_endpoints(native_inv_freq, evq_inv_freq, args.phase, args.arm)
    if args.adapter_from is not None:
        expected_method = "evq_cosh" if args.arm == "evq" and args.phase != "transition" else "native_geo"
        source_inv_freq, _, _ = load_frequency_artifact(
            args.adapter_from / "custom_inv_freq.pt",
            expected_method=expected_method,
        )
        if not torch.allclose(
            source_inv_freq.to(torch.float64),
            frequency_start.to(torch.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("adapter source frequency tensor is not the exact phase start")

    if args.phase == "warmup":
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=2 * args.lora_r,
                lora_dropout=LORA_DROPOUT,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            str(args.adapter_from),
            is_trainable=True,
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    inject_inv_freq(model, frequency_start)
    verify_model_inv_freq(model, frequency_start)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable <= 0:
        raise RuntimeError("no trainable adapter parameters found")
    forward_target = model.get_base_model() if hasattr(model, "get_base_model") else model
    if "logits_to_keep" not in inspect.signature(forward_target.forward).parameters:
        raise RuntimeError("installed Transformers lacks logits_to_keep; refusing full-sequence 128K-vocab logits")
    diagnostics = LoraPairDiagnostics(model, geometry.head_dim)
    callbacks: List[Any] = []
    transition = None
    if phase.transition:
        transition = FrequencyTransition(frequency_start, frequency_end, phase.steps)

        class FrequencyTransitionCallback(TrainerCallback):
            def on_step_begin(self, training_args, state, control, **kwargs):
                step = min(int(state.global_step), phase.steps - 1)
                inject_inv_freq(model, transition.at_step(step))
                return control

        callbacks.append(FrequencyTransitionCallback())

    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_protocol = {
        "format_version": 1,
        "purpose": "llama8b_rope_frequency_adaptation",
        **configuration,
        "model": model_identity,
        "runtime": _runtime_identity(),
        "rope": {
            "base_checkpoint_frequency": "loaded_native_geo",
            "phase_start": (
                "evq_cosh" if args.arm == "evq" and args.phase in {"exact_8k", "exact_16k"} else "native_geo"
            ),
            "target": "native_geo" if args.arm == "geo" else "evq_cosh",
            "tau": None if args.arm == "geo" else EVQ_TAU,
            "base": geometry.rope_base,
            "head_dim": geometry.head_dim,
            "path": "smoothstep_log_frequency" if phase.transition else "fixed_endpoint",
        },
        "lora": {
            "r": int(args.lora_r),
            "alpha": int(2 * args.lora_r),
            "dropout": LORA_DROPOUT,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "trainable_parameters": int(trainable),
        },
        "objective": "answer_only_tail_logits_causal_cross_entropy",
        "status": "running_no_result",
    }
    _atomic_json_dump(run_protocol, args.output_dir / "run_protocol.json")

    training_kwargs: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "max_steps": phase.steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": phase.effective_batch,
        "learning_rate": phase.learning_rate,
        "warmup_steps": 8,
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
            loss = tail_answer_cross_entropy(
                outputs.logits,
                inputs["input_ids"],
                labels,
            )
            return (loss, outputs) if return_outputs else loss

    trainer = TailAnswerTrainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=TensorAnswerDataset(bundle),
        data_collator=default_data_collator,
        callbacks=callbacks,
    )
    # PEFT forwards accept **kwargs, which can make Trainer assume the model
    # consumed num_items_in_batch. Our custom mean loss does not, so force the
    # standard division by gradient_accumulation_steps in training_step.
    trainer.model_accepts_loss_kwargs = False

    started = time.time()
    trainer.train()
    elapsed = time.time() - started
    inject_inv_freq(model, frequency_end)
    frequency_verification = verify_model_inv_freq(model, frequency_end)
    diagnostic_payload = diagnostics.finalize(model, frequency_start, frequency_end)
    diagnostic_payload.update(
        {
            "frequency_verification": frequency_verification,
            "training_seconds": elapsed,
            "global_step": int(trainer.state.global_step),
            "train_log_history": trainer.state.log_history,
        }
    )
    if int(trainer.state.global_step) != phase.steps:
        raise RuntimeError(f"training ended at step {trainer.state.global_step}, expected {phase.steps}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    frequency_method = "evq_cosh" if args.arm == "evq" else "native_geo"
    _atomic_torch_save(
        {
            "inv_freq": frequency_end.detach().cpu().to(torch.float64),
            "method": frequency_method,
            "head_dim": geometry.head_dim,
            "base": geometry.rope_base,
            "tau": EVQ_TAU if args.arm == "evq" else None,
            "midpoint": True if args.arm == "evq" else False,
            "phase": args.phase,
            "arm": args.arm,
            "path": "smoothstep_log_frequency" if phase.transition else "fixed_endpoint",
        },
        args.output_dir / "custom_inv_freq.pt",
    )
    _atomic_json_dump(diagnostic_payload, args.output_dir / "frequency_diagnostics.json")
    run_protocol["status"] = "complete_uninterpreted"
    run_protocol["artifacts"] = {
        "adapter": _adapter_model_path(args.output_dir).name,
        "frequency": "custom_inv_freq.pt",
        "diagnostics": "frequency_diagnostics.json",
    }
    _atomic_json_dump(run_protocol, args.output_dir / "run_protocol.json")
    print(
        f"completed {args.phase}/{args.arm} in {elapsed / 3600:.2f}h; "
        "capability has not been established until held-out evaluation"
    )


if __name__ == "__main__":
    main()
