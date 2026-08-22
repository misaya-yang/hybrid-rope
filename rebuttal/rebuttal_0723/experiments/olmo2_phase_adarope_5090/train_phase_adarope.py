"""Explicit CUDA trainer for the released OLMo-2 phase/AdaRoPE protocol.

CUDA actions are opt-in and require both authorization factors.  This module
does not create data, select a gate, or download a checkpoint.  Data must be
a manifest-bound view emitted by ``prepare_identifiable_data``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .objectives import ComponentGateConfig, Stage0ObjectiveConfig, document_component_gate, phase_attribution_gate, selected_hidden_lm_head, select_tournament_winner, stage0_answer_margin_loss
from .receipts import METHOD_ID, assert_peft_roundtrip, assert_sidecar_hash, atomic_json, canonical_sha256, no_cuda_preflight, require_dual_authorization, sha256_file, stage_receipt


DATA_MODULE = "rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.prepare_identifiable_data"
ATTENTION_MODULE = "rebuttal.rebuttal_0723.experiments.olmo2_phase_adarope_5090.phase_adarope"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
EXPECTED_CHECKPOINT_SHA256 = "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"
EXPECTED_PARENT_TARGET_SHA256 = "cf03385431df1084508055232853d341a903aa2d8b99fca087bc45d513d34af9"
EXPECTED_DATA_ROOT_SHA256 = "142f34ebd7d05f04c16dc53288d734c12335888e844c84098b1f64165bd3caf4"
REQUIRED_TARGET_KEYS = (
    "phase_chord_olmo_r0_lambda_0p1",
    "matched_exponential_control",
    "moment_matched_same_sign_control",
)


def _dependency_versions(torch: Any | None = None) -> dict[str, str]:
    if torch is None:
        import torch as torch_module

        torch = torch_module
    import accelerate
    import peft
    import transformers

    versions = {
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "peft": str(peft.__version__),
        "accelerate": str(accelerate.__version__),
    }
    if not (
        versions["torch"].startswith("2.8.")
        and versions["transformers"].startswith("5.15.")
        and versions["peft"] == "0.20.0"
        and versions["accelerate"] == "1.14.0"
    ):
        raise RuntimeError(f"runtime version contract drift: {versions}")
    return versions


def _checkpoint_identity(checkpoint: Path, ready_path: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    ready_path = ready_path.resolve()
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("status") != "VERIFIED_READY":
        raise ValueError("checkpoint READY status drift")
    recorded_path = ready.get("model_dir")
    if recorded_path is None or Path(str(recorded_path)).resolve() != checkpoint:
        raise ValueError("checkpoint READY path drift")
    weight = checkpoint / "model.safetensors"
    config_path = checkpoint / "config.json"
    if not weight.is_file() or not config_path.is_file():
        raise FileNotFoundError("released checkpoint weight/config is missing")
    rows = [
        row
        for row in ready.get("repository_files", {}).get("files", [])
        if Path(str(row.get("path", ""))).name == "model.safetensors"
    ]
    if len(rows) != 1 or int(rows[0].get("bytes", -1)) != weight.stat().st_size:
        raise ValueError("checkpoint READY weight receipt drift")
    weight_sha = sha256_file(weight)
    if (
        weight_sha != EXPECTED_CHECKPOINT_SHA256
        or rows[0].get("sha256") != weight_sha
        or ready.get("canonical_identity", {}).get("historical_model_sha256") != weight_sha
    ):
        raise ValueError("released checkpoint SHA drift")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "model_type": "olmo2",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 100352,
        "max_position_embeddings": 4096,
        "rope_theta": 500000,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("released checkpoint config drift")
    return {
        "path": str(checkpoint),
        "weight_sha256": weight_sha,
        "weight_bytes": int(weight.stat().st_size),
        "ready_sha256": sha256_file(ready_path),
        "config_sha256": sha256_file(config_path),
    }


def _target_manifest_identity(path: Path) -> dict[str, Any]:
    from .derive_target_manifest import canonical_content_hash

    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    derivation = payload.get("derivation_receipt")
    if not isinstance(derivation, Mapping) or derivation.get("parent_manifest_sha256") != EXPECTED_PARENT_TARGET_SHA256:
        raise ValueError("derived target parent identity drift")
    if payload.get("canonical_content_sha256") != canonical_content_hash(payload):
        raise ValueError("derived target canonical content hash drift")
    candidates = payload.get("candidates")
    if not isinstance(candidates, Mapping) or any(key not in candidates for key in REQUIRED_TARGET_KEYS):
        raise ValueError("derived target candidate set drift")
    hashes: dict[str, str] = {}
    for key in REQUIRED_TARGET_KEYS:
        record = candidates[key]
        values = np.asarray(record.get("inv_freq"), dtype="<f4") if isinstance(record, Mapping) else np.asarray([])
        actual = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        if values.shape != (64,) or not np.isfinite(values).all() or not np.all(values[:-1] > values[1:]) or record.get("inv_freq_float32_sha256") != actual:
            raise ValueError(f"derived target record drift: {key}")
        hashes[key] = actual
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "parent_sha256": EXPECTED_PARENT_TARGET_SHA256,
        "candidate_float32_sha256": hashes,
    }


def _target_name(key: str) -> str:
    if key == "moment_matched_same_sign_control":
        return "moment_matched_control"
    if key in {"matched_exponential_control", "matched_exponential_same_sign_control"}:
        return "matched_exponential"
    return "phase_chord"


def public_interfaces() -> dict[str, Any]:
    data = importlib.import_module(DATA_MODULE)
    attention = importlib.import_module(ATTENTION_MODULE)
    for name in ("load_source_tensor", "build_dataset"):
        if not hasattr(data, name):
            raise RuntimeError(f"public data interface missing: {name}")
    for name in ("PhaseAdaRoPE", "install_phase_adarope", "save_phase_adarope_state", "load_phase_adarope_state"):
        if not hasattr(attention, name):
            raise RuntimeError(f"public attention interface missing: {name}")
    return {"data": data, "attention": attention}


def stage_contract(stage: str, *, arm: str | None = None) -> dict[str, Any]:
    if stage == "stage0":
        return {"steps": 300, "training_length": 4096, "micro_batch": 2, "gradient_accumulation": 4, "eos_repair_steps": 32,
                "lora": {"rank": 64, "alpha": 128.0, "targets": list(TARGET_MODULES)},
                "objective": "answer_ce+correct_alternate_margin+eos+raw_native_4k_replay"}
    if stage == "stage1":
        if arm not in {"lora_only_null", "native_scale", "context_stretch_exp_negative", "phase_chord", "moment_matched_same_sign_control"}:
            raise ValueError("Stage1 arm must be one of the four tournament arms")
        return {"steps": 50, "training_length": 16384, "micro_batch": 1, "gradient_accumulation": 4,
                "arm": arm, "training_split": "train", "warmup_steps": 20, "gate_evaluation": "paired_component_gate16k_vs_stage1_null", "raw_native_4k_replay": True, "eos_repair_steps": 0}
    if stage == "stage2":
        if arm not in {"lora_only_null", "native_scale", "context_stretch_exp_negative", "phase_chord", "moment_matched_same_sign_control"}:
            raise ValueError("Stage2 arm must be one of the four tournament arms")
        return {"steps": 250, "training_length": 16384, "micro_batch": 1, "gradient_accumulation": 4, "arm": arm, "raw_native_4k_replay": True, "eos_repair_steps": 32,
                "joint_parameters": ["alpha", "raw_beta", "raw_gamma", "qkvo_lora"] if arm not in {"lora_only_null", "native_scale"} else (["raw_beta", "raw_gamma", "qkvo_lora"] if arm == "native_scale" else ["qkvo_lora"])}
    raise ValueError("unknown stage")


def validate_parent_for_stage(stage: str, parent: Mapping[str, Any] | None, gate: Mapping[str, Any] | None = None, *, warmup: Mapping[str, Any] | None = None, arm: str | None = None, gates: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    if stage == "stage0":
        if parent is not None:
            raise ValueError("Stage0 cannot resume an unbound parent")
        return
    expected_parent = "stage0" if stage == "stage1" else "stage1"
    if not isinstance(parent, Mapping) or parent.get("stage") != expected_parent:
        raise ValueError(f"{stage} requires the corresponding {expected_parent} bundle")
    stage0_gate = (gates or {}).get("stage0", gate)
    if stage == "stage1":
        if not isinstance(stage0_gate, Mapping) or stage0_gate.get("status") != "PASS":
            raise ValueError(f"{stage} requires a passed Stage0 4K gate")
        if float(stage0_gate.get("exact_answer_terminal_eos", -float("inf"))) < 0.90 or float(stage0_gate.get("source_follow_positive_fraction", -float("inf"))) < 0.90 or float(stage0_gate.get("terminal_eos_rate", -float("inf"))) != 1.0:
            raise ValueError("Stage0 gate does not meet exact/EOS/source-follow thresholds")
    if stage == "stage2" and (not isinstance(gate, Mapping) or gate.get("status") != "PASS"):
        raise ValueError("Stage2 requires the passed Stage1 tournament gate")


@dataclass
class PairView:
    root: Path
    correct: np.ndarray
    swapped: np.ndarray
    positions_by_row: np.ndarray
    target_tokens: np.ndarray
    generation_prompt_stops: np.ndarray | None = None

    @classmethod
    def load(cls, root: Path) -> "PairView":
        root = root.resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"data manifest is required: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("data manifest must be an object")
        _verify_child_manifest(root, manifest)
        names = ("input_ids.npy", "labels.npy", "target_positions.npy", "target_token_ids.npy")
        paths = [root / name for name in names]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError("manifest-bound input/labels/target arrays are required")
        pair_ids, labels, positions, target_tokens = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]
        if pair_ids.ndim != 3 or pair_ids.shape[1] != 2 or labels.shape != pair_ids.shape:
            raise ValueError("pair input/label shape contract drift")
        if positions.ndim != 2 or positions.shape[0] != pair_ids.shape[0] or target_tokens.shape != (pair_ids.shape[0], 2, positions.shape[1]):
            raise ValueError("target position/token shape contract drift")
        if np.any(positions < 1) or np.any(positions >= pair_ids.shape[2]):
            raise ValueError("target positions must satisfy p-1 prediction")
        correct, swapped = pair_ids[:, 0], pair_ids[:, 1]
        expected_correct = np.take_along_axis(correct, positions, axis=1)
        expected_swapped = np.take_along_axis(swapped, positions, axis=1)
        if not np.array_equal(expected_correct, target_tokens[:, 0]) or not np.array_equal(expected_swapped, target_tokens[:, 1]):
            raise ValueError("target token arrays do not match pair inputs")
        if not np.array_equal(np.take_along_axis(labels[:, 0], positions, axis=1), target_tokens[:, 0].astype(labels.dtype)) or not np.array_equal(np.take_along_axis(labels[:, 1], positions, axis=1), target_tokens[:, 1].astype(labels.dtype)):
            raise ValueError("labels do not identify target tokens")
        if manifest.get("status") != "OLMO2_PHASE_ADAROPE_IDENTIFIABLE_PAIR_VIEW_V2":
            raise ValueError("unexpected pair-view manifest status")
        shape = manifest.get("shape")
        if shape is not None and tuple(shape) != tuple(pair_ids.shape):
            raise ValueError("manifest pair shape drift")
        prompt_path = root / "generation_prompt_stops.npy"
        prompts = np.load(prompt_path, mmap_mode="r", allow_pickle=False) if prompt_path.is_file() else None
        if prompts is not None:
            if prompts.ndim != 1 or len(prompts) != len(correct):
                raise ValueError("generation prompt-stop contract drift")
            if np.all(prompts == -1):
                if bool(manifest.get("strict_autoregressive_capability")):
                    raise ValueError("strict autoregressive view lacks generation prompt stops")
                prompts = None
            elif np.any(prompts < 1) or np.any(prompts > correct.shape[1]):
                raise ValueError("generation prompt-stop contract drift")
        return cls(root, correct, swapped, positions, target_tokens, prompts)

    @property
    def positions(self) -> np.ndarray:
        return self.positions_by_row


@dataclass
class RawReplayView:
    input_ids: np.ndarray
    labels: np.ndarray

    @classmethod
    def load(cls, root: Path) -> "RawReplayView":
        root = root.resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "OLMO2_PHASE_ADAROPE_CONTINUOUS_CLM_WARMUP_V1":
            raise ValueError("raw replay must be the released continuous 4K Native view")
        _verify_child_manifest(root, manifest)
        inputs = np.load(root / "input_ids.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
        if inputs.ndim != 2 or inputs.shape[1] != 4096 or labels.shape != inputs.shape or np.any(labels[:, 0] != -100):
            raise ValueError("raw replay input/label contract drift")
        if not np.array_equal(labels[:, 1:], inputs[:, 1:].astype(labels.dtype)):
            raise ValueError("raw replay labels are not next-token Native replay")
        return cls(inputs, labels)


@dataclass
class RetentionView:
    input_ids: np.ndarray

    @classmethod
    def load(cls, root: Path) -> "RetentionView":
        root = root.resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "OLMO2_PHASE_ADAROPE_RAW_RETENTION_V1":
            raise ValueError("retention view must be the released held-out raw view")
        _verify_child_manifest(root, manifest)
        inputs = np.load(root / "input_ids.npy", mmap_mode="r", allow_pickle=False)
        if inputs.ndim != 2 or inputs.shape[1] != 4096:
            raise ValueError("retention input shape drift")
        return cls(inputs)


def _manifest_hash(root: Path) -> str:
    return sha256_file(root / "manifest.json")


def _verify_child_manifest(root: Path, manifest: Mapping[str, Any]) -> str:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"manifest has no file receipts: {root}")
    for name, receipt in files.items():
        path = root / str(name)
        if not isinstance(receipt, Mapping) or not path.is_file() or int(receipt.get("bytes", -1)) != path.stat().st_size or sha256_file(path) != str(receipt.get("sha256")):
            raise ValueError(f"manifest-bound file drift: {path}")
    child_sha = sha256_file(root / "manifest.json")
    parent_path = root.parent / "manifest.json"
    if not parent_path.is_file():
        raise FileNotFoundError(f"root data manifest is required: {parent_path}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    name = root.name
    if name in parent.get("pair_views", {}):
        expected = parent["pair_views"][name].get("manifest_sha256")
    elif name == "warmup4k_clm":
        expected = parent.get("warmup_view", {}).get("manifest_sha256")
    elif name == "retention4k_raw":
        expected = parent.get("retention_view", {}).get("manifest_sha256")
    else:
        raise ValueError(f"unregistered child data view: {name}")
    if expected != child_sha:
        raise ValueError(f"root/child manifest binding drift: {name}")
    return child_sha


def _data_root_identity(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("V3 data root manifest is required")
    root_sha = sha256_file(manifest_path)
    if root_sha != EXPECTED_DATA_ROOT_SHA256:
        raise ValueError("V3 data root manifest SHA drift")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "OLMO2_PHASE_ADAROPE_DATA_PREPARED_V3":
        raise ValueError("V3 data root status drift")
    expected_views = {
        f"{split}{length}"
        for split in ("train", "train_eos", "component_gate", "final_validation")
        for length in ("4k", "8k", "16k")
    }
    if set(payload.get("pair_views", {})) != expected_views:
        raise ValueError("V3 paired-view set drift")
    child_hashes: dict[str, str] = {}
    for name in sorted(expected_views | {"warmup4k_clm", "retention4k_raw"}):
        child = root / name
        manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
        child_hashes[name] = _verify_child_manifest(child, manifest)
    return {
        "path": str(root),
        "manifest_sha256": root_sha,
        "child_manifest_sha256": child_hashes,
    }


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "train_phase_adarope.py",
        "objectives.py",
        "phase_adarope.py",
        "prepare_identifiable_data.py",
        "derive_target_manifest.py",
        "evaluate_downstream.py",
        "receipts.py",
        "run_5090.sh",
    )
    return {name: sha256_file(root / name) for name in names if (root / name).is_file()}


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def _configure_cuda(torch: Any) -> dict[str, Any]:
    versions = _dependency_versions(torch)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    if architecture not in torch.cuda.get_arch_list():
        raise RuntimeError(f"active architecture {architecture} is unavailable")
    if "TORCHINDUCTOR_CACHE_DIR" not in os.environ or "expandable_segments:True" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        raise RuntimeError("persistent compiler cache and expandable_segments allocator are required")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    return {
        **versions,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "architecture": architecture,
        "attention": "flash_sdpa_only",
        "compile_cache": os.environ["TORCHINDUCTOR_CACHE_DIR"],
    }


def _load_base(checkpoint: Path, torch: Any) -> Any:
    from transformers import AutoModelForCausalLM
    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.train import (
        configure_flash_only_attention,
    )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    configure_flash_only_attention(model)
    return model


def _validate_model_shape(model: Any, length: int) -> dict[str, int]:
    config = _base_model(model).config if hasattr(model, "get_base_model") else model.config
    original_max = int(getattr(config, "max_position_embeddings", 0))
    if original_max not in {4096, int(length)} or (original_max > int(length)):
        raise ValueError("released checkpoint max_position_embeddings drift")
    if (int(getattr(config, "hidden_size", 0)), int(getattr(config, "num_attention_heads", 0)), int(getattr(config, "num_key_value_heads", 0))) != (2048, 16, 16):
        raise ValueError("checkpoint shape must be OLMo-2 hidden2048/MHA16")
    if float(getattr(config, "rope_theta", 500000.0)) != 500000.0:
        raise ValueError("released rope_theta drift")
    if int(length) > original_max:
        config.max_position_embeddings = int(length)
    return {"original_max_position_embeddings": original_max, "runtime_max_position_embeddings": int(getattr(config, "max_position_embeddings"))}


def _install_lora(base: Any, *, parent: Path | None = None, trainable: bool = True) -> Any:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    if parent is not None:
        adapter = parent / "adapter"
        if not adapter.is_dir():
            raise FileNotFoundError(f"Stage0 PEFT adapter is required: {adapter}")
        model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    else:
        config = LoraConfig(r=64, lora_alpha=128.0, lora_dropout=0.0, bias="none", target_modules=list(TARGET_MODULES), task_type=TaskType.CAUSAL_LM, inference_mode=not trainable)
        model = get_peft_model(base, config)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable and (not names or any("lora_A" not in name and "lora_B" not in name for name in names)):
        raise RuntimeError("trainable scope escaped QKVO LoRA")
    return model


def _base_model(model: Any) -> Any:
    model = getattr(model, "_orig_mod", model)
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _phase_bank(model: Any) -> Any | None:
    layers = getattr(getattr(_base_model(model), "model", None), "layers", [])
    return getattr(getattr(layers[0], "self_attn", None), "bank", None) if layers else None


def _load_target(manifest_path: Path, key: str, torch: Any) -> Any:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"target manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest.get("assets", manifest.get("frequencies", manifest.get("candidates", manifest)))
    record = assets.get(key) if isinstance(assets, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"target manifest lacks candidate: {key}")
    if record.get("inv_freq") is not None:
        value = np.asarray(record["inv_freq"], dtype="<f4")
        expected = record.get("inv_freq_float32_sha256")
        actual = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if expected is None or actual != str(expected):
            raise ValueError(f"inline target hash drift: {key}")
    else:
        if not record.get("path") or not record.get("sha256"):
            raise ValueError(f"target manifest lacks hash-bound asset: {key}")
        path = (manifest_path.parent / str(record["path"])).resolve()
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"target asset hash drift: {key}")
        value = np.load(path, allow_pickle=False) if path.suffix == ".npy" else torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        value = value.get("inv_freq", value.get("target_inv_freq", value))
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim != 1 or value.numel() != 64 or not torch.isfinite(value).all() or not torch.all(value[:-1] > value[1:]):
        raise ValueError("target inverse-frequency table must be finite, decreasing, and length 64")
    return value


def _install_phase(model: Any, *, target: Any, mode: str, attention: Any, target_name: str = "phase_chord") -> Any:
    """Wrap the base OLMo attention while retaining the existing PEFT modules."""
    base = _base_model(model)
    bank, _ = attention.install_phase_adarope(
        base, target_inv_freq=target, target_name=target_name, mode=mode,
        l_ref=4096, strict=True,
    )
    return bank


def _selected_logits(model: Any, input_ids: Any, target_positions: Any, *, torch: Any, backbone: Any | None = None) -> Any:
    base = _base_model(model)
    call_kwargs = {"input_ids": input_ids, "use_cache": False, "return_dict": True}
    if _phase_bank(model) is not None:
        call_kwargs["phase_context_budget"] = int(input_ids.shape[1])
    outputs = (backbone or base.model)(**call_kwargs)
    hidden = outputs.last_hidden_state
    return selected_hidden_lm_head(hidden, target_positions, _base_model(model).lm_head, input_ids=input_ids, target_tokens=input_ids.gather(1, target_positions))


def _pair_loss(model: Any, view: PairView, indices: np.ndarray, *, variant: int, device: Any, torch: Any, backbone: Any | None = None) -> tuple[Any, dict[str, float]]:
    positions = torch.as_tensor(np.asarray(view.positions_by_row[indices]), device=device, dtype=torch.long)
    inputs = view.correct if variant == 0 else view.swapped
    input_ids = torch.as_tensor(np.asarray(inputs[indices]), device=device, dtype=torch.long)
    tokens = torch.as_tensor(np.asarray(view.target_tokens[indices, variant]), device=device, dtype=torch.long)
    alternate = torch.as_tensor(np.asarray(view.target_tokens[indices, 1 - variant]), device=device, dtype=torch.long)
    logits = _selected_logits(model, input_ids, positions, torch=torch, backbone=backbone)
    eos_id = getattr(getattr(_base_model(model), "config", None), "eos_token_id", None)
    eos_mask = tokens.eq(int(eos_id)) if eos_id is not None else torch.zeros_like(tokens, dtype=torch.bool)
    return stage0_answer_margin_loss(logits=logits, answer_tokens=tokens, alternate_tokens=alternate, input_ids=input_ids, target_positions=positions, eos_mask=eos_mask, config=Stage0ObjectiveConfig())


def _raw_replay_loss(model: Any, view: RawReplayView, indices: np.ndarray, *, device: Any, torch: Any, backbone: Any | None = None) -> Any:
    input_ids = torch.as_tensor(np.asarray(view.input_ids[indices]), device=device, dtype=torch.long)
    positions = torch.arange(1, input_ids.shape[1], 64, device=device, dtype=torch.long).view(1, -1).expand(input_ids.shape[0], -1)
    tokens = input_ids.gather(1, positions)
    logits = _selected_logits(model, input_ids, positions, torch=torch, backbone=backbone)
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tokens.reshape(-1))


def _cache_parity_smoke(model: Any, view: PairView, *, torch: Any) -> dict[str, Any]:
    ids = torch.as_tensor(np.asarray(view.correct[:1]), device="cuda", dtype=torch.long)
    length = int(ids.shape[1])
    was_training = bool(model.training)
    model.eval()
    with torch.inference_mode():
        full = model(input_ids=ids, use_cache=False, return_dict=True, logits_to_keep=1, phase_context_budget=length).logits[:, -1]
        prefix = model(input_ids=ids[:, :-1], use_cache=True, return_dict=True, logits_to_keep=1, phase_context_budget=length)
        decoded = model(input_ids=ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True, return_dict=True, logits_to_keep=1, phase_context_budget=length).logits[:, -1]
    if was_training:
        model.train()
    max_abs = float((full.float() - decoded.float()).abs().max())
    top1_equal = bool(full.argmax(-1).eq(decoded.argmax(-1)).all())
    key_widths = [int(layer.keys.shape[-1]) for layer in prefix.past_key_values.layers]
    value_widths = [int(layer.values.shape[-1]) for layer in prefix.past_key_values.layers]
    close = bool(torch.allclose(full.float(), decoded.float(), rtol=0.02, atol=0.10))
    if not np.isfinite(max_abs) or not close or not top1_equal or any(width != 128 for width in key_widths + value_widths):
        raise RuntimeError(f"cache parity drift: max={max_abs}, close={close}, top1={top1_equal}, key={key_widths}, value={value_widths}")
    return {"length": length, "max_abs_logit_delta": max_abs, "rtol": 0.02, "atol": 0.10, "top1_equal": top1_equal, "key_head_dims": key_widths, "value_head_dims": value_widths, "passed": True}


def _greedy_generate(
    model: Any,
    prompt: Any,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    phase_context_budget: int | None,
    torch: Any,
) -> Any:
    """Flash-only greedy decode without materializing a padding mask."""
    generated = []
    kwargs = {
        "input_ids": prompt,
        "use_cache": True,
        "return_dict": True,
    }
    if phase_context_budget is not None:
        kwargs["phase_context_budget"] = int(phase_context_budget)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**kwargs)
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1].argmax(dim=-1)
        for _ in range(int(max_new_tokens)):
            generated.append(next_token)
            if eos_token_id is not None and bool(torch.all(next_token == int(eos_token_id))):
                break
            decode_kwargs = {
                "input_ids": next_token[:, None],
                "past_key_values": past,
                "use_cache": True,
                "return_dict": True,
            }
            if phase_context_budget is not None:
                decode_kwargs["phase_context_budget"] = int(phase_context_budget)
            outputs = model(**decode_kwargs)
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1].argmax(dim=-1)
    return torch.stack(generated, dim=1) if generated else prompt.new_empty((prompt.shape[0], 0))


def _set_trainable(model: Any, bank: Any | None, *, lora: bool, bank_trainable: bool = True) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(lora and ("lora_A" in name or "lora_B" in name)))
    if bank is not None:
        bank.alpha.requires_grad_(bank_trainable and bank.mode in {"freq_only", "joint"})
        bank.raw_beta.requires_grad_(bank_trainable and bank.mode in {"scale_only", "joint"})
        bank.raw_gamma.requires_grad_(bank_trainable and bank.mode in {"scale_only", "joint"})


def _train(model: Any, view: PairView, *, steps: int, micro_batch: int, accumulation: int, device: Any, torch: Any, smoke: bool, raw_replay: RawReplayView, backbone: Any | None = None, stage_name: str = "stage0", global_start: int = 0, global_total: int | None = None) -> dict[str, Any]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters")
    bank = _phase_bank(model)
    lora_parameters = [parameter for name, parameter in model.named_parameters() if "lora_A" in name or "lora_B" in name]
    frequency_parameters = [bank.alpha] if bank is not None and bank.alpha.requires_grad else []
    temperature_parameters = (
        [parameter for parameter in (bank.raw_beta, bank.raw_gamma) if parameter.requires_grad]
        if bank is not None
        else []
    )
    groups = [{"params": lora_parameters, "lr": 5e-5, "base_lr": 5e-5, "weight_decay": 0.0, "name": "qkvo_lora"}, {"params": frequency_parameters, "lr": 3e-3, "base_lr": 3e-3, "weight_decay": 0.0, "name": "frequency"}, {"params": temperature_parameters, "lr": 3e-3, "base_lr": 3e-3, "weight_decay": 0.0, "name": "temperature"}]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0, fused=True)
    rng = np.random.default_rng(17)
    for _ in range(int(global_start) * accumulation):
        rng.integers(0, len(view.correct), size=micro_batch, endpoint=False)
    real_steps = 1 if smoke else steps
    last: dict[str, float] = {}
    bank_before = {name: parameter.detach().clone() for name, parameter in bank.named_parameters()} if bank is not None else {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    grad_receipt: dict[str, Any] = {}
    step_rows: list[dict[str, Any]] = []
    model.train()
    for step in range(real_steps):
        optimizer.zero_grad(set_to_none=True)
        absolute_step = global_start + step + 1
        total_steps = int(global_total or steps)
        warmup_steps = 20 if stage_name in {"stage0", "stage1"} or (bank is not None and bank.mode == "joint") else 5
        if absolute_step <= warmup_steps:
            lr_scale = float(absolute_step) / float(warmup_steps)
        else:
            progress = float(absolute_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            lr_scale = 0.1 + 0.9 * 0.5 * (1.0 + float(np.cos(np.pi * min(progress, 1.0))))
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_scale
        for _ in range(accumulation):
            indices = rng.integers(0, len(view.correct), size=micro_batch, endpoint=False)
            loss, last = _pair_loss(model, view, indices, variant=(step + _) % 2, device=device, torch=torch, backbone=backbone)
            if raw_replay is not None:
                replay = _raw_replay_loss(model, raw_replay, np.asarray([step % len(raw_replay.input_ids)]), device=device, torch=torch, backbone=backbone)
                loss = loss + 0.25 * replay
                last["raw_native_4k_replay"] = float(replay.detach())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            (loss / accumulation).backward()
        if step == 0:
            active_groups = (("qkvo_lora", lora_parameters), ("frequency", frequency_parameters), ("temperature", temperature_parameters))
            if stage_name == "stage1" and bank is not None and bank.mode != "control":
                active_groups = tuple((name, group) for name, group in active_groups if name != "qkvo_lora")
            for name, group in active_groups:
                grads = [parameter.grad for parameter in group]
                grad_receipt[name] = {"parameters": len(group), "present": bool(grads and all(grad is not None for grad in grads)), "finite": bool(grads and all(torch.isfinite(grad).all() for grad in grads if grad is not None)), "nonzero": bool(grads and any(torch.count_nonzero(grad).item() for grad in grads if grad is not None))}
        if stage_name == "stage1" and bank is not None and bank.mode != "control" and step < 20:
            for parameter in lora_parameters:
                parameter.grad = None
        optimizer.step()
        bank = _phase_bank(model)
        if bank is not None:
            bank.project_parameters()
        step_rows.append({"step": step + 1, "loss": float(last.get("loss", float("nan"))), "lr": [float(group["lr"]) for group in optimizer.param_groups], "finite": bool(np.isfinite(last.get("loss", float("nan"))))})
    elapsed = max(time.perf_counter() - started, 1e-9)
    memory = {"peak_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0, "peak_reserved": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0, "free_bytes": int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else None}
    if memory["free_bytes"] is not None and memory["free_bytes"] < 1 << 30:
        raise RuntimeError("STOP: less than 1 GiB free CUDA headroom")
    if any(not value["present"] or not value["finite"] or not value["nonzero"] for value in grad_receipt.values() if value["parameters"]):
        raise RuntimeError("STOP: missing/non-finite/zero first-step gradient")
    parameter_delta = {name: float((parameter.detach() - bank_before[name]).abs().max()) for name, parameter in bank.named_parameters()} if bank is not None else {}
    if bank is not None and bank.mode == "control" and any(value != 0.0 for value in parameter_delta.values()):
        raise RuntimeError("control AdaRoPE parameters changed")
    return {"steps": real_steps, "metrics": last, "step_rows": step_rows, "trainable_parameters": sum(parameter.numel() for parameter in trainable), "first_step_gradients": grad_receipt, "memory": memory, "tokens_per_second": float(real_steps * accumulation * micro_batch * view.correct.shape[1] / elapsed), "parameter_delta": parameter_delta}


def _save_bundle(model: Any, bank: Any | None, output: Path, *, attention: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    adapter = output / "adapter"
    save_model = getattr(model, "_orig_mod", model)
    save_model.save_pretrained(adapter, safe_serialization=True)
    sidecar = None
    if bank is not None:
        sidecar = output / "phase_adarope_state.pt"
        attention.save_phase_adarope_state(sidecar, bank, metadata=dict(metadata))
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise RuntimeError("PEFT adapter_config.json was not emitted")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert_peft_roundtrip(config, {"r": 64, "lora_alpha": 128.0, "target_modules": list(TARGET_MODULES)})
    sidecar_sha = sha256_file(sidecar) if sidecar else None
    if sidecar is not None:
        assert_sidecar_hash(sidecar, sidecar_sha)
    return {"adapter": "adapter", "adapter_files": {path.name: sha256_file(path) for path in adapter.iterdir() if path.is_file()}, "adapter_config_sha256": sha256_file(config_path), "phase_sidecar": "phase_adarope_state.pt" if sidecar else None, "phase_sidecar_sha256": sidecar_sha}


def _peft_state_hash(model: Any) -> str:
    from peft import get_peft_model_state_dict
    digest = hashlib.sha256()
    for name, value in sorted(get_peft_model_state_dict(getattr(model, "_orig_mod", model)).items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _fresh_reload_check(*, checkpoint: Path, output: Path, model: Any, bank: Any | None, attention: Any, torch: Any) -> dict[str, Any]:
    """Reload the emitted PEFT/sidecar artifacts before writing COMPLETE."""
    fresh_base = _load_base(checkpoint, torch)
    fresh = _install_lora(fresh_base, parent=output, trainable=False)
    if _peft_state_hash(model) != _peft_state_hash(fresh):
        raise RuntimeError("PEFT fresh-reload state hash drift")
    if bank is not None:
        fresh_bank = _install_phase(fresh, target=bank.target_inv_freq.detach().cpu(), mode=bank.mode, attention=attention, target_name=bank.target_name)
        attention.load_phase_adarope_state(output / "phase_adarope_state.pt", fresh_bank, strict=True)
        for name in ("alpha", "raw_beta", "raw_gamma"):
            if not torch.equal(getattr(fresh_bank, name).detach().cpu(), getattr(bank, name).detach().cpu()):
                raise RuntimeError(f"AdaRoPE sidecar fresh-reload state drift: {name}")
    return {"peft_roundtrip": True, "phase_roundtrip": bank is not None}


def _run_eval(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
        configure_ruler_flash_attention,
    )
    if args.output.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output}")
    dependencies = _dependency_versions(torch)
    checkpoint_identity = _checkpoint_identity(args.checkpoint, args.checkpoint_ready)
    target_identity = _target_manifest_identity(args.target_manifest) if args.target_manifest is not None else None
    data_root_sha = sha256_file(args.data.parent / "manifest.json")
    if data_root_sha != EXPECTED_DATA_ROOT_SHA256:
        raise ValueError("V3 data root identity drift")
    interfaces = public_interfaces()
    attention = interfaces["attention"]
    if args.parent is None or args.checkpoint is None or args.data is None or args.output is None or args.checkpoint_ready is None or not args.checkpoint_ready.is_file():
        raise ValueError("eval requires --checkpoint, --checkpoint-ready, --parent, --data, and --output")
    parent_receipt = json.loads((args.parent / "receipt.json").read_text(encoding="utf-8"))
    base = _load_base(args.checkpoint, torch)
    model = _install_lora(base, parent=args.parent, trainable=False)
    eval_length = int(json.loads((args.data / "manifest.json").read_text(encoding="utf-8")).get("length", 4096))
    sidecar = args.parent / "phase_adarope_state.pt"
    if sidecar.is_file():
        if args.target_manifest is None:
            raise ValueError("phase evaluation requires --target-manifest")
        mode = {"native_scale": "scale_only", "context_stretch_exp_negative": "joint", "phase_chord": "joint", "moment_matched_same_sign_control": "joint"}.get(str(parent_receipt.get("arm")))
        if mode is None:
            raise ValueError("phase sidecar arm is not evaluable")
        if parent_receipt.get("target_key") != args.target_key:
            raise ValueError("evaluation target key differs from bundle receipt")
        bank = _install_phase(model, target=_load_target(args.target_manifest, args.target_key, torch), mode=mode, attention=attention, target_name=_target_name(args.target_key))
        attention.load_phase_adarope_state(sidecar, bank, strict=True)
    configure_ruler_flash_attention(model)
    _validate_model_shape(model, eval_length)
    model.eval().to("cuda")
    comparison = None
    if args.comparison is not None:
        comparison = _install_lora(_load_base(args.checkpoint, torch), parent=args.comparison, trainable=False)
        comparison_sidecar = args.comparison / "phase_adarope_state.pt"
        comparison_receipt = json.loads((args.comparison / "receipt.json").read_text(encoding="utf-8"))
        if comparison_receipt.get("stage") == "stage0":
            if args.target_manifest is None:
                raise ValueError("matched numeric-path comparison requires targets")
            comparison_bank = _install_phase(comparison, target=_load_target(args.target_manifest, args.target_key, torch), mode="control", attention=attention, target_name=_target_name(args.target_key))
            comparison_bank.requires_grad_(False)
        else:
            if not comparison_sidecar.is_file() or args.target_manifest is None:
                raise FileNotFoundError("comparison bundle lacks phase sidecar/targets")
            comparison_mode = {"lora_only_null": "control", "native_scale": "scale_only", "context_stretch_exp_negative": "joint", "phase_chord": "joint", "moment_matched_same_sign_control": "joint"}.get(str(comparison_receipt.get("arm")))
            comparison_key = str(comparison_receipt.get("target_key", ""))
            if comparison_mode is None or not comparison_key:
                raise ValueError("comparison receipt arm/target is not valid")
            comparison_bank = _install_phase(comparison, target=_load_target(args.target_manifest, comparison_key, torch), mode=comparison_mode, attention=attention, target_name=_target_name(comparison_key))
            interfaces["attention"].load_phase_adarope_state(comparison_sidecar, comparison_bank, strict=True)
            comparison_bank.requires_grad_(False)
        configure_ruler_flash_attention(comparison)
        _validate_model_shape(comparison, eval_length)
        comparison = comparison.to("cuda").eval()
    view = PairView.load(args.data)
    eval_split = str(json.loads((args.data / "manifest.json").read_text(encoding="utf-8")).get("split", "final_validation"))
    if view.generation_prompt_stops is None:
        raise ValueError("natural single-query eval requires generation_prompt_stops.npy")
    limit = min(len(view.correct), int(args.eval_limit))
    exact = 0
    eos_correct = 0
    source_effects: list[float] = []
    rank_hits = 0
    per_document_rows: list[dict[str, Any]] = []
    row_metadata = []
    rows_path = args.data / "rows.jsonl"
    if rows_path.is_file():
        row_metadata = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with torch.no_grad():
        for index in range(limit):
            prompt_stop = int(view.generation_prompt_stops[index])
            expected = np.asarray(view.target_tokens[index, 0], dtype=np.int64)
            prompt = torch.as_tensor(view.correct[index:index + 1, :prompt_stop], device="cuda", dtype=torch.long)
            phase_budget = int(view.correct.shape[1]) if _phase_bank(model) is not None else None
            generated = _greedy_generate(
                model,
                prompt,
                max_new_tokens=len(expected),
                eos_token_id=getattr(_base_model(model).config, "eos_token_id", None),
                phase_context_budget=phase_budget,
                torch=torch,
            )[0].detach().cpu().numpy().astype(np.int64)
            exact += int(np.array_equal(generated, expected))
            eos_correct += int(len(generated) == len(expected) and generated[-1] == expected[-1])
            positions = torch.as_tensor(view.positions_by_row[index:index + 1], device="cuda", dtype=torch.long)
            correct = torch.as_tensor(view.correct[index:index + 1], device="cuda", dtype=torch.long)
            swapped = torch.as_tensor(view.swapped[index:index + 1], device="cuda", dtype=torch.long)
            correct_logits = _selected_logits(model, correct, positions, torch=torch)
            swapped_logits = _selected_logits(model, swapped, positions, torch=torch)
            tokens = torch.as_tensor(view.target_tokens[index:index + 1, 0], device="cuda", dtype=torch.long)
            c = correct_logits.float().log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            s = swapped_logits.float().log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            effect = float((c - s).mean())
            source_effects.append(effect)
            rank_hits += int(effect > 0.0)
            first_rank = float(1 + (correct_logits[:, 0, :] > correct_logits[:, 0, tokens[:, 0].item()].view(-1, 1)).sum().item())
            candidate_metrics = {"nll": float((-c).mean()), "source_effect": effect, "first_token_gold_rank": first_rank}
            row = {"document_id": row_metadata[index].get("document_id", str(index)) if index < len(row_metadata) else str(index), "split": eval_split, "candidate": candidate_metrics}
            if comparison is not None:
                ref_correct_logits = _selected_logits(comparison, correct, positions, torch=torch)
                ref_c = ref_correct_logits.float().log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
                ref_s = _selected_logits(comparison, swapped, positions, torch=torch).float().log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
                ref_rank = float(1 + (ref_correct_logits[:, 0, :] > ref_correct_logits[:, 0, tokens[:, 0].item()].view(-1, 1)).sum().item())
                row["parent"] = {"nll": float((-ref_c).mean()), "source_effect": float((ref_c - ref_s).mean()), "first_token_gold_rank": ref_rank}
            per_document_rows.append(row)
    result = {"status": "COMPLETE", "dependencies": dependencies, "checkpoint": checkpoint_identity, "parent_receipt_sha256": sha256_file(args.parent / "receipt.json"), "comparison_receipt_sha256": sha256_file(args.comparison / "receipt.json") if args.comparison else None, "comparison_arm": json.loads((args.comparison / "receipt.json").read_text(encoding="utf-8")).get("arm") if args.comparison else None, "data_root_manifest_sha256": data_root_sha, "data_manifest_sha256": _manifest_hash(args.data), "target_manifest_sha256": sha256_file(args.target_manifest) if args.target_manifest else None, "target_identity": target_identity, "gate_sha256": {key: sha256_file(path) for key, path in {"stage0": args.stage0_gate or args.gate, "scale": args.scale_gate, "freq": args.freq_gate}.items() if path is not None}, "examples": limit, "exact_answer_terminal_eos": exact / max(limit, 1), "terminal_eos_rate": eos_correct / max(limit, 1), "source_follow_mean_logprob_delta": float(np.mean(source_effects)) if source_effects else 0.0, "source_follow_positive_fraction": rank_hits / max(limit, 1), "per_document_rows": per_document_rows, "comparison_supplied": comparison is not None, "retention_4k": None}
    if args.retention is not None:
        replay = RetentionView.load(args.retention)
        with torch.no_grad():
            native = _load_base(args.checkpoint, torch).to("cuda").eval()
            candidate_sum = native_sum = count = 0.0
            for start in range(0, len(replay.input_ids), 1):
                ids = torch.as_tensor(np.asarray(replay.input_ids[start:start + 1]), device="cuda", dtype=torch.long)
                positions = torch.arange(1, ids.shape[1], 64, device="cuda", dtype=torch.long).view(1, -1)
                tokens = ids.gather(1, positions)
                candidate_logits = _selected_logits(model, ids, positions, torch=torch)
                native_logits = _selected_logits(native, ids, positions, torch=torch)
                candidate_sum += float(torch.nn.functional.cross_entropy(candidate_logits.float().reshape(-1, candidate_logits.shape[-1]), tokens.reshape(-1), reduction="sum"))
                native_sum += float(torch.nn.functional.cross_entropy(native_logits.float().reshape(-1, native_logits.shape[-1]), tokens.reshape(-1), reduction="sum"))
                count += float(tokens.numel())
            candidate_nll = candidate_sum / max(count, 1.0)
            native_nll = native_sum / max(count, 1.0)
            result["retention_4k"] = {"candidate_nll": candidate_nll, "native_nll": native_nll, "candidate_minus_native": candidate_nll - native_nll, "selected_positions_stride": 64, "manifest_sha256": _manifest_hash(args.retention)}
    atomic_json(args.output, result)
    return result


def _run_stage(args: argparse.Namespace, stage: str, arm: str, *, smoke: bool) -> dict[str, Any]:
    import torch
    final_output = args.output
    work_output = final_output.with_name(final_output.name + ".incomplete")
    if final_output.exists() or work_output.exists():
        raise FileExistsError(f"output directory must not already exist: {final_output}")
    args.output = work_output
    if args.checkpoint_ready is None or not args.checkpoint_ready.is_file():
        raise FileNotFoundError("checkpoint READY receipt is required")
    dependencies = _dependency_versions(torch)
    checkpoint_identity = _checkpoint_identity(args.checkpoint, args.checkpoint_ready)
    target_identity = _target_manifest_identity(args.target_manifest) if args.target_manifest is not None else None
    data_root_sha = sha256_file(args.data.parent / "manifest.json")
    if data_root_sha != EXPECTED_DATA_ROOT_SHA256:
        raise ValueError("V3 data root identity drift")
    if shutil.disk_usage(args.output.parent).free < 5 * (1 << 30):
        raise RuntimeError("STOP: less than 5 GiB free disk space")
    interfaces = public_interfaces()
    attention = interfaces["attention"]
    contract = stage_contract(stage, arm=arm)
    parent_receipt = warmup_receipt = gate = None
    if stage != "stage0":
        if args.parent is None:
            raise ValueError("--parent Stage0 output is required")
        parent_receipt = json.loads((args.parent / "receipt.json").read_text(encoding="utf-8"))
        if args.gate is not None:
            gate = json.loads(args.gate.read_text(encoding="utf-8"))
        if stage == "stage1" and arm == "moment_matched_same_sign_control" and (gate or {}).get("selected_arm") != "phase_chord":
            raise ValueError("moment attribution is allowed only after phase_chord selection")
        if stage == "stage2" and arm not in {"lora_only_null", "moment_matched_same_sign_control"} and (gate or {}).get("selected_arm") != arm:
            raise ValueError(f"Stage2 arm {arm} is not the selected Stage1 winner")
        if stage == "stage2" and arm == "moment_matched_same_sign_control":
            if (gate or {}).get("selected_arm") != "phase_chord" or args.attribution_gate is None:
                raise ValueError("moment Stage2 requires phase winner and independent attribution gate")
            attribution = json.loads(args.attribution_gate.read_text(encoding="utf-8"))
            if attribution.get("status") != "PASS":
                raise ValueError("moment attribution gate did not pass")
        if stage == "stage2" and parent_receipt.get("arm") != arm:
            raise ValueError("Stage2 parent arm does not match continuation arm")
        if stage == "stage2":
            parent_file_sha = sha256_file(args.parent / "receipt.json")
            if arm == "lora_only_null":
                expected_parent_sha = (gate or {}).get("shared_null_receipt_sha256")
            elif arm == "moment_matched_same_sign_control":
                expected_parent_sha = attribution.get("moment_receipt_sha256")
            else:
                selected_rows = [row for row in (gate or {}).get("candidates", []) if row.get("arm") == arm]
                expected_parent_sha = selected_rows[0].get("candidate_receipt_sha256") if len(selected_rows) == 1 else None
            if expected_parent_sha != parent_file_sha:
                raise ValueError("Stage2 parent receipt is not the selected Stage1 bundle")
        gate_paths = {"stage0": args.stage0_gate or args.gate, "scale": args.scale_gate, "freq": args.freq_gate}
        gates = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in gate_paths.items() if path is not None}
        validate_parent_for_stage(stage, parent_receipt, gate, warmup=warmup_receipt, arm=arm, gates=gates)
    runtime = _configure_cuda(torch)
    base = _load_base(args.checkpoint, torch)
    if stage == "stage0":
        model = _install_lora(base, trainable=True)
    else:
        model = _install_lora(base, parent=args.parent, trainable=stage == "stage2")
    bank = None
    target_name = _target_name(args.target_key)
    if arm == "lora_only_null":
        bank = _install_phase(model, target=_load_target(args.target_manifest, args.target_key, torch), mode="control", attention=attention, target_name=_target_name(args.target_key))
        if stage == "stage2":
            attention.load_phase_adarope_state(args.parent / "phase_adarope_state.pt", bank, strict=True)
        _set_trainable(model, bank, lora=True, bank_trainable=False)
    elif arm == "native_scale":
        bank = _install_phase(model, target=_load_target(args.target_manifest, args.target_key, torch), mode="scale_only", attention=attention, target_name=target_name)
        if stage == "stage2":
            attention.load_phase_adarope_state(args.parent / "phase_adarope_state.pt", bank, strict=True)
        _set_trainable(model, bank, lora=True)
    elif arm in {"context_stretch_exp_negative", "phase_chord", "moment_matched_same_sign_control"}:
        bank = _install_phase(model, target=_load_target(args.target_manifest, args.target_key, torch), mode="joint", attention=attention, target_name=target_name)
        if stage == "stage2":
            attention.load_phase_adarope_state(args.parent / "phase_adarope_state.pt", bank, strict=True)
        _set_trainable(model, bank, lora=True)
    else:
        _set_trainable(model, None, lora=True)
    shape_receipt = _validate_model_shape(model, int(contract["training_length"]))
    compiled_backbone = None
    if args.compile:
        compiled_backbone = torch.compile(_base_model(model).model, mode=args.compile_mode, dynamic=False)
    model = model.to("cuda")
    view = PairView.load(args.data)
    cache_parity = _cache_parity_smoke(model, view, torch=torch) if smoke and stage in {"stage1", "stage2"} else None
    data_manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    expected_split = "train"
    if str(data_manifest.get("split")) != expected_split:
        raise ValueError(f"{stage} requires the {expected_split} split, got {data_manifest.get('split')}")
    if view.correct.shape[1] != int(contract["training_length"]):
        raise ValueError(f"{stage} requires length {contract['training_length']}, got {view.correct.shape[1]}")
    started = time.time()
    if args.raw_replay is None:
        raise ValueError("all stages require --raw-replay warmup4k_clm")
    raw_replay = RawReplayView.load(args.raw_replay)
    global_start = 50 if stage == "stage2" else 0
    result = _train(model, view, steps=int(contract["steps"]), micro_batch=int(contract["micro_batch"]), accumulation=int(contract["gradient_accumulation"]), device=torch.device("cuda"), torch=torch, smoke=smoke, raw_replay=raw_replay, backbone=compiled_backbone, stage_name=stage, global_start=global_start, global_total=300 if stage == "stage2" else None)
    eos_steps = 0
    if not smoke and stage in {"stage0", "stage2"}:
        if args.eos_data is None:
            raise ValueError(f"{stage} requires its train_eos length-matched repair view")
        eos_view = PairView.load(args.eos_data)
        eos_manifest = json.loads((args.eos_data / "manifest.json").read_text(encoding="utf-8"))
        if str(eos_manifest.get("split")) not in {"train_eos", "train"} or eos_view.correct.shape[1] != int(contract["training_length"]):
            raise ValueError("EOS repair view split/length drift")
        eos_result = _train(model, eos_view, steps=32, micro_batch=int(contract["micro_batch"]), accumulation=int(contract["gradient_accumulation"]), device=torch.device("cuda"), torch=torch, smoke=False, raw_replay=raw_replay, backbone=compiled_backbone, stage_name="eos_repair", global_start=0, global_total=32)
        eos_steps = int(eos_result["steps"])
        result["eos_repair"] = eos_result
    result["global_start"] = global_start
    result["global_total"] = 300 if stage == "stage2" else int(contract["steps"])
    result["main_steps"] = int(result["steps"])
    result["eos_steps"] = eos_steps
    args.output.mkdir(parents=True, exist_ok=False)
    step_rows = list(result.get("step_rows", []))
    if isinstance(result.get("eos_repair"), Mapping):
        step_rows.extend({**row, "phase": "eos_repair"} for row in result["eos_repair"].get("step_rows", []))
    result["step_jsonl_sha256"] = _atomic_jsonl(args.output / "train_steps.jsonl", step_rows)
    bundle = _save_bundle(model, _phase_bank(model), args.output, attention=attention, metadata={"stage": stage, "arm": arm, "parent_sha256": parent_receipt.get("content_sha256") if parent_receipt else None})
    roundtrip = _fresh_reload_check(checkpoint=args.checkpoint, output=args.output, model=model, bank=_phase_bank(model), attention=attention, torch=torch)
    receipt = stage_receipt(stage=stage, arm=arm, parent_sha256=parent_receipt.get("content_sha256") if parent_receipt else None, contract=contract, status="SMOKE_COMPLETE" if smoke else "COMPLETE")
    receipt.update({"runtime": runtime, "dependencies": dependencies, "checkpoint": checkpoint_identity, "model_shape": shape_receipt, "cache_parity": cache_parity, "checkpoint_ready_sha256": sha256_file(args.checkpoint_ready), "data_root_manifest_sha256": data_root_sha, "data_manifest_sha256": _manifest_hash(args.data), "data_examples": len(view.correct), "target_manifest": str(args.target_manifest) if args.target_manifest else None, "target_manifest_sha256": sha256_file(args.target_manifest) if args.target_manifest else None, "target_identity": target_identity, "target_key": args.target_key if args.target_manifest else None, "gate_sha256": {key: sha256_file(path) for key, path in {"stage0": args.stage0_gate or args.gate, "scale": args.scale_gate, "freq": args.freq_gate}.items() if path is not None}, "code_sha256": _code_hashes(), "train_steps_jsonl_sha256": result["step_jsonl_sha256"], "bundle": bundle, "roundtrip": roundtrip, "elapsed_seconds": time.time() - started, "smoke": smoke})
    receipt["content_sha256"] = canonical_sha256({key: value for key, value in receipt.items() if key != "content_sha256"})
    atomic_json(args.output / "receipt.json", receipt)
    args.output.replace(final_output)
    return receipt


def run_action(args: argparse.Namespace) -> dict[str, Any]:
    if args.action == "preflight":
        if args.checkpoint is None or args.checkpoint_ready is None or args.target_manifest is None or args.data_root is None:
            raise ValueError("preflight requires checkpoint, READY, derived targets, and V3 data root")
        import torch

        cuda_before = bool(torch.cuda.is_initialized())
        dependencies = _dependency_versions(torch)
        checkpoint = _checkpoint_identity(args.checkpoint, args.checkpoint_ready)
        targets = _target_manifest_identity(args.target_manifest)
        data = _data_root_identity(args.data_root)
        if bool(torch.cuda.is_initialized()) != cuda_before or cuda_before:
            raise RuntimeError("CPU preflight initialized CUDA")
        free_bytes = shutil.disk_usage(args.output.parent if args.output is not None else args.data_root).free
        if free_bytes < 5 * (1 << 30):
            raise RuntimeError("preflight requires at least 5 GiB free disk")
        interfaces = public_interfaces()
        return {
            **no_cuda_preflight(),
            "status": "PHASE_ADAROPE_CPU_PREFLIGHT_READY_V1",
            "method_id": METHOD_ID,
            "dependencies": dependencies,
            "checkpoint": checkpoint,
            "targets": targets,
            "data": data,
            "code_sha256": _code_hashes(),
            "free_disk_bytes": int(free_bytes),
            "interfaces": {key: module.__name__ for key, module in interfaces.items()},
        }
    if args.action == "gate":
        payload = json.loads(args.metrics.read_text(encoding="utf-8"))
        rows = payload.get("per_document_rows") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise ValueError("gate input must contain per_document_rows")
        final_manifest = args.final_manifest
        if final_manifest is None or not final_manifest.is_file():
            raise ValueError("gate requires a hash-bound final-validation manifest")
        final_payload = json.loads(final_manifest.read_text(encoding="utf-8"))
        expected_documents = int(final_payload.get("shape", [0])[0])
        if isinstance(payload, Mapping) and not any(isinstance(row, Mapping) and "parent" in row for row in rows):
            thresholds = {"exact_answer_terminal_eos": 0.90, "source_follow_positive_fraction": 0.90}
            status = "PASS" if all(float(payload.get(key, -float("inf"))) >= value for key, value in thresholds.items()) and float(payload.get("terminal_eos_rate", -float("inf"))) == 1.0 else "STOP"
            return {"status": status, "gate_kind": "stage0_gate", "final_manifest_sha256": sha256_file(final_manifest), "final_validation_documents": expected_documents, "metrics_payload_sha256": sha256_file(args.metrics)}
        result = document_component_gate(rows, candidate_key=args.candidate_key, config=ComponentGateConfig(), final_manifest_sha256=sha256_file(final_manifest), final_documents=expected_documents)
        result["gate_kind"] = args.gate_kind
        result["metrics_payload_sha256"] = sha256_file(args.metrics)
        return result
    if args.action == "select-winner":
        payload = json.loads(args.metrics.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("winner selection requires an arm->evaluation mapping")
        result = select_tournament_winner(payload)
        if args.output is not None:
            atomic_json(args.output, result)
        return result
    if args.action == "gate-attribution":
        phase_payload = json.loads(args.metrics.read_text(encoding="utf-8"))
        moment_payload = json.loads(args.comparison_metrics.read_text(encoding="utf-8"))
        result = phase_attribution_gate(phase_payload, moment_payload)
        result["phase_evaluation_sha256"] = sha256_file(args.metrics)
        result["moment_evaluation_sha256"] = sha256_file(args.comparison_metrics)
        if args.output is not None:
            atomic_json(args.output, result)
        return result
    require_dual_authorization(cli_authorize=bool(args.authorize))
    mapping = {"smoke": ("stage0", "native", True), "stage0": ("stage0", "native", False), "smoke-stage1-null": ("stage1", "lora_only_null", True), "smoke-stage1-native-scale": ("stage1", "native_scale", True), "smoke-stage1-exp-negative": ("stage1", "context_stretch_exp_negative", True), "smoke-stage1-phase": ("stage1", "phase_chord", True), "smoke-stage1-moment": ("stage1", "moment_matched_same_sign_control", True), "stage1-null": ("stage1", "lora_only_null", False), "stage1-native-scale": ("stage1", "native_scale", False), "stage1-exp-negative": ("stage1", "context_stretch_exp_negative", False), "stage1-phase": ("stage1", "phase_chord", False), "stage1-moment": ("stage1", "moment_matched_same_sign_control", False), "smoke-stage2-null": ("stage2", "lora_only_null", True), "smoke-stage2-native-scale": ("stage2", "native_scale", True), "smoke-stage2-exp-negative": ("stage2", "context_stretch_exp_negative", True), "smoke-stage2-phase": ("stage2", "phase_chord", True), "smoke-stage2-moment": ("stage2", "moment_matched_same_sign_control", True), "stage2-null": ("stage2", "lora_only_null", False), "stage2-native-scale": ("stage2", "native_scale", False), "stage2-exp-negative": ("stage2", "context_stretch_exp_negative", False), "stage2-phase": ("stage2", "phase_chord", False), "stage2-moment": ("stage2", "moment_matched_same_sign_control", False)}
    if args.action == "eval":
        require_dual_authorization(cli_authorize=bool(args.authorize))
        return _run_eval(args)
    stage, arm, smoke = mapping[args.action]
    return _run_stage(args, stage, arm, smoke=smoke)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "gate", "select-winner", "gate-attribution", "smoke", "stage0", "smoke-stage1-null", "smoke-stage1-native-scale", "smoke-stage1-exp-negative", "smoke-stage1-phase", "smoke-stage1-moment", "stage1-null", "stage1-native-scale", "stage1-exp-negative", "stage1-phase", "stage1-moment", "smoke-stage2-null", "smoke-stage2-native-scale", "smoke-stage2-exp-negative", "smoke-stage2-phase", "smoke-stage2-moment", "stage2-null", "stage2-native-scale", "stage2-exp-negative", "stage2-phase", "stage2-moment", "eval"))
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--comparison-metrics", type=Path)
    parser.add_argument("--final-manifest", type=Path)
    parser.add_argument("--candidate-key", default="candidate")
    parser.add_argument("--gate-kind", default="component_gate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-ready", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--raw-replay", type=Path)
    parser.add_argument("--retention", type=Path)
    parser.add_argument("--eos-data", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--target-key", default="phase_chord_olmo_r0_lambda_0p1")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--freq-warmup", type=Path)
    parser.add_argument("--scale-warmup", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--attribution-gate", type=Path)
    parser.add_argument("--stage0-gate", type=Path)
    parser.add_argument("--scale-gate", type=Path)
    parser.add_argument("--freq-gate", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--eval-limit", type=int, default=128)
    args = parser.parse_args(argv)
    if args.action == "gate" and args.metrics is None:
        parser.error("gate requires --metrics")
    if args.action == "preflight" and any(value is None for value in (args.checkpoint, args.checkpoint_ready, args.target_manifest, args.data_root, args.output)):
        parser.error("preflight requires --checkpoint, --checkpoint-ready, --target-manifest, --data-root, and --output")
    if args.action == "select-winner" and args.metrics is None:
        parser.error("select-winner requires --metrics")
    if args.action == "gate-attribution" and (args.metrics is None or args.comparison_metrics is None or args.output is None):
        parser.error("gate-attribution requires --metrics, --comparison-metrics, and --output")
    cpu_actions = {"preflight", "gate", "select-winner", "gate-attribution"}
    if args.action not in cpu_actions | {"eval"} and any(value is None for value in (args.checkpoint, args.data, args.output)):
        parser.error("CUDA training requires --checkpoint, --data, and --output")
    if args.action not in cpu_actions | {"eval"} and args.checkpoint_ready is None:
        parser.error("CUDA training requires --checkpoint-ready")
    if args.action == "eval" and any(value is None for value in (args.checkpoint, args.parent, args.data, args.output)):
        parser.error("eval requires --checkpoint, --parent, --data, and --output")
    if args.action == "eval" and args.checkpoint_ready is None:
        parser.error("eval requires --checkpoint-ready")
    if args.action == "eval" and args.retention is None:
        parser.error("eval requires the held-out --retention view")
    if args.action in {"smoke-stage1-null", "smoke-stage1-native-scale", "smoke-stage1-exp-negative", "smoke-stage1-phase", "smoke-stage1-moment", "stage1-null", "stage1-native-scale", "stage1-exp-negative", "stage1-phase", "stage1-moment", "smoke-stage2-null", "smoke-stage2-native-scale", "smoke-stage2-exp-negative", "smoke-stage2-phase", "smoke-stage2-moment", "stage2-null", "stage2-native-scale", "stage2-exp-negative", "stage2-phase", "stage2-moment"} and args.target_manifest is None:
        parser.error("this action requires --target-manifest")
    if args.action in {"smoke-stage1-null", "smoke-stage1-native-scale", "smoke-stage1-exp-negative", "smoke-stage1-phase", "smoke-stage1-moment", "stage1-null", "stage1-native-scale", "stage1-exp-negative", "stage1-phase", "stage1-moment"} and args.gate is None and args.stage0_gate is None:
        parser.error("this action requires a passed Stage0 4K gate receipt")
    if args.action in {"smoke-stage2-null", "smoke-stage2-native-scale", "smoke-stage2-exp-negative", "smoke-stage2-phase", "smoke-stage2-moment", "stage2-null", "stage2-native-scale", "stage2-exp-negative", "stage2-phase", "stage2-moment"} and args.gate is None:
        parser.error("Stage2 requires the passed Stage1 tournament gate")
    if args.action in {"smoke-stage2-moment", "stage2-moment"} and args.attribution_gate is None:
        parser.error("Stage2 moment requires --attribution-gate")
    if args.action in {"smoke-stage1-moment", "stage1-moment"} and args.gate is None:
        parser.error("moment attribution requires the phase winner selection receipt")
    if args.action in {"smoke-stage1-moment", "stage1-moment"} and args.stage0_gate is None:
        parser.error("moment attribution also requires the Stage0 gate")
    if args.action not in cpu_actions | {"eval"} and args.raw_replay is None:
        parser.error("all training stages require --raw-replay warmup4k_clm")
    if args.action in {"stage0", "stage2-null", "stage2-native-scale", "stage2-exp-negative", "stage2-phase", "stage2-moment"} and args.eos_data is None:
        parser.error("full Stage0/Stage2 actions require --eos-data; smoke omits the 32-step repair")
    result = run_action(args)
    if args.output is not None and args.action in {"preflight", "gate"}:
        atomic_json(args.output, result)
    elif args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
