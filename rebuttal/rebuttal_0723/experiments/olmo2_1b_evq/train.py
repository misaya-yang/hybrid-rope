#!/usr/bin/env python3
"""Single-GPU OLMo-2 EVQ training and Blackwell throughput probe."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AttentionInterface,
    AttentionMaskInterface,
    AutoModelForCausalLM,
)

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    ACTUAL_PARAMETER_COUNT,
    FIRST_GATE_STEPS,
    GLOBAL_BATCH_SEQUENCES,
    GLOBAL_BATCH_TOKENS,
    MODEL_CONTRACT,
    OPTIMIZER_CONTRACT,
    SEED,
    SEQUENCE_LENGTH,
    assert_model_config,
    assert_frequency_contract,
    endpoint_geo_inv_freq,
    patch_endpoint_evq,
    sha256_file,
    trainable_parameter_count,
)

PARAMETER_NORM_INTERVAL = 20
LOG_FSYNC_INTERVAL = 20
FLASH_ONLY_ATTENTION_IMPLEMENTATION = "evq_flash_only_sdpa"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class PackedOfficialStream(Dataset[tuple[torch.Tensor, bool]]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        start_instance: int = 0,
        verify_hashes: bool = False,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if self.manifest.get("status") != "STREAM_VERIFIED":
            raise RuntimeError("dataset manifest is not STREAM_VERIFIED")
        contract = self.manifest["contract"]
        if contract["dtype"] != "uint32":
            raise RuntimeError("training stream must be uint32")
        if contract["sequence_length"] != SEQUENCE_LENGTH:
            raise RuntimeError("training stream sequence length drift")
        if contract["seed"] != SEED:
            raise RuntimeError("training stream seed drift")
        stream = self.manifest["training_stream"]
        self.instances = int(stream["instances"])
        self.start_instance = int(start_instance)
        if not 0 <= self.start_instance < self.instances:
            raise ValueError("start_instance is outside the training stream")
        self.token_path = self.root / stream["path"]
        self.valid_path = self.root / stream["instance_valid_path"]
        if self.token_path.stat().st_size != (
            self.instances * SEQUENCE_LENGTH * np.dtype(np.uint32).itemsize
        ):
            raise RuntimeError("training token file size drift")
        if self.valid_path.stat().st_size != self.instances:
            raise RuntimeError("instance-valid file size drift")
        if verify_hashes:
            if sha256_file(self.token_path) != stream["sha256"]:
                raise RuntimeError("training token file hash drift")
            if (
                sha256_file(self.valid_path)
                != stream["instance_valid_sha256"]
            ):
                raise RuntimeError("instance-valid file hash drift")
        self.tokens = np.memmap(
            self.token_path,
            dtype=np.uint32,
            mode="r",
            shape=(self.instances, SEQUENCE_LENGTH),
        )
        self.valid = np.memmap(
            self.valid_path,
            dtype=np.uint8,
            mode="r",
            shape=(self.instances,),
        )

    def __len__(self) -> int:
        return self.instances - self.start_instance

    def __getitem__(self, index: int) -> tuple[torch.Tensor, bool]:
        source_index = self.start_instance + int(index)
        tokens = torch.from_numpy(
            np.asarray(self.tokens[source_index], dtype=np.int64)
        )
        return tokens, bool(self.valid[source_index])


class Backbone(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=False,
        )[0]


def flash_only_causal_mask(
    *,
    attention_mask: torch.Tensor | None = None,
    **_: Any,
) -> None:
    """Use SDPA's causal flag for packed, unpadded training sequences."""
    if attention_mask is not None:
        raise RuntimeError(
            "flash-only OLMo training does not admit padding masks"
        )
    return None


def flash_only_sdpa_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dropout: float = 0.0,
    scaling: float | None = None,
    **_: Any,
) -> tuple[torch.Tensor, None]:
    """Fail closed unless native Flash SDPA can run causal full attention."""
    del module
    if attention_mask is not None:
        raise RuntimeError(
            "flash-only OLMo attention received a materialized mask"
        )
    if query.shape[-2] != key.shape[-2]:
        raise RuntimeError("flash-only OLMo attention does not admit KV cache")
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=float(dropout),
        scale=scaling,
        is_causal=query.shape[-2] > 1,
    )
    return output.transpose(1, 2).contiguous(), None


def configure_flash_only_attention(model: nn.Module) -> None:
    AttentionInterface.register(
        FLASH_ONLY_ATTENTION_IMPLEMENTATION,
        flash_only_sdpa_forward,
    )
    AttentionMaskInterface.register(
        FLASH_ONLY_ATTENTION_IMPLEMENTATION,
        flash_only_causal_mask,
    )
    model.config._attn_implementation = FLASH_ONLY_ATTENTION_IMPLEMENTATION


class LanguageModelLoss(nn.Module):
    def __init__(self, backend: str) -> None:
        super().__init__()
        self.backend = backend
        self.z_scale = float(OPTIMIZER_CONTRACT["z_loss_multiplier"])
        self.liger: nn.Module | None = None
        if backend == "liger":
            from liger_kernel.transformers import (
                LigerFusedLinearCrossEntropyLoss,
            )

            self.liger = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                lse_square_scale=self.z_scale,
                reduction="sum",
                return_z_loss=True,
                accum_dtype=torch.float32,
            )
        elif backend != "native":
            raise ValueError(f"unknown loss backend {backend!r}")

    def forward(
        self,
        weight: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_hidden = hidden_states[:, :-1, :].contiguous().view(
            -1, hidden_states.shape[-1]
        )
        flat_labels = labels.contiguous().view(-1)
        if self.backend == "liger":
            assert self.liger is not None
            output = self.liger(weight, flat_hidden, flat_labels)
            # Liger's primary loss already includes lse_square_scale * logZ^2
            # and its backward already contains the z-loss gradient.  The
            # separate z_loss output is logging-only, so adding it again would
            # double-count it in the reported scalar.
            z_sum = output.z_loss
            if z_sum is None:
                raise RuntimeError("Liger did not return z-loss")
            total = output.loss
            ce_sum = total.detach() - z_sum.detach()
            return total, ce_sum, z_sum.detach()

        logits = F.linear(flat_hidden, weight)
        logits_fp32 = logits.float()
        ce_sum = F.cross_entropy(
            logits_fp32,
            flat_labels,
            ignore_index=-100,
            reduction="sum",
        )
        valid = flat_labels != -100
        log_z = torch.logsumexp(logits_fp32, dim=-1)
        z_sum = self.z_scale * torch.square(log_z[valid]).sum()
        return ce_sum + z_sum, ce_sum.detach(), z_sum.detach()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if OPTIMIZER_CONTRACT["precision"] != "amp_bf16":
        raise RuntimeError("precision contract must remain amp_bf16")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def load_model(
    model_path: Path, device: torch.device, *, schedule: str
) -> tuple[Any, dict]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    assert_model_config(model.config)
    model.config.use_cache = False
    configure_flash_only_attention(model)
    if trainable_parameter_count(model) != ACTUAL_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter count {trainable_parameter_count(model)} "
            f"!= {ACTUAL_PARAMETER_COUNT}"
        )
    non_fp32 = {
        str(parameter.dtype)
        for parameter in model.parameters()
        if parameter.dtype != torch.float32
    }
    if non_fp32:
        raise RuntimeError(f"model master parameters are not FP32: {non_fp32}")
    frequency = assert_frequency_contract()
    if schedule == "evq":
        frequency = patch_endpoint_evq(model)
        frequency["active_schedule"] = "evq"
        frequency["active_sha256_float32"] = frequency[
            "evq_sha256_float32"
        ]
    elif schedule == "geo":
        native = model.model.rotary_emb.inv_freq.detach().cpu().to(torch.float32)
        if not torch.equal(native, endpoint_geo_inv_freq()):
            raise RuntimeError("loaded native Geo frequency drift")
        frequency["active_schedule"] = "geo"
        frequency["active_sha256_float32"] = frequency[
            "geo_sha256_float32"
        ]
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    model.train()
    model.to(device)
    return model, frequency


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    embedding_parameters: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            embedding_parameters.add(id(module.weight))
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in embedding_parameters:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if len(decay) + len(no_decay) != len(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    ):
        raise RuntimeError("optimizer parameter grouping lost parameters")
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": OPTIMIZER_CONTRACT["weight_decay"],
            },
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=OPTIMIZER_CONTRACT["learning_rate"],
        betas=tuple(OPTIMIZER_CONTRACT["betas"]),
        eps=OPTIMIZER_CONTRACT["eps"],
        fused=True,
    )


def learning_rate(step: int) -> float:
    tokens_seen = step * GLOBAL_BATCH_TOKENS
    warmup_tokens = int(OPTIMIZER_CONTRACT["warmup_tokens"])
    initial = float(OPTIMIZER_CONTRACT["learning_rate"])
    if tokens_seen < warmup_tokens:
        return initial * tokens_seen / warmup_tokens
    progress = min(
        1.0,
        (tokens_seen - warmup_tokens)
        / (
            int(OPTIMIZER_CONTRACT["scheduler_max_tokens"])
            - warmup_tokens
        ),
    )
    alpha = float(OPTIMIZER_CONTRACT["scheduler_alpha_f"])
    return initial * (
        alpha + (1.0 - alpha) * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def make_loader(
    manifest: Path,
    *,
    microbatch: int,
    start_step: int,
    workers: int,
    verify_hashes: bool,
) -> DataLoader:
    dataset = PackedOfficialStream(
        manifest,
        start_instance=start_step * GLOBAL_BATCH_SEQUENCES,
        verify_hashes=verify_hashes,
    )
    return DataLoader(
        dataset,
        batch_size=microbatch,
        shuffle=False,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=None if workers == 0 else 4,
        persistent_workers=workers > 0,
    )


def configure_backbone(
    model: nn.Module, *, compile_mode: str, compile_enabled: bool
) -> nn.Module:
    backbone: nn.Module = Backbone(model.model)
    if compile_enabled:
        backbone = torch.compile(
            backbone,
            fullgraph=True,
            dynamic=False,
            mode=compile_mode,
        )
    return backbone


def labels_for(
    input_ids: torch.Tensor, instance_valid: torch.Tensor
) -> torch.Tensor:
    labels = input_ids[:, 1:].clone()
    labels.masked_fill_(~instance_valid[:, None], -100)
    return labels


def raw_batch_hasher() -> Any:
    return hashlib.sha256()


def update_batch_hash(digest: Any, input_ids: torch.Tensor) -> None:
    raw = (
        input_ids.detach()
        .cpu()
        .numpy()
        .astype(np.uint32, copy=False)
        .tobytes()
    )
    digest.update(raw)


def parameter_norm(model: nn.Module) -> torch.Tensor:
    norms = [
        torch.linalg.vector_norm(parameter.detach().float())
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return torch.linalg.vector_norm(torch.stack(norms))


def full_state_checkpoint(
    output_root: Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    frequency_receipt: dict[str, Any],
    data_manifest: Path,
    run_config: dict[str, Any],
) -> Path:
    final = output_root / f"step-{step:06d}-full"
    temporary = output_root / f".step-{step:06d}-full.incomplete"
    if final.exists() or temporary.exists():
        raise FileExistsError(final if final.exists() else temporary)
    temporary.mkdir(parents=True)
    cpu_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(cpu_state, temporary / "model.safetensors")
    del cpu_state
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(
        {
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        temporary / "rng.pt",
    )
    model_sha256 = sha256_file(temporary / "model.safetensors")
    optimizer_sha256 = sha256_file(temporary / "optimizer.pt")
    rng_sha256 = sha256_file(temporary / "rng.pt")
    write_json(
        temporary / "trainer_state.json",
        {
            "step": step,
            "tokens_seen": step * GLOBAL_BATCH_TOKENS,
            "data_manifest_sha256": sha256_file(data_manifest),
            "model_sha256": model_sha256,
            "optimizer_sha256": optimizer_sha256,
            "rng_sha256": rng_sha256,
            "frequency_receipt": frequency_receipt,
            "run_config": run_config,
        },
    )
    temporary.rename(final)
    return final


def validate_full_state_artifacts(
    checkpoint: Path,
    *,
    expected_step: int,
    data_manifest: Path,
    schedule: str,
    frequency_receipt: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state["step"]) != expected_step:
        raise RuntimeError(
            f"resume step {state['step']} != requested {expected_step}"
        )
    if int(state["tokens_seen"]) != expected_step * GLOBAL_BATCH_TOKENS:
        raise RuntimeError("resume token counter drift")
    if state["data_manifest_sha256"] != sha256_file(data_manifest):
        raise RuntimeError("resume data manifest drift")
    prior_runtime = state["run_config"]
    if prior_runtime["schedule"] != schedule:
        raise RuntimeError("resume schedule drift")
    if (
        state["frequency_receipt"]["active_sha256_float32"]
        != frequency_receipt["active_sha256_float32"]
    ):
        raise RuntimeError("resume frequency schedule drift")
    for name, hash_key in (
        ("model.safetensors", "model_sha256"),
        ("optimizer.pt", "optimizer_sha256"),
        ("rng.pt", "rng_sha256"),
    ):
        path = checkpoint / name
        if sha256_file(path) != state[hash_key]:
            raise RuntimeError(f"resume artifact hash drift: {name}")
    return state


def load_full_state_checkpoint(
    checkpoint: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_step: int,
    data_manifest: Path,
    schedule: str,
    frequency_receipt: dict[str, Any],
    expected_runtime: dict[str, Any],
) -> dict[str, Any]:
    state = validate_full_state_artifacts(
        checkpoint,
        expected_step=expected_step,
        data_manifest=data_manifest,
        schedule=schedule,
        frequency_receipt=frequency_receipt,
    )
    prior_runtime = state["run_config"]
    runtime_fields = (
        "gpu",
        "compute_capability",
        "torch_version",
        "cuda_version",
        "precision",
        "attention",
        "compile",
        "loss_backend",
        "sequence_length",
        "global_batch_sequences",
        "microbatch_sequences",
        "gradient_accumulation",
    )
    for field in runtime_fields:
        if prior_runtime.get(field) != expected_runtime.get(field):
            raise RuntimeError(f"resume runtime drift: {field}")

    incompatible = model.load_state_dict(
        load_file(checkpoint / "model.safetensors", device="cpu"),
        strict=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"resume model-state mismatch: {incompatible}")
    optimizer.load_state_dict(
        torch.load(
            checkpoint / "optimizer.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    rng = torch.load(
        checkpoint / "rng.pt",
        map_location="cpu",
        weights_only=False,
    )
    random.setstate(rng["python_rng"])
    np.random.set_state(rng["numpy_rng"])
    torch.set_rng_state(rng["torch_rng"])
    torch.cuda.set_rng_state_all(rng["cuda_rng"])
    return state


def recover_terminal_checkpoint(
    output: Path,
    *,
    start_step: int,
    stop_step: int,
    resume_from: Path | None,
    data_manifest: Path,
    schedule: str,
    frequency_receipt: dict[str, Any],
) -> bool:
    """Finish the narrow crash window after step-1000 state was written."""
    if (
        start_step != 500
        or stop_step != FIRST_GATE_STEPS
        or resume_from is None
    ):
        return False
    final = output / f"step-{stop_step:06d}-full"
    incomplete = output / f".step-{stop_step:06d}-full.incomplete"
    if final.exists() and incomplete.exists():
        raise RuntimeError("both final and incomplete step-1000 checkpoints exist")
    candidate = final if final.exists() else incomplete
    if not candidate.exists():
        return False
    try:
        validate_full_state_artifacts(
            candidate,
            expected_step=stop_step,
            data_manifest=data_manifest,
            schedule=schedule,
            frequency_receipt=frequency_receipt,
        )
    except Exception as error:
        raise RuntimeError(
            "step-1000 checkpoint directory exists but is not recoverable; "
            "stop before replay and inspect the incomplete artifact"
        ) from error
    log_path = output / "train.jsonl"
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [int(row["step"]) for row in rows] != list(
        range(1, stop_step + 1)
    ):
        raise RuntimeError("terminal checkpoint log is not exactly steps 1..1000")
    if candidate == incomplete:
        incomplete.rename(final)
    completion = {
        "status": "TRAINING_COMPLETE",
        "start_step": start_step,
        "stop_step": stop_step,
        "tokens_seen": stop_step * GLOBAL_BATCH_TOKENS,
        "log_sha256": sha256_file(log_path),
        "recovered_after_checkpoint_write": True,
    }
    write_json(
        output
        / f"phase_{start_step:06d}_{stop_step:06d}_completed.json",
        completion,
    )
    write_json(output / "completed.json", completion)
    print(json.dumps({"terminal_recovery": completion}, sort_keys=True))
    return True


def validate_output_for_phase(
    output: Path, *, start_step: int, resume_from: Path | None
) -> None:
    if start_step == 0:
        if resume_from is not None:
            raise ValueError("step-zero phase cannot specify --resume-from")
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        return
    if resume_from is None:
        raise ValueError("nonzero --start-step requires --resume-from")
    if not output.is_dir():
        raise FileNotFoundError(output)
    log_path = output / "train.jsonl"
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [int(row["step"]) for row in rows]
    if not rows or steps[:start_step] != list(range(1, start_step + 1)):
        raise RuntimeError("training log does not contain the complete resume prefix")
    if any(step <= start_step for step in steps[start_step:]):
        raise RuntimeError("training log has duplicated or reordered resume steps")
    completed = json.loads(
        (output / "completed.json").read_text(encoding="utf-8")
    )
    if (
        completed.get("status") != "TRAINING_COMPLETE"
        or int(completed["stop_step"]) != start_step
    ):
        raise RuntimeError("previous phase completion receipt is invalid")
    prefix = "".join(
        json.dumps(row, sort_keys=True) + "\n"
        for row in rows[:start_step]
    )
    if hashlib.sha256(prefix.encode("utf-8")).hexdigest() != completed.get(
        "log_sha256"
    ):
        raise RuntimeError("resume-prefix log does not match phase receipt")
    if len(rows) > start_step:
        # A failed 500->1000 attempt may have appended rows beyond the
        # hash-verified step-500 checkpoint. Preserve that evidence, then
        # deterministically replay from the checkpoint instead of forcing a
        # manual edit on a paid host.
        tail = rows[start_step:]
        encoded_tail = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in tail
        )
        tail_sha = hashlib.sha256(encoded_tail.encode("utf-8")).hexdigest()
        archive = output / (
            f"aborted_tail_after_{start_step:06d}_{tail_sha[:12]}.jsonl"
        )
        if archive.exists():
            if sha256_file(archive) != tail_sha:
                raise RuntimeError("interrupted-tail archive hash drift")
        else:
            archive.write_text(encoded_tail, encoding="utf-8")
        temporary = log_path.with_suffix(".jsonl.recovery.tmp")
        temporary.write_text(prefix, encoding="utf-8")
        temporary.replace(log_path)
        validation_path = output / "validation.jsonl"
        if validation_path.is_file():
            validation_rows = [
                json.loads(line)
                for line in validation_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            retained = [
                row
                for row in validation_rows
                if int(row["step"]) <= start_step
            ]
            discarded = [
                row
                for row in validation_rows
                if int(row["step"]) > start_step
            ]
            if discarded:
                discarded_text = "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in discarded
                )
                discarded_sha = hashlib.sha256(
                    discarded_text.encode("utf-8")
                ).hexdigest()
                validation_archive = output / (
                    "aborted_validation_after_"
                    f"{start_step:06d}_{discarded_sha[:12]}.jsonl"
                )
                if not validation_archive.exists():
                    validation_archive.write_text(
                        discarded_text, encoding="utf-8"
                    )
                retained_text = "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in retained
                )
                validation_tmp = validation_path.with_suffix(
                    ".jsonl.recovery.tmp"
                )
                validation_tmp.write_text(retained_text, encoding="utf-8")
                validation_tmp.replace(validation_path)
def validate_smoke_against_geo(
    *,
    evq_log: Path,
    geo_log: Path,
    output: Path,
    steps: int | None = None,
) -> dict[str, Any]:
    def read_rows(path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    evq = read_rows(evq_log)
    geo = read_rows(geo_log)
    if steps is None:
        steps = len(geo)
    if steps < 18:
        raise RuntimeError(f"Geo sentinel has only {steps} rows; require at least 18")
    if len(evq) < steps or len(geo) != steps:
        raise RuntimeError(
            f"smoke logs require EVQ >= {steps} and Geo == {steps} rows"
        )
    evq = evq[:steps]
    if [row["step"] for row in evq] != list(range(1, steps + 1)):
        raise RuntimeError("EVQ smoke step sequence drift")
    if [row["step"] for row in geo] != list(range(1, steps + 1)):
        raise RuntimeError("Geo sentinel step sequence drift")
    if any(
        left["batch_sha256_uint32"] != right["batch_sha256_uint32"]
        for left, right in zip(evq, geo)
    ):
        raise RuntimeError("Geo/EVQ smoke batches differ")
    relative_ce = [
        abs(left["train_ce_loss"] - right["train_ce_loss"])
        / max(abs(left["train_ce_loss"]), abs(right["train_ce_loss"]), 1e-12)
        for left, right in zip(evq, geo)
    ]
    grad_ratios = [
        left["grad_norm_pre_clip"]
        / max(right["grad_norm_pre_clip"], 1e-12)
        for left, right in zip(evq, geo)
    ]
    max_relative_ce = max(relative_ce)
    median_grad_ratio = float(np.median(grad_ratios))
    if max_relative_ce > 0.25:
        raise RuntimeError(
            f"EVQ smoke CE differs from Geo by {max_relative_ce:.3f}"
        )
    if not 0.2 <= median_grad_ratio <= 5.0:
        raise RuntimeError(
            f"EVQ/Geo median gradient-norm ratio {median_grad_ratio:.3f}"
        )
    receipt = {
        "status": "EVQ_SMOKE_PASS",
        "steps": steps,
        "same_batch_hashes": True,
        "max_relative_ce_difference": max_relative_ce,
        "median_grad_norm_ratio_evq_over_geo": median_grad_ratio,
        "geo_log_sha256": sha256_file(geo_log),
        "evq_log_prefix_sha256": hashlib.sha256(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in evq
            ).encode("utf-8")
        ).hexdigest(),
    }
    write_json(output, receipt)
    return receipt


def load_validation_anchors(
    manifest_path: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "EVAL_DATA_VERIFIED":
        raise RuntimeError("training validation manifest is not verified")
    arrays: list[np.ndarray] = []
    for name in ("official_validation", "long_documents"):
        row = manifest["anchors"][name]
        path = root / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"training validation anchor drift: {name}")
        arrays.append(
            np.load(path, mmap_mode="r", allow_pickle=False)
        )
    return arrays[0], arrays[1], sha256_file(manifest_path)


def validation_nll(
    backbone: nn.Module,
    lm_head_weight: torch.Tensor,
    anchors: np.ndarray,
    *,
    lengths: tuple[int, ...],
    rows: int,
    chunk_tokens: int = 256,
) -> dict[str, float]:
    if rows <= 0 or rows > len(anchors):
        raise ValueError("invalid validation row count")
    results: dict[str, float] = {}
    for length in lengths:
        if length > anchors.shape[1]:
            raise ValueError("validation length exceeds anchor width")
        values = np.asarray(
            anchors[:rows, :length], dtype=np.int64
        )
        input_ids = torch.from_numpy(values).to(
            "cuda", non_blocking=True
        )
        total = torch.zeros((), device="cuda", dtype=torch.float64)
        tokens = 0
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            hidden = backbone(input_ids)
            for start in range(0, length - 1, chunk_tokens):
                end = min(length - 1, start + chunk_tokens)
                logits = F.linear(
                    hidden[:, start:end, :], lm_head_weight
                )
                targets = input_ids[:, start + 1 : end + 1]
                total += F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                ).double()
                tokens += rows * (end - start)
        results[str(length)] = float(total / tokens)
    return results


def run_training_validation(
    *,
    step: int,
    model: nn.Module,
    backbone: nn.Module,
    official: np.ndarray,
    long_documents: np.ndarray,
    microbatch: int,
    run_long: bool,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = {
        "official_validation": validation_nll(
            backbone,
            model.lm_head.weight,
            official,
            lengths=(4_096,),
            rows=min(microbatch, 8),
        )
    }
    if run_long:
        metrics["long_documents"] = validation_nll(
            backbone,
            model.lm_head.weight,
            long_documents,
            lengths=(2_048, 4_096, 8_192, 16_384),
            rows=2,
        )
    torch.cuda.synchronize()
    if was_training:
        model.train()
    return {
        "step": step,
        "tokens_seen": step * GLOBAL_BATCH_TOKENS,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "wall_time_unix": time.time(),
    }


def build_runtime_receipt(
    *,
    args: argparse.Namespace,
    frequency_receipt: dict[str, Any],
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "amp_bf16",
        "attention": {
            "implementation": FLASH_ONLY_ATTENTION_IMPLEMENTATION,
            "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "memory_efficient_enabled": (
                torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "cudnn_enabled": (
                torch.backends.cuda.cudnn_sdp_enabled()
                if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
                else None
            ),
        },
        "compile": {
            "enabled": args.compile,
            "mode": args.compile_mode,
            "cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        },
        "loss_backend": args.loss_backend,
        "schedule": args.schedule,
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_sequences": GLOBAL_BATCH_SEQUENCES,
        "microbatch_sequences": args.microbatch,
        "gradient_accumulation": GLOBAL_BATCH_SEQUENCES // args.microbatch,
        "frequency": frequency_receipt,
    }


def run_probe(args: argparse.Namespace) -> None:
    configure_cuda()
    seed_everything(SEED)
    device = torch.device("cuda", 0)
    model, frequency = load_model(
        args.model_path, device, schedule=args.schedule
    )
    backbone = configure_backbone(
        model, compile_mode=args.compile_mode, compile_enabled=args.compile
    )
    loss_module = LanguageModelLoss(args.loss_backend).to(device)
    optimizer = build_optimizer(model)
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    loader = make_loader(
        args.data_manifest,
        microbatch=args.microbatch,
        start_step=0,
        workers=args.data_workers,
        verify_hashes=False,
    )
    iterator = iter(loader)
    times: list[float] = []
    losses: list[float] = []
    ce_losses: list[float] = []
    z_losses: list[float] = []
    grad_norms: list[float] = []
    total_iterations = args.probe_warmup + args.probe_steps
    torch.cuda.reset_peak_memory_stats()
    for index in range(total_iterations):
        input_ids, valid = next(iterator)
        input_ids = input_ids.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = backbone(input_ids)
            labels = labels_for(input_ids, valid)
            total, ce_sum, z_sum = loss_module(
                model.lm_head.weight, hidden, labels
            )
            normalized = total / float(input_ids.numel())
        if not torch.isfinite(normalized):
            raise RuntimeError("probe loss is not finite")
        normalized.backward()
        if index == 0:
            # Materialize fused AdamW's FP32 state so the probe measures
            # training-resident VRAM.  lr=0 keeps the model unchanged.
            optimizer.step()
        grad_norm = torch.nn.utils.get_total_norm(
            [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if index >= args.probe_warmup:
            times.append(elapsed)
            losses.append(float(normalized.detach()))
            ce_losses.append(float(ce_sum / float(input_ids.numel())))
            z_losses.append(float(z_sum / float(input_ids.numel())))
            grad_norms.append(float(grad_norm))
    receipt = build_runtime_receipt(args=args, frequency_receipt=frequency)
    receipt.update(
        {
            "status": "GPU_PROBE_PASS",
            "warmup_steps": args.probe_warmup,
            "measured_steps": args.probe_steps,
            "mean_step_seconds": sum(times) / len(times),
            "tokens_per_second": (
                args.microbatch * SEQUENCE_LENGTH * len(times) / sum(times)
            ),
            "first_measured_loss": losses[0],
            "last_measured_loss": losses[-1],
            "first_measured_ce_loss": ce_losses[0],
            "last_measured_ce_loss": ce_losses[-1],
            "first_measured_z_loss": z_losses[0],
            "last_measured_z_loss": z_losses[-1],
            "first_measured_grad_norm": grad_norms[0],
            "last_measured_grad_norm": grad_norms[-1],
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "optimizer_state_initialized": True,
        }
    )
    write_json(
        args.output / f"probe_{args.loss_backend}{args.probe_suffix}.json",
        receipt,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


def run_train(args: argparse.Namespace) -> None:
    configure_cuda()
    seed_everything(SEED)
    if GLOBAL_BATCH_SEQUENCES % args.microbatch:
        raise ValueError("microbatch must divide the global batch")
    if not 0 <= args.start_step < args.stop_step:
        raise ValueError("expected 0 <= start_step < stop_step")
    recovery_frequency = assert_frequency_contract()
    recovery_frequency["active_schedule"] = args.schedule
    recovery_frequency["active_sha256_float32"] = recovery_frequency[
        (
            "evq_sha256_float32"
            if args.schedule == "evq"
            else "geo_sha256_float32"
        )
    ]
    if recover_terminal_checkpoint(
        args.output,
        start_step=args.start_step,
        stop_step=args.stop_step,
        resume_from=args.resume_from,
        data_manifest=args.data_manifest,
        schedule=args.schedule,
        frequency_receipt=recovery_frequency,
    ):
        return
    validate_output_for_phase(
        args.output,
        start_step=args.start_step,
        resume_from=args.resume_from,
    )
    device = torch.device("cuda", 0)
    model, frequency = load_model(
        args.model_path, device, schedule=args.schedule
    )
    backbone = configure_backbone(
        model, compile_mode=args.compile_mode, compile_enabled=args.compile
    )
    loss_module = LanguageModelLoss(args.loss_backend).to(device)
    optimizer = build_optimizer(model)
    runtime = build_runtime_receipt(args=args, frequency_receipt=frequency)
    if args.resume_from is not None:
        load_full_state_checkpoint(
            args.resume_from,
            model=model,
            optimizer=optimizer,
            expected_step=args.start_step,
            data_manifest=args.data_manifest,
            schedule=args.schedule,
            frequency_receipt=frequency,
            expected_runtime=runtime,
        )
    loader = make_loader(
        args.data_manifest,
        microbatch=args.microbatch,
        start_step=args.start_step,
        workers=args.data_workers,
        verify_hashes=args.verify_data_hashes,
    )
    iterator: Iterator = iter(loader)
    if args.eval_manifest is None:
        raise ValueError("train mode requires --eval-manifest")
    official_validation, long_validation, eval_manifest_sha = (
        load_validation_anchors(args.eval_manifest)
    )
    accumulation = GLOBAL_BATCH_SEQUENCES // args.microbatch
    run_config = {
        **runtime,
        "start_step": args.start_step,
        "stop_step": args.stop_step,
        "data_manifest": str(args.data_manifest),
        "model_path": str(args.model_path),
        "resume_from": (
            str(args.resume_from) if args.resume_from is not None else None
        ),
        "resume_state_sha256": (
            sha256_file(args.resume_from / "trainer_state.json")
            if args.resume_from is not None
            else None
        ),
        "reference_sentinel_log": (
            str(args.reference_sentinel_log)
            if args.reference_sentinel_log is not None
            else None
        ),
        "eval_manifest": str(args.eval_manifest),
        "eval_manifest_sha256": eval_manifest_sha,
        "validation_interval": args.validation_interval,
        "long_validation_interval": args.long_validation_interval,
        "parameter_norm_interval": PARAMETER_NORM_INTERVAL,
        "log_fsync_interval": LOG_FSYNC_INTERVAL,
    }
    write_json(
        args.output
        / f"resolved_phase_{args.start_step:06d}_{args.stop_step:06d}.json",
        run_config,
    )
    log_path = args.output / "train.jsonl"
    torch.cuda.reset_peak_memory_stats()

    for step in range(args.start_step + 1, args.stop_step + 1):
        optimizer.zero_grad(set_to_none=True)
        ce_accumulator = torch.zeros((), device=device)
        z_accumulator = torch.zeros((), device=device)
        wait_seconds = 0.0
        digest = raw_batch_hasher()
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        for _ in range(accumulation):
            wait_started = time.perf_counter()
            input_ids, valid = next(iterator)
            wait_seconds += time.perf_counter() - wait_started
            update_batch_hash(digest, input_ids)
            input_ids = input_ids.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = backbone(input_ids)
                labels = labels_for(input_ids, valid)
                total, ce_sum, z_sum = loss_module(
                    model.lm_head.weight, hidden, labels
                )
                normalized = total / float(GLOBAL_BATCH_TOKENS)
            if not torch.isfinite(normalized):
                raise RuntimeError(f"step {step}: non-finite loss")
            normalized.backward()
            ce_accumulator += ce_sum / float(GLOBAL_BATCH_TOKENS)
            z_accumulator += z_sum / float(GLOBAL_BATCH_TOKENS)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(OPTIMIZER_CONTRACT["max_grad_norm"])
        )
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        record_parameter_norm = (
            step == 1
            or step % PARAMETER_NORM_INTERVAL == 0
            or step == args.stop_step
        )
        current_parameter_norm = (
            parameter_norm(model) if record_parameter_norm else None
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_started
        row = {
            "step": step,
            "tokens_seen": step * GLOBAL_BATCH_TOKENS,
            "train_ce_loss": float(ce_accumulator),
            "train_z_loss": float(z_accumulator),
            "learning_rate": lr,
            "grad_norm_pre_clip": float(grad_norm),
            "parameter_norm": (
                float(current_parameter_norm)
                if current_parameter_norm is not None
                else None
            ),
            "parameter_norm_interval": PARAMETER_NORM_INTERVAL,
            "tokens_per_second": GLOBAL_BATCH_TOKENS / elapsed,
            "step_seconds": elapsed,
            "dataloader_wait_seconds": wait_seconds,
            "batch_sha256_uint32": digest.hexdigest(),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "finite": True,
            "wall_time_unix": time.time(),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            if (
                step % LOG_FSYNC_INTERVAL == 0
                or step == args.stop_step
            ):
                os.fsync(handle.fileno())
        print(json.dumps(row, sort_keys=True), flush=True)
        if step == 20 and args.reference_sentinel_log is not None:
            validate_smoke_against_geo(
                evq_log=log_path,
                geo_log=args.reference_sentinel_log,
                output=args.output / "evq_smoke_step20.json",
            )
        if step % args.validation_interval == 0:
            validation = run_training_validation(
                step=step,
                model=model,
                backbone=backbone,
                official=official_validation,
                long_documents=long_validation,
                microbatch=args.microbatch,
                run_long=step % args.long_validation_interval == 0,
            )
            with (args.output / "validation.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(validation, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(
                json.dumps(
                    {"validation": validation}, sort_keys=True
                ),
                flush=True,
            )
        if step in args.full_save_steps:
            full_state_checkpoint(
                args.output,
                step=step,
                model=model,
                optimizer=optimizer,
                frequency_receipt=frequency,
                data_manifest=args.data_manifest,
                run_config=run_config,
            )

    completion = {
        "status": "TRAINING_COMPLETE",
        "start_step": args.start_step,
        "stop_step": args.stop_step,
        "tokens_seen": args.stop_step * GLOBAL_BATCH_TOKENS,
        "log_sha256": sha256_file(log_path),
    }
    write_json(
        args.output
        / f"phase_{args.start_step:06d}_{args.stop_step:06d}_completed.json",
        completion,
    )
    write_json(args.output / "completed.json", completion)


def parse_steps(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "train"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--data-workers", type=int, default=8)
    parser.add_argument(
        "--loss-backend", choices=("native", "liger"), default="liger"
    )
    parser.add_argument("--schedule", choices=("geo", "evq"), default="evq")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile-mode", default="max-autotune-no-cudagraphs"
    )
    parser.add_argument("--probe-warmup", type=int, default=5)
    parser.add_argument("--probe-steps", type=int, default=20)
    parser.add_argument("--probe-suffix", default="")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--stop-step", type=int, default=FIRST_GATE_STEPS)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--reference-sentinel-log", type=Path)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument(
        "--long-validation-interval", type=int, default=250
    )
    parser.add_argument(
        "--full-save-steps", type=parse_steps, default={500, 1000}
    )
    parser.add_argument("--verify-data-hashes", action="store_true")
    args = parser.parse_args()
    args.model_path = args.model_path.resolve()
    args.data_manifest = args.data_manifest.resolve()
    args.output = args.output.resolve()
    if args.resume_from is not None:
        args.resume_from = args.resume_from.resolve()
    if args.reference_sentinel_log is not None:
        args.reference_sentinel_log = args.reference_sentinel_log.resolve()
    if args.eval_manifest is not None:
        args.eval_manifest = args.eval_manifest.resolve()
    if args.validation_interval <= 0 or args.long_validation_interval <= 0:
        raise ValueError("validation intervals must be positive")
    if args.mode == "probe":
        args.output.mkdir(parents=True, exist_ok=True)
        run_probe(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
