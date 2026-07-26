#!/usr/bin/env python3
"""Matched natural-text NLL evaluation for released Geo and trained EVQ."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.contract import (
    ACTUAL_PARAMETER_COUNT,
    FIRST_GATE_STEPS,
    FIRST_GATE_TOKENS,
    assert_frequency_contract,
    assert_model_config,
    endpoint_geo_inv_freq,
    patch_endpoint_evq,
    sha256_file,
    trainable_parameter_count,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.train import (
    Backbone,
    FLASH_ONLY_ATTENTION_IMPLEMENTATION,
    configure_cuda,
    configure_flash_only_attention,
    validate_full_state_artifacts,
)


LENGTHS = (2_048, 4_096, 8_192, 16_384, 32_768)
POSITION_BUCKETS = (
    ("0-1K", 0, 1_024),
    ("1-2K", 1_024, 2_048),
    ("2-4K", 2_048, 4_096),
    ("4-8K", 4_096, 8_192),
    ("8-16K", 8_192, 16_384),
    ("16-32K", 16_384, 32_768),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_trained_checkpoint(
    checkpoint: Path,
    *,
    data_manifest: Path,
) -> dict[str, Any]:
    expected_frequency_receipt = assert_frequency_contract()
    expected_frequency_receipt["active_schedule"] = "evq"
    expected_frequency_receipt["active_sha256_float32"] = (
        expected_frequency_receipt["evq_sha256_float32"]
    )
    state = validate_full_state_artifacts(
        checkpoint,
        expected_step=FIRST_GATE_STEPS,
        data_manifest=data_manifest,
        schedule="evq",
        frequency_receipt=expected_frequency_receipt,
    )
    state_path = checkpoint / "trainer_state.json"
    model_path = checkpoint / "model.safetensors"
    model_sha = sha256_file(model_path)
    expected_frequency = assert_frequency_contract()["evq_sha256_float32"]
    frequency = state.get("frequency_receipt", {})
    if (
        frequency.get("active_schedule") != "evq"
        or frequency.get("active_sha256_float32") != expected_frequency
    ):
        raise RuntimeError("trained checkpoint EVQ frequency receipt drift")
    run_config = state.get("run_config", {})
    expected_run = {
        "schedule": "evq",
        "stop_step": FIRST_GATE_STEPS,
        "sequence_length": 4096,
        "global_batch_sequences": 512,
        "precision": "amp_bf16",
    }
    for key, expected in expected_run.items():
        if run_config.get(key) != expected:
            raise RuntimeError(f"trained checkpoint run-config drift: {key}")
    output_root = checkpoint.parent
    start_step = int(run_config.get("start_step", -1))
    if start_step not in (0, 500):
        raise RuntimeError(f"trained checkpoint start-step drift: {start_step}")
    completion_paths = (
        output_root / "completed.json",
        output_root / f"phase_{start_step:06d}_001000_completed.json",
    )
    for completion_path in completion_paths:
        completion = json.loads(
            completion_path.read_text(encoding="utf-8")
        )
        if (
            completion.get("status") != "TRAINING_COMPLETE"
            or int(completion.get("start_step", -1)) != start_step
            or int(completion.get("stop_step", -1)) != FIRST_GATE_STEPS
            or int(completion.get("tokens_seen", -1)) != FIRST_GATE_TOKENS
        ):
            raise RuntimeError(
                f"trained checkpoint completion drift: {completion_path}"
            )
    log_path = output_root / "train.jsonl"
    if sha256_file(log_path) != json.loads(
        completion_paths[0].read_text(encoding="utf-8")
    )["log_sha256"]:
        raise RuntimeError("trained checkpoint log SHA-256 drift")
    last_row = json.loads(
        [
            line
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][-1]
    )
    if int(last_row["step"]) != FIRST_GATE_STEPS:
        raise RuntimeError("trained checkpoint log does not end at step 1000")
    return {
        "trainer_state_sha256": sha256_file(state_path),
        "model_sha256": model_sha,
        "completion_sha256": sha256_file(completion_paths[0]),
        "phase_completion_sha256": sha256_file(completion_paths[1]),
        "train_log_sha256": sha256_file(log_path),
    }


def load_checkpoint(
    base_model: Path,
    checkpoint: Path,
    *,
    schedule: str,
    data_manifest: Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any], str]:
    if schedule == "geo":
        if not (checkpoint / "config.json").is_file():
            raise RuntimeError("released Geo checkpoint is missing config.json")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        weight_files = sorted(checkpoint.glob("model-*.safetensors"))
        checkpoint_sha = ":".join(sha256_file(path) for path in weight_files)
    elif schedule == "evq":
        validate_trained_checkpoint(
            checkpoint,
            data_manifest=data_manifest,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model_path = checkpoint / "model.safetensors"
        state = load_file(model_path, device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
            )
        checkpoint_sha = sha256_file(model_path)
    else:
        raise ValueError(schedule)
    assert_model_config(model.config)
    if trainable_parameter_count(model) != ACTUAL_PARAMETER_COUNT:
        raise RuntimeError("evaluation model parameter-count drift")
    configure_flash_only_attention(model)
    frequency = assert_frequency_contract()
    if schedule == "evq":
        frequency = patch_endpoint_evq(model)
        frequency["active_schedule"] = "evq"
    elif schedule == "geo":
        native = model.model.rotary_emb.inv_freq.detach().cpu().to(torch.float32)
        if not torch.equal(native, endpoint_geo_inv_freq()):
            raise RuntimeError("released Geo frequency drift")
        frequency["active_schedule"] = "geo"
    else:
        raise ValueError(schedule)
    model.config.use_cache = False
    model.eval()
    model.to(device)
    return model, frequency, checkpoint_sha


def token_nll(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    *,
    chunk_tokens: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    last = input_ids.shape[1] - 1
    for start in range(0, last, chunk_tokens):
        end = min(last, start + chunk_tokens)
        local_hidden = hidden[:, start:end, :]
        targets = input_ids[:, start + 1 : end + 1]
        logits = F.linear(local_hidden, weight)
        losses = F.cross_entropy(
            logits.float().view(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        )
        rows.append(losses.view(input_ids.shape[0], end - start))
    return torch.cat(rows, dim=1).cpu()


def summarize_row(nll: np.ndarray, max_length: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for length in LENGTHS:
        if length > max_length:
            continue
        values = nll[: length - 1]
        tail = values[-min(1_024, len(values)) :]
        buckets: dict[str, float] = {}
        for name, start, end in POSITION_BUCKETS:
            capped_end = min(end, length)
            if capped_end <= start:
                continue
            # NLL index zero predicts token position one.
            local = values[max(0, start - 1) : max(0, capped_end - 1)]
            if len(local):
                buckets[name] = float(local.mean())
        mean = float(values.mean())
        metrics[str(length)] = {
            "full_nll": mean,
            "full_ppl": math.exp(min(mean, 80.0)),
            "tail_1024_nll": float(tail.mean()),
            "position_bucket_nll": buckets,
        }
    return metrics


def evaluate_anchor_file(
    backbone: torch.nn.Module,
    lm_head_weight: torch.Tensor,
    path: Path,
    *,
    batch_size: int,
    chunk_tokens: int,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    anchors = np.load(path, allow_pickle=False, mmap_mode="r")
    if anchors.dtype != np.uint32 or anchors.ndim != 2:
        raise RuntimeError(f"invalid anchor array: {path}")
    all_nll: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(anchors), batch_size):
        batch = np.asarray(anchors[start : start + batch_size], dtype=np.int64)
        input_ids = torch.from_numpy(batch).to("cuda", non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            hidden = backbone(input_ids)
            losses = token_nll(
                hidden,
                input_ids,
                lm_head_weight,
                chunk_tokens=chunk_tokens,
            )
        losses_numpy = losses.numpy()
        all_nll.append(losses_numpy)
        for local_index, nll in enumerate(losses_numpy):
            rows.append(
                {
                    "row": start + local_index,
                    "metrics": summarize_row(nll, anchors.shape[1]),
                }
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    combined = np.concatenate(all_nll, axis=0)
    return (
        rows,
        combined,
        {
            "rows": len(anchors),
            "length": int(anchors.shape[1]),
            "tokens_per_second": int(anchors.size) / elapsed,
            "elapsed_seconds": elapsed,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--schedule", choices=("geo", "evq"), required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--extra-32k-manifest", type=Path)
    parser.add_argument("--only-extra-32k", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size-4k", type=int, default=2)
    parser.add_argument("--batch-size-16k", type=int, default=1)
    parser.add_argument("--lm-head-chunk-tokens", type=int, default=256)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile-mode", default="max-autotune-no-cudagraphs"
    )
    args = parser.parse_args()

    configure_cuda()
    device = torch.device("cuda", 0)
    model, frequency, checkpoint_sha = load_checkpoint(
        args.base_model.resolve(),
        args.checkpoint.resolve(),
        schedule=args.schedule,
        data_manifest=args.data_manifest.resolve(),
        device=device,
    )
    backbone: torch.nn.Module = Backbone(model.model)
    if args.compile:
        backbone = torch.compile(
            backbone,
            fullgraph=True,
            dynamic=False,
            mode=args.compile_mode,
        )
    eval_manifest_path = args.eval_manifest.resolve()
    eval_root = eval_manifest_path.parent
    manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "EVAL_DATA_VERIFIED":
        raise RuntimeError("evaluation manifest is not verified")
    if args.only_extra_32k and args.extra_32k_manifest is None:
        raise RuntimeError("--only-extra-32k requires --extra-32k-manifest")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {}
    nll_arrays: dict[str, np.ndarray] = {}
    standard_anchors = () if args.only_extra_32k else (
        ("official_validation", args.batch_size_4k),
        ("long_documents", args.batch_size_16k),
    )
    for name, batch_size in standard_anchors:
        anchor = manifest["anchors"][name]
        anchor_path = eval_root / anchor["path"]
        if sha256_file(anchor_path) != anchor["sha256"]:
            raise RuntimeError(f"{name} anchor hash drift")
        rows, nll, runtime = evaluate_anchor_file(
            backbone,
            model.lm_head.weight,
            anchor_path,
            batch_size=batch_size,
            chunk_tokens=args.lm_head_chunk_tokens,
        )
        results[name] = {
            "anchor_sha256": anchor["sha256"],
            "rows": rows,
            "runtime": runtime,
        }
        nll_arrays[name] = nll
    extra_manifest_sha256 = None
    if args.extra_32k_manifest is not None:
        extra_manifest_path = args.extra_32k_manifest.resolve()
        extra = json.loads(
            extra_manifest_path.read_text(encoding="utf-8")
        )
        if extra.get("status") != "EVAL32K_VERIFIED":
            raise RuntimeError("32K evaluation manifest is not verified")
        anchor = extra["anchor"]
        if int(anchor["length"]) != 32_768:
            raise RuntimeError("32K evaluation anchor length drift")
        anchor_path = extra_manifest_path.parent / anchor["path"]
        if sha256_file(anchor_path) != anchor["sha256"]:
            raise RuntimeError("32K evaluation anchor hash drift")
        metadata_path = extra_manifest_path.parent / anchor["metadata_path"]
        if sha256_file(metadata_path) != anchor["metadata_sha256"]:
            raise RuntimeError("32K evaluation metadata hash drift")
        rows, nll, runtime = evaluate_anchor_file(
            backbone,
            model.lm_head.weight,
            anchor_path,
            batch_size=1,
            chunk_tokens=args.lm_head_chunk_tokens,
        )
        name = "long_documents_32k"
        results[name] = {
            "anchor_sha256": anchor["sha256"],
            "rows": rows,
            "runtime": runtime,
        }
        nll_arrays[name] = nll
        extra_manifest_sha256 = sha256_file(extra_manifest_path)
    nll_path = output / "per_token_nll.npz"
    np.savez_compressed(nll_path, **nll_arrays)
    receipt = {
        "status": "EVALUATION_COMPLETE",
        "schedule": args.schedule,
        "checkpoint_sha256": checkpoint_sha,
        "eval_manifest_sha256": sha256_file(eval_manifest_path),
        "extra_32k_manifest_sha256": extra_manifest_sha256,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(
                torch.cuda.get_device_capability(0)
            ),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
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
                else False
            ),
        },
        "frequency": frequency,
        "compile": {
            "enabled": args.compile,
            "mode": args.compile_mode,
        },
        "results": results,
        "per_token_nll": {
            "path": nll_path.name,
            "sha256": sha256_file(nll_path),
        },
    }
    write_json(output / "results.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
