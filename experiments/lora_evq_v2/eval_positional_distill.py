#!/usr/bin/env python3
"""Evaluate the four arms of the clean positional-distillation pilot."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_positional_distill_data import sha256_file
from train_evq_lora import (
    build_training_inv_freq,
    inject_inv_freq,
    load_frequency_artifact,
    public_artifact_identifier,
    public_model_identifier,
    resolve_model_rope_geometry,
    verify_model_inv_freq,
)
from train_positional_distill import (
    FrozenSequenceDataset,
    causal_backbone,
    fingerprint_model_source,
    normalized_bucket_hidden_mse,
    position_bucket_ranges,
    validate_distill_manifest,
)


def result_filename(variant: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", variant):
        raise ValueError("variant must contain only filename-safe characters")
    return f"positional_distill_{variant}.json"


def representation_recovery(
    injection_error: float,
    adapted_error: float,
) -> Optional[float]:
    if injection_error <= 0:
        return None
    return 1.0 - max(0.0, adapted_error) / injection_error


def resolve_candidate_frequency_artifact(
    adapter_dir: Path,
    candidate_method: str,
):
    expected_method = candidate_method
    path = Path(adapter_dir) / "custom_inv_freq.pt"
    return load_frequency_artifact(path, expected_method=expected_method)


def load_candidate(
    model_name: str,
    candidate_method: str,
    tau: float,
    adapter_dir: Optional[Path],
    bf16: bool,
):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        candidate_inv_freq, _, provenance = resolve_candidate_frequency_artifact(
            adapter_dir,
            candidate_method,
        )
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        geometry = resolve_model_rope_geometry(config)
        candidate_inv_freq, schedule_meta = build_training_inv_freq(
            rope_method=candidate_method,
            head_dim=geometry.head_dim,
            base=geometry.rope_base,
            tau=tau,
        )
        provenance = {
            "artifact": None,
            "method": schedule_meta["method"],
            "tau": schedule_meta["tau"],
            "midpoint": schedule_meta["midpoint"],
        }
    inject_inv_freq(model, candidate_inv_freq)
    verification = verify_model_inv_freq(model, candidate_inv_freq)
    provenance["verified_modules"] = verification["verified_count"]
    provenance["max_error"] = verification["max_error"]
    model.eval()
    return model, tokenizer, candidate_inv_freq, provenance


def evaluate_ppl(
    model,
    tokenizer,
    text_path: Path,
    lengths: tuple[int, ...],
    chunks: int,
) -> dict:
    text = Path(text_path).read_text(encoding="utf-8")
    full_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    results = {}
    device = next(model.parameters()).device
    for length in lengths:
        losses = []
        for chunk_index in range(chunks):
            start = chunk_index * length
            if start + length > len(full_ids):
                break
            input_ids = full_ids[start : start + length].unsqueeze(0).to(device)
            with torch.no_grad():
                losses.append(float(model(input_ids, labels=input_ids).loss))
        if not losses:
            raise ValueError(f"evaluation text has insufficient tokens for {length}")
        mean_nll = float(np.mean(losses))
        results[f"{length // 1024}K"] = {
            "nll": round(mean_nll, 6),
            "ppl": round(math.exp(mean_nll), 6),
            "chunks": len(losses),
        }
    return results


def evaluate_hidden_error(
    model,
    validation_tensor: torch.Tensor,
    candidate_inv_freq: torch.Tensor,
    teacher_inv_freq: torch.Tensor,
    batches: int,
) -> dict:
    dataset = FrozenSequenceDataset(validation_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    backbone = causal_backbone(model)
    device = next(model.parameters()).device
    losses = []
    bucket_history = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        inject_inv_freq(model, teacher_inv_freq)
        disable_adapter = getattr(model, "disable_adapter", None)
        adapter_context = disable_adapter() if disable_adapter else contextlib.nullcontext()
        with adapter_context, torch.no_grad():
            teacher_hidden = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        inject_inv_freq(model, candidate_inv_freq)
        with torch.no_grad():
            candidate_hidden = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        loss, bucket_losses = normalized_bucket_hidden_mse(
            candidate_hidden,
            teacher_hidden,
            attention_mask,
            position_bucket_ranges(input_ids.shape[1]),
        )
        losses.append(float(loss.cpu()))
        bucket_history.append(bucket_losses)
    inject_inv_freq(model, candidate_inv_freq)
    if not losses:
        raise ValueError("hidden-error evaluation processed no validation batches")
    return {
        "normalized_mse": round(float(np.mean(losses)), 8),
        "bucket_normalized_mse": [
            round(float(value), 8)
            for value in np.mean(np.asarray(bucket_history), axis=0)
        ],
        "batches": len(losses),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--candidate_method",
        choices=("native_geo", "evq_cosh"),
        required=True,
    )
    parser.add_argument("--tau", type=float, default=1.414)
    parser.add_argument("--adapter_dir", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--wikitext_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ppl_lengths", default="8192,16384,32768")
    parser.add_argument("--ppl_chunks", type=int, default=5)
    parser.add_argument("--hidden_batches", type=int, default=8)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_distill_manifest(manifest, args.data_dir)
    validation_path = args.data_dir / manifest["files"]["validation"]["name"]
    validation_tensor = torch.load(
        validation_path,
        map_location="cpu",
        weights_only=True,
    )

    model, tokenizer, candidate_inv_freq, provenance = load_candidate(
        model_name=args.model_name,
        candidate_method=args.candidate_method,
        tau=args.tau,
        adapter_dir=args.adapter_dir,
        bf16=args.bf16,
    )
    config = model.config
    geometry = resolve_model_rope_geometry(config)
    teacher_inv_freq, _ = build_training_inv_freq(
        rope_method="native_geo",
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=0.0,
    )

    started = time.time()
    ppl = evaluate_ppl(
        model=model,
        tokenizer=tokenizer,
        text_path=args.wikitext_path,
        lengths=tuple(int(item) for item in args.ppl_lengths.split(",")),
        chunks=args.ppl_chunks,
    )
    hidden_error = evaluate_hidden_error(
        model=model,
        validation_tensor=validation_tensor,
        candidate_inv_freq=candidate_inv_freq,
        teacher_inv_freq=teacher_inv_freq,
        batches=args.hidden_batches,
    )
    output = {
        "format_version": 1,
        "variant": args.variant,
        "model": public_model_identifier(args.model_name),
        "base_model_fingerprint": fingerprint_model_source(args.model_name),
        "adapter": (
            public_artifact_identifier(args.adapter_dir)
            if args.adapter_dir is not None
            else None
        ),
        "adapter_sha256": (
            sha256_file(args.adapter_dir / "adapter_model.safetensors")
            if args.adapter_dir is not None
            else None
        ),
        "candidate_method": args.candidate_method,
        "frequency_provenance": provenance,
        "data_manifest_sha256": sha256_file(manifest_path),
        "ppl": ppl,
        "hidden_error": hidden_error,
        "eval_time_min": round((time.time() - started) / 60.0, 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / result_filename(args.variant)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output_path.name}")


if __name__ == "__main__":
    main()
