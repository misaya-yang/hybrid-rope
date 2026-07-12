#!/usr/bin/env python3
"""Matched Base/LoRA PPL evaluation on frozen 2026 temporal holdouts.

Each arm performs one canonical 32K backbone forward per pack. 8K and 16K
metrics are accumulated from the causally identical prefixes, avoiding two
redundant attention passes while preserving an exact nested-content contract.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

try:
    from .eval_positional_distill import causal_backbone
    from .prepare_legacy_model_manifest import validate_model_manifest
    from .prepare_positional_distill_data import tokenizer_source_fingerprint
    from .train_evq_lora import (
        build_training_inv_freq,
        inject_inv_freq,
        load_frequency_artifact,
        resolve_model_rope_geometry,
        validate_legacy_model_geometry,
        verify_model_inv_freq,
    )
    from .train_positional_distill import configure_packed_free_causal_sdpa
    from .validate_legacy_lora_artifact import validate_artifact
except ImportError:
    from eval_positional_distill import causal_backbone
    from prepare_legacy_model_manifest import validate_model_manifest
    from prepare_positional_distill_data import tokenizer_source_fingerprint
    from train_evq_lora import (
        build_training_inv_freq,
        inject_inv_freq,
        load_frequency_artifact,
        resolve_model_rope_geometry,
        validate_legacy_model_geometry,
        verify_model_inv_freq,
    )
    from train_positional_distill import configure_packed_free_causal_sdpa
    from validate_legacy_lora_artifact import validate_artifact


BUCKETS = (
    (0, 4096),
    (4096, 8192),
    (8192, 12288),
    (12288, 16384),
    (16384, 24576),
    (24576, 32768),
)
PROTOCOL_LENGTHS = (8192, 16384, 32768)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bucket_label(start: int, end: int) -> str:
    if start % 1024 == 0 and end % 1024 == 0:
        return f"{start // 1024}-{end // 1024}K"
    return f"{start}-{end}"


def _validate_file_record(root: Path, record: Mapping[str, Any]) -> Path:
    name = str(record.get("name", ""))
    if not name or Path(name).name != name:
        raise ValueError("temporal artifact file name is not a safe basename")
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record.get("size", -1)):
        raise ValueError(f"size mismatch for {name}")
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"hash mismatch for {name}")
    return path


def load_domain_artifacts(root: Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "evq_cosh.temporal_holdout_2026.v1":
        raise ValueError("temporal holdout manifest schema mismatch")
    contract = manifest.get("pack_contract", {})
    shape = contract.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or int(shape[0]) < 1
        or int(shape[1]) != 32768
    ):
        raise ValueError("temporal holdout must contain canonical 32K packs")
    if contract.get("lengths") != list(PROTOCOL_LENGTHS):
        raise ValueError("temporal holdout lengths are not the registered 8K/16K/32K prefixes")
    files = manifest.get("files", {})
    documents_path = _validate_file_record(root, files.get("documents", {}))
    input_path = _validate_file_record(root, files.get("input_ids", {}))
    mask_path = _validate_file_record(root, files.get("score_mask", {}))

    documents = [
        json.loads(line)
        for line in documents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not documents or any(
        not str(document.get("published_at", "")).startswith("2026-")
        for document in documents
    ):
        raise ValueError("temporal document ledger is empty or contains a non-2026 item")
    pack_documents = contract.get("documents_by_pack")
    if not isinstance(pack_documents, list) or len(pack_documents) != int(shape[0]):
        raise ValueError("pack document ledger is incomplete")
    flattened = [str(doc_id) for pack in pack_documents for doc_id in pack]
    if len(flattened) != len(set(flattened)):
        raise ValueError("canonical packs reuse a document")
    if not contract.get("pack_document_sets_disjoint"):
        raise ValueError("manifest does not assert disjoint pack document sets")

    input_ids = torch.load(input_path, map_location="cpu", weights_only=True)
    score_mask = torch.load(mask_path, map_location="cpu", weights_only=True)
    expected_shape = (int(shape[0]), 32768)
    if input_ids.dtype != torch.int32 or tuple(input_ids.shape) != expected_shape:
        raise ValueError("input_ids must be int32 canonical 32K packs")
    if score_mask.dtype != torch.bool or tuple(score_mask.shape) != expected_shape:
        raise ValueError("target_score_mask must be bool and match input_ids")
    starts_by_pack = contract.get("document_start_positions_by_pack")
    if not isinstance(starts_by_pack, list) or len(starts_by_pack) != expected_shape[0]:
        raise ValueError("document start ledger is incomplete")
    for pack_index, starts in enumerate(starts_by_pack):
        if not starts or int(starts[0]) != 0:
            raise ValueError("every canonical pack must begin at a document boundary")
        for position in starts:
            position = int(position)
            if position < 0 or position >= 32768 or bool(score_mask[pack_index, position]):
                raise ValueError("document-first target is not masked")
    return manifest, input_ids, score_mask


def chunked_masked_causal_sums(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    target_score_mask: torch.Tensor,
    lm_head,
    *,
    buckets: Sequence[tuple[int, int]] = BUCKETS,
    chunk_tokens: int,
) -> dict[str, dict[str, float | int]]:
    if hidden_states.shape[:2] != input_ids.shape or input_ids.shape != target_score_mask.shape:
        raise ValueError("hidden states, input IDs, and target mask axes differ")
    if input_ids.shape[0] != 1:
        raise ValueError("registered evaluator scores one independent pack at a time")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    output = {
        _bucket_label(start, end): {"nll_sum": 0.0, "scored_tokens": 0}
        for start, end in buckets
    }
    output["total"] = {"nll_sum": 0.0, "scored_tokens": 0}
    sequence_length = input_ids.shape[1]
    for hidden_start in range(0, sequence_length - 1, chunk_tokens):
        hidden_end = min(hidden_start + chunk_tokens, sequence_length - 1)
        logits = lm_head(hidden_states[:, hidden_start:hidden_end]).float()
        labels = input_ids[:, hidden_start + 1 : hidden_end + 1]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        valid = target_score_mask[:, hidden_start + 1 : hidden_end + 1]
        total_values = losses[valid]
        output["total"]["nll_sum"] += float(total_values.detach().double().sum().cpu())
        output["total"]["scored_tokens"] += int(valid.sum().cpu())
        target_positions = torch.arange(
            hidden_start + 1,
            hidden_end + 1,
            device=valid.device,
        ).unsqueeze(0)
        for bucket_start, bucket_end in buckets:
            bucket_valid = valid & (target_positions >= bucket_start) & (target_positions < bucket_end)
            values = losses[bucket_valid]
            record = output[_bucket_label(bucket_start, bucket_end)]
            record["nll_sum"] += float(values.detach().double().sum().cpu())
            record["scored_tokens"] += int(bucket_valid.sum().cpu())
        del logits, labels, losses, valid, total_values
    if int(output["total"]["scored_tokens"]) <= 0:
        raise ValueError("no scoreable temporal targets")
    return output


def _metric(nll_sum: float, scored_tokens: int) -> dict[str, float | int]:
    if scored_tokens <= 0:
        raise ValueError("metric has no scored tokens")
    nll = float(nll_sum) / int(scored_tokens)
    if not math.isfinite(nll):
        raise ValueError("non-finite NLL")
    return {
        "nll_sum": float(nll_sum),
        "scored_tokens": int(scored_tokens),
        "nll": nll,
        "ppl": math.exp(nll),
    }


def summarize_position_sums(
    sums: Mapping[str, Mapping[str, float | int]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    buckets = {
        label: _metric(float(record["nll_sum"]), int(record["scored_tokens"]))
        for label, record in sums.items()
        if label != "total"
    }
    prefix_labels = {
        "8K": ("0-4K", "4-8K"),
        "16K": ("0-4K", "4-8K", "8-12K", "12-16K"),
        "32K": tuple(_bucket_label(start, end) for start, end in BUCKETS),
    }
    prefixes = {}
    for name, labels in prefix_labels.items():
        prefixes[name] = _metric(
            sum(float(sums[label]["nll_sum"]) for label in labels),
            sum(int(sums[label]["scored_tokens"]) for label in labels),
        )
    return {"buckets": buckets, "prefixes": prefixes}


def _add_sums(
    accumulator: dict[str, dict[str, float | int]],
    values: Mapping[str, Mapping[str, float | int]],
) -> None:
    for label, record in values.items():
        target = accumulator.setdefault(label, {"nll_sum": 0.0, "scored_tokens": 0})
        target["nll_sum"] = float(target["nll_sum"]) + float(record["nll_sum"])
        target["scored_tokens"] = int(target["scored_tokens"]) + int(record["scored_tokens"])


def _probe_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(input_ids=input_ids, use_cache=False, return_dict=True).logits[:, -1].float().cpu()


def _score_arm(
    *,
    arm: str,
    model,
    backbone,
    lm_head,
    domains: Mapping[str, tuple[dict[str, Any], torch.Tensor, torch.Tensor]],
    device: torch.device,
    lm_head_chunk_tokens: int,
    disabled: bool,
    max_packs_per_domain: int | None,
) -> dict[str, Any]:
    context = model.disable_adapter() if disabled else contextlib.nullcontext()
    domain_outputs: dict[str, Any] = {}
    macro_inputs: list[dict[str, Any]] = []
    with context:
        for domain, (manifest, ids, mask) in domains.items():
            pack_count = ids.shape[0]
            if max_packs_per_domain is not None:
                pack_count = min(pack_count, max_packs_per_domain)
            aggregate: dict[str, dict[str, float | int]] = {}
            pack_outputs = []
            for pack_index in range(pack_count):
                input_ids = ids[pack_index : pack_index + 1].to(device=device, dtype=torch.long)
                score_mask = mask[pack_index : pack_index + 1].to(device=device)
                with torch.inference_mode():
                    hidden = backbone(
                        input_ids=input_ids,
                        attention_mask=None,
                        use_cache=False,
                        return_dict=True,
                    ).last_hidden_state
                    sums = chunked_masked_causal_sums(
                        hidden,
                        input_ids,
                        score_mask,
                        lm_head,
                        chunk_tokens=lm_head_chunk_tokens,
                    )
                _add_sums(aggregate, sums)
                pack_outputs.append({"pack_index": pack_index, **summarize_position_sums(sums)})
                del input_ids, score_mask, hidden
            summary = summarize_position_sums(aggregate)
            domain_outputs[domain] = {
                "manifest_sha256": sha256_file(Path(manifest["_manifest_path"])),
                "packs": pack_outputs,
                **summary,
            }
            macro_inputs.append(summary)
    # Domain macro gives each domain equal weight and avoids a long legal-text
    # domain dominating by token count.
    macro: dict[str, Any] = {"prefixes": {}, "buckets": {}}
    for section in ("prefixes", "buckets"):
        labels = macro_inputs[0][section]
        for label in labels:
            nll = sum(float(item[section][label]["nll"]) for item in macro_inputs) / len(macro_inputs)
            macro[section][label] = {"nll": nll, "ppl": math.exp(nll), "domains": len(macro_inputs)}
    return {"arm": arm, "domains": domain_outputs, "domain_macro": macro}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_manifest", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--training_data_manifest", type=Path, required=True)
    parser.add_argument("--expected_method", choices=("native_geo", "evq_cosh"), required=True)
    parser.add_argument("--expected_seed", type=int, default=42)
    parser.add_argument("--adapter_label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lm_head_chunk_tokens", type=int, default=1024)
    parser.add_argument("--max_packs_per_domain", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    collection_path = args.dataset_root / "collection_manifest.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if collection.get("schema") != "evq_cosh.temporal_holdout_2026.collection.v1":
        raise ValueError("temporal collection manifest schema mismatch")
    domains: dict[str, tuple[dict[str, Any], torch.Tensor, torch.Tensor]] = {}
    for domain, record in collection.get("domains", {}).items():
        manifest_path = args.dataset_root / str(record["manifest"])
        if sha256_file(manifest_path) != record.get("manifest_sha256"):
            raise ValueError(f"collection hash mismatch for {domain}")
        manifest, ids, mask = load_domain_artifacts(manifest_path.parent)
        manifest["_manifest_path"] = str(manifest_path)
        domains[domain] = (manifest, ids, mask)
    if len(domains) != 3:
        raise ValueError("registered temporal evaluation requires exactly three domains")

    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    validate_model_manifest(Path(args.model_name), model_manifest, verify_hashes=False)
    training_manifest_sha256 = sha256_file(args.training_data_manifest)
    adapter_metadata = validate_artifact(
        args.adapter_dir,
        expected_method=args.expected_method,
        expected_seed=args.expected_seed,
        expected_data_manifest_sha256=training_manifest_sha256,
    )
    if adapter_metadata.get("model_manifest_sha256") != sha256_file(args.model_manifest):
        raise ValueError("adapter and evaluator model manifests differ")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.model_name).is_dir(),
    )
    expected_tokenizer = tokenizer_source_fingerprint(args.model_name)
    for manifest, _, _ in domains.values():
        recorded = manifest.get("tokenizer", {})
        if (
            recorded.get("identifier") != expected_tokenizer.get("identifier")
            or recorded.get("files") != expected_tokenizer.get("files")
        ):
            raise ValueError("temporal tokens do not match the evaluator tokenizer")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=Path(args.model_name).is_dir(),
    )
    model.config.use_cache = False
    configure_packed_free_causal_sdpa(model)
    geometry = resolve_model_rope_geometry(model.config)
    validate_legacy_model_geometry(model.config, geometry)
    expected_inv_freq, schedule = build_training_inv_freq(
        rope_method=args.expected_method,
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=1.414,
    )
    artifact_inv_freq, artifact_record, frequency_provenance = load_frequency_artifact(
        args.adapter_dir / "custom_inv_freq.pt",
        expected_method=args.expected_method,
    )
    if not torch.allclose(
        artifact_inv_freq.to(torch.float64),
        expected_inv_freq.to(torch.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("adapter frequency tensor differs from the canonical schedule")

    device = torch.device("cuda")
    model.to(device)
    inject_inv_freq(model, expected_inv_freq)
    pristine_verification = verify_model_inv_freq(model, expected_inv_freq)
    first_ids = next(iter(domains.values()))[1][0:1, :512].to(device=device, dtype=torch.long)
    pristine_logits = _probe_logits(model, first_ids)

    model = PeftModel.from_pretrained(model, args.adapter_dir)
    inject_inv_freq(model, expected_inv_freq)
    wrapped_verification = verify_model_inv_freq(model, expected_inv_freq)
    model.eval()
    with model.disable_adapter():
        disabled_logits = _probe_logits(model, first_ids)
    active_logits = _probe_logits(model, first_ids)
    disabled_max_error = float((disabled_logits - pristine_logits).abs().max())
    active_mean_change = float((active_logits - disabled_logits).abs().mean())
    if disabled_max_error > 5e-3:
        raise RuntimeError(f"disabled-adapter base canary failed: max error {disabled_max_error}")
    if active_mean_change <= 1e-7:
        raise RuntimeError("active adapter does not change probe logits")

    backbone = causal_backbone(model)
    lm_head = model.get_output_embeddings()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    base = _score_arm(
        arm=f"base_{args.expected_method}",
        model=model,
        backbone=backbone,
        lm_head=lm_head,
        domains=domains,
        device=device,
        lm_head_chunk_tokens=args.lm_head_chunk_tokens,
        disabled=True,
        max_packs_per_domain=args.max_packs_per_domain,
    )
    adapted = _score_arm(
        arm=args.adapter_label,
        model=model,
        backbone=backbone,
        lm_head=lm_head,
        domains=domains,
        device=device,
        lm_head_chunk_tokens=args.lm_head_chunk_tokens,
        disabled=False,
        max_packs_per_domain=args.max_packs_per_domain,
    )
    final_verification = verify_model_inv_freq(model, expected_inv_freq)
    output = {
        "schema": "evq_cosh.temporal_holdout_2026.eval.v1",
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "training_data_manifest_sha256": training_manifest_sha256,
        "collection_manifest_sha256": sha256_file(collection_path),
        "adapter_sha256": adapter_metadata["adapter_sha256"],
        "method": args.expected_method,
        "schedule": schedule,
        "frequency_artifact": frequency_provenance,
        "frequency_artifact_metadata": {
            key: value
            for key, value in artifact_record.items()
            if key != "inv_freq"
        },
        "frequency_verification": {
            "pristine": pristine_verification,
            "wrapped": wrapped_verification,
            "final": final_verification,
        },
        "canaries": {
            "disabled_base_max_logit_error": disabled_max_error,
            "active_adapter_mean_logit_change": active_mean_change,
        },
        "efficiency_contract": (
            "one 32K causal backbone forward per arm and pack; 8K/16K are exact prefix aggregates"
        ),
        "arms": {base["arm"]: base, adapted["arm"]: adapted},
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "lm_head_chunk_tokens": args.lm_head_chunk_tokens,
            "max_packs_per_domain": args.max_packs_per_domain,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".incomplete")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "base_macro": base["domain_macro"]["prefixes"],
        "adapted_macro": adapted["domain_macro"]["prefixes"],
        "runtime_seconds": output["runtime"]["seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
