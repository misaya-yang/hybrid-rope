#!/usr/bin/env python3
"""Evaluate Geo base, Geo+LoRA, and EVQ+LoRA on one frozen 2026 corpus."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from .eval_positional_distill import causal_backbone
    from .eval_temporal_holdout_matched import (
        _probe_logits,
        _score_arm,
        load_domain_artifacts,
        sha256_file,
    )
    from .prepare_legacy_model_manifest import validate_model_manifest
    from .prepare_positional_distill_data import tokenizer_source_fingerprint
    from .train_evq_lora import (
        build_training_inv_freq,
        configure_packed_free_causal_sdpa,
        inject_inv_freq,
        load_frequency_artifact,
        resolve_model_rope_geometry,
        validate_legacy_model_geometry,
        verify_model_inv_freq,
    )
    from .validate_legacy_lora_artifact import validate_artifact
except ImportError:
    from eval_positional_distill import causal_backbone
    from eval_temporal_holdout_matched import (
        _probe_logits,
        _score_arm,
        load_domain_artifacts,
        sha256_file,
    )
    from prepare_legacy_model_manifest import validate_model_manifest
    from prepare_positional_distill_data import tokenizer_source_fingerprint
    from train_evq_lora import (
        build_training_inv_freq,
        configure_packed_free_causal_sdpa,
        inject_inv_freq,
        load_frequency_artifact,
        resolve_model_rope_geometry,
        validate_legacy_model_geometry,
        verify_model_inv_freq,
    )
    from validate_legacy_lora_artifact import validate_artifact


def temporal_arm_contract(
    *,
    geo_seed: int,
    evq_seed: int,
) -> dict[str, dict[str, Any]]:
    """Return a three-arm contract with explicit Geo and EVQ seed identities."""
    if geo_seed not in (42, 43, 44) or evq_seed not in (42, 43, 44):
        raise ValueError("temporal evaluation seeds must be 42, 43, or 44")
    return {
        "geo_base": {"frequency": "native_geo", "adapter": None},
        "geo_lora": {
            "frequency": "native_geo",
            "adapter": f"geo_longalpaca_s{geo_seed}",
        },
        "evq_lora": {
            "frequency": "evq_cosh",
            "tau": 1.414,
            "adapter": f"evq_longalpaca_tau1414_s{evq_seed}",
        },
    }


def _comparison_record(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    geo_base: str,
    geo_lora: str,
    evq_lora: str,
) -> dict[str, Any]:
    metrics = {
        name: {
            "nll": float(arms[name]["nll"]),
            "ppl": float(arms[name]["ppl"]),
            **(
                {
                    "nll_sum": float(arms[name]["nll_sum"]),
                    "scored_tokens": int(arms[name]["scored_tokens"]),
                }
                if "nll_sum" in arms[name]
                else {}
            ),
        }
        for name in (geo_base, geo_lora, evq_lora)
    }
    return {
        "arms": metrics,
        "delta_nll_geo_lora_minus_geo": metrics[geo_lora]["nll"] - metrics[geo_base]["nll"],
        "delta_nll_evq_lora_minus_geo_lora": metrics[evq_lora]["nll"] - metrics[geo_lora]["nll"],
        "delta_nll_evq_lora_minus_geo": metrics[evq_lora]["nll"] - metrics[geo_base]["nll"],
        "ppl_ratio_geo_lora_over_geo": metrics[geo_lora]["ppl"] / metrics[geo_base]["ppl"],
        "ppl_ratio_evq_lora_over_geo_lora": metrics[evq_lora]["ppl"] / metrics[geo_lora]["ppl"],
    }


def build_three_arm_comparisons(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    geo_base: str,
    geo_lora: str,
    evq_lora: str,
) -> dict[str, Any]:
    required = (geo_base, geo_lora, evq_lora)
    if any(name not in arms for name in required):
        raise ValueError("three-arm results are incomplete")
    domain_names = list(arms[geo_base]["domains"])
    if any(list(arms[name]["domains"]) != domain_names for name in required):
        raise ValueError("three arms do not contain identical ordered domains")

    macro = {}
    for prefix in ("8K", "16K", "32K"):
        macro[prefix] = _comparison_record(
            {name: arms[name]["domain_macro"]["prefixes"][prefix] for name in required},
            geo_base=geo_base,
            geo_lora=geo_lora,
            evq_lora=evq_lora,
        )

    domains: dict[str, Any] = {}
    for domain in domain_names:
        domain_record: dict[str, Any] = {"prefixes": {}, "buckets": {}, "packs": []}
        for section in ("prefixes", "buckets"):
            labels = arms[geo_base]["domains"][domain][section]
            for label in labels:
                domain_record[section][label] = _comparison_record(
                    {
                        name: arms[name]["domains"][domain][section][label]
                        for name in required
                    },
                    geo_base=geo_base,
                    geo_lora=geo_lora,
                    evq_lora=evq_lora,
                )
        base_packs = arms[geo_base]["domains"][domain]["packs"]
        other_packs = [arms[name]["domains"][domain]["packs"] for name in required[1:]]
        if any(len(packs) != len(base_packs) for packs in other_packs):
            raise ValueError(f"pack count mismatch in {domain}")
        for pack_index in range(len(base_packs)):
            if any(
                int(arms[name]["domains"][domain]["packs"][pack_index]["pack_index"])
                != pack_index
                for name in required
            ):
                raise ValueError(f"pack ordering mismatch in {domain}")
            pack_record: dict[str, Any] = {
                "pack_index": pack_index,
                "prefixes": {},
                "buckets": {},
            }
            for section in ("prefixes", "buckets"):
                labels = base_packs[pack_index][section]
                for label in labels:
                    pack_record[section][label] = _comparison_record(
                        {
                            name: arms[name]["domains"][domain]["packs"][pack_index]
                            [section][label]
                            for name in required
                        },
                        geo_base=geo_base,
                        geo_lora=geo_lora,
                        evq_lora=evq_lora,
                    )
            domain_record["packs"].append(pack_record)
        domains[domain] = domain_record
    return {"domain_macro": macro, "domains": domains}


def _json_frequency_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "inv_freq"}


def _flash_only_forward(function, *args, **kwargs):
    """Run one evaluator forward with no silent SDPA backend fallback."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return function(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_manifest", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--training_data_manifest", type=Path, required=True)
    parser.add_argument("--geo_adapter_dir", type=Path, required=True)
    parser.add_argument("--evq_adapter_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lm_head_chunk_tokens", type=int, default=1024)
    parser.add_argument("--max_packs_per_domain", type=int)
    parser.add_argument("--expected_geo_seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--expected_evq_seed", type=int, choices=(42, 43, 44), default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    collection_path = args.dataset_root / "collection_manifest.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if collection.get("schema") != "evq_cosh.temporal_holdout_2026.collection.v1":
        raise ValueError("temporal collection manifest schema mismatch")
    domains = {}
    for domain, record in collection.get("domains", {}).items():
        manifest_path = args.dataset_root / str(record["manifest"])
        if sha256_file(manifest_path) != record.get("manifest_sha256"):
            raise ValueError(f"collection manifest hash mismatch for {domain}")
        manifest, input_ids, score_mask = load_domain_artifacts(manifest_path.parent)
        manifest["_manifest_path"] = str(manifest_path)
        domains[domain] = (manifest, input_ids, score_mask)
    if len(domains) != 3:
        raise ValueError("the registered three-arm evaluation requires exactly three domains")

    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    validate_model_manifest(Path(args.model_name), model_manifest, verify_hashes=False)
    model_manifest_sha256 = sha256_file(args.model_manifest)
    training_manifest_sha256 = sha256_file(args.training_data_manifest)
    geo_metadata = validate_artifact(
        args.geo_adapter_dir,
        expected_method="native_geo",
        expected_seed=args.expected_geo_seed,
        expected_data_manifest_sha256=training_manifest_sha256,
    )
    evq_metadata = validate_artifact(
        args.evq_adapter_dir,
        expected_method="evq_cosh",
        expected_seed=args.expected_evq_seed,
        expected_data_manifest_sha256=training_manifest_sha256,
    )
    for metadata in (geo_metadata, evq_metadata):
        if metadata.get("model_manifest_sha256") != model_manifest_sha256:
            raise ValueError("adapter and evaluator model manifests differ")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.model_name).is_dir(),
    )
    tokenizer_record = tokenizer_source_fingerprint(args.model_name)
    for manifest, _, _ in domains.values():
        recorded = manifest.get("tokenizer", {})
        if (
            recorded.get("identifier") != tokenizer_record.get("identifier")
            or recorded.get("files") != tokenizer_record.get("files")
        ):
            raise ValueError("temporal tensors use a different tokenizer")

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
    geo_frequency, geo_schedule = build_training_inv_freq(
        rope_method="native_geo",
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=1.414,
    )
    evq_frequency, evq_schedule = build_training_inv_freq(
        rope_method="evq_cosh",
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=1.414,
    )
    geo_artifact, geo_artifact_record, geo_provenance = load_frequency_artifact(
        args.geo_adapter_dir / "custom_inv_freq.pt", expected_method="native_geo"
    )
    evq_artifact, evq_artifact_record, evq_provenance = load_frequency_artifact(
        args.evq_adapter_dir / "custom_inv_freq.pt", expected_method="evq_cosh"
    )
    for name, actual, expected in (
        ("geo", geo_artifact, geo_frequency),
        ("evq", evq_artifact, evq_frequency),
    ):
        if not torch.allclose(actual.to(torch.float64), expected.to(torch.float64), rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} adapter frequency differs from its canonical schedule")

    device = torch.device("cuda")
    model.to(device)
    model.eval()
    inject_inv_freq(model, geo_frequency)
    first_ids = next(iter(domains.values()))[1][0:1, :512].to(device=device, dtype=torch.long)
    pristine_logits = _flash_only_forward(_probe_logits, model, first_ids)

    model = PeftModel.from_pretrained(model, args.geo_adapter_dir, adapter_name="geo")
    model.load_adapter(args.evq_adapter_dir, adapter_name="evq")
    model.eval()
    inject_inv_freq(model, geo_frequency)
    with model.disable_adapter():
        disabled_logits = _flash_only_forward(_probe_logits, model, first_ids)
    model.set_adapter("geo")
    geo_logits = _flash_only_forward(_probe_logits, model, first_ids)
    model.set_adapter("evq")
    inject_inv_freq(model, evq_frequency)
    evq_logits = _flash_only_forward(_probe_logits, model, first_ids)
    disabled_max_error = float((disabled_logits - pristine_logits).abs().max())
    geo_mean_change = float((geo_logits - disabled_logits).abs().mean())
    evq_mean_change = float((evq_logits - disabled_logits).abs().mean())
    if disabled_max_error > 5e-3:
        raise RuntimeError(f"disabled-adapter base canary failed: {disabled_max_error}")
    if min(geo_mean_change, evq_mean_change) <= 1e-7:
        raise RuntimeError("an active adapter does not change probe logits")

    backbone = causal_backbone(model)
    lm_head = model.get_output_embeddings()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model.set_adapter("geo")
    inject_inv_freq(model, geo_frequency)
    geo_initial = verify_model_inv_freq(model, geo_frequency)
    geo_base = _flash_only_forward(
        _score_arm,
        arm="geo_base",
        model=model,
        backbone=backbone,
        lm_head=lm_head,
        domains=domains,
        device=device,
        lm_head_chunk_tokens=args.lm_head_chunk_tokens,
        disabled=True,
        max_packs_per_domain=args.max_packs_per_domain,
    )
    geo_lora = _flash_only_forward(
        _score_arm,
        arm="geo_lora",
        model=model,
        backbone=backbone,
        lm_head=lm_head,
        domains=domains,
        device=device,
        lm_head_chunk_tokens=args.lm_head_chunk_tokens,
        disabled=False,
        max_packs_per_domain=args.max_packs_per_domain,
    )
    geo_final = verify_model_inv_freq(model, geo_frequency)

    model.set_adapter("evq")
    inject_inv_freq(model, evq_frequency)
    evq_initial = verify_model_inv_freq(model, evq_frequency)
    evq_lora = _flash_only_forward(
        _score_arm,
        arm="evq_lora",
        model=model,
        backbone=backbone,
        lm_head=lm_head,
        domains=domains,
        device=device,
        lm_head_chunk_tokens=args.lm_head_chunk_tokens,
        disabled=False,
        max_packs_per_domain=args.max_packs_per_domain,
    )
    evq_final = verify_model_inv_freq(model, evq_frequency)
    arms = {record["arm"]: record for record in (geo_base, geo_lora, evq_lora)}
    comparisons = build_three_arm_comparisons(
        arms,
        geo_base="geo_base",
        geo_lora="geo_lora",
        evq_lora="evq_lora",
    )
    output = {
        "schema": "evq_cosh.temporal_holdout_2026.three_arm_eval.v1",
        "arm_contract": temporal_arm_contract(
            geo_seed=args.expected_geo_seed,
            evq_seed=args.expected_evq_seed,
        ),
        "seed_design": {
            "geo_seed": args.expected_geo_seed,
            "evq_seed": args.expected_evq_seed,
            "comparison": (
                "matched_seed"
                if args.expected_geo_seed == args.expected_evq_seed
                else "fixed_geo_reference"
            ),
        },
        "model_manifest_sha256": model_manifest_sha256,
        "training_data_manifest_sha256": training_manifest_sha256,
        "collection_manifest_sha256": sha256_file(collection_path),
        "adapter_sha256": {
            "geo_lora": geo_metadata["adapter_sha256"],
            "evq_lora": evq_metadata["adapter_sha256"],
        },
        "schedules": {"geo": geo_schedule, "evq": evq_schedule},
        "frequency_artifacts": {
            "geo": {"provenance": geo_provenance, "metadata": _json_frequency_metadata(geo_artifact_record)},
            "evq": {"provenance": evq_provenance, "metadata": _json_frequency_metadata(evq_artifact_record)},
        },
        "frequency_verification": {
            "geo_initial": geo_initial,
            "geo_final": geo_final,
            "evq_initial": evq_initial,
            "evq_final": evq_final,
        },
        "canaries": {
            "disabled_base_max_logit_error": disabled_max_error,
            "geo_active_mean_logit_change": geo_mean_change,
            "evq_active_mean_logit_change": evq_mean_change,
        },
        "efficiency_contract": "one 32K backbone forward per arm and pack; 8K/16K are exact causal-prefix aggregates",
        "arms": arms,
        "comparisons": comparisons,
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
        "domain_macro": comparisons["domain_macro"],
        "runtime_seconds": output["runtime"]["seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
