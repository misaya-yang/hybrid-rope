#!/usr/bin/env python3
"""Matched 8K/16K/32K PPL evaluation for legacy Geo/EVQ LoRA arms."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from .eval_positional_distill import causal_backbone, chunked_causal_nll
    from .legacy_lora_protocol import legacy_eval_filename, sha256_file
    from .prepare_positional_distill_data import tokenizer_source_fingerprint
    from .prepare_legacy_model_manifest import validate_model_manifest
    from .prepare_legacy_wikitext import (
        WIKITEXT_CONFIG,
        WIKITEXT_RAW_SHA256,
        WIKITEXT_REVISION,
        WIKITEXT_SOURCE,
    )
    from .train_evq_lora import (
        build_training_inv_freq,
        inject_inv_freq,
        load_frequency_artifact,
        public_model_identifier,
        resolve_model_rope_geometry,
        validate_legacy_model_geometry,
        verify_model_inv_freq,
    )
    from .train_positional_distill import configure_packed_free_causal_sdpa
    from .validate_legacy_lora_artifact import validate_artifact
except ImportError:
    from eval_positional_distill import causal_backbone, chunked_causal_nll
    from legacy_lora_protocol import legacy_eval_filename, sha256_file
    from prepare_positional_distill_data import tokenizer_source_fingerprint
    from prepare_legacy_model_manifest import validate_model_manifest
    from prepare_legacy_wikitext import (
        WIKITEXT_CONFIG,
        WIKITEXT_RAW_SHA256,
        WIKITEXT_REVISION,
        WIKITEXT_SOURCE,
    )
    from train_evq_lora import (
        build_training_inv_freq,
        inject_inv_freq,
        load_frequency_artifact,
        public_model_identifier,
        resolve_model_rope_geometry,
        validate_legacy_model_geometry,
        verify_model_inv_freq,
    )
    from train_positional_distill import configure_packed_free_causal_sdpa
    from validate_legacy_lora_artifact import validate_artifact


_GEO = re.compile(r"^geo_longalign_s(42|43|44)$")
_EVQ = re.compile(r"^evq_longalign_tau1414_s(42|43|44)$")


def variant_spec(variant: str) -> Dict[str, Any]:
    if variant == "base_geo":
        return {"method": "native_geo", "seed": None, "requires_adapter": False}
    if variant == "base_evq_tau1414":
        return {"method": "evq_cosh", "seed": None, "requires_adapter": False}
    match = _GEO.fullmatch(variant)
    if match:
        return {"method": "native_geo", "seed": int(match.group(1)), "requires_adapter": True}
    match = _EVQ.fullmatch(variant)
    if match:
        return {"method": "evq_cosh", "seed": int(match.group(1)), "requires_adapter": True}
    raise ValueError(f"unsupported legacy evaluation variant: {variant}")


def _load_eval_manifest(path: Path) -> tuple[Dict[str, Any], torch.Tensor]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "objective": "legacy_wikitext2_matched_ppl_v1",
        "lengths": [8192, 16384, 32768],
        "chunks_per_length": 5,
        "offset_rule": "chunk_index_times_context_length",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"legacy evaluation manifest mismatch for {key}")
    expected_source = {
        "source_id": WIKITEXT_SOURCE,
        "revision": WIKITEXT_REVISION,
        "config": WIKITEXT_CONFIG,
        "split": "test",
        "raw_sha256": WIKITEXT_RAW_SHA256,
    }
    source = manifest.get("source", {})
    for key, value in expected_source.items():
        if source.get(key) != value:
            raise ValueError(f"legacy evaluation source mismatch for {key}")
    token_record = manifest.get("tokens", {})
    token_path = Path(path).parent / str(token_record.get("name", ""))
    if not token_path.is_file() or sha256_file(token_path) != token_record.get("sha256"):
        raise ValueError("frozen WikiText tokens are missing or hash-mismatched")
    token_ids = torch.load(token_path, map_location="cpu", weights_only=True).to(torch.long)
    if token_ids.ndim != 1 or token_ids.numel() < 5 * 32768:
        raise ValueError("frozen WikiText token tensor is incomplete")
    return manifest, token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_manifest", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--adapter_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--lm_head_chunk_tokens", type=int, default=1024)
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def _validate_existing_result(
    path: Path,
    *,
    variant: str,
    spec: Dict[str, Any],
    model_manifest_sha256: str,
    eval_manifest_sha256: str,
    adapter_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format_version": 1,
        "variant": variant,
        "method": spec["method"],
        "seed": spec["seed"],
        "model_manifest_sha256": model_manifest_sha256,
        "eval_manifest_sha256": eval_manifest_sha256,
        "adapter_sha256": adapter_meta.get("adapter_sha256") if adapter_meta else None,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"existing evaluation mismatch for {key}")
    if record.get("frequency_provenance", {}).get("method") != spec["method"]:
        raise ValueError("existing evaluation frequency method mismatch")
    for length, context in (("8K", 8192), ("16K", 16384), ("32K", 32768)):
        metric = record.get("ppl", {}).get(length, {})
        values = [metric.get("nll"), metric.get("ppl")]
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise ValueError(f"existing evaluation has non-finite metric at {length}")
        if float(metric["ppl"]) <= 0 or metric.get("chunks") != 5:
            raise ValueError(f"existing evaluation coverage mismatch at {length}")
        if metric.get("scored_tokens") != 5 * (context - 1):
            raise ValueError(f"existing evaluation scored-token mismatch at {length}")
        chunk_nll = metric.get("per_chunk_nll")
        if not isinstance(chunk_nll, list) or len(chunk_nll) != 5 or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in chunk_nll
        ):
            raise ValueError(f"existing evaluation lacks per-chunk evidence at {length}")
    return record


def main() -> None:
    args = parse_args()
    spec = variant_spec(args.variant)
    if spec["requires_adapter"] != (args.adapter_dir is not None):
        raise ValueError("adapter presence does not match the requested variant")
    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    validate_model_manifest(Path(args.model_name), model_manifest, verify_hashes=False)
    eval_manifest, full_ids = _load_eval_manifest(args.eval_manifest)
    if args.adapter_dir is not None:
        adapter_meta = validate_artifact(
            args.adapter_dir,
            expected_method=spec["method"],
            expected_seed=spec["seed"],
        )
        if adapter_meta["model_manifest_sha256"] != sha256_file(args.model_manifest):
            raise ValueError("adapter and evaluator model manifests differ")
    else:
        adapter_meta = None
    output_path = args.output_dir / legacy_eval_filename(args.variant)
    if args.validate_only:
        record = _validate_existing_result(
            output_path,
            variant=args.variant,
            spec=spec,
            model_manifest_sha256=sha256_file(args.model_manifest),
            eval_manifest_sha256=sha256_file(args.eval_manifest),
            adapter_meta=adapter_meta,
        )
        print(json.dumps({"status": "valid", "output": str(output_path), "variant": record["variant"]}, indent=2))
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, use_fast=True, local_files_only=Path(args.model_name).is_dir()
    )
    if eval_manifest.get("tokenizer") != tokenizer_source_fingerprint(args.model_name):
        raise ValueError("evaluation tokenizer does not match frozen WikiText tokens")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=Path(args.model_name).is_dir(),
    )
    model.config.use_cache = False
    geometry = resolve_model_rope_geometry(model.config)
    validate_legacy_model_geometry(model.config, geometry)
    configure_packed_free_causal_sdpa(model)
    expected_inv_freq, expected_schedule = build_training_inv_freq(
        rope_method=spec["method"],
        head_dim=geometry.head_dim,
        base=geometry.rope_base,
        tau=1.414,
    )
    if args.adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_dir)
        inv_freq, artifact_data, frequency_provenance = load_frequency_artifact(
            args.adapter_dir / "custom_inv_freq.pt",
            expected_method=spec["method"],
        )
        if not torch.allclose(
            inv_freq.to(torch.float64),
            expected_inv_freq.to(torch.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("adapter frequency tensor does not match the canonical schedule")
        expected_artifact = {
            "head_dim": geometry.head_dim,
            "base": geometry.rope_base,
            "tau": expected_schedule["tau"],
            "midpoint": expected_schedule["midpoint"],
        }
        for key, value in expected_artifact.items():
            actual = artifact_data.get(key)
            if isinstance(value, float):
                if not isinstance(actual, (int, float)) or not math.isclose(
                    float(actual), value, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"adapter frequency artifact mismatch for {key}")
            elif actual != value:
                raise ValueError(f"adapter frequency artifact mismatch for {key}")
    else:
        inv_freq, schedule = expected_inv_freq, expected_schedule
        frequency_provenance = {
            "artifact": None,
            "method": schedule["method"],
            "tau": schedule["tau"],
            "midpoint": schedule["midpoint"],
        }
    model.to("cuda")
    inject_inv_freq(model, inv_freq)
    verification = verify_model_inv_freq(model, inv_freq)
    model.eval()
    backbone = causal_backbone(model)
    lm_head = model.get_output_embeddings()
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    ppl: Dict[str, Any] = {}
    for length in (8192, 16384, 32768):
        chunk_nll = []
        for chunk_index in range(5):
            start = chunk_index * length
            input_ids = full_ids[start : start + length].unsqueeze(0).to(device)
            with torch.inference_mode():
                hidden = backbone(
                    input_ids=input_ids,
                    attention_mask=None,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                nll = chunked_causal_nll(
                    hidden,
                    input_ids,
                    lm_head,
                    chunk_tokens=args.lm_head_chunk_tokens,
                )
            value = float(nll.cpu())
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite NLL at length={length}, chunk={chunk_index}")
            chunk_nll.append(value)
            del input_ids, hidden, nll
        mean_nll = float(np.mean(chunk_nll))
        ppl[f"{length // 1024}K"] = {
            "nll": mean_nll,
            "ppl": math.exp(mean_nll),
            "chunks": 5,
            "per_chunk_nll": chunk_nll,
            "scored_tokens": 5 * (length - 1),
        }
    final_verification = verify_model_inv_freq(model, inv_freq)
    output = {
        "format_version": 1,
        "variant": args.variant,
        "method": spec["method"],
        "seed": spec["seed"],
        "model": public_model_identifier(args.model_name),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "eval_manifest_sha256": sha256_file(args.eval_manifest),
        "adapter_sha256": adapter_meta.get("adapter_sha256") if adapter_meta else None,
        "training_data_manifest_sha256": (
            adapter_meta.get("data_manifest_sha256") if adapter_meta else None
        ),
        "training_code_sha256": adapter_meta.get("code_sha256") if adapter_meta else None,
        "training_runtime": adapter_meta.get("runtime") if adapter_meta else None,
        "frequency_provenance": frequency_provenance,
        "frequency_verification": {
            "initial": verification,
            "final": final_verification,
        },
        "ppl": ppl,
        "runtime": {
            "seconds": time.time() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "lm_head_chunk_tokens": args.lm_head_chunk_tokens,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps({
        "output": str(output_path),
        "ppl": {key: value["ppl"] for key, value in ppl.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
