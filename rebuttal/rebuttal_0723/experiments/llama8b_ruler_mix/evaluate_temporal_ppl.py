#!/usr/bin/env python3
"""Compare native base and a matched LoRA arm on frozen 2026 PPL."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

from experiments.lora_evq_v2.eval_positional_distill import causal_backbone
from experiments.lora_evq_v2.eval_temporal_holdout_matched import (
    _score_arm,
    load_domain_artifacts,
    sha256_file,
)
from experiments.lora_evq_v2.train_evq_lora import (
    inject_inv_freq,
    load_frequency_artifact,
    verify_model_inv_freq,
)
from experiments.lora_evq_v2.train_positional_distill import (
    configure_packed_free_causal_sdpa,
)


STATUS = "LLAMA8B_RULER_MIX_TEMPORAL_PPL_COMPLETE_V2"
METHOD_TO_ARM = {
    "evq_cosh": "evq_ruler_mix",
    "native_geo": "native_ruler_mix",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=sorted(METHOD_TO_ARM),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lm-head-chunk-tokens", type=int, default=1024)
    parser.add_argument(
        "--required-gpu-substring",
        default="RTX PRO 6000",
    )
    return parser.parse_args()


def load_domains(
    root: Path,
) -> tuple[dict[str, Any], dict[str, tuple[Any, torch.Tensor, torch.Tensor]]]:
    collection_path = root / "collection_manifest.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if (
        collection.get("schema")
        != "evq_cosh.temporal_holdout_2026.collection.v1"
    ):
        raise RuntimeError("temporal collection schema drift")
    domains = {}
    for domain, record in collection["domains"].items():
        manifest_path = root / str(record["manifest"])
        if sha256_file(manifest_path) != record["manifest_sha256"]:
            raise RuntimeError(f"temporal manifest drift: {domain}")
        manifest, ids, mask = load_domain_artifacts(manifest_path.parent)
        manifest["_manifest_path"] = str(manifest_path)
        domains[domain] = (manifest, ids, mask)
    if len(domains) != 3:
        raise RuntimeError("registered temporal protocol requires 3 domains")
    return collection, domains


def verify_adapter(root: Path, expected_method: str) -> dict[str, Any]:
    result_path = root / "result.json"
    metadata_path = root / "experiment_meta.json"
    model_path = root / "adapter_model.safetensors"
    frequency_path = root / "custom_inv_freq.pt"
    for path in (result_path, metadata_path, model_path, frequency_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        result.get("status")
        != "LLAMA8B_PHYSICAL_8K_RULER_MIX_COMPLETE_V1"
        or result.get("method") != expected_method
        or metadata.get("status") != "complete"
        or metadata.get("method") != expected_method
    ):
        raise RuntimeError(f"adapted {expected_method} identity drift")
    if sha256_file(model_path) != result["adapter_sha256"]:
        raise RuntimeError("adapted EVQ weight hash drift")
    if (
        sha256_file(frequency_path)
        != result["frequency_artifact_sha256"]
    ):
        raise RuntimeError("adapted EVQ frequency hash drift")
    return {
        "adapter_sha256": result["adapter_sha256"],
        "frequency_sha256": result["frequency_artifact_sha256"],
        "result_sha256": sha256_file(result_path),
        "metadata_sha256": sha256_file(metadata_path),
        "training_view_manifest_sha256": result["training_view"][
            "manifest_sha256"
        ],
    }


def macro_comparison(
    arms: dict[str, Any],
    adapted_arm: str,
) -> dict[str, Any]:
    output = {}
    for length in ("8K", "16K", "32K"):
        base = arms["native_base"]["domain_macro"]["prefixes"][length]
        adapted = arms[adapted_arm]["domain_macro"]["prefixes"][length]
        output[length] = {
            "native_base": {
                "nll": float(base["nll"]),
                "ppl": float(base["ppl"]),
            },
            adapted_arm: {
                "nll": float(adapted["nll"]),
                "ppl": float(adapted["ppl"]),
            },
            "delta_nll_adapted_minus_native": (
                float(adapted["nll"]) - float(base["nll"])
            ),
            "ppl_ratio_adapted_over_native": (
                float(adapted["ppl"]) / float(base["ppl"])
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    model_manifest = args.model_manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    adapter = args.adapter.resolve()
    output = args.output.resolve()
    adapted_arm = METHOD_TO_ARM[args.method]
    if output.exists() or output.with_suffix(output.suffix + ".incomplete").exists():
        raise FileExistsError(output)
    if not 1 <= int(args.lm_head_chunk_tokens) <= 2048:
        raise ValueError("lm-head chunk size must be in [1, 2048]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu_name = torch.cuda.get_device_name(0)
    if args.required_gpu_substring not in gpu_name:
        raise RuntimeError(f"wrong GPU: {gpu_name}")

    model_record = json.loads(model_manifest.read_text(encoding="utf-8"))
    if model_record.get("format_version") != 1:
        raise RuntimeError("model manifest format drift")
    adapter_record = verify_adapter(adapter, args.method)
    collection, domains = load_domains(dataset_root)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    configure_packed_free_causal_sdpa(model)
    model.to("cuda")
    model.eval()
    device = torch.device("cuda")
    lm_head = model.get_output_embeddings()

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        native = _score_arm(
            arm="native_base",
            model=model,
            backbone=causal_backbone(model),
            lm_head=lm_head,
            domains=domains,
            device=device,
            lm_head_chunk_tokens=int(args.lm_head_chunk_tokens),
            disabled=False,
            max_packs_per_domain=None,
        )

    inv_freq, frequency_metadata, frequency_provenance = (
        load_frequency_artifact(
            adapter / "custom_inv_freq.pt",
            expected_method=args.method,
        )
    )
    model = PeftModel.from_pretrained(
        model,
        adapter,
        is_trainable=False,
    )
    inject_inv_freq(model, inv_freq)
    frequency_verification = verify_model_inv_freq(model, inv_freq)
    model.eval()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        adapted = _score_arm(
            arm=adapted_arm,
            model=model,
            backbone=causal_backbone(model),
            lm_head=model.get_output_embeddings(),
            domains=domains,
            device=device,
            lm_head_chunk_tokens=int(args.lm_head_chunk_tokens),
            disabled=False,
            max_packs_per_domain=None,
        )

    arms = {"native_base": native, adapted_arm: adapted}
    result = {
        "status": STATUS,
        "evidence_boundary": (
            "teacher-forced natural-text NLL/PPL on timestamp-selected "
            "2026 temporal holdouts; not a downstream capability metric"
        ),
        "protocol": {
            "domains": sorted(domains),
            "packs_per_domain": 8,
            "canonical_pack_length": 32768,
            "prefix_lengths": [8192, 16384, 32768],
            "nested_prefixes": True,
            "scoring": "document-first tokens excluded by frozen mask",
            "attention": "PyTorch Flash-only SDPA",
            "lm_head_chunk_tokens": int(args.lm_head_chunk_tokens),
        },
        "method": args.method,
        "comparison": macro_comparison(arms, adapted_arm),
        "arms": arms,
        "artifacts": {
            "model_manifest_sha256": sha256_file(model_manifest),
            "collection_manifest_sha256": sha256_file(
                dataset_root / "collection_manifest.json"
            ),
            "collection_claim_boundary": collection["claim_boundary"],
            "adapter": adapter_record,
        },
        "frequency": {
            "provenance": frequency_provenance,
            "metadata": {
                key: value
                for key, value in frequency_metadata.items()
                if key != "inv_freq"
            },
            "verification": frequency_verification,
        },
        "runtime": {
            "seconds": time.time() - started,
            "gpu": gpu_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_cuda_memory_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
