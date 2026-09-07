"""Shared runtime: explicit Flash-only CUDA and exact prepared-asset checks."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch
from .contracts import sha_file
from .tables import install_static, tensor_sha, verify_static


def cuda_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU run requested but CUDA unavailable')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported')
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision('high')
    props = torch.cuda.get_device_properties(0)
    # A real kernel call is required; package availability alone is insufficient.
    q = torch.randn(1, 2, 16, 64, device='cuda', dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True)
    torch.cuda.synchronize()
    return dict(name=props.name, capability=list(torch.cuda.get_device_capability()),
                total_memory=props.total_memory, backend='Flash SDPA only', bf16=True)


def versions():
    return dict(python=platform.python_version(), **{
        n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','peft')})


def verify_prepared(prepared):
    root = Path(prepared)
    assets = json.loads((root/'assets_and_missing.json').read_text())
    for name, expected in assets['prepared_files'].items():
        if sha_file(root/name) != expected:
            raise ValueError(f'prepared artifact drift: {name}')
    for name, expected in assets['assets'].items():
        if sha_file(name) != expected:
            raise ValueError(f'source asset drift: {name}')
    return assets


def load_model(model_path, prepared, arm, *, checkpoint=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = Path(prepared)
    manifest = json.loads((root/'table_manifest.json').read_text())
    entry = manifest['arms'][arm]
    values = np.load(root/entry['path'],allow_pickle=False)
    if tensor_sha(values)!=entry['tensor_sha256']:
        raise ValueError('table hash mismatch')
    if checkpoint:
        cm=json.loads((Path(checkpoint)/'checkpoint_manifest.json').read_text())
        for name,digest in cm['files'].items():
            if sha_file(Path(checkpoint)/name)!=digest:raise ValueError('saved checkpoint file drift')
        deployment=json.loads((Path(checkpoint)/'deployment.json').read_text())
        if (deployment['arm'],deployment['table_sha256'],deployment['amplitude'])!=(arm,entry['tensor_sha256'],entry['amplitude']):
            raise ValueError('checkpoint deployment identity mismatch')
        entry={**entry,'checkpoint_manifest_sha256':sha_file(Path(checkpoint)/'checkpoint_manifest.json'),
               'training_input_tokens':cm['input_tokens'],'training_step':cm['step']}
    # from_pretrained device_map avoids a full CPU copy during a GPU run.
    is_adapter=checkpoint and (Path(checkpoint)/'adapter_config.json').exists()
    model = AutoModelForCausalLM.from_pretrained(
        model_path if is_adapter else checkpoint or model_path, dtype=torch.bfloat16, device_map={'':'cuda'},
        attn_implementation='sdpa', local_files_only=True)
    if is_adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,checkpoint,is_trainable=False).get_base_model()
    model.config.use_cache = True
    install_static(model,values,entry['amplitude'])
    verify_static(model,values,entry['amplitude'])
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    return model,tokenizer,values,entry
