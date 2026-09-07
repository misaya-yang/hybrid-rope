"""Check the actual full-length HF rotary path against elementwise FP32 phases.

Numerical audit only: no model weights, forward answers, training, or new table.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from scripts.experiments.cross_audit.runtime import cuda_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    candidate = json.loads(args.candidate.read_text())
    means = args.root/'runs/carrier_native_means_02/means.npz'
    if hashlib.sha256(means.read_bytes()).hexdigest() != candidate['source_means_sha256']:
        raise ValueError('frozen Native table source changed')
    native = np.load(means, allow_pickle=False)['native']
    table = candidate['tables']['Carrier']
    config = AutoConfig.from_pretrained(args.root/'model', local_files_only=True)
    hardware = cuda_runtime()  # same high matmul precision and backend flags as the real runs
    rotary = Qwen2RotaryEmbedding(config).cuda()
    if rotary.rope_type != 'default':
        raise ValueError('same static default rotary path required')
    records = []
    with torch.inference_mode():
        for name, array, gain in [('Native', native, 1.), ('NativeSectorCarrier', table['values_float32'], table['gain'])]:
            rotary.inv_freq = torch.tensor(np.asarray(array, dtype=np.float32), device='cuda')
            rotary.original_inv_freq = rotary.inv_freq.clone()
            rotary.attention_scaling = gain
            for length in (32768, 131072):
                pos = torch.arange(length, device='cuda')[None, :]
                dummy = torch.zeros(1, 1, 1, device='cuda', dtype=torch.float32)
                cosine, sine = rotary(dummy, pos)
                # Multiplication has no matrix-product accumulation or TF32 path.
                expected_phase = pos[:, :, None].float()*rotary.inv_freq[None, None, :]
                expected_phase = torch.cat([expected_phase, expected_phase], dim=-1)
                cosine_error = float((cosine-expected_phase.cos()*gain).abs().max())
                sine_error = float((sine-expected_phase.sin()*gain).abs().max())
                record = dict(table=name, length=length, cosine_max_abs=cosine_error, sine_max_abs=sine_error)
                records.append(record)
                print(json.dumps(record), flush=True)
                if max(cosine_error, sine_error) > 1e-6:
                    raise ValueError('full-length rotary precision differs from declared FP32 phase')
                del pos, dummy, cosine, sine, expected_phase
    result = dict(status='FULL_32K_128K_ROTARY_NUMERICS_VERIFIED_NO_CAPABILITY_CLAIM',
        hardware=hardware, matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        hf_rotary_source_sha256=hashlib.sha256(Path(inspect.getfile(Qwen2RotaryEmbedding)).read_bytes()).hexdigest(),
        candidate_file_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(), records=records)
    with args.out.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
