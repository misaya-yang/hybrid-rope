#!/usr/bin/env python3
"""Audit realized Native RoPE arithmetic without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoConfig
    from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if config.model_type != "gemma" or config.rope_parameters["rope_type"] != "default":
        raise RuntimeError("this diagnostic is registered for Native Gemma only")
    rotary = GemmaRotaryEmbedding(config).to("cuda")
    inv = rotary.inv_freq
    positions = torch.arange(config.max_position_embeddings, device="cuda", dtype=torch.int64)[None]
    reference = (inv.double()[None, :, None] * positions.double()[:, None, :]).transpose(1, 2)
    reference_cos = torch.cat((reference, reference), dim=-1).cos().to(torch.bfloat16)
    reference_sin = torch.cat((reference, reference), dim=-1).sin().to(torch.bfloat16)
    hidden = torch.empty(1, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    observations = []
    for name, precision, tf32 in (("research_high_tf32", "high", True), ("ieee_fp32", "highest", False)):
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            cos, sin = rotary(hidden, positions)
        with torch.autocast("cuda", enabled=False):
            phase = (inv[None, :, None].float() @ positions[:, None, :].float()).transpose(1, 2)
        observations.append({
            "mode": name, "phase_max_abs_vs_fp64": float((phase.double()-reference).abs().max()),
            "phase_rms_vs_fp64": float((phase.double()-reference).square().mean().sqrt()),
            "consecutive_phase_aliases": int((phase[:, 1:] == phase[:, :-1]).all(-1).sum()),
            "cos_max_abs_same_bf16": float((cos.float()-reference_cos.float()).abs().max()),
            "sin_max_abs_same_bf16": float((sin.float()-reference_sin.float()).abs().max()),
            "phase_sha256_float32": hashlib.sha256(phase.cpu().numpy().tobytes()).hexdigest(),
        })
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    result = {
        "status": "NATIVE_ROPE_ARITHMETIC_AUDIT_COMPLETE", "weights_loaded": False,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "positions_dtype": str(positions.dtype), "positions": positions.numel(),
        "unique_positions": int(positions.unique().numel()), "position_step_one": bool(torch.all(positions.diff()==1)),
        "native_dtype": str(inv.dtype),
        "native_sha256_float32": hashlib.sha256(inv.cpu().float().numpy().tobytes()).hexdigest(),
        "rotary_source_sha256": hashlib.sha256(inspect.getsource(GemmaRotaryEmbedding).encode()).hexdigest(),
        "observations": observations,
        "claim_ceiling": "arithmetic diagnostic only; no language-model or L_ref conclusion",
    }
    if args.output.exists():
        raise RuntimeError("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
