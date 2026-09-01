#!/usr/bin/env python3
"""Two-document Native Flash versus bounded FP32 attention reference audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM
    from scripts.lib.checkpoint_identity import safetensors_weight_set_sha256
    from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.evaluate_ruler import (
        configure_cuda, configure_ruler_flash_attention, ruler_causal_mask,
    )
    if args.output.exists():
        raise RuntimeError("output must be fresh")
    manifest_path = args.data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    row_path = args.data_root / manifest["files"]["calibration"]["path"]
    if hashlib.sha256(row_path.read_bytes()).hexdigest() != manifest["files"]["calibration"]["sha256"]:
        raise RuntimeError("calibration input hash changed")
    if safetensors_weight_set_sha256(args.checkpoint) != args.expected_weight_sha256:
        raise RuntimeError("weight identity changed")
    rows = [json.loads(line) for line in row_path.read_text().splitlines()]
    rows = [r for r in rows if r["family"] == "natural" and r["length"] in (4096, 8192)
            and r["sample_id"] in ("natural-calibration-000", "natural-calibration-001")]
    if len(rows) != 4:
        raise RuntimeError("expected two paired calibration documents")
    configure_cuda()
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").eval().to("cuda")
    initial_inv = model.model.rotary_emb.inv_freq.detach().clone()

    def reference_attention(module, query, key, value, attention_mask, *, scaling=None, dropout=0.0, **kwargs):
        del module, kwargs
        if attention_mask is not None or dropout or query.shape[-2] != key.shape[-2]:
            raise RuntimeError("reference admits unpadded full prefill only")
        group = query.shape[1] // key.shape[1]
        with torch.autocast("cuda", enabled=False):
            q = query.float()
            k = key.float().repeat_interleave(group, dim=1)
            v = value.float().repeat_interleave(group, dim=1)
            output = torch.empty_like(q)
            key_positions = torch.arange(k.shape[-2], device=k.device)
            for start in range(0, q.shape[-2], 128):
                stop = min(start + 128, q.shape[-2])
                scores = (q[:, :, start:stop] @ k.transpose(-1, -2)) * float(scaling)
                future = key_positions[None, :] > torch.arange(start, stop, device=k.device)[:, None]
                scores.masked_fill_(future, float("-inf"))
                output[:, :, start:stop] = scores.softmax(-1) @ v
        return output.to(query.dtype).transpose(1, 2).contiguous(), None

    AttentionInterface.register("p0_chunked_fp32_reference", reference_attention)
    AttentionMaskInterface.register("p0_chunked_fp32_reference", ruler_causal_mask)
    results = []
    torch.cuda.reset_peak_memory_stats()
    for row in rows:
        ids = torch.tensor([row["input_ids"]], device="cuda")
        target = ids[:, -256:]
        outputs = {}
        for backend in ("flash", "chunked_fp32_reference"):
            if backend == "flash":
                configure_ruler_flash_attention(model)
            else:
                model.config._attn_implementation = "p0_chunked_fp32_reference"
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, use_cache=False, logits_to_keep=257).logits[:, :-1].float()
            losses = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
            if not bool(torch.isfinite(losses).all()):
                raise RuntimeError("nonfinite reference loss")
            outputs[backend] = {"nll": float(losses.mean()), "top1": logits.argmax(-1).cpu().tolist()[0]}
            del logits, losses
        result = {"sample_id": row["sample_id"], "length": row["length"],
                  "flash_nll": outputs["flash"]["nll"],
                  "fp32_reference_nll": outputs["chunked_fp32_reference"]["nll"],
                  "top1_agreement": sum(a == b for a, b in zip(outputs["flash"]["top1"],
                                             outputs["chunked_fp32_reference"]["top1"])) / 256}
        results.append(result)
        print(json.dumps(result), flush=True)
    if not torch.equal(initial_inv, model.model.rotary_emb.inv_freq):
        raise RuntimeError("Native table changed")
    receipt = {"status": "P0_ATTENTION_REFERENCE_COMPLETE", "rows": results,
               "weights_sha256": args.expected_weight_sha256,
               "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "query_chunk": 128, "explicit_fp32_reference_not_fallback": True,
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "claim_ceiling": "same Native weights and inputs; implementation diagnostic, not length selection"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
