#!/usr/bin/env python3
"""VRAM/throughput probe for the 7B 500M CPT run (teacher-free, KL-cache mode).

Fail-fast, records the boundary. Steps:
  1. load OLMo-2-7B + install frozen Z table + gain
  2. LoRA r16 all linears + RMSNorm + embed_tokens trainable, grad-ckpt on
  3. AdamW build (measures optimizer-state memory)
  4. 3 dense 16K micro-steps (deterministic random ids; liger fused CE),
     record per-step time + peak memory after warmup step
  5. one 2K-token micro-step (proxy for the cached-KL row cost)
Verdict PASS if peak <= 29GB (3GB headroom on the 32GB card) and
sec_per_micro_16k <= 60 (else wall time for 7,648 updates > ~10 days).
Writes probe_train_result.json; never modifies any frozen artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

SEQ_LEN = 16384
VOCAB = 50304  # OLMo-2 vocab size (assert-checked against config below)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--arm", default="Z")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    result = {"status": "RUNNING", "seq_len": SEQ_LEN}

    try:
        from transformers import AutoModelForCausalLM

        manifest = json.loads((Path(args.tables) / "manifest_round12.json").read_text())
        assert manifest["status"] == "ROUND12_STATIC_TABLES_FROZEN_V1"
        entry = manifest["arms"][args.arm]
        table = np.load(Path(args.tables) / entry["path"], allow_pickle=False)
        gain = float(entry["rotary_amplitude"])

        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", low_cpu_mem_usage=True)
        assert model.config.vocab_size == VOCAB, model.config.vocab_size
        rotary = model.model.rotary_emb
        assert rotary.inv_freq.shape == table.shape
        rotary.inv_freq.copy_(torch.from_numpy(table).to(rotary.inv_freq))
        rotary.original_inv_freq = rotary.inv_freq.detach().clone()
        rotary.attention_scaling = gain
        rotary.inv_freq.requires_grad_(False)

        from peft import LoraConfig, get_peft_model
        for p in model.parameters():
            p.requires_grad_(False)
        cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"],
                         bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        n_norm = n_emb = 0
        for name, p in model.named_parameters():
            if "lm_head" in name or "rotary" in name:
                continue
            if "norm" in name and name.endswith(".weight"):
                p.requires_grad_(True)
                n_norm += 1
            elif "embed_tokens" in name:
                p.requires_grad_(True)
                n_emb += 1
        assert n_norm > 0 and n_emb > 0
        model.enable_input_require_grads()
        model.to("cuda")
        model.train()
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

        params = [p for p in model.parameters() if p.requires_grad_]
        opt = torch.optim.AdamW(params, lr=2e-5, betas=(0.9, 0.95),
                                weight_decay=0.0)

        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        flce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        inner = model.base_model.model
        backbone, head_w = inner.model, inner.lm_head.weight

        g = torch.Generator(device="cuda").manual_seed(42)
        times = []
        for i in range(3):
            ids = torch.randint(0, VOCAB, (1, SEQ_LEN + 1), device="cuda",
                                generator=g)
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.time()
            h = backbone(input_ids=ids[:, :-1], attention_mask=None).last_hidden_state
            loss = flce(h.reshape(-1, h.shape[-1]), head_w, ids[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            print(f"micro {i}: {times[-1]:.1f}s loss={loss.item():.4f} "
                  f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
        torch.cuda.reset_peak_memory_stats()
        # one more step to measure steady-state peak (post-optimizer-state init)
        ids = torch.randint(0, VOCAB, (1, SEQ_LEN + 1), device="cuda", generator=g)
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.time()
        h = backbone(input_ids=ids[:, :-1], attention_mask=None).last_hidden_state
        loss = flce(h.reshape(-1, h.shape[-1]), head_w, ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t_16k = time.time() - t0

        # 2K proxy (cached-KL row): forward+backward on 2048 tokens
        ids2 = torch.randint(0, VOCAB, (1, 2049), device="cuda", generator=g)
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.time()
        h2 = backbone(input_ids=ids2[:, :-1], attention_mask=None).last_hidden_state
        loss2 = flce(h2.reshape(-1, h2.shape[-1]), head_w, ids2[:, 1:].reshape(-1))
        loss2.backward()
        opt.step()
        torch.cuda.synchronize()
        t_2k = time.time() - t0

        peak = torch.cuda.max_memory_allocated() / 1e9
        # estimate: 7648 updates x (4 x t16k + t2k-ish KL gather) + val overhead
        est_s = 7648 * (4 * t_16k + t_2k)
        result.update({
            "peak_mem_gb_16k_step": round(peak, 2),
            "sec_per_micro_16k": round(t_16k, 1),
            "sec_per_micro_2k": round(t_2k, 1),
            "first3_micro_times": [round(t, 1) for t in times],
            "est_wall_seconds_phase_a": round(est_s),
            "est_wall_days_phase_a": round(est_s / 86400, 1),
        })
        ok_mem = peak <= 29.0
        ok_speed = t_16k <= 60.0
        result["status"] = "PASS" if (ok_mem and ok_speed) else "FAIL"
        result["fail_reasons"] = ([f"peak {peak:.1f}GB > 29GB"] if not ok_mem else []) \
            + ([f"micro 16K {t_16k:.0f}s > 60s"] if not ok_speed else [])
    except Exception as e:  # noqa: BLE001
        result["status"] = "FAIL"
        result["fail_reasons"] = [f"{type(e).__name__}: {e}"]
        print("PROBE_FAIL", result["fail_reasons"], flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print("PROBE_TRAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
