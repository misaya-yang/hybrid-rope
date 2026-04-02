#!/usr/bin/env python3
"""Test EVQ-LoRA + YaRN at 16K/32K."""
import torch, random, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ruler import make_haystack
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from train_evq_lora import inject_inv_freq, find_rotary_modules, compute_geometric_inv_freq
from pathlib import Path
import numpy as np

MODEL = "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
CKPT = "/root/autodl-tmp/lora_evq_v2/checkpoints/evq_r64_tau1414"
WIKI = "/root/autodl-tmp/data/wikitext2/wikitext2_test.txt"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def load_evq_yarn(yarn_factor):
    """Load EVQ-LoRA + YaRN scaling."""
    rope_scaling = {"type": "yarn", "factor": yarn_factor,
                    "original_max_position_embeddings": 8192}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", rope_scaling=rope_scaling)
    model = PeftModel.from_pretrained(model, CKPT)
    data = torch.load(f"{CKPT}/custom_inv_freq.pt", map_location="cpu", weights_only=True)
    inject_inv_freq(model, data["inv_freq"])
    # Verify
    mods = find_rotary_modules(model)
    if mods:
        actual = mods[0][1].inv_freq.detach().cpu().to(torch.float64)
        geo = compute_geometric_inv_freq(128, 500000.0)
        err_evq = (actual - data["inv_freq"].to(torch.float64)).abs().max().item()
        err_geo = (actual - geo).abs().max().item()
        print(f"  [ROPE] EVQ+YaRN(x{yarn_factor}): "
              f"{'EVQ active' if err_evq < err_geo else 'WARNING: GEO!'}")
    model.eval()
    return model


def load_base_yarn(yarn_factor):
    """Load base + YaRN (no EVQ, no LoRA) for comparison."""
    rope_scaling = {"type": "yarn", "factor": yarn_factor,
                    "original_max_position_embeddings": 8192}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", rope_scaling=rope_scaling)
    model.eval()
    return model


def test_ppl(model, label):
    """Positional PPL breakdown."""
    print(f"\n--- PPL: {label} ---")
    text = Path(WIKI).read_text()
    full_ids = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]
    device = next(model.parameters()).device

    ctx = 16384
    if len(full_ids) < ctx:
        print("  Not enough text for 16K")
        return {}
    chunk = full_ids[:ctx].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(chunk)
        logits = out.logits[0]
    shift_logits = logits[:-1]
    shift_labels = chunk[0, 1:]
    losses = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="none")

    results = {}
    windows = [(0, 4096, "0-4K"), (4096, 8192, "4K-8K"),
               (8192, 12288, "8K-12K"), (12288, 16383, "12K-16K")]
    for start, end, name in windows:
        end = min(end, len(losses))
        if start >= end:
            continue
        w_loss = losses[start:end].mean().item()
        w_ppl = math.exp(w_loss)
        results[name] = round(w_ppl, 2)
        print(f"  {name}: PPL={w_ppl:.2f}")
    return results


def test_niah(model, label, ctx_len=16384):
    """S-NIAH generation test."""
    print(f"\n--- S-NIAH@{ctx_len//1024}K: {label} ---")
    device = next(model.parameters()).device

    for t in range(3):
        rng = random.Random(42 + t)
        key = str(rng.randint(10000, 99999))
        needle = f"The secret passkey is {key}. Remember this number."
        depth = rng.uniform(0.2, 0.8)
        n_before = int((ctx_len - 200) * depth)
        n_after = ctx_len - 200 - n_before
        hay_b = make_haystack(n_before, tok, seed=t)
        hay_a = make_haystack(n_after, tok, seed=t + 1000)
        context = f"{hay_b}\n{needle}\n{hay_a}"

        msgs = [
            {"role": "system", "content": "Answer with only the number, nothing else."},
            {"role": "user", "content": f"{context}\n\nWhat is the 5-digit secret passkey?"},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=ctx_len)["input_ids"].to(device)

        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        found = key in resp
        print(f"  Trial {t}: key={key} depth={depth:.2f} input={ids.shape[1]} "
              f"resp=[{resp[:80]}] {'OK' if found else 'FAIL'}")


# ====== Run all configs ======

configs = [
    ("Base (no YaRN)", lambda: AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")),
    ("Base + YaRN x2", lambda: load_base_yarn(2.0)),
    ("EVQ-LoRA (no YaRN)", None),  # special handling
    ("EVQ-LoRA + YaRN x2", lambda: load_evq_yarn(2.0)),
    ("EVQ-LoRA + YaRN x4", lambda: load_evq_yarn(4.0)),
]

for name, loader in configs:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if name == "EVQ-LoRA (no YaRN)":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
        model = PeftModel.from_pretrained(model, CKPT)
        data = torch.load(f"{CKPT}/custom_inv_freq.pt", map_location="cpu", weights_only=True)
        inject_inv_freq(model, data["inv_freq"])
        model.eval()
    else:
        model = loader()
        if hasattr(model, 'eval'):
            model.eval()

    test_ppl(model, name)
    test_niah(model, name, ctx_len=16384)

    del model
    torch.cuda.empty_cache()
