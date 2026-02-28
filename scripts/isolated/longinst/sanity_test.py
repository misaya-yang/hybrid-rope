#!/usr/bin/env python3
"""
sanity_test.py — Quick verification: RoPE effect + label mask correctness.
No training. Tests 1-2 are CPU-only. Test 3 needs GPU (skips if busy).

Run: /root/miniconda3/bin/python sanity_test.py
"""
import torch
import json
import math
import subprocess
import sys
from pathlib import Path

MODEL = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
ARTIFACT_ROOT = Path("/root/autodl-tmp/dfrope/hybrid-rope/artifacts/llama8k_theory_v1")
BASE = 500000.0
HD = 128
N = HD // 2

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m⊘ SKIP\033[0m"


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def build_geometric():
    idx = torch.arange(N, dtype=torch.float64)
    return (BASE ** -(idx / N)).float()


def build_evq_cosh(tau=1.5):
    idx = torch.arange(N, dtype=torch.float64)
    u = idx / float(N)
    tau_t = torch.tensor(tau, dtype=torch.float64)
    phi = 1.0 - (1.0 / tau_t) * torch.asinh((1.0 - u) * torch.sinh(tau_t))
    return (torch.tensor(BASE, dtype=torch.float64) ** (-phi)).float()


def patch_rotary(model, inv_freq):
    count = 0
    for name, mod in model.named_modules():
        if not hasattr(mod, "inv_freq") or "rotary" not in name:
            continue
        with torch.no_grad():
            mod.inv_freq.copy_(inv_freq.to(device=mod.inv_freq.device, dtype=mod.inv_freq.dtype))
            if hasattr(mod, "original_inv_freq") and isinstance(getattr(mod, "original_inv_freq"), torch.Tensor):
                mod.original_inv_freq.copy_(inv_freq.to(
                    device=mod.original_inv_freq.device, dtype=mod.original_inv_freq.dtype
                ))
        for attr in ("_cos_cached", "_sin_cached", "cos_cached", "sin_cached"):
            if hasattr(mod, attr):
                try:
                    delattr(mod, attr)
                except Exception:
                    try:
                        setattr(mod, attr, None)
                    except Exception:
                        pass
        for attr in ("max_seq_len_cached", "seq_len_cached"):
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, 0)
                except Exception:
                    pass
        count += 1
    return count


def gpu_free_mb():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        return int(r.stdout.strip().split()[0])
    return 0


# ═══════════════════════════════════════════════════════════
# TEST 1: inv_freq 数学验证 (CPU only, <1s)
# ═══════════════════════════════════════════════════════════

def test_inv_freq_math():
    print("=" * 65)
    print("TEST 1: inv_freq Mathematical Verification (CPU)")
    print("=" * 65)
    results = []

    geo = build_geometric()
    evq = build_evq_cosh(tau=1.5)

    # 1a: EVQ != geometric
    max_diff = (geo - evq).abs().max().item()
    ok = max_diff > 0.01
    print(f"  geo[:5]       = {[f'{x:.6f}' for x in geo[:5].tolist()]}")
    print(f"  evq_1.5[:5]   = {[f'{x:.6f}' for x in evq[:5].tolist()]}")
    print(f"  max_diff       = {max_diff:.6f}")
    print(f"  {PASS if ok else FAIL}: EVQ tau=1.5 {'differs from' if ok else 'SAME AS'} geometric")
    results.append(ok)

    # 1b: EVQ properties
    pos = bool(torch.all(evq > 0))
    mono = bool(torch.all(evq[:-1] > evq[1:]))
    print(f"  positive: {pos}  strictly_decreasing: {mono}")
    print(f"  {PASS if (pos and mono) else FAIL}: EVQ basic properties")
    results.append(pos and mono)

    # 1c: Verify saved A1 inv_freq = geometric
    a1_paths = sorted(ARTIFACT_ROOT.glob("train/A1_*/artifacts/custom_inv_freq.pt"))
    if a1_paths:
        saved = torch.load(str(a1_paths[0]), map_location="cpu").float()
        a1_diff = (saved - geo).abs().max().item()
        ok = a1_diff < 1e-5
        print(f"  A1 saved vs geometric: max_diff = {a1_diff:.2e}")
        print(f"  {PASS if ok else FAIL}: A1 (tau=0.0) = geometric")
        results.append(ok)
    else:
        print(f"  {SKIP}: No A1 custom_inv_freq.pt found")

    # 1d: Check A2 inv_freq if exists (should != geometric)
    a2_paths = sorted(ARTIFACT_ROOT.glob("train/A2_*/artifacts/custom_inv_freq.pt"))
    if a2_paths:
        saved2 = torch.load(str(a2_paths[0]), map_location="cpu").float()
        a2_geo_diff = (saved2 - geo).abs().max().item()
        a2_evq_diff = (saved2 - evq).abs().max().item()
        ok = a2_geo_diff > 0.01 and a2_evq_diff < 1e-4
        print(f"  A2 saved vs geometric: max_diff = {a2_geo_diff:.6f}")
        print(f"  A2 saved vs EVQ_1.5:   max_diff = {a2_evq_diff:.2e}")
        print(f"  {PASS if ok else FAIL}: A2 (tau=1.5) = EVQ, ≠ geometric")
        results.append(ok)
    else:
        print(f"  {SKIP}: A2 not started yet (will verify later)")

    return all(results)


# ═══════════════════════════════════════════════════════════
# TEST 2: Label Mask 验证 (CPU only, <3s)
# ═══════════════════════════════════════════════════════════

def test_label_mask():
    print("\n" + "=" * 65)
    print("TEST 2: Label Mask Verification (CPU)")
    print("=" * 65)
    results = []

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    # --- 2a: Synthetic label mask check ---
    print("\n  [2a] Synthetic sample label mask:")
    messages = [
        {"role": "user", "content": "What is 2+2? Explain your reasoning step by step."},
        {"role": "assistant", "content": "The answer is 4. Two plus two equals four."},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prefix_msgs = [messages[0], {"role": "assistant", "content": ""}]
    prefix_text = tokenizer.apply_chat_template(prefix_msgs, tokenize=False)

    full_enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = full_enc["input_ids"]
    offsets = full_enc.get("offset_mapping")

    prefix_char_len = len(prefix_text)
    if offsets:
        ast_start = next((i for i, (s, e) in enumerate(offsets) if s >= prefix_char_len), len(full_ids))
        method = "offset_mapping"
    else:
        ast_start = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
        method = "prefix_len"

    labels = list(full_ids)
    for i in range(ast_start):
        labels[i] = -100

    sup_ids = [x for x in labels if x != -100]
    sup_text = tokenizer.decode(sup_ids)
    masked_count = sum(1 for x in labels if x == -100)

    print(f"  method={method} total={len(full_ids)} masked={masked_count} supervised={len(sup_ids)}")
    print(f"  supervised_text = {sup_text!r}")

    ok = len(sup_ids) > 0 and ("4" in sup_text or "four" in sup_text.lower())
    print(f"  {PASS if ok else FAIL}: supervised tokens contain expected response")
    results.append(ok)

    ok2 = masked_count > 0 and len(sup_ids) > 0
    print(f"  {PASS if ok2 else FAIL}: both masked ({masked_count}) and supervised ({len(sup_ids)}) present")
    results.append(ok2)

    # --- 2b: Check ACTUAL training data segmentation preview ---
    print("\n  [2b] Actual training data label masks:")
    data_dirs = sorted(ARTIFACT_ROOT.glob("data/A1_*"))
    if data_dirs:
        seg_path = data_dirs[0] / "segmentation_preview_10.json"
        stats_path = data_dirs[0] / "stats.json"

        if seg_path.exists():
            seg = json.loads(seg_path.read_text())
            rows = seg.get("rows", [])
            print(f"  Found {len(rows)} segmentation preview samples:")
            zero_sup = 0
            for r in rows[:5]:
                total = r.get("total_tokens", 0)
                sup = r.get("supervised_tokens", r.get("assistant_tokens", 0))
                src = r.get("source", "?")
                bnd = r.get("boundary_mode", "?")
                print(f"    src={src:>10s} total={total:>5d} sup={sup:>4d} boundary={bnd}")
                if sup == 0:
                    zero_sup += 1
            ok3 = zero_sup == 0
            print(f"  {PASS if ok3 else FAIL}: no zero-supervised samples in preview")
            results.append(ok3)
        else:
            print(f"  {SKIP}: segmentation_preview_10.json not found")

        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            lt64 = float(stats.get("assistant_tokens_lt64_ratio", -1))
            ast = stats.get("assistant_tokens", {})
            print(f"\n  assistant_tokens stats: min={ast.get('min')} p50={ast.get('p50')} "
                  f"p90={ast.get('p90')} max={ast.get('max')}")
            print(f"  lt64_ratio = {lt64}")
            ok4 = lt64 == 0.0
            print(f"  {PASS if ok4 else FAIL}: no samples with <64 supervised tokens")
            results.append(ok4)

            # Check length distribution
            tot = stats.get("total_tokens", {})
            p90 = float(tot.get("p90", 0))
            ok5 = p90 > 4000
            print(f"\n  total_tokens: p50={tot.get('p50')} p90={tot.get('p90')} max={tot.get('max')}")
            print(f"  {PASS if ok5 else FAIL}: p90 length > 4K (sufficient long-range coverage)")
            results.append(ok5)

            # Post-filter ratio check
            post = stats.get("post_token_source_token_ratio", {})
            if post:
                long_r = float(post.get("long", 0))
                syn_r = float(post.get("synthetic", 0))
                wiki_r = float(post.get("wiki", 0))
                print(f"\n  Post-filter token ratios: long={long_r:.4f} synthetic={syn_r:.4f} wiki={wiki_r:.4f}")
                ok6 = long_r > 0.5
                print(f"  {PASS if ok6 else FAIL}: long data dominates post-filter ({long_r:.1%})")
                results.append(ok6)
                if syn_r > 0.25:
                    print(f"  ⚠ WARNING: synthetic ratio {syn_r:.1%} > 25% (target 20%)")
    else:
        print(f"  {SKIP}: No A1 data directory found")

    return all(results)


# ═══════════════════════════════════════════════════════════
# TEST 3: RoPE Forward Pass Effect (needs GPU)
# ═══════════════════════════════════════════════════════════

def test_rope_forward():
    print("\n" + "=" * 65)
    print("TEST 3: RoPE Forward Pass Verification (GPU)")
    print("=" * 65)

    free = gpu_free_mb()
    if free < 8000:
        print(f"  {SKIP}: {free}MB free, need 8GB+. Run after training finishes.")
        print(f"  Command: kill training first or wait, then rerun this test.")
        return None

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import gc

    print("  Loading model in 4bit (to coexist with training)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto",
        local_files_only=True, attn_implementation="sdpa",
    )
    model.eval()

    # Need sufficient sequence length to see RoPE effect
    text = "The capital of France is Paris. " * 64 + "The final answer is"
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(model.device)
    print(f"  Input length: {input_ids.shape[1]} tokens")

    geo = build_geometric()
    evq = build_evq_cosh(tau=1.5)

    # --- Forward with geometric ---
    n_patched = patch_rotary(model, geo)
    print(f"  Patched {n_patched} rotary modules")

    # Verify inv_freq is what we set
    for name, mod in model.named_modules():
        if hasattr(mod, "inv_freq") and "rotary" in name:
            actual = mod.inv_freq[:5].float().cpu()
            expected = geo[:5]
            match = (actual - expected).abs().max().item() < 1e-5
            print(f"  Runtime inv_freq[:5] after geo patch: {actual.tolist()}")
            print(f"  Expected:                             {expected.tolist()}")
            print(f"  {PASS if match else FAIL}: inv_freq matches what we injected")
            break

    with torch.no_grad():
        logits_geo = model(input_ids).logits[0, -1, :].float().cpu()

    # --- Forward with EVQ ---
    patch_rotary(model, evq)

    for name, mod in model.named_modules():
        if hasattr(mod, "inv_freq") and "rotary" in name:
            actual = mod.inv_freq[:5].float().cpu()
            expected = evq[:5]
            match = (actual - expected).abs().max().item() < 1e-5
            print(f"  Runtime inv_freq[:5] after EVQ patch: {actual.tolist()}")
            print(f"  {PASS if match else FAIL}: inv_freq updated to EVQ values")
            break

    with torch.no_grad():
        logits_evq = model(input_ids).logits[0, -1, :].float().cpu()

    # --- Compare ---
    diff = (logits_geo - logits_evq).abs()
    top_geo = logits_geo.topk(5)
    top_evq = logits_evq.topk(5)
    print(f"\n  geo top5 tokens: {[tokenizer.decode([t]) for t in top_geo.indices.tolist()]}")
    print(f"  evq top5 tokens: {[tokenizer.decode([t]) for t in top_evq.indices.tolist()]}")
    print(f"  logits max_diff:  {diff.max().item():.4f}")
    print(f"  logits mean_diff: {diff.mean().item():.6f}")

    ok = diff.max().item() > 0.01
    print(f"  {PASS if ok else FAIL}: RoPE change {'causes' if ok else 'has NO'} output difference")

    # --- Forward with labels to verify loss mask ---
    print("\n  Verifying loss with label masks...")
    labels_all = input_ids.clone()
    labels_none = torch.full_like(input_ids, -100)
    labels_half = input_ids.clone()
    labels_half[0, : input_ids.shape[1] // 2] = -100

    patch_rotary(model, geo)
    with torch.no_grad():
        loss_all = model(input_ids=input_ids, labels=labels_all).loss.item()
        loss_none = model(input_ids=input_ids, labels=labels_none).loss
        loss_none = loss_none.item() if loss_none is not None and not math.isnan(loss_none.item() if isinstance(loss_none, torch.Tensor) else float(loss_none)) else float("nan")
        loss_half = model(input_ids=input_ids, labels=labels_half).loss.item()

    print(f"  loss(all_supervised) = {loss_all:.4f}")
    print(f"  loss(all_masked=-100) = {loss_none}")
    print(f"  loss(half_masked)     = {loss_half:.4f}")

    mask_ok = math.isnan(loss_none) or loss_none == 0.0
    print(f"  {PASS if mask_ok else FAIL}: all-masked gives NaN/0 (labels=-100 correctly ignored)")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ok


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("EVQ-Cosh RoPE + Label Mask Sanity Test")
    print("=" * 65)
    r1 = test_inv_freq_math()
    r2 = test_label_mask()
    r3 = test_rope_forward()

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Test 1 (inv_freq math):   {'PASS' if r1 else 'FAIL'}")
    print(f"  Test 2 (label mask):      {'PASS' if r2 else 'FAIL'}")
    print(f"  Test 3 (RoPE forward):    {'PASS' if r3 else ('FAIL' if r3 is False else 'SKIPPED')}")
    print("=" * 65)
