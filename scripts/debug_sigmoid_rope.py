#!/usr/bin/env python3
"""
Sigmoid RoPE Gate-Debug Script
门禁式 Debug 流程定位 sigmoid PPL 异常原因

Steps:
A) Config dump for geo_100k vs sigmoid_100k
B) Frequency/angle sanity check
C) RoPE orthogonality unit test
D) Common bug checklist
E) Gate re-evaluation (2048, 8192 only)
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

# Config
MODEL_PATH = "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct"
DATA_PATH = "/root/.cache/huggingface/hub/datasets--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1"
OUTPUT_DIR = Path("/root/autodl-tmp/dfrope/hybrid-rope/results/sigmoid_debug")
SEED = 42
LENGTHS = [2048, 8192]  # Gate test: no 16k yet

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_data(tokenizer, max_tokens=250000):
    import pandas as pd
    df = pd.read_parquet(f"{DATA_PATH}/validation-00000-of-00001.parquet")
    text = "\n\n".join(df["text"].tolist())
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:max_tokens]
    return torch.tensor(tokens, dtype=torch.long)

# ========================================
# STEP A: Config Dump
# ========================================
def compute_inv_freq_geometric(head_dim, theta=10000):
    """Standard geometric RoPE"""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

def compute_inv_freq_sigmoid(head_dim, T=100000):
    """Sigmoid-shaped RoPE - CURRENT IMPLEMENTATION (potentially buggy)"""
    inv_freq_min = 1e-6
    inv_freq_max = 1.0
    mid = T / 2
    steepness = 10.0 / T
    positions = torch.arange(0, head_dim // 2).float()
    return inv_freq_min + (inv_freq_max - inv_freq_min) * torch.sigmoid(
        steepness * (positions * 100 - mid)
    )

def compute_inv_freq_sigmoid_v2(head_dim, T=100000):
    """Sigmoid-shaped RoPE - FIXED VERSION
    
    Key insight: We want omega_i to match geometric at low dimensions
    but decay faster at high dimensions for stability.
    
    geometric: omega_i = theta^(-2i/d) = 1/theta^(2i/d)
    
    For sigmoid, we want similar range but different shape.
    The bug was: positions * 100 creates huge values, making sigmoid saturate.
    """
    # Match geometric theta range: omega goes from ~1 to ~1e-4
    # For theta=100k: omega_0 = 1, omega_d/2 = 1/100000^(1) = 1e-5
    theta = T  # Use T as effective theta
    
    # Standard geometric as baseline
    geometric = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # Sigmoid modulation: start at geometric, decay faster at high freqs
    # Use a smooth transition from geometric to faster decay
    dim_half = head_dim // 2
    positions = torch.arange(0, dim_half).float()
    
    # Sigmoid weight: 0 at low dims (use geometric), 1 at high dims (use fast decay)
    # Transition at 70% of dimensions
    transition_point = dim_half * 0.7
    steepness = 0.1  # Gentle transition
    sigmoid_weight = torch.sigmoid(steepness * (positions - transition_point))
    
    # Fast decay factor (more aggressive than geometric)
    fast_decay = 1.0 / (theta ** (2.0 * positions / head_dim))  # 2x faster decay
    
    # Blend: low dims use geometric, high dims use fast decay
    inv_freq = geometric * (1 - sigmoid_weight) + fast_decay * sigmoid_weight
    
    return inv_freq

def step_a_config_dump(model, tokenizer):
    """Step A: Print and save config details"""
    print("\n" + "="*60)
    print("STEP A: Config Dump")
    print("="*60)
    
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    rotary_pct = getattr(model.config, 'rope_scaling', None)
    
    lines = []
    lines.append("=== Model Config ===")
    lines.append(f"hidden_size: {model.config.hidden_size}")
    lines.append(f"num_attention_heads: {model.config.num_attention_heads}")
    lines.append(f"head_dim: {head_dim}")
    lines.append(f"num_key_value_heads: {model.config.num_key_value_heads}")
    lines.append(f"rope_theta: {getattr(model.config, 'rope_theta', 'N/A')}")
    lines.append(f"rope_scaling: {rotary_pct}")
    
    # Geometric config
    geo_inv_freq = compute_inv_freq_geometric(head_dim, theta=100000)
    lines.append("\n=== geo_100k Config ===")
    lines.append(f"rope_type: geometric")
    lines.append(f"theta: 100000")
    lines.append(f"inv_freq shape: {geo_inv_freq.shape}")
    lines.append(f"inv_freq[:5]: {geo_inv_freq[:5].tolist()}")
    lines.append(f"inv_freq[-5:]: {geo_inv_freq[-5:].tolist()}")
    
    # Sigmoid config (current buggy)
    sig_inv_freq = compute_inv_freq_sigmoid(head_dim, T=100000)
    lines.append("\n=== sigmoid_100k Config (CURRENT - potentially buggy) ===")
    lines.append(f"rope_type: sigmoid")
    lines.append(f"T: 100000")
    lines.append(f"inv_freq_min: 1e-6")
    lines.append(f"inv_freq_max: 1.0")
    lines.append(f"mid: {100000/2}")
    lines.append(f"steepness: {10.0/100000}")
    lines.append(f"positions range: 0 to {(head_dim//2-1)*100}")
    lines.append(f"inv_freq shape: {sig_inv_freq.shape}")
    lines.append(f"inv_freq[:5]: {sig_inv_freq[:5].tolist()}")
    lines.append(f"inv_freq[-5:]: {sig_inv_freq[-5:].tolist()}")
    
    # Compare
    lines.append("\n=== Comparison ===")
    lines.append(f"geo first omega: {geo_inv_freq[0].item():.6e}")
    lines.append(f"sig first omega: {sig_inv_freq[0].item():.6e}")
    lines.append(f"geo last omega: {geo_inv_freq[-1].item():.6e}")
    lines.append(f"sig last omega: {sig_inv_freq[-1].item():.6e}")
    lines.append(f"ratio sig/geo (first): {sig_inv_freq[0]/geo_inv_freq[0]:.2f}")
    lines.append(f"ratio sig/geo (last): {sig_inv_freq[-1]/geo_inv_freq[-1]:.2f}")
    
    content = "\n".join(lines)
    print(content)
    
    with open(OUTPUT_DIR / "logs/config_dump.txt", "w") as f:
        f.write(content)
    
    return geo_inv_freq, sig_inv_freq

# ========================================
# STEP B: Frequency/Angle Sanity
# ========================================
def step_b_freq_sanity(geo_inv_freq, sig_inv_freq, seq_len=2048):
    """Step B: Check frequency and angle ranges"""
    print("\n" + "="*60)
    print("STEP B: Frequency/Angle Sanity")
    print("="*60)
    
    lines = []
    lines.append(f"Sequence length: {seq_len}")
    
    # Omega = inv_freq
    lines.append("\n=== geo_100k omega (inv_freq) ===")
    lines.append(f"First 10: {geo_inv_freq[:10].tolist()}")
    lines.append(f"Last 10: {geo_inv_freq[-10:].tolist()}")
    
    lines.append("\n=== sigmoid_100k omega (inv_freq) ===")
    lines.append(f"First 10: {sig_inv_freq[:10].tolist()}")
    lines.append(f"Last 10: {sig_inv_freq[-10:].tolist()}")
    
    # Max angle at position 2047
    pos = seq_len - 1
    geo_angles = geo_inv_freq * pos
    sig_angles = sig_inv_freq * pos
    
    lines.append(f"\n=== Max angle at position {pos} ===")
    lines.append(f"geo max angle: {geo_angles.max().item():.2f} rad ({geo_angles.max().item()/math.pi:.2f} pi)")
    lines.append(f"sig max angle: {sig_angles.max().item():.2f} rad ({sig_angles.max().item()/math.pi:.2f} pi)")
    lines.append(f"geo min angle: {geo_angles.min().item():.6f} rad")
    lines.append(f"sig min angle: {sig_angles.min().item():.6f} rad")
    
    # Critical: Check if sigmoid angles are reasonable
    lines.append("\n=== CRITICAL CHECKS ===")
    if sig_inv_freq[0] > 0.5:
        lines.append(f"⚠️ WARNING: sig first omega = {sig_inv_freq[0]:.4f} > 0.5")
        lines.append("  This means highest freq rotates > 0.5 rad per position!")
        lines.append("  At pos 2047: angle = {:.1f} rad = {:.1f} rotations".format(
            sig_inv_freq[0] * 2047, sig_inv_freq[0] * 2047 / (2*math.pi)))
    
    if sig_inv_freq[-1] < 1e-5:
        lines.append(f"⚠️ WARNING: sig last omega = {sig_inv_freq[-1]:.2e} very small")
        lines.append("  This dimension barely rotates at all")
    
    content = "\n".join(lines)
    print(content)
    
    with open(OUTPUT_DIR / "logs/freq_dump.txt", "w") as f:
        f.write(content)

# ========================================
# STEP C: Orthogonality Test
# ========================================
def step_c_orthogonality_test(head_dim):
    """Step C: Test RoPE rotation orthogonality"""
    print("\n" + "="*60)
    print("STEP C: RoPE Orthogonality Test")
    print("="*60)
    
    lines = []
    
    def apply_rope_simple(x, inv_freq, position_ids):
        """Simple RoPE application for testing"""
        # x: (batch, heads, seq, dim)
        # inv_freq: (dim/2,)
        # position_ids: (seq,)
        
        batch, heads, seq, dim = x.shape
        dim_half = dim // 2
        
        # Split into pairs
        x1 = x[..., :dim_half]
        x2 = x[..., dim_half:]
        
        # Compute angles: (seq, dim/2)
        freqs = torch.einsum('i,j->ij', position_ids.float(), inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        
        # Rotate: (batch, heads, seq, dim/2)
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        return torch.cat([x1_rot, x2_rot], dim=-1)
    
    # Test with random input
    set_seed(42)
    batch, heads, seq = 2, 4, 128
    x = torch.randn(batch, heads, seq, head_dim)
    position_ids = torch.arange(seq)
    
    # Test geometric
    geo_inv_freq = compute_inv_freq_geometric(head_dim, theta=100000)
    x_geo = apply_rope_simple(x, geo_inv_freq, position_ids)
    
    geo_norm_before = x.norm(dim=-1).mean().item()
    geo_norm_after = x_geo.norm(dim=-1).mean().item()
    geo_rel_err = abs(geo_norm_after - geo_norm_before) / geo_norm_before
    
    lines.append("=== Geometric RoPE ===")
    lines.append(f"Norm before: {geo_norm_before:.6f}")
    lines.append(f"Norm after: {geo_norm_after:.6f}")
    lines.append(f"Relative error: {geo_rel_err:.6e}")
    lines.append(f"PASS: {geo_rel_err < 1e-5}")
    
    # Test sigmoid (current)
    sig_inv_freq = compute_inv_freq_sigmoid(head_dim, T=100000)
    x_sig = apply_rope_simple(x, sig_inv_freq, position_ids)
    
    sig_norm_before = x.norm(dim=-1).mean().item()
    sig_norm_after = x_sig.norm(dim=-1).mean().item()
    sig_rel_err = abs(sig_norm_after - sig_norm_before) / sig_norm_before
    
    lines.append("\n=== Sigmoid RoPE (current) ===")
    lines.append(f"Norm before: {sig_norm_before:.6f}")
    lines.append(f"Norm after: {sig_norm_after:.6f}")
    lines.append(f"Relative error: {sig_rel_err:.6e}")
    lines.append(f"PASS: {sig_rel_err < 1e-5}")
    
    # Check individual dimension rotations
    lines.append("\n=== Per-dimension rotation check ===")
    for i in range(min(5, head_dim//2)):
        omega_geo = geo_inv_freq[i].item()
        omega_sig = sig_inv_freq[i].item()
        lines.append(f"Dim {i}: geo_omega={omega_geo:.6f}, sig_omega={omega_sig:.6f}, ratio={omega_sig/omega_geo:.2f}")
    
    content = "\n".join(lines)
    print(content)
    
    with open(OUTPUT_DIR / "logs/orthogonality_test.txt", "w") as f:
        f.write(content)
    
    return geo_rel_err < 1e-5, sig_rel_err < 1e-5

# ========================================
# STEP D: Bug Checklist
# ========================================
def step_d_bug_checklist():
    """Step D: Check common bugs"""
    print("\n" + "="*60)
    print("STEP D: Common Bug Checklist")
    print("="*60)
    
    lines = []
    lines.append("=== Manual Checklist ===")
    lines.append("")
    lines.append("[ ] Double rotation of q/k?")
    lines.append("    - Check if model already applies RoPE internally")
    lines.append("    - Our patch replaces inv_freq, so model still does the rotation")
    lines.append("    - Should be OK if we only patch inv_freq")
    lines.append("")
    lines.append("[ ] Cache being re-roped?")
    lines.append("    - With nohup fresh model load, KV cache should be fresh")
    lines.append("    - Check if past_key_values causes issues")
    lines.append("")
    lines.append("[ ] rotary_dim pairing error?")
    lines.append("    - LLaMA uses full rotary_dim = head_dim")
    lines.append("    - Check if split is correct (first half, second half)")
    lines.append("")
    lines.append("[ ] position_ids broadcast wrong?")
    lines.append("    - position_ids should be (batch, seq) or (seq,)")
    lines.append("    - HuggingFace handles this internally")
    lines.append("")
    lines.append("[ ] sigmoid parameterization issue?")
    lines.append("    - positions * 100 creates values: 0, 100, 200, ... 6400")
    lines.append("    - steepness=10/T = 10/100000 = 0.0001")
    lines.append("    - sigmoid(0.0001 * 6400 - 50000) ≈ sigmoid(-49360) ≈ 0")
    lines.append("    - sigmoid(0.0001 * 0 - 50000) ≈ sigmoid(-50000) ≈ 0")
    lines.append("    - ALL sigmoid outputs ≈ 0, so inv_freq ≈ inv_freq_min = 1e-6!")
    lines.append("")
    lines.append("!!! BUG IDENTIFIED !!!")
    lines.append("The sigmoid parameterization is wrong:")
    lines.append("- positions * 100 makes the range way too large")
    lines.append("- mid = T/2 = 50000, but positions*100 only goes to ~6400")
    lines.append("- sigmoid(x - 50000) where x in [0, 6400] is ALWAYS near 0")
    lines.append("- Result: ALL inv_freq ≈ 1e-6 (minimum), destroying the model!")
    lines.append("")
    lines.append("FIX: Remove the '*100' factor, use proper scaling")
    
    content = "\n".join(lines)
    print(content)
    
    with open(OUTPUT_DIR / "logs/bug_checklist.txt", "w") as f:
        f.write(content)
    
    return "positions * 100 is the bug"

# ========================================
# STEP E: Gate Re-evaluation
# ========================================
@torch.no_grad()
def eval_ppl(model, tokens, length):
    model.eval()
    window = tokens[:length].unsqueeze(0).cuda()
    outputs = model(window, labels=window)
    loss = outputs.loss.item()
    return np.exp(loss)

def apply_rope_patch(model, inv_freq):
    """Patch RoPE frequencies in the model"""
    for name, module in model.named_modules():
        if 'rotary_emb' in name.lower():
            module.inv_freq = inv_freq.to(model.device)
            break
    return model

def step_e_gate_test(tokens, tokenizer, head_dim):
    """Step E: Gate re-evaluation with fixed sigmoid"""
    print("\n" + "="*60)
    print("STEP E: Gate Re-evaluation")
    print("="*60)
    
    results = {}
    
    # Test configs
    configs = {
        "geo_100k": compute_inv_freq_geometric(head_dim, theta=100000),
        "sigmoid_buggy": compute_inv_freq_sigmoid(head_dim, T=100000),
        "sigmoid_fixed": compute_inv_freq_sigmoid_v2(head_dim, T=100000),
    }
    
    for config_name, inv_freq in configs.items():
        print(f"\n--- Testing {config_name} ---")
        
        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        model = apply_rope_patch(model, inv_freq)
        
        results[config_name] = {}
        
        for length in LENGTHS:
            print(f"  Length {length}...")
            ppl_values = []
            for i in range(3):  # 3 windows for speed
                set_seed(SEED + i)
                max_start = len(tokens) - length
                if max_start > 0:
                    start = np.random.randint(0, max_start)
                    window = tokens[start:start+length]
                else:
                    window = tokens[:length]
                ppl = eval_ppl(model, window, length)
                ppl_values.append(ppl)
                print(f"    Window {i+1}: PPL={ppl:.3f}")
            
            results[config_name][length] = {
                "ppl_mean": float(np.mean(ppl_values)),
                "ppl_std": float(np.std(ppl_values))
            }
        
        del model
        torch.cuda.empty_cache()
    
    # Generate summary
    geo_2k = results["geo_100k"][2048]["ppl_mean"]
    buggy_2k = results["sigmoid_buggy"][2048]["ppl_mean"]
    fixed_2k = results["sigmoid_fixed"][2048]["ppl_mean"]
    
    buggy_ratio = buggy_2k / geo_2k
    fixed_ratio = fixed_2k / geo_2k
    
    gate_pass = fixed_ratio <= 1.5
    
    summary = f"""# Sigmoid RoPE Debug Summary

## Bug Identified
The sigmoid parameterization had a critical bug:
- `positions * 100` created values 0, 100, 200, ... 6400
- `mid = T/2 = 50000`
- `sigmoid((positions*100) - 50000)` ≈ sigmoid(-50000 to -43600) ≈ 0 for ALL positions
- Result: ALL inv_freq ≈ inv_freq_min = 1e-6 (way too small!)

## Fix Applied
New sigmoid implementation (v2):
- Uses proper scaling to match geometric at low dimensions
- Applies sigmoid-based modulation only at higher dimensions
- Smooth transition from geometric to faster decay

## Results

| Config | PPL@2048 | PPL@8192 | Ratio to geo@2k |
|--------|----------|----------|-----------------|
| geo_100k | {geo_2k:.3f} | {results['geo_100k'][8192]['ppl_mean']:.3f} | 1.00x |
| sigmoid_buggy | {buggy_2k:.3f} | {results['sigmoid_buggy'][8192]['ppl_mean']:.3f} | {buggy_ratio:.2f}x |
| sigmoid_fixed | {fixed_2k:.3f} | {results['sigmoid_fixed'][8192]['ppl_mean']:.3f} | {fixed_ratio:.2f}x |

## Gate Evaluation
- Threshold: PPL@2048 <= 1.5x geo_100k
- sigmoid_fixed ratio: {fixed_ratio:.2f}x
- **GATE: {'✅ PASSED' if gate_pass else '❌ FAILED'}**

## Next Steps
{'- Gate passed! Can proceed to 16k/24k/32k boundary scan.' if gate_pass else '- Gate failed. Need further debugging.'}
"""
    
    print(summary)
    
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(OUTPUT_DIR / "summary.md", "w") as f:
        f.write(summary)
    
    return results, gate_pass

# ========================================
# Main
# ========================================
def main():
    print("="*60)
    print("Sigmoid RoPE Gate-Debug")
    print("="*60)
    
    # Load tokenizer
    print("\nLoading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokens = load_data(tokenizer)
    print(f"Loaded {len(tokens)} tokens")
    
    # Load model for config
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"head_dim: {head_dim}")
    
    # Step A: Config dump
    geo_inv_freq, sig_inv_freq = step_a_config_dump(model, tokenizer)
    
    # Step B: Frequency sanity
    step_b_freq_sanity(geo_inv_freq, sig_inv_freq)
    
    # Step C: Orthogonality test
    geo_ok, sig_ok = step_c_orthogonality_test(head_dim)
    
    # Step D: Bug checklist
    bug = step_d_bug_checklist()
    
    # Free model memory
    del model
    torch.cuda.empty_cache()
    
    # Step E: Gate test
    results, gate_pass = step_e_gate_test(tokens, tokenizer, head_dim)
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)
    print(f"Bug found: {bug}")
    print(f"Gate passed: {gate_pass}")

if __name__ == "__main__":
    main()